"""Signal engine — reads city collector DB, outputs SignalState every poll cycle.

This module extracts 7 core signals from the collector's SQLite database.
All reads are from the local city DB. All signal logic is deterministic
and physically grounded — no ML, no black boxes.

Climate day: midnight–midnight LST (EST for KMIA) = 05:00Z–05:00Z year-round.
"""

from __future__ import annotations

import logging
import math
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

log = logging.getLogger(__name__)

# Climate day for KMIA: midnight–midnight EST = 05:00Z–05:00Z
DEFAULT_CLIMATE_DAY_START_UTC_HOUR = 5


# ---------------------------------------------------------------------------
# Configuration (passed in, not imported from trader)
# ---------------------------------------------------------------------------

@dataclass
class SignalParams:
    """Signal thresholds — self-adjust from settlement data via calibrator."""
    cape_threshold: float = 2500.0
    pw_threshold: float = 40.0
    continental_wind_min: float = 250.0
    continental_wind_max: float = 360.0
    dew_buffer_f: float = 1.5
    dew_crash_threshold_f: float = 3.0
    fawn_crash_threshold_f: float = 3.0
    nearby_divergence_threshold_f: float = 2.0
    pressure_surge_threshold_hpa: float = 1.0
    climate_day_start_utc_hour: int = DEFAULT_CLIMATE_DAY_START_UTC_HOUR


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ModelTrackingProfile:
    """Per-model tracking quality since last refresh.

    Built by comparing the model's forward-curve predictions for past hours
    against actual observations at those hours. This tells us:
      - mean_error: systematic bias (positive = model too cold)
      - recent_error: error at the most recent verified hour
      - trend: is the error growing or shrinking?
      - n_hours: how many hours of verification data we have
    """
    model: str
    mean_error: float = 0.0         # avg(obs - predicted) across verified hours
    recent_error: float = 0.0       # error at the most recent hour
    trend: float = 0.0              # recent_error - earliest_error (growing = positive)
    n_hours: int = 0                # how many hours of verification data
    errors_by_hour: dict[str, float] = field(default_factory=dict)  # {hour_utc: error}

    @property
    def correction(self) -> float:
        """Best-estimate correction to apply to this model's forward projections.

        Uses recent error (weighted 70%) blended with mean error (30%).
        Recent error reacts faster to evolving conditions; mean error
        prevents overreacting to a single noisy hour.
        """
        if self.n_hours == 0:
            return 0.0
        if self.n_hours == 1:
            return self.recent_error
        return 0.7 * self.recent_error + 0.3 * self.mean_error

    @property
    def residual(self) -> float:
        """Unexplained error after correction — how much scatter remains.

        If a model has a steady +5°F bias, correction handles it and
        residual is low (trustworthy after correction). If a model
        oscillates wildly, residual is high (untrustworthy even after correction).
        """
        if self.n_hours < 2:
            return abs(self.recent_error) * 0.5  # uncertain with little data
        errs = list(self.errors_by_hour.values())
        corrected = [e - self.correction for e in errs]
        return (sum(c ** 2 for c in corrected) / len(corrected)) ** 0.5


@dataclass
class SignalState:
    """Flat struct of signal values. No interpretation, no decisions."""

    timestamp_utc: str
    station: str
    target_date: str
    market_type: str
    hours_remaining: float

    # Signal 1: Model consensus (raw MAE-weighted, no bias adjustment)
    consensus_f: float | None = None
    consensus_sigma: float | None = None
    n_models: int = 0

    # Signal 2: Obs divergence
    obs_current_f: float | None = None
    obs_trend_2hr: float | None = None
    obs_vs_consensus: float | None = None
    projected_extreme_f: float | None = None

    # Signal 3: CAPE + PW outflow risk
    cape_current: float | None = None
    pw_mm: float | None = None
    outflow_risk: bool = False
    cape_trend_1hr: float | None = None

    # Signal 4: Wind direction
    wind_dir_deg: float | None = None
    continental_intrusion: bool = False
    wind_shift_detected: bool = False

    # Signal 5: Dew point floor
    dew_point_f: float | None = None
    evening_dew_mean_f: float | None = None
    estimated_low_floor_f: float | None = None
    dew_crash_active: bool = False

    # Signal 6: Nearby station lead
    fawn_temp_f: float | None = None
    fawn_crash_detected: bool = False
    fawn_lead_minutes: float | None = None
    nearby_divergence_f: float | None = None
    nearby_crash_count: int = 0

    # Signal 7: Pressure
    pressure_hpa: float | None = None
    pressure_3hr_trend: float | None = None
    pressure_surge: bool = False

    # Running extremes
    running_high_f: float | None = None
    running_low_f: float | None = None

    # Forward curve: per-model predicted extremes for remaining hours.
    forward_max_f: float | None = None
    forward_min_f: float | None = None
    forward_model_maxes: dict[str, float] = field(default_factory=dict)
    forward_model_mins: dict[str, float] = field(default_factory=dict)

    # Per-model tracking profiles — multi-hour error history with corrections.
    model_tracking: dict[str, ModelTrackingProfile] = field(default_factory=dict)

    # Legacy single-point tracking (kept for backward compat, derived from profiles)
    forward_tracking_errors: dict[str, float] = field(default_factory=dict)

    # Composite
    active_flags: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _circular_diff(a: float, b: float) -> float:
    """Signed angular difference a - b, in [-180, 180]."""
    return (a - b + 180) % 360 - 180


# ---------------------------------------------------------------------------
# Signal extraction
# ---------------------------------------------------------------------------

class SignalEngine:
    """Extracts SignalState from the city collector's DB."""

    def __init__(self, db: sqlite3.Connection, station: str,
                 params: SignalParams | None = None):
        self.db = db
        self.db.row_factory = sqlite3.Row
        self.station = station
        self.p = params or SignalParams()
        self._as_of_utc: datetime | None = None

    def _now_utc(self) -> datetime:
        return self._as_of_utc or datetime.now(timezone.utc)

    def _iso_utc(self, dt: datetime) -> str:
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def _climate_day_bounds(self, target_date: str) -> tuple[str, str]:
        h = self.p.climate_day_start_utc_hour
        next_day = (datetime.strptime(target_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
        return (f"{target_date}T{h:02d}:00:00Z", f"{next_day}T{h:02d}:00:00Z")

    def _hours_remaining(self, now_utc: datetime, target_date: str) -> float:
        end_str = self._climate_day_bounds(target_date)[1]
        end_dt = datetime.strptime(end_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        return max((end_dt - now_utc).total_seconds() / 3600, 0.0)

    def current_target_date(self, now_utc: datetime | None = None) -> str:
        if now_utc is None:
            now_utc = datetime.now(timezone.utc)
        h = self.p.climate_day_start_utc_hour
        d = now_utc.date() - timedelta(days=1) if now_utc.hour < h else now_utc.date()
        return d.strftime("%Y-%m-%d")

    # --- Individual signal queries ---

    def _latest_obs(self) -> dict | None:
        as_of = self._iso_utc(self._now_utc())
        row = self.db.execute(
            """SELECT temperature_f, dew_point_f, wind_speed_mph,
                      wind_heading_deg, pressure_hpa, sky_cover_pct, timestamp_utc
               FROM observations WHERE station = ? AND temperature_f IS NOT NULL
                 AND timestamp_utc <= ?
               ORDER BY timestamp_utc DESC LIMIT 1""",
            (self.station, as_of),
        ).fetchone()
        return dict(row) if row else None

    def _obs_at_offset(self, hours_ago: float) -> dict | None:
        now = self._now_utc()
        target = (now - timedelta(hours=hours_ago)).strftime("%Y-%m-%dT%H:%M:%SZ")
        lo = (now - timedelta(hours=hours_ago + 0.5)).strftime("%Y-%m-%dT%H:%M:%SZ")
        hi = (now - timedelta(hours=hours_ago - 0.5)).strftime("%Y-%m-%dT%H:%M:%SZ")
        row = self.db.execute(
            """SELECT temperature_f, dew_point_f, wind_heading_deg, pressure_hpa
               FROM observations WHERE station = ? AND temperature_f IS NOT NULL
                 AND timestamp_utc BETWEEN ? AND ?
               ORDER BY ABS(julianday(timestamp_utc) - julianday(?)) LIMIT 1""",
            (self.station, lo, hi, target),
        ).fetchone()
        return dict(row) if row else None

    def _backfill_obs_from_nearby(self, obs: dict | None) -> dict | None:
        """Fill in missing dew_point_f and pressure_hpa from closest nearby station.

        Wethr SSE often provides temp + wind but omits dew point and pressure.
        NWS ASOS stations reliably report both. This fallback queries the
        nearest station with valid data from the last 30 minutes.
        """
        if obs is None:
            return obs
        needs_dew = obs.get("dew_point_f") is None
        needs_pres = obs.get("pressure_hpa") is None
        if not needs_dew and not needs_pres:
            return obs  # nothing to backfill

        as_of = self._iso_utc(self._now_utc())
        cutoff = (self._now_utc() - timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%SZ")

        if needs_dew:
            row = self.db.execute(
                """SELECT dew_point_f FROM nearby_observations
                   WHERE dew_point_f IS NOT NULL AND timestamp_utc >= ? AND timestamp_utc <= ?
                   ORDER BY distance_mi ASC, timestamp_utc DESC LIMIT 1""",
                (cutoff, as_of),
            ).fetchone()
            if row:
                obs["dew_point_f"] = row["dew_point_f"]

        if needs_pres:
            row = self.db.execute(
                """SELECT pressure_slp_hpa FROM nearby_observations
                   WHERE pressure_slp_hpa IS NOT NULL AND timestamp_utc >= ? AND timestamp_utc <= ?
                   ORDER BY distance_mi ASC, timestamp_utc DESC LIMIT 1""",
                (cutoff, as_of),
            ).fetchone()
            if row:
                obs["pressure_hpa"] = row["pressure_slp_hpa"]

        return obs

    def _running_extremes(self, target_date: str) -> tuple[float | None, float | None]:
        """Running high/low for the climate day as-of current replay time.

        Prefers the wethr NWS envelope (whole-degree F, what Kalshi settles on)
        over MAX/MIN of temperature_f (which suffers from ASOS C→F round-trip
        artifacts, e.g. 26°C → 78.8°F when the true reading is 78°F).
        """
        start, end = self._climate_day_bounds(target_date)
        as_of = self._iso_utc(self._now_utc())
        effective_end = min(end, as_of)

        # Try NWS envelope first (authoritative for Kalshi settlement)
        row = self.db.execute(
            """SELECT MAX(wethr_high_nws_f), MIN(wethr_low_nws_f)
               FROM observations WHERE station = ?
                 AND timestamp_utc >= ? AND timestamp_utc < ?
                 AND wethr_high_nws_f IS NOT NULL""",
            (self.station, start, effective_end),
        ).fetchone()
        if row and row[0] is not None:
            return (row[0], row[1])

        # Fallback to raw temperature_f (METAR, subject to rounding)
        row = self.db.execute(
            """SELECT MAX(temperature_f), MIN(temperature_f)
               FROM observations WHERE station = ? AND temperature_f IS NOT NULL
                 AND timestamp_utc >= ? AND timestamp_utc < ?""",
            (self.station, start, effective_end),
        ).fetchone()
        return (row[0], row[1]) if row else (None, None)

    def _consensus(self, target_date: str, market_type: str) -> tuple[float | None, float | None, int]:
        as_of = self._iso_utc(self._now_utc())
        for table in ("model_consensus", "adjusted_forecasts"):
            try:
                row = self.db.execute(
                    f"""SELECT consensus_forecast_f, consensus_std_f, n_models
                        FROM {table}
                        WHERE station = ? AND forecast_date = ? AND market_type = ?
                          AND (created_at IS NULL OR created_at <= ?)
                        ORDER BY created_at DESC LIMIT 1""",
                    (self.station, target_date, market_type, as_of),
                ).fetchone()
            except sqlite3.OperationalError:
                continue
            if row:
                return row["consensus_forecast_f"], row["consensus_std_f"], row["n_models"] or 0
        return None, None, 0

    def _atmospheric(self) -> tuple[float | None, float | None]:
        """Get best available CAPE and PW from all atmospheric sources.

        Sources stored in atmospheric_data table:
          - model='HRRR': model-derived, works under clouds, hourly refresh
          - model='GOES-19-SAT': satellite-derived, clear-sky only, 10-min refresh
          - model='best_match' (Open-Meteo): model-derived, 5-min poll

        Strategy: prefer the MOST RECENT valid CAPE regardless of source.
        HRRR and GOES are both stored with timestamps — freshest wins.
        Over time, BOA and the calibration stack learn which source
        correlates better with actual outflow events in each regime.
        No manual max/min heuristic needed.

        PW comes from HRRR or Open-Meteo (GOES doesn't provide PW).
        """
        as_of = self._iso_utc(self._now_utc())

        # Get most recent CAPE from any source (freshest wins)
        cape_row = self.db.execute(
            """SELECT cape, model FROM atmospheric_data
               WHERE station = ? AND cape IS NOT NULL AND valid_time_utc <= ?
               ORDER BY valid_time_utc DESC LIMIT 1""",
            (self.station, as_of),
        ).fetchone()

        # Get most recent PW (only from model sources — GOES doesn't have PW)
        pw_row = self.db.execute(
            """SELECT precipitable_water_mm FROM atmospheric_data
               WHERE station = ? AND precipitable_water_mm IS NOT NULL
                 AND valid_time_utc <= ?
               ORDER BY valid_time_utc DESC LIMIT 1""",
            (self.station, as_of),
        ).fetchone()

        cape = cape_row["cape"] if cape_row else None
        pw = pw_row["precipitable_water_mm"] if pw_row else None

        return cape, pw

    def _cape_trend(self) -> float | None:
        as_of = self._iso_utc(self._now_utc())
        rows = self.db.execute(
            """SELECT cape, valid_time_utc FROM atmospheric_data
               WHERE station = ? AND cape IS NOT NULL AND valid_time_utc <= ?
               ORDER BY valid_time_utc DESC LIMIT 2""",
            (self.station, as_of),
        ).fetchall()
        if len(rows) >= 2 and rows[0]["cape"] is not None and rows[1]["cape"] is not None:
            return rows[0]["cape"] - rows[1]["cape"]
        return None

    def _evening_dew_mean(self, target_date: str) -> float | None:
        next_day = (datetime.strptime(target_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
        as_of = self._iso_utc(self._now_utc())
        row = self.db.execute(
            """SELECT AVG(dew_point_f) FROM observations
               WHERE station = ? AND dew_point_f IS NOT NULL
                 AND timestamp_utc >= ? AND timestamp_utc < ?
                 AND timestamp_utc <= ?""",
            (self.station, f"{target_date}T23:00:00Z", f"{next_day}T05:00:00Z", as_of),
        ).fetchone()
        return row[0] if row and row[0] else None

    def _fawn_latest(self) -> tuple[float | None, float | None]:
        as_of = self._iso_utc(self._now_utc())
        row = self.db.execute(
            """SELECT air_temp_f, dew_point_f FROM fawn_observations
               WHERE timestamp_utc <= ?
               ORDER BY timestamp_utc DESC LIMIT 1""",
            (as_of,),
        ).fetchone()
        return (row["air_temp_f"], row["dew_point_f"]) if row else (None, None)

    def _fawn_crash(self) -> bool:
        now_utc = self._now_utc()
        cutoff = (now_utc - timedelta(minutes=20)).strftime("%Y-%m-%dT%H:%M:%SZ")
        as_of = self._iso_utc(now_utc)
        rows = self.db.execute(
            """SELECT air_temp_f FROM fawn_observations
               WHERE timestamp_utc >= ? AND timestamp_utc <= ? AND air_temp_f IS NOT NULL
               ORDER BY timestamp_utc DESC LIMIT 2""",
            (cutoff, as_of),
        ).fetchall()
        if len(rows) >= 2:
            return (rows[1]["air_temp_f"] - rows[0]["air_temp_f"]) > self.p.fawn_crash_threshold_f
        return False

    def _nearby_divergence(self) -> tuple[float | None, int]:
        now_utc = self._now_utc()
        cutoff = (now_utc - timedelta(minutes=10)).strftime("%Y-%m-%dT%H:%M:%SZ")
        as_of = self._iso_utc(now_utc)
        rows = self.db.execute(
            """SELECT stid, temp_delta_vs_kmia FROM nearby_observations
               WHERE timestamp_utc >= ? AND timestamp_utc <= ? AND air_temp_f IS NOT NULL
               ORDER BY timestamp_utc DESC""",
            (cutoff, as_of),
        ).fetchall()
        if not rows:
            return None, 0
        seen = {}
        for r in rows:
            if r["stid"] not in seen and r["temp_delta_vs_kmia"] is not None:
                seen[r["stid"]] = r["temp_delta_vs_kmia"]
        if not seen:
            return None, 0
        deltas = list(seen.values())
        return (max(abs(d) for d in deltas),
                sum(1 for d in deltas if d < -self.p.nearby_divergence_threshold_f))

    MODEL_COLS = {
        "NBM": "nbm_temp_f", "GFS": "gfs_temp_f", "ECMWF": "ecmwf_temp_f",
        "HRRR": "hrrr_temp_f", "NAM": "nam_temp_f",
    }

    def _obs_by_hour(self, target_date: str) -> dict[int, float]:
        """Get actual observations keyed by UTC hour for the climate day as-of now.

        Returns {utc_hour: temperature_f} using the obs closest to the top
        of each hour.
        """
        start, end = self._climate_day_bounds(target_date)
        as_of = self._iso_utc(self._now_utc())
        effective_end = min(end, as_of)
        rows = self.db.execute(
            """SELECT temperature_f, timestamp_utc FROM observations
               WHERE station = ? AND temperature_f IS NOT NULL
                 AND timestamp_utc >= ? AND timestamp_utc < ?
               ORDER BY timestamp_utc""",
            (self.station, start, effective_end),
        ).fetchall()

        by_hour: dict[int, float] = {}
        for r in rows:
            ts = r["timestamp_utc"]
            # Parse hour from timestamp
            try:
                if "+" in ts:
                    dt = datetime.fromisoformat(ts)
                else:
                    dt = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
                h = dt.hour
                # Keep the obs closest to the top of hour (first per hour is fine)
                if h not in by_hour:
                    by_hour[h] = r["temperature_f"]
            except (ValueError, TypeError):
                continue
        return by_hour

    def _forward_curve_extremes(self, target_date: str, obs_temp_f: float | None = None) -> tuple[
        float | None, float | None, dict[str, float], dict[str, float],
        dict[str, ModelTrackingProfile], dict[str, float]
    ]:
        """Get per-model forward projections and multi-hour tracking profiles.

        Returns (forward_max, forward_min, model_maxes, model_mins,
                 model_tracking, legacy_tracking_errors).

        model_tracking:  {model: ModelTrackingProfile} with multi-hour error history
        """
        col_list = ", ".join(self.MODEL_COLS.values())

        now_utc = self._now_utc()
        as_of = self._iso_utc(now_utc)

        # Get the latest snapshot at-or-before as_of.
        rows = self.db.execute(
            f"""SELECT valid_hour_utc, hours_ahead, {col_list}
                FROM forward_curves
                WHERE station = ? AND target_date = ?
                  AND snapshot_time_utc = (
                      SELECT MAX(snapshot_time_utc) FROM forward_curves
                      WHERE station = ? AND target_date = ?
                        AND snapshot_time_utc <= ?
                  )
                ORDER BY hours_ahead""",
            (self.station, target_date, self.station, target_date, as_of),
        ).fetchall()
        if not rows:
            return None, None, {}, {}, {}, {}

        # Split into past rows (verifiable) and future rows (projections)
        future_rows = []
        past_predictions: dict[str, dict[int, float]] = {m: {} for m in self.MODEL_COLS}

        for r in rows:
            ha = r["hours_ahead"]
            valid_hr = r["valid_hour_utc"]  # e.g. "18Z"
            if ha is None:
                continue

            # Parse the valid hour
            try:
                utc_hour = int(valid_hr.replace("Z", ""))
            except (ValueError, TypeError):
                continue

            if ha >= 1:
                future_rows.append(r)

            # Past or current rows: these can be verified against obs
            if ha <= 0 or (ha <= 1 and utc_hour <= now_utc.hour):
                for name, col in self.MODEL_COLS.items():
                    v = r[col]
                    if v is not None:
                        past_predictions[name][utc_hour] = v

        # Also check earlier snapshots for predictions of hours that have now passed
        # This gives us a longer verification history
        earlier_snapshots = self.db.execute(
            f"""SELECT DISTINCT snapshot_time_utc FROM forward_curves
                WHERE station = ? AND target_date = ?
                  AND snapshot_time_utc <= ?
                ORDER BY snapshot_time_utc DESC LIMIT 6""",
            (self.station, target_date, as_of),
        ).fetchall()

        for snap_row in earlier_snapshots[1:]:  # skip latest (already processed)
            snap_time = snap_row["snapshot_time_utc"]
            snap_rows = self.db.execute(
                f"""SELECT valid_hour_utc, hours_ahead, {col_list}
                    FROM forward_curves
                    WHERE station = ? AND target_date = ? AND snapshot_time_utc = ?
                    ORDER BY hours_ahead""",
                (self.station, target_date, snap_time),
            ).fetchall()
            for r in snap_rows:
                try:
                    utc_hour = int(r["valid_hour_utc"].replace("Z", ""))
                except (ValueError, TypeError):
                    continue
                # Only use predictions for hours that have now passed
                if utc_hour <= now_utc.hour:
                    for name, col in self.MODEL_COLS.items():
                        v = r[col]
                        if v is not None and utc_hour not in past_predictions[name]:
                            past_predictions[name][utc_hour] = v

        # Get actual obs by hour for verification
        obs_by_hr = self._obs_by_hour(target_date)

        # Build tracking profiles
        tracking: dict[str, ModelTrackingProfile] = {}
        legacy_errors: dict[str, float] = {}

        for model_name in self.MODEL_COLS:
            preds = past_predictions[model_name]
            errors_by_hr: dict[str, float] = {}

            for utc_hour, pred_temp in sorted(preds.items()):
                if utc_hour in obs_by_hr:
                    err = obs_by_hr[utc_hour] - pred_temp  # + = model too cold
                    errors_by_hr[f"{utc_hour:02d}Z"] = err

            if not errors_by_hr:
                tracking[model_name] = ModelTrackingProfile(model=model_name)
                continue

            err_vals = list(errors_by_hr.values())
            profile = ModelTrackingProfile(
                model=model_name,
                mean_error=sum(err_vals) / len(err_vals),
                recent_error=err_vals[-1],
                trend=err_vals[-1] - err_vals[0] if len(err_vals) > 1 else 0.0,
                n_hours=len(err_vals),
                errors_by_hour=errors_by_hr,
            )
            tracking[model_name] = profile
            legacy_errors[model_name] = profile.recent_error

        # Compute per-model forward extremes
        per_model: dict[str, list[float]] = {m: [] for m in self.MODEL_COLS}
        for r in future_rows:
            for name, col in self.MODEL_COLS.items():
                v = r[col]
                if v is not None:
                    per_model[name].append(v)

        model_maxes = {m: max(v) for m, v in per_model.items() if v}
        model_mins = {m: min(v) for m, v in per_model.items() if v}

        fwd_max = max(model_maxes.values()) if model_maxes else None
        fwd_min = min(model_mins.values()) if model_mins else None

        return fwd_max, fwd_min, model_maxes, model_mins, tracking, legacy_errors

    def _pressure_trend(self) -> tuple[float | None, bool]:
        now = self._now_utc()
        t_3hr = (now - timedelta(hours=3)).strftime("%Y-%m-%dT%H:%M:%SZ")
        t_1hr = (now - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")

        def _p(where_extra: str = "", params: tuple = ()) -> float | None:
            row = self.db.execute(
                f"""SELECT pressure_hpa FROM observations
                    WHERE station = ? AND pressure_hpa IS NOT NULL {where_extra}
                    ORDER BY timestamp_utc DESC LIMIT 1""",
                (self.station, *params),
            ).fetchone()
            return row["pressure_hpa"] if row else None

        cur = _p()
        p3 = _p("AND timestamp_utc <= ?", (t_3hr,))
        p1 = _p("AND timestamp_utc <= ?", (t_1hr,))

        trend = (cur - p3) if cur and p3 else None
        surge = ((cur - p1) > self.p.pressure_surge_threshold_hpa) if cur and p1 else False
        return trend, surge

    # --- Main extraction ---

    def extract(self, target_date: str | None = None, now_utc: datetime | None = None) -> list[SignalState]:
        """Extract SignalState for both HIGH and LOW markets."""
        prev_as_of = self._as_of_utc
        self._as_of_utc = now_utc or datetime.now(timezone.utc)
        try:
            as_of = self._now_utc()
            if target_date is None:
                target_date = self.current_target_date(as_of)

            hours_left = self._hours_remaining(as_of, target_date)
            now_str = as_of.strftime("%Y-%m-%dT%H:%M:%SZ")

            # Shared data — backfill missing dew/pressure from nearby ASOS
            obs = self._backfill_obs_from_nearby(self._latest_obs())
            obs_2hr = self._obs_at_offset(2.0)
            obs_30m = self._obs_at_offset(0.5)
            run_hi, run_lo = self._running_extremes(target_date)
            cape, pw = self._atmospheric()
            cape_trend = self._cape_trend()
            fawn_temp, fawn_dew = self._fawn_latest()
            fawn_crash = self._fawn_crash()
            nearby_div, nearby_crash_n = self._nearby_divergence()
            pres_trend, pres_surge = self._pressure_trend()
            evening_dew = self._evening_dew_mean(target_date)
            obs_temp = obs.get("temperature_f") if obs else None
            fwd_max, fwd_min, fwd_model_maxes, fwd_model_mins, fwd_tracking, _legacy_errors =                 self._forward_curve_extremes(target_date, obs_temp)

            # Wind
            wind_dir = obs.get("wind_heading_deg") if obs else None
            wind_2hr = obs_2hr.get("wind_heading_deg") if obs_2hr else None
            sp = self.p
            continental = (sp.continental_wind_min <= wind_dir <= sp.continental_wind_max) if wind_dir is not None else False
            wind_shift = (abs(_circular_diff(wind_dir, wind_2hr)) > 90) if wind_dir is not None and wind_2hr is not None else False

            # Dew crash (30 min)
            dew_crash = False
            if obs and obs_30m:
                cur_dew = obs.get("dew_point_f")
                old_dew = obs_30m.get("dew_point_f")
                if cur_dew is not None and old_dew is not None:
                    dew_crash = (old_dew - cur_dew) > sp.dew_crash_threshold_f

            # Obs trend
            obs_trend = None
            if obs and obs_2hr:
                t1, t0 = obs.get("temperature_f"), obs_2hr.get("temperature_f")
                if t1 is not None and t0 is not None:
                    obs_trend = t1 - t0

            # Outflow
            outflow = (cape is not None and pw is not None
                       and cape > sp.cape_threshold and pw > sp.pw_threshold)

            states = []
            for mtype in ("high", "low"):
                cons_f, cons_sig, n_mod = self._consensus(target_date, mtype)

                obs_vs = None
                if obs and cons_f is not None:
                    t = obs.get("temperature_f")
                    if t is not None:
                        obs_vs = t - cons_f

                # Projected extreme
                projected = None
                if obs and obs_trend is not None and hours_left > 0.5:
                    t = obs["temperature_f"]
                    if mtype == "high" and obs_trend > 0:
                        projected = t + obs_trend * min(hours_left, 4) * 0.5 * 0.7
                    elif mtype == "low" and obs_trend < 0:
                        projected = t + obs_trend * min(hours_left, 6) * 0.5 * 0.7

                # Dew floor (low only)
                est_floor = None
                if mtype == "low":
                    cur_dew = obs.get("dew_point_f") if obs else None
                    if cur_dew is not None:
                        est_floor = cur_dew + sp.dew_buffer_f
                    elif evening_dew is not None:
                        est_floor = evening_dew + sp.dew_buffer_f

                # Flags
                flags = []
                if outflow: flags.append("OUTFLOW_RISK")
                if continental: flags.append("CONTINENTAL_INTRUSION")
                if wind_shift: flags.append("WIND_SHIFT")
                if dew_crash: flags.append("DEW_POINT_CRASH")
                if fawn_crash: flags.append("FAWN_CRASH")
                if pres_surge: flags.append("PRESSURE_SURGE")
                if nearby_crash_n >= 2: flags.append("NEARBY_CRASHING")
                if obs_vs is not None:
                    if obs_vs > 2.0: flags.append("OBS_RUNNING_HOT")
                    elif obs_vs < -2.0: flags.append("OBS_RUNNING_COLD")

                states.append(SignalState(
                    timestamp_utc=now_str, station=self.station,
                    target_date=target_date, market_type=mtype,
                    hours_remaining=hours_left,
                    consensus_f=cons_f, consensus_sigma=cons_sig, n_models=n_mod,
                    obs_current_f=obs.get("temperature_f") if obs else None,
                    obs_trend_2hr=obs_trend, obs_vs_consensus=obs_vs,
                    projected_extreme_f=projected,
                    cape_current=cape, pw_mm=pw, outflow_risk=outflow,
                    cape_trend_1hr=cape_trend,
                    wind_dir_deg=wind_dir, continental_intrusion=continental,
                    wind_shift_detected=wind_shift,
                    dew_point_f=obs.get("dew_point_f") if obs else None,
                    evening_dew_mean_f=evening_dew,
                    estimated_low_floor_f=est_floor, dew_crash_active=dew_crash,
                    fawn_temp_f=fawn_temp, fawn_crash_detected=fawn_crash,
                    fawn_lead_minutes=None,
                    nearby_divergence_f=nearby_div, nearby_crash_count=nearby_crash_n,
                    pressure_hpa=obs.get("pressure_hpa") if obs else None,
                    pressure_3hr_trend=pres_trend, pressure_surge=pres_surge,
                    running_high_f=run_hi, running_low_f=run_lo,
                    forward_max_f=fwd_max, forward_min_f=fwd_min,
                    forward_model_maxes=fwd_model_maxes,
                    forward_model_mins=fwd_model_mins,
                    forward_tracking_errors=fwd_tracking,
                    active_flags=flags,
                ))

            return states
        finally:
            self._as_of_utc = prev_as_of


def format_signal_state(s: SignalState) -> str:
    """Human-readable summary for logging."""
    lines = [
        f"=== {s.station} {s.target_date} {s.market_type.upper()} | {s.hours_remaining:.1f}hr left ===",
        f"  Consensus: {s.consensus_f}°F (σ={s.consensus_sigma}, n={s.n_models})",
        f"  Obs: {s.obs_current_f}°F  2hr: {s.obs_trend_2hr}  vs_cons: {s.obs_vs_consensus}",
        f"  Running: H={s.running_high_f} L={s.running_low_f}  Projected: {s.projected_extreme_f}",
        f"  CAPE={s.cape_current} PW={s.pw_mm}mm Outflow={s.outflow_risk}",
        f"  Wind={s.wind_dir_deg}° Cont={s.continental_intrusion} Shift={s.wind_shift_detected}",
        f"  Dew={s.dew_point_f}°F Floor={s.estimated_low_floor_f} Crash={s.dew_crash_active}",
        f"  FAWN={s.fawn_temp_f}°F Crash={s.fawn_crash_detected}  Nearby: div={s.nearby_divergence_f} crash={s.nearby_crash_count}",
        f"  Pressure={s.pressure_hpa} 3hr={s.pressure_3hr_trend} Surge={s.pressure_surge}",
        f"  Flags: {s.active_flags}",
    ]
    return "\n".join(lines)
