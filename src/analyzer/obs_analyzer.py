"""Observation analyzer — all obs-vs-forecast verification in one place.

Standalone module that compares observed data against forecasts:
1. Cloud impact: obs sky cover vs NWS gridpoint forecast sky cover
2. FAWN verification: actual solar/rain/soil vs Open-Meteo atmospheric forecasts
3. Nearby stations: spatial temperature deltas vs KMIA (lead indicators)

Designed to be extended with new obs comparisons as sources are added.
Each scorer writes to the DB via Store methods (cloud_obs table for clouds,
future tables for FAWN/nearby as needed).

Returns every individual observation (not aggregated) so upstream python
scripts can filter/cut data before handing to Claude.

Run standalone:
    cd miami-project/src
    python3 -m collector.obs_analyzer [--date YYYY-MM-DD] [--db path]
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from collector.store.db import Store

log = logging.getLogger(__name__)

CLIMATE_DAY_START_UTC_HOUR = 5  # midnight EST = 05:00Z


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CloudComparison:
    """One obs-vs-forecast sky cover pairing."""
    obs_time: str
    local_hour: int
    obs_sky_pct: float
    obs_sky_code: str | None
    forecast_sky_pct: float | None
    discrepancy_pct: float | None  # obs - forecast


@dataclass
class CloudBlockSummary:
    """Per-block cloud actual/expected/delta."""
    block_name: str
    mean_obs_pct: float
    mean_forecast_pct: float | None
    mean_discrepancy_pct: float | None  # obs - forecast
    count: int


@dataclass
class FAWNObs:
    """One FAWN observation row."""
    timestamp_utc: str
    local_hour: int
    air_temp_f: float | None
    dew_point_f: float | None
    wind_speed_mph: float | None
    wind_gust_mph: float | None
    wind_direction_deg: float | None
    solar_radiation_wm2: float | None
    rain_mm: float | None
    soil_temp_c: float | None


@dataclass
class FAWNSummary:
    """Day-level FAWN summary with forecast comparisons."""
    obs_count: int
    # Solar actual/expected/delta
    fawn_peak_solar_wm2: float | None
    forecast_peak_solar_wm2: float | None
    solar_delta_wm2: float | None
    # Rain actual/expected/delta
    fawn_total_rain_mm: float | None
    forecast_total_rain_mm: float | None
    rain_delta_mm: float | None
    # Soil actual/expected/delta
    fawn_mean_soil_temp_c: float | None
    forecast_mean_soil_temp_c: float | None
    soil_temp_delta_c: float | None
    # Air temp range
    fawn_max_air_temp_f: float | None
    fawn_min_air_temp_f: float | None
    # Dew point (obs only, day mean)
    fawn_mean_dew_point_f: float | None
    # Wind (obs only, day summary)
    fawn_mean_wind_speed_mph: float | None
    fawn_max_wind_gust_mph: float | None


@dataclass
class NearbyObs:
    """One nearby station observation."""
    stid: str
    name: str | None
    distance_mi: float | None
    bearing_deg: float | None
    timestamp_utc: str
    local_hour: int
    # Temps
    air_temp_f: float | None
    temp_delta_vs_kmia: float | None
    # Wind (obs only)
    wind_speed_mph: float | None
    wind_direction_deg: float | None
    wind_gust_mph: float | None
    # Pressure & moisture
    pressure_slp_hpa: float | None
    dew_point_f: float | None
    sky_cover_code: str | None
    # Deltas vs KMIA
    pressure_delta_hpa: float | None = None
    dew_point_delta_f: float | None = None


@dataclass
class ObsReport:
    """Full observation verification report for one station-date."""
    station: str
    target_date: str
    utc_offset: int
    # Cloud
    cloud_comparisons: list[CloudComparison] = field(default_factory=list)
    cloud_block_summaries: list[CloudBlockSummary] = field(default_factory=list)
    cloud_mean_discrepancy_pct: float | None = None
    nws_temp_error_f: float | None = None
    # FAWN
    fawn_obs: list[FAWNObs] = field(default_factory=list)
    fawn_summary: FAWNSummary | None = None
    # Nearby
    nearby_obs: list[NearbyObs] = field(default_factory=list)
    # KMIA reference
    kmia_obs_max_f: float | None = None
    kmia_obs_min_f: float | None = None
    kmia_mean_wind_speed_mph: float | None = None
    kmia_mean_pressure_hpa: float | None = None
    kmia_mean_dew_point_f: float | None = None


# ═══════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def score_observations(
    db_path: str,
    station: str = "KMIA",
    target_date: str | None = None,
    utc_offset: int = -5,
) -> ObsReport:
    """Run all observation verifications for a given date.

    Returns an ObsReport with every individual observation.
    Does NOT write to DB — caller decides persistence.
    """
    conn = sqlite3.connect(db_path, timeout=10)
    try:
        if target_date is None:
            now_local = datetime.now(timezone.utc) + timedelta(hours=utc_offset)
            target_date = now_local.strftime("%Y-%m-%d")

        day_start, day_end = _climate_day_bounds(target_date)

        report = ObsReport(
            station=station, target_date=target_date, utc_offset=utc_offset,
        )

        # KMIA obs range + reference values for the day
        kmia_stats = conn.execute(
            """SELECT MAX(temperature_f), MIN(temperature_f),
                      AVG(wind_speed_mph), AVG(pressure_hpa), AVG(dew_point_f)
               FROM observations
               WHERE station=? AND timestamp_utc >= ? AND timestamp_utc < ?
               AND temperature_f IS NOT NULL""",
            (station, day_start, day_end),
        ).fetchone()
        if kmia_stats:
            report.kmia_obs_max_f = kmia_stats[0]
            report.kmia_obs_min_f = kmia_stats[1]
            report.kmia_mean_wind_speed_mph = _r1(kmia_stats[2])
            report.kmia_mean_pressure_hpa = _r1(kmia_stats[3])
            report.kmia_mean_dew_point_f = _r1(kmia_stats[4])

        # 1. Cloud impact
        clouds = _score_clouds(conn, station, target_date, day_start, day_end, utc_offset)
        report.cloud_comparisons = clouds[0]
        report.cloud_block_summaries = clouds[1]
        report.cloud_mean_discrepancy_pct = clouds[2]
        report.nws_temp_error_f = clouds[3]

        # 2. FAWN verification
        report.fawn_obs, report.fawn_summary = _score_fawn(
            conn, station, day_start, day_end, utc_offset,
        )

        # 3. Nearby station observations
        report.nearby_obs = _score_nearby(
            conn, station, day_start, day_end, utc_offset,
            report.kmia_mean_pressure_hpa, report.kmia_mean_dew_point_f,
        )

        return report
    finally:
        conn.close()


def run_daily_obs_scoring(
    store: Store, station: str, target_date: str, utc_offset: int
) -> None:
    """Runner-facing entry point: score and persist cloud obs to DB.

    Called by the scoring loop after CLI settlement arrives.
    FAWN and nearby comparisons are computed on-demand (not persisted yet).
    """
    _persist_cloud_obs(store, station, target_date, utc_offset)


# ═══════════════════════════════════════════════════════════════════════════
# 1. CLOUD IMPACT: obs sky cover vs NWS gridpoint forecast
# ═══════════════════════════════════════════════════════════════════════════

def _score_clouds(
    conn: sqlite3.Connection,
    station: str,
    target_date: str,
    day_start: str,
    day_end: str,
    utc_offset: int,
) -> tuple[list[CloudComparison], list[CloudBlockSummary], float | None, float | None]:
    """Compare observed sky cover to NWS gridpoint forecast.

    Returns (comparisons, block_summaries, mean_discrepancy, nws_temp_error).
    """
    obs_rows = conn.execute(
        """SELECT timestamp_utc, sky_cover_pct, sky_cover_code
           FROM observations
           WHERE station = ? AND lst_date = ?
             AND sky_cover_pct IS NOT NULL
           ORDER BY timestamp_utc""",
        (station, target_date),
    ).fetchall()

    if not obs_rows:
        return [], [], None, None

    forecast_sky = _load_gridpoint_sky_cover(conn, station, target_date, utc_offset)

    comparisons = []
    discrepancies = []
    for obs_ts, obs_sky_pct, obs_sky_code in obs_rows:
        local_hour = _utc_to_local_hour(obs_ts, utc_offset)
        fcst_pct = forecast_sky.get(local_hour)
        disc = round(obs_sky_pct - fcst_pct, 1) if fcst_pct is not None else None
        if disc is not None:
            discrepancies.append(disc)

        comparisons.append(CloudComparison(
            obs_time=obs_ts,
            local_hour=local_hour,
            obs_sky_pct=obs_sky_pct,
            obs_sky_code=obs_sky_code,
            forecast_sky_pct=fcst_pct,
            discrepancy_pct=disc,
        ))

    mean_disc = round(sum(discrepancies) / len(discrepancies), 1) if discrepancies else None

    # Per-block summaries
    block_summaries = _build_cloud_block_summaries(comparisons)

    # NWS temp error
    settlement = conn.execute(
        """SELECT actual_value_f FROM event_settlements
           WHERE station=? AND settlement_date=? AND market_type='high'
           AND settlement_source='cli'""",
        (station, target_date),
    ).fetchone()

    nws_fcst = conn.execute(
        """SELECT forecast_high_f FROM model_forecasts
           WHERE station=? AND forecast_date=? AND source='nws'
           AND forecast_high_f IS NOT NULL
           ORDER BY fetch_time_utc DESC LIMIT 1""",
        (station, target_date),
    ).fetchone()

    temp_error = None
    if settlement and settlement[0] and nws_fcst and nws_fcst[0]:
        temp_error = round(nws_fcst[0] - settlement[0], 1)

    return comparisons, block_summaries, mean_disc, temp_error


def _build_cloud_block_summaries(
    comparisons: list[CloudComparison],
) -> list[CloudBlockSummary]:
    """Group cloud comparisons by time block and compute actual/expected/delta."""
    blocks = {
        "morning": range(6, 12),
        "afternoon": range(12, 18),
        "evening": range(18, 24),
    }
    summaries = []
    for block_name, hr_range in blocks.items():
        block_obs = [c for c in comparisons if c.local_hour in hr_range]
        if not block_obs:
            continue
        mean_obs = sum(c.obs_sky_pct for c in block_obs) / len(block_obs)
        paired = [c for c in block_obs if c.forecast_sky_pct is not None]
        if paired:
            mean_fcst = sum(c.forecast_sky_pct for c in paired) / len(paired)
            mean_disc = sum(c.discrepancy_pct for c in paired) / len(paired)
        else:
            mean_fcst = None
            mean_disc = None
        summaries.append(CloudBlockSummary(
            block_name=block_name,
            mean_obs_pct=round(mean_obs, 1),
            mean_forecast_pct=round(mean_fcst, 1) if mean_fcst is not None else None,
            mean_discrepancy_pct=round(mean_disc, 1) if mean_disc is not None else None,
            count=len(block_obs),
        ))
    return summaries


def _persist_cloud_obs(
    store: Store, station: str, target_date: str, utc_offset: int
) -> None:
    """Write cloud comparisons to cloud_obs table (for runner)."""
    conn = store.conn

    obs_rows = conn.execute(
        """SELECT timestamp_utc, sky_cover_pct, sky_cover_code, temperature_f
           FROM observations
           WHERE station = ? AND lst_date = ?
             AND sky_cover_pct IS NOT NULL AND temperature_f IS NOT NULL
           ORDER BY timestamp_utc""",
        (station, target_date),
    ).fetchall()

    if not obs_rows:
        return

    settlement = conn.execute(
        """SELECT actual_value_f FROM event_settlements
           WHERE station=? AND settlement_date=? AND market_type='high'
           AND settlement_source='cli'""",
        (station, target_date),
    ).fetchone()

    if not settlement or settlement[0] is None:
        return

    nws_fcst = conn.execute(
        """SELECT forecast_high_f FROM model_forecasts
           WHERE station=? AND forecast_date=? AND source='nws'
           AND forecast_high_f IS NOT NULL
           ORDER BY fetch_time_utc DESC LIMIT 1""",
        (station, target_date),
    ).fetchone()

    temp_error = None
    if nws_fcst and nws_fcst[0] is not None:
        temp_error = nws_fcst[0] - settlement[0]

    forecast_sky = _load_gridpoint_sky_cover(conn, station, target_date, utc_offset)

    scored = 0
    for obs_ts, obs_sky_pct, obs_sky_code, _ in obs_rows:
        local_hour = _utc_to_local_hour(obs_ts, utc_offset)
        forecast_sky_pct = forecast_sky.get(local_hour)

        store.insert_cloud_obs(
            station=station,
            obs_time=obs_ts,
            forecast_date=target_date,
            obs_sky_pct=obs_sky_pct,
            obs_sky_code=obs_sky_code,
            forecast_sky_pct=forecast_sky_pct,
            temp_error_f=temp_error,
        )
        scored += 1

    log.info("Cloud obs: %d records for %s", scored, target_date)


# ═══════════════════════════════════════════════════════════════════════════
# 2. FAWN VERIFICATION: actual solar/rain/soil vs Open-Meteo forecast
# ═══════════════════════════════════════════════════════════════════════════

def _score_fawn(
    conn: sqlite3.Connection,
    station: str,
    day_start: str,
    day_end: str,
    utc_offset: int,
) -> tuple[list[FAWNObs], FAWNSummary | None]:
    """Return every FAWN obs + a day-level summary with forecast comparisons."""
    rows = conn.execute(
        """SELECT timestamp_utc, air_temp_f, dew_point_f, wind_speed_mph,
                  wind_gust_mph, wind_direction_deg, solar_radiation_wm2,
                  rain_mm, soil_temp_c
           FROM fawn_observations
           WHERE timestamp_utc >= ? AND timestamp_utc < ?
           ORDER BY timestamp_utc""",
        (day_start, day_end),
    ).fetchall()

    if not rows:
        return [], None

    obs_list = []
    for r in rows:
        obs_list.append(FAWNObs(
            timestamp_utc=r[0],
            local_hour=_utc_to_local_hour(r[0], utc_offset),
            air_temp_f=r[1],
            dew_point_f=r[2],
            wind_speed_mph=r[3],
            wind_gust_mph=r[4],
            wind_direction_deg=r[5],
            solar_radiation_wm2=r[6],
            rain_mm=r[7],
            soil_temp_c=r[8],
        ))

    # Compute summary from obs
    solar_vals = [o.solar_radiation_wm2 for o in obs_list if o.solar_radiation_wm2 is not None]
    rain_vals = [o.rain_mm for o in obs_list if o.rain_mm is not None]
    soil_vals = [o.soil_temp_c for o in obs_list if o.soil_temp_c is not None]
    temp_vals = [o.air_temp_f for o in obs_list if o.air_temp_f is not None]
    dew_vals = [o.dew_point_f for o in obs_list if o.dew_point_f is not None]
    wind_vals = [o.wind_speed_mph for o in obs_list if o.wind_speed_mph is not None]
    gust_vals = [o.wind_gust_mph for o in obs_list if o.wind_gust_mph is not None]

    fawn_solar = max(solar_vals) if solar_vals else None
    fawn_rain = sum(rain_vals) if rain_vals else None
    fawn_soil = sum(soil_vals) / len(soil_vals) if soil_vals else None

    # Open-Meteo atmospheric forecasts for comparison
    atmos = conn.execute(
        """SELECT MAX(shortwave_radiation),
                  SUM(COALESCE(rain_mm, 0)),
                  AVG(soil_temperature_0_7cm)
           FROM (
               SELECT shortwave_radiation, rain_mm, soil_temperature_0_7cm
               FROM atmospheric_data
               WHERE station=? AND valid_time_utc >= ? AND valid_time_utc < ?
               GROUP BY valid_time_utc HAVING MAX(id)
           )""",
        (station, day_start, day_end),
    ).fetchone()

    fcst_solar = atmos[0] if atmos else None
    fcst_rain = atmos[1] if atmos else None
    fcst_soil = atmos[2] if atmos else None

    def _delta(obs, fcst):
        if obs is not None and fcst is not None:
            return round(obs - fcst, 1)
        return None

    summary = FAWNSummary(
        obs_count=len(obs_list),
        fawn_peak_solar_wm2=round(fawn_solar, 0) if fawn_solar is not None else None,
        forecast_peak_solar_wm2=round(fcst_solar, 0) if fcst_solar is not None else None,
        solar_delta_wm2=_delta(fawn_solar, fcst_solar),
        fawn_total_rain_mm=round(fawn_rain, 1) if fawn_rain is not None else None,
        forecast_total_rain_mm=round(fcst_rain, 1) if fcst_rain is not None else None,
        rain_delta_mm=_delta(fawn_rain, fcst_rain),
        fawn_mean_soil_temp_c=round(fawn_soil, 1) if fawn_soil is not None else None,
        forecast_mean_soil_temp_c=round(fcst_soil, 1) if fcst_soil is not None else None,
        soil_temp_delta_c=_delta(fawn_soil, fcst_soil),
        fawn_max_air_temp_f=round(max(temp_vals), 1) if temp_vals else None,
        fawn_min_air_temp_f=round(min(temp_vals), 1) if temp_vals else None,
        fawn_mean_dew_point_f=round(sum(dew_vals) / len(dew_vals), 1) if dew_vals else None,
        fawn_mean_wind_speed_mph=round(sum(wind_vals) / len(wind_vals), 1) if wind_vals else None,
        fawn_max_wind_gust_mph=round(max(gust_vals), 1) if gust_vals else None,
    )

    return obs_list, summary


# ═══════════════════════════════════════════════════════════════════════════
# 3. NEARBY STATIONS: per-observation with deltas vs KMIA
# ═══════════════════════════════════════════════════════════════════════════

def _score_nearby(
    conn: sqlite3.Connection,
    station: str,
    day_start: str,
    day_end: str,
    utc_offset: int,
    kmia_pressure: float | None,
    kmia_dew_point: float | None,
) -> list[NearbyObs]:
    """Return every nearby station observation with deltas vs KMIA."""
    rows = conn.execute(
        """SELECT no.stid, ns.name, ns.distance_mi, ns.bearing_deg,
                  no.timestamp_utc, no.air_temp_f, no.temp_delta_vs_kmia,
                  no.wind_speed_mph, no.wind_direction_deg, no.wind_gust_mph,
                  no.pressure_slp_hpa, no.dew_point_f, no.sky_cover_code
           FROM nearby_observations no
           LEFT JOIN nearby_stations ns ON ns.stid = no.stid
           WHERE no.timestamp_utc >= ? AND no.timestamp_utc < ?
             AND no.air_temp_f IS NOT NULL
           ORDER BY no.timestamp_utc, ns.distance_mi""",
        (day_start, day_end),
    ).fetchall()

    obs_list = []
    for r in rows:
        pres = r[10]
        dew = r[11]
        obs_list.append(NearbyObs(
            stid=r[0],
            name=r[1],
            distance_mi=_r1(r[2]),
            bearing_deg=_r0(r[3]),
            timestamp_utc=r[4],
            local_hour=_utc_to_local_hour(r[4], utc_offset),
            air_temp_f=r[5],
            temp_delta_vs_kmia=_r1(r[6]),
            wind_speed_mph=_r1(r[7]),
            wind_direction_deg=_r0(r[8]),
            wind_gust_mph=_r1(r[9]),
            pressure_slp_hpa=_r1(pres),
            dew_point_f=_r1(dew),
            sky_cover_code=r[12],
            pressure_delta_hpa=round(pres - kmia_pressure, 1) if pres is not None and kmia_pressure is not None else None,
            dew_point_delta_f=round(dew - kmia_dew_point, 1) if dew is not None and kmia_dew_point is not None else None,
        ))

    return obs_list


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _r1(v: float | None) -> float | None:
    """Round to 1 decimal or None."""
    return round(v, 1) if v is not None else None


def _r0(v: float | None) -> float | None:
    """Round to 0 decimals or None."""
    return round(v, 0) if v is not None else None


def _climate_day_bounds(date: str) -> tuple[str, str]:
    """Return (start_utc, end_utc) for a climate day (midnight-midnight EST = 05Z-05Z)."""
    day_start = f"{date}T{CLIMATE_DAY_START_UTC_HOUR:02d}:00:00Z"
    next_day = (datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    day_end = f"{next_day}T{CLIMATE_DAY_START_UTC_HOUR:02d}:00:00Z"
    return day_start, day_end


def _utc_to_local_hour(utc_str: str, offset: int) -> int:
    """Parse UTC timestamp and return local hour."""
    try:
        clean = utc_str.replace("Z", "+00:00")
        if "T" not in clean:
            clean = clean.replace(" ", "T")
        dt = datetime.fromisoformat(clean)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return (dt + timedelta(hours=offset)).hour
    except (ValueError, AttributeError):
        return -1


def _load_gridpoint_sky_cover(
    conn: sqlite3.Connection, station: str, target_date: str, utc_offset: int
) -> dict[int, float]:
    """Load NWS gridpoint skyCover forecast for target_date, keyed by local hour."""
    rows = conn.execute(
        """SELECT payload_json FROM sse_events
           WHERE station = ? AND event_type = 'nws_gridpoint'
           ORDER BY received_at DESC LIMIT 5""",
        (station,),
    ).fetchall()

    by_hour: dict[int, float] = {}
    for (payload_str,) in rows:
        try:
            payload = json.loads(payload_str)
        except (json.JSONDecodeError, TypeError):
            continue

        sky_cover = payload.get("skyCover", [])
        if not sky_cover:
            continue

        for entry in sky_cover:
            vt = entry.get("valid_time", "")
            val = entry.get("value")
            if not vt or val is None:
                continue
            try:
                dt = datetime.fromisoformat(vt.replace("Z", "+00:00"))
                local_dt = dt + timedelta(hours=utc_offset)
                if local_dt.strftime("%Y-%m-%d") == target_date:
                    by_hour[local_dt.hour] = float(val)
            except (ValueError, AttributeError):
                continue

        if by_hour:
            break

    return by_hour


# ---------------------------------------------------------------------------
# CLI output
# ---------------------------------------------------------------------------

def format_report(report: ObsReport) -> str:
    """Pretty-print observation verification report (compact CLI summary)."""
    lines = [
        f"═══ OBS VERIFICATION: {report.station} {report.target_date} ═══",
        f"  KMIA obs range: {report.kmia_obs_max_f}°F high / {report.kmia_obs_min_f}°F low",
        "",
    ]

    # Cloud impact
    lines.append("  ── CLOUD IMPACT ──")
    if report.cloud_comparisons:
        paired = [c for c in report.cloud_comparisons if c.discrepancy_pct is not None]
        lines.append(f"  {len(report.cloud_comparisons)} obs, "
                     f"{len(paired)} paired with NWS forecast")
        if report.cloud_mean_discrepancy_pct is not None:
            arrow = "cloudier" if report.cloud_mean_discrepancy_pct > 0 else "clearer"
            lines.append(f"  Mean discrepancy: {report.cloud_mean_discrepancy_pct:+.1f}% "
                         f"({arrow} than forecast)")
        if report.nws_temp_error_f is not None:
            lines.append(f"  NWS temp error: {report.nws_temp_error_f:+.1f}°F (fcst - CLI)")
        # Per-block actual/expected/delta
        if report.cloud_block_summaries:
            lines.append(f"    {'Block':<12s} {'Actual':>7s} {'Fcst':>7s} {'Delta':>8s}")
            for bs in report.cloud_block_summaries:
                fcst_str = f"{bs.mean_forecast_pct:.1f}%" if bs.mean_forecast_pct is not None else "   -  "
                delta_str = f"{bs.mean_discrepancy_pct:+.1f}%" if bs.mean_discrepancy_pct is not None else "   -  "
                lines.append(f"    {bs.block_name:<12s} {bs.mean_obs_pct:6.1f}% {fcst_str:>7s} "
                             f"{delta_str:>8s}  (n={bs.count})")
    else:
        lines.append("  No sky cover observations")
    lines.append("")

    # FAWN verification
    lines.append("  ── FAWN GROUND TRUTH (Homestead) ──")
    f = report.fawn_summary
    if f:
        lines.append(f"  {f.obs_count} observations ({len(report.fawn_obs)} raw)")
        if f.solar_delta_wm2 is not None:
            tag = "sunnier" if f.solar_delta_wm2 > 0 else "dimmer"
            lines.append(f"  Solar: FAWN {f.fawn_peak_solar_wm2:.0f} vs fcst "
                         f"{f.forecast_peak_solar_wm2:.0f} W/m² "
                         f"(Δ{f.solar_delta_wm2:+.0f} — {tag} than predicted)")
        if f.rain_delta_mm is not None and f.fawn_total_rain_mm is not None and f.forecast_total_rain_mm is not None:
            tag = "wetter" if f.rain_delta_mm > 0 else "drier"
            lines.append(f"  Rain: FAWN {f.fawn_total_rain_mm:.1f} vs fcst "
                         f"{f.forecast_total_rain_mm:.1f} mm "
                         f"(Δ{f.rain_delta_mm:+.1f} — {tag} than predicted)")
        elif f.fawn_total_rain_mm is not None and f.fawn_total_rain_mm > 0:
            lines.append(f"  Rain: FAWN {f.fawn_total_rain_mm:.1f}mm (no forecast comparison)")
        if f.soil_temp_delta_c is not None:
            lines.append(f"  Soil: FAWN {f.fawn_mean_soil_temp_c:.1f} vs fcst "
                         f"{f.forecast_mean_soil_temp_c:.1f}°C "
                         f"(Δ{f.soil_temp_delta_c:+.1f}°C)")
        if f.fawn_max_air_temp_f:
            lines.append(f"  FAWN air temp: {f.fawn_max_air_temp_f:.1f} high / "
                         f"{f.fawn_min_air_temp_f:.1f} low")
        # New: dew point + wind summary
        dew_str = f"{f.fawn_mean_dew_point_f:.1f}°F" if f.fawn_mean_dew_point_f is not None else "-"
        wind_str = f"{f.fawn_mean_wind_speed_mph:.1f}mph avg" if f.fawn_mean_wind_speed_mph is not None else "-"
        gust_str = f", gust {f.fawn_max_wind_gust_mph:.1f}mph" if f.fawn_max_wind_gust_mph is not None else ""
        lines.append(f"  Dew point: {dew_str}  Wind: {wind_str}{gust_str}")
    else:
        lines.append("  No FAWN data for this date")
    lines.append("")

    # Nearby stations — aggregate per station for CLI display
    lines.append("  ── NEARBY STATIONS (spatial lead) ──")
    if report.nearby_obs:
        # Aggregate per station on the fly for display
        station_agg = _aggregate_nearby_for_display(report.nearby_obs)
        lines.append(
            f"  {'Station':<8s} {'Name':<18s} {'Dist':>5s} {'Bear':>5s} "
            f"{'ΔTemp':>7s} {'High':>6s} {'Low':>6s} "
            f"{'Wind':>5s} {'Pres':>7s} {'Dew':>5s} {'ΔPres':>6s} {'ΔDew':>6s} {'N':>4s}"
        )
        lines.append("  " + "─" * 100)
        for s in station_agg:
            delta_str = f"{s['mean_delta']:+.1f}" if s['mean_delta'] is not None else "  - "
            name = (s['name'] or "")[:18]
            bearing = f"{s['bearing']:.0f}°" if s['bearing'] is not None else "  -"
            dist = f"{s['dist']:.1f}" if s['dist'] is not None else "  -"
            hi = f"{s['max_t']:.1f}" if s['max_t'] is not None else "  -"
            lo = f"{s['min_t']:.1f}" if s['min_t'] is not None else "  -"
            wind = f"{s['wind']:.1f}" if s['wind'] is not None else "  -"
            pres = f"{s['pres']:.1f}" if s['pres'] is not None else "    -"
            dew = f"{s['dew']:.1f}" if s['dew'] is not None else "  -"
            dp = f"{s['dp']:+.1f}" if s['dp'] is not None else "   -"
            dd = f"{s['dd']:+.1f}" if s['dd'] is not None else "   -"
            lines.append(
                f"  {s['stid']:<8s} {name:<18s} {dist:>5s} {bearing:>5s} "
                f"{delta_str:>7s} {hi:>6s} {lo:>6s} "
                f"{wind:>5s} {pres:>7s} {dew:>5s} {dp:>6s} {dd:>6s} {s['n']:>4d}"
            )
        # Spatial summary
        warm = [s for s in station_agg if s['mean_delta'] is not None and s['mean_delta'] > 0.5]
        cool = [s for s in station_agg if s['mean_delta'] is not None and s['mean_delta'] < -0.5]
        if warm:
            lines.append(f"  {len(warm)} stations warmer than KMIA")
        if cool:
            lines.append(f"  {len(cool)} stations cooler than KMIA")
        # KMIA reference line
        ref_parts = []
        if report.kmia_mean_wind_speed_mph is not None:
            ref_parts.append(f"wind {report.kmia_mean_wind_speed_mph:.1f}mph")
        if report.kmia_mean_pressure_hpa is not None:
            ref_parts.append(f"pres {report.kmia_mean_pressure_hpa:.1f}hPa")
        if report.kmia_mean_dew_point_f is not None:
            ref_parts.append(f"dew {report.kmia_mean_dew_point_f:.1f}°F")
        if ref_parts:
            lines.append(f"  KMIA ref: {', '.join(ref_parts)}")
        lines.append(f"  ({len(report.nearby_obs)} total individual observations)")
    else:
        lines.append("  No nearby station data")

    return "\n".join(lines)


def _aggregate_nearby_for_display(
    obs_list: list[NearbyObs],
) -> list[dict]:
    """Aggregate per-obs nearby data into per-station summaries for CLI display."""
    by_station: dict[str, list[NearbyObs]] = {}
    for o in obs_list:
        if o.stid not in by_station:
            by_station[o.stid] = []
        by_station[o.stid].append(o)

    result = []
    for stid, obs in by_station.items():
        first = obs[0]
        temps = [o.air_temp_f for o in obs if o.air_temp_f is not None]
        deltas = [o.temp_delta_vs_kmia for o in obs if o.temp_delta_vs_kmia is not None]
        winds = [o.wind_speed_mph for o in obs if o.wind_speed_mph is not None]
        pressures = [o.pressure_slp_hpa for o in obs if o.pressure_slp_hpa is not None]
        dews = [o.dew_point_f for o in obs if o.dew_point_f is not None]
        p_deltas = [o.pressure_delta_hpa for o in obs if o.pressure_delta_hpa is not None]
        d_deltas = [o.dew_point_delta_f for o in obs if o.dew_point_delta_f is not None]

        result.append({
            'stid': stid,
            'name': first.name,
            'dist': first.distance_mi,
            'bearing': first.bearing_deg,
            'mean_delta': round(sum(deltas) / len(deltas), 1) if deltas else None,
            'max_t': round(max(temps), 1) if temps else None,
            'min_t': round(min(temps), 1) if temps else None,
            'wind': round(sum(winds) / len(winds), 1) if winds else None,
            'pres': round(sum(pressures) / len(pressures), 1) if pressures else None,
            'dew': round(sum(dews) / len(dews), 1) if dews else None,
            'dp': round(sum(p_deltas) / len(p_deltas), 1) if p_deltas else None,
            'dd': round(sum(d_deltas) / len(d_deltas), 1) if d_deltas else None,
            'n': len(obs),
        })

    # Sort by distance
    result.sort(key=lambda s: s['dist'] if s['dist'] is not None else 999)
    return result


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Observation scorer (obs vs forecast)")
    parser.add_argument("--date", type=str, default=None,
                        help="Target date (YYYY-MM-DD). Default: today LST")
    parser.add_argument("--db", type=str, default="../../miami_collector.db")
    parser.add_argument("--station", type=str, default="KMIA")
    parser.add_argument("--utc-offset", type=int, default=-5)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-5s %(message)s")

    report = score_observations(args.db, args.station, args.date, args.utc_offset)
    print(format_report(report))


if __name__ == "__main__":
    main()
