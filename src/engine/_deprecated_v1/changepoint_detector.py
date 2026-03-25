"""Change detection for streaming weather observations.

Two-layer architecture:
  Layer 1: SPECI-style fixed thresholds — fires instantly on known
           event signatures (frontal passage, outflow, wind shift).
           Zero warm-up, zero historical data needed.

  Layer 2: Per-channel CUSUM on detrended residuals — statistical
           detection of sustained shifts in any obs variable.
           O(1) per update, 2 floats of state per channel.

Output feeds SwitchingKalmanFilter.notify_changepoint() to reweight
regime priors when a change is detected.

Layer 3 (added in T2.1): Bayesian Online Changepoint Detection (BOCPD)
on aggregate standardized residual stream.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .bocpd import BOCPDConfig, GaussianMeanBOCPD


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

@dataclass
class ChangeDetectorState:
    """Output from one change detection cycle."""
    fired: bool                              # did any layer fire?
    changepoint_probability: float           # 0-1 combined signal strength
    channels_fired: list[str] = field(default_factory=list)
    layer: int = 0                           # 1=threshold, 2=CUSUM, 3=BOCPD, 0=none
    cusum_values: dict[str, float] = field(default_factory=dict)
    bocpd_probability: float = 0.0
    bocpd_run_length_mode: int | None = None
    bocpd_run_length_expectation: float | None = None
    minutes_since_last_changepoint: float | None = None
    threshold_details: str = ""              # human-readable trigger description


# ---------------------------------------------------------------------------
# Channel definitions
# ---------------------------------------------------------------------------

# Each channel tracks one obs variable. The CUSUM runs on residuals
# from the diurnal expected value (detrended).

CHANNELS = [
    "temp_f",
    "dew_f",
    "pressure_hpa",
    "wind_speed_mph",
    "wind_gust_mph",
    "wind_dir_sin",
    "wind_dir_cos",
    "sky_cover_pct",
]

# CUSUM allowance (k) per channel — half the minimum shift we want to detect.
# Expressed in standard deviations of the detrended residual.
# Smaller k = more sensitive (fires on smaller shifts).
CUSUM_K = {
    "temp_f":         0.8,   # ~1.5°F shift (residual std ~2°F)
    "dew_f":          0.8,   # ~1.5°F shift
    "pressure_hpa":   0.6,   # ~0.3 hPa shift (residual std ~0.5 hPa)
    "wind_speed_mph": 0.8,   # ~3 mph shift
    "wind_gust_mph":  0.8,   # ~4 mph shift
    "wind_dir_sin":   0.5,   # ~15° effective shift
    "wind_dir_cos":   0.5,   # ~15° effective shift
    "sky_cover_pct":  0.8,   # ~15% shift
}

# CUSUM decision threshold (h) per channel — fires when cumulative
# deviation exceeds this many units. Higher = fewer false alarms.
CUSUM_H = {
    "temp_f":         5.0,
    "dew_f":          5.0,
    "pressure_hpa":   4.5,
    "wind_speed_mph": 5.0,
    "wind_gust_mph":  4.5,
    "wind_dir_sin":   4.0,
    "wind_dir_cos":   4.0,
    "sky_cover_pct":  5.5,
}


# ---------------------------------------------------------------------------
# Layer 1: SPECI-style fixed thresholds
# ---------------------------------------------------------------------------

@dataclass
class _ThresholdBuffer:
    """Rolling buffer for threshold checks over time windows."""
    values: list[float] = field(default_factory=list)
    timestamps: list[float] = field(default_factory=list)  # minutes since start
    max_minutes: float = 30.0

    def add(self, value: float, minutes: float) -> None:
        self.values.append(value)
        self.timestamps.append(minutes)
        # Prune old values
        cutoff = minutes - self.max_minutes
        while self.timestamps and self.timestamps[0] < cutoff:
            self.timestamps.pop(0)
            self.values.pop(0)

    def range_in_window(self, window_minutes: float, current_minutes: float) -> float:
        """Max - min over the last window_minutes."""
        cutoff = current_minutes - window_minutes
        recent = [v for v, t in zip(self.values, self.timestamps) if t >= cutoff]
        if len(recent) < 2:
            return 0.0
        return max(recent) - min(recent)

    def delta_in_window(self, window_minutes: float, current_minutes: float) -> float:
        """Last - first over the last window_minutes (signed)."""
        cutoff = current_minutes - window_minutes
        recent = [(v, t) for v, t in zip(self.values, self.timestamps) if t >= cutoff]
        if len(recent) < 2:
            return 0.0
        return recent[-1][0] - recent[0][0]


def _check_thresholds(
    buffers: dict[str, _ThresholdBuffer],
    minutes: float,
) -> tuple[bool, list[str], str]:
    """Check SPECI-style fixed thresholds.

    Returns (fired, channels, description).
    """
    fired_channels = []
    details = []

    # Temperature crash: >3°F drop in 15 minutes
    temp_delta = buffers["temp_f"].delta_in_window(15.0, minutes)
    if temp_delta < -3.0:
        fired_channels.append("temp_f")
        details.append(f"temp dropped {temp_delta:.1f}°F in 15min")

    # Temperature spike: >3°F rise in 15 minutes
    if temp_delta > 3.0:
        fired_channels.append("temp_f")
        details.append(f"temp rose {temp_delta:+.1f}°F in 15min")

    # Dew point jump/crash: >4°F change in 30 minutes
    dew_range = buffers["dew_f"].range_in_window(30.0, minutes)
    if dew_range > 4.0:
        fired_channels.append("dew_f")
        dew_delta = buffers["dew_f"].delta_in_window(30.0, minutes)
        details.append(f"dew point moved {dew_delta:+.1f}°F in 30min")

    # Pressure surge: >1.0 hPa change in 60 minutes
    pres_range = buffers["pressure_hpa"].range_in_window(30.0, minutes)
    if pres_range > 1.0:
        fired_channels.append("pressure_hpa")
        pres_delta = buffers["pressure_hpa"].delta_in_window(30.0, minutes)
        details.append(f"pressure moved {pres_delta:+.1f}hPa in 30min")

    # Wind direction shift: >60° in 15 minutes (detected via sin/cos range)
    sin_range = buffers["wind_dir_sin"].range_in_window(15.0, minutes)
    cos_range = buffers["wind_dir_cos"].range_in_window(15.0, minutes)
    dir_shift = math.sqrt(sin_range**2 + cos_range**2)
    if dir_shift > 0.85:  # ~60° shift
        fired_channels.append("wind_dir_sin")
        fired_channels.append("wind_dir_cos")
        approx_deg = math.degrees(math.asin(min(1.0, dir_shift)))
        details.append(f"wind shifted ~{approx_deg:.0f}° in 15min")

    # Gust spike: gust exceeds sustained by >15 mph
    if buffers["wind_gust_mph"].values and buffers["wind_speed_mph"].values:
        latest_gust = buffers["wind_gust_mph"].values[-1]
        latest_speed = buffers["wind_speed_mph"].values[-1]
        if latest_gust - latest_speed > 15.0:
            fired_channels.append("wind_gust_mph")
            details.append(f"gust {latest_gust:.0f} exceeds sustained {latest_speed:.0f} by {latest_gust - latest_speed:.0f}mph")

    # Sky cover jump: >60% change in 30 minutes (Miami is often partly cloudy)
    sky_range = buffers["sky_cover_pct"].range_in_window(30.0, minutes)
    if sky_range > 60.0:
        fired_channels.append("sky_cover_pct")
        details.append(f"sky cover changed {sky_range:.0f}% in 30min")

    fired = len(fired_channels) > 0
    return fired, list(set(fired_channels)), "; ".join(details)


# ---------------------------------------------------------------------------
# Layer 2: Per-channel CUSUM
# ---------------------------------------------------------------------------

@dataclass
class _CUSUMChannel:
    """CUSUM state for one obs channel."""
    name: str
    k: float           # allowance
    h: float           # decision threshold
    s_up: float = 0.0  # upper CUSUM
    s_dn: float = 0.0  # lower CUSUM (negative)

    def update(self, residual: float) -> bool:
        """Update CUSUM with a detrended residual. Returns True if fired."""
        self.s_up = max(0.0, self.s_up + residual - self.k)
        self.s_dn = min(0.0, self.s_dn + residual + self.k)

        if self.s_up > self.h or self.s_dn < -self.h:
            self.s_up = 0.0
            self.s_dn = 0.0
            return True
        return False


# ---------------------------------------------------------------------------
# Diurnal model (detrending)
# ---------------------------------------------------------------------------

class _DiurnalModel:
    """Simple 3-harmonic Fourier fit for expected value by time-of-day.

    Fits: E[x] = a0 + sum(a_k * cos(2*pi*k*t/24) + b_k * sin(2*pi*k*t/24))
    for k=1,2,3.

    If insufficient history, uses a flat mean (no detrending).
    """

    def __init__(self) -> None:
        self._coeffs: dict[str, np.ndarray] = {}  # channel -> [a0, a1, b1, a2, b2, a3, b3]
        self._stds: dict[str, float] = {}          # channel -> residual std

    def fit(self, channel: str, hours: np.ndarray, values: np.ndarray) -> None:
        """Fit the diurnal model from historical data."""
        if len(hours) < 14:  # need at least ~2 per harmonic coefficient
            self._coeffs[channel] = np.array([np.mean(values)] + [0.0] * 6)
            self._stds[channel] = max(float(np.std(values)), 0.5)
            return

        # Build design matrix for 3-harmonic Fourier
        X = np.column_stack([
            np.ones(len(hours)),
            np.cos(2 * np.pi * hours / 24),
            np.sin(2 * np.pi * hours / 24),
            np.cos(4 * np.pi * hours / 24),
            np.sin(4 * np.pi * hours / 24),
            np.cos(6 * np.pi * hours / 24),
            np.sin(6 * np.pi * hours / 24),
        ])

        # OLS fit
        coeffs, residuals, _, _ = np.linalg.lstsq(X, values, rcond=None)
        self._coeffs[channel] = coeffs

        # Residual std
        fitted = X @ coeffs
        resid = values - fitted
        self._stds[channel] = max(float(np.std(resid)), 0.5)

    def predict(self, channel: str, hour_lst: float) -> float:
        """Return expected value at given LST hour."""
        coeffs = self._coeffs.get(channel)
        if coeffs is None:
            return 0.0
        return float(
            coeffs[0]
            + coeffs[1] * math.cos(2 * math.pi * hour_lst / 24)
            + coeffs[2] * math.sin(2 * math.pi * hour_lst / 24)
            + coeffs[3] * math.cos(4 * math.pi * hour_lst / 24)
            + coeffs[4] * math.sin(4 * math.pi * hour_lst / 24)
            + coeffs[5] * math.cos(6 * math.pi * hour_lst / 24)
            + coeffs[6] * math.sin(6 * math.pi * hour_lst / 24)
        )

    def residual_std(self, channel: str) -> float:
        """Return the residual standard deviation for a channel."""
        return self._stds.get(channel, 1.0)

    def is_fitted(self, channel: str) -> bool:
        return channel in self._coeffs


# ---------------------------------------------------------------------------
# Main detector
# ---------------------------------------------------------------------------

class ChangeDetector:
    """Three-layer change detection for streaming weather observations.

    Layer 1: SPECI-style fixed thresholds (fires immediately on known
             event signatures like frontal passages, outflow, wind shifts).
    Layer 2: Per-channel CUSUM on detrended residuals (fires when
             sustained deviation accumulates past threshold).
    Layer 3: BOCPD on aggregate residual stream (probabilistic changepoint).

    Usage::

        detector = ChangeDetector()
        detector.fit_diurnal(db_path, station)  # optional: fit from history
        detector.reset()

        # Each cycle:
        state = detector.update(obs_dict, hour_lst, minutes_elapsed)
        if state.fired:
            skf.notify_changepoint(state.changepoint_probability)
    """

    def __init__(
        self,
        utc_offset: int = -5,
        cusum_config_path: str | None = None,
        *,
        use_bocpd: bool = True,
        bocpd_hazard: float = 0.03,
        bocpd_max_run_length: int = 120,
        bocpd_fire_threshold: float = 0.40,
    ):
        self.utc_offset = utc_offset
        self._diurnal = _DiurnalModel()
        self._cusum: dict[str, _CUSUMChannel] = {}
        self._buffers: dict[str, _ThresholdBuffer] = {}
        self._minutes = 0.0
        self._minutes_since_last_changepoint = 0.0

        self._use_bocpd = bool(use_bocpd)
        self._bocpd_fire_threshold = float(bocpd_fire_threshold)
        self._bocpd = (
            GaussianMeanBOCPD(
                BOCPDConfig(
                    hazard=float(bocpd_hazard),
                    max_run_length=int(bocpd_max_run_length),
                )
            )
            if self._use_bocpd
            else None
        )

        # Load calibrated CUSUM thresholds if available
        self._cusum_h = dict(CUSUM_H)
        self._cusum_k = dict(CUSUM_K)
        config_path = Path(cusum_config_path) if cusum_config_path else Path("analysis_data/cusum_config.json")
        if config_path.exists():
            import json
            with open(config_path) as f:
                cfg = json.load(f)
            self._cusum_h.update(cfg.get("h", {}))
            self._cusum_k.update(cfg.get("k", {}))

    def fit_diurnal(
        self,
        db_path: str,
        station: str = "KMIA",
        lookback_days: int = 14,
    ) -> None:
        """Fit the diurnal model from historical obs.

        Queries the last `lookback_days` of observations and fits a
        3-harmonic Fourier model for each channel.
        """
        import sqlite3
        from datetime import datetime, timedelta

        conn = sqlite3.connect(db_path, timeout=10)
        try:
            cutoff = (datetime.utcnow() - timedelta(days=lookback_days)).isoformat() + "Z"
            rows = conn.execute(
                """SELECT timestamp_utc, temperature_f, dew_point_f,
                          pressure_hpa, wind_speed_mph, wind_gust_mph,
                          wind_heading_deg, sky_cover_pct
                   FROM observations
                   WHERE station = ? AND timestamp_utc >= ?
                     AND temperature_f IS NOT NULL
                   ORDER BY timestamp_utc""",
                (station, cutoff),
            ).fetchall()
        finally:
            conn.close()

        if len(rows) < 50:
            return  # not enough data

        # Extract hours (LST) and values
        hours = []
        vals: dict[str, list[float]] = {ch: [] for ch in CHANNELS}

        for r in rows:
            ts, temp, dew, pres, wspd, gust, wdir, sky = r
            try:
                hr_utc = int(ts[11:13]) + int(ts[14:16]) / 60.0
                hr_lst = (hr_utc + self.utc_offset) % 24
            except (IndexError, ValueError):
                continue

            hours.append(hr_lst)
            vals["temp_f"].append(temp)
            vals["dew_f"].append(dew or temp - 10.0)
            vals["pressure_hpa"].append(pres or 1013.0)
            vals["wind_speed_mph"].append(wspd or 0.0)
            vals["wind_gust_mph"].append(gust or wspd or 0.0)
            rad = math.radians(wdir or 0.0)
            vals["wind_dir_sin"].append(math.sin(rad))
            vals["wind_dir_cos"].append(math.cos(rad))
            vals["sky_cover_pct"].append(sky or 0.0)

        hours_arr = np.array(hours)
        for ch in CHANNELS:
            self._diurnal.fit(ch, hours_arr, np.array(vals[ch]))

    def reset(self) -> None:
        """Reset all state for a new climate day. Uses calibrated thresholds if available."""
        self._cusum = {
            ch: _CUSUMChannel(name=ch, k=self._cusum_k.get(ch, CUSUM_K[ch]),
                              h=self._cusum_h.get(ch, CUSUM_H[ch]))
            for ch in CHANNELS
        }
        self._buffers = {ch: _ThresholdBuffer() for ch in CHANNELS}
        self._minutes = 0.0
        self._minutes_since_last_changepoint = 0.0
        if self._bocpd is not None:
            self._bocpd.reset()

    def update(
        self,
        obs: dict[str, float | None],
        hour_lst: float,
        minutes_elapsed: float,
    ) -> ChangeDetectorState:
        """Process one observation cycle.

        Parameters
        ----------
        obs : dict
            Raw observation values. Keys should include any of:
            temp_f, dew_f, pressure_hpa, wind_speed_mph, wind_gust_mph,
            wind_dir_deg (will be converted to sin/cos), sky_cover_pct.
        hour_lst : float
            Current hour in LST (0-24, fractional).
        minutes_elapsed : float
            Minutes elapsed since the prior update cycle.
        """
        delta_minutes = max(0.0, float(minutes_elapsed))
        self._minutes += delta_minutes
        current_minutes = self._minutes

        # Convert wind direction to sin/cos
        wdir = obs.get("wind_dir_deg")
        if wdir is not None:
            rad = math.radians(wdir)
            obs_channels = {
                "temp_f": obs.get("temp_f"),
                "dew_f": obs.get("dew_f"),
                "pressure_hpa": obs.get("pressure_hpa"),
                "wind_speed_mph": obs.get("wind_speed_mph"),
                "wind_gust_mph": obs.get("wind_gust_mph"),
                "wind_dir_sin": math.sin(rad),
                "wind_dir_cos": math.cos(rad),
                "sky_cover_pct": obs.get("sky_cover_pct"),
            }
        else:
            obs_channels = {
                "temp_f": obs.get("temp_f"),
                "dew_f": obs.get("dew_f"),
                "pressure_hpa": obs.get("pressure_hpa"),
                "wind_speed_mph": obs.get("wind_speed_mph"),
                "wind_gust_mph": obs.get("wind_gust_mph"),
                "wind_dir_sin": None,
                "wind_dir_cos": None,
                "sky_cover_pct": obs.get("sky_cover_pct"),
            }

        # Update rolling buffers for threshold checks
        for ch in CHANNELS:
            val = obs_channels.get(ch)
            if val is not None and ch in self._buffers:
                self._buffers[ch].add(val, current_minutes)

        # Layer 1: Fixed thresholds
        l1_fired, l1_channels, l1_details = _check_thresholds(
            self._buffers, current_minutes
        )

        # Layer 2: CUSUM on detrended residuals
        l2_fired_channels = []
        cusum_vals = {}
        residual_stream: list[float] = []

        for ch in CHANNELS:
            val = obs_channels.get(ch)
            if val is None:
                cusum_vals[ch] = 0.0
                continue

            # Detrend: compute residual from diurnal expectation
            if self._diurnal.is_fitted(ch):
                expected = self._diurnal.predict(ch, hour_lst)
                std = self._diurnal.residual_std(ch)
                residual = (val - expected) / std  # standardized residual
                residual_stream.append(float(residual))
            else:
                residual = 0.0  # no model → no signal

            cusum_ch = self._cusum.get(ch)
            if cusum_ch and cusum_ch.update(residual):
                l2_fired_channels.append(ch)

            cusum_vals[ch] = max(
                abs(cusum_ch.s_up) if cusum_ch else 0.0,
                abs(cusum_ch.s_dn) if cusum_ch else 0.0,
            )

        # Layer 3: BOCPD on aggregate standardized residual stream
        bocpd_prob = 0.0
        l3_fired = False
        bocpd_run_length_mode: int | None = None
        bocpd_run_length_expectation: float | None = None

        if self._bocpd is not None and residual_stream:
            aggregate_residual = float(np.mean(residual_stream))
            bocpd_prob = float(self._bocpd.update(aggregate_residual))
            l3_fired = bocpd_prob >= self._bocpd_fire_threshold

        if self._bocpd is not None:
            posterior = self._bocpd.run_length_posterior
            if posterior.size > 0:
                bocpd_run_length_mode = int(np.argmax(posterior))
                bocpd_run_length_expectation = float(
                    np.dot(np.arange(posterior.size, dtype=float), posterior)
                )

        fired_any = bool(l1_fired or l2_fired_channels or l3_fired)
        if fired_any:
            self._minutes_since_last_changepoint = 0.0
        else:
            self._minutes_since_last_changepoint += delta_minutes
        minutes_since_last_changepoint = round(self._minutes_since_last_changepoint, 3)

        # Combine layers — BOCPD is PRIMARY (T2.2: CUSUM downgraded to backup)
        # Priority: Layer 3 (BOCPD) > Layer 1 (threshold) > Layer 2 (CUSUM)
        # BOCPD provides probabilistic changepoint detection with proper run-length
        # tracking. Threshold and CUSUM serve as backup for cold-start and fallback.

        if l3_fired:
            # BOCPD fires — use its probability directly (PRIMARY detector)
            all_channels = list(set(l1_channels + l2_fired_channels))
            return ChangeDetectorState(
                fired=True,
                changepoint_probability=bocpd_prob,
                channels_fired=all_channels,
                layer=3,
                cusum_values=cusum_vals,
                bocpd_probability=bocpd_prob,
                bocpd_run_length_mode=bocpd_run_length_mode,
                bocpd_run_length_expectation=(
                    round(bocpd_run_length_expectation, 3)
                    if bocpd_run_length_expectation is not None
                    else None
                ),
                minutes_since_last_changepoint=minutes_since_last_changepoint,
                threshold_details="BOCPD fired on aggregate residual stream",
            )

        if l1_fired:
            # Layer 1 (threshold) fires as backup — compute probability
            # based on number of channels, boosted by BOCPD probability
            n_channels = len(l1_channels)
            cp_prob = min(1.0, max(bocpd_prob, 0.3 + 0.2 * n_channels))
            return ChangeDetectorState(
                fired=True,
                changepoint_probability=cp_prob,
                channels_fired=l1_channels,
                layer=1,
                cusum_values=cusum_vals,
                bocpd_probability=bocpd_prob,
                bocpd_run_length_mode=bocpd_run_length_mode,
                bocpd_run_length_expectation=(
                    round(bocpd_run_length_expectation, 3)
                    if bocpd_run_length_expectation is not None
                    else None
                ),
                minutes_since_last_changepoint=minutes_since_last_changepoint,
                threshold_details=l1_details,
            )

        if l2_fired_channels:
            # Layer 2 (CUSUM) fires as diagnostic/backup only
            # CUSUM is now secondary to BOCPD — only fires when BOCPD hasn't
            n_channels = len(l2_fired_channels)
            cp_prob = min(1.0, max(bocpd_prob, 0.2 + 0.15 * n_channels))
            return ChangeDetectorState(
                fired=True,
                changepoint_probability=cp_prob,
                channels_fired=l2_fired_channels,
                layer=2,
                cusum_values=cusum_vals,
                bocpd_probability=bocpd_prob,
                bocpd_run_length_mode=bocpd_run_length_mode,
                bocpd_run_length_expectation=(
                    round(bocpd_run_length_expectation, 3)
                    if bocpd_run_length_expectation is not None
                    else None
                ),
                minutes_since_last_changepoint=minutes_since_last_changepoint,
                threshold_details=f"CUSUM backup fired: {', '.join(l2_fired_channels)}",
            )

        # Nothing fired — report BOCPD monitoring state
            return ChangeDetectorState(
                fired=True,
                changepoint_probability=bocpd_prob,
                channels_fired=[],
                layer=3,
                cusum_values=cusum_vals,
                bocpd_probability=bocpd_prob,
                bocpd_run_length_mode=bocpd_run_length_mode,
                bocpd_run_length_expectation=(
                    round(bocpd_run_length_expectation, 3)
                    if bocpd_run_length_expectation is not None
                    else None
                ),
                minutes_since_last_changepoint=minutes_since_last_changepoint,
                threshold_details="BOCPD fired on aggregate residual stream",
            )

        return ChangeDetectorState(
            fired=False,
            changepoint_probability=bocpd_prob,
            cusum_values=cusum_vals,
            bocpd_probability=bocpd_prob,
            bocpd_run_length_mode=bocpd_run_length_mode,
            bocpd_run_length_expectation=(
                round(bocpd_run_length_expectation, 3)
                if bocpd_run_length_expectation is not None
                else None
            ),
            minutes_since_last_changepoint=minutes_since_last_changepoint,
        )
