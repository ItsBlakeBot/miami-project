"""33-feature vector extraction for DS3M Mamba encoder.

Reads from the collector SQLite database and constructs the feature
vector that feeds the Mamba encoder at each timestep.

Feature groups:
  1. NWP model outputs (11): HRRR, RAP, NBM, GFS, ECMWF, GEFS, multi-model spread
  2. Live METAR observations (6): temp, dewpoint, wind, cloud, running max
  3. Upper air / stability (6): CAPE, CIN, T925, T850, wind 925, PW
  4. Temporal encoding (6): hour sin/cos, DOY sin/cos, lead time, DST
  5. Derived / regime indicators (4): 3-day biases, dewpoint/wind changes
"""

from __future__ import annotations

import math
import logging
import sqlite3
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import torch
from torch import Tensor

log = logging.getLogger(__name__)

# Feature specification: name → (group, default_value, description)
# v3: Expanded from 33 → 52 features for Full Send GraphMamba
FEATURE_SPEC = {
    # ── NWP model outputs (0-10) ──
    "hrrr_maxt":        ("nwp", 85.0, "HRRR 2m temp max forecast"),
    "hrrr_maxt_prev":   ("nwp", 85.0, "Previous HRRR cycle MaxT"),
    "rap_maxt":         ("nwp", 85.0, "RAP 2m temp max"),
    "nbm_maxt_50th":    ("nwp", 85.0, "NBM deterministic MaxT"),
    "nbm_maxt_10th":    ("nwp", 80.0, "NBM 10th percentile MaxT"),
    "nbm_maxt_90th":    ("nwp", 90.0, "NBM 90th percentile MaxT"),
    "gfs_maxt":         ("nwp", 85.0, "GFS 2m temp max"),
    "ecmwf_maxt":       ("nwp", 85.0, "ECMWF HRES 2m temp max"),
    "gefs_mean":        ("nwp", 85.0, "GEFS ensemble mean MaxT"),
    "gefs_spread":      ("nwp", 3.0, "GEFS ensemble std dev"),
    "model_spread":     ("nwp", 4.0, "Max-min across all models"),
    # ── Live METAR observations (11-21) ──
    "obs_temp":         ("obs", 80.0, "Current temperature °F"),
    "obs_dewpoint":     ("obs", 70.0, "Current dewpoint °F"),
    "obs_wind_dir":     ("obs", 180.0, "Wind direction degrees"),
    "obs_wind_speed":   ("obs", 8.0, "Wind speed knots"),
    "obs_wind_gust":    ("obs", 12.0, "Wind gust knots"),
    "obs_cloud_cover":  ("obs", 0.5, "Cloud cover fraction"),
    "obs_temp_max_so_far": ("obs", 80.0, "Running max temp today"),
    "obs_pressure":     ("obs", 1015.0, "Station pressure hPa"),
    "obs_visibility":   ("obs", 10.0, "Visibility miles"),
    "obs_rel_humidity": ("obs", 65.0, "Relative humidity %"),
    "obs_feels_like":   ("obs", 82.0, "Feels-like temperature °F"),
    # ── Atmospheric / stability (22-32) ──
    "cape_sfc":         ("atmos", 500.0, "Surface CAPE J/kg"),
    "cin":              ("atmos", 50.0, "CIN J/kg (positive)"),
    "lifted_index":     ("atmos", -2.0, "Lifted index °C"),
    "cloud_cover_low":  ("atmos", 0.3, "Low cloud cover fraction"),
    "cloud_cover_mid":  ("atmos", 0.2, "Mid cloud cover fraction"),
    "cloud_cover_high": ("atmos", 0.2, "High cloud cover fraction"),
    "shortwave_rad":    ("atmos", 400.0, "Shortwave radiation W/m²"),
    "direct_rad":       ("atmos", 300.0, "Direct radiation W/m²"),
    "pressure_msl":     ("atmos", 1015.0, "Mean sea level pressure hPa"),
    "soil_temp":        ("atmos", 25.0, "Soil temperature 0-7cm °C"),
    "wind_925_speed":   ("atmos", 10.0, "925 hPa wind speed kt"),
    # ── Ocean (33-34) ──
    "sst":              ("ocean", 80.0, "Sea surface temperature °F"),
    "sst_air_delta":    ("ocean", 2.0, "SST minus air temp °F"),
    # ── Temporal (35-40) ──
    "hour_sin":         ("temporal", 0.0, "sin(2pi*hour/24)"),
    "hour_cos":         ("temporal", 1.0, "cos(2pi*hour/24)"),
    "doy_sin":          ("temporal", 0.0, "sin(2pi*DOY/365.25)"),
    "doy_cos":          ("temporal", 1.0, "cos(2pi*DOY/365.25)"),
    "lead_time_hours":  ("temporal", 12.0, "Hours until CLI window close"),
    "is_dst":           ("temporal", 1.0, "Binary: DST active"),
    # ── Derived / tendency (41-51) ──
    "hrrr_3day_bias":   ("derived", 0.0, "Rolling 3-day HRRR bias"),
    "nbm_3day_bias":    ("derived", 0.0, "Rolling 3-day NBM bias"),
    "dewpoint_change_1h": ("derived", 0.0, "Dewpoint change in 1h"),
    "wind_dir_change_1h": ("derived", 0.0, "Wind dir change in 1h"),
    "pressure_change_3h": ("derived", 0.0, "Pressure change over 3h hPa"),
    "temp_change_1h":   ("derived", 0.0, "Temperature change in 1h °F"),
    "gust_ratio":       ("derived", 1.5, "Gust/sustained wind ratio"),
    "cloud_height_1":   ("derived", 50.0, "Lowest cloud base height (100s ft)"),
    "humidity_change_1h": ("derived", 0.0, "Relative humidity change 1h"),
    "solar_heating_rate": ("derived", 0.0, "Temp change per shortwave unit"),
    "dewpoint_depression": ("derived", 10.0, "Temp minus dewpoint °F"),
}

FEATURE_NAMES = list(FEATURE_SPEC.keys())
N_FEATURES = len(FEATURE_NAMES)  # 33


class FeatureVectorBuilder:
    """Extracts 33-feature vectors from the collector database.

    Maintains a rolling buffer of the last 48 hours (576 timesteps at
    5-min resolution) for Mamba sequence input.
    """

    def __init__(self, db_path: str, station: str = "KMIA", buffer_hours: int = 48):
        self.db_path = db_path
        self.station = station
        self.buffer_size = buffer_hours * 12  # 5-min resolution
        self.buffer: deque[Tensor] = deque(maxlen=self.buffer_size)

        # Cache for NWP forecasts (keyed by model+run_time)
        self._nwp_cache: dict[str, dict] = {}
        self._obs_history: deque[dict] = deque(maxlen=12)  # last 1h of obs
        self._bias_history: dict[str, deque] = {
            "hrrr": deque(maxlen=3 * 288),  # 3 days of 5-min
            "nbm": deque(maxlen=3 * 288),
        }

    def build_current(self, now_utc: datetime | None = None) -> Tensor:
        """Build feature vector for the current timestep.

        Returns: (33,) tensor
        """
        if now_utc is None:
            now_utc = datetime.now(timezone.utc)

        features = {}

        try:
            conn = sqlite3.connect(self.db_path, timeout=5)
            conn.row_factory = sqlite3.Row

            # 1. Latest observation
            obs = self._query_latest_obs(conn, now_utc)
            features.update(self._extract_obs_features(obs, now_utc))

            # 2. NWP forecasts
            nwp = self._query_latest_forecasts(conn, now_utc)
            features.update(self._extract_nwp_features(nwp))

            # 3. Atmospheric data (HRRR upper air)
            atmos = self._query_atmospheric(conn, now_utc)
            features.update(self._extract_atmos_features(atmos))

            # 4. Temporal encoding
            features.update(self._compute_temporal(now_utc))

            # 5. Derived features
            features.update(self._compute_derived(obs, now_utc))

            conn.close()
        except Exception as e:
            log.warning(f"Feature extraction error: {e}")

        # Convert to tensor in canonical order
        vec = torch.tensor(
            [features.get(name, FEATURE_SPEC[name][1]) for name in FEATURE_NAMES],
            dtype=torch.float32,
        )

        # Append to buffer
        self.buffer.append(vec)

        return vec

    def get_sequence(self, max_len: int | None = None) -> Tensor:
        """Get the buffered feature sequence for Mamba.

        Returns: (T, 33) tensor where T <= buffer_size
        """
        if not self.buffer:
            # Return single zero vector if no data yet
            return torch.zeros(1, N_FEATURES)

        seq = torch.stack(list(self.buffer))
        if max_len and seq.shape[0] > max_len:
            seq = seq[-max_len:]
        return seq

    # ── Query helpers ──────────────────────────────────────────────

    def _query_latest_obs(self, conn: sqlite3.Connection, now: datetime) -> dict:
        """Get most recent observation from DB."""
        cutoff = (now - timedelta(minutes=10)).isoformat()
        row = conn.execute(
            """SELECT temp_f, dew_pt_f, wind_dir_deg, wind_speed_kt,
                      sky_cover_pct, wethr_high_nws_f, wethr_low_nws_f, source
               FROM observations
               WHERE station = ? AND timestamp_utc > ?
               ORDER BY timestamp_utc DESC LIMIT 1""",
            (self.station, cutoff),
        ).fetchone()
        if row:
            d = dict(row)
            self._obs_history.append(d)
            return d
        return {}

    def _query_latest_forecasts(self, conn: sqlite3.Connection, now: datetime) -> list[dict]:
        """Get latest forecast from each model."""
        cutoff = (now - timedelta(hours=6)).isoformat()
        rows = conn.execute(
            """SELECT model, source, forecast_high_f, forecast_low_f,
                      raw_temperature_f, run_time
               FROM model_forecasts
               WHERE station = ? AND timestamp_utc > ?
               ORDER BY timestamp_utc DESC""",
            (self.station, cutoff),
        ).fetchall()
        return [dict(r) for r in rows]

    def _query_atmospheric(self, conn: sqlite3.Connection, now: datetime) -> dict:
        """Get latest HRRR atmospheric data."""
        cutoff = (now - timedelta(hours=3)).isoformat()
        row = conn.execute(
            """SELECT cape_j_kg, pw_mm, cloud_fraction,
                      bl_height_m, dsrc_w_m2
               FROM atmospheric_data
               WHERE station = ? AND valid_time_utc > ?
               ORDER BY valid_time_utc DESC LIMIT 1""",
            (self.station, cutoff),
        ).fetchone()
        return dict(row) if row else {}

    # ── Feature extraction ─────────────────────────────────────────

    def _extract_obs_features(self, obs: dict, now: datetime) -> dict:
        features = {}
        features["obs_temp"] = obs.get("temp_f", FEATURE_SPEC["obs_temp"][1])
        features["obs_dewpoint"] = obs.get("dew_pt_f", FEATURE_SPEC["obs_dewpoint"][1])
        features["obs_wind_dir"] = obs.get("wind_dir_deg", FEATURE_SPEC["obs_wind_dir"][1])
        features["obs_wind_speed"] = obs.get("wind_speed_kt", FEATURE_SPEC["obs_wind_speed"][1])

        sky = obs.get("sky_cover_pct")
        features["obs_cloud_cover"] = (sky / 100.0) if sky is not None else 0.5

        # Running max from NWS envelope
        high_nws = obs.get("wethr_high_nws_f")
        features["obs_temp_max_so_far"] = high_nws if high_nws else features["obs_temp"]

        return features

    def _extract_nwp_features(self, forecasts: list[dict]) -> dict:
        features = {}
        by_model: dict[str, float] = {}

        for f in forecasts:
            model = (f.get("model") or "").lower()
            source = (f.get("source") or "").lower()
            high = f.get("forecast_high_f")
            if high is None:
                continue

            key = f"{source}_{model}" if model else source
            by_model[key] = high

            # Map to feature names
            if "hrrr" in key:
                features["hrrr_maxt"] = high
            elif "rap" in key:
                features["rap_maxt"] = high
            elif "nbm" in key:
                features["nbm_maxt_50th"] = high
            elif "gfs" in key:
                features["gfs_maxt"] = high
            elif "ecmwf" in key or "ifs" in key:
                features["ecmwf_maxt"] = high

        # Model spread
        if len(by_model) >= 2:
            vals = list(by_model.values())
            features["model_spread"] = max(vals) - min(vals)
            features["gefs_mean"] = np.mean(vals)
            features["gefs_spread"] = np.std(vals)

        return features

    def _extract_atmos_features(self, atmos: dict) -> dict:
        features = {}
        features["cape_sfc"] = atmos.get("cape_j_kg", FEATURE_SPEC["cape_sfc"][1])
        pw = atmos.get("pw_mm")
        if pw is not None:
            features["t_925"] = pw  # using PW as proxy if T925 not directly available
        return features

    def _compute_temporal(self, now: datetime) -> dict:
        # All time encoding in UTC (Z time) — consistent with training data
        hour_utc = now.hour + now.minute / 60.0
        doy = now.timetuple().tm_yday

        # DST check (March second Sunday to November first Sunday, approximate)
        is_dst = 3 <= now.month <= 10

        # Lead time: hours until CLI window closes at 05:00 UTC
        cli_close = now.replace(hour=5, minute=0, second=0, microsecond=0)
        if now.hour >= 5:
            cli_close += timedelta(days=1)
        lead_time = (cli_close - now).total_seconds() / 3600.0

        return {
            "hour_sin": math.sin(2 * math.pi * hour_utc / 24),
            "hour_cos": math.cos(2 * math.pi * hour_utc / 24),
            "doy_sin": math.sin(2 * math.pi * doy / 365.25),
            "doy_cos": math.cos(2 * math.pi * doy / 365.25),
            "lead_time_hours": lead_time,
            "is_dst": 1.0 if is_dst else 0.0,
        }

    def _compute_derived(self, obs: dict, now: datetime) -> dict:
        features = {}

        # Dewpoint change over last hour
        if len(self._obs_history) >= 2:
            old_dew = self._obs_history[0].get("dew_pt_f", 70)
            new_dew = self._obs_history[-1].get("dew_pt_f", 70)
            features["dewpoint_change_1h"] = (new_dew or 70) - (old_dew or 70)

            old_wind = self._obs_history[0].get("wind_dir_deg", 180)
            new_wind = self._obs_history[-1].get("wind_dir_deg", 180)
            # Circular difference
            diff = ((new_wind or 180) - (old_wind or 180) + 180) % 360 - 180
            features["wind_dir_change_1h"] = abs(diff)

        return features

    # ── Normalization ──────────────────────────────────────────────

    @staticmethod
    def normalize(vec: Tensor, means: Tensor | None = None, stds: Tensor | None = None) -> Tensor:
        """Z-score normalization. If means/stds not provided, use defaults."""
        if means is None:
            # Sensible defaults for Miami weather features
            means = torch.tensor([
                85, 85, 85, 85, 80, 90, 85, 85, 85, 3, 4,  # NWP (11)
                80, 70, 180, 8, 0.5, 80,                     # Obs (6)
                500, -50, 22, 18, 10, 180,                    # Atmos (6)
                0, 1, 0, 1, 12, 1,                            # Temporal (6)
                0, 0, 0, 0,                                   # Derived (4)
            ], dtype=torch.float32)
        if stds is None:
            stds = torch.tensor([
                5, 5, 5, 5, 5, 5, 5, 5, 5, 2, 3,
                10, 8, 90, 8, 0.3, 10,
                800, 100, 5, 5, 10, 90,
                0.7, 0.7, 0.7, 0.7, 8, 0.5,
                1.5, 1.5, 3, 30,
            ], dtype=torch.float32)
        return (vec - means) / stds.clamp(min=0.01)
