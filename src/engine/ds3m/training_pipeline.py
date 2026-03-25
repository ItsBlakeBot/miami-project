"""End-to-end training pipeline for DS3M.

Multi-task pre-training on IEM historical data, then fine-tuning on
KMIA settlement data.  Trains the full differentiable pipeline:
  Mamba encoder → DPF → NSF → CRPS loss

Training strategy:
  1. Pre-train Mamba encoder on 255K IEM observations (multi-task)
     - Predict hourly temps, remaining moves, running extremes
     - Uses 10 SE Florida stations with station embeddings
     - Gives Mamba ~17K targets/day instead of 1
  2. Fine-tune emission + PF on KMIA settlements
     - CRPS loss on daily max prediction
     - Bracket Brier score as auxiliary loss
     - 120-day rolling window, retrained nightly
  3. Train NSF on particle cloud → settlement outcome
     - Learns to convert particle statistics into optimal density
     - Trained on bracket-level outcomes (10x more data than settlements)
"""

from __future__ import annotations

import logging
import math
import sqlite3
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from engine.ds3m.mamba_encoder import MambaEncoder, MambaConfig
from engine.ds3m.graph_mamba import GraphMambaEncoder, GraphMambaConfig
from engine.ds3m.diff_particle_filter import (
    DifferentiableParticleFilter, DPFConfig, ParticleCloud,
)
from engine.ds3m.neural_spline_flow import ConditionalNSF, NSFConfig
from engine.ds3m.skew_normal import SkewNormal
from engine.ds3m.feature_vector import FeatureVectorBuilder, N_FEATURES
from engine.ds3m.augmentation import SequenceAugmentor

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Datasets
# ──────────────────────────────────────────────────────────────────────

class IEMPretrainingDataset(Dataset):
    """Enriched multi-task pre-training dataset from IEM + Open-Meteo + NDBC.

    Uses enriched_asos (hourly, all fields), enriched_atmosphere (CAPE/CIN/cloud),
    and enriched_sst (buoy SST) tables. All timestamps in UTC.

    For each 48h window at each station, targets:
      - Next-step temperature prediction (hourly)
      - Remaining move to daily max/min
      - Daily max temperature (settlement proxy from CLI backfill)

    Populates all 33 features where data is available:
      - [0-10]  NWP: filled with obs temp (no historical NWP), spread=0
      - [11-16] Obs: temp, dewpoint, wind dir/speed, cloud cover, running max
      - [17-22] Atmos: CAPE, CIN, pressure (as T925 proxy), cloud cover, wind
      - [23-28] Temporal: UTC hour sin/cos, DOY sin/cos, lead time, DST flag
      - [29-32] Derived: dewpoint change, wind change (computed from sequence)
    """

    # Sky cover code → fraction mapping
    SKY_FRAC = {"CLR": 0.0, "FEW": 0.15, "SCT": 0.35, "BKN": 0.65, "OVC": 1.0, "VV": 1.0}

    def __init__(
        self,
        db_path: str,
        stations: list[str] | None = None,
        lookback_hours: int = 48,
        step_size_hours: int = 1,
        augment: bool = True,
    ):
        self.lookback = lookback_hours
        self.step_size = step_size_hours
        self.augmentor = SequenceAugmentor(enabled=augment)

        if stations is None:
            stations = [
                "KMIA", "KOPF", "KTMB", "KHWO", "KFLL",
                "KHST", "KFXE", "KPMP", "KBCT", "KPBI",
                "KX51", "KIMM", "KPHK", "KOBE", "KSUA",
                "KMTH", "KNQX", "KEYW", "KAPF", "KRSW",
                "KFMY", "KPGD", "KFPR", "KSEF", "KVRB",
                "KMLB", "KBOW", "KSRQ", "KLAL", "KMCO",
                "KSPG", "KORL", "KTPA", "KPIE", "KSFB",
            ]

        self.station_to_idx = {s: i for i, s in enumerate(stations)}
        self.n_stations = len(stations)

        self.samples = []
        self._load_enriched(db_path, stations)

    def _load_enriched(self, db_path: str, stations: list[str]):
        """Load from enriched_asos + enriched_atmosphere + cli_daily_backfill.

        Pre-computes ALL 33 features into tensors at init time so __getitem__
        is a simple slice + augmentation (no Python loops, no dict lookups).
        """
        conn = sqlite3.connect(db_path, timeout=10)
        conn.row_factory = sqlite3.Row

        # Pre-load atmospheric data indexed by hour (UTC)
        atmos_by_hour = self._load_atmosphere(conn)
        log.info(f"Loaded {len(atmos_by_hour)} atmospheric hours (CAPE/CIN/cloud)")

        # Pre-load SST indexed by date
        sst_by_date = self._load_sst(conn)
        log.info(f"Loaded {len(sst_by_date)} SST daily averages")

        # We'll store pre-computed feature tensors per station, then slice
        self._feature_tensors = []  # list of (features_tensor, metadata_list)

        for station in stations:
            log.info(f"Loading enriched ASOS for {station}...")

            rows = conn.execute(
                """SELECT timestamp_utc, temp_f, dewpoint_f, wind_dir_deg,
                          wind_speed_kt, wind_gust_kt, pressure_hpa,
                          visibility_mi, sky_cover_1, sky_cover_2, sky_cover_3,
                          rel_humidity, precip_1h_in
                   FROM enriched_asos
                   WHERE station = ? AND temp_f IS NOT NULL
                   ORDER BY timestamp_utc""",
                (station,),
            ).fetchall()

            if len(rows) < self.lookback + 10:
                rows = self._fallback_load(conn, station)

            if len(rows) < self.lookback:
                log.warning(f"Insufficient data for {station}: {len(rows)} rows")
                continue

            n = len(rows)
            log.info(f"  {station}: {n} hourly obs")

            # Parse into arrays
            data = self._parse_rows(rows)
            daily_max = self._get_daily_max(conn, station, data["timestamps"], data["temp_f"])

            # ── Pre-compute full (N, 52) feature tensor for this station ──
            all_features = self._precompute_features(data, atmos_by_hour, n, sst_by_date)
            log.info(f"  {station}: features pre-computed ({all_features.shape})")

            # Create training windows (slices into the pre-computed tensor)
            station_idx = self.station_to_idx[station]
            for start in range(0, n - self.lookback - 1, self.step_size):
                end = start + self.lookback
                if end >= n:
                    break

                ts_end = data["timestamps"][end]
                day_key = ts_end[:10]
                cli_max = daily_max.get(day_key)
                if cli_max is None:
                    continue

                window_temps = data["temp_f"][start:end]
                next_temp = data["temp_f"][end]

                self.samples.append({
                    "station_idx": station_idx,
                    "features": all_features[start:end],  # (T, 33) pre-computed tensor
                    "next_temp": float(next_temp),
                    "remaining_high": max(0.0, float(cli_max) - float(window_temps[-1])),
                    "cli_max": float(cli_max),
                })

        conn.close()
        log.info(f"Loaded {len(self.samples)} enriched pre-training samples "
                 f"from {len(stations)} stations")

    def _precompute_features(self, data: dict, atmos_by_hour: dict, n: int,
                             sst_by_date: dict | None = None) -> torch.Tensor:
        """Pre-compute all 52 features for an entire station's time series.

        Returns (N, 52) float32 tensor. Vectorized numpy/torch ops throughout.
        Feature indices match FEATURE_SPEC order in feature_vector.py v3.
        """
        features = torch.zeros(n, N_FEATURES, dtype=torch.float32)
        timestamps = data["timestamps"]

        # ── Parse raw arrays ──
        temp_f = np.nan_to_num(data["temp_f"], nan=75.0).astype(np.float32)
        dewpoint_f = np.nan_to_num(data["dewpoint_f"], nan=65.0).astype(np.float32)
        wind_dir = np.nan_to_num(data["wind_dir_deg"], nan=180.0).astype(np.float32)
        wind_spd = np.nan_to_num(data["wind_speed_kt"], nan=5.0).astype(np.float32)
        wind_gust = np.nan_to_num(data["wind_gust_kt"], nan=8.0).astype(np.float32)
        cloud = np.nan_to_num(data["cloud_cover"], nan=0.5).astype(np.float32)
        pressure = np.nan_to_num(data["pressure_hpa"], nan=1015.0).astype(np.float32)
        visibility = np.nan_to_num(data["visibility_mi"], nan=10.0).astype(np.float32)
        rel_hum = np.nan_to_num(data["rel_humidity"], nan=65.0).astype(np.float32)
        feels_like = np.nan_to_num(data.get("feels_like_f", np.full(n, np.nan)), nan=82.0).astype(np.float32)
        sky_h1 = np.nan_to_num(data.get("sky_height_1", np.full(n, np.nan)), nan=50.0).astype(np.float32)

        # ── NWP proxies (0-10) — obs temp as stand-in for historical training ──
        features[:, 0] = torch.from_numpy(temp_f)      # hrrr_maxt
        features[:, 3] = torch.from_numpy(temp_f)      # nbm_maxt_50th
        features[:, 6] = torch.from_numpy(temp_f)      # gfs_maxt

        # ── Obs features (11-21) ──
        features[:, 11] = torch.from_numpy(temp_f)
        features[:, 12] = torch.from_numpy(dewpoint_f)
        features[:, 13] = torch.from_numpy(wind_dir)
        features[:, 14] = torch.from_numpy(wind_spd)
        features[:, 15] = torch.from_numpy(wind_gust)
        features[:, 16] = torch.from_numpy(cloud)
        features[:, 17] = torch.from_numpy(np.maximum.accumulate(temp_f))  # running max
        features[:, 18] = torch.from_numpy(pressure)
        features[:, 19] = torch.from_numpy(visibility)
        features[:, 20] = torch.from_numpy(rel_hum)
        features[:, 21] = torch.from_numpy(feels_like)

        # ── Atmospheric / stability (22-32) — from Open-Meteo ──
        cape_arr = np.full(n, 500.0, dtype=np.float32)
        cin_arr = np.full(n, 50.0, dtype=np.float32)
        li_arr = np.full(n, -2.0, dtype=np.float32)
        cloud_low = np.full(n, 0.3, dtype=np.float32)
        cloud_mid = np.full(n, 0.2, dtype=np.float32)
        cloud_high = np.full(n, 0.2, dtype=np.float32)
        sw_rad = np.full(n, 400.0, dtype=np.float32)
        dir_rad = np.full(n, 300.0, dtype=np.float32)
        pres_msl = np.full(n, 1015.0, dtype=np.float32)
        soil_t = np.full(n, 25.0, dtype=np.float32)

        for t in range(n):
            ts = timestamps[t]
            hour_key = ts[:13] if len(ts) >= 13 else ts
            atm = atmos_by_hour.get(hour_key)
            if atm:
                if atm.get("cape") is not None: cape_arr[t] = atm["cape"]
                if atm.get("cin") is not None: cin_arr[t] = abs(atm["cin"])
                if atm.get("li") is not None: li_arr[t] = atm["li"]
                if atm.get("cloud_low") is not None: cloud_low[t] = atm["cloud_low"] / 100.0
                if atm.get("cloud_mid") is not None: cloud_mid[t] = atm["cloud_mid"] / 100.0
                if atm.get("cloud_high") is not None: cloud_high[t] = atm["cloud_high"] / 100.0
                if atm.get("sw_rad") is not None: sw_rad[t] = atm["sw_rad"]
                if atm.get("dir_rad") is not None: dir_rad[t] = atm["dir_rad"]
                if atm.get("pressure") is not None: pres_msl[t] = atm["pressure"]
                if atm.get("soil_temp") is not None: soil_t[t] = atm["soil_temp"]

        features[:, 22] = torch.from_numpy(cape_arr)
        features[:, 23] = torch.from_numpy(cin_arr)
        features[:, 24] = torch.from_numpy(li_arr)
        features[:, 25] = torch.from_numpy(cloud_low)
        features[:, 26] = torch.from_numpy(cloud_mid)
        features[:, 27] = torch.from_numpy(cloud_high)
        features[:, 28] = torch.from_numpy(sw_rad)
        features[:, 29] = torch.from_numpy(dir_rad)
        features[:, 30] = torch.from_numpy(pres_msl)
        features[:, 31] = torch.from_numpy(soil_t)
        features[:, 32] = torch.from_numpy(wind_spd)  # wind_925 ≈ surface wind

        # ── Ocean (33-34) — SST from NDBC buoy ──
        if sst_by_date:
            for t in range(n):
                day_key = timestamps[t][:10]
                sst_val = sst_by_date.get(day_key)
                if sst_val is not None:
                    features[t, 33] = float(sst_val)
                    features[t, 34] = float(sst_val) - float(temp_f[t])
                else:
                    features[t, 33] = 80.0
                    features[t, 34] = 2.0
        else:
            features[:, 33] = 80.0
            features[:, 34] = 2.0

        # ── Temporal (35-40) ──
        hours_utc = np.zeros(n, dtype=np.float32)
        doys = np.zeros(n, dtype=np.float32)
        months = np.zeros(n, dtype=np.float32)

        for t in range(n):
            ts = timestamps[t]
            try:
                if " " in ts:
                    dt = datetime.strptime(ts[:19], "%Y-%m-%d %H:%M:%S")
                else:
                    dt = datetime.strptime(ts[:19], "%Y-%m-%dT%H:%M:%S")
                hours_utc[t] = dt.hour + dt.minute / 60.0
                doys[t] = dt.timetuple().tm_yday
                months[t] = dt.month
            except ValueError:
                hours_utc[t] = 12.0
                doys[t] = 180
                months[t] = 6

        two_pi = 2.0 * np.pi
        features[:, 35] = torch.from_numpy(np.sin(two_pi * hours_utc / 24.0))
        features[:, 36] = torch.from_numpy(np.cos(two_pi * hours_utc / 24.0))
        features[:, 37] = torch.from_numpy(np.sin(two_pi * doys / 365.25))
        features[:, 38] = torch.from_numpy(np.cos(two_pi * doys / 365.25))
        features[:, 39] = torch.from_numpy((5.0 - hours_utc) % 24.0)
        dst_mask = (months >= 3) & (months <= 10)
        features[:, 40] = torch.from_numpy(dst_mask.astype(np.float32))

        # ── Derived / tendency (41-51) ──
        # 41-42: model biases (zero for historical pre-training)
        # 43: dewpoint change 1h
        features[:, 43] = torch.from_numpy(np.diff(dewpoint_f, prepend=dewpoint_f[0]))
        # 44: wind dir change 1h (circular)
        wd = wind_dir.astype(np.float64)
        wd_diff = np.diff(wd, prepend=wd[0])
        wd_diff = ((wd_diff + 180) % 360) - 180
        features[:, 44] = torch.from_numpy(wd_diff.astype(np.float32))
        # 45: pressure change 3h
        pres_3h = np.zeros(n, dtype=np.float32)
        pres_3h[3:] = pressure[3:] - pressure[:-3]
        features[:, 45] = torch.from_numpy(pres_3h)
        # 46: temp change 1h
        features[:, 46] = torch.from_numpy(np.diff(temp_f, prepend=temp_f[0]))
        # 47: gust ratio (gust / sustained)
        gust_ratio = np.where(wind_spd > 0.5, wind_gust / wind_spd, 1.5)
        features[:, 47] = torch.from_numpy(gust_ratio.astype(np.float32))
        # 48: cloud base height
        features[:, 48] = torch.from_numpy(sky_h1)
        # 49: humidity change 1h
        features[:, 49] = torch.from_numpy(np.diff(rel_hum, prepend=rel_hum[0]))
        # 50: solar heating rate (temp change per shortwave unit)
        sw_safe = np.maximum(sw_rad, 1.0)
        temp_change = np.diff(temp_f, prepend=temp_f[0])
        features[:, 50] = torch.from_numpy((temp_change / sw_safe * 100).astype(np.float32))
        # 51: dewpoint depression
        features[:, 51] = torch.from_numpy(temp_f - dewpoint_f)

        # Clean up NaN
        features = torch.nan_to_num(features, nan=0.0)

        return features

    def _parse_rows(self, rows) -> dict[str, np.ndarray | list]:
        """Parse DB rows into parallel arrays."""
        n = len(rows)
        data = {
            "timestamps": [],
            "temp_f": np.full(n, np.nan, dtype=np.float32),
            "dewpoint_f": np.full(n, np.nan, dtype=np.float32),
            "wind_dir_deg": np.full(n, np.nan, dtype=np.float32),
            "wind_speed_kt": np.full(n, np.nan, dtype=np.float32),
            "wind_gust_kt": np.full(n, np.nan, dtype=np.float32),
            "pressure_hpa": np.full(n, np.nan, dtype=np.float32),
            "visibility_mi": np.full(n, np.nan, dtype=np.float32),
            "cloud_cover": np.full(n, np.nan, dtype=np.float32),
            "rel_humidity": np.full(n, np.nan, dtype=np.float32),
            "precip_1h_in": np.full(n, np.nan, dtype=np.float32),
            "feels_like_f": np.full(n, np.nan, dtype=np.float32),
            "sky_height_1": np.full(n, np.nan, dtype=np.float32),
        }
        row_keys = rows[0].keys() if len(rows) > 0 else []
        for i, r in enumerate(rows):
            data["timestamps"].append(r["timestamp_utc"])
            data["temp_f"][i] = r["temp_f"] if r["temp_f"] is not None else np.nan
            data["dewpoint_f"][i] = r["dewpoint_f"] if r["dewpoint_f"] is not None else np.nan
            data["wind_dir_deg"][i] = r["wind_dir_deg"] if r["wind_dir_deg"] is not None else np.nan
            data["wind_speed_kt"][i] = r["wind_speed_kt"] if r["wind_speed_kt"] is not None else np.nan
            data["wind_gust_kt"][i] = r["wind_gust_kt"] if r["wind_gust_kt"] is not None else np.nan
            data["pressure_hpa"][i] = r["pressure_hpa"] if r["pressure_hpa"] is not None else np.nan
            data["visibility_mi"][i] = r["visibility_mi"] if r["visibility_mi"] is not None else np.nan
            data["rel_humidity"][i] = r["rel_humidity"] if r["rel_humidity"] is not None else np.nan
            data["precip_1h_in"][i] = r["precip_1h_in"] if r["precip_1h_in"] is not None else np.nan

            if "feels_like_f" in row_keys:
                data["feels_like_f"][i] = r["feels_like_f"] if r["feels_like_f"] is not None else np.nan
            if "sky_height_1" in row_keys:
                data["sky_height_1"][i] = r["sky_height_1"] if r["sky_height_1"] is not None else np.nan

            # Parse sky cover
            for key in ("sky_cover_1", "sky_cover_2", "sky_cover_3"):
                code = r[key] if key in r.keys() else None
                if code and code in self.SKY_FRAC:
                    frac = self.SKY_FRAC[code]
                    if np.isnan(data["cloud_cover"][i]) or frac > data["cloud_cover"][i]:
                        data["cloud_cover"][i] = frac

        return data

    def _fallback_load(self, conn, station):
        """Fall back to original observation tables if enriched_asos is empty."""
        if station == "KMIA":
            return conn.execute(
                """SELECT timestamp_utc, temperature_f as temp_f,
                          dew_point_f as dewpoint_f, wind_heading_deg as wind_dir_deg,
                          wind_speed_mph as wind_speed_kt, wind_gust_mph as wind_gust_kt,
                          pressure_hpa, visibility_miles as visibility_mi,
                          sky_cover_code as sky_cover_1, NULL as sky_cover_2,
                          NULL as sky_cover_3, relative_humidity as rel_humidity,
                          precipitation_last_hour_mm as precip_1h_in
                   FROM observations
                   WHERE station = ? AND temperature_f IS NOT NULL
                   ORDER BY timestamp_utc""",
                (station,),
            ).fetchall()
        else:
            return conn.execute(
                """SELECT timestamp_utc, air_temp_f as temp_f,
                          dew_point_f as dewpoint_f, wind_direction_deg as wind_dir_deg,
                          wind_speed_mph as wind_speed_kt, wind_gust_mph as wind_gust_kt,
                          pressure_slp_hpa as pressure_hpa, NULL as visibility_mi,
                          sky_cover_code as sky_cover_1, NULL as sky_cover_2,
                          NULL as sky_cover_3, NULL as rel_humidity,
                          NULL as precip_1h_in
                   FROM nearby_observations
                   WHERE stid = ? AND air_temp_f IS NOT NULL
                   ORDER BY timestamp_utc""",
                (station,),
            ).fetchall()

    def _load_atmosphere(self, conn) -> dict[str, dict]:
        """Load atmospheric data (CAPE/CIN/cloud/radiation/soil) indexed by UTC hour."""
        atmos = {}
        try:
            rows = conn.execute(
                """SELECT timestamp_utc, cape_jkg, cin_jkg, lifted_index,
                          cloud_cover_pct, cloud_cover_low_pct, cloud_cover_mid_pct,
                          cloud_cover_high_pct, shortwave_rad_wm2, direct_rad_wm2,
                          pressure_msl_hpa, soil_temp_0_7cm_c
                   FROM enriched_atmosphere
                   ORDER BY timestamp_utc"""
            ).fetchall()
            for r in rows:
                ts = r["timestamp_utc"]
                hour_key = ts[:13] if len(ts) >= 13 else ts
                atmos[hour_key] = {
                    "cape": r["cape_jkg"],
                    "cin": r["cin_jkg"],
                    "li": r["lifted_index"],
                    "cloud_pct": r["cloud_cover_pct"],
                    "cloud_low": r["cloud_cover_low_pct"],
                    "cloud_mid": r["cloud_cover_mid_pct"],
                    "cloud_high": r["cloud_cover_high_pct"],
                    "sw_rad": r["shortwave_rad_wm2"],
                    "dir_rad": r["direct_rad_wm2"],
                    "pressure": r["pressure_msl_hpa"],
                    "soil_temp": r["soil_temp_0_7cm_c"],
                }
        except Exception:
            pass
        return atmos

    def _load_sst(self, conn) -> dict[str, float]:
        """Load daily average SST from NDBC buoy."""
        sst = {}
        try:
            rows = conn.execute(
                """SELECT SUBSTR(timestamp_utc, 1, 10) as date, AVG(water_temp_f) as avg_sst
                   FROM enriched_sst
                   WHERE water_temp_f IS NOT NULL
                   GROUP BY date"""
            ).fetchall()
            for r in rows:
                sst[r["date"]] = float(r["avg_sst"])
        except Exception:
            pass
        return sst

    def _get_daily_max(self, conn, station, timestamps, temps) -> dict[str, float]:
        """Get daily max from CLI backfill (ground truth) or compute from obs."""
        daily_max: dict[str, float] = {}

        # CLI backfill (IEM daily summaries)
        try:
            rows = conn.execute(
                "SELECT date, max_tmpf FROM cli_daily_backfill WHERE station = ? AND max_tmpf IS NOT NULL",
                (station,),
            ).fetchall()
            for r in rows:
                daily_max[r["date"]] = float(r["max_tmpf"])
        except Exception:
            pass

        # Official CLI settlements override for KMIA
        if station == "KMIA":
            try:
                rows = conn.execute(
                    """SELECT settlement_date, actual_value_f FROM event_settlements
                       WHERE station = 'KMIA' AND market_type = 'high'
                             AND actual_value_f IS NOT NULL""",
                ).fetchall()
                for r in rows:
                    daily_max[r["settlement_date"]] = float(r["actual_value_f"])
            except Exception:
                pass

        # Fill gaps from obs
        obs_daily: dict[str, list] = defaultdict(list)
        for ts, temp in zip(timestamps, temps):
            day = ts[:10] if isinstance(ts, str) else str(ts)[:10]
            if not np.isnan(temp):
                obs_daily[day].append(float(temp))
        for day, day_temps in obs_daily.items():
            if day not in daily_max and len(day_temps) >= 12:
                daily_max[day] = max(day_temps)

        return daily_max

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        # Features are pre-computed — just clone the slice
        features = s["features"].clone()

        # Apply augmentation (temporal jitter, noise, station dropout)
        features = self.augmentor(features)

        # Regime splicing: 10% chance to splice second half from another sample
        if self.augmentor.enabled and torch.rand(1).item() < 0.1:
            donor_idx = torch.randint(0, len(self.samples), (1,)).item()
            if donor_idx != idx:
                features = self.augmentor.splice(features, self.samples[donor_idx]["features"])

        return {
            "features": features,                                        # (T, 33)
            "station_idx": torch.tensor(s["station_idx"], dtype=torch.long),
            "next_temp": torch.tensor(s["next_temp"], dtype=torch.float32),
            "remaining_high": torch.tensor(s["remaining_high"], dtype=torch.float32),
            "cli_max": torch.tensor(s["cli_max"], dtype=torch.float32),
        }


class GraphPretrainingDataset(Dataset):
    """Multi-station pre-training dataset for GraphMamba.

    Each sample contains aligned feature windows from ALL stations at the same
    time period. The target is KMIA's daily max (settlement proxy).

    Returns per sample:
      - station_features: (N_stations, T, 33) — all stations' features
      - wind_dirs: (N_stations,) — wind direction per station at last timestep
      - next_temp: float — KMIA's next temperature
      - remaining_high: float — remaining move to daily max
      - cli_max: float — ground truth daily max
    """

    STATIONS = list(GraphMambaConfig().stations)  # all 35 stations from graph config

    def __init__(self, db_path: str, lookback_hours: int = 48, augment: bool = True):
        self.lookback = lookback_hours
        self.augmentor = SequenceAugmentor(enabled=augment)
        self.n_stations = len(self.STATIONS)
        self.samples = []
        self._load(db_path)

    def _load(self, db_path: str):
        """Build aligned multi-station samples using pre-computed features."""
        conn = sqlite3.connect(db_path, timeout=10)
        conn.row_factory = sqlite3.Row

        # Use IEMPretrainingDataset's loading logic to get per-station features
        inner = IEMPretrainingDataset.__new__(IEMPretrainingDataset)
        inner.lookback = self.lookback
        inner.step_size = 1
        inner.augmentor = SequenceAugmentor(enabled=False)
        inner.station_to_idx = {s: i for i, s in enumerate(self.STATIONS)}
        inner.n_stations = self.n_stations
        inner.samples = []

        atmos_by_hour = inner._load_atmosphere(conn)
        log.info(f"[GraphDS] Loaded {len(atmos_by_hour)} atmospheric hours")

        sst_by_date = inner._load_sst(conn)
        log.info(f"[GraphDS] Loaded {len(sst_by_date)} SST daily averages")

        # Pre-compute feature tensors per station
        station_features: dict[str, torch.Tensor] = {}   # station → (N_timesteps, 52)
        station_timestamps: dict[str, list[str]] = {}      # station → [timestamps]
        station_daily_max: dict[str, dict[str, float]] = {}  # station → {date: max}

        for station in self.STATIONS:
            rows = conn.execute(
                """SELECT timestamp_utc, temp_f, dewpoint_f, wind_dir_deg,
                          wind_speed_kt, wind_gust_kt, pressure_hpa,
                          visibility_mi, sky_cover_1, sky_cover_2, sky_cover_3,
                          rel_humidity, precip_1h_in
                   FROM enriched_asos
                   WHERE station = ? AND temp_f IS NOT NULL
                   ORDER BY timestamp_utc""",
                (station,),
            ).fetchall()

            if len(rows) < self.lookback:
                log.warning(f"[GraphDS] {station}: insufficient data ({len(rows)} rows)")
                continue

            data = inner._parse_rows(rows)
            n = len(rows)
            features_tensor = inner._precompute_features(data, atmos_by_hour, n, sst_by_date)

            station_features[station] = features_tensor
            station_timestamps[station] = data["timestamps"]
            station_daily_max[station] = inner._get_daily_max(
                conn, station, data["timestamps"], data["temp_f"])

            log.info(f"[GraphDS] {station}: {n} obs, features {features_tensor.shape}")

        conn.close()

        # Build aligned time index using KMIA as anchor
        if "KMIA" not in station_timestamps:
            log.error("[GraphDS] KMIA not found in data!")
            return

        kmia_ts = station_timestamps["KMIA"]
        kmia_feat = station_features["KMIA"]
        kmia_max = station_daily_max["KMIA"]

        # Build timestamp → index maps for each station
        ts_index: dict[str, dict[str, int]] = {}
        for station, ts_list in station_timestamps.items():
            # Index by hour key (YYYY-MM-DD HH) for hourly alignment
            ts_index[station] = {}
            for i, ts in enumerate(ts_list):
                hour_key = ts[:13]
                ts_index[station][hour_key] = i

        # Create samples: for each KMIA window, find matching windows at other stations
        n_kmia = len(kmia_ts)
        for start in range(0, n_kmia - self.lookback - 1):
            end = start + self.lookback
            if end >= n_kmia:
                break

            ts_end = kmia_ts[end]
            day_key = ts_end[:10]
            cli_max = kmia_max.get(day_key)
            if cli_max is None:
                continue

            # KMIA features for this window
            kmia_window = kmia_feat[start:end]
            next_temp = float(kmia_feat[end, 11])  # obs_temp at next step
            remaining = max(0.0, cli_max - float(kmia_window[-1, 11]))

            # Gather all stations' features for this time window
            all_station_feats = torch.zeros(self.n_stations, self.lookback, N_FEATURES)
            wind_dirs = torch.full((self.n_stations,), 180.0)  # default

            for s_idx, station in enumerate(self.STATIONS):
                if station not in ts_index:
                    continue

                # Find matching start index via hour key alignment
                start_hour_key = kmia_ts[start][:13]
                s_start = ts_index[station].get(start_hour_key)
                if s_start is None:
                    continue

                s_end = s_start + self.lookback
                if s_end > len(station_features[station]):
                    continue

                all_station_feats[s_idx] = station_features[station][s_start:s_end]
                wind_dirs[s_idx] = station_features[station][s_end - 1, 13]  # last wind_dir

            self.samples.append({
                "station_features": all_station_feats,  # (N_stations, T, 33)
                "wind_dirs": wind_dirs,                  # (N_stations,)
                "next_temp": next_temp,
                "remaining_high": remaining,
                "cli_max": float(cli_max),
            })

        log.info(f"[GraphDS] Built {len(self.samples)} aligned multi-station samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        features = s["station_features"].clone()  # (N, T, 33)

        # Apply augmentation per station
        if self.augmentor.enabled:
            for i in range(features.shape[0]):
                features[i] = self.augmentor(features[i])

        return {
            "station_features": features,
            "wind_dirs": s["wind_dirs"],
            "next_temp": torch.tensor(s["next_temp"], dtype=torch.float32),
            "remaining_high": torch.tensor(s["remaining_high"], dtype=torch.float32),
            "cli_max": torch.tensor(s["cli_max"], dtype=torch.float32),
        }


class SettlementDataset(Dataset):
    """Fine-tuning dataset from actual KMIA CLI settlements."""

    def __init__(self, db_path: str, lookback_days: int = 120):
        self.samples = []
        self._load(db_path, lookback_days)

    def _load(self, db_path: str, lookback_days: int):
        conn = sqlite3.connect(db_path, timeout=10)
        cutoff = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

        settlements = conn.execute(
            """SELECT settlement_date, actual_value_f, market_type
               FROM event_settlements
               WHERE station = 'KMIA' AND settlement_date >= ?
               ORDER BY settlement_date""",
            (cutoff,),
        ).fetchall()

        for date, actual, mtype in settlements:
            if actual is None or mtype != "high":
                continue

            # Get bracket outcomes for this date (10x more data)
            brackets = conn.execute(
                """SELECT ticker, floor_strike, cap_strike, winning_side
                   FROM market_settlements
                   WHERE station = 'KMIA' AND forecast_date = ? AND market_type = 'high'""",
                (date,),
            ).fetchall()

            bracket_outcomes = []
            for ticker, floor_s, cap_s, winner in brackets:
                bracket_outcomes.append({
                    "floor": float(floor_s) if floor_s else -1e6,
                    "ceiling": float(cap_s) if cap_s else 1e6,
                    "won_yes": winner == "yes",
                })

            self.samples.append({
                "date": date,
                "cli_max": float(actual),
                "bracket_outcomes": bracket_outcomes,
            })

        conn.close()
        log.info(f"Loaded {len(self.samples)} settlement samples (last {lookback_days} days)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ──────────────────────────────────────────────────────────────────────
# Training Loop
# ──────────────────────────────────────────────────────────────────────

class DS3MTrainer:
    """End-to-end trainer for the DS3M system.

    Phase 1: Pre-train Mamba on IEM multi-task
    Phase 2: Fine-tune PF + emission on settlements
    Phase 3: Train NSF on bracket outcomes
    """

    def __init__(
        self,
        mamba,  # MambaEncoder or GraphMambaEncoder
        dpf: DifferentiableParticleFilter,
        nsf: ConditionalNSF,
        db_path: str,
        device: torch.device | None = None,
    ):
        self.mamba = mamba
        self.dpf = dpf
        self.nsf = nsf
        self.db_path = db_path
        self.device = device or torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.use_graph = isinstance(mamba, GraphMambaEncoder)

        # Move models to device
        self.mamba.to(self.device)
        self.dpf.to(self.device)
        self.nsf.to(self.device)

    def pretrain_mamba(
        self,
        epochs: int = 50,
        lr: float = 1e-3,
        batch_size: int = 32,
    ) -> dict:
        """Phase 1: Pre-train Mamba encoder on IEM historical data.

        Multi-task loss:
          - Next-step temperature prediction (MSE)
          - Remaining-move prediction (MSE)
          - Daily max prediction (CRPS via skew-normal)
        """
        log.info("=== Phase 1: Pre-training Mamba on IEM data ===")

        if self.use_graph:
            log.info("Using GraphMamba — loading multi-station aligned dataset")
            dataset = GraphPretrainingDataset(self.db_path)
        else:
            dataset = IEMPretrainingDataset(self.db_path)

        if len(dataset) == 0:
            log.warning("No pre-training data available")
            return {"status": "skipped"}

        # num_workers=0: features are pre-computed tensors, __getitem__ is <0.1ms.
        # Workers add overhead (spawn/pickle) for zero benefit. GPU is the bottleneck.
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        # Prediction heads for multi-task
        d_model = self.mamba.config.mamba.d_model if self.use_graph else self.mamba.config.d_model
        next_temp_head = nn.Linear(d_model, 1).to(self.device)
        remaining_head = nn.Linear(d_model, 1).to(self.device)
        # Skew-normal head for daily max
        daily_max_head = nn.Linear(d_model, 3).to(self.device)  # mu, log_sigma, alpha

        params = (
            list(self.mamba.parameters())
            + list(next_temp_head.parameters())
            + list(remaining_head.parameters())
            + list(daily_max_head.parameters())
        )
        optimizer = optim.AdamW(params, lr=lr, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0

            self.mamba.train()
            n_total_batches = len(loader)
            for batch in loader:
                next_temp = batch["next_temp"].to(self.device)    # (B,)
                remaining = batch["remaining_high"].to(self.device)  # (B,)
                cli_max = batch["cli_max"].to(self.device)        # (B,)

                if n_batches % 500 == 0:
                    log.info(f"  Epoch {epoch} batch {n_batches}/{n_total_batches}")

                # Forward through Mamba (or GraphMamba)
                if self.use_graph:
                    station_features = batch["station_features"].to(self.device)  # (B, N, T, 33)
                    wind_dirs = batch["wind_dirs"].to(self.device)                # (B, N)
                    h_target, _ = self.mamba(station_features, wind_dirs)
                    h_seq = h_target  # (B, T, d_model)
                else:
                    features = batch["features"].to(self.device)  # (B, T, 33)
                    h_seq = self.mamba(features)  # (B, T, d_model)

                h_last = h_seq[:, -1, :]     # (B, d_model)

                # Task 1: Next-step temperature
                pred_next = next_temp_head(h_last).squeeze(-1)
                loss_next = nn.functional.mse_loss(pred_next, next_temp)

                # Task 2: Remaining move
                pred_rem = remaining_head(h_last).squeeze(-1)
                loss_rem = nn.functional.mse_loss(pred_rem, remaining)

                # Task 3: Daily max CRPS (via skew-normal)
                daily_params = daily_max_head(h_last)  # (B, 3)
                mu = daily_params[:, 0]
                sigma = torch.nn.functional.softplus(daily_params[:, 1]) + 0.1
                alpha = daily_params[:, 2].clamp(-10, 10)
                sn = SkewNormal(mu, sigma, alpha)
                loss_crps = sn.crps(cli_max).mean()

                # Combined loss (weighted)
                loss = 0.3 * loss_next + 0.3 * loss_rem + 0.4 * loss_crps

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 5.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()
            avg_loss = epoch_loss / max(n_batches, 1)

            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                # Save best Mamba weights
                prefix = "graph_" if self.use_graph else ""
                torch.save(self.mamba.state_dict(), f"analysis_data/{prefix}mamba_pretrained.pt")
            else:
                patience_counter += 1

            if epoch % 5 == 0:
                log.info(f"  Epoch {epoch}: loss={avg_loss:.4f}, best={best_loss:.4f}, "
                         f"lr={scheduler.get_last_lr()[0]:.6f}")

            if patience_counter >= 10:
                log.info(f"  Early stopping at epoch {epoch}")
                break

        log.info(f"Pre-training complete. Best loss: {best_loss:.4f}")
        return {"best_loss": best_loss, "epochs": epoch + 1}

    def train_nsf(
        self,
        epochs: int = 100,
        lr: float = 1e-3,
        batch_size: int = 32,
    ) -> dict:
        """Phase 3: Train NSF on bracket-level outcomes.

        Uses bracket settlements (10x more data than point settlements).
        Loss: CRPS + bracket Brier score.
        """
        log.info("=== Phase 3: Training NSF on bracket outcomes ===")

        dataset = SettlementDataset(self.db_path)
        if len(dataset) < 10:
            log.warning(f"Only {len(dataset)} settlements — need ≥10 for NSF training")
            return {"status": "insufficient_data"}

        optimizer = optim.AdamW(self.nsf.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            epoch_loss = 0.0
            n_samples = 0

            self.nsf.train()
            for sample in dataset:
                cli_max = torch.tensor(sample["cli_max"], dtype=torch.float32, device=self.device)

                # Build a dummy conditioning vector (in production, this comes from live PF state)
                # For training, use climatological priors
                context = torch.randn(1, self.nsf.config.d_condition, device=self.device) * 0.1
                context[0, 0] = cli_max  # hint: put actual value in first position

                # CRPS loss
                crps = self.nsf.crps(cli_max.unsqueeze(0), context)

                # Bracket Brier loss (auxiliary)
                brier_loss = torch.tensor(0.0, device=self.device)
                for bracket in sample["bracket_outcomes"]:
                    pred_p = self.nsf.bracket_prob(
                        bracket["floor"], bracket["ceiling"], context
                    )
                    actual = 1.0 if bracket["won_yes"] else 0.0
                    brier_loss = brier_loss + (pred_p - actual) ** 2

                n_brackets = max(len(sample["bracket_outcomes"]), 1)
                brier_loss = brier_loss / n_brackets

                loss = 0.6 * crps.mean() + 0.4 * brier_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.nsf.parameters(), 5.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_samples += 1

            scheduler.step()
            avg_loss = epoch_loss / max(n_samples, 1)

            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                torch.save(self.nsf.state_dict(), "analysis_data/nsf_trained.pt")
            else:
                patience_counter += 1

            if epoch % 10 == 0:
                log.info(f"  Epoch {epoch}: loss={avg_loss:.4f}, best={best_loss:.4f}")

            if patience_counter >= 10:
                log.info(f"  Early stopping at epoch {epoch}")
                break

        log.info(f"NSF training complete. Best loss: {best_loss:.4f}")
        return {"best_loss": best_loss, "epochs": epoch + 1}

    def run_nightly_training(self) -> dict:
        """Called after CLI settlement arrives.

        1. Update Mamba (if enough new data)
        2. Update transition matrix (from regime posteriors)
        3. Update NSF (from bracket outcomes)
        4. Update conformal calibrator (from settlement error)
        """
        log.info("=== Nightly training triggered ===")
        results = {}

        # Check if we have new settlement data
        dataset = SettlementDataset(self.db_path, lookback_days=120)
        if len(dataset) < 5:
            log.info("Insufficient settlement data for training")
            return {"status": "insufficient_data"}

        # Train NSF on bracket outcomes
        results["nsf"] = self.train_nsf(epochs=30, lr=5e-4)

        # Pre-training only runs on first setup or when explicitly triggered
        # results["mamba"] = self.pretrain_mamba(epochs=10, lr=1e-4)

        log.info(f"Nightly training complete: {results}")
        return results
