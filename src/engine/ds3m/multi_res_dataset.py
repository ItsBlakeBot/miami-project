"""Multi-resolution weather dataset for Weather Brain v3.1.

Produces fine (15-min), medium (hourly), and coarse (3-hourly) tensors
with feature masks, station features, and multi-task targets from the
miami_collector.db SQLite database.

Each sample is anchored to a cli_timing date and produces:
  - fine:   (32, n_fine)    — last 8 hours at 15-min resolution
  - medium: (96, n_medium)  — last 96 hours at hourly resolution
  - coarse: (112, n_coarse) — last 14 days at 3-hourly resolution
  - station_features: (n_stations, d_station)
  - feature_masks: dict of binary tensors matching each branch
  - targets: dict of scalars/ints

Author: Weather Brain training pipeline v3.1
"""

from __future__ import annotations

import logging
import math
import re
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

log = logging.getLogger(__name__)

# ── Feature counts (must match WB3Config) ──────────────────────────────
N_FINE_FEATURES = 14      # 10 weather + 4 time features
N_MEDIUM_FEATURES = 36    # 8 ASOS + 3 GFS + 3 ECMWF + 4 HRRR + 2 buoy + 3 MOS + 5 time + 2 derived + 1 spread + 5 buffer
N_COARSE_FEATURES = 18    # 4 GFS + 4 ECMWF + 3 MOS + 6 time + 1 buffer

FINE_SEQ_LEN = 32         # 8 hours × 4 per hour
MEDIUM_SEQ_LEN = 96       # 96 hours
COARSE_SEQ_LEN = 112      # 14 days × 8 per day

# ── Chronological splits ───────────────────────────────────────────────
TRAIN_END = '2025-01-01'
VAL_START = '2025-01-01'
VAL_END = '2025-10-01'
TEST_START = '2025-10-01'

# ── Stations ───────────────────────────────────────────────────────────
DEFAULT_STATIONS = [
    "KMIA", "KOPF", "KTMB", "KHWO", "KFLL",
    "KHST", "KFXE", "KPMP", "KBCT", "KPBI",
    "KX51", "KIMM", "KPHK", "KOBE", "KSUA",
    "KMTH", "KNQX", "KEYW", "KAPF", "KRSW",
    "KFMY", "KPGD", "KFPR", "KSEF", "KVRB",
    "KMLB", "KBOW", "KSRQ", "KLAL", "KMCO",
    "KSPG", "KORL", "KTPA", "KPIE", "KSFB",
]

D_STATION = 16  # per-station embedding dimension


def _parse_kalshi_date(event_ticker: str) -> str | None:
    """Parse a Kalshi event_ticker like KXHIGHMIA-26MAR27 or HIGHMIA-23AUG01 to YYYY-MM-DD."""
    m = re.search(r'(\d{2})([A-Z]{3})(\d{2})$', event_ticker)
    if not m:
        return None
    yy, mon_str, dd = m.group(1), m.group(2), m.group(3)
    months = {'JAN': '01', 'FEB': '02', 'MAR': '03', 'APR': '04',
              'MAY': '05', 'JUN': '06', 'JUL': '07', 'AUG': '08',
              'SEP': '09', 'OCT': '10', 'NOV': '11', 'DEC': '12'}
    mm = months.get(mon_str)
    if mm is None:
        return None
    return f'20{yy}-{mm}-{dd}'


# ──────────────────────────────────────────────────────────────────────
# Multi-Resolution Weather Dataset
# ──────────────────────────────────────────────────────────────────────

class MultiResolutionWeatherDataset(Dataset):
    """Produces multi-resolution inputs for WeatherBrainV3.

    Anchored on cli_timing dates. For each date, we construct the three
    temporal branches by looking back from 05:00 UTC (Kalshi settlement
    hour) on that date.

    Parameters
    ----------
    db_path : str
        Path to miami_collector.db.
    split : str
        One of 'train', 'val', 'test'.
    stations : list[str] or None
        Station IDs (defaults to 35 SE-Florida ASOS stations).
    """

    def __init__(
        self,
        db_path: str,
        split: str = 'train',
        stations: list[str] | None = None,
    ) -> None:
        self.db_path = db_path
        self.split = split
        self.stations = stations or DEFAULT_STATIONS
        self.station_to_idx = {s: i for i, s in enumerate(self.stations)}
        self.n_stations = len(self.stations)

        # Pre-load everything into memory
        log.info(f"MultiResolutionWeatherDataset: loading {split} split from {db_path}")
        conn = sqlite3.connect(db_path, timeout=30)
        conn.execute("PRAGMA busy_timeout = 30000")

        self._load_all_data(conn)
        self._build_samples(conn)

        conn.close()
        log.info(f"  {split}: {len(self.dates)} dates, "
                 f"{len(self.dates)} samples")

    # ── Data Loading ──────────────────────────────────────────────────

    def _load_all_data(self, conn: sqlite3.Connection) -> None:
        """Load all tables into pandas DataFrames in memory."""

        # enriched_asos (hourly observations)
        log.info("  Loading enriched_asos...")
        self.asos_df = pd.read_sql_query(
            "SELECT station, timestamp_utc, temp_f, dewpoint_f, wind_dir_deg, "
            "wind_speed_kt, wind_gust_kt, pressure_hpa, visibility_mi, "
            "sky_cover_1, rel_humidity, feels_like_f "
            "FROM enriched_asos WHERE temp_f IS NOT NULL ORDER BY timestamp_utc",
            conn,
        )
        self.asos_df['ts'] = pd.to_datetime(self.asos_df['timestamp_utc'])
        log.info(f"    {len(self.asos_df)} rows")

        # hrrr_subhourly (15-min resolution) — optional, may not exist
        log.info("  Loading hrrr_subhourly...")
        try:
            self.hrrr_df = pd.read_sql_query(
                "SELECT station, valid_time_utc, temperature_2m, dewpoint_2m, "
                "wind_speed_10m, wind_direction_10m, surface_pressure, cape, "
                "cloud_cover, visibility "
                "FROM hrrr_subhourly ORDER BY valid_time_utc",
                conn,
            )
            self.hrrr_df['ts'] = pd.to_datetime(self.hrrr_df['valid_time_utc'])
            log.info(f"    {len(self.hrrr_df)} rows")
        except Exception:
            log.warning("    hrrr_subhourly table not found — fine branch will use ASOS only")
            self.hrrr_df = pd.DataFrame()

        # nwp_forecast_archive (multi-model NWP)
        log.info("  Loading nwp_forecast_archive...")
        self.nwp_df = pd.read_sql_query(
            "SELECT city, model, timestamp_utc, temperature_2m, dewpoint_2m, "
            "windspeed_10m, winddirection_10m, cape, surface_pressure, "
            "cloudcover, shortwave_radiation "
            "FROM nwp_forecast_archive WHERE city='Miami' ORDER BY timestamp_utc",
            conn,
        )
        self.nwp_df['ts'] = pd.to_datetime(self.nwp_df['timestamp_utc'])
        log.info(f"    {len(self.nwp_df)} rows")

        # hrrr_archive (hourly HRRR at Miami grid point)
        log.info("  Loading hrrr_archive...")
        try:
            self.hrrr_archive_df = pd.read_sql_query(
                "SELECT model_run_utc AS timestamp_utc, temperature_2m, dewpoint_2m, "
                "wind_speed_10m AS windspeed_10m, wind_direction_10m AS winddirection_10m, "
                "cape, surface_pressure, "
                "cloud_cover AS cloudcover, shortwave_radiation "
                "FROM hrrr_archive WHERE station='KMIA' ORDER BY model_run_utc",
                conn,
            )
            self.hrrr_archive_df['ts'] = pd.to_datetime(self.hrrr_archive_df['timestamp_utc'])
            log.info(f"    {len(self.hrrr_archive_df)} rows")
        except Exception:
            log.warning("    hrrr_archive table not found — HRRR medium features unavailable")
            self.hrrr_archive_df = pd.DataFrame()

        # mos_forecasts
        log.info("  Loading mos_forecasts...")
        self.mos_df = pd.read_sql_query(
            "SELECT station, model, valid_time_utc, max_temp_f, min_temp_f, "
            "temp_f, wind_speed_kt, pop_pct "
            "FROM mos_forecasts WHERE station='KMIA' ORDER BY valid_time_utc",
            conn,
        )
        self.mos_df['ts'] = pd.to_datetime(self.mos_df['valid_time_utc'])
        log.info(f"    {len(self.mos_df)} rows")

        # buoy_observations
        log.info("  Loading buoy_observations...")
        self.buoy_df = pd.read_sql_query(
            "SELECT city, timestamp_utc, water_temp_c, air_temp_c "
            "FROM buoy_observations ORDER BY timestamp_utc",
            conn,
        )
        self.buoy_df['ts'] = pd.to_datetime(self.buoy_df['timestamp_utc'])
        log.info(f"    {len(self.buoy_df)} rows")

        # cli_timing (daily hi/lo and timing) — optional
        log.info("  Loading cli_timing...")
        try:
            self.cli_df = pd.read_sql_query(
                "SELECT station, date, high_temp_f, low_temp_f, "
                "high_hour_utc, low_hour_utc "
                "FROM cli_timing ORDER BY date",
                conn,
            )
            log.info(f"    {len(self.cli_df)} rows")
        except Exception:
            log.warning("    cli_timing not found — using cli_daily_backfill for targets")
            self.cli_df = pd.DataFrame()

        # kalshi_markets (bracket outcomes) — optional
        log.info("  Loading kalshi_markets...")
        try:
            self.kalshi_df = pd.read_sql_query(
                "SELECT event_ticker, floor_strike, cap_strike, result "
                "FROM kalshi_markets WHERE event_ticker LIKE '%HIGHMIA%'",
                conn,
            )
            self.bracket_outcomes = self._build_bracket_outcomes()
            log.info(f"    {len(self.kalshi_df)} rows, {len(self.bracket_outcomes)} settled dates")
        except Exception:
            log.warning("    kalshi_markets not found — bracket targets will be -1")
            self.kalshi_df = pd.DataFrame()
            self.bracket_outcomes = {}

        # upper_air_soundings — optional
        log.info("  Loading upper_air_soundings...")
        try:
            self.sounding_df = pd.read_sql_query(
                "SELECT city, timestamp_utc, t850_temp_c, t925_temp_c, "
                "cape, precipitable_water_mm, lifted_index "
                "FROM upper_air_soundings WHERE city='Miami' ORDER BY timestamp_utc",
                conn,
            )
            self.sounding_df['ts'] = pd.to_datetime(self.sounding_df['timestamp_utc'])
            log.info(f"    {len(self.sounding_df)} rows")
        except Exception:
            log.warning("    upper_air_soundings not found — coarse branch will use NWP only")
            self.sounding_df = pd.DataFrame()

        # Build indexed lookups for fast access
        self._build_indices()

    def _build_bracket_outcomes(self) -> dict[str, int]:
        """Map date -> winning bracket index (0-5) from kalshi_markets.

        Brackets are sorted by floor_strike ascending. The winning bracket
        is the one with result='yes'.
        """
        outcomes = {}
        # Group by event_ticker
        events = {}
        for _, row in self.kalshi_df.iterrows():
            et = row['event_ticker']
            if et not in events:
                events[et] = []
            events[et].append(row)

        for et, rows in events.items():
            date_str = _parse_kalshi_date(et)
            if date_str is None:
                continue

            # Sort brackets by floor_strike (handle None)
            bracket_rows = sorted(
                rows,
                key=lambda r: r['floor_strike'] if r['floor_strike'] is not None else -999,
            )
            for i, r in enumerate(bracket_rows):
                if r['result'] == 'yes':
                    outcomes[date_str] = min(i, 5)  # cap at 5 brackets
                    break

        return outcomes

    def _build_indices(self) -> None:
        """Build fast lookup indices for time-based queries."""
        # ASOS: station -> sorted timestamps + data arrays
        self.asos_by_station = {}
        for station, group in self.asos_df.groupby('station'):
            group = group.sort_values('ts')
            self.asos_by_station[station] = {
                'ts': group['ts'].values,
                'temp_f': group['temp_f'].values.astype(np.float32),
                'dewpoint_f': group['dewpoint_f'].fillna(65.0).values.astype(np.float32),
                'wind_dir_deg': group['wind_dir_deg'].fillna(180.0).values.astype(np.float32),
                'wind_speed_kt': group['wind_speed_kt'].fillna(5.0).values.astype(np.float32),
                'pressure_hpa': group['pressure_hpa'].fillna(1015.0).values.astype(np.float32),
                'visibility_mi': group['visibility_mi'].fillna(10.0).values.astype(np.float32),
                'sky_cover_1': group['sky_cover_1'].fillna('SCT').values,
                'rel_humidity': group['rel_humidity'].fillna(65.0).values.astype(np.float32),
            }

        # HRRR: station -> sorted timestamps + data
        self.hrrr_by_station = {}
        for station, group in self.hrrr_df.groupby('station'):
            group = group.sort_values('ts')
            self.hrrr_by_station[station] = {
                'ts': group['ts'].values,
                'temperature_2m': group['temperature_2m'].fillna(75.0).values.astype(np.float32),
                'dewpoint_2m': group['dewpoint_2m'].fillna(65.0).values.astype(np.float32),
                'wind_speed_10m': group['wind_speed_10m'].fillna(5.0).values.astype(np.float32),
                'wind_direction_10m': group['wind_direction_10m'].fillna(180.0).values.astype(np.float32),
                'surface_pressure': group['surface_pressure'].fillna(1015.0).values.astype(np.float32),
                'cape': group['cape'].fillna(0.0).values.astype(np.float32),
                'cloud_cover': group['cloud_cover'].fillna(50.0).values.astype(np.float32),
                'visibility': group['visibility'].fillna(10.0).values.astype(np.float32),
            }

        # HRRR archive: single Miami grid point (hourly)
        if len(self.hrrr_archive_df) > 0:
            ha = self.hrrr_archive_df.sort_values('ts')
            self.hrrr_archive_idx = {
                'ts': ha['ts'].values,
                'temperature_2m': ha['temperature_2m'].fillna(75.0).values.astype(np.float32),
                'dewpoint_2m': ha['dewpoint_2m'].fillna(65.0).values.astype(np.float32),
                'windspeed_10m': ha['windspeed_10m'].fillna(5.0).values.astype(np.float32),
                'surface_pressure': ha['surface_pressure'].fillna(1015.0).values.astype(np.float32),
            }
        else:
            self.hrrr_archive_idx = None

        # NWP: model -> sorted timestamps + data (city='Miami' filtered)
        self.nwp_by_model = {}
        for model, group in self.nwp_df.groupby('model'):
            group = group.sort_values('ts')
            self.nwp_by_model[model] = {
                'ts': group['ts'].values,
                'temperature_2m': group['temperature_2m'].fillna(75.0).values.astype(np.float32),
                'dewpoint_2m': group['dewpoint_2m'].fillna(65.0).values.astype(np.float32),
                'windspeed_10m': group['windspeed_10m'].fillna(5.0).values.astype(np.float32),
                'surface_pressure': group['surface_pressure'].fillna(1015.0).values.astype(np.float32),
            }

        # CLI timing indexed by (station, date)
        self.cli_by_key = {}
        for _, row in self.cli_df.iterrows():
            key = (row['station'], row['date'])
            self.cli_by_key[key] = {
                'high_temp_f': row['high_temp_f'],
                'low_temp_f': row['low_temp_f'],
                'high_hour_utc': row['high_hour_utc'],
                'low_hour_utc': row['low_hour_utc'],
            }

        # Buoy: city -> sorted timestamps + SST
        self.buoy_by_city = {}
        for city, group in self.buoy_df.groupby('city'):
            group = group.sort_values('ts')
            self.buoy_by_city[city] = {
                'ts': group['ts'].values,
                'water_temp_c': group['water_temp_c'].fillna(26.0).values.astype(np.float32),
                'air_temp_c': group['air_temp_c'].fillna(25.0).values.astype(np.float32),
            }

        # MOS: station -> sorted timestamps + data
        self.mos_by_station = {}
        for station, group in self.mos_df.groupby('station'):
            group = group.sort_values('ts')
            self.mos_by_station[station] = {
                'ts': group['ts'].values,
                'max_temp_f': group['max_temp_f'].fillna(85.0).values.astype(np.float32),
                'min_temp_f': group['min_temp_f'].fillna(70.0).values.astype(np.float32),
                'pop_pct': group['pop_pct'].fillna(0.0).values.astype(np.float32),
            }

    def _build_samples(self, conn: sqlite3.Connection) -> None:
        """Build list of sample dates filtered by split."""
        # Get all cli_timing dates for KMIA (primary station)
        all_dates = sorted(self.cli_df[self.cli_df['station'] == 'KMIA']['date'].unique())

        if self.split == 'train':
            self.dates = [d for d in all_dates if d < TRAIN_END]
        elif self.split == 'val':
            self.dates = [d for d in all_dates if VAL_START <= d < VAL_END]
        elif self.split == 'test':
            self.dates = [d for d in all_dates if d >= TEST_START]
        else:
            raise ValueError(f"Unknown split: {self.split}")

        log.info(f"  {self.split} split: {len(self.dates)} dates "
                 f"({self.dates[0] if self.dates else 'N/A'} to "
                 f"{self.dates[-1] if self.dates else 'N/A'})")

    def __len__(self) -> int:
        return len(self.dates)

    def __getitem__(self, idx: int) -> dict:
        date_str = self.dates[idx]
        # Anchor time: 05:00 UTC on the date (Kalshi settlement)
        anchor = pd.Timestamp(date_str + ' 05:00:00', tz='UTC')

        fine = self._build_fine(anchor)
        medium = self._build_medium(anchor)
        coarse = self._build_coarse(anchor)
        station_features = self._build_station_features(anchor)
        targets = self._build_targets(date_str, anchor)

        # NOTE: Fine branch now always has time features + ASOS fallback for pre-2024.
        # Medium/coarse branches always have time features. No random noise needed.

        return {
            'fine': fine['data'],
            'medium': medium['data'],
            'coarse': coarse['data'],
            'station_features': station_features,
            'feature_masks': {
                'fine': fine['mask'],
                'medium': medium['mask'],
                'coarse': coarse['mask'],
            },
            'targets': targets,
        }

    # ── Fine Branch (15-min, last 8 hours) ────────────────────────────

    def _build_fine(self, anchor: pd.Timestamp) -> dict:
        """Build fine-resolution tensor: (32, N_FINE_FEATURES).

        Sources: hrrr_subhourly + enriched_asos at 15-min intervals.
        Features: temperature_2m, dewpoint_2m, wind_speed_10m, wind_direction_10m,
                  surface_pressure, cape, cloud_cover, visibility,
                  running_max_so_far, hours_since_sunrise, + zero-padded to 64
        """
        data = torch.zeros(FINE_SEQ_LEN, N_FINE_FEATURES, dtype=torch.float32)
        mask = torch.zeros(FINE_SEQ_LEN, N_FINE_FEATURES, dtype=torch.float32)

        station = 'KMIA'
        hrrr = self.hrrr_by_station.get(station)

        start_time = anchor - pd.Timedelta(hours=8)

        if hrrr is not None and len(hrrr['ts']) > 0:
            ts_arr = hrrr['ts']
            # Find indices within [start_time, anchor)
            start_np = np.datetime64(start_time.tz_localize(None) if start_time.tzinfo else start_time)
            anchor_np = np.datetime64(anchor.tz_localize(None) if anchor.tzinfo else anchor)
            idx_mask = (ts_arr >= start_np) & (ts_arr < anchor_np)
            indices = np.where(idx_mask)[0]

            if len(indices) > 0:
                # Take the last FINE_SEQ_LEN points
                indices = indices[-FINE_SEQ_LEN:]
                n_fill = len(indices)
                offset = FINE_SEQ_LEN - n_fill

                data[offset:, 0] = torch.from_numpy(hrrr['temperature_2m'][indices]) * 9.0/5.0 + 32.0  # C→F
                data[offset:, 1] = torch.from_numpy(hrrr['dewpoint_2m'][indices]) * 9.0/5.0 + 32.0   # C→F
                data[offset:, 2] = torch.from_numpy(hrrr['wind_speed_10m'][indices])
                data[offset:, 3] = torch.from_numpy(hrrr['wind_direction_10m'][indices])
                data[offset:, 4] = torch.from_numpy(hrrr['surface_pressure'][indices])
                data[offset:, 5] = torch.from_numpy(hrrr['cape'][indices])
                data[offset:, 6] = torch.from_numpy(hrrr['cloud_cover'][indices])
                data[offset:, 7] = torch.from_numpy(hrrr['visibility'][indices])

                # Running max so far (convert to F)
                temps = hrrr['temperature_2m'][indices] * 9.0/5.0 + 32.0  # C→F
                running_max = np.maximum.accumulate(temps)
                data[offset:, 8] = torch.from_numpy(running_max)

                # Hours since sunrise (approximate: Miami sunrise ~11-13 UTC)
                sunrise_utc = 12.0  # rough average
                for j, idx in enumerate(indices):
                    ts = pd.Timestamp(ts_arr[idx])
                    hour_frac = ts.hour + ts.minute / 60.0
                    data[offset + j, 9] = max(0.0, hour_frac - sunrise_utc)

                mask[offset:, :10] = 1.0

        # If HRRR unavailable, fill from enriched_asos (interpolate hourly to 15-min)
        if mask.sum() == 0:
            asos = self.asos_by_station.get(station)
            if asos is not None and len(asos['ts']) > 0:
                ts_arr = asos['ts']
                start_np = np.datetime64(start_time.tz_localize(None) if start_time.tzinfo else start_time)
                anchor_np = np.datetime64(anchor.tz_localize(None) if anchor.tzinfo else anchor)
                idx_mask = (ts_arr >= start_np) & (ts_arr < anchor_np)
                indices = np.where(idx_mask)[0]

                if len(indices) > 0:
                    # Repeat hourly obs to fill 15-min slots
                    indices = indices[-8:]  # last 8 hours
                    for j, idx in enumerate(indices):
                        slot_start = j * 4
                        slot_end = min(slot_start + 4, FINE_SEQ_LEN)
                        for s in range(slot_start, slot_end):
                            real_s = FINE_SEQ_LEN - len(indices) * 4 + s
                            if 0 <= real_s < FINE_SEQ_LEN:
                                data[real_s, 0] = float(asos['temp_f'][idx])       # already °F
                                data[real_s, 1] = float(asos['dewpoint_f'][idx])   # already °F
                                data[real_s, 2] = float(asos['wind_speed_kt'][idx])
                                data[real_s, 3] = float(asos['wind_dir_deg'][idx])
                                data[real_s, 4] = float(asos['pressure_hpa'][idx])
                                data[real_s, 5] = 0.0   # cape (unavailable from ASOS)
                                data[real_s, 6] = 0.0   # cloud_cover (unavailable)
                                data[real_s, 7] = float(asos['visibility_mi'][idx]) if 'visibility_mi' in asos else 10.0
                                # mask=0.5 for interpolated values (partial confidence)
                                mask[real_s, :5] = 0.5
                    # Running max from ASOS temps
                    filled_temps = data[:, 0].numpy()
                    running_max = np.maximum.accumulate(filled_temps)
                    data[:, 8] = torch.from_numpy(running_max)
                    mask[data[:, 0] != 0, 8] = 0.5

        # Time features (features 10-13): always available regardless of HRRR/ASOS
        sunrise_utc = 12.0  # rough average for Miami
        for t in range(FINE_SEQ_LEN):
            minutes_back = (FINE_SEQ_LEN - t) * 15
            slot_time = anchor - pd.Timedelta(minutes=minutes_back)
            hour_frac = slot_time.hour + slot_time.minute / 60.0
            doy = slot_time.dayofyear
            data[t, 10] = math.sin(2 * math.pi * hour_frac / 24.0)
            data[t, 11] = math.cos(2 * math.pi * hour_frac / 24.0)
            data[t, 12] = math.sin(2 * math.pi * doy / 365.25)
            data[t, 13] = math.cos(2 * math.pi * doy / 365.25)
            mask[t, 10:14] = 1.0

        return {'data': data, 'mask': mask}

    # ── Medium Branch (hourly, last 96 hours) ─────────────────────────

    def _build_medium(self, anchor: pd.Timestamp) -> dict:
        """Build medium-resolution tensor: (96, N_MEDIUM_FEATURES).

        Sources: enriched_asos + nwp_forecast_archive + buoy at hourly.
        """
        data = torch.zeros(MEDIUM_SEQ_LEN, N_MEDIUM_FEATURES, dtype=torch.float32)
        mask = torch.zeros(MEDIUM_SEQ_LEN, N_MEDIUM_FEATURES, dtype=torch.float32)

        station = 'KMIA'
        start_time = anchor - pd.Timedelta(hours=96)

        # ── Observations (features 0-7) ──
        asos = self.asos_by_station.get(station)
        if asos is not None and len(asos['ts']) > 0:
            ts_arr = asos['ts']
            start_np = np.datetime64(start_time.tz_localize(None) if start_time.tzinfo else start_time)
            anchor_np = np.datetime64(anchor.tz_localize(None) if anchor.tzinfo else anchor)
            idx_mask = (ts_arr >= start_np) & (ts_arr < anchor_np)
            indices = np.where(idx_mask)[0]

            if len(indices) > 0:
                indices = indices[-MEDIUM_SEQ_LEN:]
                n_fill = len(indices)
                offset = MEDIUM_SEQ_LEN - n_fill

                data[offset:, 0] = torch.from_numpy(asos['temp_f'][indices])
                data[offset:, 1] = torch.from_numpy(asos['dewpoint_f'][indices])
                data[offset:, 2] = torch.from_numpy(asos['wind_dir_deg'][indices])
                data[offset:, 3] = torch.from_numpy(asos['wind_speed_kt'][indices])

                # Sky cover -> numeric
                sky_map = {"CLR": 0.0, "FEW": 0.15, "SCT": 0.35, "BKN": 0.65, "OVC": 1.0, "VV": 1.0}
                sky_vals = np.array([sky_map.get(str(s), 0.35) for s in asos['sky_cover_1'][indices]], dtype=np.float32)
                data[offset:, 4] = torch.from_numpy(sky_vals)

                data[offset:, 5] = torch.from_numpy(asos['visibility_mi'][indices])
                data[offset:, 6] = torch.from_numpy(asos['pressure_hpa'][indices])
                data[offset:, 7] = torch.from_numpy(asos['rel_humidity'][indices])

                mask[offset:, :8] = 1.0

                # ── Derived features (features 28-29): temp_change_1h, dewpoint_change_1h ──
                if n_fill >= 2:
                    temp_change = np.diff(asos['temp_f'][indices], prepend=asos['temp_f'][indices[0]])
                    dew_change = np.diff(asos['dewpoint_f'][indices], prepend=asos['dewpoint_f'][indices[0]])
                    data[offset:, 28] = torch.from_numpy(temp_change.astype(np.float32))
                    data[offset:, 29] = torch.from_numpy(dew_change.astype(np.float32))
                    mask[offset:, 28:30] = 1.0

        # ── NWP features (features 8-21) ──
        # GFS: temp, dewpoint, wind (features 8-10)
        gfs = self.nwp_by_model.get('gfs_seamless')
        if gfs is not None and len(gfs['ts']) > 0:
            start_np = np.datetime64(start_time.tz_localize(None))
            anchor_np = np.datetime64(anchor.tz_localize(None))
            idx_mask = (gfs['ts'] >= start_np) & (gfs['ts'] < anchor_np)
            indices = np.where(idx_mask)[0]
            if len(indices) > 0:
                indices = indices[-MEDIUM_SEQ_LEN:]
                n_fill = len(indices)
                off = MEDIUM_SEQ_LEN - n_fill
                data[off:, 8] = torch.from_numpy(gfs['temperature_2m'][indices]) * 9.0/5.0 + 32.0  # C→F
                data[off:, 9] = torch.from_numpy(gfs['dewpoint_2m'][indices]) * 9.0/5.0 + 32.0   # C→F
                data[off:, 10] = torch.from_numpy(gfs['windspeed_10m'][indices])
                mask[off:, 8:11] = 1.0

        # ECMWF: temp, dewpoint, wind (features 11-13)
        ecmwf = self.nwp_by_model.get('ecmwf_ifs04')
        if ecmwf is not None and len(ecmwf['ts']) > 0:
            start_np = np.datetime64(start_time.tz_localize(None))
            anchor_np = np.datetime64(anchor.tz_localize(None))
            idx_mask = (ecmwf['ts'] >= start_np) & (ecmwf['ts'] < anchor_np)
            indices = np.where(idx_mask)[0]
            if len(indices) > 0:
                indices = indices[-MEDIUM_SEQ_LEN:]
                n_fill = len(indices)
                off = MEDIUM_SEQ_LEN - n_fill
                data[off:, 11] = torch.from_numpy(ecmwf['temperature_2m'][indices]) * 9.0/5.0 + 32.0  # C→F
                data[off:, 12] = torch.from_numpy(ecmwf['dewpoint_2m'][indices]) * 9.0/5.0 + 32.0     # C→F
                data[off:, 13] = torch.from_numpy(ecmwf['windspeed_10m'][indices])
                mask[off:, 11:14] = 1.0

        # HRRR: temp, dewpoint, wind, pressure (features 14-17) from hrrr_archive
        hrrr = self.hrrr_archive_idx
        if hrrr is not None and len(hrrr['ts']) > 0:
            start_np = np.datetime64(start_time.tz_localize(None))
            anchor_np = np.datetime64(anchor.tz_localize(None))
            idx_mask = (hrrr['ts'] >= start_np) & (hrrr['ts'] < anchor_np)
            indices = np.where(idx_mask)[0]
            if len(indices) > 0:
                indices = indices[-MEDIUM_SEQ_LEN:]
                n_fill = len(indices)
                off = MEDIUM_SEQ_LEN - n_fill
                data[off:, 14] = torch.from_numpy(hrrr['temperature_2m'][indices]) * 9.0/5.0 + 32.0  # C→F
                data[off:, 15] = torch.from_numpy(hrrr['dewpoint_2m'][indices]) * 9.0/5.0 + 32.0     # C→F
                data[off:, 16] = torch.from_numpy(hrrr['windspeed_10m'][indices])
                data[off:, 17] = torch.from_numpy(hrrr['surface_pressure'][indices])
                mask[off:, 14:18] = 1.0

        # ── Buoy: SST, air_temp (features 18-19) ──
        buoy = self.buoy_by_city.get('Miami')
        if buoy is not None and len(buoy['ts']) > 0:
            start_np = np.datetime64(start_time.tz_localize(None))
            anchor_np = np.datetime64(anchor.tz_localize(None))
            idx_mask = (buoy['ts'] >= start_np) & (buoy['ts'] < anchor_np)
            indices = np.where(idx_mask)[0]
            if len(indices) > 0:
                indices = indices[-MEDIUM_SEQ_LEN:]
                n_fill = len(indices)
                off = MEDIUM_SEQ_LEN - n_fill
                data[off:, 18] = torch.from_numpy(buoy['water_temp_c'][indices]) * 9.0/5.0 + 32.0  # C→F
                data[off:, 19] = torch.from_numpy(buoy['air_temp_c'][indices]) * 9.0/5.0 + 32.0   # C→F
                mask[off:, 18:20] = 1.0

        # ── MOS: max_forecast, min_forecast, pop (features 20-22) ──
        mos = self.mos_by_station.get(station)
        if mos is not None and len(mos['ts']) > 0:
            start_np = np.datetime64(start_time.tz_localize(None))
            anchor_np = np.datetime64(anchor.tz_localize(None))
            idx_mask = (mos['ts'] >= start_np) & (mos['ts'] < anchor_np)
            indices = np.where(idx_mask)[0]
            if len(indices) > 0:
                indices = indices[-MEDIUM_SEQ_LEN:]
                n_fill = len(indices)
                off = MEDIUM_SEQ_LEN - n_fill
                data[off:, 20] = torch.from_numpy(mos['max_temp_f'][indices])
                data[off:, 21] = torch.from_numpy(mos['min_temp_f'][indices])
                data[off:, 22] = torch.from_numpy(mos['pop_pct'][indices])
                mask[off:, 20:23] = 1.0

        # ── Time features (features 23-27) ──
        for t in range(MEDIUM_SEQ_LEN):
            hours_back = MEDIUM_SEQ_LEN - t
            slot_time = anchor - pd.Timedelta(hours=hours_back)
            hour_frac = slot_time.hour + slot_time.minute / 60.0
            doy = slot_time.dayofyear
            data[t, 23] = math.sin(2 * math.pi * hour_frac / 24.0)
            data[t, 24] = math.cos(2 * math.pi * hour_frac / 24.0)
            data[t, 25] = math.sin(2 * math.pi * doy / 365.25)
            data[t, 26] = math.cos(2 * math.pi * doy / 365.25)
            data[t, 27] = max(0.0, hours_back)  # lead time to settlement
            mask[t, 23:28] = 1.0

        # ── Derived features (features 28-29): temp_change_1h, dewpoint_change_1h ──
        # (moved from features 30-31 after compacting)

        # NWP spread (feature 30): gfs_temp - ecmwf_temp
        if mask[:, 8].sum() > 0 and mask[:, 11].sum() > 0:
            data[:, 30] = data[:, 8] - data[:, 11]
            mask[:, 30] = mask[:, 8] * mask[:, 11]

        return {'data': data, 'mask': mask}

    # ── Coarse Branch (3-hourly, last 14 days) ────────────────────────

    def _build_coarse(self, anchor: pd.Timestamp) -> dict:
        """Build coarse-resolution tensor: (112, N_COARSE_FEATURES).

        Sources: nwp_forecast_archive + mos_forecasts at 3-hourly.
        """
        data = torch.zeros(COARSE_SEQ_LEN, N_COARSE_FEATURES, dtype=torch.float32)
        mask = torch.zeros(COARSE_SEQ_LEN, N_COARSE_FEATURES, dtype=torch.float32)

        start_time = anchor - pd.Timedelta(days=14)

        # GFS: temp, dewpoint, wind_speed, wind_dir, cape, pressure, cloud, radiation
        # (features 0-7)
        gfs = self.nwp_by_model.get('gfs_seamless')
        if gfs is not None and len(gfs['ts']) > 0:
            start_np = np.datetime64(start_time.tz_localize(None))
            anchor_np = np.datetime64(anchor.tz_localize(None))
            idx_mask = (gfs['ts'] >= start_np) & (gfs['ts'] < anchor_np)
            indices = np.where(idx_mask)[0]

            if len(indices) > 0:
                # Subsample to 3-hourly (take every 3rd point if hourly)
                indices = indices[::3][-COARSE_SEQ_LEN:]
                n_fill = len(indices)
                off = COARSE_SEQ_LEN - n_fill

                data[off:, 0] = torch.from_numpy(gfs['temperature_2m'][indices]) * 9.0/5.0 + 32.0  # C→F
                data[off:, 1] = torch.from_numpy(gfs['dewpoint_2m'][indices]) * 9.0/5.0 + 32.0   # C→F
                data[off:, 2] = torch.from_numpy(gfs['windspeed_10m'][indices])
                data[off:, 3] = torch.from_numpy(gfs['surface_pressure'][indices])
                mask[off:, :4] = 1.0

        # ECMWF: temp, dewpoint, wind_speed, pressure (features 4-7)
        ecmwf = self.nwp_by_model.get('ecmwf_ifs04')
        if ecmwf is not None and len(ecmwf['ts']) > 0:
            start_np = np.datetime64(start_time.tz_localize(None))
            anchor_np = np.datetime64(anchor.tz_localize(None))
            idx_mask = (ecmwf['ts'] >= start_np) & (ecmwf['ts'] < anchor_np)
            indices = np.where(idx_mask)[0]
            if len(indices) > 0:
                indices = indices[::3][-COARSE_SEQ_LEN:]
                n_fill = len(indices)
                off = COARSE_SEQ_LEN - n_fill
                data[off:, 4] = torch.from_numpy(ecmwf['temperature_2m'][indices]) * 9.0/5.0 + 32.0  # C→F
                data[off:, 5] = torch.from_numpy(ecmwf['dewpoint_2m'][indices]) * 9.0/5.0 + 32.0     # C→F
                data[off:, 6] = torch.from_numpy(ecmwf['windspeed_10m'][indices])
                data[off:, 7] = torch.from_numpy(ecmwf['surface_pressure'][indices])
                mask[off:, 4:8] = 1.0

        # MOS: max_temp, min_temp, pop (features 8-10)
        mos = self.mos_by_station.get('KMIA')
        if mos is not None and len(mos['ts']) > 0:
            start_np = np.datetime64(start_time.tz_localize(None))
            anchor_np = np.datetime64(anchor.tz_localize(None))
            idx_mask = (mos['ts'] >= start_np) & (mos['ts'] < anchor_np)
            indices = np.where(idx_mask)[0]
            if len(indices) > 0:
                indices = indices[::3][-COARSE_SEQ_LEN:]
                n_fill = len(indices)
                off = COARSE_SEQ_LEN - n_fill
                data[off:, 8] = torch.from_numpy(mos['max_temp_f'][indices])
                data[off:, 9] = torch.from_numpy(mos['min_temp_f'][indices])
                data[off:, 10] = torch.from_numpy(mos['pop_pct'][indices])
                mask[off:, 8:11] = 1.0

        # NOTE: Upper air soundings removed (only 150 rows out of 36K needed — too sparse)

        # Time features (features 11-16): hour_sin, hour_cos, doy_sin, doy_cos, lead_time, week_of_year
        for t in range(COARSE_SEQ_LEN):
            hours_back = (COARSE_SEQ_LEN - t) * 3
            slot_time = anchor - pd.Timedelta(hours=hours_back)
            hour_frac = slot_time.hour + slot_time.minute / 60.0
            doy = slot_time.dayofyear
            data[t, 11] = math.sin(2 * math.pi * hour_frac / 24.0)
            data[t, 12] = math.cos(2 * math.pi * hour_frac / 24.0)
            data[t, 13] = math.sin(2 * math.pi * doy / 365.25)
            data[t, 14] = math.cos(2 * math.pi * doy / 365.25)
            data[t, 15] = max(0.0, hours_back)  # lead time
            data[t, 16] = float(slot_time.isocalendar()[1]) / 52.0  # week of year
            mask[t, 11:17] = 1.0

        return {'data': data, 'mask': mask}

    # ── Station Features ──────────────────────────────────────────────

    def _build_station_features(self, anchor: pd.Timestamp) -> Tensor:
        """Build station features: (n_stations, D_STATION).

        Simple learned embeddings — return station index as one-hot
        padded to D_STATION (the model has its own station embeddings).
        """
        # For now, produce a simple feature vector per station
        # with latest available temperature and position encoding
        feats = torch.zeros(self.n_stations, D_STATION, dtype=torch.float32)
        anchor_np = np.datetime64(anchor.tz_localize(None))

        for i, station in enumerate(self.stations):
            # Station index encoding
            feats[i, 0] = float(i) / self.n_stations

            asos = self.asos_by_station.get(station)
            if asos is not None and len(asos['ts']) > 0:
                # Find latest obs before anchor
                before = asos['ts'] < anchor_np
                valid = np.where(before)[0]
                if len(valid) > 0:
                    last_idx = valid[-1]
                    feats[i, 1] = float(asos['temp_f'][last_idx]) / 100.0  # normalized
                    feats[i, 2] = float(asos['dewpoint_f'][last_idx]) / 100.0
                    feats[i, 3] = float(asos['wind_speed_kt'][last_idx]) / 50.0
                    feats[i, 4] = float(asos['pressure_hpa'][last_idx]) / 1020.0

        return feats

    # ── Targets ───────────────────────────────────────────────────────

    def _build_targets(self, date_str: str, anchor: pd.Timestamp) -> dict:
        """Build target dict from cli_timing + kalshi_markets + enriched_asos."""
        targets = {
            'daily_max': 85.0,
            'daily_min': 70.0,
            'max_hour': 20.0,
            'min_hour': 11.0,
            'bracket_target': -1,
            'next_hour_temp': 80.0,
            'nwp_bias': 0.0,
            'regime': -1,
        }

        # CLI timing data
        cli = self.cli_by_key.get(('KMIA', date_str))
        if cli is not None:
            if cli['high_temp_f'] is not None:
                targets['daily_max'] = float(cli['high_temp_f'])
            if cli['low_temp_f'] is not None:
                targets['daily_min'] = float(cli['low_temp_f'])
            if cli['high_hour_utc'] is not None:
                targets['max_hour'] = float(cli['high_hour_utc'])
            if cli['low_hour_utc'] is not None:
                targets['min_hour'] = float(cli['low_hour_utc'])

        # Bracket target from Kalshi
        bracket = self.bracket_outcomes.get(date_str, -1)
        targets['bracket_target'] = bracket

        # Next hour temp (temp at anchor + 1h)
        asos = self.asos_by_station.get('KMIA')
        if asos is not None and len(asos['ts']) > 0:
            target_time = np.datetime64((anchor + pd.Timedelta(hours=1)).tz_localize(None))
            # Find nearest observation
            diffs = np.abs(asos['ts'] - target_time)
            nearest_idx = np.argmin(diffs.astype(np.int64))
            if abs((asos['ts'][nearest_idx] - target_time).astype('timedelta64[m]').astype(int)) < 90:
                targets['next_hour_temp'] = float(asos['temp_f'][nearest_idx])

        # NWP bias: observed_max - GFS forecast max
        gfs = self.nwp_by_model.get('gfs_seamless')
        if gfs is not None and cli is not None and cli['high_temp_f'] is not None:
            # Find GFS forecast for this date
            date_start = np.datetime64(date_str)
            date_end = date_start + np.timedelta64(1, 'D')
            idx_mask = (gfs['ts'] >= date_start) & (gfs['ts'] < date_end)
            indices = np.where(idx_mask)[0]
            if len(indices) > 0:
                gfs_max_c = float(np.max(gfs['temperature_2m'][indices]))
                gfs_max_f = gfs_max_c * 9.0/5.0 + 32.0  # C→F
                targets['nwp_bias'] = targets['daily_max'] - gfs_max_f

        # Regime: -1 (unknown, HDP discovers online)
        targets['regime'] = -1

        # Convert all to tensors
        return {
            'daily_max': torch.tensor(targets['daily_max'], dtype=torch.float32),
            'daily_min': torch.tensor(targets['daily_min'], dtype=torch.float32),
            'max_hour': torch.tensor(targets['max_hour'], dtype=torch.float32),
            'min_hour': torch.tensor(targets['min_hour'], dtype=torch.float32),
            'bracket_target': torch.tensor(targets['bracket_target'], dtype=torch.long),
            'next_hour_temp': torch.tensor(targets['next_hour_temp'], dtype=torch.float32),
            'nwp_bias': torch.tensor(targets['nwp_bias'], dtype=torch.float32),
            'regime': torch.tensor(targets['regime'], dtype=torch.long),
        }


# ──────────────────────────────────────────────────────────────────────
# DataLoader builder
# ──────────────────────────────────────────────────────────────────────

def _collate_multi_res(batch: list[dict]) -> dict:
    """Custom collate for multi-resolution weather data.

    Stacks tensors and handles nested dicts properly.
    """
    fine = torch.stack([b['fine'] for b in batch])
    medium = torch.stack([b['medium'] for b in batch])
    coarse = torch.stack([b['coarse'] for b in batch])
    station_features = torch.stack([b['station_features'] for b in batch])

    feature_masks = {
        'fine': torch.stack([b['feature_masks']['fine'] for b in batch]),
        'medium': torch.stack([b['feature_masks']['medium'] for b in batch]),
        'coarse': torch.stack([b['feature_masks']['coarse'] for b in batch]),
    }

    targets = {}
    target_keys = batch[0]['targets'].keys()
    for key in target_keys:
        targets[key] = torch.stack([b['targets'][key] for b in batch])

    return {
        'fine': fine,
        'medium': medium,
        'coarse': coarse,
        'station_features': station_features,
        'feature_masks': feature_masks,
        'targets': targets,
    }


def build_weather_dataloaders(
    db_path: str,
    batch_size: int = 64,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader]:
    """Build train and validation DataLoaders for Weather Brain v3.1.

    Parameters
    ----------
    db_path : str
        Path to miami_collector.db.
    batch_size : int
        Batch size (default 64, scale up for H200).
    num_workers : int
        DataLoader workers.

    Returns
    -------
    tuple[DataLoader, DataLoader]
        (train_loader, val_loader)
    """
    train_dataset = MultiResolutionWeatherDataset(db_path, split='train')
    val_dataset = MultiResolutionWeatherDataset(db_path, split='val')

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=_collate_multi_res,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=_collate_multi_res,
    )

    log.info(f"DataLoaders built: train={len(train_dataset)} samples, "
             f"val={len(val_dataset)} samples, batch_size={batch_size}")

    return train_loader, val_loader
