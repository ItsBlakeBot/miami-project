"""SQLite store — init, insert, and query methods."""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from collector.store.schema import INDEXES, SCHEMA_VERSION, TABLES, VIEWS
from collector.types import (
    AtmosphericData,
    EventSettlement,
    FAWNObservation,
    MarketSnapshot,
    ModelForecast,
    Observation,
    PressureLevelData,
    SSTObservation,
)

log = logging.getLogger(__name__)


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _norm_num(value: float | int | None, places: int = 3) -> float | int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    return round(float(value), places)


def _has_any_value(*values: Any) -> bool:
    return any(v is not None for v in values)


class Store:
    """Thread-safe SQLite store with WAL mode."""

    def __init__(self, db_path: str | Path = "miami_collector.db"):
        self._path = str(db_path)
        self._conn: sqlite3.Connection | None = None
        self._live: Any | None = None  # LiveState, set by runner.py
        self._forecast_fingerprint_cache: dict[tuple, tuple | None] = {}
        self._market_snapshot_fingerprint_cache: dict[str, tuple | None] = {}
        self._pressure_level_fingerprint_cache: dict[tuple, tuple | None] = {}
        self._atmospheric_fingerprint_cache: dict[tuple, tuple | None] = {}

    def signal_new_data(self) -> None:
        """Notify that new data was ingested. Triggers inference if live."""
        if self._live is not None:
            self._live.signal_new_data()

    # -- Lifecycle --

    def open(self) -> None:
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._init_schema()

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    @property
    def conn(self) -> sqlite3.Connection:
        assert self._conn is not None, "Store not opened"
        return self._conn

    def _init_schema(self) -> None:
        cur = self.conn.cursor()
        for ddl in TABLES:
            cur.execute(ddl)
        for idx in INDEXES:
            cur.execute(idx)
        for view in VIEWS:
            cur.execute(view)
        # Migrations for existing DBs
        for migration in [
            "ALTER TABLE atmospheric_data ADD COLUMN soil_moisture_0_1cm REAL",
            "ALTER TABLE paper_trades ADD COLUMN strike_label TEXT",
            "ALTER TABLE bracket_estimates ADD COLUMN regime_confidence REAL",
            "ALTER TABLE bracket_estimates ADD COLUMN recommended_side TEXT",
            "ALTER TABLE bracket_estimates ADD COLUMN recommended_contracts INTEGER",
            "ALTER TABLE bracket_estimates ADD COLUMN recommended_price_cents REAL",
            "ALTER TABLE bracket_estimates ADD COLUMN recommended_probability REAL",
            "ALTER TABLE bracket_estimates ADD COLUMN recommended_edge_cents REAL",
            "ALTER TABLE bracket_estimates ADD COLUMN recommended_ev_cents REAL",
            "ALTER TABLE bracket_estimates ADD COLUMN sizing_json TEXT",
        ]:
            try:
                cur.execute(migration)
            except sqlite3.OperationalError:
                pass  # Column already exists
        cur.execute(
            "INSERT OR REPLACE INTO schema_meta(key, value) VALUES (?, ?)",
            ("version", SCHEMA_VERSION),
        )
        self.conn.commit()

    @staticmethod
    def _where_with_nullable(columns: list[str], values: tuple[Any, ...]) -> tuple[str, list[Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        for column, value in zip(columns, values, strict=True):
            if value is None:
                clauses.append(f"{column} IS NULL")
            else:
                clauses.append(f"{column} = ?")
                params.append(value)
        return " AND ".join(clauses), params

    # -- Forecast dedupe helpers --

    @staticmethod
    def _forecast_key(f: ModelForecast) -> tuple:
        return (
            f.station,
            f.forecast_date,
            f.model,
            f.source,
            f.run_time,
            f.valid_time,
        )

    @staticmethod
    def _forecast_fingerprint(f: ModelForecast) -> tuple | None:
        if not _has_any_value(
            f.forecast_high_f,
            f.forecast_low_f,
            f.raw_temperature_f,
        ):
            return None
        return (
            _norm_num(f.forecast_high_f),
            _norm_num(f.forecast_low_f),
            _norm_num(f.raw_temperature_f),
            _norm_num(f.run_age_hours),
        )

    def _load_latest_forecast_fingerprint(self, key: tuple) -> tuple | None:
        where_sql, params = self._where_with_nullable(
            ["station", "forecast_date", "model", "source", "run_time", "valid_time"],
            key,
        )
        row = self.conn.execute(
            f"""SELECT forecast_high_f, forecast_low_f, raw_temperature_f, run_age_hours
                FROM model_forecasts
                WHERE {where_sql}
                ORDER BY id DESC
                LIMIT 1""",
            params,
        ).fetchone()
        if not row:
            return None
        return (
            _norm_num(row[0]),
            _norm_num(row[1]),
            _norm_num(row[2]),
            _norm_num(row[3]),
        )

    # -- Model Forecasts --

    def insert_forecast(self, f: ModelForecast) -> int:
        key = self._forecast_key(f)
        fingerprint = self._forecast_fingerprint(f)
        if fingerprint is None:
            return 0

        cached = self._forecast_fingerprint_cache.get(key)
        if cached is None and key not in self._forecast_fingerprint_cache:
            cached = self._load_latest_forecast_fingerprint(key)
            self._forecast_fingerprint_cache[key] = cached

        if cached == fingerprint:
            return 0

        self.conn.execute(
            """INSERT INTO model_forecasts
               (station, forecast_date, model, source, run_time, valid_time,
                forecast_high_f, forecast_low_f, raw_temperature_f,
                run_age_hours, fetch_time_utc, source_record_json)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                f.station, f.forecast_date, f.model, f.source,
                f.run_time, f.valid_time,
                f.forecast_high_f, f.forecast_low_f, f.raw_temperature_f,
                f.run_age_hours, f.fetch_time_utc, f.source_record_json,
            ),
        )
        self._forecast_fingerprint_cache[key] = fingerprint
        return 1

    def insert_forecasts(self, forecasts: list[ModelForecast]) -> int:
        count = 0
        for f in forecasts:
            count += self.insert_forecast(f)
        self.conn.commit()
        return count

    # -- Observations --

    def insert_observation(self, o: Observation) -> None:
        self.conn.execute(
            """INSERT OR IGNORE INTO observations
               (station, timestamp_utc, lst_date, temperature_f, dew_point_f,
                relative_humidity, wind_speed_mph, wind_direction, wind_gust_mph,
                wind_heading_deg, visibility_miles, sky_cover_pct, sky_cover_code,
                pressure_hpa, precipitation_last_hour_mm,
                wethr_high_f, wethr_low_f, wethr_high_nws_f,
                wethr_low_nws_f, source)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                o.station, o.timestamp_utc, o.lst_date, o.temperature_f,
                o.dew_point_f, o.relative_humidity, o.wind_speed_mph,
                o.wind_direction, o.wind_gust_mph, o.wind_heading_deg,
                o.visibility_miles, o.sky_cover_pct, o.sky_cover_code,
                o.pressure_hpa, o.precipitation_last_hour_mm,
                o.wethr_high_f, o.wethr_low_f,
                o.wethr_high_nws_f, o.wethr_low_nws_f, o.source,
            ),
        )
        self.conn.commit()

    # -- Pressure level dedupe helpers --

    @staticmethod
    def _pressure_level_key(p: PressureLevelData) -> tuple:
        return (p.station, p.model, p.valid_time_utc)

    @staticmethod
    def _pressure_level_fingerprint(p: PressureLevelData) -> tuple | None:
        values = (
            _norm_num(p.temp_925_c),
            _norm_num(p.wind_speed_925),
            _norm_num(p.wind_dir_925),
            _norm_num(p.geopotential_925),
            _norm_num(p.temp_850_c),
            _norm_num(p.wind_speed_850),
            _norm_num(p.wind_dir_850),
            _norm_num(p.geopotential_850),
            _norm_num(p.rh_850),
            _norm_num(p.rh_700),
            _norm_num(p.temp_700_c),
            _norm_num(p.wind_speed_700),
            _norm_num(p.wind_dir_700),
            _norm_num(p.geopotential_700),
            _norm_num(p.temp_500_c),
            _norm_num(p.wind_speed_500),
            _norm_num(p.wind_dir_500),
        )
        return values if _has_any_value(*values) else None

    def _load_latest_pressure_level_fingerprint(self, key: tuple) -> tuple | None:
        row = self.conn.execute(
            """SELECT temp_925_c, wind_speed_925, wind_dir_925, geopotential_925,
                      temp_850_c, wind_speed_850, wind_dir_850, geopotential_850,
                      rh_850, rh_700,
                      temp_700_c, wind_speed_700, wind_dir_700, geopotential_700,
                      temp_500_c, wind_speed_500, wind_dir_500
               FROM pressure_levels
               WHERE station = ? AND model = ? AND valid_time_utc = ?
               ORDER BY id DESC
               LIMIT 1""",
            key,
        ).fetchone()
        if not row:
            return None
        return tuple(_norm_num(value) for value in row)

    # -- Pressure Levels --

    def insert_pressure_level(self, p: PressureLevelData) -> int:
        key = self._pressure_level_key(p)
        fingerprint = self._pressure_level_fingerprint(p)
        if fingerprint is None:
            return 0

        cached = self._pressure_level_fingerprint_cache.get(key)
        if cached is None and key not in self._pressure_level_fingerprint_cache:
            cached = self._load_latest_pressure_level_fingerprint(key)
            self._pressure_level_fingerprint_cache[key] = cached

        if cached == fingerprint:
            return 0

        self.conn.execute(
            """INSERT INTO pressure_levels
               (station, valid_time_utc, model, fetch_time_utc,
                temp_925_c, wind_speed_925, wind_dir_925, geopotential_925,
                temp_850_c, wind_speed_850, wind_dir_850, geopotential_850,
                rh_850, rh_700,
                temp_700_c, wind_speed_700, wind_dir_700, geopotential_700,
                temp_500_c, wind_speed_500, wind_dir_500)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                p.station, p.valid_time_utc, p.model, p.fetch_time_utc,
                p.temp_925_c, p.wind_speed_925, p.wind_dir_925, p.geopotential_925,
                p.temp_850_c, p.wind_speed_850, p.wind_dir_850, p.geopotential_850,
                p.rh_850, p.rh_700,
                p.temp_700_c, p.wind_speed_700, p.wind_dir_700, p.geopotential_700,
                p.temp_500_c, p.wind_speed_500, p.wind_dir_500,
            ),
        )
        self._pressure_level_fingerprint_cache[key] = fingerprint
        return 1

    def insert_pressure_levels(self, levels: list[PressureLevelData]) -> int:
        count = 0
        for p in levels:
            count += self.insert_pressure_level(p)
        self.conn.commit()
        return count

    # -- Market snapshot dedupe helpers --

    @staticmethod
    def _market_snapshot_fingerprint(m: MarketSnapshot) -> tuple:
        # Only best bid/ask and last trade price trigger new rows.
        # Volume, depth, and liquidity fields still get stored on every
        # row but don't cause new rows on their own — volume ticks with
        # every 1-contract trade (noise), and depth flickers with every
        # deep-book delta. Price changes are the signal.
        return (
            _norm_num(m.best_yes_bid_cents),
            _norm_num(m.best_yes_ask_cents),
            _norm_num(m.best_no_bid_cents),
            _norm_num(m.best_no_ask_cents),
            _norm_num(m.last_price_cents),
        )

    def _load_latest_market_snapshot_fingerprint(self, ticker: str) -> tuple | None:
        row = self.conn.execute(
            """SELECT best_yes_bid_cents, best_yes_ask_cents,
                      best_no_bid_cents, best_no_ask_cents,
                      last_price_cents
               FROM market_snapshots
               WHERE ticker = ?
               ORDER BY id DESC
               LIMIT 1""",
            (ticker,),
        ).fetchone()
        if not row:
            return None
        return (
            _norm_num(row[0]),
            _norm_num(row[1]),
            _norm_num(row[2]),
            _norm_num(row[3]),
            _norm_num(row[4]),
        )

    # -- Market Snapshots --

    def insert_market_snapshot(self, m: MarketSnapshot) -> int:
        fingerprint = self._market_snapshot_fingerprint(m)
        ticker = m.ticker

        cached = self._market_snapshot_fingerprint_cache.get(ticker)
        if cached is None and ticker not in self._market_snapshot_fingerprint_cache:
            cached = self._load_latest_market_snapshot_fingerprint(ticker)
            self._market_snapshot_fingerprint_cache[ticker] = cached

        if cached == fingerprint:
            return 0

        self.conn.execute(
            """INSERT INTO market_snapshots
               (ticker, event_ticker, series_ticker, market_type, forecast_date,
                floor_strike, cap_strike, best_yes_bid_cents, best_yes_ask_cents,
                best_no_bid_cents, best_no_ask_cents, last_price_cents, volume,
                yes_bid_qty, yes_ask_qty, no_bid_qty, no_ask_qty,
                total_yes_depth, total_no_depth, spread_cents,
                num_yes_levels, num_no_levels, snapshot_time)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                m.ticker, m.event_ticker, m.series_ticker, m.market_type,
                m.forecast_date, m.floor_strike, m.cap_strike,
                m.best_yes_bid_cents, m.best_yes_ask_cents,
                m.best_no_bid_cents, m.best_no_ask_cents,
                m.last_price_cents, m.volume,
                m.yes_bid_qty, m.yes_ask_qty, m.no_bid_qty, m.no_ask_qty,
                m.total_yes_depth, m.total_no_depth, m.spread_cents,
                m.num_yes_levels, m.num_no_levels, m.snapshot_time,
            ),
        )
        self.conn.commit()
        self._market_snapshot_fingerprint_cache[ticker] = fingerprint
        return 1

    # -- Event Settlements --

    def insert_event_settlement(self, s: EventSettlement) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO event_settlements
               (station, settlement_date, market_type, actual_value_f,
                settlement_source, raw_text, received_at)
               VALUES (?,?,?,?,?,?,?)""",
            (
                s.station, s.settlement_date, s.market_type,
                s.actual_value_f, s.settlement_source, s.raw_text,
                s.received_at,
            ),
        )
        self.conn.commit()

    # -- Market Settlements --

    def insert_market_settlement(
        self,
        ticker: str,
        event_ticker: str,
        series_ticker: str,
        station: str,
        forecast_date: str,
        market_type: str,
        floor_strike: float | None,
        cap_strike: float | None,
        winning_side: str,
        settled_at: str,
    ) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO market_settlements
               (ticker, event_ticker, series_ticker, station, forecast_date,
                market_type, floor_strike, cap_strike, winning_side, settled_at)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                ticker, event_ticker, series_ticker, station, forecast_date,
                market_type, floor_strike, cap_strike, winning_side, settled_at,
            ),
        )
        self.conn.commit()

    # -- Nearby Observations (IEM / Synoptic) --

    def insert_nearby_observation(
        self,
        stid: str,
        name: str,
        network: str,
        latitude: float,
        longitude: float,
        distance_mi: float,
        elevation_m: float | None,
        timestamp_utc: str,
        lst_date: str,
        air_temp_f: float | None,
        dew_point_f: float | None,
        wind_speed_mph: float | None,
        wind_direction_deg: float | None,
        wind_gust_mph: float | None,
        pressure_slp_hpa: float | None,
        sky_cover_code: str | None,
        temp_delta_vs_kmia: float | None = None,
    ) -> None:
        self.conn.execute(
            """INSERT OR IGNORE INTO nearby_observations
               (stid, name, network, latitude, longitude, distance_mi,
                elevation_m, timestamp_utc, lst_date, air_temp_f, dew_point_f,
                wind_speed_mph, wind_direction_deg, wind_gust_mph,
                pressure_slp_hpa, sky_cover_code, temp_delta_vs_kmia)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                stid, name, network, latitude, longitude, distance_mi,
                elevation_m, timestamp_utc, lst_date, air_temp_f, dew_point_f,
                wind_speed_mph, wind_direction_deg, wind_gust_mph,
                pressure_slp_hpa, sky_cover_code, temp_delta_vs_kmia,
            ),
        )

    def insert_nearby_observations(self, obs_list: list[tuple]) -> int:
        """Batch insert nearby observations. Skips duplicates via UNIQUE(stid, timestamp_utc)."""
        count = 0
        for args in obs_list:
            cur = self.conn.execute(
                """INSERT OR IGNORE INTO nearby_observations
                   (stid, name, network, latitude, longitude, distance_mi,
                    elevation_m, timestamp_utc, lst_date, air_temp_f, dew_point_f,
                    wind_speed_mph, wind_direction_deg, wind_gust_mph,
                    pressure_slp_hpa, sky_cover_code, temp_delta_vs_kmia)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                args,
            )
            count += cur.rowcount
        self.conn.commit()
        return count

    def upsert_nearby_station(
        self,
        stid: str,
        name: str,
        network: str,
        latitude: float,
        longitude: float,
        distance_mi: float,
        elevation_m: float | None,
        bearing_deg: float | None = None,
    ) -> None:
        self.conn.execute(
            """INSERT INTO nearby_stations
               (stid, name, network, latitude, longitude, distance_mi, bearing_deg, elevation_m, last_seen_at)
               VALUES (?,?,?,?,?,?,?,?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
               ON CONFLICT(stid) DO UPDATE SET
                 bearing_deg = COALESCE(excluded.bearing_deg, bearing_deg),
                 last_seen_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now'),
                 is_active = 1""",
            (stid, name, network, latitude, longitude, distance_mi, bearing_deg, elevation_m),
        )
        self.conn.commit()

    def get_latest_kmia_temp(self) -> float | None:
        """Get the most recent KMIA temperature for computing deltas."""
        row = self.conn.execute(
            """SELECT temperature_f FROM observations
               WHERE station = 'KMIA' AND temperature_f IS NOT NULL
               ORDER BY timestamp_utc DESC LIMIT 1""",
        ).fetchone()
        return row[0] if row else None

    # -- SSE Events --

    def insert_sse_event(self, station: str, event_type: str, payload: dict) -> None:
        self.conn.execute(
            """INSERT INTO sse_events (station, event_type, payload_json)
               VALUES (?,?,?)""",
            (station, event_type, json.dumps(payload)),
        )
        self.conn.commit()

    # -- Diurnal Scores --

    def insert_diurnal_score(
        self,
        station: str,
        model: str,
        forecast_date: str,
        valid_time_utc: str,
        valid_hour_local: int,
        predicted_f: float,
        observed_f: float,
        obs_timestamp_utc: str,
    ) -> None:
        error_f = predicted_f - observed_f
        self.conn.execute(
            """INSERT INTO diurnal_scores
               (station, model, forecast_date, valid_time_utc, valid_hour_local,
                predicted_f, observed_f, error_f, obs_timestamp_utc)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                station, model, forecast_date, valid_time_utc, valid_hour_local,
                predicted_f, observed_f, error_f, obs_timestamp_utc,
            ),
        )
        self.conn.commit()

    # -- Cloud Obs --

    def insert_cloud_obs(
        self,
        station: str,
        obs_time: str,
        forecast_date: str,
        obs_sky_pct: float | None,
        obs_sky_code: str | None,
        forecast_sky_pct: float | None,
        temp_error_f: float | None,
    ) -> None:
        disc = None
        if obs_sky_pct is not None and forecast_sky_pct is not None:
            disc = obs_sky_pct - forecast_sky_pct
        self.conn.execute(
            """INSERT INTO cloud_obs
               (station, obs_time, forecast_date, obs_sky_pct, obs_sky_code,
                forecast_sky_pct, cloud_discrepancy_pct, temp_error_f)
               VALUES (?,?,?,?,?,?,?,?)""",
            (
                station, obs_time, forecast_date, obs_sky_pct, obs_sky_code,
                forecast_sky_pct, disc, temp_error_f,
            ),
        )
        self.conn.commit()

    # -- Model Scores --

    def insert_model_score(
        self,
        station: str,
        model: str,
        score_date: str,
        market_type: str,
        mae: float,
        bias: float,
        rmse: float,
        sample_count: int,
        window_days: int,
    ) -> None:
        self.conn.execute(
            """INSERT INTO model_scores
               (station, model, score_date, market_type, mae, bias, rmse,
                sample_count, window_days)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (station, model, score_date, market_type, mae, bias, rmse,
             sample_count, window_days),
        )
        self.conn.commit()

    # -- Adjusted Forecasts --

    def save_model_consensus(
        self,
        station: str,
        forecast_date: str,
        market_type: str,
        rows: list[dict],
    ) -> None:
        """Replace all adjusted forecast rows for a station/date/market_type."""
        self.conn.execute(
            "DELETE FROM model_consensus WHERE station=? AND forecast_date=? AND market_type=?",
            (station, forecast_date, market_type),
        )
        for r in rows:
            self.conn.execute(
                """INSERT INTO model_consensus
                   (station, forecast_date, market_type, model, source,
                    run_time, run_age_hours, raw_forecast_f, bias,
                    forecast_f, mae, skill_weight, decay_factor,
                    final_weight, sample_count, consensus_forecast_f,
                    consensus_std_f, n_models, window_days, run_time_utc)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (station, forecast_date, market_type,
                 r["model"], r["source"], r.get("run_time"),
                 r.get("run_age_hours"), r.get("raw_forecast_f"),
                 r.get("bias"), r.get("forecast_f"),
                 r.get("mae"), r.get("skill_weight"),
                 r.get("decay_factor"), r.get("final_weight"),
                 r.get("sample_count"), r.get("consensus_forecast_f"),
                 r.get("consensus_std_f"), r.get("n_models"),
                 r.get("window_days"), r.get("run_time_utc")),
            )
        self.conn.commit()

    def get_adjusted_consensus(
        self,
        station: str,
        forecast_date: str,
        market_type: str,
    ) -> list[dict] | None:
        """Read latest adjusted forecast rows for a station/date/market_type."""
        rows = self.conn.execute(
            """SELECT model, source, run_time, run_age_hours,
                      raw_forecast_f, bias, forecast_f, mae,
                      skill_weight, decay_factor, final_weight, sample_count,
                      consensus_forecast_f, consensus_std_f, n_models, window_days
               FROM model_consensus
               WHERE station=? AND forecast_date=? AND market_type=?
               ORDER BY final_weight DESC""",
            (station, forecast_date, market_type),
        ).fetchall()
        if not rows:
            return None
        return [
            dict(
                model=r[0], source=r[1], run_time=r[2], run_age_hours=r[3],
                raw_forecast_f=r[4], bias=r[5], forecast_f=r[6],
                mae=r[7], skill_weight=r[8], decay_factor=r[9],
                final_weight=r[10], sample_count=r[11],
                consensus_forecast_f=r[12], consensus_std_f=r[13],
                n_models=r[14], window_days=r[15],
            )
            for r in rows
        ]

    # -- Forward Curves --

    def save_forward_curve(
        self,
        station: str,
        snapshot_time_utc: str,
        target_date: str,
        rows: list[dict],
    ) -> int:
        """Save forward curve rows (DELETE+INSERT for idempotent re-runs)."""
        self.conn.execute(
            "DELETE FROM forward_curves WHERE station=? AND snapshot_time_utc=?",
            (station, snapshot_time_utc),
        )
        count = 0
        for r in rows:
            self.conn.execute(
                """INSERT INTO forward_curves (
                       station, snapshot_time_utc, target_date, valid_hour_utc,
                       hours_ahead, nbm_temp_f, gfs_temp_f, ecmwf_temp_f,
                       hrrr_temp_f, nam_temp_f, model_min_f, model_max_f,
                       model_spread_f, cape, pw_mm, precip_prob, precip_mm,
                       solar_wm2, temp_850_c, temp_925_c, wind_dir_850,
                       wind_speed_850
                   ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    station, snapshot_time_utc, target_date, r["valid_hour_utc"],
                    r["hours_ahead"], r.get("nbm_temp_f"), r.get("gfs_temp_f"),
                    r.get("ecmwf_temp_f"), r.get("hrrr_temp_f"), r.get("nam_temp_f"),
                    r.get("model_min_f"), r.get("model_max_f"), r.get("model_spread_f"),
                    r.get("cape"), r.get("pw_mm"), r.get("precip_prob"),
                    r.get("precip_mm"), r.get("solar_wm2"), r.get("temp_850_c"),
                    r.get("temp_925_c"), r.get("wind_dir_850"), r.get("wind_speed_850"),
                ),
            )
            count += 1
        self.conn.commit()
        return count

    # -- Collection Runs --

    def log_collection_run(
        self,
        collector: str,
        status: str,
        records: int = 0,
        error_text: str | None = None,
    ) -> None:
        now = _now_utc()
        self.conn.execute(
            """INSERT INTO collection_runs
               (collector, started_at, completed_at, status, records_collected, error_text)
               VALUES (?,?,?,?,?,?)""",
            (collector, now, now, status, records, error_text),
        )
        self.conn.commit()

    # -- Atmospheric dedupe helpers --

    @staticmethod
    def _atmospheric_key(a: AtmosphericData) -> tuple:
        return (a.station, a.model, a.valid_time_utc)

    @staticmethod
    def _atmospheric_fingerprint(a: AtmosphericData) -> tuple | None:
        values = (
            _norm_num(a.shortwave_radiation),
            _norm_num(a.direct_radiation),
            _norm_num(a.diffuse_radiation),
            _norm_num(a.cape),
            _norm_num(a.lifted_index),
            _norm_num(a.boundary_layer_height),
            _norm_num(a.precipitable_water_mm),
            _norm_num(a.soil_temperature_0_7cm),
            _norm_num(a.soil_moisture_0_1cm),
            _norm_num(a.precipitation_mm),
            _norm_num(a.rain_mm),
            _norm_num(a.showers_mm),
            _norm_num(a.precipitation_probability),
        )
        return values if _has_any_value(*values) else None

    def _load_latest_atmospheric_fingerprint(self, key: tuple) -> tuple | None:
        row = self.conn.execute(
            """SELECT shortwave_radiation, direct_radiation, diffuse_radiation,
                      cape, lifted_index, boundary_layer_height,
                      precipitable_water_mm, soil_temperature_0_7cm, soil_moisture_0_1cm,
                      precipitation_mm, rain_mm, showers_mm, precipitation_probability
               FROM atmospheric_data
               WHERE station = ? AND model = ? AND valid_time_utc = ?
               ORDER BY id DESC
               LIMIT 1""",
            key,
        ).fetchone()
        if not row:
            return None
        return tuple(_norm_num(value) for value in row)

    # -- Atmospheric Data --

    def insert_atmospheric_data(self, a: AtmosphericData) -> int:
        key = self._atmospheric_key(a)
        fingerprint = self._atmospheric_fingerprint(a)
        if fingerprint is None:
            return 0

        cached = self._atmospheric_fingerprint_cache.get(key)
        if cached is None and key not in self._atmospheric_fingerprint_cache:
            cached = self._load_latest_atmospheric_fingerprint(key)
            self._atmospheric_fingerprint_cache[key] = cached

        if cached == fingerprint:
            return 0

        self.conn.execute(
            """INSERT INTO atmospheric_data
               (station, valid_time_utc, model, fetch_time_utc,
                shortwave_radiation, direct_radiation, diffuse_radiation,
                cape, lifted_index, boundary_layer_height,
                precipitable_water_mm, soil_temperature_0_7cm, soil_moisture_0_1cm,
                precipitation_mm, rain_mm, showers_mm, precipitation_probability)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                a.station, a.valid_time_utc, a.model, a.fetch_time_utc,
                a.shortwave_radiation, a.direct_radiation, a.diffuse_radiation,
                a.cape, a.lifted_index, a.boundary_layer_height,
                a.precipitable_water_mm, a.soil_temperature_0_7cm, a.soil_moisture_0_1cm,
                a.precipitation_mm, a.rain_mm, a.showers_mm,
                a.precipitation_probability,
            ),
        )
        self._atmospheric_fingerprint_cache[key] = fingerprint
        return 1

    def insert_atmospheric_batch(self, data: list[AtmosphericData]) -> int:
        count = 0
        for a in data:
            count += self.insert_atmospheric_data(a)
        self.conn.commit()
        return count

    # -- SST Observations --

    def insert_sst_observation(self, s: SSTObservation) -> None:
        self.conn.execute(
            """INSERT OR IGNORE INTO sst_observations
               (station_id, name, timestamp_utc, water_temp_c, water_temp_f,
                air_temp_c, wind_speed_mps, wind_dir_deg, pressure_hpa, distance_mi)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                s.station_id, s.name, s.timestamp_utc,
                s.water_temp_c, s.water_temp_f, s.air_temp_c,
                s.wind_speed_mps, s.wind_dir_deg, s.pressure_hpa,
                s.distance_mi,
            ),
        )
        self.conn.commit()

    def insert_sst_observations(self, obs: list[SSTObservation]) -> int:
        count = 0
        for s in obs:
            cur = self.conn.execute(
                """INSERT OR IGNORE INTO sst_observations
                   (station_id, name, timestamp_utc, water_temp_c, water_temp_f,
                    air_temp_c, wind_speed_mps, wind_dir_deg, pressure_hpa, distance_mi)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    s.station_id, s.name, s.timestamp_utc,
                    s.water_temp_c, s.water_temp_f, s.air_temp_c,
                    s.wind_speed_mps, s.wind_dir_deg, s.pressure_hpa,
                    s.distance_mi,
                ),
            )
            count += cur.rowcount
        self.conn.commit()
        return count

    # -- FAWN Observations --

    def insert_fawn_observations(self, obs: list[FAWNObservation]) -> int:
        """Batch insert FAWN observations. Skips duplicates via UNIQUE(station_id, timestamp_utc)."""
        count = 0
        for o in obs:
            cur = self.conn.execute(
                """INSERT OR IGNORE INTO fawn_observations
                   (station_id, station_name, timestamp_utc,
                    air_temp_f, air_temp_c, dew_point_f, relative_humidity,
                    wind_speed_mph, wind_gust_mph, wind_direction_deg,
                    solar_radiation_wm2, soil_temp_c, soil_temp_f,
                    rain_mm, rain_in)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    o.station_id, o.station_name, o.timestamp_utc,
                    o.air_temp_f, o.air_temp_c, o.dew_point_f, o.relative_humidity,
                    o.wind_speed_mph, o.wind_gust_mph, o.wind_direction_deg,
                    o.solar_radiation_wm2, o.soil_temp_c, o.soil_temp_f,
                    o.rain_mm, o.rain_in,
                ),
            )
            count += cur.rowcount
        self.conn.commit()
        return count

    # -- RTMA-RU Observations --

    def insert_rtma_ru_observations(
        self, obs: list[Any]
    ) -> int:
        """Batch insert RTMA-RU observations.

        Accepts list of RTMAObservation dataclass instances.
        Skips duplicates via UNIQUE(timestamp_utc, lat, lon).
        """
        count = 0
        for o in obs:
            cur = self.conn.execute(
                """INSERT OR IGNORE INTO rtma_ru_observations
                   (timestamp_utc, lat, lon, temperature_2m, dewpoint_2m,
                    wind_speed_10m, wind_direction_10m, surface_pressure,
                    wind_gust_10m, cloud_cover_pct, visibility_m)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    o.timestamp_utc, o.lat, o.lon,
                    o.temperature_2m, o.dewpoint_2m,
                    o.wind_speed_10m, o.wind_direction_10m, o.surface_pressure,
                    o.wind_gust_10m, o.cloud_cover_pct, o.visibility_m,
                ),
            )
            count += cur.rowcount
        self.conn.commit()
        return count

    # -- Query helpers --

    def get_nearest_observation(
        self, station: str, target_utc: str
    ) -> tuple[str, float] | None:
        """Find the observation nearest to target_utc. Returns (timestamp, temp_f) or None."""
        row = self.conn.execute(
            """SELECT timestamp_utc, temperature_f FROM observations
               WHERE station = ? AND temperature_f IS NOT NULL
               ORDER BY ABS(julianday(timestamp_utc) - julianday(?))
               LIMIT 1""",
            (station, target_utc),
        ).fetchone()
        return row if row else None

    def get_forecasts_with_valid_time(
        self, station: str, forecast_date: str
    ) -> list[tuple[str, str, str, float]]:
        """Get all hourly point forecasts for a date.
        Returns list of (model, valid_time, fetch_time_utc, raw_temperature_f)."""
        rows = self.conn.execute(
            """SELECT model, valid_time, fetch_time_utc, raw_temperature_f
               FROM model_forecasts
               WHERE station = ? AND forecast_date = ?
                 AND valid_time IS NOT NULL AND raw_temperature_f IS NOT NULL
               ORDER BY model, valid_time""",
            (station, forecast_date),
        ).fetchall()
        return rows

    def get_daily_forecasts(
        self, station: str, forecast_date: str
    ) -> list[tuple[str, str, float | None, float | None, str | None]]:
        """Get daily high/low forecasts. Returns (model, source, high, low, run_time)."""
        rows = self.conn.execute(
            """SELECT model, source, forecast_high_f, forecast_low_f, run_time
               FROM model_forecasts
               WHERE station = ? AND forecast_date = ?
                 AND (forecast_high_f IS NOT NULL OR forecast_low_f IS NOT NULL)
                 AND valid_time IS NULL
               ORDER BY model""",
            (station, forecast_date),
        ).fetchall()
        return rows

    # -- Regime Labels --

    def upsert_regime_label(
        self,
        station: str,
        target_date: str,
        regimes_active: str,
        *,
        path_class: str | None = None,
        confidence_tags: str | None = None,
        phase_summary: str | None = None,
        model_performance: str | None = None,
        signal_labels: str | None = None,
        signal_families_active: str | None = None,
        threshold_recommendations: str | None = None,
        review_path: str | None = None,
        review_source: str = "ai",
    ) -> None:
        """Insert or replace a regime label for a station/date."""
        self.conn.execute(
            """INSERT OR REPLACE INTO regime_labels
               (station, target_date, regimes_active, path_class,
                confidence_tags, phase_summary, model_performance,
                signal_labels, signal_families_active,
                threshold_recommendations, review_path, review_source)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (station, target_date, regimes_active, path_class,
             confidence_tags, phase_summary, model_performance,
             signal_labels, signal_families_active,
             threshold_recommendations, review_path, review_source),
        )
        self.conn.commit()

    def get_regime_labels(
        self, station: str, target_date: str
    ) -> dict | None:
        """Get regime label for a station/date."""
        row = self.conn.execute(
            """SELECT regimes_active, path_class, confidence_tags,
                      phase_summary, model_performance, signal_labels,
                      signal_families_active, threshold_recommendations,
                      review_path, review_source
               FROM regime_labels
               WHERE station = ? AND target_date = ?""",
            (station, target_date),
        ).fetchone()
        if not row:
            return None
        return dict(
            regimes_active=row[0], path_class=row[1],
            confidence_tags=row[2], phase_summary=row[3],
            model_performance=row[4], signal_labels=row[5],
            signal_families_active=row[6],
            threshold_recommendations=row[7],
            review_path=row[8], review_source=row[9],
        )

    def get_recent_regime_labels(
        self, station: str, n: int = 3
    ) -> list[dict]:
        """Get the N most recent regime labels."""
        rows = self.conn.execute(
            """SELECT target_date, regimes_active, path_class,
                      confidence_tags, phase_summary, review_source
               FROM regime_labels
               WHERE station = ?
               ORDER BY target_date DESC LIMIT ?""",
            (station, n),
        ).fetchall()
        return [
            dict(target_date=r[0], regimes_active=r[1], path_class=r[2],
                 confidence_tags=r[3], phase_summary=r[4], review_source=r[5])
            for r in rows
        ]
