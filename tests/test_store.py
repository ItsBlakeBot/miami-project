"""Tests for the SQLite store layer."""

import os
import tempfile

import pytest

from collector.store.db import Store
from collector.store.schema import SCHEMA_VERSION
from collector.types import (
    EventSettlement,
    MarketSnapshot,
    ModelForecast,
    Observation,
    PressureLevelData,
)


@pytest.fixture
def store():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    s = Store(path)
    s.open()
    yield s
    s.close()
    os.unlink(path)


def test_schema_creation(store):
    """All 13 tables should be created."""
    tables = store.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    table_names = {t[0] for t in tables}

    expected = {
        "schema_meta", "model_forecasts", "observations", "pressure_levels",
        "market_snapshots", "event_settlements", "market_settlements",
        "model_scores", "diurnal_scores", "cloud_obs", "sse_events",
        "collection_runs",
    }
    assert expected.issubset(table_names)


def test_schema_version(store):
    row = store.conn.execute(
        "SELECT value FROM schema_meta WHERE key='version'"
    ).fetchone()
    assert row is not None
    assert row[0] == SCHEMA_VERSION


def test_insert_forecast(store):
    f = ModelForecast(
        station="KMIA",
        forecast_date="2026-03-14",
        model="GFS-Global",
        source="openmeteo",
        forecast_high_f=85.0,
        forecast_low_f=72.0,
        fetch_time_utc="2026-03-14T12:00:00Z",
    )
    assert store.insert_forecasts([f]) == 1

    # Duplicate should skip
    f2 = ModelForecast(
        station="KMIA",
        forecast_date="2026-03-14",
        model="GFS-Global",
        source="openmeteo",
        forecast_high_f=85.0,
        forecast_low_f=72.0,
        fetch_time_utc="2026-03-14T12:00:01Z",
    )
    assert store.insert_forecasts([f2]) == 0  # Values match, skip

    rows = store.conn.execute("SELECT * FROM model_forecasts").fetchall()
    assert len(rows) == 1


def test_insert_hourly_forecast(store):
    f = ModelForecast(
        station="KMIA",
        forecast_date="2026-03-14",
        model="GFS-Global",
        source="openmeteo",
        valid_time="2026-03-14T18:00:00Z",
        raw_temperature_f=83.5,
        fetch_time_utc="2026-03-14T12:00:00Z",
    )
    store.insert_forecasts([f])

    rows = store.get_forecasts_with_valid_time("KMIA", "2026-03-14")
    assert len(rows) == 1
    assert rows[0][0] == "GFS-Global"
    assert rows[0][3] == 83.5


def test_insert_observation(store):
    o = Observation(
        station="KMIA",
        timestamp_utc="2026-03-14T15:30:00Z",
        lst_date="2026-03-14",
        temperature_f=84.2,
        dew_point_f=72.1,
        wind_speed_mph=12.0,
        sky_cover_pct=50.0,
        sky_cover_code="SCT",
        wethr_high_nws_f=85.0,
        wethr_low_nws_f=71.0,
        source="wethr_1min",
    )
    store.insert_observation(o)

    rows = store.conn.execute("SELECT * FROM observations").fetchall()
    assert len(rows) == 1


def test_insert_pressure_level(store):
    p = PressureLevelData(
        station="KMIA",
        valid_time_utc="2026-03-14T18:00:00Z",
        model="gfs_seamless",
        temp_925_c=22.5,
        wind_speed_925=8.3,
        wind_dir_925=180.0,
        temp_850_c=15.2,
    )
    count = store.insert_pressure_levels([p])
    assert count == 1


def test_insert_market_snapshot(store):
    m = MarketSnapshot(
        ticker="KXHIGHMIA-26MAR15-T82",
        event_ticker="KXHIGHMIA-26MAR15",
        series_ticker="KXHIGHMIA",
        market_type="high",
        floor_strike=82.0,
        best_yes_bid_cents=45.0,
        best_yes_ask_cents=48.0,
        snapshot_time="2026-03-14T15:00:00Z",
    )
    store.insert_market_snapshot(m)

    rows = store.conn.execute("SELECT * FROM market_snapshots").fetchall()
    assert len(rows) == 1


def test_insert_event_settlement(store):
    s = EventSettlement(
        station="KMIA",
        settlement_date="2026-03-13",  # Climate day, NOT issue date
        market_type="high",
        actual_value_f=87.0,
        settlement_source="cli",
        received_at="2026-03-14T10:30:00Z",
    )
    store.insert_event_settlement(s)

    rows = store.conn.execute("SELECT * FROM event_settlements").fetchall()
    assert len(rows) == 1
    assert rows[0][2] == "2026-03-13"  # settlement_date


def test_settlement_upsert(store):
    """Inserting same station+date+type should replace."""
    s1 = EventSettlement(
        station="KMIA", settlement_date="2026-03-13", market_type="high",
        actual_value_f=87.0, settlement_source="cli",
    )
    s2 = EventSettlement(
        station="KMIA", settlement_date="2026-03-13", market_type="high",
        actual_value_f=88.0, settlement_source="cli",
    )
    store.insert_event_settlement(s1)
    store.insert_event_settlement(s2)

    rows = store.conn.execute("SELECT actual_value_f FROM event_settlements").fetchall()
    assert len(rows) == 1
    assert rows[0][0] == 88.0


def test_insert_diurnal_score(store):
    store.insert_diurnal_score(
        station="KMIA",
        model="GFS-Global",
        forecast_date="2026-03-14",
        valid_time_utc="2026-03-14T19:00:00Z",
        valid_hour_local=14,
        predicted_f=85.0,
        observed_f=83.5,
        obs_timestamp_utc="2026-03-14T18:58:00Z",
    )

    rows = store.conn.execute("SELECT error_f FROM diurnal_scores").fetchall()
    assert len(rows) == 1
    assert abs(rows[0][0] - 1.5) < 0.01  # 85.0 - 83.5


def test_insert_cloud_obs(store):
    store.insert_cloud_obs(
        station="KMIA",
        obs_time="2026-03-14T18:00:00Z",
        forecast_date="2026-03-14",
        obs_sky_pct=75.0,
        obs_sky_code="BKN",
        forecast_sky_pct=25.0,
        temp_error_f=2.5,
    )

    rows = store.conn.execute(
        "SELECT cloud_discrepancy_pct FROM cloud_obs"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == 50.0  # 75 - 25


def test_get_nearest_observation(store):
    for i, temp in enumerate([80.0, 82.0, 84.0]):
        store.insert_observation(Observation(
            station="KMIA",
            timestamp_utc=f"2026-03-14T{14+i}:00:00Z",
            lst_date="2026-03-14",
            temperature_f=temp,
            source="test",
        ))

    result = store.get_nearest_observation("KMIA", "2026-03-14T15:10:00Z")
    assert result is not None
    assert result[1] == 82.0  # 15:00 is closest to 15:10


def test_collection_run_log(store):
    store.log_collection_run("forecasts", "ok", 150)
    store.log_collection_run("nws", "error", 0, "timeout")

    rows = store.conn.execute(
        "SELECT collector, status, records_collected FROM collection_runs ORDER BY id"
    ).fetchall()
    assert len(rows) == 2
    assert rows[0] == ("forecasts", "ok", 150)
    assert rows[1] == ("nws", "error", 0)
