from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from analyzer.archive_audit import audit_connection, report_to_markdown

UTC = timezone.utc


def _build_minimal_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE model_forecasts (
            station TEXT,
            forecast_date TEXT,
            model TEXT,
            source TEXT,
            run_time TEXT,
            valid_time TEXT,
            raw_temperature_f REAL,
            fetch_time_utc TEXT
        );

        CREATE TABLE observations (
            station TEXT,
            timestamp_utc TEXT,
            temperature_f REAL,
            dew_point_f REAL,
            pressure_hpa REAL,
            sky_cover_pct REAL,
            source TEXT
        );

        CREATE TABLE nearby_observations (
            timestamp_utc TEXT,
            air_temp_f REAL,
            temp_delta_vs_kmia REAL
        );

        CREATE TABLE fawn_observations (
            timestamp_utc TEXT,
            air_temp_f REAL,
            dew_point_f REAL
        );

        CREATE TABLE atmospheric_data (
            station TEXT,
            valid_time_utc TEXT,
            cape REAL,
            precipitable_water_mm REAL
        );

        CREATE TABLE forward_curves (
            station TEXT,
            snapshot_time_utc TEXT,
            target_date TEXT,
            valid_hour_utc TEXT,
            hours_ahead REAL
        );

        CREATE TABLE market_snapshots (
            ticker TEXT,
            forecast_date TEXT,
            market_type TEXT,
            snapshot_time TEXT,
            best_yes_bid_cents REAL,
            best_yes_ask_cents REAL,
            best_no_bid_cents REAL,
            best_no_ask_cents REAL
        );

        CREATE TABLE active_brackets (
            ticker TEXT,
            target_date TEXT,
            market_type TEXT,
            settlement_floor REAL,
            settlement_ceil REAL
        );

        CREATE TABLE bracket_estimates (
            station TEXT,
            target_date TEXT,
            ticker TEXT,
            probability REAL,
            timestamp_utc TEXT
        );

        CREATE TABLE market_settlements (
            ticker TEXT,
            station TEXT,
            forecast_date TEXT,
            market_type TEXT,
            winning_side TEXT
        );

        CREATE TABLE event_settlements (
            station TEXT,
            settlement_date TEXT,
            market_type TEXT,
            actual_value_f REAL
        );
        """
    )


def test_archive_audit_detects_missing_table_and_columns() -> None:
    conn = sqlite3.connect(":memory:")
    conn.execute(
        """CREATE TABLE observations (
                station TEXT,
                timestamp_utc TEXT,
                temperature_f REAL
            )"""
    )

    report = audit_connection(conn, db_path=":memory:", station="KMIA")

    assert "model_forecasts" in report.missing_tables
    obs = next(t for t in report.table_audits if t.table == "observations")
    assert obs.present is True
    assert "dew_point_f" in obs.missing_columns
    assert report.is_replay_ready is False


def test_archive_audit_freshness_minutes_and_status() -> None:
    conn = sqlite3.connect(":memory:")
    _build_minimal_schema(conn)

    conn.execute(
        "INSERT INTO observations (station, timestamp_utc, temperature_f, dew_point_f, pressure_hpa, sky_cover_pct, source) VALUES (?,?,?,?,?,?,?)",
        ("KMIA", "2026-03-21T10:00:00+00:00", 78.0, 69.0, 1012.0, 55.0, "wethr"),
    )

    report = audit_connection(
        conn,
        db_path=":memory:",
        station="KMIA",
        now_utc=datetime(2026, 3, 21, 10, 30, tzinfo=UTC),
    )

    obs = next(t for t in report.table_audits if t.table == "observations")
    assert obs.latest_timestamp == "2026-03-21T10:00:00+00:00"
    assert obs.freshness_mode == "realtime"
    assert obs.freshness_status == "stale"
    assert obs.freshness_minutes == 30.0


def test_archive_audit_future_expected_and_unparseable() -> None:
    conn = sqlite3.connect(":memory:")
    _build_minimal_schema(conn)

    conn.execute(
        "INSERT INTO atmospheric_data (station, valid_time_utc, cape, precipitable_water_mm) VALUES (?,?,?,?)",
        ("KMIA", "2026-03-21T14:00:00Z", 1000.0, 35.0),
    )
    conn.execute(
        "INSERT INTO market_snapshots (ticker, forecast_date, market_type, snapshot_time, best_yes_bid_cents, best_yes_ask_cents, best_no_bid_cents, best_no_ask_cents) VALUES (?,?,?,?,?,?,?,?)",
        ("TEST", "2026-03-21", "high", "bad-ts", 10, 11, 89, 90),
    )

    report = audit_connection(
        conn,
        db_path=":memory:",
        station="KMIA",
        now_utc=datetime(2026, 3, 21, 10, 30, tzinfo=UTC),
    )

    atm = next(t for t in report.table_audits if t.table == "atmospheric_data")
    assert atm.freshness_mode == "forecast_validity"
    assert atm.freshness_status == "future_expected"
    assert atm.freshness_minutes is not None and atm.freshness_minutes < 0

    mkt = next(t for t in report.table_audits if t.table == "market_snapshots")
    assert mkt.freshness_mode == "realtime"
    assert mkt.freshness_status == "unparseable"
    assert mkt.freshness_minutes is None


def test_report_markdown_contains_summary() -> None:
    conn = sqlite3.connect(":memory:")
    _build_minimal_schema(conn)

    report = audit_connection(conn, db_path="demo.db", station="KMIA")
    md = report_to_markdown(report)

    assert "# Archive Audit Report" in md
    assert "`demo.db`" in md
    assert "Replay-ready" in md
    assert "freshness status" in md
