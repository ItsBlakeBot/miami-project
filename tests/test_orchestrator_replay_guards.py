from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from engine.orchestrator import _build_remaining_move_snapshots

UTC = timezone.utc


def _make_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE model_forecasts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            station TEXT,
            forecast_date TEXT,
            model TEXT,
            source TEXT,
            valid_time TEXT,
            run_time TEXT,
            fetch_time_utc TEXT,
            raw_temperature_f REAL,
            source_record_json TEXT
        );

        CREATE TABLE model_scores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            station TEXT,
            model TEXT,
            market_type TEXT,
            mae REAL,
            sample_count INTEGER,
            score_date TEXT
        );
        """
    )
    return conn


def test_remaining_snapshot_skips_unparseable_valid_time() -> None:
    conn = _make_conn()
    conn.execute(
        """INSERT INTO model_forecasts
           (station, forecast_date, model, source, valid_time, run_time, fetch_time_utc, raw_temperature_f)
           VALUES (?,?,?,?,?,?,?,?)""",
        ("KMIA", "2026-03-21", "gfs", "ncep", "not-a-ts", "2026-03-21T10:00:00Z", "2026-03-21T10:00:00Z", 80.0),
    )

    snaps = _build_remaining_move_snapshots(
        conn,
        station="KMIA",
        target_date="2026-03-21",
        now_utc=datetime(2026, 3, 21, 11, 0, tzinfo=UTC),
        running_high_f=75.0,
        running_low_f=None,
    )

    assert snaps == []


def test_remaining_snapshot_skips_future_issued_rows() -> None:
    conn = _make_conn()
    conn.execute(
        """INSERT INTO model_forecasts
           (station, forecast_date, model, source, valid_time, run_time, fetch_time_utc, raw_temperature_f)
           VALUES (?,?,?,?,?,?,?,?)""",
        (
            "KMIA",
            "2026-03-21",
            "hrrr",
            "ncep",
            "2026-03-21T12:00:00Z",
            "2026-03-21T12:00:00Z",
            "2026-03-21T12:00:00Z",  # future issued relative to now below
            81.0,
        ),
    )

    snaps = _build_remaining_move_snapshots(
        conn,
        station="KMIA",
        target_date="2026-03-21",
        now_utc=datetime(2026, 3, 21, 11, 0, tzinfo=UTC),
        running_high_f=75.0,
        running_low_f=None,
    )

    assert snaps == []


def test_remaining_snapshot_accepts_valid_forward_row() -> None:
    conn = _make_conn()
    conn.execute(
        """INSERT INTO model_forecasts
           (station, forecast_date, model, source, valid_time, run_time, fetch_time_utc, raw_temperature_f)
           VALUES (?,?,?,?,?,?,?,?)""",
        (
            "KMIA",
            "2026-03-21",
            "hrrr",
            "ncep",
            "2026-03-21T12:00:00Z",
            "2026-03-21T10:00:00Z",
            "2026-03-21T10:00:00Z",
            81.0,
        ),
    )

    snaps = _build_remaining_move_snapshots(
        conn,
        station="KMIA",
        target_date="2026-03-21",
        now_utc=datetime(2026, 3, 21, 11, 0, tzinfo=UTC),
        running_high_f=75.0,
        running_low_f=None,
    )

    assert len(snaps) == 1
    assert snaps[0].source_name == "hrrr"
    assert snaps[0].forecast_high_f is not None
