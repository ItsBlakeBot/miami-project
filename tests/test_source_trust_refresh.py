from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

from analyzer.source_trust_refresh import run_refresh

UTC = timezone.utc


def _init_db(path: str) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            """CREATE TABLE model_forecasts (
                station TEXT,
                forecast_date TEXT,
                model TEXT,
                source TEXT,
                run_time TEXT,
                valid_time TEXT,
                forecast_high_f REAL,
                forecast_low_f REAL,
                fetch_time_utc TEXT
            )"""
        )
        conn.execute(
            """CREATE TABLE event_settlements (
                station TEXT,
                settlement_date TEXT,
                market_type TEXT,
                actual_value_f REAL
            )"""
        )

        conn.executemany(
            "INSERT INTO event_settlements (station, settlement_date, market_type, actual_value_f) VALUES (?, ?, ?, ?)",
            [
                ("KMIA", "2026-03-08", "high", 84.0),
                ("KMIA", "2026-03-08", "low", 69.0),
                ("KMIA", "2026-03-09", "high", 83.0),
                ("KMIA", "2026-03-09", "low", 68.0),
            ],
        )

        conn.executemany(
            """INSERT INTO model_forecasts (
                station, forecast_date, model, source, run_time, valid_time,
                forecast_high_f, forecast_low_f, fetch_time_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                ("KMIA", "2026-03-08", "gfs", "openmeteo", "2026-03-08T00:00:00Z", None, 85.0, 68.0, None),
                ("KMIA", "2026-03-09", "gfs", "openmeteo", "2026-03-09T00:00:00Z", None, 82.0, 69.0, None),
                ("KMIA", "2026-03-08", "ecmwf", "wethr", "2026-03-08T00:00:00Z", None, 83.0, 70.0, None),
                ("KMIA", "2026-03-09", "ecmwf", "wethr", "2026-03-09T00:00:00Z", None, 84.0, 67.0, None),
            ],
        )
        conn.commit()
    finally:
        conn.close()


def test_source_trust_refresh_writes_summary_and_state(tmp_path) -> None:
    db_path = str(tmp_path / "test.sqlite")
    _init_db(db_path)

    result = run_refresh(
        db_path,
        station="KMIA",
        lookback_days=30,
        min_days=1,
        as_of_utc=datetime(2026, 3, 10, 12, 0, tzinfo=UTC),
        summary_out=tmp_path / "source_trust_backfill.json",
        summary_md_out=tmp_path / "source_trust_backfill.md",
        state_path=str(tmp_path / "source_trust_state.json"),
        max_step_per_refresh=0.05,
    )

    assert result["covered_days"] == 2
    assert result["sufficient_days"] is True
    assert "openmeteo" in result["family_multipliers"]

    summary = json.loads((tmp_path / "source_trust_backfill.json").read_text(encoding="utf-8"))
    state = json.loads((tmp_path / "source_trust_state.json").read_text(encoding="utf-8"))

    assert summary["covered_days"] == 2
    assert "family_multipliers" in state
    assert state["family_multipliers"]["openmeteo"] == result["family_multipliers"]["openmeteo"]
