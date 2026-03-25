"""Trader's own SQLite schema — signal snapshots, trades, outcomes, parameter history.

This DB is separate from the city collector's DB. The trader reads from the
collector but writes only to its own database.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path


TABLES = [
    # -- Every poll cycle's signal state + estimated distribution --
    """
    CREATE TABLE IF NOT EXISTS signal_snapshots (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp_utc TEXT NOT NULL,
        station TEXT NOT NULL,
        target_date TEXT NOT NULL,
        market_type TEXT NOT NULL,
        hours_remaining REAL,

        -- Signal 1: Consensus
        consensus_f REAL,
        consensus_sigma REAL,
        raw_consensus_f REAL,
        n_models INTEGER,

        -- Signal 2: Obs divergence
        obs_current_f REAL,
        obs_trend_2hr REAL,
        obs_vs_consensus REAL,
        projected_extreme_f REAL,

        -- Signal 3: CAPE + PW
        cape REAL,
        pw_mm REAL,
        outflow_risk INTEGER,
        cape_trend_1hr REAL,

        -- Signal 4: Wind
        wind_dir_deg REAL,
        continental INTEGER,
        wind_shift INTEGER,

        -- Signal 5: Dew point
        dew_point_f REAL,
        evening_dew_mean_f REAL,
        estimated_floor_f REAL,
        dew_crash INTEGER,

        -- Signal 6: Nearby
        fawn_temp_f REAL,
        fawn_crash INTEGER,
        fawn_lead_minutes REAL,
        nearby_divergence_f REAL,
        nearby_crash_count INTEGER,

        -- Signal 7: Pressure
        pressure_hpa REAL,
        pressure_3hr_trend REAL,
        pressure_surge INTEGER,

        -- Estimator output
        estimated_mu REAL,
        estimated_sigma REAL,
        active_flags TEXT,          -- JSON list of triggered signal names
        adjustments TEXT,           -- JSON list of adjustment descriptions

        -- Running extremes
        running_high_f REAL,
        running_low_f REAL,

        created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
    )
    """,

    # -- Every trade decision (shadow, paper, or live) --
    """
    CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp_utc TEXT NOT NULL,
        station TEXT NOT NULL,
        target_date TEXT NOT NULL,
        market_type TEXT NOT NULL,
        ticker TEXT NOT NULL,
        side TEXT NOT NULL,              -- "yes" or "no"
        action TEXT NOT NULL DEFAULT 'buy',
        price_cents INTEGER,
        contracts INTEGER,
        our_probability REAL,
        market_probability REAL,
        edge REAL,
        kelly_size REAL,
        settlement_floor REAL,
        settlement_ceil REAL,
        signal_snapshot_id INTEGER REFERENCES signal_snapshots(id),
        mode TEXT NOT NULL,              -- "shadow", "paper", "live"

        -- Kalshi order tracking (live mode only)
        kalshi_order_id TEXT,
        order_status TEXT DEFAULT 'pending',
        fill_price_cents INTEGER,

        created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
    )
    """,

    # -- Settlement outcomes + P/L --
    """
    CREATE TABLE IF NOT EXISTS trade_outcomes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        trade_id INTEGER NOT NULL REFERENCES trades(id),
        target_date TEXT NOT NULL,
        market_type TEXT NOT NULL,
        cli_value_f REAL,
        winning_side TEXT,              -- "yes" or "no"
        our_side_won INTEGER,           -- boolean
        pnl_cents INTEGER,              -- per contract
        total_pnl_cents INTEGER,        -- pnl * contracts
        created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
    )
    """,

    # -- Parameter tuning audit trail --
    """
    CREATE TABLE IF NOT EXISTS parameter_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        parameter_name TEXT NOT NULL,
        old_value REAL,
        new_value REAL,
        reason TEXT,
        calibration_date TEXT,
        sample_count INTEGER,
        created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
    )
    """,
]

INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_snap_date ON signal_snapshots(station, target_date, market_type)",
    "CREATE INDEX IF NOT EXISTS idx_snap_time ON signal_snapshots(timestamp_utc)",
    "CREATE INDEX IF NOT EXISTS idx_trades_date ON trades(station, target_date, market_type)",
    "CREATE INDEX IF NOT EXISTS idx_trades_ticker ON trades(ticker)",
    "CREATE INDEX IF NOT EXISTS idx_trades_mode ON trades(mode, target_date)",
    "CREATE INDEX IF NOT EXISTS idx_outcomes_date ON trade_outcomes(target_date, market_type)",
    "CREATE INDEX IF NOT EXISTS idx_outcomes_trade ON trade_outcomes(trade_id)",
    "CREATE INDEX IF NOT EXISTS idx_params_name ON parameter_history(parameter_name, calibration_date)",
]


def init_trader_db(db_path: str | Path) -> sqlite3.Connection:
    """Create trader DB and initialize schema. Returns connection."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")

    for ddl in TABLES:
        conn.execute(ddl)
    for idx in INDEXES:
        conn.execute(idx)

    # Migrations for existing DBs.
    for migration in [
        "ALTER TABLE trades ADD COLUMN settlement_floor REAL",
        "ALTER TABLE trades ADD COLUMN settlement_ceil REAL",
    ]:
        try:
            conn.execute(migration)
        except sqlite3.OperationalError:
            pass

    conn.commit()
    return conn
