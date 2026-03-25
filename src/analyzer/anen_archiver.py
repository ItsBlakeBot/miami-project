"""TA8.5: HRRR + obs archiving for future Analog Ensemble (AnEn).

The NCAR Analog Ensemble method (Delle Monache et al. 2013) finds N most
similar historical forecasts to the current forecast, then uses the
corresponding historical observations as ensemble members.

This requires a systematic archive of paired (forecast, observation) data.
The longer the archive, the better the AnEn performs. Minimum useful: 1 year.
Target: 2+ years for seasonal coverage.

This module archives forecast-observation pairs daily after settlement.
Start archiving NOW — the data is the bottleneck, not the code.

Archive schema:
  For each (station, settlement_date, model, run_time):
    - forecast_high_f, forecast_low_f
    - cli_high_f, cli_low_f (settlement truth)
    - atmospheric context: cape, pw, cloud, wind_850
    - error: forecast - cli (for analog distance computation)
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

log = logging.getLogger(__name__)


def ensure_anen_archive_table(db: sqlite3.Connection) -> None:
    """Create the AnEn archive table if it doesn't exist."""
    db.execute("""
        CREATE TABLE IF NOT EXISTS anen_archive (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            station TEXT NOT NULL,
            settlement_date TEXT NOT NULL,
            model TEXT NOT NULL,
            source TEXT NOT NULL,
            run_time TEXT,
            forecast_high_f REAL,
            forecast_low_f REAL,
            cli_high_f REAL,
            cli_low_f REAL,
            error_high_f REAL,
            error_low_f REAL,
            cape REAL,
            precipitable_water_mm REAL,
            cloud_cover_pct REAL,
            wind_speed_850_ms REAL,
            wind_dir_850_deg REAL,
            created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
            UNIQUE(station, settlement_date, model, source, run_time)
        )
    """)
    db.execute("""
        CREATE INDEX IF NOT EXISTS idx_anen_station_date
        ON anen_archive(station, settlement_date)
    """)
    db.commit()


def archive_settlement_day(
    db: sqlite3.Connection,
    station: str,
    settlement_date: str,
) -> int:
    """Archive forecast-observation pairs for one settlement day.

    Called daily after CLI settlement arrives. Pairs each model forecast
    with the actual settlement values.

    Returns:
        Number of archive rows inserted.
    """
    ensure_anen_archive_table(db)

    # Get settlements
    settlements = db.execute(
        """SELECT market_type, actual_value_f
           FROM event_settlements
           WHERE station = ? AND settlement_date = ?
             AND actual_value_f IS NOT NULL""",
        (station, settlement_date),
    ).fetchall()

    if not settlements:
        return 0

    cli: dict[str, float] = {}
    for row in settlements:
        cli[row["market_type"]] = row["actual_value_f"]

    cli_high = cli.get("high")
    cli_low = cli.get("low")

    if cli_high is None and cli_low is None:
        return 0

    # Get model forecasts
    forecasts = db.execute(
        """SELECT model, source, forecast_high_f, forecast_low_f, run_time
           FROM model_forecasts
           WHERE station = ? AND forecast_date = ?
             AND (forecast_high_f IS NOT NULL OR forecast_low_f IS NOT NULL)
           ORDER BY id DESC""",
        (station, settlement_date),
    ).fetchall()

    # Deduplicate: latest per (model, source, run_time)
    seen: dict[tuple, dict] = {}
    for row in forecasts:
        key = (row["model"], row["source"] or "unknown", row["run_time"] or "")
        if key not in seen:
            seen[key] = row

    # Get atmospheric context (latest before settlement)
    atmos = db.execute(
        """SELECT cape, precipitable_water_mm
           FROM atmospheric_data
           WHERE station = ? AND valid_time_utc <= ? || 'T23:59:59Z'
           ORDER BY valid_time_utc DESC LIMIT 1""",
        (station, settlement_date),
    ).fetchone()

    cape = atmos["cape"] if atmos else None
    pw = atmos["precipitable_water_mm"] if atmos else None

    count = 0
    for (model, source, run_time), row in seen.items():
        fh = row["forecast_high_f"]
        fl = row["forecast_low_f"]
        eh = (fh - cli_high) if fh is not None and cli_high is not None else None
        el = (fl - cli_low) if fl is not None and cli_low is not None else None

        try:
            db.execute(
                """INSERT OR IGNORE INTO anen_archive
                   (station, settlement_date, model, source, run_time,
                    forecast_high_f, forecast_low_f,
                    cli_high_f, cli_low_f,
                    error_high_f, error_low_f,
                    cape, precipitable_water_mm)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (station, settlement_date, model, source, run_time,
                 fh, fl, cli_high, cli_low, eh, el, cape, pw),
            )
            count += 1
        except Exception:
            pass

    db.commit()
    return count


def backfill_anen_archive(
    db: sqlite3.Connection,
    station: str,
    lookback_days: int = 90,
    reference_date: str | None = None,
) -> int:
    """Backfill the AnEn archive from historical data.

    Call once to populate the archive with all available historical
    forecast-observation pairs.
    """
    if reference_date is None:
        ref = date.today()
    else:
        ref = date.fromisoformat(reference_date)

    total = 0
    for i in range(1, lookback_days + 1):
        d = (ref - timedelta(days=i)).isoformat()
        n = archive_settlement_day(db, station, d)
        total += n

    log.info("AnEn backfill: %d rows archived over %d days", total, lookback_days)
    return total


def main() -> None:
    """Backfill AnEn archive from historical data."""
    import argparse

    parser = argparse.ArgumentParser(description="Archive forecast-obs pairs for AnEn")
    parser.add_argument("--db", required=True)
    parser.add_argument("--station", default="KMIA")
    parser.add_argument("--lookback-days", type=int, default=90)
    parser.add_argument("--reference-date", default=None)
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row

    total = backfill_anen_archive(conn, args.station, args.lookback_days, args.reference_date)
    conn.close()

    print(f"AnEn archive: {total} rows backfilled for {args.station}")


if __name__ == "__main__":
    main()
