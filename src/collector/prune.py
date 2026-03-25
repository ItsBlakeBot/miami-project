"""Database retention and pruning helpers."""

from __future__ import annotations

import argparse
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

UTC = timezone.utc
log = logging.getLogger(__name__)

DEFAULT_RETENTION_DAYS = 60


@dataclass(frozen=True)
class RetentionRule:
    table: str
    time_column: str
    keep_days: int = DEFAULT_RETENTION_DAYS


# Default retention rules. Override via config.toml [retention] section.
#
# Tables NOT listed here are kept permanently: event_settlements,
# market_settlements, signal_calibration, active_brackets, nearby_stations,
# schema_meta, model_scores, regime_labels, regime_labels_hdp_test,
# paper_trades, paper_trade_settlements, paper_trade_marks.
#
# DS3M training data strategy:
#   - model_forecasts + observations: PERMANENT. These are the core
#     forecast-observation pairs needed for EMOS/BOA/DIMMPF training.
#     At ~500 rows/day for forecasts and ~1500/day for obs, a full year
#     is ~730K rows (~200MB). This is manageable and essential.
#   - nearby_observations: PERMANENT for LETKF spatial training.
#   - fawn/sst: PERMANENT for regime/microclimate signal training.
#   - atmospheric_data: PERMANENT for CAPE/PW regime characterization.
#   - market_snapshots: 90 days. Useful for market-implied density
#     backtesting but not critical for DS3M core.
#   - bracket_estimates: 90 days. Can be regenerated from forecasts.
#   - Ephemeral/diagnostic tables: aggressive pruning (14-30 days).
RETENTION_RULES = (
    # DS3M TRAINING DATA — keep permanently (no rule = no pruning)
    # model_forecasts: not listed → permanent
    # observations: not listed → permanent
    # nearby_observations: not listed → permanent
    # fawn_observations: not listed → permanent
    # sst_observations: not listed → permanent
    # atmospheric_data: not listed → permanent

    # TRADING DATA — extended retention for backtesting
    RetentionRule("market_snapshots", "snapshot_time", 90),
    RetentionRule("bracket_estimates", "timestamp_utc", 90),
    RetentionRule("forward_curves", "snapshot_time_utc", 90),

    # DIAGNOSTIC/EPHEMERAL — aggressive pruning
    RetentionRule("paper_trade_marks", "mark_time", 30),
    RetentionRule("ds3m_paper_trade_marks", "mark_time", 30),
    RetentionRule("pressure_levels", "valid_time_utc", 60),
    RetentionRule("collection_runs", "started_at", 120),
    RetentionRule("sse_events", "received_at", 14),
    RetentionRule("model_consensus", "created_at", 60),
    RetentionRule("diurnal_scores", "created_at", 60),
    RetentionRule("cloud_obs", "created_at", 30),
    RetentionRule("signal_events", "created_at", 60),
    RetentionRule("signal_scores", "created_at", 60),
)


def default_db_path() -> Path:
    root = Path(__file__).resolve().parents[2]
    primary = root / "miami_collector.db"
    if primary.exists():
        return primary
    return root / "src" / "miami_collector.db"


def prune_database(
    db_path: str | Path,
    *,
    now_utc: datetime | None = None,
    retention_overrides: dict[str, int] | None = None,
) -> dict[str, int]:
    """Delete rows older than their retention period.

    Args:
        db_path: Path to the SQLite database.
        now_utc: Override current time (for testing).
        retention_overrides: {table_name: keep_days} to override defaults.
    """
    now_utc = now_utc or datetime.now(tz=UTC)
    db_path = Path(db_path)
    overrides = retention_overrides or {}

    conn = sqlite3.connect(db_path)
    try:
        counts: dict[str, int] = {}
        for rule in RETENTION_RULES:
            keep_days = overrides.get(rule.table, rule.keep_days)
            cutoff = (now_utc - timedelta(days=keep_days)).strftime("%Y-%m-%dT%H:%M:%SZ")
            sql = f"DELETE FROM {rule.table} WHERE {rule.time_column} < ?"
            try:
                cur = conn.execute(sql, (cutoff,))
                counts[rule.table] = cur.rowcount if cur.rowcount is not None else 0
            except sqlite3.OperationalError:
                counts[rule.table] = 0
        conn.commit()
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.execute("VACUUM")
        return counts
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prune old rows from the Miami collector DB.")
    parser.add_argument("--db", default=str(default_db_path()), help="Path to collector DB")
    args = parser.parse_args(argv)

    counts = prune_database(args.db)
    total = sum(counts.values())
    print(f"pruned {total} rows from {args.db}")
    for table, count in counts.items():
        if count > 0:
            print(f"  {table}: {count}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
