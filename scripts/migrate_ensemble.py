"""One-time migration: aggregate individual ensemble member rows into single summary rows.

Run on a DB copy first:
    cp miami_collector.db miami_collector_backup.db
    python3 scripts/migrate_ensemble.py --db miami_collector.db

This script:
1. Finds all individual ensemble member rows (GFS-Ensemble-mXX, ECMWF-IFS-Ensemble-mXX)
2. Groups them by (station, forecast_date, run_time, fetch_time_utc)
3. Computes aggregate stats (mean, std, percentiles)
4. Inserts one summary row per group
5. Deletes the individual member rows
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
from pathlib import Path


ENSEMBLE_PREFIXES = ["GFS-Ensemble-m", "ECMWF-IFS-Ensemble-m"]
FAMILY_NAMES = {
    "GFS-Ensemble-m": "GFS-Ensemble",
    "ECMWF-IFS-Ensemble-m": "ECMWF-IFS-Ensemble",
}


def _percentile(vals: list[float], p: float) -> float:
    s = sorted(vals)
    k = (len(s) - 1) * p / 100
    f_idx = int(k)
    c_idx = min(f_idx + 1, len(s) - 1)
    frac = k - f_idx
    return round(s[f_idx] + frac * (s[c_idx] - s[f_idx]), 2)


def _stats(vals: list[float]) -> dict:
    if not vals:
        return {}
    return {
        "mean": round(statistics.mean(vals), 2),
        "std": round(statistics.stdev(vals), 2) if len(vals) > 1 else 0.0,
        "min": round(min(vals), 2),
        "max": round(max(vals), 2),
        "p10": _percentile(vals, 10),
        "p25": _percentile(vals, 25),
        "p50": _percentile(vals, 50),
        "p75": _percentile(vals, 75),
        "p90": _percentile(vals, 90),
        "n_members": len(vals),
    }


def migrate(db_path: str, *, dry_run: bool = False) -> None:
    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row

    # Count existing member rows
    total_members = 0
    for prefix in ENSEMBLE_PREFIXES:
        row = conn.execute(
            "SELECT COUNT(*) FROM model_forecasts WHERE model LIKE ?",
            (f"{prefix}%",),
        ).fetchone()
        count = row[0]
        total_members += count
        print(f"  {prefix}*: {count} rows")

    if total_members == 0:
        print("No ensemble member rows to migrate.")
        conn.close()
        return

    print(f"\nTotal member rows to process: {total_members}")

    # Get distinct groups
    groups = conn.execute(
        """SELECT DISTINCT station, forecast_date, source, run_time, fetch_time_utc
           FROM model_forecasts
           WHERE model LIKE 'GFS-Ensemble-m%' OR model LIKE 'ECMWF-IFS-Ensemble-m%'
           ORDER BY forecast_date""",
    ).fetchall()

    print(f"Found {len(groups)} distinct (station, date, run_time, fetch_time) groups\n")

    inserted = 0
    deleted = 0

    for g in groups:
        station, forecast_date, source, run_time, fetch_time = (
            g["station"], g["forecast_date"], g["source"], g["run_time"], g["fetch_time_utc"],
        )

        for prefix, family_name in FAMILY_NAMES.items():
            rows = conn.execute(
                """SELECT forecast_high_f, forecast_low_f FROM model_forecasts
                   WHERE station=? AND forecast_date=? AND source=? AND model LIKE ?
                   AND (run_time=? OR (run_time IS NULL AND ? IS NULL))
                   AND (fetch_time_utc=? OR (fetch_time_utc IS NULL AND ? IS NULL))""",
                (station, forecast_date, source, f"{prefix}%",
                 run_time, run_time, fetch_time, fetch_time),
            ).fetchall()

            if not rows:
                continue

            member_highs = [r["forecast_high_f"] for r in rows if r["forecast_high_f"] is not None]
            member_lows = [r["forecast_low_f"] for r in rows if r["forecast_low_f"] is not None]

            if not member_highs and not member_lows:
                continue

            ensemble_stats = {
                "high": _stats(member_highs),
                "low": _stats(member_lows),
            }

            if dry_run:
                print(f"  [DRY] {family_name} {forecast_date} {run_time}: "
                      f"{len(rows)} members -> 1 aggregate "
                      f"(high={ensemble_stats['high'].get('mean')}, "
                      f"low={ensemble_stats['low'].get('mean')})")
            else:
                # Insert aggregate row
                conn.execute(
                    """INSERT INTO model_forecasts
                       (station, forecast_date, model, source, run_time,
                        forecast_high_f, forecast_low_f, fetch_time_utc,
                        source_record_json)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (station, forecast_date, family_name, source, run_time,
                     ensemble_stats["high"].get("mean"),
                     ensemble_stats["low"].get("mean"),
                     fetch_time,
                     json.dumps({"ensemble_stats": ensemble_stats})),
                )
                inserted += 1

                # Delete individual member rows
                cur = conn.execute(
                    """DELETE FROM model_forecasts
                       WHERE station=? AND forecast_date=? AND source=? AND model LIKE ?
                       AND (run_time=? OR (run_time IS NULL AND ? IS NULL))
                       AND (fetch_time_utc=? OR (fetch_time_utc IS NULL AND ? IS NULL))""",
                    (station, forecast_date, source, f"{prefix}%",
                     run_time, run_time, fetch_time, fetch_time),
                )
                deleted += cur.rowcount

    if not dry_run:
        conn.commit()
        print(f"Inserted {inserted} aggregate rows, deleted {deleted} member rows")
        print("Running VACUUM to reclaim space...")
        conn.execute("VACUUM")
        print("Done.")
    else:
        print(f"\n[DRY RUN] Would insert {inserted} aggregate rows")

    conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate ensemble members to aggregate rows")
    parser.add_argument("--db", required=True, help="Path to miami_collector.db")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without changing DB")
    args = parser.parse_args()

    if not Path(args.db).exists():
        print(f"Error: {args.db} not found")
        return

    print(f"Migrating ensemble data in {args.db}")
    if args.dry_run:
        print("(DRY RUN — no changes will be made)\n")
    migrate(args.db, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
