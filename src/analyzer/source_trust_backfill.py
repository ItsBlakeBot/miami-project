"""Backfill source trust metrics from historical forecast-vs-settlement data.

Phase mapping:
- T0.4 source-trust backfill bootstrap

Produces a machine-readable artifact with per-source and per-family error
metrics plus basic time-to-settlement reliability cuts.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from engine.replay_context import parse_utc

UTC = timezone.utc


@dataclass(frozen=True)
class ErrorRecord:
    forecast_date: str
    market_type: str
    source: str
    model: str
    source_key: str
    source_family: str
    issued_at_utc: str
    hours_to_settlement: float
    ttl_bucket: str
    error_f: float
    abs_error_f: float


def _climate_day_end_utc(forecast_date: str) -> datetime:
    d = datetime.strptime(forecast_date, "%Y-%m-%d") + timedelta(days=1)
    return datetime(d.year, d.month, d.day, 5, 0, 0, tzinfo=UTC)


def _ttl_bucket(hours_to_settlement: float) -> str:
    h = max(0.0, float(hours_to_settlement))
    if h < 3:
        return "0-3h"
    if h < 6:
        return "3-6h"
    if h < 12:
        return "6-12h"
    if h < 24:
        return "12-24h"
    return "24h+"


def _metrics(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "n": 0,
            "mae": None,
            "rmse": None,
            "bias": None,
        }
    n = len(values)
    abs_vals = [abs(v) for v in values]
    mse = sum(v * v for v in values) / n
    return {
        "n": n,
        "mae": round(sum(abs_vals) / n, 6),
        "rmse": round(math.sqrt(mse), 6),
        "bias": round(sum(values) / n, 6),
    }


def run_backfill(
    db_path: str | Path,
    *,
    station: str,
    lookback_days: int = 90,
    min_days: int = 30,
    as_of_utc: datetime | None = None,
) -> dict:
    now = (as_of_utc or datetime.now(tz=UTC)).astimezone(UTC)
    end_date = now.strftime("%Y-%m-%d")
    start_date = (now - timedelta(days=max(1, int(lookback_days)))).strftime("%Y-%m-%d")

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    try:
        settle_rows = conn.execute(
            """SELECT settlement_date, market_type, actual_value_f
               FROM event_settlements
               WHERE station = ?
                 AND settlement_date >= ?
                 AND settlement_date <= ?
                 AND actual_value_f IS NOT NULL""",
            (station, start_date, end_date),
        ).fetchall()

        actuals: dict[tuple[str, str], float] = {}
        for row in settle_rows:
            market_type = str(row["market_type"] or "").lower().strip()
            if market_type not in {"high", "low"}:
                continue
            actuals[(str(row["settlement_date"]), market_type)] = float(row["actual_value_f"])

        forecast_rows = conn.execute(
            """SELECT forecast_date, model, source, run_time, fetch_time_utc,
                      forecast_high_f, forecast_low_f
               FROM model_forecasts
               WHERE station = ?
                 AND forecast_date >= ?
                 AND forecast_date <= ?
                 AND valid_time IS NULL
                 AND (forecast_high_f IS NOT NULL OR forecast_low_f IS NOT NULL)
               ORDER BY forecast_date, model, source, COALESCE(fetch_time_utc, run_time)""",
            (station, start_date, end_date),
        ).fetchall()
    finally:
        conn.close()

    records: list[ErrorRecord] = []
    missing_actual_rows = 0
    missing_issued_rows = 0

    for row in forecast_rows:
        forecast_date = str(row["forecast_date"])
        issued_at = parse_utc(row["fetch_time_utc"] or row["run_time"])
        if issued_at is None:
            missing_issued_rows += 1
            continue

        settle_end = _climate_day_end_utc(forecast_date)
        ttl_h = max(0.0, (settle_end - issued_at).total_seconds() / 3600.0)
        bucket = _ttl_bucket(ttl_h)

        source = str(row["source"] or "unknown")
        model = str(row["model"] or "unknown")
        source_key = f"{source}:{model}"

        for market_type, forecast_value in (
            ("high", row["forecast_high_f"]),
            ("low", row["forecast_low_f"]),
        ):
            if forecast_value is None:
                continue
            actual = actuals.get((forecast_date, market_type))
            if actual is None:
                missing_actual_rows += 1
                continue
            err = float(forecast_value) - float(actual)
            records.append(
                ErrorRecord(
                    forecast_date=forecast_date,
                    market_type=market_type,
                    source=source,
                    model=model,
                    source_key=source_key,
                    source_family=source,
                    issued_at_utc=issued_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    hours_to_settlement=round(ttl_h, 4),
                    ttl_bucket=bucket,
                    error_f=round(err, 6),
                    abs_error_f=round(abs(err), 6),
                )
            )

    by_source_key: dict[str, list[float]] = defaultdict(list)
    by_family: dict[str, list[float]] = defaultdict(list)
    by_family_bucket: dict[tuple[str, str], list[float]] = defaultdict(list)

    for rec in records:
        by_source_key[rec.source_key].append(rec.error_f)
        by_family[rec.source_family].append(rec.error_f)
        by_family_bucket[(rec.source_family, rec.ttl_bucket)].append(rec.error_f)

    source_metrics = {
        key: _metrics(vals)
        for key, vals in sorted(by_source_key.items())
    }
    family_metrics = {
        key: _metrics(vals)
        for key, vals in sorted(by_family.items())
    }

    family_reliability: dict[str, dict[str, dict[str, float | int | None]]] = {}
    for (family, bucket), vals in sorted(by_family_bucket.items()):
        family_reliability.setdefault(family, {})[bucket] = _metrics(vals)

    covered_days = sorted({r.forecast_date for r in records})

    summary = {
        "station": station,
        "as_of_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "lookback_days": int(lookback_days),
        "min_days_target": int(min_days),
        "start_date": start_date,
        "end_date": end_date,
        "records": len(records),
        "covered_days": len(covered_days),
        "covered_days_list": covered_days,
        "sufficient_days": len(covered_days) >= int(min_days),
        "missing_actual_rows": int(missing_actual_rows),
        "missing_issued_rows": int(missing_issued_rows),
        "metrics_by_source_key": source_metrics,
        "metrics_by_family": family_metrics,
        "reliability_by_family_ttl_bucket": family_reliability,
        "ttl_buckets": ["0-3h", "3-6h", "6-12h", "12-24h", "24h+"],
    }

    return summary


def _write_markdown(summary: dict, out_path: Path) -> None:
    lines: list[str] = []
    lines.append("# Source Trust Backfill Summary")
    lines.append("")
    lines.append(f"- Station: {summary['station']}")
    lines.append(f"- Window: {summary['start_date']} → {summary['end_date']}")
    lines.append(f"- Records: {summary['records']}")
    lines.append(f"- Covered days: {summary['covered_days']}")
    lines.append(f"- Sufficient days (target {summary['min_days_target']}): {summary['sufficient_days']}")
    lines.append("")

    lines.append("## Per-family metrics")
    for family, metrics in summary["metrics_by_family"].items():
        lines.append(
            f"- {family}: n={metrics['n']}, mae={metrics['mae']}, rmse={metrics['rmse']}, bias={metrics['bias']}"
        )
    lines.append("")

    lines.append("## Reliability by family / time-to-settlement bucket")
    for family, buckets in summary["reliability_by_family_ttl_bucket"].items():
        lines.append(f"### {family}")
        for bucket, metrics in buckets.items():
            lines.append(
                f"- {bucket}: n={metrics['n']}, mae={metrics['mae']}, rmse={metrics['rmse']}, bias={metrics['bias']}"
            )
        lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Backfill source trust metrics from settled weather outcomes")
    parser.add_argument("--db", default="miami_collector.db")
    parser.add_argument("--station", default="KMIA")
    parser.add_argument("--lookback-days", type=int, default=90)
    parser.add_argument("--min-days", type=int, default=30)
    parser.add_argument("--out", default="analysis_data/source_trust_backfill.json")
    parser.add_argument("--md-out", default="analysis_data/source_trust_backfill.md")
    parser.add_argument("--as-of-utc", help="Optional replay anchor timestamp (ISO)")
    args = parser.parse_args(argv)

    as_of = parse_utc(args.as_of_utc) if args.as_of_utc else None
    summary = run_backfill(
        args.db,
        station=args.station,
        lookback_days=max(1, int(args.lookback_days)),
        min_days=max(1, int(args.min_days)),
        as_of_utc=as_of,
    )

    out_json = Path(args.out)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    _write_markdown(summary, Path(args.md_out))

    print(json.dumps({
        "ok": True,
        "records": summary["records"],
        "covered_days": summary["covered_days"],
        "sufficient_days": summary["sufficient_days"],
        "out": str(out_json),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
