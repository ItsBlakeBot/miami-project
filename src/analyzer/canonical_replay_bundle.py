"""Canonical replay bundle entry point.

Phase mapping:
- T0.2 canonical replay harness hardening
- T0.3 canonical metric/report package

Runs the remaining-only replay backtest with unified date selection and emits
one bundled report artifact for model comparison.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from analyzer.changepoint_replay_compare import run_compare
from analyzer.replay_remaining_backtest import _settled_dates, run_replay


def _to_markdown(bundle: dict) -> str:
    replay = bundle["replay_summary"]
    lines: list[str] = []
    lines.append("# Canonical Replay Bundle")
    lines.append("")
    lines.append(f"- Station: {bundle['station']}")
    lines.append(f"- Dates: {bundle['date_window']['start']} → {bundle['date_window']['end']}")
    lines.append(f"- Resolved dates: {bundle['date_window']['resolved_dates']}")
    lines.append("")

    lines.append("## Core metrics")
    lines.append(f"- Brier (all estimates): {replay.get('brier_all_estimates')}")
    lines.append(f"- Log-loss (all estimates): {replay.get('logloss_all_estimates')}")
    lines.append(f"- Trade Brier: {replay.get('trade_brier')}")
    lines.append(f"- Sharpness |p-0.5| mean: {replay.get('sharpness_mean_abs_p_minus_0_5')}")
    lines.append("")

    remaining = replay.get("remaining_target_metrics") or {}
    if remaining:
        lines.append("## Remaining-target metrics")
        for market_type in ("high", "low"):
            row = remaining.get(market_type) or {}
            lines.append(
                f"- {market_type}: n={row.get('n')}, mae={row.get('mae')}, rmse={row.get('rmse')}, crps={row.get('crps')}"
            )
        if remaining.get("crps_method"):
            lines.append(f"- CRPS method: {remaining.get('crps_method')}")
        lines.append("")

    lines.append("## Trading outcomes")
    lines.append(f"- Trades: {replay.get('trades')}")
    lines.append(f"- Contracts: {replay.get('contracts')}")
    lines.append(f"- Realized PnL cents: {replay.get('pnl_cents')}")
    lines.append(f"- Expected value cents total: {replay.get('expected_value_cents_total')}")
    lines.append(f"- Expected - realized cents: {replay.get('expected_minus_realized_cents')}")
    lines.append("")

    trade_quality = replay.get("trade_quality_cuts") or {}
    if trade_quality:
        lines.append("## Trade-quality cuts")
        for row in trade_quality.get("by_market_type") or []:
            lines.append(
                f"- market={row.get('market_type')}: n={row.get('n')}, win_rate={row.get('win_rate')}, avg_ev={row.get('avg_expected_value_cents')}, avg_pnl={row.get('avg_realized_pnl_cents')}"
            )
        lines.append("")

    regime_cuts = replay.get("regime_cuts") or []
    if regime_cuts:
        lines.append("## Regime cuts")
        for row in regime_cuts[:12]:
            lines.append(
                f"- {row.get('market_type')} / {row.get('regime')}: n={row.get('n')}, mae={row.get('remaining_mae')}, rmse={row.get('remaining_rmse')}, crps={row.get('remaining_crps')}"
            )
        lines.append("")

    lines.append("## Worst-day diagnostics")
    wd = replay.get("worst_day_diagnostics") or {}
    lines.append(f"- Highest Brier day: {wd.get('highest_brier_day')}")
    lines.append(f"- Lowest realized PnL day: {wd.get('lowest_realized_pnl_day')}")
    lines.append("")

    ttl_cuts = replay.get("ttl_cuts") or {}
    if ttl_cuts:
        lines.append("## TTL cuts")
        for bucket in ("0-3h", "3-6h", "6-12h", "12-24h", "24h+"):
            row = ttl_cuts.get(bucket)
            if not row:
                continue
            rem = row.get("remaining_targets") or {}
            lines.append(
                f"- {bucket}: n={row.get('n')}, brier={row.get('brier')}, logloss={row.get('logloss')}, rem_high_mae={rem.get('remaining_high_mae')}, rem_low_mae={rem.get('remaining_low_mae')}"
            )
        lines.append("")

    cp = bundle.get("changepoint_compare")
    if cp:
        lines.append("## Changepoint compare (BOCPD vs CUSUM)")
        lines.append(f"- Events: {cp['counts']['events']}")
        lines.append(f"- BOCPD fires: {cp['bocpd_enabled']['total_fires']}")
        lines.append(f"- CUSUM fires: {cp['cusum_only']['total_fires']}")
        lines.append(f"- Delta false-positive proxy: {cp['delta']['false_positive_proxy']}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Canonical replay bundle runner")
    parser.add_argument("--db", default="miami_collector.db")
    parser.add_argument("--station", default="KMIA")
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--date", action="append", dest="dates")
    parser.add_argument("--step-minutes", type=int, default=30)
    parser.add_argument("--max-cycles-per-day", type=int)
    parser.add_argument("--max-trades-per-ticker", type=int, default=3)
    parser.add_argument("--min-reentry-minutes", type=int, default=90)
    parser.add_argument("--adaptive-lookback-days", type=int, default=2)
    parser.add_argument("--include-changepoint-compare", action="store_true")
    parser.add_argument("--out", default="analysis_data/canonical_replay_bundle.json")
    parser.add_argument("--md-out", default="analysis_data/canonical_replay_bundle.md")
    args = parser.parse_args(argv)

    conn = sqlite3.connect(str(args.db))
    try:
        dates = _settled_dates(conn, args.station, args.start_date, args.end_date, args.dates)
    finally:
        conn.close()

    replay_summary, _trades = run_replay(
        db_path=args.db,
        station=args.station,
        dates=dates,
        step_minutes=max(1, int(args.step_minutes)),
        max_cycles_per_day=args.max_cycles_per_day,
        max_trades_per_ticker=max(1, int(args.max_trades_per_ticker)),
        min_reentry_minutes=max(0, int(args.min_reentry_minutes)),
        adaptive_lookback_days=max(1, int(args.adaptive_lookback_days)),
    )

    cp_compare = None
    if args.include_changepoint_compare:
        cp_compare = run_compare(
            args.db,
            station=args.station,
            start_date=args.start_date,
            end_date=args.end_date,
        )

    bundle = {
        "station": args.station,
        "date_window": {
            "start": args.start_date,
            "end": args.end_date,
            "resolved_dates": len(dates),
            "dates": dates,
        },
        "replay_summary": replay_summary,
        "changepoint_compare": cp_compare,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    md_out = Path(args.md_out)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.write_text(_to_markdown(bundle), encoding="utf-8")

    print(json.dumps({"ok": True, "out": str(out), "dates": len(dates)}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
