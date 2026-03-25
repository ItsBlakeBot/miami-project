"""Rolling source-trust refresh runner.

Phase mapping:
- T1.4 adaptive source trust multipliers

Refreshes the source-trust backfill artifact for a rolling window and updates
bounded family multipliers via the persisted source-trust state file.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from analyzer.source_trust_backfill import _write_markdown, run_backfill
from engine.replay_context import parse_utc
from engine.source_trust import SourceTrustConfig, load_source_trust_priors


def run_refresh(
    db_path: str | Path,
    *,
    station: str = "KMIA",
    lookback_days: int = 45,
    min_days: int = 15,
    as_of_utc: datetime | None = None,
    summary_out: str | Path = "analysis_data/source_trust_backfill.json",
    summary_md_out: str | Path = "analysis_data/source_trust_backfill.md",
    state_path: str | None = "analysis_data/source_trust_state.json",
    min_family_samples: int = 80,
    clip_low: float = 0.7,
    clip_high: float = 1.3,
    max_step_per_refresh: float = 0.08,
) -> dict:
    summary = run_backfill(
        db_path,
        station=station,
        lookback_days=max(1, int(lookback_days)),
        min_days=max(1, int(min_days)),
        as_of_utc=as_of_utc,
    )

    out_json = Path(summary_out)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    _write_markdown(summary, Path(summary_md_out))

    priors = load_source_trust_priors(
        out_json,
        cfg=SourceTrustConfig(
            min_family_samples=int(min_family_samples),
            clip_low=float(clip_low),
            clip_high=float(clip_high),
            max_step_per_refresh=float(max_step_per_refresh),
            state_path=state_path,
        ),
    )

    result = {
        "station": station,
        "summary_path": str(out_json),
        "summary_md_path": str(summary_md_out),
        "state_path": state_path,
        "covered_days": summary.get("covered_days"),
        "sufficient_days": summary.get("sufficient_days"),
        "global_mae": priors.global_mae,
        "family_multipliers": priors.family_multipliers,
        "source": priors.source,
    }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Refresh rolling source-trust priors and persisted multiplier state")
    parser.add_argument("--db", default="miami_collector.db")
    parser.add_argument("--station", default="KMIA")
    parser.add_argument("--lookback-days", type=int, default=45)
    parser.add_argument("--min-days", type=int, default=15)
    parser.add_argument("--as-of-utc")
    parser.add_argument("--summary-out", default="analysis_data/source_trust_backfill.json")
    parser.add_argument("--summary-md-out", default="analysis_data/source_trust_backfill.md")
    parser.add_argument("--state-path", default="analysis_data/source_trust_state.json")
    parser.add_argument("--min-family-samples", type=int, default=80)
    parser.add_argument("--clip-low", type=float, default=0.7)
    parser.add_argument("--clip-high", type=float, default=1.3)
    parser.add_argument("--max-step-per-refresh", type=float, default=0.08)
    parser.add_argument("--out", default="analysis_data/source_trust_refresh.json")
    args = parser.parse_args(argv)

    as_of = parse_utc(args.as_of_utc) if args.as_of_utc else None
    result = run_refresh(
        args.db,
        station=args.station,
        lookback_days=args.lookback_days,
        min_days=args.min_days,
        as_of_utc=as_of,
        summary_out=args.summary_out,
        summary_md_out=args.summary_md_out,
        state_path=args.state_path,
        min_family_samples=args.min_family_samples,
        clip_low=args.clip_low,
        clip_high=args.clip_high,
        max_step_per_refresh=args.max_step_per_refresh,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"ok": True, "out": str(out), "covered_days": result["covered_days"]}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
