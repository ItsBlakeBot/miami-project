from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

from analyzer.ai_review_parser import parse_review, store_review
from analyzer.daily_data_builder import DailyDataBuilder
from analyzer.hdp_regime_discovery import HDPShadowRunner
from analyzer.post_settlement_calibrator import (
    PROPOSED_CUSUM_CONFIG_PATH,
    PROPOSED_SKF_CONFIG_PATH,
    TRAINED_SKF_CANDIDATE_PATH,
    apply_statistical_promotion,
    run_post_settlement_calibration,
)
from analyzer.prompt_builder import (
    build_boa_state_text,
    build_calibration_context,
    build_collection_health_text,
    build_ds3m_comparison_text,
    build_ds3m_progress_metrics,
    build_ds3m_regime_discoveries_text,
    build_ds3m_tracker_text,
    build_emos_state_text,
    build_exit_tuner_state_text,
    build_hdp_hmm_results_text,
    build_letkf_diagnostics_text,
    build_letkf_tracker_metrics,
    build_market_density_text,
    build_paper_trade_summary_text,
    build_platt_state_text,
    build_prompt,
    build_recent_context,
    save_prompt,
)
from analyzer.train_skf import SKFTrainer
from collector.store.db import Store
from engine.orchestrator import run_inference_cycle

log = logging.getLogger(__name__)


def default_target_date(now_utc: datetime | None = None, utc_offset_hours: int = -5) -> str:
    now_utc = now_utc or datetime.now(timezone.utc)
    lst_now = now_utc + timedelta(hours=utc_offset_hours)
    return (lst_now.date() - timedelta(days=1)).isoformat()


def prepare_review(target_date: str, db_path: str | Path, *, station: str = "KMIA") -> dict:
    """Prepare phase: run calibration to generate proposals, then build the prompt.

    The calibration must run *before* prompt assembly so the AI can see and
    audit the proposed statistical corrections.
    """
    # 1. Run deterministic statistical calibration → writes proposed configs
    calibration_summary = run_post_settlement_calibration(target_date, db_path, station=station)

    # 2. Build daily data package
    builder = DailyDataBuilder(str(db_path))
    pkg = builder.build(target_date)
    store = Store(str(db_path))
    store.open()
    try:
        recent = store.get_recent_regime_labels(station, n=3)
    finally:
        store.close()

    # 3. Build prompt with calibration proposal context (proposals now exist on disk)
    calibration_context = build_calibration_context(
        live_cusum_config_path=Path("analysis_data/cusum_config.json"),
        proposed_cusum_config_path=PROPOSED_CUSUM_CONFIG_PATH,
        live_skf_config_path=Path("analysis_data/skf_config.json"),
        proposed_skf_config_path=PROPOSED_SKF_CONFIG_PATH,
    )

    letkf_metrics = build_letkf_tracker_metrics()
    ds3m_metrics = build_ds3m_progress_metrics(db_path, station=station)

    prompt = build_prompt(
        pkg.to_prompt_text(),
        target_date,
        recent_context=build_recent_context(recent),
        calibration_context=calibration_context,
        hdp_hmm_results=build_hdp_hmm_results_text(db_path, station=station, target_date=target_date),
        promoted_regimes="_No deterministic promoted-regime ledger yet; use HDP/DS3M recommendations._",
        letkf_diagnostics=build_letkf_diagnostics_text(),
        letkf_sigma_max_weight=letkf_metrics["sigma_max_weight"],
        letkf_min_updates=letkf_metrics["min_updates"],
        emos_state=build_emos_state_text(),
        boa_state=build_boa_state_text(),
        cusum_proposals=json.dumps(calibration_summary, indent=2),
        platt_state=build_platt_state_text(),
        market_density=build_market_density_text(db_path, station=station, target_date=target_date),
        paper_trade_summary=build_paper_trade_summary_text(db_path, station=station),
        ds3m_paper_trading_comparison=build_ds3m_comparison_text(db_path, station=station),
        ds3m_regime_discoveries=(
            f"{build_ds3m_regime_discoveries_text()}\n\n"
            f"{build_ds3m_tracker_text(db_path, station=station)}"
        ),
        collection_health=build_collection_health_text(db_path),
        n_settlements=ds3m_metrics["n_settlements"],
        ds3m_n_cycles=ds3m_metrics["ds3m_n_cycles"],
        letkf_n_updates=letkf_metrics["n_updates"],
        ds3m_regimes_discovered=ds3m_metrics["ds3m_regimes_discovered"],
        ds3m_unnamed_regimes=ds3m_metrics["ds3m_unnamed_regimes"],
        ds3m_vs_prod_crps=ds3m_metrics["ds3m_vs_prod_crps"],
        months_to_ds3m_production=ds3m_metrics["months_to_ds3m_production"],
        exit_tuner_state=build_exit_tuner_state_text(),
    )
    prompt_path = save_prompt(prompt, target_date, Path("reviews/prompts"))
    return {
        "target_date": target_date,
        "prompt_path": str(prompt_path),
        "prompt_chars": len(prompt),
        "calibration": calibration_summary,
    }


def finalize_review(target_date: str, db_path: str | Path, review_input: str | Path, *, station: str = "KMIA") -> dict:
    """Finalize phase: parse AI review, promote configs, train SKF candidate, run HDP shadow.

    Calibration proposals were already generated during prepare.  This step
    consumes the AI audit to decide what gets promoted to live.
    """
    # 1. Parse and store the AI review
    raw = Path(review_input).read_text()
    parsed = parse_review(raw)
    if not parsed.target_date:
        parsed.target_date = target_date
    store_review(parsed, db_path)

    # 2. Promote proposed configs using the AI's statistical audit verdict
    promotion_summary = apply_statistical_promotion(
        parsed.target_date,
        parsed.statistical_audit,
        db_path=db_path,
    )

    # 3. Train SKF into a *candidate* artifact (never overwrites live config)
    trained_ok = False
    try:
        trainer = SKFTrainer(str(db_path))
        trained = trainer.train(min_days_per_regime=2)
        trainer.save_config(trained, str(TRAINED_SKF_CANDIDATE_PATH))
        trained_ok = True
    except Exception as exc:
        log.warning("SKF training failed (non-fatal): %s", exc)

    # 4. HDP shadow — runs after parse/store, never touches live pipeline
    hdp_regimes = None
    try:
        hdp_runner = HDPShadowRunner(db_path, station=station)
        hdp_result = hdp_runner.run(parsed.target_date)
        if hdp_result is not None:
            hdp_runner.store(hdp_result)
            hdp_regimes = hdp_result.n_regimes_discovered
    except Exception as exc:
        log.warning("HDP shadow run failed (non-fatal): %s", exc)

    # 5. DS3M: settle shadow paper trades + daily training + comparison report
    ds3m_summary = {}
    try:
        from engine.ds3m.paper_trader import settle_ds3m_trades, compare_paper_trading
        settle_result = settle_ds3m_trades(db_path, station=station, target_date=parsed.target_date)
        comparison = compare_paper_trading(db_path, station=station, lookback_days=30)
        ds3m_summary = {
            "settled": settle_result.get("settled", 0),
            "comparison": comparison,
        }
        log.info("DS3M post-settlement: %d trades settled", settle_result.get("settled", 0))
    except Exception as exc:
        log.warning("DS3M post-settlement failed (non-fatal): %s", exc)

    try:
        from engine.ds3m.trainer import run_daily_training
        from engine.ds3m.state import DS3MState
        ds3m_state = DS3MState.load()
        training_result = run_daily_training(db_path, ds3m_state, station=station)
        ds3m_state.save()
        ds3m_summary["training"] = training_result
        log.info("DS3M daily training complete: %s", training_result.get("status", "unknown"))
    except Exception as exc:
        log.warning("DS3M daily training failed (non-fatal): %s", exc)

    # 6. Exit policy auto-tuner: replay all trades under different params,
    # find optimal Sharpe-maximizing settings, EMA-blend toward them.
    # Runs for BOTH production and DS3M paper traders.
    exit_tuner_summary = {}
    try:
        from engine.exit_tuner import run_exit_tuning
        prod_tuning = run_exit_tuning(db_path, station=station, table_prefix="")
        exit_tuner_summary["production"] = {
            "status": prod_tuning.get("status"),
            "improvement_cents": prod_tuning.get("improvement_cents"),
            "blended_params": prod_tuning.get("blended_params"),
        }
        log.info("Exit tuner (production): improvement=%.1f¢", prod_tuning.get("improvement_cents", 0))

        ds3m_tuning = run_exit_tuning(db_path, station=station, table_prefix="ds3m_")
        exit_tuner_summary["ds3m"] = {
            "status": ds3m_tuning.get("status"),
            "improvement_cents": ds3m_tuning.get("improvement_cents"),
            "blended_params": ds3m_tuning.get("blended_params"),
        }
        log.info("Exit tuner (DS3M): improvement=%.1f¢", ds3m_tuning.get("improvement_cents", 0))
    except Exception as exc:
        log.warning("Exit tuner failed (non-fatal): %s", exc)

    return {
        "target_date": parsed.target_date,
        "regimes_active": parsed.regimes_active,
        "promotion": promotion_summary,
        "trained_candidate_written": trained_ok,
        "trained_candidate_path": str(TRAINED_SKF_CANDIDATE_PATH),
        "hdp_shadow_regimes": hdp_regimes,
        "ds3m": ds3m_summary,
        "exit_tuner": exit_tuner_summary,
    }


def run_full(target_date: str, db_path: str | Path, review_input: str | Path, *, station: str = "KMIA", run_inference: bool = False) -> dict:
    prepare = prepare_review(target_date, db_path, station=station)
    finalize = finalize_review(target_date, db_path, review_input, station=station)
    result = {"prepare": prepare, "finalize": finalize}
    if run_inference:
        run_inference_cycle(db_path=str(db_path))
        result["inference"] = {"ran": True}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministic daily post-settlement driver")
    sub = parser.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prepare")
    prep.add_argument("--db", required=True)
    prep.add_argument("--date")
    prep.add_argument("--station", default="KMIA")

    fin = sub.add_parser("finalize")
    fin.add_argument("--db", required=True)
    fin.add_argument("--date")
    fin.add_argument("--station", default="KMIA")
    fin.add_argument("--review-input", required=True)

    run = sub.add_parser("run")
    run.add_argument("--db", required=True)
    run.add_argument("--date")
    run.add_argument("--station", default="KMIA")
    run.add_argument("--review-input", required=True)
    run.add_argument("--run-inference", action="store_true")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-5s %(name)s: %(message)s")
    target_date = args.date or default_target_date()

    if args.command == "prepare":
        print(json.dumps(prepare_review(target_date, args.db, station=args.station), indent=2))
    elif args.command == "finalize":
        print(json.dumps(finalize_review(target_date, args.db, args.review_input, station=args.station), indent=2))
    elif args.command == "run":
        print(json.dumps(run_full(target_date, args.db, args.review_input, station=args.station, run_inference=args.run_inference), indent=2))


if __name__ == "__main__":
    main()
