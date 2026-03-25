"""Automated daily recalibration pipeline.

Full pipeline after settlement:
  1. Archive forecast-obs pairs for Analog Ensemble (TA8.5)
  2. Run concept drift detection (T7.3)
  3. If drift → refit EMOS with shorter window (T1.2)
  4. Update BOA weights from settlement (T1.5)
  5. Run P&L regime detection (T7.2)
  6. Regime catalog review: compare today's regime inference against catalog,
     flag unrecognized patterns for HDP-HMM discovery + OpenClaw naming (T5.1-T5.2)
  7. Health monitoring (T7.1)

Called daily after CLI settlement arrives (~04:45 LST).
This is the bot's self-improvement loop — every settlement makes it better.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import date
from pathlib import Path

log = logging.getLogger(__name__)


def run_daily_recalibration(
    db: sqlite3.Connection,
    station: str = "KMIA",
    reference_date: str | None = None,
    emos_state_path: str = "analysis_data/emos_state.json",
    boa_state_path: str = "analysis_data/boa_state.json",
    health_report_path: str = "analysis_data/health_report.json",
    drift_report_path: str = "analysis_data/drift_report.json",
    anen_archive: bool = True,
) -> dict:
    """Run the full daily post-settlement recalibration pipeline.

    Order:
      1. Archive forecast-observation pairs for AnEn (TA8.5)
      2. Run drift detection (T7.3)
      3. If drift detected → refit EMOS with shorter window (T1.2)
      4. Update BOA weights with today's settlement (T1.5)
      5. Run P&L regime detection (T7.2)
      6. Run health monitoring (T7.1)
      7. Save all reports

    Returns:
        Summary dict of what was done.
    """
    if reference_date is None:
        ref_str = date.today().isoformat()
    else:
        ref_str = reference_date

    summary: dict = {"date": ref_str, "station": station, "actions": []}

    # 1. Archive for AnEn
    try:
        from analyzer.anen_archiver import archive_settlement_day
        n_archived = archive_settlement_day(db, station, ref_str)
        summary["anen_archived"] = n_archived
        summary["actions"].append(f"anen: {n_archived} rows archived")
    except Exception as e:
        log.warning("AnEn archiving failed: %s", e)
        summary["actions"].append(f"anen: failed ({e})")

    # 2. Drift detection
    drift_detected = False
    try:
        from analyzer.drift_detector import detect_concept_drift
        drift = detect_concept_drift(db, station, reference_days=30, recent_days=7, reference_date=ref_str)
        Path(drift_report_path).parent.mkdir(parents=True, exist_ok=True)
        Path(drift_report_path).write_text(json.dumps(drift.to_dict(), indent=2))
        drift_detected = drift.recalibration_recommended
        summary["drift_detected"] = drift_detected
        summary["drift_alerts"] = len(drift.alerts)
        summary["actions"].append(f"drift: {len(drift.alerts)} alerts, recal={'yes' if drift_detected else 'no'}")
    except Exception as e:
        log.warning("Drift detection failed: %s", e)
        summary["actions"].append(f"drift: failed ({e})")

    # 3. EMOS recalibration (always refit — use shorter window if drift detected)
    try:
        from engine.emos import fit_emos_from_db
        # If drift detected, use a shorter window (more recent data weighted more)
        lookback = 20 if drift_detected else 40
        min_samples = 8  # accept thin data during early operation
        emos_state = fit_emos_from_db(db, station, lookback_days=lookback, min_samples=min_samples, reference_date=ref_str)
        emos_state.save(emos_state_path)

        high_info = f"a={emos_state.high.a:.2f},b={emos_state.high.b:.2f}" if emos_state.high else "insufficient"
        low_info = f"a={emos_state.low.a:.2f},b={emos_state.low.b:.2f}" if emos_state.low else "insufficient"
        summary["emos_refit"] = True
        summary["emos_lookback_days"] = lookback
        summary["actions"].append(f"emos: refit (lookback={lookback}d) high=[{high_info}] low=[{low_info}]")
    except Exception as e:
        log.warning("EMOS refit failed: %s", e)
        summary["actions"].append(f"emos: failed ({e})")

    # 4. BOA weight update
    try:
        from engine.boa import BOAManager
        boa = BOAManager.load(boa_state_path)
        # Get today's settlement
        settlements = db.execute(
            """SELECT market_type, actual_value_f FROM event_settlements
               WHERE station = ? AND settlement_date = ? AND actual_value_f IS NOT NULL""",
            (station, ref_str),
        ).fetchall()

        for row in settlements:
            mt = row["market_type"]
            obs = row["actual_value_f"]
            # Get model forecasts for BOA update
            fcst_rows = db.execute(
                """SELECT model, source, forecast_high_f, forecast_low_f, run_time
                   FROM model_forecasts
                   WHERE station = ? AND forecast_date = ?
                     AND (forecast_high_f IS NOT NULL OR forecast_low_f IS NOT NULL)
                   ORDER BY id DESC""",
                (station, ref_str),
            ).fetchall()

            seen: dict[tuple, dict] = {}
            for fr in fcst_rows:
                key = (fr["model"], fr["source"] or "unknown")
                if key not in seen:
                    seen[key] = fr

            source_fcsts: dict[str, tuple[float, float]] = {}
            awake: set[str] = set()

            import numpy as np
            all_vals: list[float] = []
            for (model, source), fr in seen.items():
                val = fr["forecast_high_f"] if mt == "high" else fr["forecast_low_f"]
                if val is not None:
                    all_vals.append(float(val))
                    sk = f"{source}:{model}"
                    source_fcsts[sk] = (float(val), 1.5)
                    awake.add(sk)  # all sources with today's forecast are awake

            if source_fcsts and len(all_vals) >= 2:
                spread = float(np.std(all_vals))
                source_fcsts = {k: (mu, max(0.5, spread)) for k, (mu, _) in source_fcsts.items()}
                boa.update(mt, source_fcsts, obs, awake_sources=awake)

        boa.save(boa_state_path)
        summary["boa_updated"] = True
        summary["actions"].append("boa: weights updated from settlement")
    except Exception as e:
        log.warning("BOA update failed: %s", e)
        summary["actions"].append(f"boa: failed ({e})")

    # 5. P&L regime detection
    try:
        from analyzer.pnl_regime_detector import detect_pnl_regime
        pnl = detect_pnl_regime(db, lookback_days=30, reference_date=ref_str)
        summary["pnl_regime"] = pnl.regime
        summary["pnl_size_multiplier"] = pnl.recommended_size_multiplier
        summary["actions"].append(f"pnl: regime={pnl.regime}, size_mult={pnl.recommended_size_multiplier:.2f}")
    except Exception as e:
        log.warning("P&L regime detection failed: %s", e)

    # 6. Regime catalog review — compare today's conditions to live catalog
    # Flags patterns that don't fit known regimes for HDP-HMM discovery + OpenClaw naming
    try:
        regime_review = _review_regime_catalog(db, station, ref_str)
        summary["regime_review"] = regime_review
        if regime_review.get("unrecognized"):
            summary["actions"].append(
                f"regime: UNRECOGNIZED pattern detected — "
                f"confidence={regime_review['max_confidence']:.0%}, "
                f"flagged for HDP-HMM discovery + OpenClaw naming"
            )
            # Write proposal file for OpenClaw cron to pick up
            proposal_path = Path("analysis_data/regime_proposals")
            proposal_path.mkdir(parents=True, exist_ok=True)
            proposal_file = proposal_path / f"{ref_str}.json"
            proposal_file.write_text(json.dumps(regime_review, indent=2))
        else:
            summary["actions"].append(
                f"regime: {regime_review['primary']} ({regime_review['max_confidence']:.0%}) — catalog match OK"
            )
    except Exception as e:
        log.warning("Regime catalog review failed: %s", e)
        summary["actions"].append(f"regime: review failed ({e})")

    # 7. Health monitoring
    try:
        from analyzer.health_monitor import run_health_check, save_health_report
        health = run_health_check(db, station, lookback_days=30, reference_date=ref_str)
        save_health_report(health, health_report_path)
        summary["health_alerts"] = len(health.alerts)
        summary["trading_regime"] = health.trading_regime
        summary["actions"].append(f"health: {len(health.alerts)} alerts, regime={health.trading_regime}")
    except Exception as e:
        log.warning("Health monitoring failed: %s", e)

    return summary


def _review_regime_catalog(
    db: sqlite3.Connection,
    station: str,
    settlement_date: str,
) -> dict:
    """Review whether today's weather matched a known regime in the catalog.

    If no regime has confidence > 0.4, flag as "unrecognized" — this pattern
    may represent a new regime that the HDP-HMM should discover and OpenClaw
    should name.

    The proposal file written to analysis_data/regime_proposals/{date}.json
    is picked up by the OpenClaw scheduled task for AI-assisted naming.
    """
    from engine.regime_catalog import infer_live_regime

    # Get today's atmospheric context at peak heating time (~20:00 UTC / 3PM LST)
    atmos = db.execute(
        """SELECT cape, precipitable_water_mm FROM atmospheric_data
           WHERE station = ? AND valid_time_utc LIKE ? || '%'
             AND cape IS NOT NULL
           ORDER BY valid_time_utc DESC LIMIT 1""",
        (station, settlement_date),
    ).fetchone()

    obs = db.execute(
        """SELECT temperature_f, dew_point_f, wind_heading_deg, sky_cover_pct
           FROM observations
           WHERE station = ? AND timestamp_utc LIKE ? || '%'
             AND temperature_f IS NOT NULL
           ORDER BY timestamp_utc DESC LIMIT 1""",
        (station, settlement_date),
    ).fetchone()

    cape = atmos["cape"] if atmos else None
    pw = atmos["precipitable_water_mm"] if atmos else None
    wind_dir = obs["wind_heading_deg"] if obs else None

    # Get cloud estimate
    cloud_fraction = None
    try:
        cloud_row = db.execute(
            """SELECT sky_cover_code FROM nearby_observations
               WHERE timestamp_utc LIKE ? || '%' AND sky_cover_code IS NOT NULL
               ORDER BY timestamp_utc DESC LIMIT 5""",
            (settlement_date,),
        ).fetchall()
        if cloud_row:
            from engine.cloud_cover import metar_sky_to_fraction
            fracs = [metar_sky_to_fraction(r["sky_cover_code"]) for r in cloud_row]
            valid = [f for f in fracs if f is not None]
            if valid:
                cloud_fraction = sum(valid) / len(valid)
    except Exception:
        pass

    regime = infer_live_regime(
        cape=cape,
        pw_mm=pw,
        wind_dir_deg=wind_dir,
        cloud_fraction=cloud_fraction,
        hour_lst=15.0,  # review at peak heating
    )

    # Self-adjusting unrecognized threshold: adapts to the system's own confidence
    # distribution. Uses mean - 2*std of recent regime confidence scores.
    # This means "unrecognized" = statistically unusual for THIS system, not an
    # arbitrary fixed cutoff. If the system typically runs 60-80% confidence,
    # the threshold might be ~40%. If it's noisier (40-60%), threshold drops to ~25%.
    # Falls back to 0.30 only during cold start (<20 historical confidence scores).
    unrecognized_threshold = 0.30  # cold-start fallback
    try:
        hist_rows = db.execute(
            """SELECT regime_confidence FROM bracket_estimates
               WHERE station = ? AND regime_confidence IS NOT NULL
               ORDER BY timestamp_utc DESC LIMIT 500""",
            (station,),
        ).fetchall()
        if len(hist_rows) >= 20:
            import numpy as np
            hist_conf = np.array([r["regime_confidence"] for r in hist_rows])
            # Threshold = mean - 2*std (captures ~2.3% tail of normal distribution)
            # Clamped to [0.15, 0.60] to prevent degenerate values
            mean_conf = float(np.mean(hist_conf))
            std_conf = float(np.std(hist_conf))
            unrecognized_threshold = max(0.15, min(0.60, mean_conf - 2.0 * std_conf))
    except Exception:
        pass

    review = {
        "settlement_date": settlement_date,
        "station": station,
        "primary": regime.primary.value,
        "max_confidence": regime.confidence,
        "probabilities": regime.probabilities,
        "unrecognized": regime.confidence < unrecognized_threshold,
        "unrecognized_threshold": round(unrecognized_threshold, 3),
        "context": {
            "cape": cape,
            "pw_mm": pw,
            "wind_dir": wind_dir,
            "cloud_fraction": cloud_fraction,
        },
    }

    if review["unrecognized"]:
        review["proposal"] = {
            "action": "investigate",
            "reason": f"No regime matched with confidence > 40% (best: {regime.primary.value} at {regime.confidence:.0%})",
            "recommended_steps": [
                "Run HDP-HMM on recent 30-day window including this day",
                "Check if this pattern appears in the discovered regime candidates",
                "If new regime: OpenClaw names it and proposes catalog addition",
                "Human approval required before catalog update",
            ],
        }

    return review


def main() -> None:
    """Run daily recalibration pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description="Daily post-settlement recalibration")
    parser.add_argument("--db", required=True)
    parser.add_argument("--station", default="KMIA")
    parser.add_argument("--reference-date", default=None)
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row

    summary = run_daily_recalibration(conn, args.station, args.reference_date)
    conn.close()

    print(f"\nDaily recalibration for {summary['station']} ({summary['date']}):")
    for action in summary.get("actions", []):
        print(f"  ✓ {action}")


if __name__ == "__main__":
    main()
