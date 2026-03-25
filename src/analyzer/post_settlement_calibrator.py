"""Post-settlement calibration for CUSUM and SKF parameters.

Runs after each CLI settlement. Uses the now-known ground truth to:
1. Tune CUSUM thresholds (ARL feedback — raise h on false alarms, lower on misses)
2. Tune SKF regime parameters (EMA updates on mu_shift, sigma_scale, A, Q)

Both update JSON config files that the live modules read on next startup.
The AI daily review serves as sanity check — it sees the current parameters
and the auto-tuned proposals, and can override or adjust.

This module is the *statistical* feedback loop. The AI review is the
*reasoning* feedback loop. Both run; the AI's recommendations take
precedence when they conflict.
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration paths
# ---------------------------------------------------------------------------

CUSUM_CONFIG_PATH = Path("analysis_data/cusum_config.json")
SKF_CONFIG_PATH = Path("analysis_data/skf_config.json")
PROPOSED_CUSUM_CONFIG_PATH = Path("analysis_data/cusum_config.proposed.json")
PROPOSED_SKF_CONFIG_PATH = Path("analysis_data/skf_config.proposed.json")
TRAINED_SKF_CANDIDATE_PATH = Path("analysis_data/skf_trained_candidate.json")
CALIBRATION_LOG_PATH = Path("analysis_data/calibration_log.jsonl")
CALIBRATION_PROMOTION_LOG_PATH = Path("analysis_data/calibration_promotion_log.jsonl")

CORRECTION_KEYS = (
    "mu_shift_high",
    "mu_shift_low",
    "sigma_scale_high",
    "sigma_scale_low",
)

STRUCTURAL_KEYS = (
    "A",
    "b",
    "Q",
    "R",
    "self_transition_prob",
    "active_families",
)

# EMA smoothing factor — same as bias_memory.py for consistency.
# 0.15 means ~87% weight on history, ~13% on today's observation.
# Conservative enough to not overreact to single-day outliers.
ALPHA = 0.15

# Maximum per-day adjustment to prevent runaway tuning
MAX_H_ADJUSTMENT = 0.3        # CUSUM h can move ±0.3 per day
MAX_MU_SHIFT_ADJUSTMENT = 0.3  # SKF mu_shift can move ±0.3°F per day
MAX_SIGMA_SCALE_ADJUSTMENT = 0.05  # SKF sigma_scale can move ±0.05 per day


def _load_json_dict(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _clamp_factor(value: float | int | None, default: float) -> float:
    try:
        factor = float(value)
    except (TypeError, ValueError):
        factor = default
    return max(0.0, min(1.0, factor))


def _regime_map(raw: dict) -> dict[str, dict]:
    regimes = raw.get("regimes", raw) if isinstance(raw, dict) else {}
    if isinstance(regimes, dict):
        regime_list = list(regimes.values())
    elif isinstance(regimes, list):
        regime_list = regimes
    else:
        regime_list = []
    result: dict[str, dict] = {}
    for regime in regime_list:
        if isinstance(regime, dict) and regime.get("name"):
            result[str(regime["name"])] = regime
    return result


def _blend_numeric(current, proposed, factor: float):
    if current is None:
        return proposed
    if proposed is None:
        return current
    return round(float(current) + factor * (float(proposed) - float(current)), 4)


def _copy_structural_fields(target: dict, source: dict) -> None:
    for key in STRUCTURAL_KEYS:
        if key in source:
            target[key] = source[key]


def _blend_regime_corrections(live: dict, proposed: dict, factor: float) -> dict:
    out = dict(live)
    _copy_structural_fields(out, proposed if proposed else live)
    for key in CORRECTION_KEYS:
        out[key] = _blend_numeric(live.get(key), proposed.get(key), factor)
    return out


def _normalize_audit(audit: dict | None) -> dict:
    audit = audit or {}
    if "statistical_audit" in audit and isinstance(audit["statistical_audit"], dict):
        audit = audit["statistical_audit"]
    verdict = str(audit.get("overall_verdict") or audit.get("promotion_action") or "hold").strip().lower()
    factor = _clamp_factor(audit.get("dampening_factor"), 0.0)
    if verdict == "approve":
        factor = 1.0
    elif verdict == "approve_with_dampening":
        factor = _clamp_factor(audit.get("dampening_factor"), 0.5)
    elif verdict in {"hold", "reject"}:
        factor = 0.0
    else:
        verdict = "hold"
        factor = 0.0
    return {**audit, "overall_verdict": verdict, "dampening_factor": factor}


# ---------------------------------------------------------------------------
# CUSUM calibration
# ---------------------------------------------------------------------------

@dataclass
class CUSUMCalibrationResult:
    """Result of one day's CUSUM calibration."""
    channel_updates: dict[str, dict]  # channel -> {old_h, new_h, reason}
    false_alarms: list[str]           # channels that fired without a real event
    missed_detections: list[str]      # channels that should have fired but didn't
    good_detections: list[str]        # channels that correctly fired


def calibrate_cusum(
    target_date: str,
    db_path: str | Path,
    station: str = "KMIA",
    input_config_path: Path = CUSUM_CONFIG_PATH,
    output_config_path: Path | None = None,
) -> CUSUMCalibrationResult:
    """Calibrate CUSUM thresholds based on one day's results.

    Compares what the CUSUM detected vs. what the AI review labeled
    as actual regime transitions. Adjusts h per channel.
    """
    from engine.changepoint_detector import CUSUM_H, CUSUM_K

    db_path = Path(db_path)

    # Load current config (or defaults)
    if output_config_path is None:
        output_config_path = input_config_path

    if input_config_path.exists():
        config = _load_json_dict(input_config_path)
        current_h = config.get("h", dict(CUSUM_H))
        current_k = config.get("k", dict(CUSUM_K))
    else:
        current_h = dict(CUSUM_H)
        current_k = dict(CUSUM_K)

    # Get AI-labeled regime transitions for this day
    conn = sqlite3.connect(str(db_path), timeout=10)
    try:
        row = conn.execute(
            """SELECT phase_summary, signal_labels
               FROM regime_labels
               WHERE station = ? AND target_date = ?""",
            (station, target_date),
        ).fetchone()
    finally:
        conn.close()

    if not row:
        log.info("No regime labels for %s — skipping CUSUM calibration", target_date)
        return CUSUMCalibrationResult({}, [], [], [])

    # Parse AI-labeled transitions
    phases = json.loads(row[0]) if row[0] else []
    signal_labels = json.loads(row[1]) if row[1] else []

    # Determine which channels SHOULD have fired (from AI labels)
    ai_detected_channels = set()
    for label in signal_labels:
        signal = label.get("signal", "")
        # Map signal names to CUSUM channels
        if "temp" in signal.lower():
            ai_detected_channels.add("temp_f")
        if "dew" in signal.lower():
            ai_detected_channels.add("dew_f")
        if "pressure" in signal.lower() or "pres" in signal.lower():
            ai_detected_channels.add("pressure_hpa")
        if "wind" in signal.lower():
            ai_detected_channels.add("wind_speed_mph")
            ai_detected_channels.add("wind_dir_sin")
        if "gust" in signal.lower():
            ai_detected_channels.add("wind_gust_mph")
        if "sky" in signal.lower() or "cloud" in signal.lower():
            ai_detected_channels.add("sky_cover_pct")

    # Determine which channels actually fired (from signal_events or
    # by re-running the detector — for now, use a heuristic based on
    # whether AI labeled significant transitions)
    has_regime_transition = len(phases) > 1

    # Update thresholds
    updates = {}
    false_alarms = []
    missed = []
    good = []

    for channel in current_h:
        old_h = current_h[channel]
        new_h = old_h

        if channel in ai_detected_channels and has_regime_transition:
            # Channel was relevant to a real transition → it should fire easily
            # Nudge h down (more sensitive)
            new_h = old_h - ALPHA * MAX_H_ADJUSTMENT
            good.append(channel)
            reason = "AI confirmed relevant transition"
        elif channel not in ai_detected_channels and not has_regime_transition:
            # No transition happened, channel wasn't relevant → leave alone
            reason = "no change needed"
        else:
            # Either false alarm (channel not in AI labels but transition
            # happened on other channels) or quiet day → slight nudge up
            new_h = old_h + ALPHA * MAX_H_ADJUSTMENT * 0.5
            reason = "slight desensitize"

        # Clamp to reasonable range
        new_h = max(2.0, min(8.0, new_h))

        if abs(new_h - old_h) > 0.001:
            updates[channel] = {
                "old_h": round(old_h, 3),
                "new_h": round(new_h, 3),
                "reason": reason,
            }
            current_h[channel] = round(new_h, 3)

    # Save updated config
    config = {"h": current_h, "k": current_k}
    _write_json(output_config_path, config)

    if updates:
        log.info("CUSUM calibration for %s: %d channels updated", target_date, len(updates))

    return CUSUMCalibrationResult(updates, false_alarms, missed, good)


# ---------------------------------------------------------------------------
# SKF calibration
# ---------------------------------------------------------------------------

@dataclass
class SKFCalibrationResult:
    """Result of one day's SKF calibration."""
    regime_updates: dict[str, dict]  # regime -> {param: {old, new, reason}}
    dominant_regime: str
    settlement_error_high: float | None
    settlement_error_low: float | None


def calibrate_skf(
    target_date: str,
    db_path: str | Path,
    station: str = "KMIA",
    input_config_path: Path = SKF_CONFIG_PATH,
    output_config_path: Path | None = None,
) -> SKFCalibrationResult:
    """Calibrate SKF parameters based on one day's settlement results.

    Uses the settlement error (model prediction vs CLI truth) conditioned
    on which regime the SKF thought was active to update:
    - mu_shift_high/low: EMA update toward the observed error
    - sigma_scale_high/low: widen if settlement fell outside predicted CI, tighten if inside
    - Q (process noise): increase if innovation residuals were consistently large
    """
    db_path = Path(db_path)

    # Load current SKF config
    if output_config_path is None:
        output_config_path = input_config_path

    if not input_config_path.exists():
        log.info("No SKF config at %s — skipping calibration", input_config_path)
        return SKFCalibrationResult({}, "unknown", None, None)

    raw = _load_json_dict(input_config_path)

    regimes = raw.get("regimes", raw)
    if isinstance(regimes, dict):
        regimes = list(regimes.values())

    # Get settlement truth
    conn = sqlite3.connect(str(db_path), timeout=10)
    try:
        settlements = {}
        for row in conn.execute(
            """SELECT market_type, actual_value_f
               FROM event_settlements
               WHERE station = ? AND settlement_date = ?""",
            (station, target_date),
        ).fetchall():
            settlements[row[0]] = row[1]

        # Get AI-labeled dominant regime for this day
        regime_row = conn.execute(
            """SELECT regimes_active, phase_summary
               FROM regime_labels
               WHERE station = ? AND target_date = ?""",
            (station, target_date),
        ).fetchone()

        # Get what the consensus predicted
        consensus_rows = conn.execute(
            """SELECT market_type, consensus_forecast_f
               FROM model_consensus
               WHERE station = ? AND forecast_date = ?
               ORDER BY id DESC""",
            (station, target_date),
        ).fetchall()
    finally:
        conn.close()

    if not settlements:
        log.info("No settlements for %s — skipping SKF calibration", target_date)
        return SKFCalibrationResult({}, "unknown", None, None)

    # Determine dominant regime
    dominant = "mixed_uncertain"
    if regime_row:
        regimes_active = json.loads(regime_row[0]) if regime_row[0] else []
        if regimes_active:
            dominant = regimes_active[0]

    # Compute consensus forecast (latest per market type)
    consensus = {}
    seen_types = set()
    for mkt, forecast_f in consensus_rows:
        if mkt not in seen_types and forecast_f is not None:
            consensus[mkt] = forecast_f
            seen_types.add(mkt)

    # Compute settlement errors
    error_high = None
    error_low = None
    if "high" in settlements and "high" in consensus:
        error_high = settlements["high"] - consensus["high"]
    if "low" in settlements and "low" in consensus:
        error_low = settlements["low"] - consensus["low"]

    # Update the matching regime's parameters
    updates = {}
    for regime_dict in regimes:
        name = regime_dict.get("name", "")
        if name != dominant:
            continue

        regime_updates = {}

        # mu_shift updates (EMA toward observed error)
        if error_high is not None:
            old_mu_h = regime_dict.get("mu_shift_high", 0.0)
            raw_delta = ALPHA * (error_high - old_mu_h)
            delta = max(-MAX_MU_SHIFT_ADJUSTMENT, min(MAX_MU_SHIFT_ADJUSTMENT, raw_delta))
            new_mu_h = old_mu_h + delta
            regime_dict["mu_shift_high"] = round(new_mu_h, 4)
            if abs(delta) > 0.001:
                regime_updates["mu_shift_high"] = {
                    "old": round(old_mu_h, 4),
                    "new": round(new_mu_h, 4),
                    "error": round(error_high, 2),
                }

        if error_low is not None:
            old_mu_l = regime_dict.get("mu_shift_low", 0.0)
            raw_delta = ALPHA * (error_low - old_mu_l)
            delta = max(-MAX_MU_SHIFT_ADJUSTMENT, min(MAX_MU_SHIFT_ADJUSTMENT, raw_delta))
            new_mu_l = old_mu_l + delta
            regime_dict["mu_shift_low"] = round(new_mu_l, 4)
            if abs(delta) > 0.001:
                regime_updates["mu_shift_low"] = {
                    "old": round(old_mu_l, 4),
                    "new": round(new_mu_l, 4),
                    "error": round(error_low, 2),
                }

        # sigma_scale updates
        # If the error was larger than expected (outside ~1.5 sigma), widen
        # If the error was smaller than expected, tighten slightly
        for market_type, error_val in [("high", error_high), ("low", error_low)]:
            if error_val is None:
                continue
            key = f"sigma_scale_{market_type}"
            old_sigma = regime_dict.get(key, 1.0)

            # Rough expected sigma from the current scale (~2°F baseline * scale)
            expected_sigma = 2.0 * old_sigma
            if abs(error_val) > 1.5 * expected_sigma:
                # Error outside CI → widen
                adjustment = min(MAX_SIGMA_SCALE_ADJUSTMENT, ALPHA * 0.1)
                new_sigma = old_sigma + adjustment
            elif abs(error_val) < 0.5 * expected_sigma:
                # Error well inside CI → tighten slightly
                adjustment = min(MAX_SIGMA_SCALE_ADJUSTMENT * 0.5, ALPHA * 0.05)
                new_sigma = old_sigma - adjustment
            else:
                new_sigma = old_sigma

            # Clamp to reasonable range
            new_sigma = max(0.5, min(2.0, new_sigma))
            regime_dict[key] = round(new_sigma, 4)

            if abs(new_sigma - old_sigma) > 0.001:
                regime_updates[key] = {
                    "old": round(old_sigma, 4),
                    "new": round(new_sigma, 4),
                    "error": round(error_val, 2),
                }

        if regime_updates:
            updates[name] = regime_updates

    # Save updated config
    if updates:
        _write_json(output_config_path, {"regimes": regimes})
        log.info("SKF calibration for %s: updated regime '%s'", target_date, dominant)

    return SKFCalibrationResult(
        regime_updates=updates,
        dominant_regime=dominant,
        settlement_error_high=error_high,
        settlement_error_low=error_low,
    )


# ---------------------------------------------------------------------------
# Combined calibration
# ---------------------------------------------------------------------------

def run_post_settlement_calibration(
    target_date: str,
    db_path: str | Path,
    station: str = "KMIA",
) -> dict:
    """Run both CUSUM and SKF calibration for a completed day.

    Writes deterministic *proposals* only. Live configs are changed later by
    apply_statistical_promotion() after the AI audit is parsed.
    """
    db_path = Path(db_path)

    cusum_result = calibrate_cusum(
        target_date,
        db_path,
        station,
        input_config_path=CUSUM_CONFIG_PATH,
        output_config_path=PROPOSED_CUSUM_CONFIG_PATH,
    )
    skf_result = calibrate_skf(
        target_date,
        db_path,
        station,
        input_config_path=SKF_CONFIG_PATH,
        output_config_path=PROPOSED_SKF_CONFIG_PATH,
    )

    # LETKF autotuning: compare analysis to settlement, adjust R/alpha
    letkf_tune_summary: dict = {}
    try:
        from engine.letkf import LETKFState, LETKFTuneState, SE_FLORIDA_CLUSTER, autotune_letkf

        # Get settlement values for LETKF comparison
        conn = sqlite3.connect(str(db_path), timeout=10)
        try:
            settle_rows = conn.execute(
                """SELECT market_type, actual_value_f
                   FROM event_settlements
                   WHERE station = ? AND settlement_date = ?""",
                (station, target_date),
            ).fetchall()
            settlements_for_letkf = {r[0]: r[1] for r in settle_rows}
        finally:
            conn.close()

        observed_high = settlements_for_letkf.get("high")
        observed_low = settlements_for_letkf.get("low")

        if observed_high is not None or observed_low is not None:
            # Load tune state (no live LETKFState available post-settlement,
            # so we create a minimal one from persisted diagnostics).
            # The autotune function will record innovations and adjust R.
            tune_state = LETKFTuneState.load()
            # Create a fresh LETKFState to hold R estimates from tune state
            letkf_state = LETKFState(cluster=SE_FLORIDA_CLUSTER)
            # Restore R estimates if any previous tune state exists
            if tune_state.initial_r_estimates:
                letkf_state.r_estimate = dict(tune_state.initial_r_estimates)
            # Initialize ensemble to settlement values for innovation calculation
            forecast_means = {
                s.code: (observed_high or observed_low or 75.0)
                for s in SE_FLORIDA_CLUSTER.stations
            }
            letkf_state.initialize_from_forecasts(forecast_means, forecast_spread=1.0)

            letkf_tune_summary = autotune_letkf(
                letkf_state,
                observed_high_f=observed_high,
                observed_low_f=observed_low,
                tune_state=tune_state,
            )
            log.info("LETKF autotune for %s: %s", target_date, letkf_tune_summary.get("status", "unknown"))
    except Exception as exc:
        log.warning("LETKF autotune failed (non-fatal): %s", exc)
        letkf_tune_summary = {"status": "error", "error": str(exc)}

    summary = {
        "target_date": target_date,
        "station": station,
        "cusum": {
            "channels_updated": len(cusum_result.channel_updates),
            "updates": cusum_result.channel_updates,
        },
        "skf": {
            "dominant_regime": skf_result.dominant_regime,
            "error_high": skf_result.settlement_error_high,
            "error_low": skf_result.settlement_error_low,
            "updates": skf_result.regime_updates,
        },
        "letkf": letkf_tune_summary,
    }

    # Append to calibration log
    log_path = CALIBRATION_LOG_PATH
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps(summary) + "\n")

    log.info(
        "Post-settlement calibration for %s: CUSUM=%d updates, SKF regime=%s",
        target_date,
        len(cusum_result.channel_updates),
        skf_result.dominant_regime,
    )

    return summary


def apply_statistical_promotion(
    target_date: str,
    audit: dict | None,
    *,
    db_path: str | Path | None = None,
    live_cusum_config_path: Path = CUSUM_CONFIG_PATH,
    proposed_cusum_config_path: Path = PROPOSED_CUSUM_CONFIG_PATH,
    live_skf_config_path: Path = SKF_CONFIG_PATH,
    proposed_skf_config_path: Path = PROPOSED_SKF_CONFIG_PATH,
    trained_skf_candidate_path: Path = TRAINED_SKF_CANDIDATE_PATH,
    promotion_log_path: Path = CALIBRATION_PROMOTION_LOG_PATH,
) -> dict:
    """Apply AI-audited promotion from proposed configs into live configs.

    Structural SKF parameters come from the latest trained candidate when present.
    Correction terms remain owned by the audited promotion flow.
    """
    norm_audit = _normalize_audit(audit)
    verdict = norm_audit["overall_verdict"]
    factor = norm_audit["dampening_factor"]

    live_cusum = _load_json_dict(live_cusum_config_path)
    proposed_cusum = _load_json_dict(proposed_cusum_config_path)
    live_skf = _load_json_dict(live_skf_config_path)
    proposed_skf = _load_json_dict(proposed_skf_config_path)
    trained_skf = _load_json_dict(trained_skf_candidate_path)

    promoted_cusum = json.loads(json.dumps(live_cusum or {}))
    if proposed_cusum:
        promoted_cusum.setdefault("k", proposed_cusum.get("k", live_cusum.get("k", {})))
        promoted_cusum["h"] = {}
        channels = sorted(set((live_cusum.get("h", {}) if live_cusum else {})) | set(proposed_cusum.get("h", {})))
        for channel in channels:
            promoted_cusum["h"][channel] = _blend_numeric(
                (live_cusum.get("h", {}) if live_cusum else {}).get(channel),
                proposed_cusum.get("h", {}).get(channel),
                factor,
            )

    live_map = _regime_map(live_skf)
    proposed_map = _regime_map(proposed_skf)
    trained_map = _regime_map(trained_skf)
    promoted_regimes: list[dict] = []
    for name in sorted(set(live_map) | set(proposed_map) | set(trained_map)):
        base = dict(trained_map.get(name) or live_map.get(name) or proposed_map.get(name) or {"name": name})
        live_regime = dict(live_map.get(name) or base)
        proposed_regime = dict(proposed_map.get(name) or live_regime)
        merged = _blend_regime_corrections(live_regime, proposed_regime, factor)
        _copy_structural_fields(merged, base)
        merged.setdefault("name", name)
        promoted_regimes.append(merged)

    promoted_skf = {"regimes": promoted_regimes} if promoted_regimes else live_skf

    should_apply = verdict in {"approve", "approve_with_dampening"}

    # "hold" and "reject" must not mutate live configs; those verdicts explicitly
    # defer or deny promotion until more evidence arrives.
    if should_apply:
        if promoted_cusum:
            _write_json(live_cusum_config_path, promoted_cusum)
        if promoted_skf:
            _write_json(live_skf_config_path, promoted_skf)

    summary = {
        "target_date": target_date,
        "audit": norm_audit,
        "applied": should_apply,
        "live_cusum_updated": should_apply and bool(promoted_cusum),
        "live_skf_updated": should_apply and bool(promoted_regimes),
        "trained_candidate_used": bool(trained_skf),
    }
    promotion_log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(promotion_log_path, "a") as f:
        f.write(json.dumps(summary) + "\n")
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run post-settlement calibration for CUSUM and SKF"
    )
    parser.add_argument("--db", required=True, help="Path to collector DB")
    parser.add_argument("--date", help="Target date (YYYY-MM-DD). Defaults to yesterday.")
    parser.add_argument("--station", default="KMIA")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
    )

    if args.date:
        target_date = args.date
    else:
        target_date = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")

    result = run_post_settlement_calibration(target_date, args.db, args.station)

    print(f"Calibration for {target_date}:")
    print(f"  CUSUM: {result['cusum']['channels_updated']} channels updated")
    if result["skf"]["updates"]:
        for regime, params in result["skf"]["updates"].items():
            for param, vals in params.items():
                print(f"  SKF {regime}.{param}: {vals['old']} → {vals['new']} (error={vals['error']})")
    else:
        print("  SKF: no updates")


if __name__ == "__main__":
    main()
