"""Model scorer + MAE-weighted consensus — single module for all model evaluation.

Standalone module that:
1. Reads raw model forecasts from both sources (Wethr SSE + Open-Meteo)
2. Deduplicates models that appear in both sources (canonical mapping)
3. Groups ensemble members into family means
4. Scores each canonical model against NWS CLI settlement ONLY (no DSM)
5. Computes MAE and bias separately for HIGH and LOW
6. Produces MAE-weighted consensus forecasts (bias stored for diagnostics only)
7. Time-decay weighting: fresh model runs weighted higher (0.5^(age/6h))

Replaces both the old scoring.py (daily scoring) and bias_adjust.py (consensus).

Run standalone:
    cd miami-project/src
    python3 -m collector.model_scorer [--date YYYY-MM-DD] [--window 15] [--db path]
    python3 -m collector.model_scorer --consensus  # also show MAE-weighted consensus
"""

from __future__ import annotations

import argparse
import logging
import math
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCORING_WINDOW_DAYS = 15
MIN_SAMPLES = 3           # settled days needed for bias/MAE to be meaningful
MAE_FLOOR = 0.5           # minimum MAE to prevent extreme weights
UNSCORED_MAE = 4.0        # assumed MAE for models with no track record
WEIGHT_POWER = 2           # 1/MAE^power — amplifies skill differences
RUN_AGE_HALFLIFE_HOURS = 6 # time-decay: weight halves every N hours

# ---------------------------------------------------------------------------
# Model deduplication
#
# When a model appears in both Wethr and Open-Meteo, prefer Open-Meteo
# (explicit forecast_high_f / forecast_low_f vs JSON extraction).
# Map: (db_model_name, db_source) → canonical_name | None (skip)
# ---------------------------------------------------------------------------

MODEL_CANONICAL: dict[tuple[str, str], str | None] = {
    # --- Open-Meteo deterministic (preferred for cross-source dupes) ---
    ("ECMWF-IFS", "openmeteo"):     "ECMWF-IFS",
    ("ECMWF-AIFS", "openmeteo"):    "ECMWF-AIFS",
    ("GFS-Global", "openmeteo"):    "GFS",
    ("GFS-HRRR", "openmeteo"):      "HRRR",
    ("GEM-Global", "openmeteo"):    "GEM",
    ("JMA-GSM", "openmeteo"):       "JMA",
    ("UKMO-Global", "openmeteo"):   "UKMO",
    ("ICON-Global", "openmeteo"):   "ICON",
    ("GFS-GraphCast", "openmeteo"): "GFS-GraphCast",
    ("KNMI-Harmonie", "openmeteo"): "KNMI-Harmonie",
    ("MetNo-Nordic", "openmeteo"):  "MetNo-Nordic",

    # --- Wethr-only models (no OpenMeteo equivalent) ---
    ("NBM", "wethr"):       "NBM",
    ("NAM", "wethr"):       "NAM",
    ("NAM4KM", "wethr"):    "NAM-4km",
    ("ARPEGE", "wethr"):    "ARPEGE",
    ("RAP", "wethr"):       "RAP",
    ("LAV-MOS", "wethr"):   "LAV-MOS",
    ("GFS-MOS", "wethr"):   "GFS-MOS",
    ("NAM-MOS", "wethr"):   "NAM-MOS",
    ("NBS-MOS", "wethr"):   "NBS-MOS",

    # --- Wethr duplicates of OpenMeteo → SKIP ---
    ("ECMWF-IFS", "wethr"): None,
    ("GFS", "wethr"):       None,
    ("HRRR", "wethr"):      None,
    ("GEM-GDPS", "wethr"):  None,
    ("JMA", "wethr"):       None,
    ("UKMO", "wethr"):      None,
    ("ICON", "wethr"):      None,
    ("GEFS", "wethr"):      None,
}

# Ensemble family prefixes → canonical group name
# NOTE: Ensemble means are EXCLUDED from consensus scoring. With 19+
# deterministic models, the ensemble mean just double-counts the parent
# (e.g. ECMWF-Ensemble ≈ ECMWF-IFS, GFS-Ensemble ≈ GFS-Global).
# Ensemble spread is still used for uncertainty (sigma) in signals.py.
# To re-enable, uncomment the values below:
ENSEMBLE_FAMILIES: dict[str, str] = {
    # "ECMWF-IFS-Ensemble": "ECMWF-Ensemble",
    # "GFS-Ensemble":       "GFS-Ensemble",
}

# Ensemble member prefixes to SKIP entirely from consensus (not grouped, just dropped)
ENSEMBLE_SKIP_PREFIXES = [
    "ECMWF-IFS-Ensemble-m",
    "GFS-Ensemble-m",
]

# Reverse lookup: canonical → preferred DB (model, source) for forecast queries
# Used by bias adjustment to map canonical scores back to DB model names
CANONICAL_TO_DB: dict[str, tuple[str, str]] = {
    v: k for k, v in MODEL_CANONICAL.items() if v is not None
}


# ---------------------------------------------------------------------------
# Data structures — Scoring
# ---------------------------------------------------------------------------

@dataclass
class ModelScore:
    """Scoring result for one canonical model, one market_type."""
    canonical_name: str
    market_type: str          # "high" or "low"
    mae: float
    bias: float               # positive = warm bias
    rmse: float
    sample_count: int
    errors: list[float] = field(default_factory=list)
    db_models: list[str] = field(default_factory=list)
    source: str = ""


@dataclass
class ScoringReport:
    """Full scoring report for one station."""
    station: str
    score_date: str
    window_days: int
    settlement_dates: list[str] = field(default_factory=list)
    high_scores: list[ModelScore] = field(default_factory=list)
    low_scores: list[ModelScore] = field(default_factory=list)

    def get_score(self, canonical_name: str, market_type: str) -> ModelScore | None:
        """Look up score by canonical name and market type."""
        scores = self.high_scores if market_type == "high" else self.low_scores
        for s in scores:
            if s.canonical_name == canonical_name:
                return s
        return None


# ---------------------------------------------------------------------------
# Data structures — MAE-weighted consensus
# ---------------------------------------------------------------------------

@dataclass
class ModelAdjustment:
    """One model run's contribution to the consensus."""
    model: str               # canonical name
    source: str
    run_time: str | None
    run_age_hours: float | None
    raw_forecast_f: float     # the value used in consensus (no bias applied)
    bias: float               # diagnostic only — stored for signal evaluator
    mae: float
    skill_weight: float
    decay_factor: float
    final_weight: float
    sample_count: int


@dataclass
class ConsensusResult:
    """MAE-weighted consensus for one market_type."""
    station: str
    forecast_date: str
    market_type: str
    consensus_f: float
    consensus_std_f: float
    n_models: int
    window_days: int
    models: list[ModelAdjustment] = field(default_factory=list)
    run_time_utc: str = ""


# ═══════════════════════════════════════════════════════════════════════════
# PART 1: MODEL SCORING (CLI-only ground truth)
# ═══════════════════════════════════════════════════════════════════════════

def score_models(
    db_path: str,
    station: str = "KMIA",
    target_date: str | None = None,
    window_days: int = SCORING_WINDOW_DAYS,
) -> ScoringReport:
    """Score all models against CLI settlement for HIGH and LOW.

    CLI (NWS Climate Summary) is the ONLY accepted ground truth.
    DSM is never used for scoring.
    """
    conn = sqlite3.connect(db_path, timeout=10)
    try:
        if target_date is None:
            now_local = datetime.now(timezone.utc) + timedelta(hours=-5)
            target_date = now_local.strftime("%Y-%m-%d")

        report = ScoringReport(
            station=station, score_date=target_date, window_days=window_days,
        )

        start_dt = datetime.strptime(target_date, "%Y-%m-%d") - timedelta(days=window_days)
        start_date = start_dt.strftime("%Y-%m-%d")

        for market_type in ("high", "low"):
            settlements = _get_cli_settlements(conn, station, market_type, start_date, target_date)
            if not settlements:
                log.info("No CLI settlements for %s %s in [%s, %s]",
                         station, market_type, start_date, target_date)
                continue

            if market_type == "high":
                report.settlement_dates = sorted(settlements.keys())

            scores = _score_all_models(conn, station, market_type, settlements)
            if market_type == "high":
                report.high_scores = sorted(scores, key=lambda s: s.mae)
            else:
                report.low_scores = sorted(scores, key=lambda s: s.mae)

        return report
    finally:
        conn.close()


def _get_cli_settlements(
    conn: sqlite3.Connection,
    station: str,
    market_type: str,
    start_date: str,
    end_date: str,
) -> dict[str, float]:
    """Get CLI-only settlements. Returns {date: actual_value_f}.

    CLI (NWS Climate Summary) is the ONLY accepted source.
    DSM is excluded entirely — it can be premature/inaccurate.
    """
    rows = conn.execute(
        """SELECT settlement_date, actual_value_f
           FROM event_settlements
           WHERE station = ? AND market_type = ?
             AND settlement_date BETWEEN ? AND ?
             AND actual_value_f IS NOT NULL
             AND settlement_source = 'cli'""",
        (station, market_type, start_date, end_date),
    ).fetchall()

    return {date_str: value for date_str, value in rows}


def _score_all_models(
    conn: sqlite3.Connection,
    station: str,
    market_type: str,
    settlements: dict[str, float],
) -> list[ModelScore]:
    """Score every canonical model against CLI settlements."""
    # {canonical_name: {date: (forecast_f, [db_models], source)}}
    model_forecasts: dict[str, dict[str, tuple[float, list[str], str]]] = {}

    fcst_col = "forecast_high_f" if market_type == "high" else "forecast_low_f"

    # --- Pass 1: Models with explicit forecast_high_f / forecast_low_f ---
    for date_str in settlements:
        rows = conn.execute(
            f"""SELECT model, source, {fcst_col}
                FROM model_forecasts
                WHERE station = ? AND forecast_date = ?
                  AND {fcst_col} IS NOT NULL AND valid_time IS NULL
                GROUP BY model, source
                HAVING MAX(fetch_time_utc)""",
            (station, date_str),
        ).fetchall()

        for model, source, val in rows:
            if val is None:
                continue
            canonical = _resolve_canonical(model, source)
            if canonical is None:
                continue

            if canonical not in model_forecasts:
                model_forecasts[canonical] = {}
            if date_str not in model_forecasts[canonical]:
                model_forecasts[canonical][date_str] = (float(val), [model], source)

    # --- Pass 2: Wethr models with hourly JSON (derive high/low) ---
    agg_fn = "MAX" if market_type == "high" else "MIN"

    for date_str in settlements:
        wethr_models = conn.execute(
            """SELECT DISTINCT model FROM model_forecasts
               WHERE station = ? AND source = 'wethr'
                 AND forecast_date = ? AND valid_time IS NOT NULL
                 AND json_extract(source_record_json, '$.temperature_f') IS NOT NULL""",
            (station, date_str),
        ).fetchall()

        for (model,) in wethr_models:
            canonical = _resolve_canonical(model, "wethr")
            if canonical is None:
                continue
            if canonical in model_forecasts and date_str in model_forecasts[canonical]:
                continue

            row = conn.execute(
                f"""SELECT {agg_fn}(CAST(temp_f AS REAL)) FROM (
                        SELECT json_extract(source_record_json, '$.temperature_f') as temp_f
                        FROM model_forecasts
                        WHERE station = ? AND model = ? AND forecast_date = ?
                          AND source = 'wethr' AND valid_time IS NOT NULL
                          AND json_extract(source_record_json, '$.temperature_f') IS NOT NULL
                        GROUP BY valid_time
                        HAVING MAX(fetch_time_utc)
                    )""",
                (station, model, date_str),
            ).fetchone()

            if row and row[0] is not None:
                if canonical not in model_forecasts:
                    model_forecasts[canonical] = {}
                model_forecasts[canonical][date_str] = (float(row[0]), [model], "wethr")

    # --- Pass 3: Ensemble members → group means ---
    ensemble_raw: dict[str, dict[str, list[float]]] = {}

    for date_str in settlements:
        rows = conn.execute(
            f"""SELECT model, source, {fcst_col}
                FROM model_forecasts
                WHERE station = ? AND forecast_date = ?
                  AND {fcst_col} IS NOT NULL AND valid_time IS NULL
                GROUP BY model, source
                HAVING MAX(fetch_time_utc)""",
            (station, date_str),
        ).fetchall()

        for model, source, val in rows:
            if val is None:
                continue
            group = _resolve_ensemble_group(model)
            if group is None:
                continue
            if group not in ensemble_raw:
                ensemble_raw[group] = {}
            if date_str not in ensemble_raw[group]:
                ensemble_raw[group][date_str] = []
            ensemble_raw[group][date_str].append(float(val))

    for group, dates in ensemble_raw.items():
        if group not in model_forecasts:
            model_forecasts[group] = {}
        for date_str, vals in dates.items():
            if date_str not in model_forecasts[group]:
                mean_val = sum(vals) / len(vals)
                model_forecasts[group][date_str] = (
                    mean_val, [f"{group}(n={len(vals)})"], "openmeteo"
                )

    # --- Compute scores ---
    scores: list[ModelScore] = []

    for canonical, date_map in model_forecasts.items():
        errors = []
        for date_str, actual in settlements.items():
            if date_str in date_map:
                forecast_f = date_map[date_str][0]
                errors.append(forecast_f - actual)

        if len(errors) < MIN_SAMPLES:
            continue

        mae = sum(abs(e) for e in errors) / len(errors)
        bias = sum(errors) / len(errors)
        rmse = math.sqrt(sum(e ** 2 for e in errors) / len(errors))

        latest_date = max(date_map.keys())
        _, db_models, source = date_map[latest_date]

        scores.append(ModelScore(
            canonical_name=canonical,
            market_type=market_type,
            mae=round(mae, 2),
            bias=round(bias, 2),
            rmse=round(rmse, 2),
            sample_count=len(errors),
            errors=errors,
            db_models=db_models,
            source=source,
        ))

    return scores


# ═══════════════════════════════════════════════════════════════════════════
# PART 2: BIAS-ADJUSTED CONSENSUS
# ═══════════════════════════════════════════════════════════════════════════

def run_bias_adjustment(
    db_path: str,
    station: str,
    forecast_date: str,
    window_days: int = SCORING_WINDOW_DAYS,
) -> tuple[ConsensusResult | None, ConsensusResult | None]:
    """Compute MAE-weighted consensus for HIGH and LOW, save to DB.

    Uses CLI-only model scores for MAE-based skill weighting.
    Returns (high_result, low_result).
    """
    conn = sqlite3.connect(db_path, timeout=10)
    try:
        # Score models first (CLI-only)
        report = score_models(db_path, station, forecast_date, window_days)

        high = _compute_consensus(conn, station, forecast_date, "high", report)
        low = _compute_consensus(conn, station, forecast_date, "low", report)

        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        for result in (high, low):
            if result is not None:
                result.run_time_utc = now_utc
                _save_consensus(conn, result)

        return high, low
    finally:
        conn.close()


def _compute_consensus(
    conn: sqlite3.Connection,
    station: str,
    forecast_date: str,
    market_type: str,
    report: ScoringReport,
) -> ConsensusResult | None:
    """Compute MAE-weighted consensus for one market_type (bias diagnostic only)."""
    now_utc = datetime.now(timezone.utc)

    # Get raw forecasts (with dedup + ensemble grouping)
    raw_forecasts = _get_latest_forecasts(conn, station, forecast_date, market_type)
    if not raw_forecasts:
        log.info("No forecasts for %s %s %s", station, forecast_date, market_type)
        return None

    adjustments: list[ModelAdjustment] = []

    for canonical, source, run_time, forecast_f in raw_forecasts:
        # Look up score from CLI-only scoring
        score = report.get_score(canonical, market_type)
        has_score = score is not None and score.sample_count >= MIN_SAMPLES

        bias = score.bias if has_score else 0.0  # diagnostic only — not applied
        mae = max(score.mae, MAE_FLOOR) if has_score else UNSCORED_MAE
        sample_count = score.sample_count if has_score else 0

        skill_weight = 1.0 / (mae ** WEIGHT_POWER)

        run_age = _compute_run_age(run_time, now_utc)
        decay = _compute_time_decay(run_age) if run_age is not None else 0.5

        adjustments.append(ModelAdjustment(
            model=canonical,
            source=source,
            run_time=run_time,
            run_age_hours=round(run_age, 1) if run_age is not None else None,
            raw_forecast_f=round(forecast_f, 1),
            bias=round(bias, 2),
            mae=round(mae, 2),
            skill_weight=round(skill_weight, 4),
            decay_factor=round(decay, 4),
            final_weight=0.0,
            sample_count=sample_count,
        ))

    if not adjustments:
        return None

    # Normalize weights
    raw_weights = [a.skill_weight * a.decay_factor for a in adjustments]
    total_weight = sum(raw_weights)
    if total_weight < 1e-10:
        return None

    for a, rw in zip(adjustments, raw_weights):
        a.final_weight = round(rw / total_weight, 4)

    # Weighted consensus (raw forecasts, no bias shift)
    consensus = sum(a.final_weight * a.raw_forecast_f for a in adjustments)
    weighted_var = sum(
        a.final_weight * (a.raw_forecast_f - consensus) ** 2 for a in adjustments
    )
    consensus_std = math.sqrt(weighted_var) if weighted_var > 0 else 0.0

    adjustments.sort(key=lambda a: a.final_weight, reverse=True)

    return ConsensusResult(
        station=station,
        forecast_date=forecast_date,
        market_type=market_type,
        consensus_f=round(consensus, 1),
        consensus_std_f=round(consensus_std, 1),
        n_models=len(adjustments),
        window_days=report.window_days if any(
            a.sample_count >= MIN_SAMPLES for a in adjustments
        ) else 0,
        models=adjustments,
    )


def _get_latest_forecasts(
    conn: sqlite3.Connection,
    station: str,
    forecast_date: str,
    market_type: str,
) -> list[tuple[str, str, str | None, float]]:
    """Get latest forecasts, deduped and with ensembles grouped.

    Returns [(canonical_name, source, run_time, forecast_f), ...].
    """
    fcst_col = "forecast_high_f" if market_type == "high" else "forecast_low_f"

    # Get all forecasts — latest fetch per (model, run_time)
    rows = conn.execute(
        f"""SELECT model, source, run_time, {fcst_col}, fetch_time_utc
            FROM model_forecasts
            WHERE station = ? AND forecast_date = ?
              AND valid_time IS NULL
            GROUP BY model, COALESCE(run_time, fetch_time_utc)
            HAVING MAX(id)""",
        (station, forecast_date),
    ).fetchall()

    # Also try wethr models with JSON high/low
    json_key = "high_temperature_f" if market_type == "high" else "low_temperature_f"
    wethr_rows = conn.execute(
        f"""SELECT model, source, run_time,
                   json_extract(source_record_json, '$.{json_key}'),
                   fetch_time_utc
            FROM model_forecasts
            WHERE station = ? AND forecast_date = ? AND source = 'wethr'
              AND valid_time IS NULL AND {fcst_col} IS NULL
              AND source_record_json IS NOT NULL
            GROUP BY model, COALESCE(run_time, fetch_time_utc)
            HAVING MAX(id)""",
        (station, forecast_date),
    ).fetchall()

    # Combine, dedup, resolve canonical names
    seen: set[tuple[str, str | None]] = set()
    deterministic: list[tuple[str, str, str | None, float]] = []
    ensemble_buckets: dict[str, list[tuple[str, str | None, float]]] = {}

    for model, source, run_time, val, fetch_time in list(rows) + list(wethr_rows):
        if val is None:
            continue

        # Skip individual ensemble members entirely — they double-count
        # the deterministic parent model. Spread is captured in signals.py.
        if any(model.startswith(p) for p in ENSEMBLE_SKIP_PREFIXES):
            continue

        # Check ensemble (only active if ENSEMBLE_FAMILIES has entries)
        group = _resolve_ensemble_group(model)
        if group is not None:
            if group not in ensemble_buckets:
                ensemble_buckets[group] = []
            ensemble_buckets[group].append((source, run_time or fetch_time, float(val)))
            continue

        canonical = _resolve_canonical(model, source)
        if canonical is None:
            continue

        key = (canonical, run_time or fetch_time)
        if key in seen:
            continue
        seen.add(key)
        deterministic.append((canonical, source, run_time or fetch_time, float(val)))

    # Add ensemble means — use ONLY the latest run's members
    # Build lookup of deterministic parents for sanity checks (use latest run only)
    det_by_family: dict[str, tuple[str, float]] = {}  # group -> (run_time, value)
    for name, _src, rt, val in deterministic:
        # Map deterministic names to ensemble families for cross-validation
        family = None
        if "GFS" in name and "Ensemble" not in name and "GraphCast" not in name:
            family = "GFS-Ensemble"
        elif "ECMWF" in name and "Ensemble" not in name:
            family = "ECMWF-Ensemble"
        if family is not None:
            existing = det_by_family.get(family)
            if existing is None or (rt and (existing[0] is None or rt > existing[0])):
                det_by_family[family] = (rt, val)

    for group, members in ensemble_buckets.items():
        if not members:
            continue
        # Find the latest run_time across all members in this ensemble
        latest_rt = max((rt for _, rt, _ in members if rt), default=None)
        if latest_rt is not None:
            # Filter to only members from the latest run
            latest_members = [(s, rt, v) for s, rt, v in members if rt == latest_rt]
        else:
            latest_members = members
        if not latest_members:
            continue
        mean_val = sum(v for _, _, v in latest_members) / len(latest_members)

        # Sanity check: if ensemble mean diverges >5°F from its deterministic
        # parent, the ensemble data is likely stale — skip it entirely
        parent_entry = det_by_family.get(group)
        parent_val = parent_entry[1] if parent_entry else None
        if parent_val is not None and abs(mean_val - parent_val) > 5.0:
            log.warning(
                "Suppressing %s: ensemble mean %.1f°F diverges %.1f°F from "
                "deterministic parent %.1f°F (likely stale upstream data)",
                group, mean_val, abs(mean_val - parent_val), parent_val,
            )
            continue

        deterministic.append((group, latest_members[0][0], latest_rt, mean_val))

    return deterministic


# ---------------------------------------------------------------------------
# Consensus persistence
# ---------------------------------------------------------------------------

def _save_consensus(conn: sqlite3.Connection, result: ConsensusResult) -> None:
    """Write consensus to model_consensus table (idempotent)."""
    conn.execute(
        "DELETE FROM model_consensus WHERE station=? AND forecast_date=? AND market_type=?",
        (result.station, result.forecast_date, result.market_type),
    )
    for a in result.models:
        conn.execute(
            """INSERT INTO model_consensus
               (station, forecast_date, market_type, model, source,
                run_time, run_age_hours, raw_forecast_f, bias,
                forecast_f, mae, skill_weight, decay_factor,
                final_weight, sample_count, consensus_forecast_f,
                consensus_std_f, n_models, window_days, run_time_utc)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (result.station, result.forecast_date, result.market_type,
             a.model, a.source, a.run_time, a.run_age_hours,
             a.raw_forecast_f, a.bias, a.raw_forecast_f,  # adjusted=raw (bias not applied)
             a.mae, a.skill_weight, a.decay_factor, a.final_weight,
             a.sample_count, result.consensus_f, result.consensus_std_f,
             result.n_models, result.window_days, result.run_time_utc),
        )
    conn.commit()
    log.info("Saved %d model adjustments for %s %s %s (consensus=%.1f°F)",
             len(result.models), result.station, result.forecast_date,
             result.market_type, result.consensus_f)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _resolve_canonical(model: str, source: str) -> str | None:
    """Map (model, source) to canonical name. Returns None to skip duplicates."""
    if _resolve_ensemble_group(model) is not None:
        return None
    key = (model, source)
    if key in MODEL_CANONICAL:
        return MODEL_CANONICAL[key]
    log.debug("Unmapped model: %s/%s — using raw name", model, source)
    return model


def _resolve_ensemble_group(model: str) -> str | None:
    """If model is an ensemble member, return the group name. Else None."""
    for prefix, group in ENSEMBLE_FAMILIES.items():
        if model.startswith(prefix) and model != prefix:
            return group
    return None


def _compute_run_age(run_time: str | None, now_utc: datetime) -> float | None:
    """Hours between run_time and now."""
    if not run_time:
        return None
    try:
        rt_str = run_time.replace("Z", "+00:00")
        if "+" not in rt_str and "-" not in rt_str[10:]:
            rt_str += "+00:00"
        rt = datetime.fromisoformat(rt_str)
        if rt.tzinfo is None:
            rt = rt.replace(tzinfo=timezone.utc)
        delta = now_utc - rt
        return max(delta.total_seconds() / 3600, 0)
    except (ValueError, TypeError):
        return None


def _compute_time_decay(run_age_hours: float | None) -> float:
    """Exponential decay: 0.5^(age/halflife). Fresh run → 1.0, stale → 0."""
    if run_age_hours is None or run_age_hours < 0:
        return 0.5
    return 0.5 ** (run_age_hours / RUN_AGE_HALFLIFE_HOURS)


# ---------------------------------------------------------------------------
# CLI output
# ---------------------------------------------------------------------------

def format_report(report: ScoringReport) -> str:
    """Pretty-print scoring report."""
    lines = [
        f"═══ MODEL SCORES: {report.station} as of {report.score_date} "
        f"({report.window_days}-day window, {len(report.settlement_dates)} settled days) ═══",
        f"  Settled dates: {', '.join(report.settlement_dates)}",
        "",
    ]

    for market_type, scores in [("HIGH", report.high_scores), ("LOW", report.low_scores)]:
        if not scores:
            lines.append(f"  {market_type}: no scores available")
            lines.append("")
            continue

        lines.append(f"  ── {market_type} ──")
        lines.append(
            f"  {'Model':<22s} {'MAE':>6s} {'Bias':>7s} {'RMSE':>6s} {'N':>3s}  Source"
        )
        lines.append("  " + "─" * 60)

        for s in scores:
            bias_arrow = "→" if abs(s.bias) < 0.5 else ("↑warm" if s.bias > 0 else "↓cool")
            lines.append(
                f"  {s.canonical_name:<22s} {s.mae:6.2f} {s.bias:+7.2f} {s.rmse:6.2f} "
                f"{s.sample_count:3d}  {s.source} {bias_arrow}"
            )

        best = scores[0]
        worst = scores[-1]
        lines.append(f"  Best: {best.canonical_name} (MAE {best.mae:.2f}°F)")
        lines.append(f"  Worst: {worst.canonical_name} (MAE {worst.mae:.2f}°F)")
        lines.append("")

    return "\n".join(lines)


def format_consensus(result: ConsensusResult) -> str:
    """Pretty-print a consensus result."""
    lines = [
        f"=== MAE-Weighted Consensus: {result.forecast_date} "
        f"{result.market_type.upper()} ({result.station}) ===",
    ]

    if result.window_days == 0:
        lines.append("  WARNING: Cold start — equal-weight (< 3 CLI settled days)")
    else:
        lines.append(f"  Window: {result.window_days} days (CLI-only)")

    lines.append(
        f"  Consensus: {result.consensus_f}°F  (std: {result.consensus_std_f}°F, "
        f"{result.n_models} models)"
    )
    lines.append("")
    lines.append(
        f"  {'Model':<25s} {'Src':<10s} {'Fcst':>6s} {'Bias*':>6s} "
        f"{'MAE':>5s} {'Skill':>6s} {'Decay':>5s} {'Wt%':>6s} {'Age':>5s} {'N':>3s}"
    )
    lines.append("  " + "─" * 88)

    for a in result.models:
        age_str = f"{a.run_age_hours:.0f}h" if a.run_age_hours is not None else "?"
        lines.append(
            f"  {a.model:<25s} {a.source:<10s} {a.raw_forecast_f:6.1f} "
            f"{a.bias:+6.2f} "
            f"{a.mae:5.2f} {a.skill_weight:6.4f} {a.decay_factor:5.3f} "
            f"{a.final_weight * 100:5.1f}% {age_str:>5s} {a.sample_count:3d}"
        )

    lines.append("  * Bias is diagnostic only — not applied to consensus.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Score persistence
# ---------------------------------------------------------------------------

def _save_scores_to_db(db_path: str, report: ScoringReport) -> None:
    """Write per-model scores to model_scores table (idempotent per score_date)."""
    conn = sqlite3.connect(db_path, timeout=10)
    try:
        # Clear old scores for this date/station
        conn.execute(
            "DELETE FROM model_scores WHERE station=? AND score_date=?",
            (report.station, report.score_date),
        )
        count = 0
        for scores in (report.high_scores, report.low_scores):
            for s in scores:
                conn.execute(
                    """INSERT INTO model_scores
                       (station, model, score_date, market_type, mae, bias, rmse,
                        sample_count, window_days)
                       VALUES (?,?,?,?,?,?,?,?,?)""",
                    (report.station, s.canonical_name, report.score_date,
                     s.market_type, s.mae, s.bias, s.rmse,
                     s.sample_count, report.window_days),
                )
                count += 1
        conn.commit()
        log.info("Saved %d model scores for %s %s", count, report.station, report.score_date)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Model scorer + MAE-weighted consensus (CLI settlement only)"
    )
    parser.add_argument("--date", type=str, default=None,
                        help="Target date (YYYY-MM-DD). Default: today LST")
    parser.add_argument("--window", type=int, default=SCORING_WINDOW_DAYS)
    parser.add_argument("--db", type=str, default="../../miami_collector.db")
    parser.add_argument("--station", type=str, default="KMIA")
    parser.add_argument("--consensus", action="store_true",
                        help="Also compute and show MAE-weighted consensus")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-5s %(message)s")

    # Scoring
    report = score_models(args.db, args.station, args.date, args.window)
    print(format_report(report))

    # Persist scores to model_scores table
    _save_scores_to_db(args.db, report)

    # Consensus
    if args.consensus:
        print()
        high, low = run_bias_adjustment(args.db, args.station,
                                         args.date or report.score_date, args.window)
        if high:
            print(format_consensus(high))
        print()
        if low:
            print(format_consensus(low))


if __name__ == "__main__":
    main()
