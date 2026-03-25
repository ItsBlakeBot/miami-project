"""End-to-end inference orchestrator.

Current primary path (remaining-only target):
  signal_engine (running extremes + obs)
    → build remaining snapshots from forward valid_time > now
    → baseline_engine weighting (freshness + skill)
    → single regime correction layer (SKF + changepoint stack: threshold/CUSUM/BOCPD)
    → bracket_pricer

Runnable standalone:
    python -m engine.orchestrator --db miami_collector.db --date 2026-03-19

Or importable:
    from engine.orchestrator import run_inference_cycle
    results = run_inference_cycle(db_path, target_date)
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from engine.baseline_engine import BaselineEngine, BaselineEngineConfig
from engine.bracket_pricer import Bracket, bracket_yes_probability, fit_mixture_from_baseline
from engine.changepoint_detector import ChangeDetector
from engine.kalman_regimes import ObsVector, SwitchingKalmanFilter
from engine.cloud_cover import estimate_cloud_fraction
from engine.boa import BOAManager
from engine.emos import EMOSCoefficients, EMOSState
from engine.quantile_combiner import PredictiveDistribution, recalibrate_distribution, shift_distribution, clamp_lower, clamp_upper, scale_sigma
from engine.regime_classifier import RegimeState
from engine.replay_context import parse_utc
from engine.residual_estimator import AdjustedBelief, ResidualAdjustment
from engine.signal_engine import SignalEngine, SignalState
from engine.edge_detector import kalshi_fee_cents
from engine.source_registry import ForecastSourceSnapshot

log = logging.getLogger(__name__)

UTC = timezone.utc

SKF_CONFIG_PATH = Path("analysis_data/skf_config.json")
CUSUM_CONFIG_PATH = Path("analysis_data/cusum_config.json")

# Lazy-initialized sigma climatology singleton
_sigma_climatology: "SigmaClimatology | None" = None


def _get_sigma_climatology() -> "SigmaClimatology":
    global _sigma_climatology
    if _sigma_climatology is None:
        from engine.sigma_climatology import SigmaClimatology
        _sigma_climatology = SigmaClimatology("KMIA")
    return _sigma_climatology


# Model → family mapping (matches what the collector writes)
SOURCE_FAMILY: dict[str, str] = {
    "openmeteo": "openmeteo",
    "wethr": "wethr",
    "nws": "nws",
    "herbie": "herbie",
}


def _resolve_emos_coefficients(
    emos_state: EMOSState,
    market_type: str,
    notes: list[str],
) -> EMOSCoefficients:
    """Return EMOS coefficients with a cold-start identity fallback.

    If fitted coefficients are unavailable, we still activate the EMOS path with
    an identity remaining-move mapping and spread-preserving sigma model. This
    keeps behavior stable on day 1 while allowing automatic improvement once
    recalibration writes fitted coefficients.
    """
    coeff = emos_state.high if market_type == "high" else emos_state.low
    if coeff is not None:
        return coeff

    notes.append(f"{market_type} emos cold-start identity coefficients")
    return EMOSCoefficients(
        a=0.0,
        b=1.0,
        c=0.0,
        d=1.0,
        market_type=market_type,
        n_training_samples=0,
        fit_utc="cold_start_identity",
    )


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MarketQuote:
    ticker: str
    last_price_cents: float | None = None
    best_yes_ask_cents: float | None = None
    best_no_ask_cents: float | None = None
    yes_ask_qty: int = 0
    no_ask_qty: int = 0


@dataclass
class BracketEstimate:
    ticker: str
    market_type: str
    floor_f: float | None
    ceiling_f: float | None
    directional: str | None
    model_probability: float  # YES probability
    mu: float | None
    sigma: float | None
    market_price: float | None  # YES last price in cents
    edge: float | None  # YES edge: model_prob * 100 - market_price
    no_probability: float = 0.0  # 1 - model_probability
    no_market_price: float | None = None  # 100 - market_price
    no_edge: float | None = None  # NO edge: no_prob * 100 - no_market_price
    active_signals: list[str] = field(default_factory=list)
    regime_confidence: float | None = None


@dataclass
class InferenceCycleResult:
    target_date: str
    timestamp_utc: str
    high_belief: AdjustedBelief | None = None
    low_belief: AdjustedBelief | None = None
    high_regime: RegimeState | None = None
    low_regime: RegimeState | None = None
    estimates: list[BracketEstimate] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class InferenceConfig:
    """Tunable thresholds for the inference pipeline.

    Every threshold is configurable — no magic numbers.
    Load from JSON or override programmatically.
    """

    # BOA: minimum settlement updates before BOA overrides source trust (T1.5)
    boa_min_updates: int = 5

    # Regime catalog: minimum confidence to apply sigma/mu conditioning
    regime_confidence_gate: float = 0.3

    # Regime catalog: minimum absolute mu bias to apply (°F)
    regime_mu_bias_min_f: float = 0.1

    # SKF: dormant mode — still runs + logs but corrections NOT applied
    # Set to True to reactivate SKF corrections (not recommended until
    # LETKF + regime catalog have proven superior in replay)
    skf_corrections_enabled: bool = False

    # LETKF: minimum observations for spatial assimilation update
    letkf_min_obs: int = 2

    # LETKF Sage-Husa: forgetting factor for adaptive R estimation
    letkf_sage_husa_b: float = 0.97

    # LETKF: max inflation factor for RTPS
    letkf_max_inflation: float = 2.0

    # LETKF: minimum n_updates before analysis spread influences baseline sigma
    letkf_min_updates_for_sigma: int = 3

    # LETKF: max blending weight for analysis spread on baseline sigma (0.0-1.0)
    letkf_sigma_max_weight: float = 0.4

    # LETKF: spatial divergence threshold (°F) for logging spatial signals
    letkf_spatial_divergence_f: float = 0.5

    # (obs-lock sigma compression removed — replaced by _apply_remaining_move_prior
    # which uses sigma climatology + changepoint awareness instead of raw obs gaps)

    @classmethod
    def load(cls, path: str | Path = "analysis_data/inference_config.json") -> "InferenceConfig":
        p = Path(path)
        if not p.exists():
            return cls()
        try:
            data = json.loads(p.read_text())
            return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        except (json.JSONDecodeError, TypeError):
            return cls()


@dataclass
class InferenceRuntimeState:
    """State carried across inference loop iterations.

    Keeps ChangeDetector/SKF/LETKF state warm so CUSUM, regime probabilities,
    and spatial analysis can accumulate across cycles. Also tracks config file
    mtimes to hot-reload when live configs change.
    """

    change_detector: ChangeDetector | None = None
    skf: SwitchingKalmanFilter | None = None
    active_target_date: str | None = None
    last_cycle_utc: datetime | None = None
    skf_config_mtime: float | None = None
    cusum_config_mtime: float | None = None
    # LETKF state for spatial assimilation (T3.1-T3.3)
    letkf_state: object | None = None  # LETKFState, initialized by runner on startup
    # Tunable inference config — reloaded when file mtime changes
    config: InferenceConfig = field(default_factory=InferenceConfig)
    config_mtime: float | None = None


# ---------------------------------------------------------------------------
# Forecast snapshot builder — reads model_forecasts, wraps in ForecastSourceSnapshot
# ---------------------------------------------------------------------------

def _build_forecast_snapshots(
    db: sqlite3.Connection,
    station: str,
    target_date: str,
    as_of_utc: datetime | None = None,
) -> list[ForecastSourceSnapshot]:
    """Read daily forecasts from DB, dedup to latest per model/source,
    wrap in ForecastSourceSnapshot objects with synthetic quantiles."""
    from statistics import NormalDist

    rows = db.execute(
        """SELECT model, source, forecast_high_f, forecast_low_f, run_time
           FROM model_forecasts
           WHERE station = ? AND forecast_date = ?
             AND (forecast_high_f IS NOT NULL OR forecast_low_f IS NOT NULL)
             AND valid_time IS NULL
           ORDER BY run_time DESC""",
        (station, target_date),
    ).fetchall()

    # Dedup: keep latest run per (model, source)
    seen: set[tuple[str, str]] = set()
    snapshots: list[ForecastSourceSnapshot] = []

    for row in rows:
        model = row["model"]
        source = row["source"]
        key = (model, source)
        if key in seen:
            continue
        seen.add(key)

        family = SOURCE_FAMILY.get(source, source)

        # Parse run_time to datetime
        run_time_str = row["run_time"]
        if run_time_str:
            issued_at = _parse_timestamp(run_time_str)
            if issued_at is None:
                continue
        else:
            issued_at = datetime.now(tz=UTC)

        if as_of_utc is not None and issued_at > as_of_utc:
            continue

        # Generate synthetic quantiles from point forecasts.
        # Default spread: 2.0°F sigma (typical for 24h forecasts in Miami)
        default_sigma = 2.0
        norm = NormalDist(0, 1)
        quantile_grid = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)

        q_high = {}
        if row["forecast_high_f"] is not None:
            mu = row["forecast_high_f"]
            q_high = {q: round(mu + norm.inv_cdf(q) * default_sigma, 3) for q in quantile_grid}

        q_low = {}
        if row["forecast_low_f"] is not None:
            mu = row["forecast_low_f"]
            q_low = {q: round(mu + norm.inv_cdf(q) * default_sigma, 3) for q in quantile_grid}

        snapshots.append(ForecastSourceSnapshot(
            source_name=model,
            family_name=family,
            issued_at_utc=issued_at,
            target_date=target_date,
            forecast_high_f=row["forecast_high_f"],
            forecast_low_f=row["forecast_low_f"],
            quantiles_high=q_high,
            quantiles_low=q_low,
        ))

    return snapshots


def _parse_timestamp(ts: str) -> datetime | None:
    """Parse UTC-ish timestamp strings for replay-safe filtering.

    Returns None for unparseable values so callers can skip rows instead of
    silently treating bad timestamps as 'now'.
    """
    return parse_utc(ts)


def _climate_day_end_utc(target_date: str) -> datetime:
    next_day = datetime.strptime(target_date, "%Y-%m-%d") + timedelta(days=1)
    return datetime(next_day.year, next_day.month, next_day.day, 5, 0, 0, tzinfo=UTC)


def _load_latest_skill_mae(
    db: sqlite3.Connection,
    station: str,
    market_type: str,
) -> dict[str, float]:
    """Latest per-model MAE proxy used as baseline tracking_error input.

    Smaller values imply higher trust in BaselineEngine weighting.
    """
    rows = db.execute(
        """SELECT model, mae, sample_count
           FROM model_scores
           WHERE station = ? AND market_type = ?
           ORDER BY score_date DESC, id DESC""",
        (station, market_type),
    ).fetchall()

    skills: dict[str, float] = {}
    for row in rows:
        model = row["model"]
        if model in skills:
            continue
        mae = row["mae"]
        if mae is None:
            continue
        sample_count = int(row["sample_count"] or 0)
        # Small-sample penalty so sparse score histories don't dominate.
        penalty = 0.0
        if sample_count < 5:
            penalty = 1.25
        elif sample_count < 10:
            penalty = 0.60
        skills[model] = round(max(0.5, float(mae) + penalty), 3)
    return skills


def _build_remaining_move_snapshots(
    db: sqlite3.Connection,
    station: str,
    target_date: str,
    now_utc: datetime,
    running_high_f: float | None,
    running_low_f: float | None,
) -> list[ForecastSourceSnapshot]:
    """Build final-high/low snapshots from *future-only* hourly forecasts.

    Core target alignment:
      R_high = final_high - running_high_now
      R_low  = running_low_now - final_low

    We compute per-model remaining extremes from valid_time > now, then convert
    back to final-high/final-low distributions so downstream bracket pricing can
    stay unchanged.
    """
    from statistics import NormalDist

    if running_high_f is None and running_low_f is None:
        return []

    climate_end = _climate_day_end_utc(target_date)
    skill_high = _load_latest_skill_mae(db, station, "high")
    skill_low = _load_latest_skill_mae(db, station, "low")

    rows = db.execute(
        """SELECT model,
                  source,
                  valid_time,
                  run_time,
                  fetch_time_utc,
                  COALESCE(
                      raw_temperature_f,
                      json_extract(source_record_json, '$.temperature_f')
                  ) AS temp_f
           FROM model_forecasts
           WHERE station = ?
             AND forecast_date = ?
             AND valid_time IS NOT NULL
             AND (
                 raw_temperature_f IS NOT NULL
                 OR json_extract(source_record_json, '$.temperature_f') IS NOT NULL
             )
           ORDER BY id DESC""",
        (station, target_date),
    ).fetchall()

    per_source: dict[tuple[str, str], dict] = {}
    for row in rows:
        valid_time = row["valid_time"]
        if not valid_time:
            continue
        valid_dt = _parse_timestamp(valid_time)
        if valid_dt is None:
            continue
        if valid_dt <= now_utc or valid_dt > climate_end:
            continue

        temp_f = row["temp_f"]
        if temp_f is None:
            continue

        model = row["model"]
        source = row["source"] or "unknown"
        key = (model, source)

        issued_raw = row["fetch_time_utc"] or row["run_time"] or valid_time
        issued_at = _parse_timestamp(issued_raw)
        if issued_at is None:
            continue
        if issued_at > now_utc:
            continue

        entry = per_source.setdefault(
            key,
            {
                "temps": [],
                "seen_valid": set(),
                "issued_at": issued_at,
                "source": source,
                "model": model,
            },
        )
        valid_key = valid_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        if valid_key in entry["seen_valid"]:
            continue
        entry["seen_valid"].add(valid_key)
        entry["temps"].append(float(temp_f))
        if issued_at > entry["issued_at"]:
            entry["issued_at"] = issued_at

    if not per_source:
        return []

    quantiles = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
    std_norm = NormalDist(0.0, 1.0)
    snapshots: list[ForecastSourceSnapshot] = []

    for (model, source), payload in sorted(per_source.items()):
        temps = payload["temps"]
        if not temps:
            continue

        src_max = max(temps)
        src_min = min(temps)
        src_spread = max(0.0, src_max - src_min)

        if running_high_f is not None:
            remaining_high = max(0.0, src_max - running_high_f)
            high_mu = running_high_f + remaining_high
        else:
            high_mu = src_max

        if running_low_f is not None:
            remaining_low = max(0.0, running_low_f - src_min)
            low_mu = running_low_f - remaining_low
        else:
            low_mu = src_min

        # Remaining-move uncertainty proxy from per-source forward spread.
        sigma_base = max(0.55, min(3.75, 0.35 * src_spread + 0.45))
        sigma_high = sigma_base
        sigma_low = sigma_base

        q_high = {
            q: round(high_mu + std_norm.inv_cdf(q) * sigma_high, 3)
            for q in quantiles
        }
        q_low = {
            q: round(low_mu + std_norm.inv_cdf(q) * sigma_low, 3)
            for q in quantiles
        }

        if running_high_f is not None:
            q_high = {q: max(v, running_high_f) for q, v in q_high.items()}
        if running_low_f is not None:
            q_low = {q: min(v, running_low_f) for q, v in q_low.items()}

        snapshots.append(
            ForecastSourceSnapshot(
                source_name=model,
                family_name=SOURCE_FAMILY.get(source, source),
                issued_at_utc=payload["issued_at"],
                target_date=target_date,
                forecast_high_f=round(high_mu, 3),
                forecast_low_f=round(low_mu, 3),
                quantiles_high=q_high,
                quantiles_low=q_low,
                tracking_error_high=skill_high.get(model),
                tracking_error_low=skill_low.get(model),
                metadata={
                    "mode": "remaining_only",
                    "future_points": len(temps),
                    "source": source,
                },
            )
        )

    return snapshots


def _apply_single_regime_layer(
    baseline,
    *,
    market_type: str,
    running_high_f: float | None,
    running_low_f: float | None,
    skf_state,
    cp_state,
    n_training_days: int,
) -> tuple[AdjustedBelief, RegimeState]:
    """One correction layer only: SKF mean/sigma + changepoint volatility widening."""
    dist = baseline.distribution
    delta_mu = 0.0
    sigma_multiplier = 1.0
    tags: list[str] = []

    primary = "mixed_uncertain"
    confidence = 0.35

    if skf_state is not None:
        primary = skf_state.most_likely_regime
        confidence = float(skf_state.regime_confidence)
        if market_type == "high":
            delta_mu += float(skf_state.mu_shift_high)
            sigma_multiplier *= float(skf_state.sigma_scale_high)
        else:
            delta_mu += float(skf_state.mu_shift_low)
            sigma_multiplier *= float(skf_state.sigma_scale_low)
        tags.append(f"skf:{primary}")

    if cp_state is not None and cp_state.fired:
        cp_prob = float(cp_state.changepoint_probability)
        sigma_multiplier *= (1.0 + 0.65 * cp_prob)
        tags.append(f"cusum_cp:{cp_prob:.2f}")
        confidence = max(confidence, min(0.95, 0.40 + 0.45 * cp_prob))

    if abs(delta_mu) > 1e-6:
        dist = shift_distribution(dist, delta_mu, note=f"single_regime_mu_shift={delta_mu:+.2f}")
    if abs(sigma_multiplier - 1.0) > 1e-6:
        dist = scale_sigma(dist, sigma_multiplier, note=f"single_regime_sigma={sigma_multiplier:.3f}")

    # Keep physics hard constraints in the distribution itself.
    if market_type == "high" and running_high_f is not None:
        dist = clamp_lower(dist, running_high_f, note=f"running_high_floor={running_high_f:.1f}°F")
    if market_type == "low" and running_low_f is not None:
        dist = clamp_upper(dist, running_low_f, note=f"running_low_ceiling={running_low_f:.1f}°F")

    regime = RegimeState(
        primary_regime=primary,
        path_class="remaining_only",
        confidence=round(confidence, 3),
        tags=tags,
    )

    if skf_state is not None:
        regime.skf_probabilities = {
            **skf_state.regime_probabilities,
            "_n_training_days": n_training_days,
        }
        regime.skf_mu_shift_high = float(skf_state.mu_shift_high)
        regime.skf_mu_shift_low = float(skf_state.mu_shift_low)
        regime.skf_sigma_scale_high = float(skf_state.sigma_scale_high)
        regime.skf_sigma_scale_low = float(skf_state.sigma_scale_low)
        regime.skf_active_families = dict(skf_state.active_families)

    adjusted = AdjustedBelief(
        market_type=market_type,
        target_date=baseline.target_date,
        baseline=baseline,
        distribution=dist,
        adjustment=ResidualAdjustment(
            delta_mu_f=round(delta_mu, 3),
            sigma_multiplier=round(sigma_multiplier, 3),
            clamps=[],
            notes=tags,
        ),
    )
    return adjusted, regime


# ---------------------------------------------------------------------------
# Bracket loading — reads active_brackets table
# ---------------------------------------------------------------------------

def _load_brackets(
    db: sqlite3.Connection,
    target_date: str,
) -> list[Bracket]:
    """Load brackets from active_brackets table.

    Uses settlement_floor/settlement_ceil for probability calculation — these
    represent the actual settlement range, not the ticker strike values.
    """
    rows = db.execute(
        """SELECT ticker, market_type, floor_strike, cap_strike,
                  settlement_floor, settlement_ceil
           FROM active_brackets
           WHERE target_date = ?
           ORDER BY market_type, floor_strike""",
        (target_date,),
    ).fetchall()

    brackets: list[Bracket] = []
    for row in rows:
        ticker = row["ticker"]
        market_type = row["market_type"]
        floor_strike = row["floor_strike"]
        cap_strike = row["cap_strike"]
        settlement_floor = row["settlement_floor"]
        settlement_ceil = row["settlement_ceil"]

        directional = _infer_directional(ticker, floor_strike, cap_strike, market_type)

        # Use settlement bounds for probability calculation
        if directional == "under":
            brackets.append(Bracket(
                ticker=ticker, market_type=market_type,
                ceiling_f=settlement_ceil, directional="under",
            ))
        elif directional == "over":
            brackets.append(Bracket(
                ticker=ticker, market_type=market_type,
                floor_f=settlement_floor, directional="over",
            ))
        else:
            brackets.append(Bracket(
                ticker=ticker, market_type=market_type,
                floor_f=settlement_floor, ceiling_f=settlement_ceil,
            ))

    return brackets


def _load_brackets_from_market_snapshots(
    db: sqlite3.Connection,
    target_date: str,
    as_of_utc: datetime,
) -> list[Bracket]:
    """Fallback bracket loader for historical replay/backtests.

    Uses market_snapshots metadata as-of a replay timestamp so we can backtest
    days where active_brackets no longer contains the historical contracts.
    """
    as_of = as_of_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = db.execute(
        """SELECT ms.ticker, ms.market_type, ms.floor_strike, ms.cap_strike
           FROM market_snapshots ms
           JOIN (
               SELECT ticker, MAX(id) AS max_id
               FROM market_snapshots
               WHERE forecast_date = ? AND snapshot_time <= ?
               GROUP BY ticker
           ) latest ON latest.max_id = ms.id
           ORDER BY ms.market_type, ms.floor_strike""",
        (target_date, as_of),
    ).fetchall()

    brackets: list[Bracket] = []
    for row in rows:
        ticker = row["ticker"]
        market_type = row["market_type"]
        floor_strike = row["floor_strike"]
        cap_strike = row["cap_strike"]

        settlement_floor, settlement_ceil = _settlement_bounds(
            market_type, floor_strike, cap_strike, ticker
        )
        directional = _infer_directional(ticker, floor_strike, cap_strike, market_type)

        if directional == "under":
            brackets.append(Bracket(
                ticker=ticker,
                market_type=market_type,
                ceiling_f=settlement_ceil,
                directional="under",
            ))
        elif directional == "over":
            brackets.append(Bracket(
                ticker=ticker,
                market_type=market_type,
                floor_f=settlement_floor,
                directional="over",
            ))
        else:
            brackets.append(Bracket(
                ticker=ticker,
                market_type=market_type,
                floor_f=settlement_floor,
                ceiling_f=settlement_ceil,
            ))

    return brackets


def _infer_directional(
    ticker: str, floor_strike: float | None, cap_strike: float | None, market_type: str,
) -> str | None:
    """Infer whether a bracket is directional (under/over) or a range bracket.

    Kalshi convention:
    - B prefix = range bracket, floor is 0.5 below the number
    - T prefix with lower value = "under" threshold
    - T prefix with higher value = "over" threshold

    We detect this from the ticker pattern and settlement_floor/ceil extremes.
    """
    # Check for T prefix (directional) vs B prefix (range bracket)
    parts = ticker.split("-")
    if len(parts) >= 3:
        strike_part = parts[-1]  # e.g., "T73", "B77.5", "T80"
        if strike_part.startswith("T"):
            # Directional — need to figure out under vs over
            # Convention: if cap_strike < floor_strike → under (T with lower value)
            # if cap_strike is None or empty → over (T with higher value)
            if cap_strike is not None and cap_strike < floor_strike:
                return "under"
            if cap_strike is None or cap_strike == 0:
                return "over"
            # Heuristic from the active_brackets data:
            # under: floor=73, cap=72 (cap < floor)
            # over: floor=80, cap=None
            if cap_strike < floor_strike:
                return "under"
            return "over"
    return None


# ---------------------------------------------------------------------------
# Latest market books for edge + sizing
# ---------------------------------------------------------------------------

def _load_market_quotes(
    db: sqlite3.Connection,
    target_date: str,
    as_of_utc: datetime | None = None,
) -> dict[str, MarketQuote]:
    """Get latest per-ticker market snapshot fields used by sizing."""
    if as_of_utc is not None:
        as_of = as_of_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
        rows = db.execute(
            """SELECT ms.ticker,
                      ms.last_price_cents,
                      ms.best_yes_ask_cents,
                      ms.best_no_ask_cents,
                      ms.yes_ask_qty,
                      ms.no_ask_qty
               FROM market_snapshots ms
               JOIN (
                   SELECT ticker, MAX(id) AS max_id
                   FROM market_snapshots
                   WHERE forecast_date = ? AND snapshot_time <= ?
                   GROUP BY ticker
               ) latest ON latest.max_id = ms.id""",
            (target_date, as_of),
        ).fetchall()
    else:
        rows = db.execute(
            """SELECT ms.ticker,
                      ms.last_price_cents,
                      ms.best_yes_ask_cents,
                      ms.best_no_ask_cents,
                      ms.yes_ask_qty,
                      ms.no_ask_qty
               FROM market_snapshots ms
               JOIN (
                   SELECT ticker, MAX(id) AS max_id
                   FROM market_snapshots
                   WHERE forecast_date = ?
                   GROUP BY ticker
               ) latest ON latest.max_id = ms.id""",
            (target_date,),
        ).fetchall()

    quotes: dict[str, MarketQuote] = {}
    for r in rows:
        quotes[r["ticker"]] = MarketQuote(
            ticker=r["ticker"],
            last_price_cents=r["last_price_cents"],
            best_yes_ask_cents=r["best_yes_ask_cents"],
            best_no_ask_cents=r["best_no_ask_cents"],
            yes_ask_qty=int(r["yes_ask_qty"] or 0),
            no_ask_qty=int(r["no_ask_qty"] or 0),
        )
    return quotes


def _mtime_or_none(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except FileNotFoundError:
        return None


_INFERENCE_CONFIG_PATH = Path("analysis_data/inference_config.json")


def _ensure_runtime_models(
    runtime_state: InferenceRuntimeState,
    target_date: str,
    notes: list[str],
) -> tuple[ChangeDetector, SwitchingKalmanFilter]:
    """Ensure runtime detector/SKF exist and are fresh.

    Rebuilds when config files change, and resets state only on climate-day
    rollover (or model reload). Also hot-reloads inference_config.json.
    """
    # Hot-reload InferenceConfig when file changes
    inf_cfg_mtime = _mtime_or_none(_INFERENCE_CONFIG_PATH)
    if inf_cfg_mtime != runtime_state.config_mtime:
        runtime_state.config = InferenceConfig.load(_INFERENCE_CONFIG_PATH)
        runtime_state.config_mtime = inf_cfg_mtime
        notes.append("inference_config_reloaded")

    skf_mtime = _mtime_or_none(SKF_CONFIG_PATH)
    cusum_mtime = _mtime_or_none(CUSUM_CONFIG_PATH)

    needs_rebuild = (
        runtime_state.change_detector is None
        or runtime_state.skf is None
        or runtime_state.skf_config_mtime != skf_mtime
        or runtime_state.cusum_config_mtime != cusum_mtime
    )

    if needs_rebuild:
        runtime_state.change_detector = ChangeDetector()
        runtime_state.change_detector.reset()
        runtime_state.skf = SwitchingKalmanFilter()
        runtime_state.skf.reset(target_date)
        runtime_state.active_target_date = target_date
        runtime_state.skf_config_mtime = skf_mtime
        runtime_state.cusum_config_mtime = cusum_mtime
        notes.append("runtime_models_reloaded")
    elif runtime_state.active_target_date != target_date:
        runtime_state.change_detector.reset()
        runtime_state.skf.reset(target_date)
        runtime_state.active_target_date = target_date
        notes.append(f"runtime_models_reset_for_new_day={target_date}")

    return runtime_state.change_detector, runtime_state.skf


def _apply_remaining_move_prior(
    baseline,
    market_type: str,
    target_date: str,
    hour_lst: int,
    running_high_f: float | None,
    running_low_f: float | None,
    notes: list[str],
    cp_probability: float = 0.0,
):
    """Blend baseline remaining-move belief with weighted climatology prior.

    This is intentionally upstream of regime/SKF modifiers so regime logic acts
    on a physically realistic prior for *remaining* movement.
    """
    mu = baseline.distribution.mu
    if mu is None:
        return baseline

    sigma = baseline.distribution.sigma or 1.0
    model_trust = min(max(baseline.model_trust, 0.0), 1.0)

    clim = _get_sigma_climatology()
    profile = clim.get_remaining_profile(target_date, hour_lst, market_type)
    rem_prior = float(profile.get("mean_remaining", 0.0))
    rem_q90 = float(profile.get("q90_remaining", rem_prior + 1.2816 * sigma))
    p_lock = float(profile.get("p_lock", 0.0))

    if market_type == "high" and running_high_f is not None:
        rem_model = max(0.0, mu - running_high_f)
        # Soft blend: seasonal prior regularizes, but changepoint confidence
        # restores model weight during active regime shifts.
        alpha_base = max(0.25, model_trust * (1.0 - 0.5 * p_lock))
        alpha = min(0.95, alpha_base + 0.5 * cp_probability)
        rem_blend = alpha * rem_model + (1.0 - alpha) * rem_prior
        rem_blend = min(rem_blend, rem_q90)
        mu_target = running_high_f + rem_blend

        # Sigma compression from climatology: when p_lock is high (extreme is
        # likely locked), compress sigma toward the remaining potential.
        # But preserve width if changepoint probability is elevated.
        sigma_clim_remaining = float(profile.get("sigma_remaining", sigma))
        sigma_cap = sigma_clim_remaining * (1.0 + 2.0 * cp_probability)  # widen on regime break
        new_sigma = min(sigma, max(0.25, sigma_cap))  # floor at 0.25°F

        mu_changed = abs(mu_target - mu) > 1e-6
        sigma_changed = abs(new_sigma - sigma) > 0.01

        if mu_changed or sigma_changed:
            dist = baseline.distribution
            if mu_changed:
                dist = shift_distribution(dist, mu_target - mu, note="")
            if sigma_changed:
                dist = scale_sigma(dist, new_sigma / max(sigma, 0.01), note="")
            dist._notes = dist._notes if hasattr(dist, '_notes') else []
            baseline = dataclasses.replace(
                baseline,
                distribution=recalibrate_distribution(
                    baseline.distribution,
                    mu_target if mu_changed else mu,
                    new_sigma if sigma_changed else sigma,
                    note=(
                        f"remaining_prior_high rem={rem_blend:.2f} "
                        f"p_lock={p_lock:.2f} sigma={sigma:.2f}→{new_sigma:.2f} "
                        f"cp={cp_probability:.2f}"
                    ),
                ),
            )
            notes.append(
                f"high remaining prior: rem={rem_blend:.2f} p_lock={p_lock:.2f} "
                f"sigma={sigma:.2f}→{new_sigma:.2f} cp={cp_probability:.2f}"
            )

    elif market_type == "low" and running_low_f is not None:
        rem_model = max(0.0, running_low_f - mu)
        alpha_base = max(0.25, model_trust * (1.0 - 0.5 * p_lock))
        alpha = min(0.95, alpha_base + 0.5 * cp_probability)
        rem_blend = alpha * rem_model + (1.0 - alpha) * rem_prior
        rem_blend = min(rem_blend, rem_q90)
        mu_target = running_low_f - rem_blend

        # Sigma compression from climatology (symmetric with HIGH path).
        sigma_clim_remaining = float(profile.get("sigma_remaining", sigma))
        sigma_cap = sigma_clim_remaining * (1.0 + 2.0 * cp_probability)
        new_sigma = min(sigma, max(0.25, sigma_cap))

        mu_changed = abs(mu_target - mu) > 1e-6
        sigma_changed = abs(new_sigma - sigma) > 0.01

        if mu_changed or sigma_changed:
            baseline = dataclasses.replace(
                baseline,
                distribution=recalibrate_distribution(
                    baseline.distribution,
                    mu_target if mu_changed else mu,
                    new_sigma if sigma_changed else sigma,
                    note=(
                        f"remaining_prior_low rem={rem_blend:.2f} "
                        f"p_lock={p_lock:.2f} sigma={sigma:.2f}→{new_sigma:.2f} "
                        f"cp={cp_probability:.2f}"
                    ),
                ),
            )
            notes.append(
                f"low remaining prior: rem={rem_blend:.2f} p_lock={p_lock:.2f} "
                f"sigma={sigma:.2f}→{new_sigma:.2f} cp={cp_probability:.2f}"
            )

    return baseline


# ---------------------------------------------------------------------------
# Core inference cycle
# ---------------------------------------------------------------------------

def run_inference_cycle(
    db_path: str | Path,
    target_date: str | None = None,
    station: str = "KMIA",
    market_prices_override: dict[str, float] | None = None,
    runtime_state: InferenceRuntimeState | None = None,
    eval_time_utc: datetime | None = None,
) -> InferenceCycleResult:
    """Run the full inference pipeline for a given target date.

    Returns InferenceCycleResult with per-bracket probability estimates.
    """
    db = sqlite3.connect(str(db_path))
    db.row_factory = sqlite3.Row
    now_utc = (eval_time_utc.astimezone(UTC) if eval_time_utc is not None else datetime.now(tz=UTC))
    timestamp_utc = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

    try:
        return _run_inference(
            db,
            target_date,
            station,
            now_utc,
            timestamp_utc,
            market_prices_override,
            runtime_state=runtime_state,
        )
    finally:
        db.close()


def _run_inference(
    db: sqlite3.Connection,
    target_date: str | None,
    station: str,
    now_utc: datetime,
    timestamp_utc: str,
    market_prices_override: dict[str, float] | None = None,
    runtime_state: InferenceRuntimeState | None = None,
) -> InferenceCycleResult:
    notes: list[str] = []

    # 1. Signal extraction
    sig_engine = SignalEngine(db, station)
    if target_date is None:
        target_date = sig_engine.current_target_date(now_utc)
    signals = sig_engine.extract(target_date, now_utc=now_utc)

    if not signals:
        return InferenceCycleResult(
            target_date=target_date,
            timestamp_utc=timestamp_utc,
            notes=["No signals extracted — missing obs data?"],
        )

    # Index by market_type
    sig_by_type: dict[str, SignalState] = {s.market_type: s for s in signals}
    notes.append(f"extracted {len(signals)} signal states")

    # 1b. LETKF spatial assimilation (optional — runs when cluster state is available)
    # Assimilates ALL nearby observations (ASOS + FAWN + buoy + primary) via formal
    # observation operators (T3.2) to produce analyzed temperature at trading stations.
    letkf_analysis: dict[str, tuple[float, float]] | None = None
    if runtime_state is not None and hasattr(runtime_state, 'letkf_state') and runtime_state.letkf_state is not None:
        from engine.observation_operators import build_letkf_obs_from_db
        try:
            obs_list = build_letkf_obs_from_db(db, station, timestamp_utc)
            if len(obs_list) >= 2:
                letkf_analysis = runtime_state.letkf_state.update(obs_list)
                if letkf_analysis:
                    for stn, (mu, spread) in letkf_analysis.items():
                        notes.append(f"letkf {stn}: analyzed_temp={mu:.1f}°F ±{spread:.2f}")
                    notes.append(f"letkf: {len(obs_list)} obs assimilated")
        except Exception:
            log.debug("LETKF update skipped", exc_info=True)

    # 2. Build forecast snapshots from *remaining* forward hours only.
    high_sig = sig_by_type.get("high")
    low_sig = sig_by_type.get("low")
    running_high_now = high_sig.running_high_f if high_sig else None
    running_low_now = low_sig.running_low_f if low_sig else None

    snapshots = _build_remaining_move_snapshots(
        db,
        station,
        target_date,
        now_utc,
        running_high_now,
        running_low_now,
    )
    if snapshots:
        notes.append(f"loaded {len(snapshots)} remaining-only forecast snapshots")
    else:
        snapshots = _build_forecast_snapshots(db, station, target_date, as_of_utc=now_utc)
        notes.append(f"remaining snapshots unavailable; fallback daily snapshots={len(snapshots)}")

    # 3. Build baseline beliefs (one per market type)
    # Load BOA weights (T1.5) — when available with sufficient updates,
    # BOA family multipliers override static source trust (T1.4).
    boa_state_path = Path("analysis_data/boa_state.json")
    boa_manager = BOAManager.load(boa_state_path)
    boa_overrides: dict[str, dict[str, float]] = {}  # {market_type: {family: mult}}
    inf_cfg = runtime_state.config if runtime_state else InferenceConfig()
    boa_min_updates = inf_cfg.boa_min_updates

    for mt in ("high", "low"):
        boa_weights = boa_manager.get_weights(mt)
        boa_state = boa_manager.high if mt == "high" else boa_manager.low
        if boa_state.total_updates >= boa_min_updates and boa_weights:
            # Aggregate source-level weights to family-level multipliers.
            # BOA keys are "source:model" (e.g., "openmeteo:gfs_seamless").
            # Family is everything before the first ":".
            family_sums: dict[str, float] = {}
            family_counts: dict[str, int] = {}
            for src_key, w in boa_weights.items():
                family = src_key.split(":")[0] if ":" in src_key else src_key
                family_sums[family] = family_sums.get(family, 0.0) + w
                family_counts[family] = family_counts.get(family, 0) + 1

            # Convert to relative multipliers (mean weight → multiplier vs equal weight)
            n_families = len(family_sums)
            if n_families > 0:
                equal_share = 1.0 / n_families
                family_mults = {}
                for fam, total_w in family_sums.items():
                    avg_w = total_w / family_counts[fam]
                    # Multiplier relative to equal share, clipped to [0.5, 2.0]
                    mult = max(0.5, min(2.0, avg_w / max(equal_share, 1e-8)))
                    family_mults[fam] = round(mult, 4)
                boa_overrides[mt] = family_mults
                notes.append(
                    f"{mt} boa: {boa_state.total_updates} updates, "
                    f"family_mults={family_mults}"
                )

    baselines: dict[str, object] = {}
    for market_type in ("high", "low"):
        sig = sig_by_type.get(market_type)
        hours_to_settlement = float(sig.hours_remaining) if sig and sig.hours_remaining is not None else None

        # Use BOA overrides for this market type if available, else fall back to source trust
        engine_config = BaselineEngineConfig(
            station=station,
            boa_family_overrides=boa_overrides.get(market_type),
        )
        baseline_engine = BaselineEngine(engine_config)

        baselines[market_type] = baseline_engine.build_baseline(
            snapshots,
            market_type=market_type,
            target_date=target_date,
            eval_time_utc=now_utc,
            hours_to_settlement=hours_to_settlement,
        )

    # 3b. EMOS calibration on REMAINING MOVE (not absolute forecast).
    # EMOS corrects the forward curve's predicted remaining move, which
    # naturally scales with time-of-day: late in the day, the predicted
    # remaining move is small → EMOS correction is small.
    emos_state_path = Path("analysis_data/emos_state.json")
    emos_state = EMOSState.load(emos_state_path)
    for market_type in ("high", "low"):
        baseline = baselines[market_type]
        if baseline.distribution.mu is None or baseline.distribution.sigma is None:
            continue

        sig = sig_by_type.get(market_type)
        if sig is None:
            continue

        # Compute predicted remaining move from forward curve baseline
        mu_baseline = baseline.distribution.mu
        if market_type == "high" and sig.running_high_f is not None:
            predicted_remaining = max(0.0, mu_baseline - sig.running_high_f)
            running_extreme = sig.running_high_f
        elif market_type == "low" and sig.running_low_f is not None:
            predicted_remaining = max(0.0, sig.running_low_f - mu_baseline)
            running_extreme = sig.running_low_f
        else:
            continue

        # Compute ensemble spread
        forecast_vals = []
        for snap in snapshots:
            val = snap.forecast_high_f if market_type == "high" else snap.forecast_low_f
            if val is not None:
                forecast_vals.append(val)
        ens_spread = (
            float(max(0.01, (max(forecast_vals) - min(forecast_vals)) / 4.0))
            if len(forecast_vals) >= 2
            else float(max(0.01, baseline.distribution.sigma or 1.0))
        )

        coeff = _resolve_emos_coefficients(emos_state, market_type, notes)
        cal_remaining, sigma_cal = coeff.predict_remaining(predicted_remaining, spread=ens_spread)

        # Convert calibrated remaining move back to absolute mu
        if market_type == "high":
            mu_cal = running_extreme + cal_remaining
        else:
            mu_cal = running_extreme - cal_remaining

        mu_old = baseline.distribution.mu
        sigma_old = baseline.distribution.sigma

        baselines[market_type] = dataclasses.replace(
            baseline,
            distribution=recalibrate_distribution(
                baseline.distribution,
                mu_cal,
                sigma_cal,
                note=(
                    f"emos_remaining: pred_rem={predicted_remaining:.1f} "
                    f"cal_rem={cal_remaining:.1f} "
                    f"mu={mu_old:.1f}→{mu_cal:.1f}"
                ),
            ),
        )
        notes.append(
            f"{market_type} emos: rem={predicted_remaining:.1f}→{cal_remaining:.1f}, "
            f"mu={mu_old:.1f}→{mu_cal:.1f}, sigma={sigma_old:.2f}→{sigma_cal:.2f}"
        )

    # 4. Change detection + SKF (persist state across cycles when runtime_state is provided)
    if runtime_state is not None:
        cd, skf = _ensure_runtime_models(runtime_state, target_date, notes)
        elapsed_minutes = 5.0
        if runtime_state.last_cycle_utc is not None:
            dt_minutes = (now_utc - runtime_state.last_cycle_utc).total_seconds() / 60.0
            elapsed_minutes = max(0.5, min(60.0, dt_minutes))
        runtime_state.last_cycle_utc = now_utc
    else:
        cd = ChangeDetector()
        cd.reset()
        skf = SwitchingKalmanFilter()
        skf.reset(target_date)
        elapsed_minutes = 5.0

    # Count regime label training days for SKF trust diagnostics
    n_training_days = db.execute(
        "SELECT COUNT(*) FROM regime_labels WHERE station = ?", (station,)
    ).fetchone()[0]

    cp_state = None
    skf_state = None

    # Approximate hour_lst from fixed LST frame (UTC-5 for KMIA climate day rules).
    hour_lst = (now_utc.hour - 5) % 24 + now_utc.minute / 60.0

    # Estimate cloud cover from METAR obs + FAWN solar proxy
    cloud_est = None
    try:
        cloud_est = estimate_cloud_fraction(db, station, now_utc)
        if cloud_est is not None:
            notes.append(f"cloud: {cloud_est.fraction:.2f} ({cloud_est.source}, n_metar={cloud_est.n_metar_stations})")
    except Exception:
        pass  # cloud estimation is optional; don't block inference

    if high_sig and high_sig.obs_current_f is not None:
        sky_cover_pct = cloud_est.fraction * 100.0 if cloud_est is not None else None
        obs_dict = {
            "temp_f": high_sig.obs_current_f,
            "dew_f": high_sig.dew_point_f,
            "pressure_hpa": high_sig.pressure_hpa,
            "wind_speed_mph": None,
            "wind_gust_mph": None,
            "wind_dir_deg": high_sig.wind_dir_deg,
            "sky_cover_pct": sky_cover_pct,
        }
        cp_state = cd.update(obs_dict, hour_lst, minutes_elapsed=elapsed_minutes)

        if cp_state.fired:
            skf.notify_changepoint(cp_state.changepoint_probability, cp_state.channels_fired)
            notes.append(
                f"changepoint fired: p={cp_state.changepoint_probability:.2f}, "
                f"channels={cp_state.channels_fired}, "
                f"bocpd={cp_state.bocpd_probability:.2f}, "
                f"runlen={cp_state.bocpd_run_length_mode}"
            )
        elif cp_state.bocpd_run_length_mode is not None:
            notes.append(
                f"changepoint monitor: bocpd={cp_state.bocpd_probability:.2f}, "
                f"runlen={cp_state.bocpd_run_length_mode}, "
                f"mins_since_cp={cp_state.minutes_since_last_changepoint}"
            )

        if high_sig.wind_dir_deg is not None:
            skf_obs = ObsVector(
                temp_f=high_sig.obs_current_f,
                dew_f=high_sig.dew_point_f or high_sig.obs_current_f - 10.0,
                pressure_hpa=high_sig.pressure_hpa or 1013.25,
                wind_dir_deg=high_sig.wind_dir_deg,
                timestamp_utc=timestamp_utc,
            )
            skf_state = skf.update(skf_obs, hours_elapsed=elapsed_minutes / 60.0)

    # 4b. Live regime catalog inference (T4.1-T4.2)
    # Infer the current weather regime from signals + atmospheric context.
    # The regime conditions sigma scaling and mu bias for bracket pricing.
    live_regime_state = None
    try:
        from engine.regime_catalog import infer_live_regime
        live_regime_state = infer_live_regime(
            cape=high_sig.cape_current if high_sig else None,
            pw_mm=high_sig.pw_mm if high_sig else None,
            wind_dir_deg=high_sig.wind_dir_deg if high_sig else None,
            cloud_fraction=cloud_est.fraction if cloud_est is not None else None,
            dew_crash_active=high_sig.dew_crash_active if high_sig else False,
            pressure_surge=high_sig.pressure_surge if high_sig else False,
            bocpd_changepoint_prob=cp_state.bocpd_probability if cp_state else 0.0,
            hour_lst=hour_lst,
            letkf_spread=(
                float(letkf_analysis[station][1])
                if letkf_analysis and station in letkf_analysis
                else None
            ),
        )
        notes.append(
            f"regime: {live_regime_state.primary.value} "
            f"({live_regime_state.confidence:.0%}), "
            f"sigma_mult={live_regime_state.sigma_multiplier:.2f}, "
            f"mu_bias_high={live_regime_state.mu_bias_high_f:+.1f}°F, "
            f"mu_bias_low={live_regime_state.mu_bias_low_f:+.1f}°F"
        )
    except Exception:
        log.debug("Live regime inference skipped", exc_info=True)

    # 5. Single-layer correction only: SKF/changepoint widening on top of remaining-target baseline.
    adjusted: dict[str, AdjustedBelief] = {}
    regimes: dict[str, RegimeState] = {}

    for market_type in ("high", "low"):
        sig = sig_by_type.get(market_type)
        if sig is None:
            continue

        baseline = baselines[market_type]
        if baseline.distribution.mu is None:
            continue

        # Apply live regime conditioning (sigma scaling + mu bias) before anchoring
        if live_regime_state is not None and live_regime_state.confidence > inf_cfg.regime_confidence_gate:
            if live_regime_state.sigma_multiplier != 1.0:
                baseline = dataclasses.replace(
                    baseline,
                    distribution=scale_sigma(
                        baseline.distribution,
                        live_regime_state.sigma_multiplier,
                        note=f"regime_sigma_{live_regime_state.primary.value}={live_regime_state.sigma_multiplier:.2f}",
                    ),
                )
            # Use market-type-specific mu bias (cloud suppression has opposite
            # effects on highs vs lows)
            mu_bias = (
                live_regime_state.mu_bias_high_f if market_type == "high"
                else live_regime_state.mu_bias_low_f
            )
            if abs(mu_bias) > inf_cfg.regime_mu_bias_min_f:
                baseline = dataclasses.replace(
                    baseline,
                    distribution=shift_distribution(
                        baseline.distribution,
                        mu_bias,
                        note=f"regime_mu_bias_{live_regime_state.primary.value}_{market_type}={mu_bias:+.1f}°F",
                    ),
                )

        # LETKF spatial sigma constraint (T3.3): use analysis spread to
        # tighten or widen baseline sigma.  Gated by n_updates >= 3 so the
        # ensemble has had enough assimilation cycles to be trustworthy.
        letkf_n_updates = (
            runtime_state.letkf_state.n_updates
            if runtime_state is not None
            and hasattr(runtime_state, "letkf_state")
            and runtime_state.letkf_state is not None
            else 0
        )
        if (
            letkf_analysis is not None
            and station in letkf_analysis
            and letkf_n_updates >= inf_cfg.letkf_min_updates_for_sigma
            and baseline.distribution.sigma is not None
        ):
            letkf_mu, letkf_spread = letkf_analysis[station]
            baseline_sigma = baseline.distribution.sigma

            # Log spatial signal if LETKF analysis diverges from raw obs
            obs_current = (
                sig.obs_current_f if sig.obs_current_f is not None else None
            )
            if obs_current is not None and abs(letkf_mu - obs_current) > inf_cfg.letkf_spatial_divergence_f:
                notes.append(
                    f"{market_type} letkf_spatial_signal: "
                    f"analysis={letkf_mu:.1f}°F vs obs={obs_current:.1f}°F "
                    f"(delta={letkf_mu - obs_current:+.1f}°F)"
                )

            # Blend LETKF spread with baseline sigma.
            # Weight LETKF more as n_updates grows (max weight 0.4 at 10+ updates).
            letkf_weight = min(inf_cfg.letkf_sigma_max_weight, 0.1 * (letkf_n_updates - 2))

            if letkf_spread < baseline_sigma:
                # LETKF says spatial field is tighter -> compress sigma
                blended_sigma = (1.0 - letkf_weight) * baseline_sigma + letkf_weight * letkf_spread
                baseline = dataclasses.replace(
                    baseline,
                    distribution=scale_sigma(
                        baseline.distribution,
                        blended_sigma / max(baseline_sigma, 0.01),
                        note=(
                            f"letkf_tighten: spread={letkf_spread:.2f} < sigma={baseline_sigma:.2f}, "
                            f"w={letkf_weight:.2f} -> {blended_sigma:.2f}"
                        ),
                    ),
                )
            elif letkf_spread > baseline_sigma:
                # LETKF says spatial disagreement -> widen sigma
                blended_sigma = (1.0 - letkf_weight) * baseline_sigma + letkf_weight * letkf_spread
                baseline = dataclasses.replace(
                    baseline,
                    distribution=scale_sigma(
                        baseline.distribution,
                        blended_sigma / max(baseline_sigma, 0.01),
                        note=(
                            f"letkf_widen: spread={letkf_spread:.2f} > sigma={baseline_sigma:.2f}, "
                            f"w={letkf_weight:.2f} -> {blended_sigma:.2f}"
                        ),
                    ),
                )

            notes.append(
                f"{market_type} letkf_sigma: spread={letkf_spread:.2f}, "
                f"baseline_sigma={baseline_sigma:.2f}, "
                f"weight={letkf_weight:.2f}, n_updates={letkf_n_updates}"
            )

        # Anchor baseline to current running extremes before regime corrections.
        if market_type == "high" and sig.running_high_f is not None:
            mu = baseline.distribution.mu
            if mu is not None and mu < sig.running_high_f:
                baseline = dataclasses.replace(
                    baseline,
                    distribution=shift_distribution(
                        baseline.distribution,
                        sig.running_high_f - mu,
                        note=f"remaining_anchor_high={sig.running_high_f:.1f}°F",
                    ),
                )
        elif market_type == "low" and sig.running_low_f is not None:
            mu = baseline.distribution.mu
            if mu is not None and mu > sig.running_low_f:
                baseline = dataclasses.replace(
                    baseline,
                    distribution=shift_distribution(
                        baseline.distribution,
                        sig.running_low_f - mu,
                        note=f"remaining_anchor_low={sig.running_low_f:.1f}°F",
                    ),
                )

        # Apply remaining-move prior: blends the forward curve's remaining-move
        # prediction with sigma climatology (IEM ASOS 6yr empirical data).
        # This naturally compresses sigma late in the day when the climatological
        # remaining potential is small, while preserving width during regime breaks
        # (changepoint probability widens the climatological cap).
        cp_prob = cp_state.bocpd_probability if cp_state else 0.0
        baseline = _apply_remaining_move_prior(
            baseline,
            market_type=market_type,
            target_date=target_date,
            hour_lst=int(hour_lst),
            running_high_f=sig.running_high_f,
            running_low_f=sig.running_low_f,
            notes=notes,
            cp_probability=cp_prob,
        )

        # SKF: dormant unless explicitly re-enabled via inference_config.json.
        # Still runs and logs state for research/comparison, but mu/sigma
        # corrections are NOT applied to the live distribution.
        # Superseded by: regime catalog (T4.1-T4.2), EMOS (T1.2), LETKF (T3.3).
        skf_for_corrections = skf_state if inf_cfg.skf_corrections_enabled else None
        if skf_state is not None and not inf_cfg.skf_corrections_enabled:
            # Log SKF state for research even though corrections are dormant
            notes.append(
                f"{market_type} skf_dormant: regime_probs={skf_state.regime_probabilities!r}, "
                f"mu_shift={getattr(skf_state, 'mu_shift', 'N/A')}"
            )

        adj, regime = _apply_single_regime_layer(
            baseline,
            market_type=market_type,
            running_high_f=sig.running_high_f,
            running_low_f=sig.running_low_f,
            skf_state=skf_for_corrections,
            cp_state=cp_state,
            n_training_days=n_training_days,
        )

        adjusted[market_type] = adj
        regimes[market_type] = regime

        mu_note = adj.distribution.mu if adj.distribution.mu is not None else float("nan")
        sigma_note = adj.distribution.sigma if adj.distribution.sigma is not None else float("nan")
        regime_label = live_regime_state.primary.value if live_regime_state else regime.primary_regime
        regime_conf = live_regime_state.confidence if live_regime_state else regime.confidence
        notes.append(
            f"{market_type}: mu={mu_note:.1f}°F, "
            f"sigma={sigma_note:.2f}°F, "
            f"regime={regime_label}({regime_conf:.2f})"
        )

    # 8. Load brackets from active_brackets, fallback to market_snapshots for replay.
    brackets = _load_brackets(db, target_date)
    if not brackets:
        brackets = _load_brackets_from_market_snapshots(db, target_date, now_utc)
        if brackets:
            notes.append("active_brackets empty; using market_snapshots fallback")
        else:
            notes.append("No active brackets for target date")

    # Use live override prices if provided, else latest full quotes from DB.
    if market_prices_override:
        market_quotes = {
            ticker: MarketQuote(ticker=ticker, last_price_cents=price)
            for ticker, price in market_prices_override.items()
        }
    else:
        market_quotes = _load_market_quotes(db, target_date, as_of_utc=now_utc)

    estimates: list[BracketEstimate] = []
    for bracket in brackets:
        belief = adjusted.get(bracket.market_type)
        if belief is None:
            continue

        # Pass truncation bounds so the CDF integrates over the
        # physically possible range only.
        high_sig = sig_by_type.get("high")
        low_sig = sig_by_type.get("low")
        trunc_lower = None
        trunc_upper = None
        if bracket.market_type == "high" and high_sig and high_sig.running_high_f is not None:
            trunc_lower = high_sig.running_high_f  # high can't go down
        elif bracket.market_type == "low" and low_sig and low_sig.running_low_f is not None:
            trunc_upper = low_sig.running_low_f  # low can't go up

        # Use 2-component Gaussian mixture for fatter tails.
        # This fixes the sigma-too-tight problem (1.3x ratio) by putting 25% weight
        # on a wide tail component, preventing overconfident tail bracket bets.
        mixture = fit_mixture_from_baseline(belief.distribution)
        prob = bracket_yes_probability(belief.distribution, bracket, trunc_lower, trunc_upper, mixture=mixture)
        if prob is None:
            continue

        # Physical constraint: zero out impossible brackets.
        if bracket.market_type == "high" and high_sig and high_sig.running_high_f is not None:
            running_h = high_sig.running_high_f
            # Bracket ceiling below running high → impossible
            if bracket.ceiling_f is not None and bracket.ceiling_f <= running_h and bracket.directional != "under":
                prob = 0.0
            # "Under" bracket with threshold below running high → impossible
            if bracket.directional == "under" and bracket.ceiling_f is not None and bracket.ceiling_f <= running_h:
                prob = 0.0
        elif bracket.market_type == "low" and low_sig and low_sig.running_low_f is not None:
            running_l = low_sig.running_low_f
            # Bracket floor above running low → impossible (low can only go lower)
            if bracket.floor_f is not None and bracket.floor_f >= running_l and bracket.directional != "over":
                prob = 0.0
            if bracket.directional == "over" and bracket.floor_f is not None and bracket.floor_f >= running_l:
                prob = 0.0

        quote = market_quotes.get(bracket.ticker)
        mkt_price = quote.last_price_cents if quote else None
        if mkt_price is None and quote and quote.best_yes_ask_cents is not None:
            mkt_price = quote.best_yes_ask_cents

        active_signals = []
        regime = regimes.get(bracket.market_type)
        if regime:
            active_signals = regime.tags[:]

        edge = None
        yes_price_for_edge = quote.best_yes_ask_cents if quote and quote.best_yes_ask_cents is not None else mkt_price
        if yes_price_for_edge is not None:
            yes_fee = kalshi_fee_cents(yes_price_for_edge)
            edge = round(prob * 100.0 - yes_price_for_edge - yes_fee, 1)

        no_prob = round(1.0 - prob, 4)
        if quote and quote.best_no_ask_cents is not None:
            no_mkt = quote.best_no_ask_cents
        else:
            no_mkt = round(100.0 - mkt_price, 1) if mkt_price is not None else None
        no_edge = None
        if no_mkt is not None:
            no_fee = kalshi_fee_cents(no_mkt)
            no_edge = round(no_prob * 100.0 - no_mkt - no_fee, 1)

        estimates.append(BracketEstimate(
            ticker=bracket.ticker,
            market_type=bracket.market_type,
            floor_f=bracket.floor_f,
            ceiling_f=bracket.ceiling_f,
            directional=bracket.directional,
            model_probability=round(prob, 4),
            mu=belief.distribution.mu,
            sigma=belief.distribution.sigma,
            market_price=mkt_price,
            edge=edge,
            no_probability=no_prob,
            no_market_price=no_mkt,
            no_edge=no_edge,
            active_signals=active_signals,
            regime_confidence=(regime.confidence if regime else None),
        ))

    # 8b. Apply Platt calibration post-fix (when trained)
    try:
        from engine.platt_calibrator import PlattCalibratorState
        platt_state = PlattCalibratorState.load()
        if platt_state.high.n_samples >= 10 or platt_state.low.n_samples >= 10:
            for est in estimates:
                cal = platt_state.high if est.market_type == "high" else platt_state.low
                if cal.n_samples >= 10:
                    raw_p = est.model_probability
                    est_dict = est.__dict__
                    est_dict["model_probability"] = round(cal.calibrate(raw_p), 4)
                    est_dict["no_probability"] = round(1.0 - est_dict["model_probability"], 4)
            notes.append("platt_calibration_applied")
    except Exception:
        pass  # Platt calibration is optional

    # 8c. Market-implied density extraction and trade ranking
    try:
        from engine.market_implied_density import extract_market_density, rank_trades_by_edge
        bracket_data = []
        for est in estimates:
            if est.market_price is not None:
                bracket_data.append((
                    est.ticker, est.floor_f, est.ceiling_f,
                    est.market_type, est.market_price,
                ))
        if len(bracket_data) >= 3:
            market_density = extract_market_density(bracket_data)
            notes.append(
                f"market_implied: mu={market_density.mu_implied:.1f}°F, "
                f"sigma={market_density.sigma_implied:.2f}°F"
            )
    except Exception:
        log.debug("Market-implied density extraction skipped", exc_info=True)

    # 9. Write estimates to bracket_estimates table
    for est in estimates:
        try:
            db.execute(
                """INSERT OR REPLACE INTO bracket_estimates
                   (station, target_date, market_type, ticker, probability, mu, sigma,
                    active_signals, timestamp_utc, regime_confidence)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    station,
                    target_date,
                    est.market_type,
                    est.ticker,
                    est.model_probability,
                    est.mu,
                    est.sigma,
                    json.dumps(est.active_signals),
                    timestamp_utc,
                    est.regime_confidence,
                ),
            )
        except Exception as e:
            log.warning("Failed to write bracket estimate for %s: %s", est.ticker, e)
    db.commit()

    # 10. Bracket arbitrage scan (T6.4) — diagnostic monitor
    try:
        from engine.bracket_arbitrage import scan_bracket_arbitrage
        arb_opps = scan_bracket_arbitrage(db, target_date, station)
        for opp in arb_opps:
            notes.append(
                f"arb_detected: {opp.type} {opp.market_type} "
                f"profit={opp.profit_cents:.1f}¢ "
                f"sum_asks={opp.sum_asks:.0f} sum_bids={opp.sum_bids:.0f}"
            )
    except Exception:
        log.debug("Bracket arbitrage scan skipped", exc_info=True)

    return InferenceCycleResult(
        target_date=target_date,
        timestamp_utc=timestamp_utc,
        high_belief=adjusted.get("high"),
        low_belief=adjusted.get("low"),
        high_regime=regimes.get("high"),
        low_regime=regimes.get("low"),
        estimates=estimates,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Bracket discovery — populate active_brackets from Kalshi REST
# ---------------------------------------------------------------------------

def discover_and_populate_brackets(
    db_path: str | Path,
    target_date: str | None = None,
    station: str = "KMIA",
) -> int:
    """Discover today's Kalshi brackets via REST API and write to active_brackets.

    Returns the number of brackets written.
    """
    # Import here to avoid circular deps / requiring Kalshi auth for basic orchestrator use
    from collector.sources.kalshi_rest import KalshiREST

    db = sqlite3.connect(str(db_path))
    db.row_factory = sqlite3.Row

    try:
        if target_date is None:
            sig_engine = SignalEngine(db, station)
            target_date = sig_engine.current_target_date()

        # Clear stale brackets for different dates
        db.execute("DELETE FROM active_brackets WHERE target_date != ?", (target_date,))

        # Discover markets from Kalshi
        kalshi = KalshiREST()
        high_tickers = kalshi.get_all_tickers()  # Returns all active KMIA high+low tickers

        # Get metadata for each ticker (already parsed by kalshi_rest)
        count = 0
        for ticker in high_tickers:
            meta = kalshi.get_market_metadata(ticker)
            if meta is None:
                continue

            forecast_date = meta.get("forecast_date")
            if forecast_date != target_date:
                continue

            market_type = meta.get("market_type", "high")
            floor_strike = meta.get("floor_strike")
            cap_strike = meta.get("cap_strike")

            # Determine settlement bounds
            settlement_floor, settlement_ceil = _settlement_bounds(
                market_type, floor_strike, cap_strike, ticker,
            )

            db.execute(
                """INSERT OR REPLACE INTO active_brackets
                   (ticker, market_type, target_date, floor_strike, cap_strike,
                    settlement_floor, settlement_ceil)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (ticker, market_type, target_date, floor_strike, cap_strike,
                 settlement_floor, settlement_ceil),
            )
            count += 1

        db.commit()
        log.info("Populated %d active brackets for %s", count, target_date)
        return count
    finally:
        db.close()


def _settlement_bounds(
    market_type: str,
    floor_strike: float | None,
    cap_strike: float | None,
    ticker: str,
) -> tuple[float, float]:
    """Compute continuous settlement bounds for CDF integration.

    Kalshi settles on whole-degree F (NWS rounding). To integrate a continuous
    distribution we apply half-degree continuity correction:

      B77.5 wins if reported value is 77 or 78.
        78.49… rounds to 78 → YES.  78.5 rounds to 79 → NO.
        → bounds = [floor - 0.5, floor + 1.5)
        → B77.5: [76.5, 78.5)

      T73 (under) wins if reported value ≤ 72  (i.e. <73).
        72.49… rounds to 72 → YES.  72.5 rounds to 73 → NO.
        → bounds = (-∞, floor - 0.5)
        → T73: (-∞, 72.5)

      T80 (over) wins if reported value ≥ 81  (i.e. >80).
        B79.5 covers 79 and 80, so T80 starts at 81.
        80.5 rounds to 81 → YES.  80.49… rounds to 80 → NO.
        → bounds = [floor + 0.5, ∞)
        → T80: [80.5, ∞)
    """
    parts = ticker.split("-")
    strike_part = parts[-1] if len(parts) >= 3 else ""

    if strike_part.startswith("T"):
        f = floor_strike or 0.0
        if cap_strike is not None and floor_strike is not None and cap_strike < floor_strike:
            # Under: T73 → reported ≤ 72 → true temp < 72.5
            return -50.0, f - 0.5
        else:
            # Over: T80 → reported ≥ 81 (i.e. >80) → true temp ≥ 80.5
            return f + 0.5, 200.0
    elif strike_part.startswith("B"):
        # B77.5 → reported 77 or 78 → true temp in [76.5, 78.5)
        f = floor_strike or 0.0
        return f - 0.5, f + 1.5
    else:
        return floor_strike or 0.0, cap_strike or 200.0


def _format_strike(est: BracketEstimate) -> str:
    """Format a bracket's strike info for display, including direction for T markets.

    Derives whole-degree labels from the ticker name (not settlement bounds,
    which include half-degree continuity correction).
    """
    parts = est.ticker.split("-")
    strike_part = parts[-1] if len(parts) >= 3 else ""

    label = est.market_type[0].upper()  # "H" or "L"

    if strike_part.startswith("T"):
        try:
            val = int(float(strike_part[1:]))
        except ValueError:
            val = 0
        if est.directional == "under":
            return f"{label} <{val}"
        else:
            return f"{label} >{val}"
    elif strike_part.startswith("B"):
        try:
            val = float(strike_part[1:])
            lo = int(val - 0.5)  # B77.5 → 77
            hi = lo + 1          # → 78
        except ValueError:
            lo, hi = 0, 0
        return f"{label} [{lo}-{hi}]"
    return est.market_type


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run Miami inference cycle")
    parser.add_argument("--db", default="miami_collector.db", help="Path to collector DB")
    parser.add_argument("--date", default=None, help="Target date (YYYY-MM-DD)")
    parser.add_argument("--discover", action="store_true", help="Discover brackets from Kalshi REST first")
    parser.add_argument("--station", default="KMIA")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    if args.discover:
        n = discover_and_populate_brackets(args.db, args.date, args.station)
        print(f"Discovered {n} brackets")

    result = run_inference_cycle(args.db, args.date, args.station)

    # Print results
    print(f"\n{'='*72}")
    print(f"Inference for {result.target_date} at {result.timestamp_utc}")
    print(f"{'='*72}")

    for note in result.notes:
        print(f"  {note}")

    if result.high_belief:
        b = result.high_belief
        print(f"\nHIGH: mu={b.distribution.mu:.1f}°F  sigma={b.distribution.sigma:.2f}°F  "
              f"regime={result.high_regime.primary_regime if result.high_regime else '?'}")
    if result.low_belief:
        b = result.low_belief
        print(f"LOW:  mu={b.distribution.mu:.1f}°F  sigma={b.distribution.sigma:.2f}°F  "
              f"regime={result.low_regime.primary_regime if result.low_regime else '?'}")

    if result.estimates:
        print(f"\n{'Ticker':<30} {'Strike':<12} {'YES%':>6} {'YES¢':>6} {'YEdge':>7} {'NO%':>6} {'NO¢':>6} {'NEdge':>7} {'Best':>4}")
        print("-" * 100)
        for est in result.estimates:
            y_mkt = f"{est.market_price:.0f}" if est.market_price is not None else "---"
            y_edge = f"{est.edge:+.1f}" if est.edge is not None else "---"
            n_mkt = f"{est.no_market_price:.0f}" if est.no_market_price is not None else "---"
            n_edge = f"{est.no_edge:+.1f}" if est.no_edge is not None else "---"
            strike = _format_strike(est)

            # Flag the better side
            best = ""
            if est.edge is not None and est.no_edge is not None:
                if est.edge > 0 and est.edge >= est.no_edge:
                    best = "YES"
                elif est.no_edge > 0 and est.no_edge > est.edge:
                    best = "NO"

            print(
                f"{est.ticker:<30} {strike:<12} "
                f"{est.model_probability*100:>5.1f} {y_mkt:>6} {y_edge:>7} "
                f"{est.no_probability*100:>5.1f} {n_mkt:>6} {n_edge:>7} "
                f"{best:>4}"
            )
    else:
        print("\nNo bracket estimates generated.")


if __name__ == "__main__":
    main()
