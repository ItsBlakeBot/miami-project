"""Dynamic source-weighting helpers for baseline aggregation.

Phase mapping:
- T1.1 dynamic weighting engine

This module keeps weighting logic modular so aggregation can evolve without
rewriting baseline construction internals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import exp


@dataclass(frozen=True)
class DynamicWeightingConfig:
    freshness_scale_hours: float = 6.0
    tracking_scale: float = 4.0
    min_weight: float = 0.01

    # As settlement gets closer, stale runs should decay faster.
    horizon_scale_hours: float = 8.0
    max_urgency_freshness_tightening: float = 0.45

    # Optional multipliers for source families (e.g. tuned from rolling skill).
    family_multipliers: dict[str, float] = field(default_factory=dict)

    # Changepoint sensitivity hook for old runs.
    break_probability_old_run_penalty: float = 0.75
    old_run_hours_threshold: float = 1.5


@dataclass(frozen=True)
class DynamicWeightContext:
    age_hours: float
    tracking_error_f: float | None
    hours_to_settlement: float | None = None
    regime_confidence: float | None = None
    break_probability: float | None = None
    source_family: str | None = None


@dataclass(frozen=True)
class WeightBreakdown:
    freshness_weight: float
    tracking_weight: float
    family_multiplier: float
    break_multiplier: float
    regime_multiplier: float
    effective_freshness_scale_hours: float
    final_weight: float


def _effective_freshness_scale(cfg: DynamicWeightingConfig, hours_to_settlement: float | None) -> float:
    if hours_to_settlement is None:
        return cfg.freshness_scale_hours

    ttl = max(0.0, float(hours_to_settlement))
    urgency = exp(-ttl / max(0.1, cfg.horizon_scale_hours))
    factor = 1.0 - cfg.max_urgency_freshness_tightening * urgency
    factor = max(0.2, min(1.0, factor))
    return max(0.1, cfg.freshness_scale_hours * factor)


def compute_dynamic_weight(
    ctx: DynamicWeightContext,
    cfg: DynamicWeightingConfig,
) -> WeightBreakdown:
    age = max(0.0, float(ctx.age_hours))

    eff_scale = _effective_freshness_scale(cfg, ctx.hours_to_settlement)
    freshness_weight = exp(-age / eff_scale)

    if ctx.tracking_error_f is None:
        tracking_weight = 0.85
    else:
        err = float(ctx.tracking_error_f)
        tracking_weight = exp(-((err ** 2) / max(0.001, cfg.tracking_scale)))

    family_multiplier = 1.0
    if ctx.source_family:
        family_multiplier = float(cfg.family_multipliers.get(ctx.source_family, 1.0))

    break_multiplier = 1.0
    if (
        ctx.break_probability is not None
        and float(ctx.break_probability) >= 0.7
        and age >= cfg.old_run_hours_threshold
    ):
        break_multiplier = cfg.break_probability_old_run_penalty

    regime_multiplier = 1.0
    if ctx.regime_confidence is not None and float(ctx.regime_confidence) < 0.35:
        regime_multiplier = 0.9

    raw = freshness_weight * tracking_weight * family_multiplier * break_multiplier * regime_multiplier
    final = max(cfg.min_weight, raw)

    return WeightBreakdown(
        freshness_weight=round(freshness_weight, 6),
        tracking_weight=round(tracking_weight, 6),
        family_multiplier=round(family_multiplier, 6),
        break_multiplier=round(break_multiplier, 6),
        regime_multiplier=round(regime_multiplier, 6),
        effective_freshness_scale_hours=round(eff_scale, 6),
        final_weight=round(final, 6),
    )
