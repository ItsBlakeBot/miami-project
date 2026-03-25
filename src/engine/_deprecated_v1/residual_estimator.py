"""Data structures for adjusted beliefs in the inference pipeline.

The old apply_residual_adjustment() function has been removed — it was
superseded by the remaining-move architecture + regime_catalog + sigma
climatology, all composed directly in orchestrator._apply_single_regime_layer().
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .baseline_engine import BaselineBelief
from .quantile_combiner import PredictiveDistribution


@dataclass
class ResidualAdjustment:
    delta_mu_f: float = 0.0
    sigma_multiplier: float = 1.0
    clamps: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class AdjustedBelief:
    market_type: str
    target_date: str
    baseline: BaselineBelief
    distribution: PredictiveDistribution
    adjustment: ResidualAdjustment
