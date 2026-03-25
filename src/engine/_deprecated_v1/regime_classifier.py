"""RegimeState data structure for SKF/changepoint correction output.

The old classify_regime() function and RegimeInput class have been removed —
they were superseded by regime_catalog.infer_live_regime() which uses
atmospheric signals (CAPE, PW, cloud fraction, wind, BOCPD) directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RegimeState:
    primary_regime: str
    path_class: str
    confidence: float
    tags: list[str] = field(default_factory=list)
    # SKF-derived fields (populated when Switching Kalman Filter is active)
    skf_probabilities: dict[str, float] = field(default_factory=dict)
    skf_mu_shift_high: float = 0.0
    skf_mu_shift_low: float = 0.0
    skf_sigma_scale_high: float = 1.0
    skf_sigma_scale_low: float = 1.0
    skf_active_families: dict[str, float] = field(default_factory=dict)
