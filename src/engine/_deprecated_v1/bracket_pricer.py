"""Bracket pricing from predictive distributions.

Supports truncated normal distributions via optional truncation_bound.
For highs, the distribution is truncated below the running high (can't go down).
For lows, the distribution is truncated above the running low (can't go up).

Also supports 2-component Gaussian mixture models for fatter tails.
The mixture puts 25% weight on a wide component (2x sigma), which
fattens tails without changing the central estimate. This prevents
overconfident pricing on tail brackets.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import erf, sqrt

from engine.quantile_combiner import PredictiveDistribution


@dataclass(frozen=True)
class Bracket:
    ticker: str
    market_type: str
    floor_f: float | None = None
    ceiling_f: float | None = None
    directional: str | None = None  # 'under' or 'over'


@dataclass(frozen=True)
class MixtureDistribution:
    """2-component Gaussian mixture parameterization."""
    mu1: float
    sigma1: float
    mu2: float
    sigma2: float
    weight1: float  # weight2 = 1 - weight1

    @property
    def weight2(self) -> float:
        return 1.0 - self.weight1


def _normal_cdf(x: float, mu: float, sigma: float) -> float:
    if sigma <= 0:
        return 1.0 if x >= mu else 0.0
    z = (x - mu) / (sigma * sqrt(2.0))
    return 0.5 * (1.0 + erf(z))


def _truncated_cdf(
    x: float, mu: float, sigma: float,
    lower: float | None, upper: float | None,
) -> float:
    """CDF of a truncated normal on [lower, upper].

    P(X <= x | lower <= X <= upper) = (Φ(x) - Φ(lower)) / (Φ(upper) - Φ(lower))
    where Φ is the standard normal CDF evaluated at (·-mu)/sigma.
    """
    raw = _normal_cdf(x, mu, sigma)
    lo_cdf = _normal_cdf(lower, mu, sigma) if lower is not None else 0.0
    hi_cdf = _normal_cdf(upper, mu, sigma) if upper is not None else 1.0

    denom = hi_cdf - lo_cdf
    if denom <= 1e-12:
        # All mass is outside the truncation window — point mass behavior
        return 1.0 if x >= mu else 0.0

    # Clamp raw to the truncation window
    raw = max(raw, lo_cdf)
    raw = min(raw, hi_cdf)

    return (raw - lo_cdf) / denom


def _mixture_cdf(
    x: float,
    mix: MixtureDistribution,
    lower: float | None,
    upper: float | None,
) -> float:
    """CDF of a 2-component truncated Gaussian mixture.

    mixture_CDF(x) = w1 * truncated_normal_cdf(x, mu1, sigma1)
                    + w2 * truncated_normal_cdf(x, mu2, sigma2)

    Each component is independently truncated to [lower, upper].
    """
    cdf1 = _truncated_cdf(x, mix.mu1, mix.sigma1, lower, upper)
    cdf2 = _truncated_cdf(x, mix.mu2, mix.sigma2, lower, upper)
    return mix.weight1 * cdf1 + mix.weight2 * cdf2


def fit_mixture_from_baseline(dist: PredictiveDistribution) -> MixtureDistribution | None:
    """Create a 2-component mixture from a single-Gaussian PredictiveDistribution.

    Component 1 (core):  mu, sigma * 0.7, weight = 0.75
    Component 2 (tail):  mu, sigma * 2.0, weight = 0.25

    This gives heavier tails than a single Gaussian while keeping the
    same mean. The core component is tighter (0.7x) to compensate for
    the wide tail component, so the blended 50th-percentile region stays
    close to the original.
    """
    mu = dist.mu
    sigma = dist.sigma
    if mu is None or sigma is None:
        return None
    return MixtureDistribution(
        mu1=mu,
        sigma1=sigma * 0.7,
        mu2=mu,
        sigma2=sigma * 2.0,
        weight1=0.75,
    )


def bracket_yes_probability_mixture(
    mixture: MixtureDistribution,
    bracket: Bracket,
    truncation_lower: float | None = None,
    truncation_upper: float | None = None,
) -> float | None:
    """Compute P(YES) for a bracket using a 2-component mixture distribution."""

    def cdf(x: float) -> float:
        return _mixture_cdf(x, mixture, truncation_lower, truncation_upper)

    if bracket.directional == "under" and bracket.ceiling_f is not None:
        return cdf(bracket.ceiling_f)
    if bracket.directional == "over" and bracket.floor_f is not None:
        return 1.0 - cdf(bracket.floor_f)
    if bracket.floor_f is not None and bracket.ceiling_f is not None:
        upper = cdf(bracket.ceiling_f)
        lower = cdf(bracket.floor_f)
        return max(0.0, min(1.0, upper - lower))
    return None


def bracket_yes_probability(
    distribution: PredictiveDistribution,
    bracket: Bracket,
    truncation_lower: float | None = None,
    truncation_upper: float | None = None,
    mixture: MixtureDistribution | None = None,
) -> float | None:
    """Compute P(YES) for a bracket.

    truncation_lower: hard floor (e.g., running high for high markets)
    truncation_upper: hard ceiling (e.g., running low for low markets)
    mixture: optional 2-component mixture; when provided, uses mixture CDF
             instead of single Gaussian for fatter tails.
    """
    if mixture is not None:
        return bracket_yes_probability_mixture(
            mixture, bracket, truncation_lower, truncation_upper,
        )

    mu = distribution.mu
    sigma = distribution.sigma
    if mu is None or sigma is None:
        return None

    def cdf(x: float) -> float:
        return _truncated_cdf(x, mu, sigma, truncation_lower, truncation_upper)

    if bracket.directional == "under" and bracket.ceiling_f is not None:
        return cdf(bracket.ceiling_f)
    if bracket.directional == "over" and bracket.floor_f is not None:
        return 1.0 - cdf(bracket.floor_f)
    if bracket.floor_f is not None and bracket.ceiling_f is not None:
        upper = cdf(bracket.ceiling_f)
        lower = cdf(bracket.floor_f)
        return max(0.0, min(1.0, upper - lower))
    return None
