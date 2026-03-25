"""Quantile-based probabilistic combination utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from statistics import NormalDist

from .source_registry import ForecastSourceSnapshot

DEFAULT_QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)


@dataclass
class PredictiveDistribution:
    market_type: str
    target_date: str
    quantiles: dict[float, float]
    source_weights: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def p50(self) -> float | None:
        return self.quantiles.get(0.50)

    @property
    def mu(self) -> float | None:
        return self.p50

    @property
    def sigma(self) -> float | None:
        p50 = self.quantiles.get(0.50)
        p90 = self.quantiles.get(0.90)
        if p50 is None or p90 is None:
            return None
        z90 = NormalDist().inv_cdf(0.90)
        if z90 == 0:
            return None
        return max(0.0, (p90 - p50) / z90)

    def with_quantiles(self, quantiles: dict[float, float], note: str | None = None) -> "PredictiveDistribution":
        notes = list(self.notes)
        if note:
            notes.append(note)
        return PredictiveDistribution(
            market_type=self.market_type,
            target_date=self.target_date,
            quantiles=quantiles,
            source_weights=dict(self.source_weights),
            notes=notes,
        )


def combine_quantiles(
    snapshots: list[ForecastSourceSnapshot],
    *,
    market_type: str,
    target_date: str,
    weights: dict[str, float] | None = None,
    quantile_grid: tuple[float, ...] = DEFAULT_QUANTILES,
) -> PredictiveDistribution:
    """Combine per-source quantiles into a weighted predictive distribution.

    If a source lacks explicit quantiles, we fall back to its point forecast as a
    degenerate distribution at all requested quantiles. That keeps v1 simple.
    """
    if not snapshots:
        return PredictiveDistribution(market_type=market_type, target_date=target_date, quantiles={})

    resolved_weights: dict[str, float] = {}
    for snapshot in snapshots:
        key = snapshot.source_key
        resolved_weights[key] = max(0.0, (weights or {}).get(key, 1.0))

    total_weight = sum(resolved_weights.values()) or 1.0
    normalized = {k: v / total_weight for k, v in resolved_weights.items()}

    combined: dict[float, float] = {}
    for q in quantile_grid:
        acc = 0.0
        used = 0.0
        for snapshot in snapshots:
            key = snapshot.source_key
            source_quantiles = snapshot.quantiles_high if market_type == "high" else snapshot.quantiles_low
            if source_quantiles:
                value = source_quantiles.get(q)
            else:
                value = snapshot.forecast_high_f if market_type == "high" else snapshot.forecast_low_f
            if value is None:
                continue
            w = normalized[key]
            acc += w * value
            used += w
        if used > 0:
            combined[q] = round(acc / used, 3)

    return PredictiveDistribution(
        market_type=market_type,
        target_date=target_date,
        quantiles=combined,
        source_weights={k: round(v, 6) for k, v in normalized.items()},
    )


def clamp_upper(distribution: PredictiveDistribution, upper_bound_f: float, *, note: str) -> PredictiveDistribution:
    quantiles = {q: min(v, upper_bound_f) for q, v in distribution.quantiles.items()}
    return distribution.with_quantiles(quantiles, note=note)


def clamp_lower(distribution: PredictiveDistribution, lower_bound_f: float, *, note: str) -> PredictiveDistribution:
    quantiles = {q: max(v, lower_bound_f) for q, v in distribution.quantiles.items()}
    return distribution.with_quantiles(quantiles, note=note)


def shift_distribution(distribution: PredictiveDistribution, delta_f: float, *, note: str) -> PredictiveDistribution:
    quantiles = {q: round(v + delta_f, 3) for q, v in distribution.quantiles.items()}
    return distribution.with_quantiles(quantiles, note=note)


def scale_sigma(distribution: PredictiveDistribution, multiplier: float, *, note: str) -> PredictiveDistribution:
    if not distribution.quantiles or distribution.p50 is None:
        return distribution
    median = distribution.p50
    quantiles = {}
    for q, v in distribution.quantiles.items():
        offset = v - median
        quantiles[q] = round(median + offset * multiplier, 3)
    return distribution.with_quantiles(quantiles, note=note)


def recalibrate_distribution(
    distribution: PredictiveDistribution,
    new_mu: float,
    new_sigma: float,
    *,
    note: str,
) -> PredictiveDistribution:
    """Rebuild quantiles from calibrated (mu, sigma) using Normal assumption.

    Replaces the distribution's quantile grid with Normal(new_mu, new_sigma)
    quantiles at the same probability levels. Used by EMOS calibration to
    replace the baseline distribution with the calibrated one.
    """
    std_norm = NormalDist(0.0, 1.0)
    quantiles = {}
    for q in distribution.quantiles:
        z = std_norm.inv_cdf(q)
        quantiles[q] = round(new_mu + z * new_sigma, 3)
    return distribution.with_quantiles(quantiles, note=note)
