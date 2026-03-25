"""Layer 2: Weighted empirical CDF from particle cloud to bracket probabilities.

Converts the DIMMPF particle cloud into P(YES) for each Kalshi bracket
using a weighted empirical CDF.  Optionally provides smoothed KDE
density points for visualization.
"""

from __future__ import annotations

import logging

import numpy as np
from numpy import ndarray

from engine.bracket_pricer import Bracket

log = logging.getLogger(__name__)


class ParticleDensityEstimator:
    """Weighted empirical CDF from particle cloud -> bracket P(YES)."""

    def __init__(self, bandwidth_f: float = 0.5):
        self.bandwidth_f = bandwidth_f

    # ------------------------------------------------------------------
    # Core CDF
    # ------------------------------------------------------------------

    def weighted_ecdf(self, particles: ndarray, weights: ndarray, x: float) -> float:
        """P(X <= x) = sum of weights where particle <= x."""
        mask = particles <= x
        return float(np.sum(weights[mask]))

    # ------------------------------------------------------------------
    # Single bracket
    # ------------------------------------------------------------------

    def bracket_probability(
        self,
        particles: ndarray,
        weights: ndarray,
        bracket: Bracket,
        truncation_lower: float | None = None,
        truncation_upper: float | None = None,
    ) -> float | None:
        """P(YES) for a single bracket.  Handles under/over/range.

        Parameters
        ----------
        particles : 1-D array of particle values (deg F).
        weights   : 1-D array of normalised particle weights (sum to 1).
        bracket   : Bracket with floor_f, ceiling_f, directional.
        truncation_lower / truncation_upper : optional physical bounds
            (e.g. running high for high markets).  Particles outside
            the window are dropped and weights renormalised.
        """
        p, w = self._truncate(particles, weights, truncation_lower, truncation_upper)
        if len(p) == 0:
            log.warning("No particles remain after truncation for %s", bracket.ticker)
            return None

        if bracket.directional == "under":
            if bracket.ceiling_f is None:
                return None
            return self.weighted_ecdf(p, w, bracket.ceiling_f)

        if bracket.directional == "over":
            if bracket.floor_f is None:
                return None
            return 1.0 - self.weighted_ecdf(p, w, bracket.floor_f)

        # Range bracket: P(floor < X <= ceiling)
        if bracket.floor_f is None or bracket.ceiling_f is None:
            return None
        cdf_hi = self.weighted_ecdf(p, w, bracket.ceiling_f)
        cdf_lo = self.weighted_ecdf(p, w, bracket.floor_f)
        return max(0.0, cdf_hi - cdf_lo)

    # ------------------------------------------------------------------
    # All brackets at once
    # ------------------------------------------------------------------

    def all_bracket_probabilities(
        self,
        particles_high: ndarray,
        particles_low: ndarray,
        weights: ndarray,
        brackets: list[Bracket],
        running_high_f: float | None,
        running_low_f: float | None,
    ) -> dict[str, float]:
        """Compute P(YES) for every bracket.

        Parameters
        ----------
        particles_high : particles for high-temperature markets.
        particles_low  : particles for low-temperature markets.
        weights        : shared particle weights (normalised).
        brackets       : all active Bracket objects.
        running_high_f : current running high (truncation lower bound for highs).
        running_low_f  : current running low (truncation upper bound for lows).
        """
        probs: dict[str, float] = {}
        for b in brackets:
            if b.market_type == "high":
                p = self.bracket_probability(
                    particles_high, weights, b,
                    truncation_lower=running_high_f,
                    truncation_upper=None,
                )
            elif b.market_type == "low":
                p = self.bracket_probability(
                    particles_low, weights, b,
                    truncation_lower=None,
                    truncation_upper=running_low_f,
                )
            else:
                p = self.bracket_probability(particles_high, weights, b)

            if p is not None:
                probs[b.ticker] = p
            else:
                log.debug("Skipped bracket %s — no probability", b.ticker)
        return probs

    # ------------------------------------------------------------------
    # Smoothed KDE density (for visualization)
    # ------------------------------------------------------------------

    def density_points(
        self, particles: ndarray, weights: ndarray, n_grid: int = 500,
    ) -> list[tuple[float, float]]:
        """Smoothed KDE density for visualization.

        Returns list of (x, density) tuples over a grid spanning the
        particle range +/- 3 * bandwidth.
        """
        lo = float(np.min(particles)) - 3.0 * self.bandwidth_f
        hi = float(np.max(particles)) + 3.0 * self.bandwidth_f
        grid = np.linspace(lo, hi, n_grid)

        # Gaussian kernel: K(u) = exp(-0.5 * u^2) / sqrt(2*pi)
        h = self.bandwidth_f
        density = np.zeros(n_grid)
        for xi, wi in zip(particles, weights):
            u = (grid - xi) / h
            density += wi * np.exp(-0.5 * u * u)
        density /= h * np.sqrt(2.0 * np.pi)

        return [(float(x), float(d)) for x, d in zip(grid, density)]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _truncate(
        particles: ndarray,
        weights: ndarray,
        lower: float | None,
        upper: float | None,
    ) -> tuple[ndarray, ndarray]:
        """Drop particles outside [lower, upper] and renormalise weights."""
        mask = np.ones(len(particles), dtype=bool)
        if lower is not None:
            mask &= particles >= lower
        if upper is not None:
            mask &= particles <= upper

        p = particles[mask]
        w = weights[mask]
        total = w.sum()
        if total > 0:
            w = w / total
        return p, w
