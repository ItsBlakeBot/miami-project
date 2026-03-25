"""DS3M regime dynamics — learnable per-regime transition matrix and state evolution.

Each regime defines drift (mean change per cycle) and sigma (innovation std)
separately for HIGH and LOW remaining-move markets. The transition matrix is
row-stochastic and updated from soft regime posteriors via MLE with a sticky
Dirichlet prior.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from engine.regime_catalog import REGIME_PARAMS, RegimeType

log = logging.getLogger(__name__)

# Canonical ordering of regimes — used for indexing into arrays.
_REGIME_ORDER: list[RegimeType] = [
    RegimeType.MARINE_STABLE,
    RegimeType.INLAND_HEATING,
    RegimeType.CLOUD_SUPPRESSION,
    RegimeType.CONVECTIVE_OUTFLOW,
    RegimeType.FRONTAL_PASSAGE,
    RegimeType.TRANSITION,
]


@dataclass
class RegimeDynamics:
    """Learnable per-regime transition matrix and dynamics parameters."""

    k_regimes: int
    regime_names: list[str]  # e.g. ["marine_stable", "inland_heating", ...]
    transition_matrix: np.ndarray  # (K, K) row-stochastic
    drift_high: np.ndarray  # (K,) mean drift per cycle for HIGH remaining move
    sigma_high: np.ndarray  # (K,) innovation std per cycle for HIGH
    drift_low: np.ndarray  # (K,) mean drift per cycle for LOW remaining move
    sigma_low: np.ndarray  # (K,) innovation std per cycle for LOW

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def from_catalog_defaults(
        cls,
        k_regimes: int = 6,
        self_prob: float = 0.85,
    ) -> RegimeDynamics:
        """Initialize dynamics from regime_catalog.REGIME_PARAMS.

        Transition matrix is diagonal-heavy (self_prob on diagonal, uniform
        off-diagonal).  drift comes from mu_bias, sigma from sigma_multiplier.
        """
        if k_regimes > len(_REGIME_ORDER):
            raise ValueError(
                f"k_regimes={k_regimes} exceeds catalog size {len(_REGIME_ORDER)}"
            )

        regime_types = _REGIME_ORDER[:k_regimes]
        names = [rt.value for rt in regime_types]
        k = k_regimes

        # Transition matrix: self_prob on diagonal, rest uniform
        off_diag = (1.0 - self_prob) / max(k - 1, 1)
        T = np.full((k, k), off_diag)
        np.fill_diagonal(T, self_prob)

        drift_high = np.zeros(k)
        sigma_high = np.ones(k)
        drift_low = np.zeros(k)
        sigma_low = np.ones(k)

        for i, rt in enumerate(regime_types):
            params = REGIME_PARAMS[rt]
            # Use mu_bias as drift (directional tendency)
            drift_high[i] = params.get("mu_bias_high_f", 0.0)
            drift_low[i] = params.get("mu_bias_low_f", 0.0)
            # Use sigma_multiplier as base innovation std
            sigma_high[i] = params.get("sigma_multiplier", 1.0)
            sigma_low[i] = params.get("sigma_multiplier", 1.0)

        log.info(
            "RegimeDynamics initialized from catalog: k=%d, self_prob=%.2f",
            k,
            self_prob,
        )
        return cls(
            k_regimes=k,
            regime_names=names,
            transition_matrix=T,
            drift_high=drift_high,
            sigma_high=sigma_high,
            drift_low=drift_low,
            sigma_low=sigma_low,
        )

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------
    def predict_regime(
        self,
        current_regime_ids: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Sample new regime IDs from transition rows.

        Args:
            current_regime_ids: (N,) int array of current regime indices.
            rng: numpy Generator for reproducibility.

        Returns:
            (N,) int array of new regime indices.
        """
        n = len(current_regime_ids)
        new_ids = np.empty(n, dtype=np.intp)
        for i in range(n):
            row = self.transition_matrix[current_regime_ids[i]]
            new_ids[i] = rng.choice(self.k_regimes, p=row)
        return new_ids

    def predict_state(
        self,
        remaining_high: np.ndarray,
        remaining_low: np.ndarray,
        regime_ids: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Propagate state forward: add per-regime drift + innovation noise.

        Args:
            remaining_high: (N,) current HIGH remaining-move values.
            remaining_low: (N,) current LOW remaining-move values.
            regime_ids: (N,) int regime indices for each particle.
            rng: numpy Generator.

        Returns:
            (new_high, new_low) each (N,).
        """
        n = len(remaining_high)
        new_high = remaining_high.copy()
        new_low = remaining_low.copy()

        for k in range(self.k_regimes):
            mask = regime_ids == k
            count = int(np.sum(mask))
            if count == 0:
                continue
            new_high[mask] += self.drift_high[k] + self.sigma_high[k] * rng.standard_normal(count)
            new_low[mask] += self.drift_low[k] + self.sigma_low[k] * rng.standard_normal(count)

        return new_high, new_low

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------
    def update_transition_from_history(
        self,
        regime_posteriors: list[np.ndarray],
        sticky_kappa: float,
        alpha: float,
    ) -> None:
        """MLE update of transition matrix from soft regime posteriors.

        Uses a sequence of posterior regime probability vectors to accumulate
        expected transition counts, then adds a sticky Dirichlet prior and
        normalizes.

        Args:
            regime_posteriors: list of (K,) probability vectors, one per time step.
            sticky_kappa: extra pseudo-count added to diagonal (self-transition).
            alpha: Dirichlet concentration for all entries.
        """
        if len(regime_posteriors) < 2:
            return

        k = self.k_regimes
        counts = np.full((k, k), alpha)
        # Add sticky prior to diagonal
        np.fill_diagonal(counts, alpha + sticky_kappa)

        # Accumulate soft expected transition counts: E[z_{t-1}=i, z_t=j]
        for t in range(1, len(regime_posteriors)):
            prev = regime_posteriors[t - 1]  # (K,)
            curr = regime_posteriors[t]  # (K,)
            # Outer product gives joint probability under independence assumption
            counts += np.outer(prev, curr)

        # Normalize rows to get row-stochastic matrix
        row_sums = counts.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-12)
        self.transition_matrix = counts / row_sums

        log.debug(
            "Transition matrix updated from %d posteriors, diag=%.3f..%.3f",
            len(regime_posteriors),
            self.transition_matrix.diagonal().min(),
            self.transition_matrix.diagonal().max(),
        )

    # ------------------------------------------------------------------
    # Regime expansion
    # ------------------------------------------------------------------
    def add_regime(
        self,
        name: str,
        drift_high: float,
        sigma_high: float,
        drift_low: float,
        sigma_low: float,
        self_prob: float = 0.85,
    ) -> None:
        """Expand K by 1 — grow all matrices to accommodate a new regime.

        The new regime gets self_prob on its diagonal; existing rows are
        renormalized to accommodate the new column.
        """
        old_k = self.k_regimes
        new_k = old_k + 1

        # Expand transition matrix
        new_T = np.zeros((new_k, new_k))
        # Copy old rows, shrink existing entries to make room for new column
        scale = (1.0 - self_prob / new_k)  # heuristic: small leak to new regime
        new_T[:old_k, :old_k] = self.transition_matrix * scale
        # New column for existing rows
        new_T[:old_k, old_k] = (1.0 - new_T[:old_k, :old_k].sum(axis=1))
        # New row
        off_diag = (1.0 - self_prob) / max(old_k, 1)
        new_T[old_k, :old_k] = off_diag
        new_T[old_k, old_k] = self_prob

        # Renormalize all rows for safety
        row_sums = new_T.sum(axis=1, keepdims=True)
        new_T = new_T / np.maximum(row_sums, 1e-12)

        self.transition_matrix = new_T
        self.drift_high = np.append(self.drift_high, drift_high)
        self.sigma_high = np.append(self.sigma_high, sigma_high)
        self.drift_low = np.append(self.drift_low, drift_low)
        self.sigma_low = np.append(self.sigma_low, sigma_low)
        self.regime_names.append(name)
        self.k_regimes = new_k

        log.info("Added regime '%s' — K now %d", name, new_k)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "k_regimes": self.k_regimes,
            "regime_names": self.regime_names,
            "transition_matrix": self.transition_matrix.tolist(),
            "drift_high": self.drift_high.tolist(),
            "sigma_high": self.sigma_high.tolist(),
            "drift_low": self.drift_low.tolist(),
            "sigma_low": self.sigma_low.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> RegimeDynamics:
        return cls(
            k_regimes=d["k_regimes"],
            regime_names=d["regime_names"],
            transition_matrix=np.array(d["transition_matrix"]),
            drift_high=np.array(d["drift_high"]),
            sigma_high=np.array(d["sigma_high"]),
            drift_low=np.array(d["drift_low"]),
            sigma_low=np.array(d["sigma_low"]),
        )
