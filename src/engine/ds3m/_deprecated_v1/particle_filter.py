"""DS3M DIMMPF — Dirichlet-Informed Multi-Model Particle Filter.

Core particle filter with predict-update-resample cycle. All weight
arithmetic is in log-space to prevent underflow with 200 particles.
Resampling uses the systematic algorithm.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from engine.ds3m.config import DS3MConfig
from engine.ds3m.observation_model import ObservationModel
from engine.ds3m.regime_dynamics import RegimeDynamics

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Particle state
# ---------------------------------------------------------------------------
@dataclass
class ParticleState:
    """Snapshot of the particle cloud at one point in time."""

    remaining_high: np.ndarray  # (N,)
    remaining_low: np.ndarray  # (N,)
    regime_ids: np.ndarray  # (N,) int
    log_weights: np.ndarray  # (N,)
    n_particles: int
    k_regimes: int

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------
    @property
    def weights(self) -> np.ndarray:
        """Normalized weights via log-sum-exp."""
        max_lw = np.max(self.log_weights)
        shifted = np.exp(self.log_weights - max_lw)
        return shifted / shifted.sum()

    @property
    def ess(self) -> float:
        """Effective sample size: 1 / sum(w_i^2)."""
        w = self.weights
        return 1.0 / np.sum(w ** 2)

    @property
    def regime_posterior(self) -> np.ndarray:
        """(K,) vector of weighted regime probabilities."""
        w = self.weights
        posterior = np.zeros(self.k_regimes)
        for k in range(self.k_regimes):
            posterior[k] = w[self.regime_ids == k].sum()
        return posterior

    @property
    def weighted_mean_high(self) -> float:
        return float(np.dot(self.weights, self.remaining_high))

    @property
    def weighted_mean_low(self) -> float:
        return float(np.dot(self.weights, self.remaining_low))

    @property
    def weighted_std_high(self) -> float:
        w = self.weights
        mu = np.dot(w, self.remaining_high)
        var = np.dot(w, (self.remaining_high - mu) ** 2)
        return float(np.sqrt(max(var, 0.0)))

    @property
    def weighted_std_low(self) -> float:
        w = self.weights
        mu = np.dot(w, self.remaining_low)
        var = np.dot(w, (self.remaining_low - mu) ** 2)
        return float(np.sqrt(max(var, 0.0)))

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "remaining_high": self.remaining_high.tolist(),
            "remaining_low": self.remaining_low.tolist(),
            "regime_ids": self.regime_ids.tolist(),
            "log_weights": self.log_weights.tolist(),
            "n_particles": self.n_particles,
            "k_regimes": self.k_regimes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ParticleState:
        return cls(
            remaining_high=np.array(d["remaining_high"]),
            remaining_low=np.array(d["remaining_low"]),
            regime_ids=np.array(d["regime_ids"], dtype=np.intp),
            log_weights=np.array(d["log_weights"]),
            n_particles=d["n_particles"],
            k_regimes=d["k_regimes"],
        )


# ---------------------------------------------------------------------------
# Systematic resampling
# ---------------------------------------------------------------------------
def _systematic_resample(
    weights: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Systematic resampling: one uniform U ~ [0, 1/N], then U+k/N.

    Args:
        weights: (N,) normalized weight vector.
        rng: numpy Generator.

    Returns:
        (N,) int array of resampled indices.
    """
    n = len(weights)
    cumsum = np.cumsum(weights)
    # Safety: ensure last entry is exactly 1.0
    cumsum[-1] = 1.0

    u0 = rng.uniform(0.0, 1.0 / n)
    positions = u0 + np.arange(n) / n

    indices = np.empty(n, dtype=np.intp)
    j = 0
    for i in range(n):
        while cumsum[j] < positions[i]:
            j += 1
        indices[i] = j
    return indices


# ---------------------------------------------------------------------------
# DIMMPF
# ---------------------------------------------------------------------------
class DIMMPF:
    """Dirichlet-Informed Multi-Model Particle Filter.

    Shadow-mode filter that runs alongside production inference.
    Maintains a cloud of weighted particles, each carrying a
    (remaining_high, remaining_low, regime_id) state.
    """

    def __init__(
        self,
        config: DS3MConfig,
        dynamics: RegimeDynamics,
        obs_model: ObservationModel,
        rng_seed: int = 42,
    ) -> None:
        self.config = config
        self.dynamics = dynamics
        self.obs_model = obs_model
        self.rng = np.random.default_rng(rng_seed)

    # ------------------------------------------------------------------
    # Initialize
    # ------------------------------------------------------------------
    def initialize(
        self,
        mu_high: float,
        sigma_high: float,
        mu_low: float,
        sigma_low: float,
        regime_probs: np.ndarray | None = None,
    ) -> ParticleState:
        """Create initial particle cloud.

        Samples remaining-move values from N(mu, sigma) per market.
        Regime IDs are drawn from prior probabilities (uniform if None).

        Args:
            mu_high: prior mean for HIGH remaining move.
            sigma_high: prior std for HIGH remaining move.
            mu_low: prior mean for LOW remaining move.
            sigma_low: prior std for LOW remaining move.
            regime_probs: (K,) prior regime probabilities, or None for uniform.

        Returns:
            Initialized ParticleState with uniform log-weights.
        """
        n = self.config.n_particles
        k = self.dynamics.k_regimes

        remaining_high = mu_high + sigma_high * self.rng.standard_normal(n)
        remaining_low = mu_low + sigma_low * self.rng.standard_normal(n)

        if regime_probs is None:
            regime_probs = np.ones(k) / k
        else:
            regime_probs = np.asarray(regime_probs)
            regime_probs = regime_probs / regime_probs.sum()

        regime_ids = self.rng.choice(k, size=n, p=regime_probs)

        # Uniform log-weights: log(1/N) = -log(N)
        log_weights = np.full(n, -np.log(n))

        state = ParticleState(
            remaining_high=remaining_high,
            remaining_low=remaining_low,
            regime_ids=regime_ids,
            log_weights=log_weights,
            n_particles=n,
            k_regimes=k,
        )
        log.info(
            "DIMMPF initialized: N=%d, K=%d, mu_high=%.2f, mu_low=%.2f",
            n,
            k,
            mu_high,
            mu_low,
        )
        return state

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------
    def predict(self, state: ParticleState) -> ParticleState:
        """Propagate particles forward one cycle.

        1. Sample new regime IDs from transition matrix.
        2. Evolve state with per-regime drift + noise.

        Weights are unchanged (prediction does not update weights).
        """
        new_regime_ids = self.dynamics.predict_regime(state.regime_ids, self.rng)
        new_high, new_low = self.dynamics.predict_state(
            state.remaining_high,
            state.remaining_low,
            new_regime_ids,
            self.rng,
        )
        return ParticleState(
            remaining_high=new_high,
            remaining_low=new_low,
            regime_ids=new_regime_ids,
            log_weights=state.log_weights.copy(),
            n_particles=state.n_particles,
            k_regimes=state.k_regimes,
        )

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------
    def update(
        self,
        state: ParticleState,
        obs_remaining_high: float | None,
        obs_remaining_low: float | None,
        source_key: str,
    ) -> ParticleState:
        """Incorporate observation(s) — update weights in log-space.

        Computes Gaussian log-likelihood for each particle and adds to
        existing log-weights. Also updates adaptive R via Sage-Husa.
        """
        new_log_weights = state.log_weights.copy()

        if obs_remaining_high is not None:
            ll_high = self.obs_model.log_likelihood(
                state.remaining_high, obs_remaining_high, source_key
            )
            new_log_weights += ll_high
            # Sage-Husa R update using weighted mean as prediction
            w = state.weights
            pred_high = float(np.dot(w, state.remaining_high))
            self.obs_model.update_r(
                f"{source_key}_high", obs_remaining_high - pred_high
            )

        if obs_remaining_low is not None:
            ll_low = self.obs_model.log_likelihood(
                state.remaining_low, obs_remaining_low, source_key
            )
            new_log_weights += ll_low
            w = state.weights
            pred_low = float(np.dot(w, state.remaining_low))
            self.obs_model.update_r(
                f"{source_key}_low", obs_remaining_low - pred_low
            )

        return ParticleState(
            remaining_high=state.remaining_high.copy(),
            remaining_low=state.remaining_low.copy(),
            regime_ids=state.regime_ids.copy(),
            log_weights=new_log_weights,
            n_particles=state.n_particles,
            k_regimes=state.k_regimes,
        )

    # ------------------------------------------------------------------
    # Resample
    # ------------------------------------------------------------------
    def resample_if_needed(self, state: ParticleState) -> ParticleState:
        """Systematic resample when ESS drops below threshold.

        After resampling, weights are reset to uniform (log(1/N)).
        """
        threshold = self.config.ess_threshold_fraction * state.n_particles
        if state.ess >= threshold:
            return state

        log.debug(
            "Resampling: ESS=%.1f < threshold=%.1f",
            state.ess,
            threshold,
        )

        indices = _systematic_resample(state.weights, self.rng)
        n = state.n_particles

        return ParticleState(
            remaining_high=state.remaining_high[indices],
            remaining_low=state.remaining_low[indices],
            regime_ids=state.regime_ids[indices],
            log_weights=np.full(n, -np.log(n)),
            n_particles=n,
            k_regimes=state.k_regimes,
        )

    # ------------------------------------------------------------------
    # Full cycle
    # ------------------------------------------------------------------
    def step(
        self,
        state: ParticleState,
        obs_remaining_high: float | None,
        obs_remaining_low: float | None,
        source_key: str,
    ) -> ParticleState:
        """Full predict-update-resample cycle.

        Args:
            state: current ParticleState.
            obs_remaining_high: observed HIGH remaining move (None to skip).
            obs_remaining_low: observed LOW remaining move (None to skip).
            source_key: observation source identifier.

        Returns:
            Updated ParticleState.
        """
        state = self.predict(state)
        state = self.update(state, obs_remaining_high, obs_remaining_low, source_key)
        state = self.resample_if_needed(state)
        return state
