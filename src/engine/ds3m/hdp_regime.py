"""HDP-Sticky regime discovery and management for DS3M.

Replaces the rule-based regime_catalog.py with a Bayesian nonparametric
approach that can discover an unbounded number of weather regimes.

Architecture:
  - Hierarchical Dirichlet Process prior → unbounded regime count
  - Sticky kappa → prevents degenerate rapid switching
  - 5 initialized physical regimes with soft-label priors
  - HDP "new table" mechanism for regime birth
  - Physical transition prior as KL regularizer
  - Automatic naming from atmospheric signatures

Reference: Fox et al. 2011 — "Sticky HDP-HMM"
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import torch
from torch import Tensor

log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Physical regime definitions (Miami KMIA)
# ──────────────────────────────────────────────────────────────────────

INITIAL_REGIMES = {
    0: {
        "name": "continental",
        "display_name": "Continental",
        "description": "Light/offshore winds, clear skies, UHI-enhanced heating",
        "color": "#FF6B35",  # orange
        "sigma_multiplier": 1.0,
        "mu_bias_high": 0.0,
        "mu_bias_low": 0.0,
        "position_sizing_mult": 1.0,
        "min_edge_override": None,
    },
    1: {
        "name": "sea_breeze",
        "display_name": "Sea Breeze",
        "description": "Onshore flow, afternoon temp suppression, dew point jump",
        "color": "#004E89",  # blue
        "sigma_multiplier": 0.85,
        "mu_bias_high": -1.5,
        "mu_bias_low": 1.0,
        "position_sizing_mult": 0.7,
        "min_edge_override": None,
    },
    2: {
        "name": "frontal",
        "display_name": "Frontal",
        "description": "Strong gradient, high model disagreement, rapid transitions",
        "color": "#7D3C98",  # purple
        "sigma_multiplier": 1.4,
        "mu_bias_high": -1.0,
        "mu_bias_low": -1.5,
        "position_sizing_mult": 0.5,
        "min_edge_override": 6.0,  # wider minimum edge (cents)
    },
    3: {
        "name": "tropical_moist",
        "display_name": "Tropical/Moist",
        "description": "Deep moisture, CAPE-driven convection, afternoon storms cap temp",
        "color": "#27AE60",  # green
        "sigma_multiplier": 1.2,
        "mu_bias_high": -0.5,
        "mu_bias_low": 0.0,
        "position_sizing_mult": 0.8,
        "min_edge_override": None,
    },
    4: {
        "name": "nocturnal_warm",
        "display_name": "Nocturnal Warm",
        "description": "Low-level jet, overnight warm advection, DST overnight high risk",
        "color": "#C0392B",  # red
        "sigma_multiplier": 1.3,
        "mu_bias_high": 2.0,
        "mu_bias_low": 0.0,
        "position_sizing_mult": 1.2,
        "min_edge_override": None,
    },
}

# Physical transition prior: encodes meteorological constraints
# Higher values = more likely transition
TRANSITION_PRIOR = np.array([
    #  Cont   SeaBr  Front  Trop   Noct
    [0.70,  0.15,  0.10,  0.05,  0.00],  # Continental
    [0.10,  0.65,  0.05,  0.20,  0.00],  # Sea Breeze
    [0.15,  0.05,  0.50,  0.10,  0.20],  # Frontal
    [0.05,  0.15,  0.10,  0.65,  0.05],  # Tropical
    [0.30,  0.00,  0.20,  0.05,  0.45],  # Nocturnal
], dtype=np.float64)

# Color palette for auto-discovered regimes
DISCOVERY_COLORS = [
    "#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6",
    "#1ABC9C", "#E67E22", "#34495E", "#16A085", "#D35400",
]


# ──────────────────────────────────────────────────────────────────────
# Regime metadata
# ──────────────────────────────────────────────────────────────────────

@dataclass
class RegimeInfo:
    """Metadata for a single regime (physical or discovered)."""
    id: int
    name: str
    display_name: str
    description: str
    color: str
    is_physical: bool  # True for the 5 initialized regimes
    birth_date: str | None = None  # ISO format, when discovered
    n_observations: int = 0
    sigma_multiplier: float = 1.0
    mu_bias_high: float = 0.0
    mu_bias_low: float = 0.0
    position_sizing_mult: float = 0.6  # conservative for new regimes
    min_edge_override: float | None = None
    characteristic_features: dict = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────
# HDP-Sticky Regime Manager
# ──────────────────────────────────────────────────────────────────────

class HDPRegimeManager:
    """Manages weather regimes via HDP-Sticky HMM dynamics.

    Key mechanisms:
      1. Sticky self-transition: kappa biases toward staying in current regime
      2. HDP "new table": probability mass on creating a new regime
      3. Physical prior: KL regularization toward known transition constraints
      4. Auto-naming: atmospheric signature → descriptive name
      5. Merge detection: regimes with similar dynamics get merged
    """

    def __init__(
        self,
        gamma: float = 2.0,        # HDP concentration (controls # of global regimes)
        alpha0: float = 5.0,       # DP concentration per regime's transition row
        kappa: float = 50.0,       # sticky self-transition bonus
        transition_prior_weight: float = 0.05,  # KL regularization strength
        merge_threshold: float = 0.3,  # dynamics similarity for merge
        merge_min_cooccurrence: int = 100,  # min cycles before merge check
    ):
        self.gamma = gamma
        self.alpha0 = alpha0
        self.kappa = kappa
        self.transition_prior_weight = transition_prior_weight
        self.merge_threshold = merge_threshold
        self.merge_min_cooccurrence = merge_min_cooccurrence

        # Initialize with physical regimes
        self.regimes: dict[int, RegimeInfo] = {}
        self.K = 0  # current number of active regimes
        self._next_id = 0

        for rid, params in INITIAL_REGIMES.items():
            self.regimes[rid] = RegimeInfo(
                id=rid,
                name=params["name"],
                display_name=params["display_name"],
                description=params["description"],
                color=params["color"],
                is_physical=True,
                sigma_multiplier=params["sigma_multiplier"],
                mu_bias_high=params["mu_bias_high"],
                mu_bias_low=params["mu_bias_low"],
                position_sizing_mult=params["position_sizing_mult"],
                min_edge_override=params["min_edge_override"],
            )
            self.K += 1
            self._next_id = max(self._next_id, rid + 1)

        # Transition matrix: (K, K) — learnable
        self.transition_matrix = np.array(TRANSITION_PRIOR, dtype=np.float64)
        # Add sticky mass to diagonal
        self._apply_sticky()

        # Global beta (HDP base measure weights)
        self.beta = np.ones(self.K, dtype=np.float64) / self.K

        # Co-occurrence tracking for merge detection
        self.cooccurrence = np.zeros((self.K, self.K), dtype=np.int64)

        # HDP "new table" tracking
        self._low_likelihood_streak = 0
        self._new_table_threshold = 3  # consecutive bad cycles to trigger birth
        self._new_table_avg_ll = 0.0

    def _apply_sticky(self):
        """Add sticky kappa to transition matrix diagonal."""
        K = self.transition_matrix.shape[0]
        for k in range(K):
            row = self.transition_matrix[k]
            row_sum = row.sum()
            if row_sum > 0:
                self.transition_matrix[k] = (
                    self.alpha0 * self.beta[:K] + self.kappa * (np.arange(K) == k)
                )
                self.transition_matrix[k] /= self.transition_matrix[k].sum()

    # ── Regime Transition ──────────────────────────────────────────

    def sample_transition(self, current_regime: int) -> int:
        """Sample next regime from transition row (with new-table probability)."""
        K = self.K
        row = self.transition_matrix[current_regime].copy()

        # HDP new table probability
        p_new = self.gamma / (self.gamma + self.alpha0 + self.kappa)
        row = row * (1 - p_new)

        # Sample
        r = np.random.random()
        if r < p_new:
            return -1  # signal for new regime birth
        cumsum = np.cumsum(row)
        idx = np.searchsorted(cumsum, r - p_new)
        return min(idx, K - 1)

    def get_transition_row(self, current_regime: int) -> np.ndarray:
        """Get transition probabilities (including new-table mass)."""
        row = self.transition_matrix[current_regime].copy()
        p_new = self.gamma / (self.gamma + self.alpha0 + self.kappa)
        return row * (1 - p_new)

    # ── Soft-Label Prior ───────────────────────────────────────────

    def compute_regime_prior(self, features: dict[str, float]) -> np.ndarray:
        """Compute soft prior over regimes from atmospheric features.

        Used to initialize/regularize the discrete latent variable.
        Light touch — biases toward physical regimes but lets model refine.
        """
        priors = np.zeros(self.K, dtype=np.float64)

        wd = features.get("wind_dir_10m", 0)
        ws = features.get("wind_speed_10m", 0)
        dd = features.get("dewpoint_change_1h", 0)
        cape = features.get("cape", 0)
        spread = features.get("model_spread", 0)
        hour = features.get("hour_local", 12)
        w925 = features.get("wind_925_speed", 0)
        w925d = features.get("wind_925_dir", 0)

        # Continental: offshore (W/NW/N) or light winds
        if self.K > 0:
            if (270 <= wd <= 360 or 0 <= wd <= 45) or ws < 5:
                priors[0] += 2.0

        # Sea Breeze: onshore (E/SE/S), dewpoint jump, afternoon
        if self.K > 1:
            if 90 <= wd <= 210 and dd > 1.5 and 12 <= hour <= 18:
                priors[1] += 3.0
            elif 90 <= wd <= 210 and ws > 8:
                priors[1] += 1.5

        # Frontal: high model spread, strong winds
        if self.K > 2:
            if spread > 4.0:
                priors[2] += 2.5
            if ws > 15:
                priors[2] += 1.0

        # Tropical/Moist: high CAPE
        if self.K > 3:
            if cape > 1000:
                priors[3] += 2.0
            if cape > 500 and 90 <= wd <= 210:
                priors[3] += 1.0

        # Nocturnal Warm: strong 925hPa southerly, nighttime
        if self.K > 4:
            if 150 <= w925d <= 240 and w925 > 20 and (hour >= 20 or hour <= 6):
                priors[4] += 3.0
            elif 150 <= w925d <= 240 and w925 > 15:
                priors[4] += 1.0

        # Softmax
        priors = np.exp(priors - priors.max())
        return priors / priors.sum()

    # ── Regime Birth (HDP New Table) ───────────────────────────────

    def check_new_regime(self, avg_log_likelihood: float) -> RegimeInfo | None:
        """Check if a new regime should be born.

        Triggered when the particle filter's average log-likelihood is
        persistently low, meaning no existing regime explains the data.
        """
        if avg_log_likelihood < -5.0:
            self._low_likelihood_streak += 1
            self._new_table_avg_ll = (
                0.7 * self._new_table_avg_ll + 0.3 * avg_log_likelihood
            )
        else:
            self._low_likelihood_streak = 0
            self._new_table_avg_ll = 0.0

        if self._low_likelihood_streak >= self._new_table_threshold:
            self._low_likelihood_streak = 0
            return self._birth_regime()
        return None

    def _birth_regime(self) -> RegimeInfo:
        """Create a new regime with conservative defaults."""
        new_id = self._next_id
        self._next_id += 1

        color = DISCOVERY_COLORS[new_id % len(DISCOVERY_COLORS)]
        placeholder_name = f"discovered_{new_id}"

        regime = RegimeInfo(
            id=new_id,
            name=placeholder_name,
            display_name=f"Regime {new_id} (New)",
            description="Auto-discovered regime — awaiting characterization",
            color=color,
            is_physical=False,
            birth_date=datetime.utcnow().isoformat(),
            sigma_multiplier=1.2,  # conservative: wide sigma
            mu_bias_high=0.0,
            mu_bias_low=0.0,
            position_sizing_mult=0.4,  # very conservative sizing for unknown regimes
        )

        self.regimes[new_id] = regime
        self.K += 1

        # Expand transition matrix
        old_T = self.transition_matrix
        new_T = np.zeros((self.K, self.K), dtype=np.float64)
        new_T[:self.K-1, :self.K-1] = old_T
        # New regime: 85% self, redistribute rest
        new_T[self.K-1, self.K-1] = 0.85
        for k in range(self.K - 1):
            new_T[self.K-1, k] = 0.15 / (self.K - 1)
            # Steal a bit from existing regimes' self-transition for new regime
            new_T[k, self.K-1] = 0.03
            new_T[k, :] /= new_T[k, :].sum()
        new_T[self.K-1, :] /= new_T[self.K-1, :].sum()
        self.transition_matrix = new_T

        # Expand beta
        self.beta = np.append(self.beta, self.gamma / (self.gamma + self.K))
        self.beta /= self.beta.sum()

        # Expand co-occurrence
        old_co = self.cooccurrence
        new_co = np.zeros((self.K, self.K), dtype=np.int64)
        new_co[:self.K-1, :self.K-1] = old_co
        self.cooccurrence = new_co

        log.info(
            f"HDP regime birth: id={new_id}, name='{placeholder_name}', "
            f"total regimes={self.K}"
        )
        return regime

    # ── Regime Naming ──────────────────────────────────────────────

    def name_regime(
        self,
        regime_id: int,
        characteristic_obs: dict[str, float],
        force_name: str | None = None,
    ) -> str:
        """Generate a descriptive name from atmospheric signature.

        Falls back to heuristic naming if no LLM is available.
        """
        if force_name:
            self.regimes[regime_id].name = force_name
            self.regimes[regime_id].display_name = force_name.replace("_", " ").title()
            return force_name

        # Heuristic naming from atmospheric signature
        parts = []
        temp = characteristic_obs.get("temp_f", 80)
        dew = characteristic_obs.get("dew_point_f", 70)
        wind_speed = characteristic_obs.get("wind_speed_kt", 5)
        wind_dir = characteristic_obs.get("wind_dir", 0)
        cape = characteristic_obs.get("cape", 0)

        if temp > 88:
            parts.append("hot")
        elif temp < 70:
            parts.append("cool")

        if dew > 72:
            parts.append("humid")
        elif dew < 55:
            parts.append("dry")

        if wind_speed > 15:
            parts.append("windy")
        elif wind_speed < 3:
            parts.append("calm")

        if 45 <= wind_dir <= 135:
            parts.append("onshore")
        elif 225 <= wind_dir <= 315:
            parts.append("offshore")

        if cape > 1500:
            parts.append("unstable")

        if not parts:
            parts.append("transitional")

        name = f"discovered_{'_'.join(parts)}"
        self.regimes[regime_id].name = name
        self.regimes[regime_id].display_name = " ".join(p.title() for p in parts)
        self.regimes[regime_id].characteristic_features = characteristic_obs

        log.info(f"Regime {regime_id} named: '{name}'")
        return name

    # ── Regime Merge ───────────────────────────────────────────────

    def update_cooccurrence(self, regime_posterior: np.ndarray):
        """Track which regimes co-occur (both have >20% posterior)."""
        K = min(len(regime_posterior), self.K)
        active = np.where(regime_posterior[:K] > 0.2)[0]
        for i in active:
            for j in active:
                if i != j and i < self.K and j < self.K:
                    self.cooccurrence[i, j] += 1

    def check_merge(
        self,
        drift_high: np.ndarray,
        sigma_high: np.ndarray,
        drift_low: np.ndarray,
        sigma_low: np.ndarray,
    ) -> tuple[int, int] | None:
        """Check if any two regimes should be merged.

        Merge when:
          - Co-occurrence > merge_min_cooccurrence
          - Dynamics are similar (drift diff < threshold, sigma diff < threshold)
          - Neither is a core physical regime being merged into a discovered one
        """
        for i in range(self.K):
            for j in range(i + 1, self.K):
                if self.cooccurrence[i, j] < self.merge_min_cooccurrence:
                    continue
                if i >= len(drift_high) or j >= len(drift_high):
                    continue

                drift_diff = abs(drift_high[i] - drift_high[j]) + abs(drift_low[i] - drift_low[j])
                sigma_diff = abs(sigma_high[i] - sigma_high[j]) + abs(sigma_low[i] - sigma_low[j])

                if drift_diff < self.merge_threshold and sigma_diff < self.merge_threshold:
                    # Don't merge two physical regimes
                    ri = self.regimes.get(i)
                    rj = self.regimes.get(j)
                    if ri and rj and ri.is_physical and rj.is_physical:
                        continue
                    # Merge j into i (keep whichever is physical)
                    keep, remove = (i, j) if (ri and ri.is_physical) else (j, i)
                    log.info(
                        f"Regime merge: {remove} → {keep} "
                        f"(co-occurrence={self.cooccurrence[i, j]}, "
                        f"drift_diff={drift_diff:.3f}, sigma_diff={sigma_diff:.3f})"
                    )
                    return (keep, remove)
        return None

    def execute_merge(self, keep: int, remove: int):
        """Execute a regime merge: absorb 'remove' into 'keep'."""
        if remove in self.regimes:
            keep_info = self.regimes[keep]
            remove_info = self.regimes[remove]
            keep_info.n_observations += remove_info.n_observations
            del self.regimes[remove]

        # Note: transition matrix and dynamics arrays must be updated by caller
        # (RegimeDynamics.merge_regime handles the array manipulation)

    # ── Transition Matrix Training ─────────────────────────────────

    def update_transition_matrix(self, regime_posterior_history: list[np.ndarray]):
        """Update transition matrix via MLE + Dirichlet + sticky prior.

        Uses soft counts from regime posteriors (not hard assignments).
        """
        if len(regime_posterior_history) < 10:
            return

        K = self.K
        counts = np.zeros((K, K), dtype=np.float64)

        for t in range(1, len(regime_posterior_history)):
            prev = regime_posterior_history[t - 1]
            curr = regime_posterior_history[t]
            K_t = min(len(prev), len(curr), K)
            for i in range(K_t):
                for j in range(K_t):
                    counts[i, j] += prev[i] * curr[j]

        # Add Dirichlet + sticky prior
        for k in range(K):
            counts[k, :] += self.alpha0 * self.beta[:K]
            counts[k, k] += self.kappa

        # Normalize to get transition matrix
        for k in range(K):
            row_sum = counts[k, :].sum()
            if row_sum > 0:
                self.transition_matrix[k, :K] = counts[k, :K] / row_sum

        # KL regularization toward physical prior (for physical regimes only)
        n_physical = min(5, K)
        for k in range(n_physical):
            for j in range(n_physical):
                learned = self.transition_matrix[k, j]
                prior = TRANSITION_PRIOR[k, j]
                # Soft pull toward prior
                self.transition_matrix[k, j] = (
                    (1 - self.transition_prior_weight) * learned
                    + self.transition_prior_weight * prior
                )
            # Renormalize
            self.transition_matrix[k, :K] /= self.transition_matrix[k, :K].sum()

    # ── Serialization ──────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "K": self.K,
            "gamma": self.gamma,
            "alpha0": self.alpha0,
            "kappa": self.kappa,
            "transition_matrix": self.transition_matrix.tolist(),
            "beta": self.beta.tolist(),
            "cooccurrence": self.cooccurrence.tolist(),
            "next_id": self._next_id,
            "regimes": {
                str(k): {
                    "id": v.id, "name": v.name, "display_name": v.display_name,
                    "description": v.description, "color": v.color,
                    "is_physical": v.is_physical, "birth_date": v.birth_date,
                    "n_observations": v.n_observations,
                    "sigma_multiplier": v.sigma_multiplier,
                    "mu_bias_high": v.mu_bias_high, "mu_bias_low": v.mu_bias_low,
                    "position_sizing_mult": v.position_sizing_mult,
                    "min_edge_override": v.min_edge_override,
                    "characteristic_features": v.characteristic_features,
                }
                for k, v in self.regimes.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> HDPRegimeManager:
        mgr = cls(
            gamma=data.get("gamma", 2.0),
            alpha0=data.get("alpha0", 5.0),
            kappa=data.get("kappa", 50.0),
        )
        mgr.K = data["K"]
        mgr.transition_matrix = np.array(data["transition_matrix"], dtype=np.float64)
        mgr.beta = np.array(data["beta"], dtype=np.float64)
        mgr.cooccurrence = np.array(data["cooccurrence"], dtype=np.int64)
        mgr._next_id = data.get("next_id", mgr.K)

        mgr.regimes = {}
        for k_str, v in data.get("regimes", {}).items():
            rid = int(k_str)
            mgr.regimes[rid] = RegimeInfo(
                id=v["id"], name=v["name"], display_name=v["display_name"],
                description=v["description"], color=v["color"],
                is_physical=v["is_physical"], birth_date=v.get("birth_date"),
                n_observations=v.get("n_observations", 0),
                sigma_multiplier=v.get("sigma_multiplier", 1.0),
                mu_bias_high=v.get("mu_bias_high", 0.0),
                mu_bias_low=v.get("mu_bias_low", 0.0),
                position_sizing_mult=v.get("position_sizing_mult", 0.6),
                min_edge_override=v.get("min_edge_override"),
                characteristic_features=v.get("characteristic_features", {}),
            )
        return mgr

    # ── Dashboard helpers ──────────────────────────────────────────

    def get_regime_display(self, regime_posterior: np.ndarray) -> dict:
        """Get dashboard-ready regime information."""
        K = min(len(regime_posterior), self.K)
        top_regime = int(np.argmax(regime_posterior[:K]))
        info = self.regimes.get(top_regime)

        return {
            "active_regime": info.display_name if info else f"Regime {top_regime}",
            "active_regime_name": info.name if info else f"regime_{top_regime}",
            "color": info.color if info else "#999999",
            "confidence": float(regime_posterior[top_regime]) if top_regime < len(regime_posterior) else 0.0,
            "regime_probs": {
                self.regimes[k].display_name: float(regime_posterior[k])
                for k in range(K)
                if k in self.regimes and k < len(regime_posterior)
            },
            "n_active_regimes": self.K,
            "is_new_discovery": info is not None and not info.is_physical,
        }
