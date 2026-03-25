"""Bayesian Online Changepoint Detection (BOCPD) utilities.

Phase mapping:
- T2.1 BOCPD prototype

Implements a lightweight Gaussian-mean BOCPD with fixed observation variance.
Designed for online residual streams (e.g., detrended weather residuals).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BOCPDConfig:
    hazard: float = 0.03
    max_run_length: int = 120
    obs_variance: float = 1.0
    mu0: float = 0.0
    kappa0: float = 1.0


class GaussianMeanBOCPD:
    """BOCPD with unknown segment mean and known observation variance.

    References: Adams & MacKay (2007) style run-length recursion.
    """

    def __init__(self, config: BOCPDConfig | None = None) -> None:
        self.cfg = config or BOCPDConfig()
        self.reset()

    @staticmethod
    def _norm_pdf(x: float, mean: np.ndarray, var: np.ndarray) -> np.ndarray:
        var = np.maximum(1e-9, var)
        z = (x - mean) / np.sqrt(var)
        return np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi * var)

    def reset(self) -> None:
        m = int(self.cfg.max_run_length)
        self._run_probs = np.zeros(m + 1, dtype=float)
        self._run_probs[0] = 1.0

        self._mu = np.zeros(m + 1, dtype=float)
        self._kappa = np.zeros(m + 1, dtype=float)
        self._mu[0] = float(self.cfg.mu0)
        self._kappa[0] = float(self.cfg.kappa0)

    def update(self, x: float) -> float:
        """Update with one residual observation.

        Returns changepoint probability p(r_t = 0 | x_1:t).
        """
        m = int(self.cfg.max_run_length)
        x = float(x)

        # Predictive likelihood for each previous run-length state.
        pred_var = float(self.cfg.obs_variance) * (1.0 + 1.0 / np.maximum(1e-9, self._kappa))
        pred = self._norm_pdf(x, self._mu, pred_var)

        # BOCPD recursion:
        # - growth terms use p(x_t | r_{t-1}, x_{t-r:t-1})
        # - changepoint term uses p(x_t | prior), not per-run-length predictive
        h = float(self.cfg.hazard)
        growth = self._run_probs * (1.0 - h) * pred

        prior_pred_var = float(self.cfg.obs_variance) * (1.0 + 1.0 / max(1e-9, float(self.cfg.kappa0)))
        prior_pred = float(self._norm_pdf(x, np.array([float(self.cfg.mu0)]), np.array([prior_pred_var]))[0])
        cp_mass = np.sum(self._run_probs * h) * prior_pred

        new_probs = np.zeros_like(self._run_probs)
        new_probs[0] = cp_mass
        new_probs[1:] = growth[:-1]

        total = float(np.sum(new_probs))
        if total <= 1e-15:
            # Numerical fallback
            new_probs[:] = 0.0
            new_probs[0] = 1.0
        else:
            new_probs /= total

        # Posterior params for each run length after seeing x.
        new_mu = np.zeros_like(self._mu)
        new_kappa = np.zeros_like(self._kappa)

        # New segment (r=0): update prior with first observation.
        k0 = float(self.cfg.kappa0)
        mu0 = float(self.cfg.mu0)
        k0_post = k0 + 1.0
        mu0_post = (k0 * mu0 + x) / k0_post
        new_kappa[0] = k0_post
        new_mu[0] = mu0_post

        # Growth states (r >= 1) come from previous r-1 posterior updated with x.
        prev_k = self._kappa[:-1]
        prev_mu = self._mu[:-1]
        grown_k = prev_k + 1.0
        grown_mu = (prev_k * prev_mu + x) / np.maximum(1e-9, grown_k)

        new_kappa[1:] = grown_k
        new_mu[1:] = grown_mu

        self._run_probs = new_probs
        self._kappa = new_kappa
        self._mu = new_mu

        return float(self._run_probs[0])

    @property
    def run_length_posterior(self) -> np.ndarray:
        return self._run_probs.copy()
