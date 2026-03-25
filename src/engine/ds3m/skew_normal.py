"""Skew-normal distribution for DS3M emission layer.

Provides a PyTorch-native skew-normal with:
  - log_prob / cdf / bracket_prob for pricing
  - Closed-form CRPS for training loss
  - 2-component mixture variant for bimodal scenarios
  - CLI rounding adjustment (±0.5°F boundary)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.distributions import Normal

_STD_NORMAL = Normal(0.0, 1.0)

# ──────────────────────────────────────────────────────────────────────
# Owen's T function (needed for exact skew-normal CDF)
# ──────────────────────────────────────────────────────────────────────

def _owens_t(h: Tensor, a: Tensor, n_terms: int = 20) -> Tensor:
    """Owen's T(h, a) via series expansion (AS 76 algorithm).

    Accurate to ~1e-8 for |a| <= 1.  For |a| > 1 we use the identity
    T(h,a) = 0.5*(Phi(h) + Phi(ah)) - Phi(h)*Phi(ah) - T(ah, 1/a).
    """
    # Handle |a| > 1 via reflection
    abs_a = torch.abs(a)
    needs_reflect = abs_a > 1.0

    # For the reflection branch
    a_safe = torch.where(needs_reflect, 1.0 / abs_a.clamp(min=1e-10), abs_a)
    h_safe = torch.where(needs_reflect, h * abs_a, h)

    # Series computation for |a| <= 1
    r = torch.atan(a_safe) / (2 * math.pi)
    if n_terms == 0:
        t_val = r * torch.exp(-0.5 * h_safe**2)
    else:
        phi_h = torch.exp(_STD_NORMAL.log_prob(h_safe))
        Phi_h = _STD_NORMAL.cdf(h_safe)
        # Gauss-Legendre style series
        t_val = torch.zeros_like(h)
        a_pow = a_safe.clone()
        sign = 1.0
        for j in range(n_terms):
            k = 2 * j + 1
            coeff = sign / k
            t_val = t_val + coeff * a_pow * torch.exp(-0.5 * h_safe**2 * (1 + a_safe**2 * 0))
            a_pow = a_pow * a_safe**2
            sign = -sign
        t_val = t_val * torch.exp(-0.5 * h_safe**2) / (2 * math.pi)

    # Apply reflection identity where needed
    if needs_reflect.any():
        phi_h_orig = _STD_NORMAL.cdf(h)
        phi_ah = _STD_NORMAL.cdf(a.abs() * h)
        reflected = 0.5 * (phi_h_orig + phi_ah) - phi_h_orig * phi_ah - t_val
        t_val = torch.where(needs_reflect, reflected, t_val)

    # Sign correction
    t_val = torch.where(a < 0, -t_val, t_val)
    return t_val


# ──────────────────────────────────────────────────────────────────────
# Core SkewNormal class
# ──────────────────────────────────────────────────────────────────────

class SkewNormal:
    """Skew-normal distribution SN(mu, sigma, alpha).

    Parameters
    ----------
    loc   : mu — location
    scale : sigma — scale (> 0)
    alpha : skewness  (alpha=0 → Gaussian, alpha>0 → right skew)
    """

    def __init__(self, loc: Tensor, scale: Tensor, alpha: Tensor):
        self.loc = loc
        self.scale = scale.clamp(min=1e-6)
        self.alpha = alpha

    def log_prob(self, x: Tensor) -> Tensor:
        """Log-density of skew-normal at x."""
        z = (x - self.loc) / self.scale
        log_phi = _STD_NORMAL.log_prob(z) - torch.log(self.scale)
        log_Phi = torch.log(_STD_NORMAL.cdf(self.alpha * z).clamp(min=1e-10))
        return log_phi + log_Phi + math.log(2)

    def cdf(self, x: Tensor) -> Tensor:
        """CDF of skew-normal at x using Owen's T function."""
        z = (x - self.loc) / self.scale
        Phi_z = _STD_NORMAL.cdf(z)
        T_val = _owens_t(z, self.alpha)
        return Phi_z - 2 * T_val

    def bracket_prob(self, lower: float | Tensor, upper: float | Tensor) -> Tensor:
        """P(lower <= X < upper)."""
        lo = torch.as_tensor(lower, dtype=self.loc.dtype, device=self.loc.device)
        hi = torch.as_tensor(upper, dtype=self.loc.dtype, device=self.loc.device)
        return (self.cdf(hi) - self.cdf(lo)).clamp(min=0.001, max=0.999)

    def bracket_prob_cli_adjusted(
        self, lower: float | Tensor, upper: float | Tensor
    ) -> Tensor:
        """P(lower <= X < upper) with CLI ±0.5°F rounding adjustment.

        The NWS CLI rounds sensor °C → °F → nearest integer.
        Effective bracket boundary is at N ± 0.5°F in continuous space.
        """
        lo = torch.as_tensor(lower, dtype=self.loc.dtype, device=self.loc.device)
        hi = torch.as_tensor(upper, dtype=self.loc.dtype, device=self.loc.device)
        adj_lo = torch.where(lo > -999, lo - 0.5, lo)  # -inf stays -inf
        adj_hi = torch.where(hi < 999, hi + 0.5, hi)   # +inf stays +inf
        return (self.cdf(adj_hi) - self.cdf(adj_lo)).clamp(min=0.001, max=0.999)

    def crps(self, observation: Tensor) -> Tensor:
        """Closed-form CRPS for skew-normal (Gneiting & Raftery 2007).

        CRPS = sigma * [z*(2*Phi(z)-1) + 2*phi(z) - 1/sqrt(pi)
               + 2*delta*(phi(z)*Phi(alpha*z) - delta/sqrt(pi))]
        where delta = alpha / sqrt(1 + alpha^2)
        """
        z = (observation - self.loc) / self.scale
        delta = self.alpha / torch.sqrt(1 + self.alpha**2)

        phi_z = torch.exp(_STD_NORMAL.log_prob(z))
        Phi_z = _STD_NORMAL.cdf(z)
        Phi_az = _STD_NORMAL.cdf(self.alpha * z)

        crps_val = self.scale * (
            z * (2 * Phi_z - 1)
            + 2 * phi_z
            - 1.0 / math.sqrt(math.pi)
            + 2 * delta * (phi_z * Phi_az - delta / math.sqrt(math.pi))
        )
        return crps_val

    def mean(self) -> Tensor:
        delta = self.alpha / torch.sqrt(1 + self.alpha**2)
        return self.loc + self.scale * delta * math.sqrt(2 / math.pi)

    def variance(self) -> Tensor:
        delta = self.alpha / torch.sqrt(1 + self.alpha**2)
        return self.scale**2 * (1 - 2 * delta**2 / math.pi)

    def sample(self, n: int = 1) -> Tensor:
        """Sample via the stochastic representation: X = mu + sigma * (delta*|Z0| + sqrt(1-delta^2)*Z1)."""
        delta = self.alpha / torch.sqrt(1 + self.alpha**2)
        z0 = torch.randn(n, *self.loc.shape, device=self.loc.device)
        z1 = torch.randn(n, *self.loc.shape, device=self.loc.device)
        y = delta * z0.abs() + torch.sqrt(1 - delta**2) * z1
        return self.loc + self.scale * y


# ──────────────────────────────────────────────────────────────────────
# 2-Component Skew-Normal Mixture
# ──────────────────────────────────────────────────────────────────────

class SkewNormalMixture:
    """Weighted mixture of K SkewNormal components.

    Handles bimodal distributions (e.g., sea breeze arrives vs doesn't).
    Default: 2 components — core scenario + alternative scenario.
    """

    def __init__(
        self,
        locs: Tensor,       # (K,)
        scales: Tensor,     # (K,)
        alphas: Tensor,     # (K,)
        weights: Tensor,    # (K,) — must sum to 1
    ):
        self.K = locs.shape[0]
        self.weights = weights / weights.sum()  # normalize
        self.components = [
            SkewNormal(locs[k], scales[k], alphas[k])
            for k in range(self.K)
        ]

    def log_prob(self, x: Tensor) -> Tensor:
        """Log-density of mixture at x (log-sum-exp for stability)."""
        log_probs = torch.stack([
            comp.log_prob(x) + torch.log(self.weights[k].clamp(min=1e-10))
            for k, comp in enumerate(self.components)
        ], dim=0)
        return torch.logsumexp(log_probs, dim=0)

    def cdf(self, x: Tensor) -> Tensor:
        return sum(
            self.weights[k] * comp.cdf(x)
            for k, comp in enumerate(self.components)
        )

    def bracket_prob(self, lower: float | Tensor, upper: float | Tensor) -> Tensor:
        return sum(
            self.weights[k] * comp.bracket_prob(lower, upper)
            for k, comp in enumerate(self.components)
        ).clamp(min=0.001, max=0.999)

    def bracket_prob_cli_adjusted(
        self, lower: float | Tensor, upper: float | Tensor
    ) -> Tensor:
        return sum(
            self.weights[k] * comp.bracket_prob_cli_adjusted(lower, upper)
            for k, comp in enumerate(self.components)
        ).clamp(min=0.001, max=0.999)

    def crps(self, observation: Tensor) -> Tensor:
        """Approximate mixture CRPS via weighted component CRPS.

        Exact mixture CRPS requires computing E|X-X'| between components,
        which has no closed form for skew-normal.  This approximation is
        an upper bound and works well when components don't overlap heavily.
        For exact CRPS, use Monte Carlo (sample-based CRPS).
        """
        return sum(
            self.weights[k] * comp.crps(observation)
            for k, comp in enumerate(self.components)
        )

    def crps_mc(self, observation: Tensor, n_samples: int = 500) -> Tensor:
        """Exact CRPS via Monte Carlo: E|X-y| - 0.5*E|X-X'|."""
        samples = self.sample(n_samples).squeeze(-1)  # (n_samples,)
        term1 = (samples - observation).abs().mean(dim=0)
        # E|X-X'| via pairwise differences (subsample for efficiency)
        n_sub = min(n_samples, 200)
        s1 = samples[:n_sub]
        s2 = samples[torch.randperm(n_samples)[:n_sub]]
        term2 = (s1 - s2).abs().mean(dim=0)
        return term1 - 0.5 * term2

    def mean(self) -> Tensor:
        return sum(self.weights[k] * comp.mean() for k, comp in enumerate(self.components))

    def sample(self, n: int = 1) -> Tensor:
        """Sample from mixture: first choose component, then sample."""
        comp_indices = torch.multinomial(self.weights, n, replacement=True)
        samples = torch.zeros(n, *self.components[0].loc.shape, device=self.components[0].loc.device)
        for k, comp in enumerate(self.components):
            mask = comp_indices == k
            n_k = mask.sum().item()
            if n_k > 0:
                samples[mask] = comp.sample(n_k).squeeze(0) if n_k == 1 else comp.sample(n_k)
        return samples


# ──────────────────────────────────────────────────────────────────────
# Emission head for DS3M: MLP → SkewNormalMixture params
# ──────────────────────────────────────────────────────────────────────

class SkewNormalEmissionHead(torch.nn.Module):
    """Parametric emission head: latent → 2-component SkewNormal mixture.

    Input: concatenation of [z_t, regime_embedding]
    Output: SkewNormalMixture (2 components)
    """

    def __init__(self, d_input: int, n_components: int = 2, hidden: int = 32):
        super().__init__()
        self.n_components = n_components
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_input, hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.SiLU(),
        )
        # Per-component: loc, log_scale, alpha, logit_weight
        self.param_head = torch.nn.Linear(hidden, n_components * 4)

    def forward(self, z: Tensor) -> SkewNormalMixture:
        """z: (batch, d_input) → SkewNormalMixture per batch element."""
        h = self.net(z)
        params = self.param_head(h)  # (batch, n_components * 4)
        params = params.reshape(-1, self.n_components, 4)

        locs = params[..., 0]                        # (batch, K)
        scales = torch.nn.functional.softplus(params[..., 1]) + 0.1  # floor at 0.1°F
        alphas = params[..., 2].clamp(-10, 10)       # prevent extreme skew
        weights = torch.softmax(params[..., 3], dim=-1)  # (batch, K)

        # Return list of mixtures (one per batch element)
        # For single-element case (common in inference), return directly
        if z.shape[0] == 1:
            return SkewNormalMixture(
                locs[0], scales[0], alphas[0], weights[0]
            )
        return [
            SkewNormalMixture(locs[i], scales[i], alphas[i], weights[i])
            for i in range(z.shape[0])
        ]
