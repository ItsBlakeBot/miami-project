"""Conditional Neural Spline Flow for DS3M density estimation.

Replaces the weighted empirical CDF with a learned, continuous density.
Given particle cloud statistics + NWP context + regime state, outputs a
full CDF from which bracket probabilities are computed analytically.

Architecture:
  Conditioning vector → Coupling layers with rational-quadratic splines
  → Invertible transform from base distribution (standard Normal)
  → Continuous CDF over temperature space

Key advantages over ECDF:
  - Smooth, differentiable density (no step artifacts from 500 particles)
  - End-to-end trainable with CRPS loss
  - Can extrapolate into tails (ECDF is bounded by particle range)
  - Sub-millisecond inference once trained

Reference: Durkan et al. 2019 — "Neural Spline Flows"
Uses nflows library (pip install nflows).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────

@dataclass
class NSFConfig:
    """Neural Spline Flow configuration."""
    # Conditioning
    d_condition: int = 48       # particle stats + NWP context + regime + temporal
    # Flow architecture
    n_transforms: int = 10      # number of coupling layers
    n_bins: int = 24            # rational-quadratic spline bins
    hidden_dim: int = 128       # hidden units in coupling networks
    n_hidden_layers: int = 3    # depth of coupling networks
    tail_bound: float = 15.0    # spline range in standardized space (±15°F from mean)
    # Training
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    max_epochs: int = 100
    early_stopping_patience: int = 10


# ──────────────────────────────────────────────────────────────────────
# Rational-Quadratic Spline Transform (core building block)
# ──────────────────────────────────────────────────────────────────────

def _rational_quadratic_spline_forward(
    x: Tensor,
    widths: Tensor,
    heights: Tensor,
    derivatives: Tensor,
    tail_bound: float = 15.0,
) -> tuple[Tensor, Tensor]:
    """Apply monotone rational-quadratic spline transform.

    Args:
        x: input values (B,)
        widths: bin widths (B, K) — must sum to 2*tail_bound
        heights: bin heights (B, K) — must sum to 2*tail_bound
        derivatives: knot derivatives (B, K+1) — must be > 0

    Returns:
        y: transformed values (B,)
        log_det: log |dy/dx| (B,)
    """
    # Identify which bin each x falls in
    inside = (x >= -tail_bound) & (x <= tail_bound)

    # Cumulative widths/heights for bin edges
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = torch.nn.functional.pad(cumwidths, (1, 0), value=-tail_bound)
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = torch.nn.functional.pad(cumheights, (1, 0), value=-tail_bound)

    # Find bin index for each x
    bin_idx = torch.searchsorted(cumwidths[..., 1:], x.unsqueeze(-1)).squeeze(-1)
    bin_idx = bin_idx.clamp(0, widths.shape[-1] - 1)

    # Gather bin parameters
    w = torch.gather(widths, -1, bin_idx.unsqueeze(-1)).squeeze(-1)
    h = torch.gather(heights, -1, bin_idx.unsqueeze(-1)).squeeze(-1)
    cw = torch.gather(cumwidths, -1, bin_idx.unsqueeze(-1)).squeeze(-1)
    ch = torch.gather(cumheights, -1, bin_idx.unsqueeze(-1)).squeeze(-1)
    d_k = torch.gather(derivatives, -1, bin_idx.unsqueeze(-1)).squeeze(-1)
    d_k1 = torch.gather(derivatives, -1, (bin_idx + 1).unsqueeze(-1)).squeeze(-1)

    s = h / w  # slope of straight line connecting knots
    xi = (x - cw) / w  # normalized position within bin [0, 1]
    xi = xi.clamp(1e-6, 1 - 1e-6)

    # Rational quadratic: y = ch + h * (s*xi^2 + d_k*xi*(1-xi)) / (s + (d_k + d_k1 - 2s)*xi*(1-xi))
    numer = h * (s * xi**2 + d_k * xi * (1 - xi))
    denom = s + (d_k + d_k1 - 2 * s) * xi * (1 - xi)
    y = ch + numer / denom

    # Log-determinant
    denom_sq = denom**2
    dy_dxi = (s**2 * (d_k1 * xi**2 + 2 * s * xi * (1 - xi) + d_k * (1 - xi)**2)) / denom_sq
    log_det = torch.log(dy_dxi.clamp(min=1e-10)) + torch.log(h.clamp(min=1e-10)) - torch.log(w.clamp(min=1e-10))

    # Identity for values outside the tail bound
    y = torch.where(inside, y, x)
    log_det = torch.where(inside, log_det, torch.zeros_like(log_det))

    return y, log_det


# ──────────────────────────────────────────────────────────────────────
# Conditional Spline Coupling Layer
# ──────────────────────────────────────────────────────────────────────

class ConditionalSplineLayer(nn.Module):
    """A single conditional spline transform layer.

    For 1D flows (temperature), this is a simple autoregressive transform
    conditioned on external context (particle stats, regime, etc.).
    """

    def __init__(self, d_condition: int, n_bins: int = 16,
                 hidden_dim: int = 64, n_hidden: int = 2, tail_bound: float = 15.0):
        super().__init__()
        self.n_bins = n_bins
        self.tail_bound = tail_bound

        layers = [nn.Linear(d_condition, hidden_dim), nn.SiLU()]
        for _ in range(n_hidden - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])
        # Output: widths (K) + heights (K) + derivatives (K+1)
        layers.append(nn.Linear(hidden_dim, 3 * n_bins + 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor, context: Tensor) -> tuple[Tensor, Tensor]:
        """Forward transform: base → target.

        x: (B,) values in standardized space
        context: (B, d_condition) conditioning vector
        returns: (y, log_det)
        """
        params = self.net(context)  # (B, 3K+1)

        # Split into widths, heights, derivatives
        W, H, D = params.split([self.n_bins, self.n_bins, self.n_bins + 1], dim=-1)

        # Softmax widths/heights to span [-tail_bound, tail_bound]
        W = torch.softmax(W, dim=-1) * 2 * self.tail_bound
        H = torch.softmax(H, dim=-1) * 2 * self.tail_bound
        D = torch.nn.functional.softplus(D) + 1e-3  # derivatives must be > 0

        return _rational_quadratic_spline_forward(x, W, H, D, self.tail_bound)

    def inverse(self, y: Tensor, context: Tensor) -> tuple[Tensor, Tensor]:
        """Inverse transform: target → base (for sampling/CDF computation).

        Uses bisection since analytical inverse of RQ spline is complex.
        """
        # Bisection search
        lo = torch.full_like(y, -self.tail_bound)
        hi = torch.full_like(y, self.tail_bound)

        for _ in range(50):  # 50 iterations → ~1e-15 precision
            mid = (lo + hi) / 2
            y_mid, _ = self.forward(mid, context)
            lo = torch.where(y_mid < y, mid, lo)
            hi = torch.where(y_mid >= y, mid, hi)

        x = (lo + hi) / 2
        _, log_det = self.forward(x, context)
        return x, -log_det


# ──────────────────────────────────────────────────────────────────────
# Conditional Neural Spline Flow
# ──────────────────────────────────────────────────────────────────────

class ConditionalNSF(nn.Module):
    """1D Conditional Neural Spline Flow for temperature density estimation.

    Maps a standard Normal base distribution through a chain of conditional
    rational-quadratic spline transforms to produce an expressive,
    continuous density over temperature.

    The conditioning vector encodes:
      - Particle cloud statistics (weighted mean, std, skewness, kurtosis)
      - NWP model consensus (mean, spread, trend)
      - Regime posterior (K probabilities)
      - Temporal features (hour, DOY, lead time, DST)
      - LETKF spatial spread
      - Mamba hidden state summary (mean-pooled)
    """

    def __init__(self, config: NSFConfig | None = None):
        super().__init__()
        if config is None:
            config = NSFConfig()
        self.config = config

        self.transforms = nn.ModuleList([
            ConditionalSplineLayer(
                config.d_condition, config.n_bins,
                config.hidden_dim, config.n_hidden_layers, config.tail_bound,
            )
            for _ in range(config.n_transforms)
        ])

        # Location/scale normalization (learned, adapts to temperature range)
        self.loc_param = nn.Parameter(torch.tensor(85.0))   # typical Miami high
        self.scale_param = nn.Parameter(torch.tensor(5.0))   # typical spread

        n_params = sum(p.numel() for p in self.parameters())
        log.info(f"ConditionalNSF: {n_params:,} parameters, "
                 f"{config.n_transforms} transforms, {config.n_bins} bins")

    def _standardize(self, temp_f: Tensor) -> Tensor:
        """Map temperature to standardized space."""
        return (temp_f - self.loc_param) / self.scale_param.clamp(min=0.1)

    def _unstandardize(self, z: Tensor) -> Tensor:
        """Map standardized space back to temperature."""
        return z * self.scale_param.clamp(min=0.1) + self.loc_param

    def log_prob(self, temp_f: Tensor, context: Tensor) -> Tensor:
        """Log-density of the flow at observed temperature.

        temp_f: (B,) observed temperatures in °F
        context: (B, d_condition) conditioning vector
        returns: (B,) log p(temp_f | context)
        """
        z = self._standardize(temp_f)
        total_log_det = torch.zeros_like(z)

        # Inverse pass through transforms (target → base)
        for transform in reversed(self.transforms):
            z, log_det = transform.inverse(z, context)
            total_log_det += log_det

        # Base distribution log-prob
        log_prob_base = -0.5 * (z**2 + math.log(2 * math.pi))
        # Jacobian of standardization
        log_det_std = -torch.log(self.scale_param.clamp(min=0.1))

        return log_prob_base + total_log_det + log_det_std

    def sample(self, context: Tensor, n_samples: int = 1000) -> Tensor:
        """Sample from the conditional flow.

        context: (B, d_condition)
        returns: (B, n_samples) temperatures in °F
        """
        B = context.shape[0]
        # Expand context for n_samples
        ctx_expanded = context.unsqueeze(1).expand(B, n_samples, -1).reshape(B * n_samples, -1)

        # Sample from base
        z = torch.randn(B * n_samples, device=context.device)

        # Forward pass through transforms
        for transform in self.transforms:
            z, _ = transform.forward(z, ctx_expanded)

        temp_f = self._unstandardize(z)
        return temp_f.reshape(B, n_samples)

    def cdf(self, temp_f: Tensor, context: Tensor, n_quad: int = 200) -> Tensor:
        """CDF at given temperature(s) via numerical integration.

        For bracket pricing, we need P(T <= x).  Compute via dense sampling
        from the flow and empirical CDF, which is smoother than quadrature
        for complex densities.

        temp_f: (B,) or scalar — evaluation points
        context: (B, d_condition)
        returns: (B,) CDF values
        """
        samples = self.sample(context, n_samples=2000)  # (B, 2000)
        temp_f_expanded = temp_f.unsqueeze(-1) if temp_f.dim() > 0 else temp_f
        cdf_vals = (samples <= temp_f_expanded).float().mean(dim=-1)
        return cdf_vals

    def bracket_prob(
        self, lower: float, upper: float, context: Tensor, n_samples: int = 2000
    ) -> Tensor:
        """P(lower <= T < upper) from the flow.

        context: (B, d_condition)
        returns: (B,) bracket probabilities
        """
        samples = self.sample(context, n_samples)  # (B, n_samples)

        # CLI rounding adjustment
        adj_lower = lower - 0.5 if lower > -900 else -1e6
        adj_upper = upper + 0.5 if upper < 900 else 1e6

        in_bracket = ((samples >= adj_lower) & (samples < adj_upper)).float()
        prob = in_bracket.mean(dim=-1)
        return prob.clamp(min=0.001, max=0.999)

    def all_bracket_probs(
        self, brackets: list[tuple[float, float]], context: Tensor,
        n_samples: int = 3000,
    ) -> dict[str, Tensor]:
        """Compute P(YES) for all brackets in one pass.

        brackets: list of (lower, upper) in °F
        context: (1, d_condition) — single prediction context
        returns: dict of bracket_label → probability tensor
        """
        samples = self.sample(context, n_samples)  # (1, n_samples)
        samples = samples.squeeze(0)  # (n_samples,)

        probs = {}
        for lower, upper in brackets:
            adj_lo = lower - 0.5 if lower > -900 else -1e6
            adj_hi = upper + 0.5 if upper < 900 else 1e6
            in_bracket = ((samples >= adj_lo) & (samples < adj_hi)).float()
            p = in_bracket.mean().clamp(min=0.001, max=0.999)
            label = self._bracket_label(lower, upper)
            probs[label] = p
        return probs

    def crps(self, observation: Tensor, context: Tensor, n_samples: int = 1000) -> Tensor:
        """CRPS via Monte Carlo: E|X-y| - 0.5*E|X-X'|.

        observation: (B,) actual temperatures
        context: (B, d_condition)
        returns: (B,) CRPS values
        """
        samples = self.sample(context, n_samples)  # (B, n_samples)
        term1 = (samples - observation.unsqueeze(-1)).abs().mean(dim=-1)
        # E|X-X'| via subsampled pairwise
        n_sub = min(n_samples, 300)
        s1 = samples[:, :n_sub]
        s2 = samples[:, torch.randperm(n_samples, device=samples.device)[:n_sub]]
        term2 = (s1 - s2).abs().mean(dim=-1)
        return term1 - 0.5 * term2

    @staticmethod
    def _bracket_label(lower: float, upper: float) -> str:
        if lower < -900:
            return f"≤{int(upper)}°F"
        elif upper > 900:
            return f"≥{int(lower)}°F"
        else:
            return f"{int(lower)}-{int(upper)}°F"


# ──────────────────────────────────────────────────────────────────────
# Conditioning Vector Builder
# ──────────────────────────────────────────────────────────────────────

class NSFConditionBuilder:
    """Builds the conditioning vector for the NSF from DS3M state.

    Extracts and concatenates:
      - Particle cloud: weighted mean, std, skew, kurtosis, ESS (5)
      - Per-particle regime posterior: (K) regime probs
      - Mamba hidden state: mean-pooled d_model features (d_model)
      - NWP consensus: mean forecast, spread, latest bias (3)
      - Temporal: hour_sin, hour_cos, doy_sin, doy_cos, lead_time, is_dst (6)
      - LETKF: spatial spread, n_updates (2)
      - Running extremes: running_high, running_low, current_temp (3)
    """

    def __init__(self, k_regimes: int = 5, d_model: int = 64):
        self.k_regimes = k_regimes
        self.d_model = d_model
        # Total: 5 + K + d_model + 3 + 6 + 2 + 3 = 19 + K + d_model
        self.d_condition = 19 + k_regimes + d_model

    def build(
        self,
        particle_weights: Tensor,       # (N,) normalized weights
        particle_values: Tensor,         # (N,) remaining_high or remaining_low
        regime_posterior: Tensor,        # (K,)
        mamba_h: Tensor,                 # (d_model,)
        nwp_mean: float,
        nwp_spread: float,
        nwp_bias: float,
        hour_local: float,
        day_of_year: int,
        lead_time_hours: float,
        is_dst: bool,
        letkf_spread: float,
        letkf_n_updates: int,
        running_high: float,
        running_low: float,
        current_temp: float,
    ) -> Tensor:
        """Build conditioning vector (1, d_condition)."""
        w = particle_weights
        v = particle_values

        # Particle cloud statistics
        w_mean = (w * v).sum()
        w_var = (w * (v - w_mean)**2).sum()
        w_std = w_var.sqrt().clamp(min=0.01)
        w_skew = (w * ((v - w_mean) / w_std)**3).sum()
        w_kurt = (w * ((v - w_mean) / w_std)**4).sum()
        ess = 1.0 / (w**2).sum()

        cloud_stats = torch.tensor([w_mean, w_std, w_skew, w_kurt, ess],
                                   dtype=torch.float32)

        # Temporal encoding
        hour_sin = math.sin(2 * math.pi * hour_local / 24)
        hour_cos = math.cos(2 * math.pi * hour_local / 24)
        doy_sin = math.sin(2 * math.pi * day_of_year / 365.25)
        doy_cos = math.cos(2 * math.pi * day_of_year / 365.25)
        temporal = torch.tensor(
            [hour_sin, hour_cos, doy_sin, doy_cos, lead_time_hours, float(is_dst)],
            dtype=torch.float32,
        )

        nwp = torch.tensor([nwp_mean, nwp_spread, nwp_bias], dtype=torch.float32)
        spatial = torch.tensor([letkf_spread, float(letkf_n_updates)], dtype=torch.float32)
        extremes = torch.tensor([running_high, running_low, current_temp], dtype=torch.float32)

        # Concatenate all
        condition = torch.cat([
            cloud_stats,         # 5
            regime_posterior,    # K
            mamba_h,             # d_model
            nwp,                 # 3
            temporal,            # 6
            spatial,             # 2
            extremes,            # 3
        ])

        return condition.unsqueeze(0)  # (1, d_condition)
