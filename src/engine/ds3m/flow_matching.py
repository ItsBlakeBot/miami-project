"""
Flow Matching Density Estimator — replaces Neural Spline Flow (NSF).

Why Flow Matching > NSF:
  1. No invertibility constraint → simpler, more expressive networks
  2. Smooth tails (no bin clipping) → better calibration for extreme temps
  3. Simple MSE training on velocity field → clean gradients, fast convergence
  4. Scales naturally to high-dim conditioning (GraphMamba 384-dim embedding)
  5. Continuous-time ODE → arbitrary precision at inference via adaptive solver

Architecture:
  - Learns a time-dependent velocity field v(x, t; θ) that transports
    samples from N(0,1) at t=0 to the target distribution at t=1
  - Conditioning: Mamba embedding + regime posterior + time features
  - Inference: solve ODE dx/dt = v(x, t) from t=0 to t=1
  - Training: regress v against the optimal transport path (linear interpolant)

Reference: Lipman et al. "Flow Matching for Generative Modeling" (2023)
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


@dataclass
class FlowMatchingConfig:
    """Configuration for the Flow Matching density estimator."""
    d_condition: int = 408       # Conditioning dimension (mamba_embed + regime + time + bracket)
    d_hidden: int = 256          # Hidden dimension
    n_hidden_layers: int = 6     # Depth of velocity network
    d_data: int = 1              # Data dimension (temperature or bracket prob)
    n_ode_steps: int = 30        # Euler steps at inference (more = more accurate)
    dropout: float = 0.1
    use_adaptive_solver: bool = False  # Use adaptive RK45 instead of fixed Euler
    sigma_min: float = 1e-4      # Minimum noise level


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal embedding for the flow time variable t ∈ [0, 1]."""

    def __init__(self, d_embed: int = 64):
        super().__init__()
        self.d_embed = d_embed

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) or (B, 1) → (B, d_embed)"""
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        half = self.d_embed // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
        args = t * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class VelocityNetwork(nn.Module):
    """
    Time-conditioned velocity field v(x, t, c; θ).

    Predicts the velocity that transports x from noise (t=0)
    to the target distribution (t=1), conditioned on c.

    Architecture: ResNet-style MLP with time embedding + conditioning.
    """

    def __init__(self, config: FlowMatchingConfig):
        super().__init__()
        self.config = config

        d_time = 64
        self.time_embed = SinusoidalTimeEmbedding(d_time)

        # Input: x (d_data) + t_embed (d_time) + condition (d_condition)
        d_input = config.d_data + d_time + config.d_condition

        # Build residual blocks
        layers = []
        layers.append(nn.Linear(d_input, config.d_hidden))
        layers.append(nn.SiLU())
        layers.append(nn.Dropout(config.dropout))

        for _ in range(config.n_hidden_layers):
            layers.append(ResBlock(config.d_hidden, config.dropout))

        layers.append(nn.Linear(config.d_hidden, config.d_data))

        self.net = nn.Sequential(*layers)

        # Conditioning projection (FiLM-style)
        self.cond_scale = nn.Linear(config.d_condition, config.d_hidden)
        self.cond_shift = nn.Linear(config.d_condition, config.d_hidden)

    def forward(
        self,
        x: torch.Tensor,       # (B, d_data)
        t: torch.Tensor,       # (B,)
        condition: torch.Tensor # (B, d_condition)
    ) -> torch.Tensor:
        """Predict velocity v(x, t, c)."""
        t_embed = self.time_embed(t)  # (B, 64)
        inp = torch.cat([x, t_embed, condition], dim=-1)

        h = self.net[0](inp)   # Linear → d_hidden
        h = self.net[1](h)     # SiLU
        h = self.net[2](h)     # Dropout

        # FiLM conditioning
        scale = self.cond_scale(condition)  # (B, d_hidden)
        shift = self.cond_shift(condition)  # (B, d_hidden)
        h = h * (1 + scale) + shift

        # Residual blocks
        for layer in self.net[3:-1]:
            h = layer(h)

        # Output projection
        v = self.net[-1](h)    # (B, d_data)
        return v


class ResBlock(nn.Module):
    """Residual block with SiLU activation."""

    def __init__(self, d_hidden: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(d_hidden),
            nn.Linear(d_hidden, d_hidden * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden * 2, d_hidden),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class FlowMatchingDensity(nn.Module):
    """
    Flow Matching density estimator for weather bracket probabilities.

    Training:
      loss = MSE(v_pred, v_target)
      where v_target = x1 - x0 (optimal transport velocity)
      and x_t = (1-t)*x0 + t*x1 + sigma*noise

    Inference:
      Sample x0 ~ N(0,1)
      Solve ODE: dx/dt = v(x, t, condition) from t=0 to t=1
      x1 is the predicted temperature / bracket probability

    Density estimation:
      log p(x1) computed via change of variables + trace estimator
    """

    def __init__(self, config: Optional[FlowMatchingConfig] = None):
        super().__init__()
        self.config = config or FlowMatchingConfig()
        self.velocity_net = VelocityNetwork(self.config)

    def forward(
        self,
        x1: torch.Tensor,          # (B, d_data) target samples
        condition: torch.Tensor,     # (B, d_condition) conditioning
    ) -> dict:
        """
        Training forward pass.

        Computes the flow matching loss:
          1. Sample t ~ U(0, 1)
          2. Sample x0 ~ N(0, 1)
          3. Compute x_t = (1 - (1-σ)t) * x0 + t * x1
          4. Target velocity = x1 - (1-σ) * x0
          5. Loss = ||v_pred(x_t, t, c) - v_target||²
        """
        B = x1.shape[0]
        sigma = self.config.sigma_min

        # Sample flow time
        t = torch.rand(B, device=x1.device)

        # Sample source noise
        x0 = torch.randn_like(x1)

        # Interpolate
        t_expand = t.unsqueeze(-1)
        x_t = (1 - (1 - sigma) * t_expand) * x0 + t_expand * x1

        # Target velocity (optimal transport)
        v_target = x1 - (1 - sigma) * x0

        # Predicted velocity
        v_pred = self.velocity_net(x_t, t, condition)

        # Flow matching loss
        loss = F.mse_loss(v_pred, v_target)

        return {
            "loss": loss,
            "v_pred": v_pred,
            "v_target": v_target,
        }

    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,     # (B, d_condition)
        n_samples: int = 1000,
        n_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate samples by solving the flow ODE.

        Returns: (B, n_samples, d_data) samples from the learned distribution.
        """
        B = condition.shape[0]
        n_steps = n_steps or self.config.n_ode_steps
        d = self.config.d_data

        # Source samples: x0 ~ N(0, 1)
        x = torch.randn(B, n_samples, d, device=condition.device)

        # Expand condition for n_samples
        cond = condition.unsqueeze(1).expand(B, n_samples, -1)  # (B, n_samples, d_cond)

        # Euler integration from t=0 to t=1
        dt = 1.0 / n_steps

        for step in range(n_steps):
            t = torch.full((B * n_samples,), step * dt, device=x.device)

            x_flat = x.reshape(B * n_samples, d)
            cond_flat = cond.reshape(B * n_samples, -1)

            v = self.velocity_net(x_flat, t, cond_flat)
            v = v.reshape(B, n_samples, d)

            x = x + dt * v

        return x

    @torch.no_grad()
    def log_prob(
        self,
        x1: torch.Tensor,           # (B, d_data) points to evaluate
        condition: torch.Tensor,     # (B, d_condition)
        n_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Estimate log probability via the Hutchinson trace estimator.

        Solves the ODE backwards from t=1 to t=0 while accumulating
        the log-determinant of the Jacobian.

        Returns: (B,) log probabilities.
        """
        n_steps = n_steps or self.config.n_ode_steps
        B, d = x1.shape

        x = x1.clone()
        log_det = torch.zeros(B, device=x.device)

        dt = -1.0 / n_steps  # Integrate backwards

        for step in range(n_steps):
            t_val = 1.0 - step / n_steps
            t = torch.full((B,), t_val, device=x.device)

            # Hutchinson trace estimator
            x.requires_grad_(True)
            v = self.velocity_net(x, t, condition)

            # Estimate trace of Jacobian via random projection
            epsilon = torch.randn_like(x)
            (vjp,) = torch.autograd.grad(v, x, epsilon, create_graph=False)
            trace_est = (vjp * epsilon).sum(dim=-1)

            x = x.detach() + dt * v.detach()
            log_det = log_det + dt * trace_est.detach()

        # Base distribution log prob
        log_p0 = -0.5 * (x ** 2 + math.log(2 * math.pi)).sum(dim=-1)

        return log_p0 + log_det

    @torch.no_grad()
    def bracket_probabilities(
        self,
        condition: torch.Tensor,     # (1, d_condition) single inference
        bracket_edges: list[float],  # e.g., [80, 82, 84, 86, 88, 90]
        n_samples: int = 5000,
    ) -> list[float]:
        """
        Compute bracket probabilities for Kalshi markets.

        Generates samples from the learned density and counts
        how many fall in each bracket.

        Returns: list of probabilities summing to ~1.0
        """
        if condition.dim() == 1:
            condition = condition.unsqueeze(0)

        samples = self.sample(condition, n_samples=n_samples)  # (1, n_samples, 1)
        samples = samples.squeeze(0).squeeze(-1)  # (n_samples,)

        n_brackets = len(bracket_edges) + 1
        probs = []

        # Below first edge
        probs.append((samples < bracket_edges[0]).float().mean().item())

        # Between edges
        for i in range(len(bracket_edges) - 1):
            mask = (samples >= bracket_edges[i]) & (samples < bracket_edges[i + 1])
            probs.append(mask.float().mean().item())

        # Above last edge
        probs.append((samples >= bracket_edges[-1]).float().mean().item())

        return probs

    def get_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
