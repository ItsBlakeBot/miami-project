"""Rectified Flow Matching + Consistency Distillation for Weather Brain v3.1.

Replaces the Neural Spline Flow (NSF) with rectified flow matching for
continuous density estimation over temperature predictions.

Key improvements over NSF and v1 Flow Matching:
  1. Rectified flow: straight ODE paths (not curved) -> faster convergence
  2. Consistency distillation: reduce inference from 30 to 8-10 ODE steps
  3. Higher capacity velocity network with FiLM conditioning
  4. Larger conditioning vector: 896 (mamba) + 8 (regime) + 192 (DPF) = 1096

Rectified Flow (Liu et al. 2022):
  - Learn v(x_t, t, c) where x_t = (1-t)*noise + t*target
  - Target velocity: v* = target - noise (straight path)
  - Much straighter ODE trajectories -> fewer steps needed

Consistency Distillation:
  - After training the flow, distill to predict the final state x_1
    directly from any intermediate (x_t, t), enabling 8-10 step inference
    instead of 30+ Euler steps.

Reference:
  Liu et al. "Flow Straight and Fast" (2022)
  Song et al. "Consistency Models" (2023)
"""

from __future__ import annotations

import math
import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from engine.ds3m.wb3_config import FlowMatchingV2Config

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Sinusoidal Time Embedding
# ──────────────────────────────────────────────────────────────────────

class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal embedding for flow time t in [0, 1]."""

    def __init__(self, d_embed: int = 64) -> None:
        super().__init__()
        self.d_embed = d_embed

    def forward(self, t: Tensor) -> Tensor:
        """Embed scalar time values.

        Parameters
        ----------
        t : Tensor
            Shape (B,) or (B, 1).

        Returns
        -------
        Tensor
            Shape (B, d_embed).
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        half = self.d_embed // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device, dtype=t.dtype)
            / half
        )
        args = t * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


# ──────────────────────────────────────────────────────────────────────
# Residual Block
# ──────────────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    """Pre-norm residual block with SiLU activation."""

    def __init__(self, d_hidden: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(d_hidden),
            nn.Linear(d_hidden, d_hidden * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden * 2, d_hidden),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.block(x)


# ──────────────────────────────────────────────────────────────────────
# Velocity Network (with FiLM conditioning)
# ──────────────────────────────────────────────────────────────────────

class VelocityFieldV2(nn.Module):
    """Time-conditioned velocity field v(x_t, t, condition).

    Predicts the velocity that transports x from noise (t=0) to
    the target distribution (t=1), using rectified flow paths.

    Architecture: ResNet-style MLP with FiLM conditioning.

    Parameters
    ----------
    config : FlowMatchingV2Config
        Configuration for the velocity network.
    """

    def __init__(self, config: FlowMatchingV2Config) -> None:
        super().__init__()
        self.config = config

        d_time = config.d_time_embed
        self.time_embed = SinusoidalTimeEmbedding(d_time)

        # Input: x (d_data) + t_embed (d_time) + condition (d_condition)
        d_input = config.d_data + d_time + config.d_condition

        # Build network
        self.input_proj = nn.Linear(d_input, config.d_hidden)
        self.act = nn.SiLU()
        self.input_drop = nn.Dropout(config.dropout)

        self.res_blocks = nn.ModuleList([
            ResBlock(config.d_hidden, config.dropout)
            for _ in range(config.n_layers)
        ])

        self.output_proj = nn.Linear(config.d_hidden, config.d_data)

        # FiLM conditioning (applied after input projection)
        self.cond_scale = nn.Linear(config.d_condition, config.d_hidden)
        self.cond_shift = nn.Linear(config.d_condition, config.d_hidden)

    def forward(
        self,
        x: Tensor,
        t: Tensor,
        condition: Tensor,
    ) -> Tensor:
        """Predict velocity v(x, t, condition).

        Parameters
        ----------
        x : Tensor
            Shape (B, d_data) — noisy sample.
        t : Tensor
            Shape (B,) — flow time in [0, 1].
        condition : Tensor
            Shape (B, d_condition) — conditioning vector.

        Returns
        -------
        Tensor
            Shape (B, d_data) — predicted velocity.
        """
        t_embed = self.time_embed(t)
        inp = torch.cat([x, t_embed, condition], dim=-1)

        h = self.input_proj(inp)
        h = self.act(h)
        h = self.input_drop(h)

        # FiLM conditioning
        scale = self.cond_scale(condition)
        shift = self.cond_shift(condition)
        h = h * (1.0 + scale) + shift

        # Residual blocks
        for block in self.res_blocks:
            h = block(h)

        return self.output_proj(h)


# ──────────────────────────────────────────────────────────────────────
# Rectified Flow Matching
# ──────────────────────────────────────────────────────────────────────

class RectifiedFlowMatching(nn.Module):
    """Continuous density estimation via rectified flow.

    Training: learn velocity field v(x_t, t, condition) that transports
    noise to data along straight paths (rectified flow).

    Inference: ODE solve from noise -> data in 8-10 steps
    (consistency distilled).

    Parameters
    ----------
    config : FlowMatchingV2Config or None
        Configuration. If None, uses defaults.
    """

    def __init__(self, config: FlowMatchingV2Config | None = None) -> None:
        super().__init__()
        if config is None:
            config = FlowMatchingV2Config()
        self.config = config
        self.velocity_field = VelocityFieldV2(config)

        # Consistency distillation: target network (EMA of velocity)
        self.velocity_field_ema = VelocityFieldV2(config)
        self.velocity_field_ema.load_state_dict(
            self.velocity_field.state_dict()
        )
        for p in self.velocity_field_ema.parameters():
            p.requires_grad_(False)

        # EMA decay rate
        self.ema_decay = 0.999

        # Location/scale normalization (learned)
        self.loc_param = nn.Parameter(torch.tensor(85.0))   # typical Miami high
        self.scale_param = nn.Parameter(torch.tensor(5.0))  # typical spread

        n_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        log.info(
            f"RectifiedFlowMatching: {n_params:,} trainable parameters, "
            f"d_condition={config.d_condition}, "
            f"n_steps_infer={config.n_ode_steps_infer}"
        )

    def _standardize(self, temp_f: Tensor) -> Tensor:
        """Map temperature to standardized space."""
        return (temp_f - self.loc_param) / self.scale_param.clamp(min=0.1)

    def _unstandardize(self, z: Tensor) -> Tensor:
        """Map standardized space back to temperature."""
        return z * self.scale_param.clamp(min=0.1) + self.loc_param

    @torch.no_grad()
    def _update_ema(self) -> None:
        """Update EMA target network for consistency distillation."""
        for p_ema, p in zip(
            self.velocity_field_ema.parameters(),
            self.velocity_field.parameters(),
        ):
            p_ema.data.mul_(self.ema_decay).add_(
                p.data, alpha=1.0 - self.ema_decay
            )

    def forward(
        self,
        condition: Tensor,
        target: Tensor | None = None,
    ) -> Tensor | dict:
        """Forward pass: training loss or sampling.

        Parameters
        ----------
        condition : Tensor
            Shape (B, d_condition) — conditioning vector.
        target : Tensor or None
            Shape (B, 1) — target temperature in degF. If provided,
            computes training loss. If None, samples from the model.

        Returns
        -------
        dict (training) or Tensor (inference)
            Training: dict with 'loss', 'flow_loss', 'consistency_loss'.
            Inference: (B, 1) sampled temperatures.
        """
        if target is not None:
            return self.training_loss(condition, target)
        else:
            return self.sample(condition)

    def training_loss(
        self,
        condition: Tensor,
        target: Tensor,
    ) -> dict[str, Tensor]:
        """Compute rectified flow training loss.

        Parameters
        ----------
        condition : Tensor
            Shape (B, d_condition).
        target : Tensor
            Shape (B, 1) — target values in degF.

        Returns
        -------
        dict[str, Tensor]
            Contains 'loss', 'flow_loss', and 'consistency_loss'.
        """
        # Standardize target
        target_std = self._standardize(target)

        # ── Rectified flow loss ───────────────────────────────────
        # Sample from [0.025, 0.975] to avoid boundary artifacts with dt_teacher
        t = torch.rand(target_std.shape[0], device=target_std.device) * 0.95 + 0.025
        noise = torch.randn_like(target_std)

        # Straight path interpolation: x_t = (1-t)*noise + t*target
        t_expand = t.unsqueeze(-1)
        x_t = (1.0 - t_expand) * noise + t_expand * target_std

        # Predicted velocity
        v_pred = self.velocity_field(x_t, t, condition)

        # Target velocity (straight path): v* = target - noise
        v_target = target_std - noise

        flow_loss = F.mse_loss(v_pred, v_target)

        # ── Consistency distillation loss ─────────────────────────
        # Derivation (Song et al. 2023, adapted for rectified flow):
        #   The consistency property requires f(x_t, t) = f(x_{t'}, t') for
        #   any (x_t, t) and (x_{t'}, t') on the same ODE trajectory.
        #   Here f(x_t, t) = x_t + v(x_t, t) * (1 - t) estimates x_1 (the endpoint).
        #
        #   Teacher estimate: take one Euler step t -> t+dt using EMA velocity,
        #   then estimate x_1 from the new point. This gives a 2-step estimate.
        #   Student estimate: directly estimate x_1 from (x_t, t) in one shot.
        #   The loss enforces that both estimates agree, enabling few-step inference.
        with torch.no_grad():
            # Teacher: one Euler step from t to t+dt
            dt_teacher = 1.0 / self.config.n_ode_steps_train
            v_teacher = self.velocity_field_ema(x_t, t, condition)
            x_t_next = x_t + v_teacher * dt_teacher

            # Teacher's prediction at t+dt
            t_next = (t + dt_teacher).clamp(max=1.0)
            v_teacher_next = self.velocity_field_ema(
                x_t_next, t_next, condition
            )
            # Teacher's full trajectory endpoint estimate: x_1 ≈ x_{t+dt} + v(x_{t+dt}, t+dt) * (1 - (t+dt))
            teacher_endpoint = x_t_next + v_teacher_next * (1.0 - t_next).unsqueeze(-1)

        # Student: direct prediction from (x_t, t) to t=1
        # x_1 ≈ x_t + v(x_t, t) * (1 - t)
        v_student = self.velocity_field(x_t, t, condition)
        student_endpoint = x_t + v_student * (1.0 - t_expand)

        consistency_loss = F.mse_loss(student_endpoint, teacher_endpoint)

        # Combined loss
        loss = flow_loss + 0.1 * consistency_loss

        # Update EMA
        self._update_ema()

        return {
            "loss": loss,
            "flow_loss": flow_loss,
            "consistency_loss": consistency_loss,
        }

    @torch.no_grad()
    def sample(
        self,
        condition: Tensor,
        n_steps: int | None = None,
        n_samples: int = 1,
    ) -> Tensor:
        """Generate samples via Euler ODE solve.

        Uses consistency-distilled steps for fast inference (8-10 steps
        instead of 30+).

        Parameters
        ----------
        condition : Tensor
            Shape (B, d_condition).
        n_steps : int or None
            Number of ODE steps. Defaults to config.n_ode_steps_infer.
        n_samples : int
            Number of samples per condition.

        Returns
        -------
        Tensor
            Shape (B, n_samples, 1) — sampled temperatures in degF.
        """
        if n_steps is None:
            n_steps = self.config.n_ode_steps_infer
        B = condition.shape[0]

        # Start from noise
        x = torch.randn(
            B, n_samples, self.config.d_data, device=condition.device
        )

        # Expand condition for n_samples
        cond = condition.unsqueeze(1).expand(B, n_samples, -1)

        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = torch.full(
                (B * n_samples,), i * dt, device=x.device
            )
            x_flat = x.reshape(B * n_samples, -1)
            cond_flat = cond.reshape(B * n_samples, -1)

            v = self.velocity_field(x_flat, t, cond_flat)
            v = v.reshape(B, n_samples, -1)
            x = x + v * dt

        # Unstandardize
        return self._unstandardize(x)

    @torch.no_grad()
    def bracket_probabilities(
        self,
        condition: Tensor,
        bracket_edges: list[float],
        n_samples: int = 5000,
    ) -> list[float]:
        """Compute bracket probabilities for Kalshi markets.

        Parameters
        ----------
        condition : Tensor
            Shape (1, d_condition) or (B, d_condition).
        bracket_edges : list[float]
            Temperature edges defining brackets.
        n_samples : int
            Number of Monte Carlo samples.

        Returns
        -------
        list[float]
            Bracket probabilities summing to ~1.0.
        """
        if condition.dim() == 1:
            condition = condition.unsqueeze(0)

        samples = self.sample(condition, n_samples=n_samples)
        samples = samples.squeeze(-1)  # (B, n_samples)

        # For single-batch inference
        if samples.shape[0] == 1:
            samples = samples.squeeze(0)

        probs = []
        probs.append((samples < bracket_edges[0]).float().mean().item())
        for i in range(len(bracket_edges) - 1):
            mask = (samples >= bracket_edges[i]) & (
                samples < bracket_edges[i + 1]
            )
            probs.append(mask.float().mean().item())
        probs.append((samples >= bracket_edges[-1]).float().mean().item())

        return probs

    @torch.no_grad()
    def crps(
        self,
        observation: Tensor,
        condition: Tensor,
        n_samples: int = 2000,
    ) -> Tensor:
        """CRPS via Monte Carlo: E|X-y| - 0.5*E|X-X'|.

        Parameters
        ----------
        observation : Tensor
            Shape (B,) — actual temperatures.
        condition : Tensor
            Shape (B, d_condition).
        n_samples : int
            Number of samples for CRPS estimation.

        Returns
        -------
        Tensor
            Shape (B,) — CRPS values.
        """
        samples = self.sample(condition, n_samples=n_samples)
        samples = samples.squeeze(-1)  # (B, n_samples)

        term1 = (samples - observation.unsqueeze(-1)).abs().mean(dim=-1)

        n_sub = min(n_samples, 500)
        s1 = samples[:, :n_sub]
        perm = torch.randperm(n_samples, device=samples.device)[:n_sub]
        s2 = samples[:, perm]
        term2 = (s1 - s2).abs().mean(dim=-1)

        return term1 - 0.5 * term2
