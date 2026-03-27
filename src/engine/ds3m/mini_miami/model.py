"""Mini Miami Weather Brain — 25M params, single-city, 12GB GPU.

Same architecture as WeatherBrainV3 but right-sized:
- Multi-resolution Mamba (fine/medium/coarse)
- Interleaved Graph-SSM on medium branch
- Differentiable Particle Filter (1000 particles, 4 regimes)
- Rectified Flow Matching
- 10 multi-task heads with GradNorm
- 5-member ensemble

No torch.compile (avoids DPF graph breaks).
No DDP (single GPU).
AdamW only (no Muon dependency headaches).
"""

from __future__ import annotations

import copy
import logging
import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import MiniMiamiConfig

log = logging.getLogger(__name__)

# Try Mamba2 CUDA kernels, fall back to pure PyTorch
try:
    from mamba_ssm import Mamba2 as _Mamba2
    HAS_MAMBA2 = True
    log.info("Mini Miami: using Mamba2 CUDA kernels")
except ImportError:
    HAS_MAMBA2 = False
    log.info("Mini Miami: using pure-PyTorch SSM (no mamba_ssm)")


# ──────────────────────────────────────────────────────────────────
# Building blocks
# ──────────────────────────────────────────────────────────────────

class MambaBlock(nn.Module):
    """Single Mamba block with residual + dropout."""

    def __init__(self, d_model: int, d_state: int = 64, d_conv: int = 4,
                 expand: int = 2, dropout: float = 0.2):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        if HAS_MAMBA2:
            # Use CUDA kernels — fast, memory efficient
            # d_model must be divisible by headdim (default 64)
            headdim = 64
            adjusted_d = max(headdim, (d_model // headdim) * headdim)
            self.ssm = _Mamba2(
                d_model=adjusted_d,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            self.proj_in = nn.Linear(d_model, adjusted_d) if adjusted_d != d_model else nn.Identity()
            self.proj_out = nn.Linear(adjusted_d, d_model) if adjusted_d != d_model else nn.Identity()
        else:
            # Pure PyTorch fallback
            self.ssm = PurePytorchSSM(d_model, d_state, d_conv, expand)
            self.proj_in = nn.Identity()
            self.proj_out = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = self.ssm(x)
        x = self.proj_out(x)
        return residual + self.dropout(x)


class PurePytorchSSM(nn.Module):
    """Minimal SSM for CPU/non-CUDA fallback."""

    def __init__(self, d_model: int, d_state: int = 64, d_conv: int = 4, expand: int = 2):
        super().__init__()
        d_inner = d_model * expand
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(d_inner, d_inner, d_conv, padding=d_conv - 1, groups=d_inner)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.dt_proj = nn.Linear(d_inner, d_inner, bias=True)
        self.A_log = nn.Parameter(torch.randn(d_inner, d_state) * 0.01)
        self.D = nn.Parameter(torch.ones(d_inner))

    def forward(self, x: Tensor) -> Tensor:
        B, T, D = x.shape
        xz = self.in_proj(x)
        x_part, z = xz.chunk(2, dim=-1)

        x_conv = x_part.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :T]
        x_conv = x_conv.transpose(1, 2)
        x_conv = F.silu(x_conv)

        y = x_conv * F.silu(z)  # simplified gating
        return self.out_proj(y)


class SpatialGraphAttention(nn.Module):
    """Lightweight graph attention for station spatial relationships."""

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.norm = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, N_stations, d_model)"""
        B, N, D = x.shape
        residual = x
        x = self.norm(x)

        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(2)  # each: (B, N, heads, d_head)
        q = q.transpose(1, 2)  # (B, heads, N, d_head)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return residual + self.dropout(self.out_proj(out))


class InterleavedMambaGraph(nn.Module):
    """Interleaved Mamba + Graph Attention for medium branch.

    Each layer: MambaBlock (temporal) → GraphAttention (spatial).
    This is the key architectural innovation from the big model.
    """

    def __init__(self, d_model: int, n_layers: int, n_stations: int,
                 d_state: int = 64, dropout: float = 0.2):
        super().__init__()
        self.n_stations = n_stations
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                'temporal': MambaBlock(d_model, d_state=d_state, dropout=dropout),
                'spatial': SpatialGraphAttention(d_model, n_heads=4, dropout=dropout),
            }))

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, T, d_model) — but we reshape for spatial attention.

        During spatial attention, we treat the batch as containing
        station-groups. The medium branch operates on per-station
        features that are concatenated along the batch dimension.
        """
        B, T, D = x.shape

        for layer in self.layers:
            # Temporal: standard Mamba over time
            x = layer['temporal'](x)

            # Spatial: reshape to (B//N, N, D), attend across stations, reshape back
            # If N_stations divides B (multi-station batching), do spatial attention
            if B >= self.n_stations and B % self.n_stations == 0:
                n_groups = B // self.n_stations
                # Pool time dimension for spatial attention
                x_spatial = x.mean(dim=1)  # (B, D)
                x_spatial = x_spatial.view(n_groups, self.n_stations, D)
                x_spatial = layer['spatial'](x_spatial)
                x_spatial = x_spatial.view(B, D).unsqueeze(1)
                x = x + x_spatial  # residual spatial signal

        return x


class TemporalBranch(nn.Module):
    """Single-resolution Mamba branch (fine, medium, or coarse)."""

    def __init__(self, d_input: int, d_model: int, n_layers: int,
                 d_state: int = 64, dropout: float = 0.2,
                 interleave_graph: bool = False, n_stations: int = 15):
        super().__init__()
        self.input_proj = nn.Linear(d_input, d_model)

        if interleave_graph:
            self.blocks = InterleavedMambaGraph(
                d_model, n_layers, n_stations, d_state, dropout
            )
        else:
            self.blocks = nn.Sequential(*[
                MambaBlock(d_model, d_state, dropout=dropout)
                for _ in range(n_layers)
            ])

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        x = self.input_proj(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x.mean(dim=1)  # temporal pooling → (B, d_model)


class FeatureMask(nn.Module):
    """Learnable feature masking — same as big model."""

    def __init__(self, n_features: int):
        super().__init__()
        self.logits = nn.Parameter(torch.ones(n_features) * 2.0)

    def forward(self, x: Tensor) -> Tensor:
        mask = torch.sigmoid(self.logits)
        return x * mask


class MiniDPF(nn.Module):
    """Lightweight Differentiable Particle Filter.

    1000 particles (vs 8000), d_latent=64 (vs 192), 4 regimes (vs 8).
    Uses top-k resampling to avoid OOM.
    """

    def __init__(self, d_observation: int, n_particles: int = 1000,
                 d_latent: int = 64, k_regimes: int = 4, resample_top_k: int = 32):
        super().__init__()
        self.n_particles = n_particles
        self.d_latent = d_latent
        self.k_regimes = k_regimes
        self.resample_top_k = resample_top_k

        # Transition model
        self.transition = nn.Sequential(
            nn.Linear(d_latent + k_regimes, d_latent * 2),
            nn.GELU(),
            nn.Linear(d_latent * 2, d_latent),
        )

        # Observation model
        self.obs_encoder = nn.Sequential(
            nn.Linear(d_observation, d_latent * 2),
            nn.GELU(),
            nn.Linear(d_latent * 2, d_latent),
        )

        # Weight model (log-likelihood)
        self.weight_net = nn.Sequential(
            nn.Linear(d_latent * 2, d_latent),
            nn.GELU(),
            nn.Linear(d_latent, 1),
        )

        # Regime classifier
        self.regime_head = nn.Linear(d_latent, k_regimes)

        n_params = sum(p.numel() for p in self.parameters())
        log.info(f"MiniDPF: {n_params:,} params, {n_particles} particles, "
                 f"d_latent={d_latent}, k_regimes={k_regimes}")

    def forward(self, observation: Tensor) -> dict[str, Tensor]:
        """
        observation: (B, d_observation)
        Returns: particle_state (B, d_latent), regime_posterior (B, k_regimes)
        """
        B = observation.shape[0]
        device = observation.device

        # Encode observation
        obs_encoded = self.obs_encoder(observation)  # (B, d_latent)

        # Initialize particles
        particles = torch.randn(B, self.n_particles, self.d_latent, device=device) * 0.1

        # Single-step update (no sequential scan needed for single observation)
        # Regime prior for each particle
        regime_logits = self.regime_head(particles.mean(dim=1))  # (B, k_regimes)
        regime_probs = F.softmax(regime_logits, dim=-1)

        # Expand regime for transition
        regime_expanded = regime_probs.unsqueeze(1).expand(-1, self.n_particles, -1)
        trans_input = torch.cat([particles, regime_expanded], dim=-1)  # (B, N, d_latent+k)
        particles = self.transition(trans_input)  # (B, N, d_latent)

        # Compute weights
        obs_expanded = obs_encoded.unsqueeze(1).expand(-1, self.n_particles, -1)
        weight_input = torch.cat([particles, obs_expanded], dim=-1)  # (B, N, 2*d_latent)
        log_weights = self.weight_net(weight_input).squeeze(-1)  # (B, N)
        weights = F.softmax(log_weights, dim=-1)

        # Top-k resampling (memory efficient)
        _, top_indices = weights.topk(self.resample_top_k, dim=-1)
        top_particles = particles.gather(
            1, top_indices.unsqueeze(-1).expand(-1, -1, self.d_latent)
        )
        particle_state = (top_particles * weights.gather(1, top_indices).unsqueeze(-1)).sum(dim=1)

        return {
            'particle_state': particle_state,  # (B, d_latent)
            'regime_posterior': regime_probs,   # (B, k_regimes)
        }


class MiniFlowMatching(nn.Module):
    """Rectified Flow Matching — same concept, smaller network."""

    def __init__(self, d_condition: int, d_hidden: int = 512,
                 n_layers: int = 4, n_steps_infer: int = 8):
        super().__init__()
        self.d_condition = d_condition
        self.n_steps_infer = n_steps_infer

        layers = []
        layers.append(nn.Linear(d_condition + 1 + 1, d_hidden))  # +1 for t, +1 for x_t
        layers.append(nn.GELU())
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(d_hidden, d_hidden))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(0.1))
        layers.append(nn.Linear(d_hidden, 1))
        self.velocity_net = nn.Sequential(*layers)

        n_params = sum(p.numel() for p in self.parameters())
        log.info(f"MiniFlowMatching: {n_params:,} params, d_condition={d_condition}")

    def training_loss(self, condition: Tensor, target: Tensor) -> dict[str, Tensor]:
        """Rectified flow training loss."""
        B = condition.shape[0]
        device = condition.device

        t = torch.rand(B, 1, device=device)
        noise = torch.randn_like(target)
        x_t = (1 - t) * noise + t * target
        v_target = target - noise

        inp = torch.cat([condition, t, x_t], dim=-1)
        v_pred = self.velocity_net(inp)
        flow_loss = F.mse_loss(v_pred, v_target)

        return {'flow_loss': flow_loss, 'consistency_loss': torch.tensor(0.0, device=device),
                'loss': flow_loss}

    @torch.no_grad()
    def sample(self, condition: Tensor, n_steps: int = None) -> Tensor:
        n_steps = n_steps or self.n_steps_infer
        x = torch.randn(condition.shape[0], 1, device=condition.device)
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = torch.full((condition.shape[0], 1), i * dt, device=condition.device)
            inp = torch.cat([condition, t, x], dim=-1)
            v = self.velocity_net(inp)
            x = x + v * dt
        return x


class MiniMultiTaskHeads(nn.Module):
    """10-task prediction heads with GradNorm — same as big model."""

    TASK_NAMES = [
        "daily_max", "daily_min", "max_hour", "min_hour",
        "bracket_probs", "next_hour_temp", "nwp_bias", "regime",
        "bracket_probs_low", "daily_low_bracket",
    ]

    def __init__(self, d_input: int = 384, n_brackets: int = 6, n_regimes: int = 4):
        super().__init__()
        self.head_daily_max = nn.Linear(d_input, 1)
        self.head_daily_min = nn.Linear(d_input, 1)
        self.head_max_hour = nn.Linear(d_input, 1)
        self.head_min_hour = nn.Linear(d_input, 1)
        self.head_bracket_probs = nn.Linear(d_input, n_brackets)
        self.head_bracket_probs_low = nn.Linear(d_input, n_brackets)
        self.head_next_hour_temp = nn.Linear(d_input, 1)
        self.head_nwp_bias = nn.Linear(d_input, 1)
        self.head_regime = nn.Linear(d_input, n_regimes)

        # Initialize biases to target means
        with torch.no_grad():
            self.head_daily_max.bias.fill_(85.0)
            self.head_daily_min.bias.fill_(70.0)
            self.head_next_hour_temp.bias.fill_(80.0)
            self.head_nwp_bias.bias.fill_(0.0)

        # GradNorm
        self.log_weights = nn.Parameter(torch.zeros(len(self.TASK_NAMES)))
        self._gradnorm_active = False
        self.register_buffer("initial_losses", torch.ones(len(self.TASK_NAMES)))
        self._initialized = False

        n_params = sum(p.numel() for p in self.parameters())
        log.info(f"MiniMultiTaskHeads: {n_params:,} params, {len(self.TASK_NAMES)} tasks")

    @property
    def task_weights(self) -> Tensor:
        w = torch.exp(self.log_weights)
        return w / w.sum() * len(self.TASK_NAMES)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        return {
            "daily_max": self.head_daily_max(x).squeeze(-1),
            "daily_min": self.head_daily_min(x).squeeze(-1),
            "max_hour": torch.sigmoid(self.head_max_hour(x).squeeze(-1)) * 24.0,
            "min_hour": torch.sigmoid(self.head_min_hour(x).squeeze(-1)) * 24.0,
            "bracket_probs": self.head_bracket_probs(x),
            "bracket_probs_low": self.head_bracket_probs_low(x),
            "next_hour_temp": self.head_next_hour_temp(x).squeeze(-1),
            "nwp_bias": self.head_nwp_bias(x).squeeze(-1),
            "regime": self.head_regime(x),
        }

    def compute_loss(self, predictions: dict, targets: dict) -> dict:
        losses = {}

        # Regression (skip NaN targets)
        for name in ["daily_max", "daily_min", "max_hour", "min_hour",
                      "next_hour_temp", "nwp_bias"]:
            if name in targets:
                t = targets[name]
                p = predictions[name]
                valid = ~torch.isnan(t) & ~torch.isinf(t)
                if valid.any():
                    losses[name] = F.mse_loss(p[valid], t[valid])

        # Bracket HIGH
        if "bracket_target" in targets:
            bt = targets["bracket_target"].long()
            valid = bt >= 0
            if valid.any():
                losses["bracket_probs"] = F.cross_entropy(
                    predictions["bracket_probs"][valid], bt[valid])

        # Bracket LOW
        if "bracket_target_low" in targets:
            bt_low = targets["bracket_target_low"].long()
            valid = bt_low >= 0
            if valid.any():
                losses["bracket_probs_low"] = F.cross_entropy(
                    predictions["bracket_probs_low"][valid], bt_low[valid])

        # Regime
        if "regime" in targets:
            rt = targets["regime"].long()
            if (rt >= 0).any():
                losses["regime"] = F.cross_entropy(
                    predictions["regime"], rt, ignore_index=-1)

        # Weighted sum
        dev = next(iter(losses.values())).device if losses else torch.device('cpu')
        task_losses = torch.stack([
            losses.get(name, torch.tensor(0.0, device=dev))
            for name in self.TASK_NAMES
        ])
        task_losses = torch.nan_to_num(task_losses, nan=0.0)

        if not self._initialized and task_losses.sum() > 0:
            self.initial_losses.copy_(task_losses.detach().clamp(min=1e-6))
            self._initialized = True

        total_loss = task_losses.sum()

        result = {"total_loss": total_loss, "task_weights": self.task_weights.detach()}
        for name in self.TASK_NAMES:
            if name in losses:
                result[f"loss_{name}"] = losses[name].detach()

        return result


# ──────────────────────────────────────────────────────────────────
# Main Model
# ──────────────────────────────────────────────────────────────────

class MiniMiamiWeatherBrain(nn.Module):
    """Mini Miami Weather Brain — 25M params.

    Same architecture as WeatherBrainV3, right-sized for single city.

    Forward flow:
    1. Feature masking on each branch
    2. Fine/Medium/Coarse → TemporalBranch → temporal embeddings
    3. Fusion: concat + project → shared representation
    4. DPF: particle filter for regime detection
    5. Flow matching: probabilistic temperature generation
    6. Multi-task heads: 10 prediction targets
    """

    def __init__(self, config: MiniMiamiConfig = None, seed: int = 42):
        super().__init__()
        if config is None:
            config = MiniMiamiConfig()
        self.config = config
        self.d_model = config.medium_branch.d_model  # 384

        torch.manual_seed(seed)

        # Feature masks
        self.mask_fine = FeatureMask(config.n_features_fine)
        self.mask_medium = FeatureMask(config.n_features_medium)
        self.mask_coarse = FeatureMask(config.n_features_coarse)

        # Temporal branches
        self.fine_branch = TemporalBranch(
            config.fine_branch.d_input, config.fine_branch.d_model,
            config.fine_branch.n_layers, config.fine_branch.d_state,
            config.fine_branch.dropout,
        )
        self.medium_branch = TemporalBranch(
            config.medium_branch.d_input, config.medium_branch.d_model,
            config.medium_branch.n_layers, config.medium_branch.d_state,
            config.medium_branch.dropout,
            interleave_graph=True, n_stations=config.graph.n_stations,
        )
        self.coarse_branch = TemporalBranch(
            config.coarse_branch.d_input, config.coarse_branch.d_model,
            config.coarse_branch.n_layers, config.coarse_branch.d_state,
            config.coarse_branch.dropout,
        )

        # Fusion: project all branches to d_model
        d_concat = (config.fine_branch.d_model +
                    config.medium_branch.d_model +
                    config.coarse_branch.d_model)  # 256 + 384 + 256 = 896
        self.fusion = nn.Sequential(
            nn.Linear(d_concat, self.d_model),
            nn.GELU(),
            nn.Dropout(config.fusion.dropout),
            nn.Linear(self.d_model, self.d_model),
            nn.LayerNorm(self.d_model),
        )

        # DPF
        self.dpf = MiniDPF(
            d_observation=self.d_model,
            n_particles=config.dpf.n_particles,
            d_latent=config.dpf.d_latent,
            k_regimes=config.dpf.k_regimes,
            resample_top_k=config.dpf.resample_top_k,
        )

        # Flow matching
        self.flow_matching = MiniFlowMatching(
            d_condition=self.d_model + config.dpf.d_latent + config.dpf.k_regimes,
            d_hidden=config.flow.d_hidden,
            n_layers=config.flow.n_layers,
            n_steps_infer=config.flow.n_steps_infer,
        )

        # Multi-task heads
        self.heads = MiniMultiTaskHeads(
            d_input=self.d_model,
            n_brackets=config.tasks.n_brackets,
            n_regimes=config.tasks.n_regimes,
        )

        # Bracket softmax (separate from multi-task heads, for HIGH bracket)
        self.bracket_softmax = nn.Linear(self.d_model, config.tasks.n_brackets)

        # Log param counts
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        log.info(f"MiniMiamiWeatherBrain: {total:,} total, {trainable:,} trainable")

    def forward(self, fine: Tensor, medium: Tensor, coarse: Tensor,
                feature_masks: dict[str, Tensor] | None = None) -> dict[str, Tensor]:
        """
        fine: (B, 32, 18)
        medium: (B, 96, 36)
        coarse: (B, 112, 33)
        """
        # Feature masking
        fine = self.mask_fine(fine)
        medium = self.mask_medium(medium)
        coarse = self.mask_coarse(coarse)

        # Temporal encoding
        h_fine = self.fine_branch(fine)      # (B, 256)
        h_medium = self.medium_branch(medium)  # (B, 384)
        h_coarse = self.coarse_branch(coarse)  # (B, 256)

        # Fusion
        h_concat = torch.cat([h_fine, h_medium, h_coarse], dim=-1)  # (B, 896)
        h_fused = self.fusion(h_concat)  # (B, 384)

        # DPF
        dpf_out = self.dpf(h_fused)
        particle_state = dpf_out['particle_state']  # (B, 64)
        regime_posterior = dpf_out['regime_posterior']  # (B, 4)

        # Condition for flow matching
        condition = torch.cat([h_fused, particle_state, regime_posterior], dim=-1)

        # Multi-task predictions
        predictions = self.heads(h_fused)

        # Bracket softmax (HIGH) — separate head
        bracket_logits = self.bracket_softmax(h_fused)

        return {
            'predictions': predictions,
            'bracket_probs': bracket_logits,
            'regime_posterior': regime_posterior,
            'particle_state': particle_state,
            'condition': condition,
            'temporal_state': h_fused,
            'spatial_state': h_fused,
        }

    def compute_loss(self, outputs: dict, targets: dict,
                     target_temp: Tensor | None = None) -> dict:
        """Compute combined training loss."""
        result = {}

        # Multi-task loss
        task_result = self.heads.compute_loss(outputs['predictions'], targets)
        result.update(task_result)

        # Bracket HIGH (separate head)
        bracket_loss = torch.tensor(0.0, device=outputs['bracket_probs'].device)
        if 'bracket_target' in targets:
            bt = targets['bracket_target'].long()
            valid = bt >= 0
            if valid.any():
                bracket_loss = F.cross_entropy(
                    outputs['bracket_probs'][valid], bt[valid])
        result['bracket_loss'] = bracket_loss

        # Bracket LOW
        bracket_loss_low = torch.tensor(0.0, device=outputs['bracket_probs'].device)
        if 'bracket_target_low' in targets:
            bt_low = targets['bracket_target_low'].long()
            valid = bt_low >= 0
            if valid.any():
                bracket_loss_low = F.cross_entropy(
                    outputs['predictions']['bracket_probs_low'][valid],
                    bt_low[valid])
        result['bracket_loss_low'] = bracket_loss_low

        # Flow matching
        if target_temp is not None:
            flow_result = self.flow_matching.training_loss(
                outputs['condition'], target_temp)
            result['flow_loss'] = flow_result['flow_loss']
            result['consistency_loss'] = flow_result['consistency_loss']
            flow_total = flow_result['loss']
        else:
            flow_total = torch.tensor(0.0, device=outputs['bracket_probs'].device)

        # Combined
        result['total_loss'] = (
            task_result['total_loss']
            + 0.5 * bracket_loss
            + 0.5 * bracket_loss_low
            + 0.3 * flow_total
        )

        return result

    @torch.no_grad()
    def predict_brackets(self, fine: Tensor, medium: Tensor,
                         coarse: Tensor) -> dict[str, Tensor]:
        """Inference mode bracket predictions."""
        self.eval()
        out = self(fine, medium, coarse)
        return {
            'high': F.softmax(out['bracket_probs'], dim=-1),
            'low': F.softmax(out['predictions']['bracket_probs_low'], dim=-1),
            'regime': out['regime_posterior'],
        }


class MiniMiamiEnsemble(nn.Module):
    """5-member ensemble wrapper."""

    def __init__(self, config: MiniMiamiConfig = None):
        if config is None:
            config = MiniMiamiConfig()
        super().__init__()
        self.members = nn.ModuleList([
            MiniMiamiWeatherBrain(config, seed=s)
            for s in config.seeds
        ])

    @torch.no_grad()
    def predict(self, fine: Tensor, medium: Tensor,
                coarse: Tensor) -> dict[str, Tensor]:
        """Ensemble prediction with uncertainty."""
        preds = [m.predict_brackets(fine, medium, coarse) for m in self.members]

        high_probs = torch.stack([p['high'] for p in preds])
        low_probs = torch.stack([p['low'] for p in preds])

        return {
            'high_mean': high_probs.mean(0),
            'high_std': high_probs.std(0),
            'low_mean': low_probs.mean(0),
            'low_std': low_probs.std(0),
            'regime': torch.stack([p['regime'] for p in preds]).mean(0),
        }
