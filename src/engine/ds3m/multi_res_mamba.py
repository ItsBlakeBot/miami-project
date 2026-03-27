"""Multi-Resolution Temporal Mamba for Weather Brain v3.1.

Three parallel Mamba branches at different temporal resolutions, fused
via a hybrid Mamba-Transformer cross-attention mechanism (Jamba/SST style).

Branches:
  Fine   (15-min, 32 steps):  d_model=640, d_state=64, 8 layers   → ~22M params
  Medium (hourly, 96 steps):  d_model=896, d_state=64, 12 layers  → ~50M params
  Coarse (3-hourly, 112 steps): d_model=512, d_state=64, 6 layers → ~12M params

Cross-Resolution Fusion (SST-style):
  4 SSTFusionBlock layers with learned Mamba/Transformer router
  d_model=896, 12 heads, regime-conditioned routing
  Bidirectional: fine <-> medium <-> coarse                        → ~9M params

Total temporal encoding: ~93M params

Pure PyTorch implementation — works on MPS, CPU, CUDA without
external mamba_ssm dependency. Optional CUDA kernel acceleration.
"""

from __future__ import annotations

import math
import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from engine.ds3m.wb3_config import MambaBranchConfig, FusionConfig, GraphMambaV3Config

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Core Selective SSM Block (Pure PyTorch)
# ──────────────────────────────────────────────────────────────────────

class SelectiveSSMBlock(nn.Module):
    """Pure PyTorch selective SSM — works on MPS, CPU, CUDA.

    Implements the Mamba selective state space mechanism:
      Input projection -> Conv1d -> SSM scan -> Output projection

    The selectivity comes from input-dependent B, C, delta parameters
    that allow the model to selectively remember or forget information.

    Parameters
    ----------
    d_model : int
        Model dimension.
    d_state : int
        SSM state dimension per channel.
    d_conv : int
        Local convolution width.
    expand : int
        Inner dimension expansion factor.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand
        self.d_conv = d_conv

        # Input projection: x -> (z, x_ssm) split
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Depthwise convolution (local context)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=d_conv - 1,
            groups=self.d_inner, bias=True,
        )

        # SSM parameters: input-dependent B, C, delta
        self.x_proj = nn.Linear(
            self.d_inner, d_state * 2 + 1, bias=False
        )  # B, C, dt

        # A is a learnable diagonal (log-space for stability)
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1, dtype=torch.float32))
            .unsqueeze(0).expand(self.d_inner, -1).clone()
        )

        # dt bias (learned initialization of discretization step)
        self.dt_bias = nn.Parameter(torch.zeros(self.d_inner))
        nn.init.uniform_(self.dt_bias, math.log(0.001), math.log(0.1))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # D skip connection (Mamba3 style)
        self.D = nn.Parameter(torch.ones(self.d_inner))

    def forward(self, x: Tensor) -> Tensor:
        """Full sequence processing (training mode).

        Parameters
        ----------
        x : Tensor
            Shape (batch, seq_len, d_model).

        Returns
        -------
        Tensor
            Shape (batch, seq_len, d_model).
        """
        batch, seq_len, _ = x.shape

        # Project and split into gate (z) and SSM input (x_ssm)
        xz = self.in_proj(x)
        x_ssm, z = xz.chunk(2, dim=-1)

        # Depthwise conv (causal: trim future)
        x_conv = x_ssm.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]
        x_conv = x_conv.transpose(1, 2)
        x_conv = F.silu(x_conv)

        # Compute input-dependent SSM parameters
        ssm_params = self.x_proj(x_conv)
        B = ssm_params[..., :self.d_state]
        C = ssm_params[..., self.d_state:2 * self.d_state]
        dt = ssm_params[..., -1:]  # (batch, seq_len, 1) — keep dim for per-channel bias

        # Discretization step (softplus + per-channel bias)
        dt = F.softplus(dt + self.dt_bias.unsqueeze(0).unsqueeze(0))  # (batch, seq_len, d_inner)

        # A matrix (negative for stability)
        A = -torch.exp(self.A_log)

        # Parallel scan
        y = self._parallel_scan(x_conv, A, B, C, dt)

        # Skip connection
        y = y + self.D.unsqueeze(0).unsqueeze(0) * x_conv

        # Gated output
        y = y * F.silu(z)

        return self.out_proj(y)

    def _parallel_scan(
        self,
        x: Tensor,
        A: Tensor,
        B: Tensor,
        C: Tensor,
        dt: Tensor,
    ) -> Tensor:
        """Memory-efficient chunked recurrent scan.

        Processes in chunks of 16 timesteps with detached hidden states
        between chunks to limit gradient graph size.
        """
        batch, seq_len, d_inner = x.shape
        chunk_size = 16

        h = torch.zeros(batch, d_inner, self.d_state, device=x.device, dtype=x.dtype)
        all_outputs = []

        for chunk_start in range(0, seq_len, chunk_size):
            chunk_end = min(chunk_start + chunk_size, seq_len)
            chunk_outputs = []

            # Detach hidden state between chunks to limit graph size
            if chunk_start > 0:
                h = h.detach()

            for t in range(chunk_start, chunk_end):
                dt_t = dt[:, t].unsqueeze(-1)
                A_bar_t = torch.exp(A.unsqueeze(0) * dt_t)
                B_t = B[:, t, :]
                C_t = C[:, t, :]
                x_t = x[:, t, :]

                h = A_bar_t * h + dt_t * B_t.unsqueeze(1) * x_t.unsqueeze(-1)
                y_t = (C_t.unsqueeze(1) * h).sum(dim=-1)
                chunk_outputs.append(y_t)

            all_outputs.extend(chunk_outputs)

        return torch.stack(all_outputs, dim=1)

    def step(
        self, x: Tensor, state: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """Single-step recurrent update (inference mode).

        Parameters
        ----------
        x : Tensor
            Shape (batch, d_model).
        state : Tensor or None
            Shape (batch, d_inner, d_state).

        Returns
        -------
        tuple[Tensor, Tensor]
            (output, new_state) — output is (batch, d_model).
        """
        batch = x.shape[0]

        if state is None:
            state = torch.zeros(
                batch, self.d_inner, self.d_state,
                device=x.device, dtype=x.dtype,
            )

        xz = self.in_proj(x)
        x_ssm, z = xz.chunk(2, dim=-1)
        x_conv = F.silu(x_ssm)

        ssm_params = self.x_proj(x_conv)
        B = ssm_params[..., :self.d_state]
        C = ssm_params[..., self.d_state:2 * self.d_state]
        dt_raw = ssm_params[..., -1]
        dt = F.softplus(dt_raw + self.dt_bias.mean())

        A = -torch.exp(self.A_log)
        dt_exp = dt.unsqueeze(-1).unsqueeze(-1)
        A_bar = torch.exp(A.unsqueeze(0) * dt_exp)
        new_state = A_bar * state + dt_exp * B.unsqueeze(1) * x_conv.unsqueeze(-1)

        y = (C.unsqueeze(1) * new_state).sum(dim=-1)
        y = y + self.D.unsqueeze(0) * x_conv
        y = y * F.silu(z)
        output = self.out_proj(y)

        return output, new_state


# ──────────────────────────────────────────────────────────────────────
# Mamba Block (LayerNorm + Residual + Dropout)
# ──────────────────────────────────────────────────────────────────────

class MambaBlock(nn.Module):
    """Single Mamba block: LayerNorm -> SSM -> Residual + Dropout.

    Uses mamba_ssm.Mamba2 CUDA kernels when available (1000x faster,
    950x less memory). Falls back to pure-PyTorch SelectiveSSMBlock
    for CPU/MPS inference.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Use CUDA kernels if available, else pure PyTorch
        try:
            from mamba_ssm import Mamba2
            self.ssm = Mamba2(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                chunk_size=32,
            )
            self._use_cuda_ssm = True
            log.info(f"  MambaBlock: using mamba_ssm.Mamba2 CUDA kernels (d={d_model})")
        except (ImportError, Exception):
            self.ssm = SelectiveSSMBlock(d_model, d_state, d_conv, expand)
            self._use_cuda_ssm = False
            log.info(f"  MambaBlock: using pure-PyTorch SSM fallback (d={d_model})")

    def forward(self, x: Tensor) -> Tensor:
        return x + self.dropout(self.ssm(self.norm(x)))


# ──────────────────────────────────────────────────────────────────────
# Interleaved Mamba + Graph Block (for medium branch Full Graph-SSM)
# ──────────────────────────────────────────────────────────────────────

class InterleavedMambaGraphBlock(nn.Module):
    """Mamba temporal + Graph spatial at every layer.

    Each layer first applies a MambaBlock for temporal processing along
    the time axis, then a SpatialGraphAttentionV3 for spatial processing
    across stations. This interleaving lets the model learn spatial-
    temporal interactions like sea-breeze onset propagation.

    Parameters
    ----------
    d_model : int
        Model dimension.
    d_state : int
        SSM state dimension.
    n_heads : int
        Number of graph attention heads.
    dropout : float
        Dropout rate.
    graph_config : GraphMambaV3Config or None
        Config for graph attention layer.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        n_heads: int = 14,
        dropout: float = 0.1,
        graph_config: GraphMambaV3Config | None = None,
    ) -> None:
        super().__init__()
        from engine.ds3m.graph_mamba_v3 import SpatialGraphAttentionV3

        self.mamba = MambaBlock(d_model, d_state, d_conv, expand, dropout)
        self.graph_attn = SpatialGraphAttentionV3(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            use_dynamic_edges=graph_config.use_dynamic_edges if graph_config else True,
            rope_base=graph_config.rope_base if graph_config else 10000.0,
        )

    def forward(
        self,
        x: Tensor,
        adj: Tensor | None = None,
        wind_dirs: Tensor | None = None,
        bearings: Tensor | None = None,
    ) -> Tensor:
        """Forward pass: temporal Mamba then spatial graph attention.

        Parameters
        ----------
        x : Tensor
            Shape (B, T, N_stations, d_model) — spatiotemporal tensor.
        adj : Tensor or None
            Shape (N, N) — static adjacency matrix.
        wind_dirs : Tensor or None
            Shape (B, N) — wind direction per station.
        bearings : Tensor or None
            Shape (N, N) — bearing matrix.

        Returns
        -------
        Tensor
            Shape (B, T, N_stations, d_model).
        """
        B, T, N, D = x.shape

        # Temporal: reshape to (B*N, T, D) so MambaBlock processes each station's time series
        x_time = x.permute(0, 2, 1, 3).reshape(B * N, T, D)
        x_time = self.mamba(x_time)
        x_time = x_time.reshape(B, N, T, D).permute(0, 2, 1, 3)  # (B, T, N, D)

        # Spatial: reshape to (B*T, N, D) so graph attention processes each timestep's stations
        if adj is not None:
            x_space = x_time.reshape(B * T, N, D)
            x_space = self.graph_attn(
                x_space, adj,
                wind_dirs=wind_dirs.unsqueeze(1).expand(B, T, N).reshape(B * T, N) if wind_dirs is not None else None,
                bearings=bearings,
            )
            x_time = x_space.reshape(B, T, N, D)

        return x_time


# ──────────────────────────────────────────────────────────────────────
# Temporal Branch (single resolution)
# ──────────────────────────────────────────────────────────────────────

class TemporalBranch(nn.Module):
    """Single-resolution temporal Mamba branch.

    Processes a fixed-resolution time series through an input projection
    and a stack of Mamba blocks. Optionally interleaves graph attention
    at every layer (for the medium branch's Full Graph-SSM mode).

    Parameters
    ----------
    config : MambaBranchConfig
        Branch hyperparameters.
    interleave_graph : bool
        If True, use InterleavedMambaGraphBlock instead of plain MambaBlock.
        Only the medium branch uses this.
    graph_config : GraphMambaV3Config or None
        Config for graph attention layers (required if interleave_graph=True).
    """

    def __init__(
        self,
        config: MambaBranchConfig,
        interleave_graph: bool = False,
        graph_config: GraphMambaV3Config | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.interleave_graph = interleave_graph
        self.input_proj = nn.Linear(config.d_input, config.d_model)

        if interleave_graph:
            if graph_config is None:
                graph_config = GraphMambaV3Config()
            self.layers = nn.ModuleList([
                InterleavedMambaGraphBlock(
                    d_model=config.d_model,
                    d_state=config.d_state,
                    d_conv=config.d_conv,
                    expand=config.expand,
                    n_heads=graph_config.n_heads,
                    dropout=config.dropout,
                    graph_config=graph_config,
                )
                for _ in range(config.n_layers)
            ])
        else:
            self.layers = nn.ModuleList([
                MambaBlock(
                    config.d_model, config.d_state, config.d_conv,
                    config.expand, config.dropout,
                )
                for _ in range(config.n_layers)
            ])
        self.output_norm = nn.LayerNorm(config.d_model)

        n_params = sum(p.numel() for p in self.parameters())
        log.info(
            f"TemporalBranch: {n_params:,} params, "
            f"d_model={config.d_model}, d_state={config.d_state}, "
            f"n_layers={config.n_layers}, seq_len={config.seq_len}, "
            f"interleave_graph={interleave_graph}"
        )

    def forward(
        self,
        x: Tensor,
        adj: Tensor | None = None,
        wind_dirs: Tensor | None = None,
        bearings: Tensor | None = None,
        station_embed: Tensor | None = None,
    ) -> Tensor:
        """Process temporal sequence.

        Parameters
        ----------
        x : Tensor
            Shape (batch, seq_len, d_input) for standard mode, or
            (batch, seq_len, N_stations, d_input) for interleaved mode.
        adj : Tensor or None
            Shape (N, N) — adjacency (only used in interleaved mode).
        wind_dirs : Tensor or None
            Shape (B, N) — wind direction per station (interleaved mode).
        bearings : Tensor or None
            Shape (N, N) — bearing matrix (interleaved mode).
        station_embed : Tensor or None
            Shape (N, d_model) — station embeddings to add after input projection.

        Returns
        -------
        Tensor
            Shape (batch, seq_len, d_model) for standard mode, or
            (batch, seq_len, N_stations, d_model) for interleaved mode.
        """
        from torch.utils.checkpoint import checkpoint

        if self.interleave_graph:
            # x: (B, T, N, d_input) -> project -> (B, T, N, d_model)
            h = self.input_proj(x)
            # Add station embeddings
            if station_embed is not None:
                h = h + station_embed.unsqueeze(0).unsqueeze(0)  # (1, 1, N, d_model)
            for layer in self.layers:
                h = checkpoint(
                    layer, h, adj, wind_dirs, bearings,
                    use_reentrant=False,
                )
            return self.output_norm(h)
        else:
            h = self.input_proj(x)
            for layer in self.layers:
                h = checkpoint(layer, h, use_reentrant=False)
            return self.output_norm(h)


# ──────────────────────────────────────────────────────────────────────
# Hybrid Mamba-Transformer Fusion Block
# ──────────────────────────────────────────────────────────────────────

class HybridFusionBlock(nn.Module):
    """Alternates Mamba (selective forgetting) + Transformer (global routing).

    Jamba/SST hybrid style: Mamba layers handle sequential dependencies
    and selective state compression, while Transformer layers handle
    global cross-resolution routing via multi-head attention.

    Parameters
    ----------
    d_model : int
        Model dimension.
    n_heads : int
        Number of attention heads for Transformer layers.
    d_state : int
        SSM state dimension for Mamba layers.
    d_conv : int
        Convolution width for Mamba layers.
    expand : int
        Expansion factor for Mamba layers.
    dropout : float
        Dropout rate.
    dim_feedforward : int
        FFN hidden dimension in Transformer layers.
    """

    def __init__(
        self,
        d_model: int = 896,
        n_heads: int = 12,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        dim_feedforward: int = 2048,
    ) -> None:
        super().__init__()

        # Layer 1: Mamba (selective forgetting)
        self.mamba1 = MambaBlock(d_model, d_state, d_conv, expand, dropout)

        # Layer 2: Transformer (global routing)
        self.transformer1 = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        # Layer 3: Mamba (selective forgetting)
        self.mamba2 = MambaBlock(d_model, d_state, d_conv, expand, dropout)

        # Layer 4: Transformer (global routing)
        self.transformer2 = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Process concatenated multi-resolution tokens.

        Parameters
        ----------
        x : Tensor
            Shape (batch, total_tokens, d_model) — concatenated
            fine + medium + coarse tokens.

        Returns
        -------
        Tensor
            Shape (batch, total_tokens, d_model).
        """
        x = self.mamba1(x)
        x = self.transformer1(x)
        x = self.mamba2(x)
        x = self.transformer2(x)
        return x


class SSTFusionBlock(nn.Module):
    """SST-style State Space Transformer fusion.
    Learned router softly mixes long-range Mamba expert + short-range Transformer expert.
    Beats naive alternation by 8-12% on multi-scale time series.
    """

    def __init__(self, d_model: int, n_heads: int, d_state: int = 64, regime_dim: int = 8) -> None:
        super().__init__()
        # Use CUDA Mamba2 kernels if available (1000x faster, 950x less memory)
        try:
            from mamba_ssm import Mamba2
            self.mamba_expert = Mamba2(
                d_model=d_model, d_state=d_state, d_conv=4,
                expand=2, chunk_size=32,
            )
            log.info(f"  SSTFusionBlock: using mamba_ssm.Mamba2 CUDA kernels (d={d_model})")
        except (ImportError, Exception):
            self.mamba_expert = SelectiveSSMBlock(d_model, d_state)
            log.info(f"  SSTFusionBlock: using pure-PyTorch SSM fallback (d={d_model})")
        self.transformer_expert = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True, norm_first=True
        )
        # Learned router conditioned on token + regime
        self.router = nn.Sequential(
            nn.Linear(d_model + regime_dim + 1, 128),  # +1 for branch_type flag
            nn.SiLU(),
            nn.Linear(128, 2),  # 2 experts: mamba vs transformer
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: Tensor,
        regime_posterior: Tensor | None = None,
        branch_type: Tensor | None = None,
    ) -> Tensor:
        # regime_posterior: (B, regime_dim) -> expand to (B, T, regime_dim)
        # branch_type: (B, T, 1) already per-token, or (B, 1) scalar to expand
        if regime_posterior is None:
            # Infer regime_dim from router input size: d_model + regime_dim + 1
            _regime_dim = self.router[0].in_features - x.size(-1) - 1
            regime_posterior = torch.zeros(x.size(0), _regime_dim, device=x.device)
        regime_expanded = regime_posterior.unsqueeze(1).expand(-1, x.size(1), -1)
        if branch_type is None:
            branch_expanded = torch.zeros(x.size(0), x.size(1), 1, device=x.device)
        elif branch_type.dim() == 2:
            # (B, 1) -> (B, T, 1)
            branch_expanded = branch_type.unsqueeze(1).expand(-1, x.size(1), -1)
        else:
            # Already (B, T, 1)
            branch_expanded = branch_type
        router_input = torch.cat([x, regime_expanded, branch_expanded], dim=-1)
        weights = F.softmax(self.router(router_input), dim=-1)  # (B, T, 2)

        mamba_out = self.mamba_expert(x)
        transformer_out = self.transformer_expert(x)

        mixed = weights[..., 0:1] * mamba_out + weights[..., 1:2] * transformer_out
        return self.norm(x + mixed)


class SSTFusionStack(nn.Module):
    """Stack of 4 SSTFusionBlock layers replacing the old HybridFusionBlock.

    Parameters
    ----------
    d_model : int
        Model dimension.
    n_heads : int
        Number of attention heads.
    d_state : int
        SSM state dimension.
    regime_dim : int
        Dimension of regime posterior vector.
    n_layers : int
        Number of SST fusion layers.
    """

    def __init__(
        self,
        d_model: int = 896,
        n_heads: int = 12,
        d_state: int = 64,
        regime_dim: int = 8,
        n_layers: int = 4,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            SSTFusionBlock(d_model, n_heads, d_state, regime_dim)
            for _ in range(n_layers)
        ])

    def forward(
        self,
        x: Tensor,
        regime_posterior: Tensor | None = None,
        branch_type: Tensor | None = None,
    ) -> Tensor:
        for layer in self.layers:
            x = layer(x, regime_posterior=regime_posterior, branch_type=branch_type)
        return x


# ──────────────────────────────────────────────────────────────────────
# Multi-Resolution Mamba (full assembly)
# ──────────────────────────────────────────────────────────────────────

class MultiResolutionMamba(nn.Module):
    """Multi-Resolution Temporal Mamba for Weather Brain v3.1.

    Three parallel branches at different temporal resolutions, fused
    via hybrid Mamba-Transformer cross-attention.

    Branches:
      Fine   — 15-min resolution, 32 steps (HRRR sub-hourly + RTMA-RU + ASOS)
      Medium — hourly resolution, 96 steps (full NWP + obs + buoy)
      Coarse — 3-hourly resolution, 112 steps (GFS/ECMWF/MOS/soundings)

    The fusion mechanism allows bidirectional information flow between
    resolutions: fine-grained signals inform coarse trends, and coarse
    patterns provide context for fine-scale predictions.

    Parameters
    ----------
    fine_config : MambaBranchConfig
        Configuration for the fine (15-min) branch.
    medium_config : MambaBranchConfig
        Configuration for the medium (hourly) branch.
    coarse_config : MambaBranchConfig
        Configuration for the coarse (3-hourly) branch.
    fusion_config : FusionConfig
        Configuration for cross-resolution fusion.
    """

    def __init__(
        self,
        fine_config: MambaBranchConfig | None = None,
        medium_config: MambaBranchConfig | None = None,
        coarse_config: MambaBranchConfig | None = None,
        fusion_config: FusionConfig | None = None,
    ) -> None:
        super().__init__()

        # Use defaults if not provided
        if fine_config is None:
            fine_config = MambaBranchConfig(
                d_input=64, d_model=640, d_state=64, n_layers=8, seq_len=32,
            )
        if medium_config is None:
            medium_config = MambaBranchConfig(
                d_input=64, d_model=896, d_state=64, n_layers=12, seq_len=96,
            )
        if coarse_config is None:
            coarse_config = MambaBranchConfig(
                d_input=32, d_model=512, d_state=64, n_layers=6, seq_len=112,
            )
        if fusion_config is None:
            fusion_config = FusionConfig()

        self.fine_config = fine_config
        self.medium_config = medium_config
        self.coarse_config = coarse_config
        self.fusion_config = fusion_config

        # ── Graph config for interleaved medium branch ───────────
        graph_config = GraphMambaV3Config()

        # ── Temporal branches ─────────────────────────────────────
        self.fine_branch = TemporalBranch(fine_config)
        self.medium_branch = TemporalBranch(
            medium_config,
            interleave_graph=True,
            graph_config=graph_config,
        )
        self.coarse_branch = TemporalBranch(coarse_config)

        # ── Graph structure for interleaved medium branch ─────────
        from engine.ds3m.graph_mamba_v3 import (
            ALL_STATIONS_V3,
            build_adjacency_matrix_v3,
            build_bearing_matrix_v3,
        )
        self.interleave_stations = ALL_STATIONS_V3
        self.interleave_n_stations = len(ALL_STATIONS_V3)
        self.interleave_target_idx = ALL_STATIONS_V3.index("KMIA")
        self.interleave_station_embed = nn.Embedding(
            self.interleave_n_stations, medium_config.d_model
        )
        adj = build_adjacency_matrix_v3(ALL_STATIONS_V3, graph_config.max_distance_km)
        bearings = build_bearing_matrix_v3(ALL_STATIONS_V3)
        self.register_buffer("interleave_adj", adj)
        self.register_buffer("interleave_bearings", bearings)

        # ── Projection layers to fusion dimension ─────────────────
        d_fusion = fusion_config.d_model
        self.fine_proj = nn.Linear(fine_config.d_model, d_fusion)
        self.medium_proj = nn.Linear(medium_config.d_model, d_fusion)
        self.coarse_proj = nn.Linear(coarse_config.d_model, d_fusion)

        # ── Resolution embeddings (learnable) ─────────────────────
        self.resolution_embed = nn.Embedding(3, d_fusion)  # 0=fine, 1=medium, 2=coarse

        # ── Cross-resolution fusion (SST-style learned routing) ──
        self.fusion_block = SSTFusionStack(
            d_model=fusion_config.d_model,
            n_heads=fusion_config.n_heads,
            d_state=fusion_config.d_state,
            regime_dim=fusion_config.regime_dim,
            n_layers=4,
        )

        # ── Output projection (back to medium branch dim for downstream) ──
        self.output_norm = nn.LayerNorm(d_fusion)

        # ── Parameter counting ────────────────────────────────────
        n_fine = sum(p.numel() for p in self.fine_branch.parameters())
        n_medium = sum(p.numel() for p in self.medium_branch.parameters())
        n_coarse = sum(p.numel() for p in self.coarse_branch.parameters())
        n_fusion = (
            sum(p.numel() for p in self.fusion_block.parameters())
            + sum(p.numel() for p in self.fine_proj.parameters())
            + sum(p.numel() for p in self.medium_proj.parameters())
            + sum(p.numel() for p in self.coarse_proj.parameters())
            + sum(p.numel() for p in self.resolution_embed.parameters())
            + sum(p.numel() for p in self.output_norm.parameters())
        )
        n_total = sum(p.numel() for p in self.parameters())

        log.info(
            f"MultiResolutionMamba: {n_total:,} total parameters\n"
            f"  Fine branch:   {n_fine:,} params "
            f"(d={fine_config.d_model}, L={fine_config.n_layers}, seq={fine_config.seq_len})\n"
            f"  Medium branch: {n_medium:,} params "
            f"(d={medium_config.d_model}, L={medium_config.n_layers}, seq={medium_config.seq_len})\n"
            f"  Coarse branch: {n_coarse:,} params "
            f"(d={coarse_config.d_model}, L={coarse_config.n_layers}, seq={coarse_config.seq_len})\n"
            f"  Fusion:        {n_fusion:,} params "
            f"(d={fusion_config.d_model}, heads={fusion_config.n_heads})"
        )

    def forward(
        self,
        fine_input: Tensor,
        medium_input: Tensor,
        coarse_input: Tensor,
        regime_posterior: Tensor | None = None,
        wind_dirs: Tensor | None = None,
    ) -> Tensor:
        """Process multi-resolution temporal inputs and fuse.

        Parameters
        ----------
        fine_input : Tensor
            Shape (batch, 32, d_input_fine) — 15-min resolution features.
        medium_input : Tensor
            Shape (batch, 96, d_input_medium) — hourly resolution features.
        coarse_input : Tensor
            Shape (batch, 112, d_input_coarse) — 3-hourly resolution features.
        regime_posterior : Tensor or None
            Shape (batch, regime_dim) — regime posterior for SST router.
            If None, defaults to zeros.
        wind_dirs : Tensor or None
            Shape (B, N_stations) — wind direction per station (for graph attention).

        Returns
        -------
        Tensor
            Shape (batch, d_fusion) — fused temporal state (last token pooled).
        """
        batch = fine_input.shape[0]
        device = fine_input.device
        regime_dim = getattr(self.fusion_block.layers[0], 'router', None)
        # Infer regime_dim from the first SSTFusionBlock router input size
        _sst_regime_dim = self.fusion_block.layers[0].router[0].in_features - self.fusion_config.d_model - 1
        if regime_posterior is None:
            regime_posterior = torch.zeros(batch, _sst_regime_dim, device=device)

        # ── Step 1: Process each branch independently ─────────────
        h_fine = self.fine_branch(fine_input)      # (B, 32, 640)

        # Medium branch: interleaved Mamba+Graph — expand to (B, T, N, d_input)
        N = self.interleave_n_stations
        medium_expanded = medium_input.unsqueeze(2).expand(
            batch, medium_input.shape[1], N, medium_input.shape[2]
        )  # (B, T, N, d_input)
        station_embed = self.interleave_station_embed(
            torch.arange(N, device=device)
        )  # (N, d_model)
        h_medium_full = self.medium_branch(
            medium_expanded,
            adj=self.interleave_adj,
            wind_dirs=wind_dirs,
            bearings=self.interleave_bearings,
            station_embed=station_embed,
        )  # (B, T, N, d_model)
        # Pool: extract KMIA station to get (B, T, d_model)
        h_medium = h_medium_full[:, :, self.interleave_target_idx, :]  # (B, 96, 896)

        h_coarse = self.coarse_branch(coarse_input)  # (B, 112, 512)

        # ── Step 2: Project to common fusion dimension ────────────
        h_fine_proj = self.fine_proj(h_fine)        # (B, 32, d_fusion)
        h_medium_proj = self.medium_proj(h_medium)  # (B, 96, d_fusion)
        h_coarse_proj = self.coarse_proj(h_coarse)  # (B, 112, d_fusion)

        # ── Step 3: Add resolution embeddings ─────────────────────
        res_ids = torch.arange(3, device=device)
        res_embeds = self.resolution_embed(res_ids)  # (3, d_fusion)

        h_fine_proj = h_fine_proj + res_embeds[0].unsqueeze(0).unsqueeze(0)
        h_medium_proj = h_medium_proj + res_embeds[1].unsqueeze(0).unsqueeze(0)
        h_coarse_proj = h_coarse_proj + res_embeds[2].unsqueeze(0).unsqueeze(0)

        # ── Step 4: Concatenate and fuse (SST-style routing) ──────
        # Build per-token branch_type indicator for SST router
        n_fine = fine_input.shape[1]
        n_medium = medium_input.shape[1]
        n_coarse = coarse_input.shape[1]
        branch_type = torch.cat([
            torch.zeros(batch, n_fine, 1, device=device),    # 0 = fine
            torch.ones(batch, n_medium, 1, device=device),   # 1 = medium
            torch.full((batch, n_coarse, 1), 2.0, device=device),  # 2 = coarse
        ], dim=1)  # (B, 240, 1)

        fused = torch.cat(
            [h_fine_proj, h_medium_proj, h_coarse_proj], dim=1
        )  # (B, 32+96+112=240, d_fusion)

        fused = self.fusion_block(
            fused, regime_posterior=regime_posterior, branch_type=branch_type
        )  # (B, 240, d_fusion)

        # ── Step 5: Extract medium branch output (last token) ─────
        medium_start = n_fine
        medium_end = medium_start + n_medium
        h_medium_fused = fused[:, medium_start:medium_end, :]  # (B, 96, d_fusion)

        # Pool: take last token of the medium branch
        output = self.output_norm(h_medium_fused[:, -1, :])  # (B, d_fusion)

        return output

    def forward_full_sequence(
        self,
        fine_input: Tensor,
        medium_input: Tensor,
        coarse_input: Tensor,
        regime_posterior: Tensor | None = None,
        wind_dirs: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Full-sequence forward returning all branch outputs after fusion.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            (fine_fused, medium_fused, coarse_fused) — each branch's
            fused representation across all timesteps.
        """
        batch = fine_input.shape[0]
        device = fine_input.device
        _sst_regime_dim = self.fusion_block.layers[0].router[0].in_features - self.fusion_config.d_model - 1
        if regime_posterior is None:
            regime_posterior = torch.zeros(batch, _sst_regime_dim, device=device)

        h_fine = self.fine_branch(fine_input)

        # Medium branch: interleaved Mamba+Graph
        N = self.interleave_n_stations
        medium_expanded = medium_input.unsqueeze(2).expand(
            batch, medium_input.shape[1], N, medium_input.shape[2]
        )
        station_embed = self.interleave_station_embed(
            torch.arange(N, device=device)
        )
        h_medium_full = self.medium_branch(
            medium_expanded,
            adj=self.interleave_adj,
            wind_dirs=wind_dirs,
            bearings=self.interleave_bearings,
            station_embed=station_embed,
        )
        h_medium = h_medium_full[:, :, self.interleave_target_idx, :]

        h_coarse = self.coarse_branch(coarse_input)

        h_fine_proj = self.fine_proj(h_fine)
        h_medium_proj = self.medium_proj(h_medium)
        h_coarse_proj = self.coarse_proj(h_coarse)

        res_ids = torch.arange(3, device=device)
        res_embeds = self.resolution_embed(res_ids)
        h_fine_proj = h_fine_proj + res_embeds[0]
        h_medium_proj = h_medium_proj + res_embeds[1]
        h_coarse_proj = h_coarse_proj + res_embeds[2]

        n_fine = fine_input.shape[1]
        n_medium = medium_input.shape[1]
        n_coarse = coarse_input.shape[1]

        # Build per-token branch_type indicator for SST router
        branch_type = torch.cat([
            torch.zeros(batch, n_fine, 1, device=device),
            torch.ones(batch, n_medium, 1, device=device),
            torch.full((batch, n_coarse, 1), 2.0, device=device),
        ], dim=1)

        fused = torch.cat([h_fine_proj, h_medium_proj, h_coarse_proj], dim=1)
        fused = self.fusion_block(
            fused, regime_posterior=regime_posterior, branch_type=branch_type
        )

        fine_out = fused[:, :n_fine, :]
        medium_out = fused[:, n_fine:n_fine + n_medium, :]
        coarse_out = fused[:, n_fine + n_medium:, :]

        return fine_out, medium_out, coarse_out
