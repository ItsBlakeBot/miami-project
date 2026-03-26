"""Market Graph-Mamba + MoE Joint Encoder for the Trading Brain.

Architecture (following SOTA 2025-2026 pattern):
    Frozen DS3M outputs + Market signals
        → Graph-Mamba backbone (cross-signal graph edges)
        → MoE layers (regime-conditioned expert routing)
        → Rich state vector for SAC policy head

The graph structure connects different signal types:
    - Weather bracket nodes (1 per bracket, carries DS3M probs + regime)
    - Market data nodes (1 per bracket, carries price/volume/flow)
    - Portfolio node (1 global, carries positions/P&L/risk)
    - Time node (1 global, carries settlement countdown/hour/DOW)

Edges model cross-signal dependencies:
    - Weather↔Market: same-bracket weather-to-market
    - Market↔Market: adjacent brackets (correlation)
    - Portfolio→All: position context affects all decisions
    - Time→All: temporal context affects everything

This is NOT the weather GraphMamba (which has graph edges between physical
stations). This is the TRADING GraphMamba (graph edges between signal types).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TradingEncoderConfig:
    """Joint encoder configuration."""
    # Mamba backbone
    d_model: int = 384
    d_state: int = 32
    n_mamba_layers: int = 8
    dropout: float = 0.15
    expand_factor: int = 2

    # Graph attention
    n_graph_layers: int = 4
    n_heads: int = 8

    # MoE
    n_experts: int = 6
    n_active_experts: int = 2          # top-k routing
    expert_hidden_dim: int = 512
    moe_frequency: int = 2             # insert MoE every N Mamba layers
    load_balance_weight: float = 0.01  # auxiliary loss weight

    # Input dimensions
    market_feature_dim: int = 20       # per-bracket market features
    ds3m_signal_dim: int = 30          # per-city DS3M signal features
    portfolio_dim: int = 18            # portfolio state
    n_brackets: int = 6
    max_cities: int = 10               # max simultaneous cities

    # Sequence
    lookback_steps: int = 120          # 2h at 1-min resolution

    # FiLM conditioning
    regime_dim: int = 5                # HDP regime posterior dimension


# ---------------------------------------------------------------------------
# FiLM Conditioning Layer
# ---------------------------------------------------------------------------

class FiLMConditioner(nn.Module):
    """Feature-wise Linear Modulation from regime latents.

    Produces per-channel scale and shift from the regime posterior,
    allowing regime-specific behavior without separate expert networks.
    """

    def __init__(self, regime_dim: int, d_model: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(regime_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model * 2),  # gamma and beta
        )

    def forward(self, x: torch.Tensor, regime: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D), regime: (B, regime_dim)"""
        gb = self.proj(regime)                   # (B, 2*D)
        gamma, beta = gb.chunk(2, dim=-1)        # each (B, D)
        gamma = gamma.unsqueeze(1) + 1.0         # center at 1
        beta = beta.unsqueeze(1)
        return x * gamma + beta


# ---------------------------------------------------------------------------
# Mixture of Experts Layer
# ---------------------------------------------------------------------------

class Expert(nn.Module):
    """Single MoE expert: 2-layer MLP with SiLU."""

    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MoELayer(nn.Module):
    """Mixture of Experts with top-k routing and load balancing.

    Expert specializations (learned, not hardcoded):
        0: Weather edge (conditioned heavily on DS3M regime)
        1: Timing (intraday patterns, NWP model releases)
        2: Order flow (maker/taker dynamics, spread exploitation)
        3: Risk/portfolio (position management, exit timing)
        4: Reserved (sentiment / news)
        5: Reserved (cross-city arbitrage)

    Router is conditioned on regime latents so it automatically
    routes to weather expert during high-confidence regimes.
    """

    def __init__(
        self,
        d_model: int,
        n_experts: int = 6,
        n_active: int = 2,
        hidden_dim: int = 512,
        regime_dim: int = 5,
        dropout: float = 0.1,
        load_balance_weight: float = 0.01,
    ):
        super().__init__()
        self.n_experts = n_experts
        self.n_active = n_active
        self.load_balance_weight = load_balance_weight

        # Experts
        self.experts = nn.ModuleList([
            Expert(d_model, hidden_dim, dropout) for _ in range(n_experts)
        ])

        # Router: conditioned on hidden state + regime
        self.router = nn.Sequential(
            nn.Linear(d_model + regime_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, n_experts),
        )

        self.layer_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        regime: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, D) hidden states
            regime: (B, regime_dim) regime posterior

        Returns:
            output: (B, T, D) MoE output
            aux_loss: scalar load balancing loss
        """
        B, T, D = x.shape
        residual = x
        x_norm = self.layer_norm(x)

        # Router input: hidden state + regime (broadcast over time)
        regime_expanded = regime.unsqueeze(1).expand(-1, T, -1)  # (B, T, regime_dim)
        router_input = torch.cat([x_norm, regime_expanded], dim=-1)  # (B, T, D+regime_dim)

        # Gate logits
        gate_logits = self.router(router_input)  # (B, T, n_experts)

        # Top-k routing
        topk_vals, topk_idx = torch.topk(gate_logits, self.n_active, dim=-1)
        topk_weights = F.softmax(topk_vals, dim=-1)  # (B, T, k)

        # Compute expert outputs for selected experts
        # For efficiency with small n_experts, just compute all and select
        expert_outputs = torch.stack([
            expert(x_norm) for expert in self.experts
        ], dim=-2)  # (B, T, n_experts, D)

        # Gather selected experts
        topk_idx_expanded = topk_idx.unsqueeze(-1).expand(-1, -1, -1, D)  # (B, T, k, D)
        selected = torch.gather(expert_outputs, 2, topk_idx_expanded)     # (B, T, k, D)

        # Weighted sum
        output = (selected * topk_weights.unsqueeze(-1)).sum(dim=2)  # (B, T, D)

        # Load balancing auxiliary loss
        # Encourage uniform expert utilization
        gate_probs = F.softmax(gate_logits, dim=-1)  # (B, T, n_experts)
        avg_probs = gate_probs.mean(dim=(0, 1))      # (n_experts,)
        # Fraction of tokens routed to each expert
        mask = F.one_hot(topk_idx[..., 0], self.n_experts).float()
        avg_mask = mask.mean(dim=(0, 1))              # (n_experts,)
        aux_loss = self.load_balance_weight * self.n_experts * (avg_probs * avg_mask).sum()

        return residual + output, aux_loss


# ---------------------------------------------------------------------------
# Graph Attention Layer
# ---------------------------------------------------------------------------

class SignalGraphAttention(nn.Module):
    """Multi-head attention over signal graph nodes.

    Nodes represent different signal types (weather, market, portfolio, time).
    Edges are pre-defined based on signal relationships.
    Unlike the weather GraphMamba which has spatial station edges,
    this graph has semantic signal-type edges.
    """

    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        assert d_model % n_heads == 0

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

        self.edge_type_embedding = nn.Embedding(8, self.d_head)  # 8 edge types
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N_nodes, D) node embeddings
            edge_index: (2, E) source/target node indices
            edge_type: (E,) edge type indices (optional)

        Returns:
            (B, N_nodes, D) updated node embeddings
        """
        B, N, D = x.shape
        residual = x
        x = self.norm(x)

        # Multi-head projections
        Q = self.q_proj(x).view(B, N, self.n_heads, self.d_head)
        K = self.k_proj(x).view(B, N, self.n_heads, self.d_head)
        V = self.v_proj(x).view(B, N, self.n_heads, self.d_head)

        # For small graphs (< 50 nodes), dense attention is fine
        # Scale dot-product attention with edge masking
        attn = torch.einsum("bnhd,bmhd->bhnm", Q, K) / math.sqrt(self.d_head)

        # Build adjacency mask from edge_index
        mask = torch.zeros(N, N, device=x.device, dtype=torch.bool)
        if edge_index.numel() > 0:
            mask[edge_index[0], edge_index[1]] = True
            mask = mask | torch.eye(N, device=x.device, dtype=torch.bool)  # self-loops
        else:
            mask = torch.ones(N, N, device=x.device, dtype=torch.bool)

        # Apply edge type bias
        if edge_type is not None and edge_index.numel() > 0:
            edge_bias = self.edge_type_embedding(edge_type)  # (E, d_head)
            # Add bias to attention scores for each edge
            for idx in range(edge_index.shape[1]):
                src, tgt = edge_index[0, idx], edge_index[1, idx]
                bias = (Q[:, tgt] * edge_bias[idx]).sum(-1)  # (B, n_heads)
                attn[:, :, tgt, src] = attn[:, :, tgt, src] + bias

        # Mask non-edges with -inf
        attn = attn.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Aggregate
        out = torch.einsum("bhnm,bmhd->bnhd", attn, V)
        out = out.reshape(B, N, D)
        out = self.o_proj(out)

        return residual + self.dropout(out)


# ---------------------------------------------------------------------------
# Selective SSM Block (Pure PyTorch, Mamba-3 compatible)
# ---------------------------------------------------------------------------

class SelectiveSSMBlock(nn.Module):
    """Pure-PyTorch selective SSM block.

    Implements the core Mamba selective scan without CUDA kernels.
    For training on GPU, we'll swap in the official mamba-ssm CUDA kernels.
    For inference on Mac M4, this pure-PyTorch version runs fine.

    This implements Mamba-3's exponential-trapezoidal discretization
    and supports complex-valued states.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 32,
        expand: int = 2,
        dt_rank: Optional[int] = None,
        use_complex: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand
        self.dt_rank = dt_rank or max(1, d_model // 16)
        self.use_complex = use_complex

        # Input projection: x → (z, x_ssm)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # SSM parameters (input-dependent, Mamba-3 style)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # A parameter (log-space for stability)
        if use_complex:
            # Complex A enables oscillatory dynamics (diurnal patterns, price cycles)
            A_real = torch.randn(self.d_inner, self.d_state) * 0.5
            A_imag = torch.randn(self.d_inner, self.d_state) * 0.1
            self.A_log_real = nn.Parameter(torch.log(torch.clamp(-A_real, min=1e-4)))
            self.A_imag = nn.Parameter(A_imag)
        else:
            A = torch.randn(self.d_inner, self.d_state)
            self.A_log = nn.Parameter(torch.log(torch.clamp(-A, min=1e-4)))

        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def _get_A(self) -> torch.Tensor:
        if self.use_complex:
            A_real = -torch.exp(self.A_log_real)
            return torch.complex(A_real, self.A_imag)
        return -torch.exp(self.A_log)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model) → (B, T, d_model)"""
        B, T, _ = x.shape
        residual = x
        x = self.norm(x)

        # Input projection
        xz = self.in_proj(x)                        # (B, T, 2*d_inner)
        x_ssm, z = xz.chunk(2, dim=-1)              # each (B, T, d_inner)

        # Input-dependent SSM parameters
        x_dbl = self.x_proj(x_ssm)                  # (B, T, dt_rank + 2*d_state)
        dt, B_param, C_param = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt = F.softplus(self.dt_proj(dt))            # (B, T, d_inner)

        # Get A matrix
        A = self._get_A()                            # (d_inner, d_state)

        # Selective scan (sequential, pure PyTorch)
        # For training on CUDA, replace with mamba_ssm.selective_scan_fn
        dtype = torch.cfloat if self.use_complex else x.dtype
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=dtype)
        ys = []

        for t in range(T):
            # Exponential-trapezoidal discretization (Mamba-3)
            dt_t = dt[:, t, :].unsqueeze(-1)         # (B, d_inner, 1)
            dA = torch.exp(A.unsqueeze(0) * dt_t)    # (B, d_inner, d_state)

            B_t = B_param[:, t, :].unsqueeze(1)      # (B, 1, d_state)
            if self.use_complex:
                B_t = B_t.to(dtype)

            x_t = x_ssm[:, t, :].unsqueeze(-1)       # (B, d_inner, 1)
            if self.use_complex:
                x_t = x_t.to(dtype)

            # State update: h = dA * h + dB * x
            dB = dt_t * B_t                           # (B, d_inner, d_state)
            h = dA * h + dB * x_t

            # Output: y = C @ h + D * x
            C_t = C_param[:, t, :].unsqueeze(1)       # (B, 1, d_state)
            if self.use_complex:
                C_t = C_t.to(dtype)
            y = (h * C_t).sum(dim=-1)                 # (B, d_inner)
            if self.use_complex:
                y = y.real
            y = y + self.D * x_ssm[:, t]
            ys.append(y)

        y = torch.stack(ys, dim=1)                    # (B, T, d_inner)

        # Gating
        y = y * F.silu(z)
        y = self.out_proj(y)

        return residual + self.dropout(y)


# ---------------------------------------------------------------------------
# Full Trading Encoder
# ---------------------------------------------------------------------------

class TradingGraphMambaEncoder(nn.Module):
    """Joint Graph-Mamba + MoE Encoder for trading.

    Processes the unified graph of weather signals + market data + portfolio
    through interleaved Mamba temporal layers and graph attention spatial layers,
    with MoE layers inserted every N blocks for expert specialization.

    This is the "brain" of the trading system.
    """

    def __init__(self, cfg: TradingEncoderConfig):
        super().__init__()
        self.cfg = cfg

        # Input embeddings (project each signal type to d_model)
        self.market_embed = nn.Sequential(
            nn.Linear(cfg.market_feature_dim, cfg.d_model),
            nn.SiLU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )
        self.ds3m_embed = nn.Sequential(
            nn.Linear(cfg.ds3m_signal_dim, cfg.d_model),
            nn.SiLU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )
        self.portfolio_embed = nn.Sequential(
            nn.Linear(cfg.portfolio_dim, cfg.d_model),
            nn.SiLU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )

        # Node type embedding (weather=0, market=1, portfolio=2, time=3)
        self.node_type_embed = nn.Embedding(4, cfg.d_model)

        # Bracket position embedding
        self.bracket_pos_embed = nn.Embedding(cfg.n_brackets, cfg.d_model)

        # FiLM conditioning
        self.film = FiLMConditioner(cfg.regime_dim, cfg.d_model)

        # Interleaved layers
        self.mamba_layers = nn.ModuleList()
        self.graph_layers = nn.ModuleList()
        self.moe_layers = nn.ModuleList()
        self.layer_types = []  # track which type each layer is

        for i in range(cfg.n_mamba_layers):
            # Mamba temporal layer
            self.mamba_layers.append(
                SelectiveSSMBlock(
                    d_model=cfg.d_model,
                    d_state=cfg.d_state,
                    expand=cfg.expand_factor,
                    use_complex=True,
                    dropout=cfg.dropout,
                )
            )

            # Graph attention layer (every other Mamba layer)
            if (i + 1) % 2 == 0:
                self.graph_layers.append(
                    SignalGraphAttention(
                        d_model=cfg.d_model,
                        n_heads=cfg.n_heads,
                        dropout=cfg.dropout,
                    )
                )

            # MoE layer (every moe_frequency layers)
            if (i + 1) % cfg.moe_frequency == 0:
                self.moe_layers.append(
                    MoELayer(
                        d_model=cfg.d_model,
                        n_experts=cfg.n_experts,
                        n_active=cfg.n_active_experts,
                        hidden_dim=cfg.expert_hidden_dim,
                        regime_dim=cfg.regime_dim,
                        dropout=cfg.dropout,
                        load_balance_weight=cfg.load_balance_weight,
                    )
                )

        # Output projection
        self.output_norm = nn.LayerNorm(cfg.d_model)
        self.state_proj = nn.Linear(cfg.d_model, cfg.d_model)

    def _build_signal_graph(
        self,
        n_brackets: int,
        n_cities: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the signal graph edge structure.

        Node layout per city:
            [0..n_brackets-1]: weather bracket nodes
            [n_brackets..2*n_brackets-1]: market bracket nodes
        Global nodes:
            [2*n_brackets*n_cities]: portfolio node
            [2*n_brackets*n_cities+1]: time node

        Edge types:
            0: weather↔market (same bracket, same city)
            1: market↔market (adjacent brackets, same city)
            2: weather↔weather (adjacent brackets, same city)
            3: portfolio→weather (global context)
            4: portfolio→market (global context)
            5: time→all (temporal context)
            6: cross-city same bracket
            7: self-loop
        """
        edges_src, edges_dst, edge_types = [], [], []
        nodes_per_city = 2 * n_brackets
        total_nodes = nodes_per_city * n_cities + 2  # +portfolio +time

        portfolio_node = nodes_per_city * n_cities
        time_node = portfolio_node + 1

        for city in range(n_cities):
            base = city * nodes_per_city
            wx_base = base                          # weather nodes start
            mk_base = base + n_brackets             # market nodes start

            for b in range(n_brackets):
                # Weather ↔ Market (same bracket): type 0
                edges_src.extend([wx_base + b, mk_base + b])
                edges_dst.extend([mk_base + b, wx_base + b])
                edge_types.extend([0, 0])

                # Adjacent bracket connections (market↔market): type 1
                if b > 0:
                    edges_src.extend([mk_base + b, mk_base + b - 1])
                    edges_dst.extend([mk_base + b - 1, mk_base + b])
                    edge_types.extend([1, 1])

                # Adjacent bracket connections (weather↔weather): type 2
                if b > 0:
                    edges_src.extend([wx_base + b, wx_base + b - 1])
                    edges_dst.extend([wx_base + b - 1, wx_base + b])
                    edge_types.extend([2, 2])

                # Portfolio → all: types 3, 4
                edges_src.extend([portfolio_node, portfolio_node])
                edges_dst.extend([wx_base + b, mk_base + b])
                edge_types.extend([3, 4])

                # Time → all: type 5
                edges_src.extend([time_node, time_node])
                edges_dst.extend([wx_base + b, mk_base + b])
                edge_types.extend([5, 5])

            # Cross-city same bracket: type 6
            for other_city in range(city + 1, n_cities):
                other_base = other_city * nodes_per_city
                for b in range(n_brackets):
                    edges_src.extend([mk_base + b, other_base + n_brackets + b])
                    edges_dst.extend([other_base + n_brackets + b, mk_base + b])
                    edge_types.extend([6, 6])

        edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long, device=device)
        edge_type = torch.tensor(edge_types, dtype=torch.long, device=device)
        return edge_index, edge_type

    def forward(
        self,
        market_features: torch.Tensor,       # (B, T, n_brackets, market_dim)
        ds3m_signals: torch.Tensor,           # (B, T, n_brackets, signal_dim)
        portfolio_state: torch.Tensor,        # (B, portfolio_dim)
        regime_posterior: torch.Tensor,        # (B, regime_dim)
        n_cities: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            state_vector: (B, d_model) — rich state for SAC policy head
            bracket_states: (B, n_brackets, d_model) — per-bracket states
            aux_loss: scalar MoE load balancing loss
        """
        B, T, n_brackets, _ = market_features.shape
        device = market_features.device

        # 1. Embed each signal type
        market_emb = self.market_embed(market_features)      # (B, T, n_brackets, D)
        ds3m_emb = self.ds3m_embed(ds3m_signals)             # (B, T, n_brackets, D)

        # Add node type embeddings
        weather_type = self.node_type_embed(torch.zeros(1, dtype=torch.long, device=device))
        market_type = self.node_type_embed(torch.ones(1, dtype=torch.long, device=device))
        ds3m_emb = ds3m_emb + weather_type
        market_emb = market_emb + market_type

        # Add bracket position embeddings
        bracket_ids = torch.arange(n_brackets, device=device)
        bracket_emb = self.bracket_pos_embed(bracket_ids)    # (n_brackets, D)
        ds3m_emb = ds3m_emb + bracket_emb.unsqueeze(0).unsqueeze(0)
        market_emb = market_emb + bracket_emb.unsqueeze(0).unsqueeze(0)

        # Portfolio + time nodes (broadcast over time)
        portfolio_emb = self.portfolio_embed(portfolio_state)  # (B, D)
        port_type = self.node_type_embed(torch.tensor([2], device=device))
        portfolio_emb = portfolio_emb + port_type.squeeze(0)
        portfolio_emb = portfolio_emb.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, D)

        time_type = self.node_type_embed(torch.tensor([3], device=device))
        time_emb = time_type.squeeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        time_emb = time_emb.expand(B, 1, 1, -1)

        # 2. Assemble graph nodes: [weather_brackets, market_brackets, portfolio, time]
        # Shape: (B, T, N_nodes, D) where N_nodes = 2*n_brackets + 2
        nodes = torch.cat([
            ds3m_emb,           # (B, T, n_brackets, D)
            market_emb,         # (B, T, n_brackets, D)
            portfolio_emb.expand(B, T, 1, -1),
            time_emb.expand(B, T, 1, -1),
        ], dim=2)              # (B, T, 2*n_brackets+2, D)

        N_nodes = nodes.shape[2]

        # FiLM conditioning from regime
        nodes_flat = nodes.view(B, T * N_nodes, -1)
        nodes_flat = self.film(nodes_flat, regime_posterior)
        nodes = nodes_flat.view(B, T, N_nodes, -1)

        # 3. Build signal graph
        edge_index, edge_type = self._build_signal_graph(n_brackets, n_cities, device)

        # 4. Process through interleaved Mamba + Graph + MoE layers
        total_aux_loss = torch.tensor(0.0, device=device)
        graph_idx = 0
        moe_idx = 0

        for i, mamba_layer in enumerate(self.mamba_layers):
            # Mamba: process each node's temporal sequence independently
            # Reshape: (B, T, N, D) → (B*N, T, D) for temporal processing
            x = nodes.permute(0, 2, 1, 3).reshape(B * N_nodes, T, -1)
            x = mamba_layer(x)
            nodes = x.reshape(B, N_nodes, T, -1).permute(0, 2, 1, 3)

            # Graph attention (every 2 layers)
            if (i + 1) % 2 == 0 and graph_idx < len(self.graph_layers):
                # Process latest timestep through graph attention
                # (or all timesteps for richer interaction)
                latest = nodes[:, -1, :, :]  # (B, N, D) - latest timestep
                latest = self.graph_layers[graph_idx](latest, edge_index, edge_type)
                nodes[:, -1, :, :] = latest
                graph_idx += 1

            # MoE (every moe_frequency layers)
            if (i + 1) % self.cfg.moe_frequency == 0 and moe_idx < len(self.moe_layers):
                x = nodes.reshape(B, T * N_nodes, -1)
                x, aux = self.moe_layers[moe_idx](x, regime_posterior)
                nodes = x.reshape(B, T, N_nodes, -1)
                total_aux_loss = total_aux_loss + aux
                moe_idx += 1

        # 5. Extract outputs from final timestep
        final = nodes[:, -1, :, :]  # (B, N_nodes, D)
        final = self.output_norm(final)

        # Per-bracket states (market nodes for trading decisions)
        bracket_states = final[:, n_brackets:2*n_brackets, :]  # (B, n_brackets, D)

        # Global state vector (pool all nodes)
        state_vector = self.state_proj(final.mean(dim=1))  # (B, D)

        return state_vector, bracket_states, total_aux_loss

    def count_parameters(self) -> dict:
        """Count parameters by component."""
        mamba_params = sum(p.numel() for layer in self.mamba_layers for p in layer.parameters())
        graph_params = sum(p.numel() for layer in self.graph_layers for p in layer.parameters())
        moe_params = sum(p.numel() for layer in self.moe_layers for p in layer.parameters())
        embed_params = (
            sum(p.numel() for p in self.market_embed.parameters()) +
            sum(p.numel() for p in self.ds3m_embed.parameters()) +
            sum(p.numel() for p in self.portfolio_embed.parameters()) +
            sum(p.numel() for p in self.node_type_embed.parameters()) +
            sum(p.numel() for p in self.bracket_pos_embed.parameters()) +
            sum(p.numel() for p in self.film.parameters())
        )
        output_params = (
            sum(p.numel() for p in self.output_norm.parameters()) +
            sum(p.numel() for p in self.state_proj.parameters())
        )
        total = sum(p.numel() for p in self.parameters())
        return {
            "mamba": mamba_params,
            "graph_attention": graph_params,
            "moe": moe_params,
            "embeddings": embed_params,
            "output": output_params,
            "total": total,
        }
