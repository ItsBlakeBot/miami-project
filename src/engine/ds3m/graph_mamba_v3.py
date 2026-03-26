"""GraphMamba Spatial Encoder v3 for Weather Brain v3.1.

Upgraded from graph_mamba.py with:
  - 12 graph attention layers (up from 4)
  - d_model=896, 12 heads
  - 47 nodes (35 FL + 12 expansion)
  - RoPE relative position encoding
  - Dynamic wind-direction-weighted adjacency
  - 1 SHARED EXPERT (always active, DeepSeek-R1/LLaMA-4 style) as
    global weather prior that stabilizes multi-hop propagation during
    rare events

Architecture:
  Per-node temporal embedding -> [GraphAttention + SharedExpert] x 12
  -> Target station (KMIA) fused representation

The shared expert acts as a global weather prior: a 3-layer MLP that
is added to every node's representation after each graph attention
layer. This prevents catastrophic forgetting during rare weather
events where the graph attention's learned spatial patterns may not
generalize.
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from engine.ds3m.graph_mamba import (
    STATION_COORDS,
    DEFAULT_STATIONS,
    compute_distance_km,
    build_adjacency_matrix,
    build_bearing_matrix,
)
from engine.ds3m.wb3_config import GraphMambaV3Config

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Expansion stations (12 additional for v3)
# ──────────────────────────────────────────────────────────────────────

EXPANSION_STATIONS = {
    # Northern FL / Georgia border — cold air intrusion early warning
    "KJAX": (30.49, -81.69),   # Jacksonville
    "KGNV": (29.69, -82.27),   # Gainesville
    "KDAB": (29.18, -81.06),   # Daytona Beach
    # Gulf buoys / offshore
    "FWYF1": (25.59, -80.10),  # Fowey Rocks (offshore Miami)
    "MLRF1": (25.01, -80.38),  # Molasses Reef
    "SMKF1": (24.63, -81.11),  # Sombrero Key
    # Bahamas / Caribbean — tropical moisture source
    "MYNN": (25.04, -77.47),   # Nassau
    "MYBC": (25.08, -77.88),   # Bimini (closest Bahamas to Miami)
    # Lake Okeechobee region
    "KCLW": (27.98, -82.76),   # Clearwater
    "KVNC": (27.07, -82.44),   # Venice
    # Interior / ridge
    "KAVO": (27.59, -81.53),   # Avon Park
    "KWRB": (28.14, -80.72),   # Patrick SFB
}

# Combined station list: 35 original + 12 expansion = 47
ALL_STATIONS_V3 = DEFAULT_STATIONS + list(EXPANSION_STATIONS.keys())

# Merge coordinates
ALL_STATION_COORDS = {**STATION_COORDS, **EXPANSION_STATIONS}


def build_adjacency_matrix_v3(
    stations: list[str],
    max_distance_km: float = 400.0,
) -> Tensor:
    """Build distance-weighted adjacency for the expanded 47-station graph."""
    n = len(stations)
    adj = torch.zeros(n, n)
    scale = 40.0  # km — slightly wider than v2 for the larger network

    for i, si in enumerate(stations):
        for j, sj in enumerate(stations):
            if i == j:
                adj[i, j] = 1.0
                continue
            ci = ALL_STATION_COORDS.get(si)
            cj = ALL_STATION_COORDS.get(sj)
            if ci is None or cj is None:
                continue
            d = compute_distance_km(ci[0], ci[1], cj[0], cj[1])
            if d <= max_distance_km:
                adj[i, j] = math.exp(-d / scale)

    return adj


def build_bearing_matrix_v3(stations: list[str]) -> Tensor:
    """Build bearing matrix for the expanded 47-station graph."""
    n = len(stations)
    bearings = torch.zeros(n, n)

    for i, si in enumerate(stations):
        for j, sj in enumerate(stations):
            if i == j:
                continue
            ci = ALL_STATION_COORDS.get(si)
            cj = ALL_STATION_COORDS.get(sj)
            if ci is None or cj is None:
                continue
            dlat = math.radians(cj[0] - ci[0])
            dlon = math.radians(cj[1] - ci[1])
            x = math.sin(dlon) * math.cos(math.radians(cj[0]))
            y = (
                math.cos(math.radians(ci[0])) * math.sin(math.radians(cj[0]))
                - math.sin(math.radians(ci[0]))
                * math.cos(math.radians(cj[0]))
                * math.cos(dlon)
            )
            bearing = math.degrees(math.atan2(x, y)) % 360
            bearings[i, j] = bearing

    return bearings


# ──────────────────────────────────────────────────────────────────────
# RoPE Relative Position Encoding
# ──────────────────────────────────────────────────────────────────────

class RotaryPositionEncoding(nn.Module):
    """Rotary Position Encoding (RoPE) for graph attention.

    Encodes relative spatial positions using geographic distance and
    bearing between stations. Applied to Q and K in attention.

    Parameters
    ----------
    d_head : int
        Dimension per attention head.
    base : float
        RoPE frequency base.
    """

    def __init__(self, d_head: int, base: float = 10000.0) -> None:
        super().__init__()
        self.d_head = d_head
        inv_freq = 1.0 / (
            base ** (torch.arange(0, d_head, 2).float() / d_head)
        )
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, positions: Tensor) -> tuple[Tensor, Tensor]:
        """Compute RoPE cos/sin from position indices.

        Parameters
        ----------
        positions : Tensor
            Shape (N,) — position indices or continuous values.

        Returns
        -------
        tuple[Tensor, Tensor]
            (cos, sin) each of shape (N, d_head).
        """
        freqs = torch.einsum("i,j->ij", positions.float(), self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)  # (N, d_head)
        return emb.cos(), emb.sin()


def apply_rotary(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply RoPE rotation to tensor x.

    Parameters
    ----------
    x : Tensor
        Shape (..., d_head).
    cos, sin : Tensor
        Shape (N, d_head) — broadcastable with x.

    Returns
    -------
    Tensor
        Rotated tensor, same shape as x.
    """
    d_half = x.shape[-1] // 2
    x1 = x[..., :d_half]
    x2 = x[..., d_half:]
    cos_half = cos[..., :d_half]
    sin_half = sin[..., :d_half]
    return torch.cat([
        x1 * cos_half - x2 * sin_half,
        x2 * cos_half + x1 * sin_half,
    ], dim=-1)


# ──────────────────────────────────────────────────────────────────────
# Shared Expert (DeepSeek-R1 / LLaMA-4 Style)
# ──────────────────────────────────────────────────────────────────────

class SharedExpert(nn.Module):
    """Always-active global weather prior (DeepSeek-R1 style).

    A 3-layer MLP that is added to every node's representation after
    graph attention. Acts as a stabilizing force:

      - Provides a shared inductive bias across all stations
      - Prevents catastrophic forgetting during rare events
      - Stabilizes multi-hop propagation (12 layers deep)
      - Encodes global weather priors (e.g., temperature conservation)

    Parameters
    ----------
    d_model : int
        Input/output dimension.
    d_hidden : int
        Hidden layer dimension.
    """

    def __init__(self, d_model: int, d_hidden: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, d_model),
        )
        # Initialize last layer near zero for residual-friendly start
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

        n_params = sum(p.numel() for p in self.parameters())
        log.info(f"SharedExpert: {n_params:,} params, d_hidden={d_hidden}")

    def forward(self, x: Tensor) -> Tensor:
        """Apply shared expert with residual connection.

        Parameters
        ----------
        x : Tensor
            Shape (..., d_model).

        Returns
        -------
        Tensor
            Shape (..., d_model).
        """
        return x + self.mlp(x)


# ──────────────────────────────────────────────────────────────────────
# Graph Attention Layer v3 (with RoPE)
# ──────────────────────────────────────────────────────────────────────

class SpatialGraphAttentionV3(nn.Module):
    """Multi-head graph attention with RoPE relative position encoding.

    Upgrades from v2:
      - RoPE for relative spatial encoding
      - Larger d_model (896) and more heads (12)
      - Wind-direction-weighted dynamic adjacency

    Parameters
    ----------
    d_model : int
        Hidden dimension per node.
    n_heads : int
        Number of attention heads.
    dropout : float
        Dropout rate.
    use_dynamic_edges : bool
        If True, modulate attention by wind direction.
    rope_base : float
        RoPE frequency base.
    """

    def __init__(
        self,
        d_model: int = 896,
        n_heads: int = 12,
        dropout: float = 0.1,
        use_dynamic_edges: bool = True,
        rope_base: float = 10000.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        assert d_model % n_heads == 0, (
            f"d_model={d_model} not divisible by n_heads={n_heads}"
        )

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # RoPE for relative position encoding
        self.rope = RotaryPositionEncoding(self.d_head, rope_base)

        # Dynamic wind-direction edges
        self.use_dynamic_edges = use_dynamic_edges
        if use_dynamic_edges:
            self.wind_proj = nn.Sequential(
                nn.Linear(2, n_heads),
                nn.Tanh(),
            )

    def forward(
        self,
        node_features: Tensor,
        adj: Tensor,
        wind_dirs: Tensor | None = None,
        bearings: Tensor | None = None,
    ) -> Tensor:
        """Graph attention with RoPE.

        Parameters
        ----------
        node_features : Tensor
            Shape (B, N, d_model).
        adj : Tensor
            Shape (N, N) — static adjacency.
        wind_dirs : Tensor or None
            Shape (B, N) — wind direction per station in degrees.
        bearings : Tensor or None
            Shape (N, N) — bearing matrix.

        Returns
        -------
        Tensor
            Shape (B, N, d_model) — updated node features.
        """
        B, N, D = node_features.shape
        residual = node_features
        x = self.norm(node_features)

        # Multi-head QKV
        Q = self.W_q(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_k(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)

        # Apply RoPE to Q and K
        positions = torch.arange(N, device=x.device)
        cos, sin = self.rope(positions)  # (N, d_head)
        # Broadcast across batch and heads
        cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, N, d_head)
        sin = sin.unsqueeze(0).unsqueeze(0)
        Q = apply_rotary(Q, cos, sin)
        K = apply_rotary(K, cos, sin)

        # Scaled dot-product attention
        scale = math.sqrt(self.d_head)
        attn = torch.matmul(Q, K.transpose(-2, -1)) / scale

        # Apply static adjacency mask
        adj_mask = (adj > 0).unsqueeze(0).unsqueeze(0)
        attn = attn.masked_fill(~adj_mask, float("-inf"))

        # Dynamic wind-direction modulation
        if self.use_dynamic_edges and wind_dirs is not None and bearings is not None:
            reverse_bearings = (bearings.transpose(0, 1) + 180) % 360
            wind_expanded = wind_dirs.unsqueeze(-1).expand(B, N, N)
            bearing_expanded = reverse_bearings.unsqueeze(0).expand(B, N, N)
            alignment = torch.cos(
                torch.deg2rad(wind_expanded - bearing_expanded)
            )
            dist_feat = adj.unsqueeze(0).expand(B, N, N)
            edge_features = torch.stack([alignment, dist_feat], dim=-1)
            wind_bias = self.wind_proj(edge_features).permute(0, 3, 1, 2)
            attn = attn + wind_bias

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        out = self.W_o(out)

        return residual + self.dropout(out)


# ──────────────────────────────────────────────────────────────────────
# GraphMamba Spatial Encoder v3 (full assembly)
# ──────────────────────────────────────────────────────────────────────

class GraphMambaSpatialV3(nn.Module):
    """GraphMamba Spatial Encoder v3 for Weather Brain v3.1.

    47-node graph with 12 attention layers, RoPE encoding, and a
    shared expert that stabilizes multi-hop propagation.

    Architecture:
      Input projection -> Station embedding
      -> [GraphAttention + SharedExpert] x 12
      -> Extract target station (KMIA)

    Parameters
    ----------
    config : GraphMambaV3Config
        Configuration for the spatial encoder.
    stations : list[str] or None
        Station list. Defaults to ALL_STATIONS_V3 (47 stations).
    target_station : str
        Primary station for trading (default: KMIA).
    """

    def __init__(
        self,
        config: GraphMambaV3Config | None = None,
        stations: list[str] | None = None,
        target_station: str = "KMIA",
    ) -> None:
        super().__init__()

        if config is None:
            config = GraphMambaV3Config()
        self.config = config

        stations = stations or ALL_STATIONS_V3
        self.stations = stations
        self.station_to_idx = {s: i for i, s in enumerate(stations)}
        self.target_station = target_station
        self.target_idx = self.station_to_idx[target_station]
        n_nodes = len(stations)

        # ── Input projection ──────────────────────────────────────
        # Temporal Mamba output -> graph node dimension
        # (d_fusion from MultiResolutionMamba -> d_model for graph)
        self.input_proj = nn.Linear(config.d_model, config.d_model)

        # ── Station embeddings ────────────────────────────────────
        self.station_embed = nn.Embedding(n_nodes, config.d_model)

        # ── Graph attention layers + shared expert ────────────────
        self.graph_layers = nn.ModuleList([
            SpatialGraphAttentionV3(
                d_model=config.d_model,
                n_heads=config.n_heads,
                dropout=config.graph_dropout,
                use_dynamic_edges=config.use_dynamic_edges,
                rope_base=config.rope_base,
            )
            for _ in range(config.n_graph_layers)
        ])

        # Shared expert (1 instance, applied after every graph layer)
        self.shared_expert = SharedExpert(
            config.d_model, config.shared_expert_d_hidden
        )

        # ── Pre-compute static graph structure ────────────────────
        adj = build_adjacency_matrix_v3(stations, config.max_distance_km)
        bearings = build_bearing_matrix_v3(stations)
        self.register_buffer("adj", adj)
        self.register_buffer("bearings", bearings)

        # ── Parameter counting ────────────────────────────────────
        n_graph = sum(
            sum(p.numel() for p in layer.parameters())
            for layer in self.graph_layers
        )
        n_expert = sum(p.numel() for p in self.shared_expert.parameters())
        n_total = sum(p.numel() for p in self.parameters())

        log.info(
            f"GraphMambaSpatialV3: {n_total:,} total parameters\n"
            f"  Graph layers: {config.n_graph_layers} x {n_graph // config.n_graph_layers:,} "
            f"= {n_graph:,} params\n"
            f"  Shared expert: {n_expert:,} params\n"
            f"  Stations: {n_nodes}, target: {target_station}"
        )

    def forward(
        self,
        temporal_state: Tensor,
        station_features: Tensor | None = None,
        wind_dirs: Tensor | None = None,
    ) -> Tensor:
        """Forward pass through spatial graph encoder.

        Parameters
        ----------
        temporal_state : Tensor
            Shape (B, d_model) — fused temporal state from MultiResolutionMamba.
            Broadcast to all nodes, then refined by station embeddings.
        station_features : Tensor or None
            Shape (B, N_nodes, d_model) — per-station features if available.
            If provided, used directly instead of broadcasting temporal_state.
        wind_dirs : Tensor or None
            Shape (B, N_nodes) — wind direction per station in degrees.

        Returns
        -------
        Tensor
            Shape (B, d_model) — target station's graph-enriched representation.
        """
        B = temporal_state.shape[0]
        N = len(self.stations)
        device = temporal_state.device

        if station_features is not None:
            # Use provided per-station features
            h = self.input_proj(station_features)  # (B, N, d_model)
        else:
            # Broadcast temporal state to all nodes
            h = temporal_state.unsqueeze(1).expand(B, N, -1)  # (B, N, d_model)
            h = self.input_proj(h)

        # Add station embeddings
        station_ids = torch.arange(N, device=device)
        s_embed = self.station_embed(station_ids)
        h = h + s_embed.unsqueeze(0)

        # ── Graph attention + shared expert stack ─────────────────
        for graph_layer in self.graph_layers:
            h = graph_layer(
                h, self.adj, wind_dirs=wind_dirs, bearings=self.bearings
            )
            h = self.shared_expert(h)

        # ── Extract target station ────────────────────────────────
        h_target = h[:, self.target_idx, :]  # (B, d_model)

        return h_target

    def forward_all_stations(
        self,
        temporal_state: Tensor,
        station_features: Tensor | None = None,
        wind_dirs: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Forward pass returning both target and all station representations.

        Returns
        -------
        tuple[Tensor, Tensor]
            (h_target, h_all) — target station (B, d_model) and
            all stations (B, N, d_model).
        """
        B = temporal_state.shape[0]
        N = len(self.stations)
        device = temporal_state.device

        if station_features is not None:
            h = self.input_proj(station_features)
        else:
            h = temporal_state.unsqueeze(1).expand(B, N, -1)
            h = self.input_proj(h)

        station_ids = torch.arange(N, device=device)
        s_embed = self.station_embed(station_ids)
        h = h + s_embed.unsqueeze(0)

        for graph_layer in self.graph_layers:
            h = graph_layer(
                h, self.adj, wind_dirs=wind_dirs, bearings=self.bearings
            )
            h = self.shared_expert(h)

        h_target = h[:, self.target_idx, :]
        return h_target, h
