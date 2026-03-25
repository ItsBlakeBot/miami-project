"""GraphMamba: Spatial-Temporal Mamba with Graph Attention.

Extends the base Mamba encoder with spatial message passing across
SE Florida weather stations. Each station runs its own temporal Mamba,
then graph attention layers propagate spatial signals between stations.

Key insight: we only TRADE at KMIA, but neighboring stations provide
leading indicators that arrive before KMIA's own observations:
  - KFLL sea breeze → KMIA gets it 20-40 min later
  - KPBI cold front → KMIA gets it 2-3 hours later
  - KHST inland heating → signals urban/coastal lag at KMIA

Architecture:
  Per-station Mamba (shared weights) → Graph Attention → Fused output

The graph can be:
  - Static: geographic distance-based adjacency
  - Dynamic: wind-direction-dependent (upwind stations get higher attention)

For inference, only KMIA's fused hidden state is used for bracket pricing.
Other stations are context providers.
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from engine.ds3m.mamba_encoder import MambaEncoder, MambaConfig

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Station graph definition
# ──────────────────────────────────────────────────────────────────────

# Florida ASOS stations within 350km of KMIA (35 stations)
# Covers: SE FL core, Keys, Gulf coast, Central FL, Tampa/Orlando corridor
STATION_COORDS = {
    # SE Florida core (0-100km)
    "KMIA": (25.80, -80.29),   # Miami Intl — TARGET
    "KOPF": (25.91, -80.28),   # Opa Locka — 13km
    "KTMB": (25.65, -80.43),   # Kendall-Tamiami — 22km
    "KHWO": (26.00, -80.24),   # Hollywood — 23km
    "KFLL": (26.07, -80.15),   # Ft Lauderdale — 33km
    "KHST": (25.49, -80.38),   # Homestead AFB — 35km
    "KFXE": (26.20, -80.17),   # Ft Laud Exec — 46km
    "KPMP": (26.25, -80.11),   # Pompano Beach — 54km
    "KBCT": (26.38, -80.11),   # Boca Raton — 67km
    "KPBI": (26.68, -80.10),   # W Palm Beach — 100km
    # Inland / Everglades (100-200km)
    "KX51": (26.69, -80.86),   # Homestead Gen — 115km
    "KIMM": (26.43, -81.40),   # Immokalee — 132km
    "KPHK": (26.79, -80.69),   # Pahokee — east side of Okeechobee
    "KOBE": (27.26, -80.85),   # Okeechobee — 151km
    "KSUA": (27.18, -80.22),   # Stuart — 154km
    # Keys (140-202km)
    "KMTH": (24.73, -81.05),   # Marathon — 141km
    "KNQX": (24.58, -81.69),   # Key West NAS — 195km
    "KEYW": (24.56, -81.76),   # Key West Intl — 202km (tropical regime)
    # Gulf coast (154-211km)
    "KAPF": (26.15, -81.78),   # Naples — 154km
    "KRSW": (26.54, -81.76),   # SW FL Intl — 169km
    "KFMY": (26.59, -81.86),   # Page Field — 180km
    "KPGD": (26.92, -81.99),   # Punta Gorda — 211km
    # Treasure Coast (190-210km)
    "KFPR": (27.50, -80.37),   # Ft Pierce — 190km
    "KSEF": (27.46, -81.34),   # Sebring — 190km
    "KVRB": (27.66, -80.42),   # Vero Beach — 208km
    # Central FL + Tampa corridor (259-345km) — frontal early warning
    "KMLB": (28.10, -80.64),   # Melbourne — 259km
    "KBOW": (27.94, -81.78),   # Bartow — 281km
    "KSRQ": (27.40, -82.55),   # Sarasota — 287km
    "KLAL": (27.99, -82.02),   # Lakeland — 298km
    "KMCO": (28.43, -81.31),   # Orlando Intl — 310km
    "KSPG": (27.77, -82.63),   # St Petersburg — 320km
    "KORL": (28.55, -81.33),   # Orlando Exec — 323km
    "KTPA": (27.98, -82.53),   # Tampa Intl — 329km (4-6hr frontal lead)
    "KPIE": (27.91, -82.69),   # St Pete-Clearwater — 335km
    "KSFB": (28.78, -81.24),   # Sanford — 345km
}

DEFAULT_STATIONS = list(STATION_COORDS.keys())


def compute_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance in km."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def build_adjacency_matrix(stations: list[str], max_distance_km: float = 120.0) -> Tensor:
    """Build distance-weighted adjacency matrix for station graph.

    Edge weight = exp(-distance / scale), zero if > max_distance_km.
    Returns (N, N) float tensor. Self-loops included with weight 1.0.
    """
    n = len(stations)
    adj = torch.zeros(n, n)
    scale = 30.0  # km — controls how quickly attention falls off with distance

    for i, si in enumerate(stations):
        for j, sj in enumerate(stations):
            if i == j:
                adj[i, j] = 1.0
                continue
            ci, cj = STATION_COORDS.get(si), STATION_COORDS.get(sj)
            if ci is None or cj is None:
                continue
            d = compute_distance_km(ci[0], ci[1], cj[0], cj[1])
            if d <= max_distance_km:
                adj[i, j] = math.exp(-d / scale)

    return adj


def build_bearing_matrix(stations: list[str]) -> Tensor:
    """Build bearing matrix: bearing[i, j] = direction from station i to station j.

    Returns (N, N) tensor of bearings in degrees [0, 360).
    Used for wind-direction-dependent dynamic attention.
    """
    n = len(stations)
    bearings = torch.zeros(n, n)

    for i, si in enumerate(stations):
        for j, sj in enumerate(stations):
            if i == j:
                continue
            ci, cj = STATION_COORDS.get(si), STATION_COORDS.get(sj)
            if ci is None or cj is None:
                continue
            dlat = math.radians(cj[0] - ci[0])
            dlon = math.radians(cj[1] - ci[1])
            x = math.sin(dlon) * math.cos(math.radians(cj[0]))
            y = math.cos(math.radians(ci[0])) * math.sin(math.radians(cj[0])) - \
                math.sin(math.radians(ci[0])) * math.cos(math.radians(cj[0])) * math.cos(dlon)
            bearing = math.degrees(math.atan2(x, y)) % 360
            bearings[i, j] = bearing

    return bearings


# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────

@dataclass
class GraphMambaConfig:
    """Configuration for GraphMamba spatial-temporal model."""
    # Temporal Mamba (shared across stations) — FULL SEND config
    mamba: MambaConfig = field(default_factory=lambda: MambaConfig(
        d_input=52, d_model=384, d_state=48, n_layers=8, dropout=0.15,
    ))

    # Graph attention — 4 layers for deep spatial propagation
    n_graph_layers: int = 4          # number of graph attention layers
    n_heads: int = 6                 # multi-head attention heads (384/6=64 per head)
    graph_dropout: float = 0.1
    use_dynamic_edges: bool = True   # wind-direction-dependent attention
    max_distance_km: float = 350.0   # max station distance for edges (covers Tampa/Orlando)

    # Station config
    stations: list[str] = field(default_factory=lambda: DEFAULT_STATIONS)
    target_station: str = "KMIA"     # primary station for trading


# ──────────────────────────────────────────────────────────────────────
# Graph Attention Layer
# ──────────────────────────────────────────────────────────────────────

class SpatialGraphAttention(nn.Module):
    """Multi-head graph attention over station nodes.

    Each node's representation is the Mamba hidden state h_t for that station.
    Attention is masked by geographic adjacency and optionally modulated by
    wind direction (upwind stations attend more strongly).

    Parameters
    ----------
    d_model : int — hidden dimension per node
    n_heads : int — number of attention heads
    dropout : float
    use_dynamic_edges : bool — if True, modulate attention by wind direction
    """

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1,
                 use_dynamic_edges: bool = True):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        assert d_model % n_heads == 0

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.use_dynamic_edges = use_dynamic_edges
        if use_dynamic_edges:
            # Learn to modulate attention based on wind-direction alignment
            # Input: cos(wind_dir - bearing) feature per edge
            self.wind_proj = nn.Sequential(
                nn.Linear(2, n_heads),  # (cos_alignment, distance_weight) → per-head bias
                nn.Tanh(),
            )

    def forward(
        self,
        node_features: Tensor,       # (B, N_stations, d_model)
        adj: Tensor,                  # (N, N) static adjacency
        wind_dirs: Tensor | None = None,  # (B, N) wind direction per station in degrees
        bearings: Tensor | None = None,    # (N, N) bearing matrix
    ) -> Tensor:
        """
        Returns: (B, N_stations, d_model) updated node features
        """
        B, N, D = node_features.shape
        residual = node_features
        x = self.norm(node_features)

        # Multi-head QKV
        Q = self.W_q(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)  # (B, H, N, d_h)
        K = self.W_k(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)

        # Scaled dot-product attention
        scale = math.sqrt(self.d_head)
        attn = torch.matmul(Q, K.transpose(-2, -1)) / scale  # (B, H, N, N)

        # Apply static adjacency mask (zero out non-connected stations)
        adj_mask = (adj > 0).unsqueeze(0).unsqueeze(0)  # (1, 1, N, N)
        attn = attn.masked_fill(~adj_mask, float("-inf"))

        # Dynamic wind-direction modulation
        if self.use_dynamic_edges and wind_dirs is not None and bearings is not None:
            # For each edge (i→j), compute how aligned the wind at station i
            # is with the bearing from j to i (upwind = wind blowing FROM j TO i)
            # cos(wind_dir_i - bearing_ji) ≈ 1 means j is upwind of i
            reverse_bearings = (bearings.transpose(0, 1) + 180) % 360  # bearing from j to i
            wind_expanded = wind_dirs.unsqueeze(-1).expand(B, N, N)  # (B, N, N)
            bearing_expanded = reverse_bearings.unsqueeze(0).expand(B, N, N)

            # Cosine alignment: 1.0 = perfectly upwind, -1.0 = perfectly downwind
            alignment = torch.cos(torch.deg2rad(wind_expanded - bearing_expanded))  # (B, N, N)

            # Distance feature from adjacency
            dist_feat = adj.unsqueeze(0).expand(B, N, N)  # (B, N, N)

            # Stack features and project to per-head bias
            edge_features = torch.stack([alignment, dist_feat], dim=-1)  # (B, N, N, 2)
            wind_bias = self.wind_proj(edge_features)  # (B, N, N, n_heads)
            wind_bias = wind_bias.permute(0, 3, 1, 2)  # (B, H, N, N)

            attn = attn + wind_bias

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, V)  # (B, H, N, d_h)
        out = out.transpose(1, 2).contiguous().view(B, N, D)  # (B, N, D)
        out = self.W_o(out)

        return residual + self.dropout(out)


# ──────────────────────────────────────────────────────────────────────
# GraphMamba Encoder
# ──────────────────────────────────────────────────────────────────────

class GraphMambaEncoder(nn.Module):
    """Spatial-Temporal Mamba encoder with graph attention.

    Architecture:
      1. Each station's feature sequence → shared Mamba encoder → h_t per station
      2. Graph attention layers propagate spatial context between stations
      3. Target station (KMIA) gets enriched hidden state for downstream tasks

    Training: all stations processed in parallel (batched across stations)
    Inference: process available stations, extract KMIA's fused representation
    """

    def __init__(self, config: GraphMambaConfig | None = None):
        super().__init__()
        if config is None:
            config = GraphMambaConfig()
        self.config = config

        # Shared temporal Mamba encoder (same weights for all stations)
        self.mamba = MambaEncoder(config.mamba)

        # Station embedding (learnable per-station identity)
        self.station_embed = nn.Embedding(len(config.stations), config.mamba.d_model)

        # Graph attention layers
        self.graph_layers = nn.ModuleList([
            SpatialGraphAttention(
                config.mamba.d_model,
                config.n_heads,
                config.graph_dropout,
                config.use_dynamic_edges,
            )
            for _ in range(config.n_graph_layers)
        ])

        # Pre-compute static graph structure
        self.station_to_idx = {s: i for i, s in enumerate(config.stations)}
        self.target_idx = self.station_to_idx[config.target_station]

        # Register adjacency and bearing matrices as buffers (move with model)
        adj = build_adjacency_matrix(config.stations, config.max_distance_km)
        bearings = build_bearing_matrix(config.stations)
        self.register_buffer("adj", adj)
        self.register_buffer("bearings", bearings)

        n_params = sum(p.numel() for p in self.parameters())
        n_mamba = sum(p.numel() for p in self.mamba.parameters())
        n_graph = n_params - n_mamba
        log.info(f"GraphMambaEncoder: {n_params:,} total parameters "
                 f"(Mamba: {n_mamba:,}, Graph: {n_graph:,})")
        log.info(f"  Stations: {len(config.stations)}, target: {config.target_station}")
        log.info(f"  Graph layers: {config.n_graph_layers}, heads: {config.n_heads}")

    def forward(
        self,
        station_features: dict[str, Tensor] | Tensor,
        wind_dirs: Tensor | None = None,
    ) -> Tensor:
        """Forward pass through temporal Mamba + spatial graph attention.

        Parameters
        ----------
        station_features : dict[str, Tensor] mapping station_id → (B, T, 33)
            OR Tensor of shape (B, N_stations, T, 33) if pre-stacked
        wind_dirs : (B, N_stations) wind direction per station (degrees)
            Used for dynamic edge weighting. If None, uses static adjacency only.

        Returns
        -------
        h_target : (B, T, d_model) — target station's graph-enriched hidden state
        h_all : (B, N_stations, T, d_model) — all stations' hidden states (for training)
        """
        N = len(self.config.stations)
        d_model = self.config.mamba.d_model

        if isinstance(station_features, dict):
            # Dict input: stack into (B, N, T, 33)
            # Get batch size and seq len from first available station
            first = next(iter(station_features.values()))
            B, T, D = first.shape
            device = first.device

            stacked = torch.zeros(B, N, T, D, device=device)
            station_mask = torch.zeros(N, device=device, dtype=torch.bool)

            for station_id, feat in station_features.items():
                idx = self.station_to_idx.get(station_id)
                if idx is not None:
                    stacked[:, idx] = feat
                    station_mask[idx] = True
        else:
            # Pre-stacked (B, N, T, 33)
            stacked = station_features
            B, N_in, T, D = stacked.shape
            device = stacked.device
            station_mask = torch.ones(N, device=device, dtype=torch.bool)

        # ── Step 1: Temporal Mamba per station (shared weights) ──
        # Reshape to process all stations as a batch: (B*N, T, 33)
        flat = stacked.view(B * N, T, D)
        h_flat = self.mamba(flat)  # (B*N, T, d_model)
        h_per_station = h_flat.view(B, N, T, d_model)  # (B, N, T, d_model)

        # Add station embeddings
        station_ids = torch.arange(N, device=device)
        s_embed = self.station_embed(station_ids)  # (N, d_model)
        h_per_station = h_per_station + s_embed.unsqueeze(0).unsqueeze(2)

        # ── Step 2: Graph attention at each timestep ──
        # For efficiency, apply graph attention on the last hidden state
        # (or optionally every K timesteps for richer spatial interaction)
        h_last = h_per_station[:, :, -1, :]  # (B, N, d_model)

        for graph_layer in self.graph_layers:
            h_last = graph_layer(
                h_last,
                self.adj,
                wind_dirs=wind_dirs,
                bearings=self.bearings,
            )

        # ── Step 3: Inject graph context back into temporal sequence ──
        # The graph-enriched last-step representation replaces the original
        h_per_station = h_per_station.clone()
        h_per_station[:, :, -1, :] = h_last

        # Extract target station
        h_target = h_per_station[:, self.target_idx]  # (B, T, d_model)

        return h_target, h_per_station

    def forward_target_only(
        self,
        station_features: dict[str, Tensor],
        wind_dirs: Tensor | None = None,
    ) -> Tensor:
        """Convenience: return only the target station's hidden state.

        Returns: (B, T, d_model)
        """
        h_target, _ = self.forward(station_features, wind_dirs)
        return h_target

    def get_target_h_last(
        self,
        station_features: dict[str, Tensor],
        wind_dirs: Tensor | None = None,
    ) -> Tensor:
        """Get target station's last hidden state (for downstream DPF/NSF).

        Returns: (B, d_model)
        """
        h_target, _ = self.forward(station_features, wind_dirs)
        return h_target[:, -1, :]  # (B, d_model)
