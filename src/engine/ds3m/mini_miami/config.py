"""Mini Miami Configuration — 25M params, fits on RTX 5070 (12GB).

Design philosophy (backed by research):
- Keep WIDTH (d_model) — width helps generalization, not just memorization
- Keep DEPTH — deeper networks promote low-rank solutions that regularize
- Cut PARTICLES — DPF from 8000→1000, d_latent 192→64 (Miami has ~4 regimes, not 8)
- Cut STATIONS — 47→15 (South FL stations only, not all of Florida)
- Increase REGULARIZATION — dropout 0.10→0.20, weight decay 0.01→0.05
- Same data pipeline — reuses multi_res_dataset.py identically

Architecture: Multi-Scale Mamba (inspired by ms-Mamba, ICLR 2025)
- Fine branch: d=256, 4 layers (32 timesteps × 18 features)
- Medium branch: d=384, 6 layers, interleaved Graph-SSM (96 × 36)
- Coarse branch: d=256, 4 layers (112 × 33)
- DPF: 1000 particles, d_latent=64, k_regimes=4
- Flow matching: d_condition=456
- 10 multi-task heads (identical to big model)
- 5-member ensemble

Estimated: ~25M params/member, ~125M total ensemble
VRAM: ~4GB per member at batch=16 (fits in 12GB easily)
Training: ~45 min on 5070, ~8 min on H200
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class MiniMambaBranchConfig:
    d_input: int = 18
    d_model: int = 256
    d_state: int = 64
    d_conv: int = 4
    expand: int = 2
    n_layers: int = 4
    dropout: float = 0.20
    seq_len: int = 32


@dataclass
class MiniFusionConfig:
    d_model: int = 384
    n_heads: int = 8
    dropout: float = 0.20


@dataclass
class MiniGraphConfig:
    d_model: int = 384
    n_graph_layers: int = 6
    n_stations: int = 15      # South FL only
    target_station: str = "KMIA"
    dropout: float = 0.20
    d_expert_hidden: int = 384


@dataclass
class MiniDPFConfig:
    n_particles: int = 1000   # Down from 8000
    d_latent: int = 64        # Down from 192
    d_observation: int = 384
    k_regimes: int = 4        # Miami has ~4 distinct weather regimes
    resample_top_k: int = 32  # Down from 64


@dataclass
class MiniFlowConfig:
    d_condition: int = 456    # 384 + 64 + 8 (spatial + particle + regime)
    d_hidden: int = 512
    n_layers: int = 4         # Down from 6
    n_steps_train: int = 4
    n_steps_infer: int = 8    # Down from 10
    sigma_min: float = 0.001


@dataclass
class MiniMultiTaskConfig:
    d_input: int = 384
    n_brackets: int = 6
    n_regimes: int = 4        # Down from 8
    gradnorm_alpha: float = 1.5
    gradnorm_delay_epochs: int = 10


@dataclass
class MiniMiamiConfig:
    """Full config for Mini Miami Weather Brain."""

    fine_branch: MiniMambaBranchConfig = field(default_factory=lambda: MiniMambaBranchConfig(
        d_input=18, d_model=256, d_state=64, d_conv=4, expand=2,
        n_layers=4, dropout=0.20, seq_len=32,
    ))
    medium_branch: MiniMambaBranchConfig = field(default_factory=lambda: MiniMambaBranchConfig(
        d_input=36, d_model=384, d_state=64, d_conv=4, expand=2,
        n_layers=6, dropout=0.20, seq_len=96,
    ))
    coarse_branch: MiniMambaBranchConfig = field(default_factory=lambda: MiniMambaBranchConfig(
        d_input=33, d_model=256, d_state=64, d_conv=4, expand=2,
        n_layers=4, dropout=0.20, seq_len=112,
    ))

    fusion: MiniFusionConfig = field(default_factory=MiniFusionConfig)
    graph: MiniGraphConfig = field(default_factory=MiniGraphConfig)
    dpf: MiniDPFConfig = field(default_factory=MiniDPFConfig)
    flow: MiniFlowConfig = field(default_factory=MiniFlowConfig)
    tasks: MiniMultiTaskConfig = field(default_factory=MiniMultiTaskConfig)

    # Feature masking
    n_features_fine: int = 18
    n_features_medium: int = 36
    n_features_coarse: int = 33

    # Training
    learning_rate: float = 3e-4
    weight_decay: float = 0.05     # Up from 0.01 — stronger regularization
    warmup_epochs: int = 5
    supervised_epochs: int = 50
    byol_epochs: int = 5
    flow_epochs: int = 15
    swa_epochs: int = 5

    # Ensemble
    n_ensemble: int = 5
    seeds: list[int] = field(default_factory=lambda: [42, 137, 256, 512, 1024])

    # South FL stations (15 nearest to Miami)
    stations: list[str] = field(default_factory=lambda: [
        "KMIA", "KFLL", "KPBI", "KHWO", "KOPF",  # Miami metro
        "KFXE", "KBCT", "KTMB", "KHST", "KEYW",  # SE FL
        "KAPF", "KRSW", "KFMY", "KOBE", "KSPG",  # SW FL coast
    ])
