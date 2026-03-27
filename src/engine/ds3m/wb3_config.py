"""Weather Brain v3.1 'Thermonuclear' configuration.

All hyperparameters for the 112M-parameter per-model architecture.
5x ensemble = 560M total parameters.

This config is separate from the legacy DS3MConfig to avoid breaking
existing v2 code. Import as:
    from engine.ds3m.wb3_config import WB3Config
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Sub-configs for each component
# ──────────────────────────────────────────────────────────────────────

@dataclass
class MambaBranchConfig:
    """Configuration for a single Mamba temporal branch."""
    d_input: int = 64
    d_model: int = 640
    d_state: int = 64
    d_conv: int = 4
    expand: int = 2
    n_layers: int = 8
    dropout: float = 0.10
    seq_len: int = 32


@dataclass
class FusionConfig:
    """Cross-resolution Mamba-Transformer fusion."""
    d_model: int = 896
    n_heads: int = 14  # 896 / 14 = 64
    n_layers: int = 4          # 2 Mamba + 2 Transformer, alternating
    d_state: int = 64
    d_conv: int = 4
    expand: int = 2
    dropout: float = 0.10
    dim_feedforward: int = 2048
    regime_dim: int = 8        # dimension of regime posterior for SST router


@dataclass
class GraphMambaV3Config:
    """GraphMamba Spatial Encoder v3."""
    d_model: int = 896
    n_heads: int = 14  # 896 / 14 = 64
    n_graph_layers: int = 12
    n_nodes: int = 47            # 35 FL + 12 expansion
    graph_dropout: float = 0.10
    use_dynamic_edges: bool = True
    max_distance_km: float = 400.0
    rope_base: float = 10000.0
    shared_expert_d_hidden: int = 896


@dataclass
class DPFv3Config:
    """Differentiable Particle Filter v3."""
    n_particles: int = 8000
    d_latent: int = 192
    k_regimes: int = 8
    d_regime_embed: int = 32
    d_mamba: int = 896           # conditioning dim from Mamba
    ess_threshold_frac: float = 0.5
    resample_temperature: float = 0.5
    proposal_hidden: int = 256


@dataclass
class FlowMatchingV2Config:
    """Rectified Flow + Consistency Distillation."""
    d_condition: int = 1096      # 896 (mamba) + 8 (regime) + 192 (DPF latent)
    d_hidden: int = 512
    n_layers: int = 3
    d_data: int = 1
    n_ode_steps_train: int = 30  # full steps during training
    n_ode_steps_infer: int = 10  # consistency-distilled inference
    dropout: float = 0.10
    sigma_min: float = 1e-4
    d_time_embed: int = 64


@dataclass
class MultiTaskConfig:
    """Multi-task prediction heads."""
    d_input: int = 896
    n_brackets: int = 6
    n_regimes: int = 8
    n_tasks: int = 10           # 8 original + bracket_probs_low + regime (10 total)
    gradnorm_alpha: float = 1.5  # GradNorm restoring force


# ──────────────────────────────────────────────────────────────────────
# Master config
# ──────────────────────────────────────────────────────────────────────

@dataclass
class WB3Config:
    """Weather Brain v3.1 'Thermonuclear' master configuration.

    Target: ~112M parameters per model, x5 ensemble = 560M total.

    Architecture:
      Multi-Resolution Mamba (fine/medium/coarse) → ~84M
      Cross-Resolution Fusion                     → ~9M
      GraphMamba Spatial v3                        → ~10M
      DPF v3                                      → ~4M
      Flow Matching v2                             → ~2M
      Multi-Task Heads                             → ~1M
      Feature Masking                              → ~0.01M
      ─────────────────────────────────────────
      Total                                        → ~112M
    """

    # ── Branch configs ────────────────────────────────────────────
    fine_branch: MambaBranchConfig = field(default_factory=lambda: MambaBranchConfig(
        d_input=18, d_model=640, d_state=64, d_conv=4, expand=2,
        n_layers=8, dropout=0.10, seq_len=32,
    ))
    medium_branch: MambaBranchConfig = field(default_factory=lambda: MambaBranchConfig(
        d_input=36, d_model=896, d_state=64, d_conv=4, expand=2,
        n_layers=12, dropout=0.10, seq_len=96,
    ))
    coarse_branch: MambaBranchConfig = field(default_factory=lambda: MambaBranchConfig(
        d_input=24, d_model=512, d_state=64, d_conv=4, expand=2,
        n_layers=6, dropout=0.10, seq_len=112,
    ))

    # ── Component configs ─────────────────────────────────────────
    fusion: FusionConfig = field(default_factory=FusionConfig)
    graph: GraphMambaV3Config = field(default_factory=GraphMambaV3Config)
    dpf: DPFv3Config = field(default_factory=DPFv3Config)
    flow: FlowMatchingV2Config = field(default_factory=FlowMatchingV2Config)
    tasks: MultiTaskConfig = field(default_factory=MultiTaskConfig)

    # ── Feature masking ───────────────────────────────────────────
    n_features_fine: int = 18
    n_features_medium: int = 36
    n_features_coarse: int = 24

    # ── Training ──────────────────────────────────────────────────
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 2000
    max_grad_norm: float = 1.0
    bf16: bool = True

    # ── Ensemble ──────────────────────────────────────────────────
    n_ensemble: int = 5
    ensemble_seed_base: int = 42

    # ── Persistence ───────────────────────────────────────────────
    state_path: str = "analysis_data/wb3_state.pt"

    @classmethod
    def load(cls, path: str | Path) -> WB3Config:
        """Load config from JSON."""
        p = Path(path)
        if not p.exists():
            return cls()
        try:
            data = json.loads(p.read_text())
            return cls(**{k: v for k, v in data.items()
                         if k in cls.__dataclass_fields__})
        except (json.JSONDecodeError, TypeError):
            return cls()

    def save(self, path: str | Path) -> None:
        """Save config to JSON."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), indent=2))
