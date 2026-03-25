"""DS3M v2 configuration — all hyperparameters in one place."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class DS3MConfig:
    """Unified DS3M v2 hyperparameters."""

    # ── Mamba Encoder ─────────────────────────────────────────────
    mamba_d_input: int = 33
    mamba_d_model: int = 64
    mamba_d_state: int = 16
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    mamba_n_layers: int = 2
    mamba_dropout: float = 0.05

    # ── Differentiable Particle Filter ────────────────────────────
    n_particles: int = 500
    dpf_d_state: int = 2
    dpf_d_latent: int = 16
    dpf_d_regime_embed: int = 8
    resample_temperature: float = 0.5
    ess_threshold_fraction: float = 0.5

    # ── Neural Spline Flow ────────────────────────────────────────
    nsf_n_transforms: int = 4
    nsf_n_bins: int = 16
    nsf_hidden_dim: int = 64
    nsf_tail_bound: float = 15.0

    # ── HDP-Sticky Regime Discovery ───────────────────────────────
    k_regimes_initial: int = 5
    max_regimes: int = 15
    hdp_gamma: float = 2.0
    hdp_alpha0: float = 5.0
    hdp_kappa: float = 50.0
    regime_likelihood_gap: float = -5.0
    regime_gap_cycles: int = 3

    # ── Regime-Conditioned Kalman Filter ──────────────────────────
    kf_continental_K: float = 0.25
    kf_continental_Q: float = 0.4
    kf_continental_R: float = 1.0
    kf_sea_breeze_K: float = 0.40
    kf_sea_breeze_Q: float = 0.2
    kf_sea_breeze_R: float = 0.6
    kf_frontal_K: float = 0.15
    kf_frontal_Q: float = 1.5
    kf_frontal_R: float = 2.0

    # ── Conformal Calibration ─────────────────────────────────────
    conformal_window: int = 30
    conformal_alpha: float = 0.10

    # ── Bracket Pricer ────────────────────────────────────────────
    min_edge_cents: float = 4.0
    min_ev_cents: float = 2.0
    min_price_cents: float = 3.0
    max_price_cents: float = 80.0
    taker_edge_threshold: float = 6.0

    # ── Paper Trader ──────────────────────────────────────────────
    bankroll_cents: int = 100_000
    kelly_fraction: float = 0.15
    max_position_pct: float = 0.05
    max_open_trades: int = 8
    max_per_bracket: int = 2
    max_daily_trades: int = 15
    cooldown_minutes: int = 10

    # ── Training ──────────────────────────────────────────────────
    pretrain_epochs: int = 50
    pretrain_lr: float = 1e-3
    nsf_train_epochs: int = 100
    nsf_train_lr: float = 5e-4
    crps_weight: float = 0.6
    brier_weight: float = 0.4
    training_lookback_days: int = 30
    min_settlements_for_training: int = 5

    # ── Orchestrator ──────────────────────────────────────────────
    cycle_secs: int = 5
    mamba_forward_interval_secs: int = 60
    state_persist_interval: int = 10  # every N cycles

    # ── Dashboard ─────────────────────────────────────────────────
    dashboard_port: int = 8050

    # ── Persistence ───────────────────────────────────────────────
    state_path: str = "analysis_data/ds3m_v2_state.json"
    analysis_dir: str = "analysis_data"

    @classmethod
    def load(cls, path: str | Path = "analysis_data/ds3m_v2_config.json") -> DS3MConfig:
        p = Path(path)
        if not p.exists():
            return cls()
        try:
            data = json.loads(p.read_text())
            return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        except (json.JSONDecodeError, TypeError):
            return cls()

    def save(self, path: str | Path = "analysis_data/ds3m_v2_config.json"):
        from dataclasses import asdict
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), indent=2))
