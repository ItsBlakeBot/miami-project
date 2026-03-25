"""Differentiable Particle Filter for DS3M.

End-to-end gradient-based training through the full inference pipeline:
  Mamba encoder → regime dynamics → observation model → density estimation

Uses the "optimal transport resampling" trick (Corenflos et al. 2021) to
make resampling differentiable, enabling backprop through the entire PF.

Key differences from the old DIMMPF:
  1. All operations are torch Tensors (not numpy) — gradients flow everywhere
  2. Resampling uses soft/differentiable variant (not hard systematic)
  3. Mamba hidden state conditions regime transitions and emission
  4. Skew-normal emission per particle (not Gaussian)
  5. Trained end-to-end with CRPS loss via the NSF density head

Reference: Corenflos et al. 2021 — "Differentiable Particle Filtering"
           Chen et al. 2023 — "PyDPF: Differentiable Particle Filters"
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from engine.ds3m.skew_normal import SkewNormal, SkewNormalMixture

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────

@dataclass
class DPFConfig:
    n_particles: int = 500
    d_state: int = 2          # (remaining_high, remaining_low)
    d_latent: int = 32        # continuous latent z_t per particle
    k_regimes: int = 5        # initial (grows via HDP)
    d_regime_embed: int = 16  # regime embedding dimension
    ess_threshold_frac: float = 0.5
    resample_temperature: float = 0.5  # soft resampling temperature (lower = harder)
    gradient_clip: float = 5.0


# ──────────────────────────────────────────────────────────────────────
# Particle State (differentiable)
# ──────────────────────────────────────────────────────────────────────

class ParticleCloud:
    """Differentiable particle cloud state.

    All tensors retain gradients for end-to-end training.
    """

    def __init__(
        self,
        remaining_high: Tensor,   # (N,) — °F above current running high
        remaining_low: Tensor,    # (N,) — °F below current running low
        z: Tensor,                # (N, d_latent) — continuous latent
        regime_logits: Tensor,    # (N, K) — soft regime assignment (logits)
        log_weights: Tensor,      # (N,) — unnormalized log weights
    ):
        self.remaining_high = remaining_high
        self.remaining_low = remaining_low
        self.z = z
        self.regime_logits = regime_logits
        self.log_weights = log_weights
        self.N = remaining_high.shape[0]

    @property
    def weights(self) -> Tensor:
        """Normalized weights via log-sum-exp."""
        return F.softmax(self.log_weights, dim=0)

    @property
    def ess(self) -> Tensor:
        """Effective sample size."""
        w = self.weights
        return 1.0 / (w ** 2).sum()

    @property
    def regime_probs(self) -> Tensor:
        """Per-particle regime probabilities (N, K)."""
        return F.softmax(self.regime_logits, dim=-1)

    @property
    def regime_posterior(self) -> Tensor:
        """Weight-averaged regime posterior (K,)."""
        return (self.weights.unsqueeze(-1) * self.regime_probs).sum(dim=0)

    @property
    def weighted_mean_high(self) -> Tensor:
        return (self.weights * self.remaining_high).sum()

    @property
    def weighted_std_high(self) -> Tensor:
        mu = self.weighted_mean_high
        return ((self.weights * (self.remaining_high - mu) ** 2).sum()).sqrt().clamp(min=0.01)

    @property
    def weighted_mean_low(self) -> Tensor:
        return (self.weights * self.remaining_low).sum()

    @property
    def weighted_std_low(self) -> Tensor:
        mu = self.weighted_mean_low
        return ((self.weights * (self.remaining_low - mu) ** 2).sum()).sqrt().clamp(min=0.01)


# ──────────────────────────────────────────────────────────────────────
# Differentiable Transition Model
# ──────────────────────────────────────────────────────────────────────

class DifferentiableTransition(nn.Module):
    """Learnable state transition conditioned on Mamba hidden state + regime.

    For each particle:
      1. Regime transition: logits updated from Mamba h_t
      2. State dynamics: drift + noise conditioned on regime + h_t
      3. Latent z update: GRU-like update from h_t + regime
    """

    def __init__(self, config: DPFConfig, d_mamba: int = 64):
        super().__init__()
        self.config = config

        # Regime embedding
        self.regime_embed = nn.Embedding(config.k_regimes + 10, config.d_regime_embed)  # +10 for growth

        # Regime transition network: h_t → regime logit update
        self.regime_transition_net = nn.Sequential(
            nn.Linear(d_mamba + config.d_regime_embed, 32),
            nn.SiLU(),
            nn.Linear(32, config.k_regimes + 10),  # over-allocate for regime growth
        )

        # State dynamics network: (h_t, regime_embed, z) → (drift_high, drift_low, log_sigma_high, log_sigma_low)
        d_dyn_input = d_mamba + config.d_regime_embed + config.d_latent
        self.dynamics_net = nn.Sequential(
            nn.Linear(d_dyn_input, 64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 4),  # drift_h, drift_l, log_sigma_h, log_sigma_l
        )

        # Latent z update (GRU-like)
        self.z_gate = nn.Linear(d_mamba + config.d_regime_embed + config.d_latent, config.d_latent)
        self.z_candidate = nn.Linear(d_mamba + config.d_regime_embed + config.d_latent, config.d_latent)

    def forward(
        self, cloud: ParticleCloud, h_t: Tensor, k_active: int,
    ) -> ParticleCloud:
        """Predict step: advance all particles one timestep.

        cloud: current particle cloud
        h_t: (d_mamba,) Mamba hidden state (shared across particles)
        k_active: number of currently active regimes
        """
        N = cloud.N
        h_expanded = h_t.unsqueeze(0).expand(N, -1)  # (N, d_mamba)

        # 1. Regime transition
        # Get current regime embedding (soft: weighted sum)
        regime_probs = cloud.regime_probs[:, :k_active]  # (N, K)
        regime_ids = torch.arange(k_active, device=h_t.device)
        regime_embeds_all = self.regime_embed(regime_ids)  # (K, d_regime_embed)
        # Soft regime embedding per particle
        regime_embed = torch.matmul(regime_probs, regime_embeds_all)  # (N, d_regime_embed)

        # Update regime logits
        regime_input = torch.cat([h_expanded, regime_embed], dim=-1)
        regime_logit_update = self.regime_transition_net(regime_input)[:, :k_active]
        # Soft update: blend old logits with new (don't completely overwrite)
        new_regime_logits = cloud.regime_logits.clone()
        new_regime_logits[:, :k_active] = (
            0.7 * cloud.regime_logits[:, :k_active] + 0.3 * regime_logit_update
        )

        # 2. State dynamics (conditioned on regime + Mamba)
        dyn_input = torch.cat([h_expanded, regime_embed, cloud.z], dim=-1)
        dyn_params = self.dynamics_net(dyn_input)  # (N, 4)
        drift_high = dyn_params[:, 0]
        drift_low = dyn_params[:, 1]
        log_sigma_high = dyn_params[:, 2]
        log_sigma_low = dyn_params[:, 3]

        sigma_high = F.softplus(log_sigma_high) + 0.01  # floor
        sigma_low = F.softplus(log_sigma_low) + 0.01

        # Sample innovations (reparameterization trick for differentiability)
        eps_high = torch.randn(N, device=h_t.device)
        eps_low = torch.randn(N, device=h_t.device)

        new_remaining_high = (
            cloud.remaining_high + drift_high + sigma_high * eps_high
        ).clamp(min=0.0)
        new_remaining_low = (
            cloud.remaining_low + drift_low + sigma_low * eps_low
        ).clamp(min=0.0)

        # 3. Latent z update (GRU-style)
        z_input = torch.cat([h_expanded, regime_embed, cloud.z], dim=-1)
        gate = torch.sigmoid(self.z_gate(z_input))
        candidate = torch.tanh(self.z_candidate(z_input))
        new_z = gate * cloud.z + (1 - gate) * candidate

        return ParticleCloud(
            remaining_high=new_remaining_high,
            remaining_low=new_remaining_low,
            z=new_z,
            regime_logits=new_regime_logits,
            log_weights=cloud.log_weights,  # weights unchanged in predict step
        )


# ──────────────────────────────────────────────────────────────────────
# Differentiable Observation Model
# ──────────────────────────────────────────────────────────────────────

class DifferentiableObservation(nn.Module):
    """Learnable observation model with skew-normal likelihood.

    Per-source observation noise R learned as parameters.
    Skewness alpha learned from Mamba context.
    """

    def __init__(self, config: DPFConfig, d_mamba: int = 64, n_sources: int = 3):
        super().__init__()
        self.config = config

        # Per-source log-R (learnable observation noise)
        # Sources: 0=wethr, 1=nws, 2=iem
        self.log_R = nn.Parameter(
            torch.tensor([math.log(0.5), math.log(0.4), math.log(0.6)])
        )

        # ── Per-regime observation noise scale ───────────────────
        # Each regime gets its own learned obs noise scale so that e.g.
        # "frontal" can have wide emission noise while "continental"
        # has tight noise.  Stored as log-scale; actual scale =
        # softplus(log_obs_scale[k]).  Over-allocated (+10) like
        # regime_logits for HDP growth.
        k_total = config.k_regimes + 10
        self.log_obs_scale = nn.Parameter(torch.zeros(k_total))
        # Initialise physical regimes with rough priors (log-space vals
        # chosen so softplus ≈ desired multiplier):
        #   continental → 1.0, sea_breeze → 0.85, frontal → 1.4,
        #   tropical → 1.2, nocturnal → 1.3
        _init_scales = [1.0, 0.85, 1.4, 1.2, 1.3]
        for i, s in enumerate(_init_scales):
            if i < k_total:
                # softplus(x) ≈ s  →  x = log(exp(s) - 1)
                self.log_obs_scale.data[i] = math.log(math.exp(s) - 1.0)

        # Context-dependent alpha (skewness of observation noise)
        self.alpha_net = nn.Sequential(
            nn.Linear(d_mamba, 16),
            nn.SiLU(),
            nn.Linear(16, 1),
        )

    def log_likelihood(
        self,
        cloud: ParticleCloud,
        obs_remaining: Tensor,  # scalar — observed remaining move
        source_idx: int,
        h_t: Tensor,            # (d_mamba,) Mamba hidden state
        market_type: str = "high",
        k_active: int = 5,
    ) -> Tensor:
        """Compute per-particle log-likelihood of observation.

        Observation noise is regime-conditioned: each particle's noise
        scale is the soft-weighted combination of per-regime scales.

        Returns: (N,) log p(obs | particle_i)
        """
        R_source = torch.exp(self.log_R[source_idx]).clamp(min=0.15, max=2.5)
        alpha = self.alpha_net(h_t.unsqueeze(0)).squeeze()  # scalar

        if market_type == "high":
            predicted = cloud.remaining_high
        else:
            predicted = cloud.remaining_low

        # ── Per-regime observation noise ─────────────────────────
        # regime_scales: (K,) positive via softplus
        regime_scales = F.softplus(self.log_obs_scale[:k_active])  # (K,)
        regime_probs = cloud.regime_probs[:, :k_active]             # (N, K)
        # Per-particle scale = soft weighted combination of regime scales
        per_particle_scale = torch.matmul(regime_probs, regime_scales)  # (N,)
        # Multiply source-level R by regime-specific multiplier
        R = (R_source * per_particle_scale).clamp(min=0.15, max=4.0)

        # Skew-normal log-likelihood
        dist = SkewNormal(predicted, R, alpha.expand_as(predicted))
        return dist.log_prob(obs_remaining.expand_as(predicted))


# ──────────────────────────────────────────────────────────────────────
# Differentiable Soft Resampling
# ──────────────────────────────────────────────────────────────────────

def soft_resample(cloud: ParticleCloud, temperature: float = 0.5) -> ParticleCloud:
    """Differentiable soft resampling via Gumbel-Softmax relaxation.

    Instead of hard systematic resampling (which breaks gradients),
    we use a soft weighted combination:
      new_particle_i = sum_j(softmax(log_w_j / tau) * particle_j)

    At low temperature (tau → 0), this approaches hard resampling.
    At high temperature, it's a uniform mixture (minimal information loss).

    After resampling, weights are reset to uniform.
    """
    N = cloud.N
    w = cloud.weights  # (N,)

    # Soft assignment matrix via Gumbel-Softmax
    # Each new particle is a soft mixture of old particles
    log_w = cloud.log_weights.unsqueeze(0).expand(N, -1)  # (N, N)
    # Add Gumbel noise for stochasticity
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(log_w) + 1e-10) + 1e-10)
    soft_assignment = F.softmax((log_w + gumbel_noise) / temperature, dim=-1)  # (N, N)

    # Soft resample each state dimension
    new_remaining_high = torch.matmul(soft_assignment, cloud.remaining_high)
    new_remaining_low = torch.matmul(soft_assignment, cloud.remaining_low)
    new_z = torch.matmul(soft_assignment, cloud.z)  # (N, d_latent)
    new_regime_logits = torch.matmul(soft_assignment, cloud.regime_logits)  # (N, K)

    return ParticleCloud(
        remaining_high=new_remaining_high,
        remaining_low=new_remaining_low,
        z=new_z,
        regime_logits=new_regime_logits,
        log_weights=torch.zeros(N, device=cloud.log_weights.device),  # uniform after resample
    )


# ──────────────────────────────────────────────────────────────────────
# Full Differentiable Particle Filter
# ──────────────────────────────────────────────────────────────────────

class DifferentiableParticleFilter(nn.Module):
    """End-to-end differentiable particle filter for DS3M.

    Full pipeline: predict → update → soft_resample
    All operations are differentiable — CRPS loss backprops through
    the entire inference chain.

    Components:
      - DifferentiableTransition: Mamba-conditioned state dynamics
      - DifferentiableObservation: Skew-normal likelihood (per-regime noise)
      - Soft resampling: Gumbel-Softmax relaxation
      - Learnable HDP concentration α (softplus-transformed)
      - Merge-split moves for regime housekeeping
    """

    def __init__(self, config: DPFConfig | None = None, d_mamba: int = 64):
        super().__init__()
        if config is None:
            config = DPFConfig()
        self.config = config

        self.transition = DifferentiableTransition(config, d_mamba)
        self.observation = DifferentiableObservation(config, d_mamba)

        # ── Learnable HDP concentration parameter α ──────────────
        # Controls how eagerly new regimes are birthed.
        # Stored in unconstrained space; use self.alpha (property) for
        # the positive value via softplus.  Initialized so that
        # softplus(raw) ≈ 1.0  →  raw = log(exp(1) - 1) ≈ 0.5413.
        self._log_alpha = nn.Parameter(
            torch.tensor(math.log(math.exp(1.0) - 1.0))
        )

        # ── Hysteresis damping state for merge/split ────────────
        self._merge_split_cooldown: int = 0
        self._regime_stability_counter: dict[int, int] = {}
        self._last_alpha: float | None = None
        self._ess_cache: float | None = None

        n_params = sum(p.numel() for p in self.parameters())
        log.info(f"DifferentiableParticleFilter: {n_params:,} parameters, "
                 f"{config.n_particles} particles")

    # ── Concentration α accessor ─────────────────────────────────

    @property
    def alpha(self) -> Tensor:
        """Positive concentration parameter via softplus transform."""
        return F.softplus(self._log_alpha)

    # ── Merge-Split Moves ────────────────────────────────────────

    @torch.no_grad()
    def merge_split_check(
        self,
        regime_posteriors: Tensor,
        threshold: float = 0.05,
        k_active: int | None = None,
    ) -> dict:
        """Housekeeping: merge similar regimes, split bimodal ones.

        Should be called periodically (e.g. every N timesteps or at
        end-of-day) — NOT every particle-filter step.

        Args:
            regime_posteriors: (K,) weight-averaged regime posterior
                from ParticleCloud.regime_posterior.
            threshold: KL-divergence threshold for merging.
            k_active: number of active regimes (defaults to config).

        Returns:
            dict with keys:
              action: "merge" | "split" | "none"
              detail: human-readable description (for logging)
              merge_pair: (keep, remove) indices if merge
              split_from: source index if split
              split_into: (new_a, new_b) indices if split
        """
        if k_active is None:
            k_active = self.config.k_regimes

        # ── Hysteresis guards ────────────────────────────────────
        # 1. Cooldown: block merge/split for 3 cycles after last action
        if self._merge_split_cooldown > 0:
            self._merge_split_cooldown -= 1
            return {"action": "blocked_cooldown", "detail": f"cooldown active ({self._merge_split_cooldown + 1} cycles remaining)"}

        # 2. ESS check: block when ESS is too low (noisy posterior)
        # ESS is cached by cache_ess() which should be called each cycle.
        if hasattr(self, '_ess_cache') and self._ess_cache is not None:
            if self._ess_cache < 0.6 * self.config.n_particles:
                return {"action": "blocked_low_ess", "detail": f"ESS too low ({self._ess_cache:.1f} < {0.6 * self.config.n_particles:.1f})"}

        # 3. Alpha stability: block if alpha changed dramatically
        current_alpha = self.alpha.item()
        if self._last_alpha is not None:
            alpha_delta = abs(math.log(current_alpha) - math.log(self._last_alpha))
            if alpha_delta >= 0.5:
                return {"action": "blocked_alpha_unstable", "detail": f"|log(alpha) - log(last_alpha)| = {alpha_delta:.3f} >= 0.5"}
        self._last_alpha = current_alpha

        result: dict = {"action": "none", "detail": "no merge/split triggered"}
        obs_scale = F.softplus(self.observation.log_obs_scale[:k_active])  # (K,)
        posteriors = regime_posteriors[:k_active]

        # ── MERGE: KL divergence between emission parameters ─────
        # We treat each regime's emission as N(0, scale_k) and compute
        # closed-form KL(k || j).  For two univariate Gaussians with
        # the same mean (regime-conditioned *scale* only):
        #   KL(k||j) = log(σ_j/σ_k) + (σ_k² / (2σ_j²)) - 0.5
        for i in range(k_active):
            for j in range(i + 1, k_active):
                si, sj = obs_scale[i], obs_scale[j]
                kl_ij = torch.log(sj / si) + (si ** 2) / (2 * sj ** 2) - 0.5
                kl_ji = torch.log(si / sj) + (sj ** 2) / (2 * si ** 2) - 0.5
                sym_kl = 0.5 * (kl_ij + kl_ji)  # symmetric KL

                if sym_kl.item() < threshold:
                    # 4. Stability guard: both regimes must be stable >= 3 cycles
                    stable_i = self._regime_stability_counter.get(i, 0) >= 3
                    stable_j = self._regime_stability_counter.get(j, 0) >= 3
                    if not (stable_i and stable_j):
                        continue  # skip this pair, try next
                    # Merge: average parameters, combine posterior mass
                    avg_raw = 0.5 * (
                        self.observation.log_obs_scale.data[i]
                        + self.observation.log_obs_scale.data[j]
                    )
                    # Keep the lower-index regime, absorb the higher
                    keep, remove = (i, j) if posteriors[i] >= posteriors[j] else (j, i)
                    self.observation.log_obs_scale.data[keep] = avg_raw
                    # Zero out the removed regime's scale (will be
                    # unused until k_active is decremented by caller)
                    self.observation.log_obs_scale.data[remove] = 0.0

                    result = {
                        "action": "merge",
                        "detail": (
                            f"Merged regime {remove} into {keep} "
                            f"(sym-KL={sym_kl.item():.4f} < {threshold})"
                        ),
                        "merge_pair": (int(keep), int(remove)),
                    }
                    log.info(f"merge_split_check: {result['detail']}")
                    self._merge_split_cooldown = 3
                    return result

        # ── SPLIT: dominant regime with high variance ─────────────
        # If one regime holds > 40 % posterior mass AND has the highest
        # obs-scale (bimodal indicator), split it.
        max_posterior_idx = int(torch.argmax(posteriors).item())
        max_posterior_val = posteriors[max_posterior_idx].item()
        max_scale_idx = int(torch.argmax(obs_scale).item())

        # 4. Stability guard: regime must be dominant >= 5 cycles to split
        dominant_stable = self._regime_stability_counter.get(max_posterior_idx, 0) >= 5

        if (
            max_posterior_val > 0.4
            and max_posterior_idx == max_scale_idx
            and k_active < len(self.observation.log_obs_scale)  # room to grow
            and dominant_stable
        ):
            src = max_posterior_idx
            new_idx = k_active  # first unused slot

            src_raw = self.observation.log_obs_scale.data[src].clone()
            # Perturb: one child gets +δ, the other −δ
            delta = 0.3 * src_raw.abs().clamp(min=0.1)
            self.observation.log_obs_scale.data[src] = src_raw - delta
            self.observation.log_obs_scale.data[new_idx] = src_raw + delta

            result = {
                "action": "split",
                "detail": (
                    f"Split regime {src} into ({src}, {new_idx}) — "
                    f"posterior={max_posterior_val:.3f}, "
                    f"scale={obs_scale[src].item():.3f}"
                ),
                "split_from": int(src),
                "split_into": (int(src), int(new_idx)),
            }
            log.info(f"merge_split_check: {result['detail']}")
            self._merge_split_cooldown = 3
            return result

        return result

    def update_stability(self, regime_posteriors: Tensor) -> None:
        """Update regime stability counters and ESS cache.

        Should be called each inference cycle (before merge_split_check)
        to track how many consecutive cycles each regime has been dominant.
        Also caches ESS for the low-ESS guard in merge_split_check.

        Args:
            regime_posteriors: (K,) weight-averaged regime posterior
                from ParticleCloud.regime_posterior.
        """
        dominant_idx = int(torch.argmax(regime_posteriors).item())

        # Increment dominant regime, reset all others
        for idx in list(self._regime_stability_counter.keys()):
            if idx != dominant_idx:
                self._regime_stability_counter[idx] = 0
        self._regime_stability_counter[dominant_idx] = (
            self._regime_stability_counter.get(dominant_idx, 0) + 1
        )

    def cache_ess(self, ess_value: float) -> None:
        """Cache the current ESS value for merge_split_check gating.

        Call this each inference cycle with cloud.ess.item().
        """
        self._ess_cache = ess_value

    def initialize(
        self,
        mu_high: float = 5.0,
        sigma_high: float = 2.0,
        mu_low: float = 3.0,
        sigma_low: float = 2.0,
        device: torch.device | None = None,
    ) -> ParticleCloud:
        """Initialize particle cloud from prior."""
        N = self.config.n_particles
        K = self.config.k_regimes

        if device is None:
            device = next(self.parameters()).device

        remaining_high = torch.randn(N, device=device) * sigma_high + mu_high
        remaining_high = remaining_high.clamp(min=0.0)
        remaining_low = torch.randn(N, device=device) * sigma_low + mu_low
        remaining_low = remaining_low.clamp(min=0.0)

        z = torch.randn(N, self.config.d_latent, device=device) * 0.1
        regime_logits = torch.zeros(N, K + 10, device=device)  # over-allocate
        # Initialize with slight prior toward regime 0 (continental)
        regime_logits[:, 0] = 0.5

        log_weights = torch.zeros(N, device=device)

        return ParticleCloud(remaining_high, remaining_low, z, regime_logits, log_weights)

    def step(
        self,
        cloud: ParticleCloud,
        h_t: Tensor,
        obs_remaining_high: Tensor | None,
        obs_remaining_low: Tensor | None,
        source_idx: int = 0,
        k_active: int = 5,
    ) -> ParticleCloud:
        """Full predict → update → resample cycle.

        cloud: current particle state
        h_t: (d_mamba,) Mamba hidden state
        obs_remaining_high/low: observed remaining moves (None if no obs)
        source_idx: data source index for R selection
        k_active: current number of active regimes
        """
        # 1. Predict: advance particles
        cloud = self.transition(cloud, h_t, k_active)

        # 2. Update: weight particles by observation likelihood
        if obs_remaining_high is not None:
            ll_high = self.observation.log_likelihood(
                cloud, obs_remaining_high, source_idx, h_t, "high",
                k_active=k_active,
            )
            cloud.log_weights = cloud.log_weights + ll_high

        if obs_remaining_low is not None:
            ll_low = self.observation.log_likelihood(
                cloud, obs_remaining_low, source_idx, h_t, "low",
                k_active=k_active,
            )
            cloud.log_weights = cloud.log_weights + ll_low

        # 3. Resample if ESS too low
        ess = cloud.ess
        threshold = self.config.n_particles * self.config.ess_threshold_frac
        if ess < threshold:
            cloud = soft_resample(cloud, self.config.resample_temperature)

        return cloud

    def forward(
        self,
        cloud: ParticleCloud,
        h_sequence: Tensor,
        obs_high_sequence: list[Tensor | None],
        obs_low_sequence: list[Tensor | None],
        source_indices: list[int],
        k_active: int = 5,
    ) -> list[ParticleCloud]:
        """Process a full sequence (training mode).

        h_sequence: (T, d_mamba) — Mamba hidden states per timestep
        obs_*_sequence: list of T observation tensors (None = missing)
        source_indices: list of T source indices
        k_active: number of active regimes

        Returns: list of T ParticleClouds (one per timestep)
        """
        T = h_sequence.shape[0]
        clouds = []

        for t in range(T):
            cloud = self.step(
                cloud, h_sequence[t],
                obs_high_sequence[t], obs_low_sequence[t],
                source_indices[t], k_active,
            )
            clouds.append(cloud)

        return clouds
