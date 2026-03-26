"""Differentiable Particle Filter v3 for Weather Brain v3.1.

Upgraded from diff_particle_filter.py with:
  - n_particles: 8000 (up from 500)
  - d_latent: 192 (up from 32)
  - k_regimes: 8 (up from 5)
  - Learned proposal distribution (instead of prior proposal)
  - Soft resampling with Gumbel-Softmax relaxation

Key insight: the learned proposal distribution uses the Mamba hidden
state to propose particles that are more likely to match the next
observation, dramatically improving particle efficiency. This is
critical at 8000 particles where the curse of dimensionality in the
192-dim latent space would otherwise require millions of particles.

Reference:
  Corenflos et al. 2021 — "Differentiable Particle Filtering"
  Chen et al. 2023 — "PyDPF: Differentiable Particle Filters"
  Gu et al. 2015 — "Neural Adaptive Sequential Monte Carlo"
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from engine.ds3m.wb3_config import DPFv3Config

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Particle Cloud State
# ──────────────────────────────────────────────────────────────────────

class ParticleCloudV3:
    """Differentiable particle cloud state for v3.

    All tensors retain gradients for end-to-end training.
    Batch dimension B is explicit for ensemble/batch training.

    Attributes
    ----------
    z : Tensor
        Shape (B, N, d_latent) — continuous latent state per particle.
    regime_logits : Tensor
        Shape (B, N, K) — soft regime assignment logits.
    log_weights : Tensor
        Shape (B, N) — unnormalized log weights.
    """

    def __init__(
        self,
        z: Tensor,
        regime_logits: Tensor,
        log_weights: Tensor,
    ) -> None:
        self.z = z
        self.regime_logits = regime_logits
        self.log_weights = log_weights

    @property
    def weights(self) -> Tensor:
        """Normalized weights via log-sum-exp. Shape (B, N)."""
        return F.softmax(self.log_weights, dim=-1)

    @property
    def ess(self) -> Tensor:
        """Effective sample size per batch. Shape (B,)."""
        w = self.weights
        return 1.0 / (w ** 2).sum(dim=-1)

    @property
    def regime_probs(self) -> Tensor:
        """Per-particle regime probabilities. Shape (B, N, K)."""
        return F.softmax(self.regime_logits, dim=-1)

    @property
    def regime_posterior(self) -> Tensor:
        """Weight-averaged regime posterior. Shape (B, K)."""
        w = self.weights.unsqueeze(-1)  # (B, N, 1)
        return (w * self.regime_probs).sum(dim=1)  # (B, K)

    @property
    def weighted_mean_z(self) -> Tensor:
        """Weighted mean of latent state. Shape (B, d_latent)."""
        w = self.weights.unsqueeze(-1)  # (B, N, 1)
        return (w * self.z).sum(dim=1)  # (B, d_latent)


# ──────────────────────────────────────────────────────────────────────
# Learned Proposal Distribution
# ──────────────────────────────────────────────────────────────────────

class LearnedProposal(nn.Module):
    """Learned proposal distribution for particle filter.

    Instead of proposing from the prior (which wastes particles in
    high-dimensional latent spaces), this network uses the Mamba
    hidden state to propose particles that are likely to match
    the next observation.

    The proposal outputs:
      - z_mean: (B, N, d_latent) — mean of proposed latent state
      - z_log_std: (B, N, d_latent) — log std of proposed latent state
      - regime_logits: (B, N, K) — proposed regime logits

    Parameters
    ----------
    config : DPFv3Config
        Particle filter configuration.
    """

    def __init__(self, config: DPFv3Config) -> None:
        super().__init__()
        self.config = config

        d_input = config.d_mamba + config.d_latent + config.d_regime_embed
        d_hidden = config.proposal_hidden

        # Latent proposal network
        self.z_net = nn.Sequential(
            nn.Linear(d_input, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, config.d_latent * 2),  # mean + log_std
        )

        # Regime proposal network
        self.regime_net = nn.Sequential(
            nn.Linear(d_input, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, config.k_regimes),
        )

        # Regime embedding for input conditioning
        self.regime_embed = nn.Embedding(
            config.k_regimes + 10, config.d_regime_embed
        )

    def forward(
        self,
        cloud: ParticleCloudV3,
        h_t: Tensor,
        k_active: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Propose new particle states.

        Parameters
        ----------
        cloud : ParticleCloudV3
            Current particle cloud.
        h_t : Tensor
            Shape (B, d_mamba) — Mamba hidden state.
        k_active : int
            Number of active regimes.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            (z_proposed, regime_logits, log_q) — proposed latent states,
            proposed regime logits, and log proposal density.
        """
        B, N, d_latent = cloud.z.shape
        device = cloud.z.device

        # Compute soft regime embedding per particle
        regime_probs = cloud.regime_probs[:, :, :k_active]  # (B, N, K)
        regime_ids = torch.arange(k_active, device=device)
        regime_embeds = self.regime_embed(regime_ids)  # (K, d_regime_embed)
        regime_embed = torch.matmul(
            regime_probs, regime_embeds
        )  # (B, N, d_regime_embed)

        # Expand h_t across particles
        h_expanded = h_t.unsqueeze(1).expand(B, N, -1)  # (B, N, d_mamba)

        # Build input
        inp = torch.cat([h_expanded, cloud.z, regime_embed], dim=-1)

        # Propose latent state
        z_params = self.z_net(inp)  # (B, N, 2*d_latent)
        z_mean = z_params[..., :d_latent]
        z_log_std = z_params[..., d_latent:].clamp(-5.0, 2.0)

        # Reparameterization trick
        eps = torch.randn_like(z_mean)
        z_proposed = z_mean + eps * z_log_std.exp()

        # Log proposal density (diagonal Gaussian)
        log_q = (
            -0.5 * (eps ** 2)
            - z_log_std
            - 0.5 * math.log(2 * math.pi)
        ).sum(dim=-1)  # (B, N)

        # Propose regime logits
        regime_logits = self.regime_net(inp)  # (B, N, k_regimes)
        # Pad to full K if needed
        if regime_logits.shape[-1] < cloud.regime_logits.shape[-1]:
            pad = torch.zeros(
                B, N,
                cloud.regime_logits.shape[-1] - regime_logits.shape[-1],
                device=device,
            )
            regime_logits = torch.cat([regime_logits, pad], dim=-1)

        return z_proposed, regime_logits, log_q


# ──────────────────────────────────────────────────────────────────────
# Differentiable Transition Model v3
# ──────────────────────────────────────────────────────────────────────

class TransitionV3(nn.Module):
    """Learnable state transition conditioned on Mamba hidden state + regime.

    Upgrades from v2:
      - Larger latent space (d_latent=192)
      - 8 regimes
      - GRU-like latent update with per-regime gating

    Parameters
    ----------
    config : DPFv3Config
        Particle filter configuration.
    """

    def __init__(self, config: DPFv3Config) -> None:
        super().__init__()
        self.config = config

        self.regime_embed = nn.Embedding(
            config.k_regimes + 10, config.d_regime_embed
        )

        d_input = config.d_mamba + config.d_regime_embed + config.d_latent

        # Regime transition network
        self.regime_transition = nn.Sequential(
            nn.Linear(config.d_mamba + config.d_regime_embed, 128),
            nn.SiLU(),
            nn.Linear(128, config.k_regimes + 10),
        )

        # Latent dynamics (GRU-style)
        self.z_gate = nn.Linear(d_input, config.d_latent)
        self.z_candidate = nn.Linear(d_input, config.d_latent)

    def forward(
        self,
        cloud: ParticleCloudV3,
        h_t: Tensor,
        k_active: int,
    ) -> ParticleCloudV3:
        """Advance all particles one timestep.

        Parameters
        ----------
        cloud : ParticleCloudV3
            Current particle cloud.
        h_t : Tensor
            Shape (B, d_mamba) — Mamba hidden state.
        k_active : int
            Number of active regimes.

        Returns
        -------
        ParticleCloudV3
            Updated particle cloud (predict step, weights unchanged).
        """
        B, N, d_latent = cloud.z.shape
        device = h_t.device

        h_expanded = h_t.unsqueeze(1).expand(B, N, -1)

        # Soft regime embedding
        regime_probs = cloud.regime_probs[:, :, :k_active]
        regime_ids = torch.arange(k_active, device=device)
        regime_embeds = self.regime_embed(regime_ids)
        regime_embed = torch.matmul(regime_probs, regime_embeds)

        # Regime transition
        regime_input = torch.cat([h_expanded, regime_embed], dim=-1)
        regime_logit_update = self.regime_transition(regime_input)
        new_regime_logits = cloud.regime_logits.clone()
        k_total = min(k_active, new_regime_logits.shape[-1])
        new_regime_logits[:, :, :k_total] = (
            0.7 * cloud.regime_logits[:, :, :k_total]
            + 0.3 * regime_logit_update[:, :, :k_total]
        )

        # Latent dynamics (GRU-style)
        z_input = torch.cat([h_expanded, regime_embed, cloud.z], dim=-1)
        gate = torch.sigmoid(self.z_gate(z_input))
        candidate = torch.tanh(self.z_candidate(z_input))
        new_z = gate * cloud.z + (1.0 - gate) * candidate

        return ParticleCloudV3(
            z=new_z,
            regime_logits=new_regime_logits,
            log_weights=cloud.log_weights,
        )


# ──────────────────────────────────────────────────────────────────────
# Differentiable Observation Model v3
# ──────────────────────────────────────────────────────────────────────

class ObservationV3(nn.Module):
    """Learnable observation model for the particle filter.

    Maps latent particle state to predicted observations and computes
    log-likelihood of actual observations.

    Parameters
    ----------
    config : DPFv3Config
        Particle filter configuration.
    """

    def __init__(self, config: DPFv3Config) -> None:
        super().__init__()
        self.config = config

        # Emission network: z -> (predicted_temp, log_sigma)
        self.emission_net = nn.Sequential(
            nn.Linear(config.d_latent + config.d_mamba, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 2),  # predicted_value, log_sigma
        )

        # Per-regime observation noise scale
        self.log_obs_scale = nn.Parameter(
            torch.zeros(config.k_regimes + 10)
        )
        # Initialize with reasonable priors
        _init = [1.0, 0.85, 1.4, 1.2, 1.3, 1.1, 0.9, 1.15]
        for i, s in enumerate(_init):
            if i < len(self.log_obs_scale):
                self.log_obs_scale.data[i] = math.log(math.exp(s) - 1.0)

    def log_likelihood(
        self,
        cloud: ParticleCloudV3,
        observation: Tensor,
        h_t: Tensor,
        k_active: int,
    ) -> Tensor:
        """Compute per-particle log-likelihood.

        Parameters
        ----------
        cloud : ParticleCloudV3
            Current particle cloud.
        observation : Tensor
            Shape (B,) or (B, 1) — observed value.
        h_t : Tensor
            Shape (B, d_mamba) — Mamba hidden state.
        k_active : int
            Number of active regimes.

        Returns
        -------
        Tensor
            Shape (B, N) — log p(obs | particle_i).
        """
        B, N, d_latent = cloud.z.shape

        if observation.dim() == 1:
            observation = observation.unsqueeze(-1)

        h_expanded = h_t.unsqueeze(1).expand(B, N, -1)
        emission_input = torch.cat([cloud.z, h_expanded], dim=-1)
        emission = self.emission_net(emission_input)  # (B, N, 2)

        pred = emission[:, :, 0]  # (B, N)
        log_sigma_base = emission[:, :, 1]  # (B, N)

        # Per-regime noise scaling
        regime_scales = F.softplus(self.log_obs_scale[:k_active])
        regime_probs = cloud.regime_probs[:, :, :k_active]
        per_particle_scale = torch.matmul(
            regime_probs, regime_scales
        )  # (B, N)

        sigma = (F.softplus(log_sigma_base) * per_particle_scale).clamp(
            min=0.1, max=5.0
        )

        # Gaussian log-likelihood
        obs_expanded = observation.expand(B, N)
        ll = (
            -0.5 * ((obs_expanded - pred) / sigma) ** 2
            - torch.log(sigma)
            - 0.5 * math.log(2 * math.pi)
        )

        return ll


# ──────────────────────────────────────────────────────────────────────
# Soft Resampling
# ──────────────────────────────────────────────────────────────────────

def soft_resample_v3(
    cloud: ParticleCloudV3,
    temperature: float = 0.5,
) -> ParticleCloudV3:
    """Differentiable soft resampling via Gumbel-Softmax relaxation.

    Uses a soft weighted combination instead of hard systematic
    resampling to maintain gradient flow.

    Parameters
    ----------
    cloud : ParticleCloudV3
        Particle cloud to resample.
    temperature : float
        Gumbel-Softmax temperature (lower = harder resampling).

    Returns
    -------
    ParticleCloudV3
        Resampled cloud with uniform weights.
    """
    B, N, _ = cloud.z.shape

    log_w = cloud.log_weights.unsqueeze(1).expand(B, N, N)  # (B, N, N)
    gumbel_noise = -torch.log(
        -torch.log(torch.rand_like(log_w) + 1e-10) + 1e-10
    )
    soft_assignment = F.softmax(
        (log_w + gumbel_noise) / temperature, dim=-1
    )  # (B, N, N)

    # Soft resample latent state
    new_z = torch.bmm(soft_assignment, cloud.z)  # (B, N, d_latent)

    # Soft resample regime logits
    new_regime_logits = torch.bmm(
        soft_assignment, cloud.regime_logits
    )  # (B, N, K)

    return ParticleCloudV3(
        z=new_z,
        regime_logits=new_regime_logits,
        log_weights=torch.zeros(B, N, device=cloud.z.device),
    )


# ──────────────────────────────────────────────────────────────────────
# Full Differentiable Particle Filter v3
# ──────────────────────────────────────────────────────────────────────

class DifferentiableParticleFilterV3(nn.Module):
    """End-to-end differentiable particle filter v3.

    Full pipeline: propose -> transition -> update -> soft_resample.

    Upgrades from v2:
      - 8000 particles (vs 500)
      - 192-dim latent space (vs 32)
      - 8 regimes (vs 5)
      - Learned proposal distribution
      - Batched operations for ensemble training

    Parameters
    ----------
    config : DPFv3Config or None
        Configuration. Defaults to DPFv3Config().
    """

    def __init__(self, config: DPFv3Config | None = None) -> None:
        super().__init__()
        if config is None:
            config = DPFv3Config()
        self.config = config

        self.proposal = LearnedProposal(config)
        self.transition = TransitionV3(config)
        self.observation = ObservationV3(config)

        # Learnable HDP concentration parameter alpha
        self._log_alpha = nn.Parameter(
            torch.tensor(math.log(math.exp(1.0) - 1.0))
        )

        n_params = sum(p.numel() for p in self.parameters())
        log.info(
            f"DifferentiableParticleFilterV3: {n_params:,} parameters, "
            f"{config.n_particles} particles, "
            f"d_latent={config.d_latent}, k_regimes={config.k_regimes}"
        )

    @property
    def alpha(self) -> Tensor:
        """Positive concentration parameter via softplus."""
        return F.softplus(self._log_alpha)

    def initialize(
        self,
        batch_size: int = 1,
        device: torch.device | None = None,
    ) -> ParticleCloudV3:
        """Initialize particle cloud from prior.

        Parameters
        ----------
        batch_size : int
            Number of batch elements.
        device : torch.device or None
            Device for tensors. Defaults to model's device.

        Returns
        -------
        ParticleCloudV3
            Initialized particle cloud.
        """
        N = self.config.n_particles
        K = self.config.k_regimes
        d_latent = self.config.d_latent

        if device is None:
            device = next(self.parameters()).device

        z = torch.randn(batch_size, N, d_latent, device=device) * 0.1
        regime_logits = torch.zeros(
            batch_size, N, K + 10, device=device
        )
        regime_logits[:, :, 0] = 0.5  # prior toward continental

        log_weights = torch.zeros(batch_size, N, device=device)

        return ParticleCloudV3(z, regime_logits, log_weights)

    def step(
        self,
        cloud: ParticleCloudV3,
        h_t: Tensor,
        observation: Tensor | None = None,
        k_active: int = 8,
    ) -> ParticleCloudV3:
        """Full predict -> update -> resample cycle.

        Parameters
        ----------
        cloud : ParticleCloudV3
            Current particle state.
        h_t : Tensor
            Shape (B, d_mamba) — Mamba hidden state.
        observation : Tensor or None
            Shape (B,) or (B, 1) — observed value (None if no obs).
        k_active : int
            Current number of active regimes.

        Returns
        -------
        ParticleCloudV3
            Updated particle cloud.
        """
        # 1. Propose (learned proposal)
        z_proposed, regime_proposed, log_q = self.proposal(
            cloud, h_t, k_active
        )
        cloud = ParticleCloudV3(
            z=z_proposed,
            regime_logits=regime_proposed,
            log_weights=cloud.log_weights - log_q,
        )

        # 2. Transition
        cloud = self.transition(cloud, h_t, k_active)

        # 3. Update weights with observation
        if observation is not None:
            ll = self.observation.log_likelihood(
                cloud, observation, h_t, k_active
            )
            cloud.log_weights = cloud.log_weights + ll

        # 4. Resample if ESS too low
        ess = cloud.ess  # (B,)
        threshold = self.config.n_particles * self.config.ess_threshold_frac
        if ess.min().item() < threshold:
            cloud = soft_resample_v3(
                cloud, self.config.resample_temperature
            )

        return cloud

    def forward(
        self,
        spatial_state: Tensor,
        regime_posterior: Tensor,
    ) -> Tensor:
        """Extract latent summary for downstream conditioning.

        This is the interface used by WeatherBrainV3: given the
        spatial encoder output and regime posterior, returns a
        summary of the particle filter's latent state.

        Parameters
        ----------
        spatial_state : Tensor
            Shape (B, d_mamba) — spatial encoder output.
        regime_posterior : Tensor
            Shape (B, K) — regime posterior probabilities.

        Returns
        -------
        Tensor
            Shape (B, d_latent) — particle filter latent summary.
        """
        B = spatial_state.shape[0]
        device = spatial_state.device

        # Initialize particles
        cloud = self.initialize(batch_size=B, device=device)

        # Run one step with spatial state as conditioning
        cloud = self.step(
            cloud, spatial_state,
            observation=None,
            k_active=self.config.k_regimes,
        )

        # Return weighted mean of latent state
        return cloud.weighted_mean_z
