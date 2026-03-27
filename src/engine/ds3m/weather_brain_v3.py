"""Weather Brain v3.1 'Thermonuclear' — Full Model Assembly.

Assembles all components into the complete 112M-parameter Weather Brain
v3.1 model for Miami weather prediction and Kalshi bracket trading.

Architecture (per model, x5 ensemble = 560M total):
  1. Feature Masking      — learned defaults for missing data
  2. Multi-Resolution Mamba — 3 temporal branches + fusion (~93M)
  3. GraphMamba Spatial v3  — 47-node graph attention + shared expert (~10M)
  4. DPF v3               — 8000-particle differentiable filter (~4M)
  5. HDP Regime Discovery  — Bayesian nonparametric regime clustering
  6. Multi-Task Heads      — 8 prediction targets with GradNorm (~1M)
  7. Rectified Flow Matching — density estimation via ODE solve (~2M)
  8. Bracket Softmax       — direct bracket probability prediction

Data flow:
  Fine/Medium/Coarse inputs
    -> FeatureMasking (fill missing)
    -> MultiResolutionMamba (temporal encoding + cross-resolution fusion)
    -> GraphMambaSpatialV3 (spatial encoding across stations)
    -> DPF v3 (particle filter latent summary)
    -> HDP (regime posterior)
    -> MultiTaskHeads (8-target predictions)
    -> RectifiedFlowMatching (continuous density for bracket pricing)

Supports:
  - BF16 mixed precision training
  - Random seed initialization for ensemble diversity
  - Pure PyTorch (no CUDA-only dependencies)
  - MPS / CPU / CUDA
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from engine.ds3m.wb3_config import WB3Config
from engine.ds3m.feature_mask import FeatureMasking
from engine.ds3m.multi_res_mamba import MultiResolutionMamba
from engine.ds3m.graph_mamba_v3 import GraphMambaSpatialV3
from engine.ds3m.dpf_v3 import DifferentiableParticleFilterV3
from engine.ds3m.hdp_regime import HDPRegimeManager
from engine.ds3m.multi_task_heads import MultiTaskHeads
from engine.ds3m.flow_matching_v2 import RectifiedFlowMatching

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# HDP Regime Discovery (torch wrapper)
# ──────────────────────────────────────────────────────────────────────

class HDPRegimeDiscovery(nn.Module):
    """Thin nn.Module wrapper around HDPRegimeManager for regime posteriors.

    Computes a soft regime posterior from the spatial encoder output
    using a learnable classification head. The HDP manager handles
    regime birth/merge/naming outside the forward pass.

    Parameters
    ----------
    d_input : int
        Input dimension (from spatial encoder).
    k_regimes : int
        Number of regime classes.
    """

    def __init__(self, d_input: int = 896, k_regimes: int = 8) -> None:
        super().__init__()
        self.regime_head = nn.Sequential(
            nn.Linear(d_input, 256),
            nn.SiLU(),
            nn.Linear(256, k_regimes),
        )
        self.k_regimes = k_regimes

    def forward(self, x: Tensor) -> Tensor:
        """Compute regime posterior probabilities.

        Parameters
        ----------
        x : Tensor
            Shape (B, d_input) — spatial encoder output.

        Returns
        -------
        Tensor
            Shape (B, k_regimes) — regime posterior probabilities.
        """
        logits = self.regime_head(x)
        return F.softmax(logits, dim=-1)


# ──────────────────────────────────────────────────────────────────────
# Weather Brain v3.1 — Full Model
# ──────────────────────────────────────────────────────────────────────

class WeatherBrainV3(nn.Module):
    """Weather Brain v3.1 'Thermonuclear' — 112M parameter weather model.

    Complete architecture for Miami weather prediction and Kalshi
    bracket market trading.

    Parameters
    ----------
    config : WB3Config or None
        Full model configuration. If None, uses defaults.
    seed : int or None
        Random seed for reproducible weight initialization.
        Used for ensemble diversity (each member gets a different seed).
    """

    def __init__(
        self,
        config: WB3Config | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__()

        if config is None:
            config = WB3Config()
        self.config = config

        # Set seed for reproducible initialization (ensemble diversity)
        if seed is not None:
            torch.manual_seed(seed)

        # ── Component assembly ────────────────────────────────────

        # 1. Feature masking (learned defaults for missing data)
        self.feature_mask_fine = FeatureMasking(
            n_features=config.n_features_fine
        )
        self.feature_mask_medium = FeatureMasking(
            n_features=config.n_features_medium
        )
        self.feature_mask_coarse = FeatureMasking(
            n_features=config.n_features_coarse
        )

        # 2. Multi-resolution temporal Mamba
        self.multi_res_mamba = MultiResolutionMamba(
            fine_config=config.fine_branch,
            medium_config=config.medium_branch,
            coarse_config=config.coarse_branch,
            fusion_config=config.fusion,
        )

        # 3. GraphMamba Spatial Encoder v3
        self.graph_mamba = GraphMambaSpatialV3(config=config.graph)

        # 4. DPF v3 (Differentiable Particle Filter)
        self.dpf = DifferentiableParticleFilterV3(config=config.dpf)

        # 5. HDP Regime Discovery
        self.hdp = HDPRegimeDiscovery(
            d_input=config.graph.d_model,
            k_regimes=config.dpf.k_regimes,
        )

        # 6. Multi-task prediction heads
        self.heads = MultiTaskHeads(config=config.tasks)

        # 7. Rectified Flow Matching density estimator
        self.flow_matching = RectifiedFlowMatching(config=config.flow)

        # 8. Direct bracket softmax head
        self.bracket_softmax = nn.Linear(
            config.graph.d_model, config.tasks.n_brackets
        )

        # ── Parameter counting ────────────────────────────────────
        self._print_param_counts()

    def _print_param_counts(self) -> None:
        """Print per-component parameter counts."""
        components = {
            "FeatureMasking (fine)": self.feature_mask_fine,
            "FeatureMasking (medium)": self.feature_mask_medium,
            "FeatureMasking (coarse)": self.feature_mask_coarse,
            "MultiResolutionMamba": self.multi_res_mamba,
            "GraphMambaSpatialV3": self.graph_mamba,
            "DPF v3": self.dpf,
            "HDP Regime": self.hdp,
            "MultiTaskHeads": self.heads,
            "RectifiedFlowMatching": self.flow_matching,
            "BracketSoftmax": self.bracket_softmax,
        }

        total = 0
        lines = ["Weather Brain v3.1 'Thermonuclear' — Parameter Breakdown:"]
        for name, module in components.items():
            n = sum(p.numel() for p in module.parameters())
            total += n
            lines.append(f"  {name:30s}: {n:>12,} params")

        # Handle double-counted params (shared parameters)
        actual_total = sum(p.numel() for p in self.parameters())
        lines.append(f"  {'─' * 44}")
        lines.append(f"  {'Sum':30s}: {total:>12,} params")
        lines.append(f"  {'Actual (deduplicated)':30s}: {actual_total:>12,} params")
        lines.append(f"  {'x5 Ensemble':30s}: {actual_total * 5:>12,} params")

        log.info("\n".join(lines))

    def forward(
        self,
        fine_input: Tensor,
        medium_input: Tensor,
        coarse_input: Tensor,
        station_features: Tensor | None = None,
        feature_masks: dict[str, Tensor] | None = None,
        wind_dirs: Tensor | None = None,
    ) -> dict[str, Tensor | dict]:
        """Full forward pass through Weather Brain v3.1.

        Parameters
        ----------
        fine_input : Tensor
            Shape (B, 32, 64) — 15-min resolution features
            (HRRR sub-hourly + RTMA-RU + ASOS).
        medium_input : Tensor
            Shape (B, 96, 64) — hourly resolution features
            (full NWP + obs + buoy).
        coarse_input : Tensor
            Shape (B, 112, 32) — 3-hourly resolution features
            (GFS/ECMWF/MOS/soundings).
        station_features : Tensor or None
            Shape (B, N_nodes, d_model) — per-station features.
            If None, temporal state is broadcast to all nodes.
        feature_masks : dict[str, Tensor] or None
            Masks for missing features. Keys: 'fine', 'medium', 'coarse'.
            Each tensor has shape matching the corresponding input.
            1.0 = available, 0.0 = missing. If None, all features
            assumed available.
        wind_dirs : Tensor or None
            Shape (B, N_nodes) — wind direction per station (degrees).

        Returns
        -------
        dict[str, Tensor | dict]
            Contains:
              - 'predictions': dict of 8-task predictions
              - 'bracket_probs': (B, n_brackets) softmax bracket probs
              - 'regime_posterior': (B, k_regimes) regime posterior
              - 'particle_state': (B, d_latent) DPF latent summary
              - 'condition': (B, d_condition) flow matching condition vector
              - 'temporal_state': (B, d_fusion) fused temporal representation
              - 'spatial_state': (B, d_model) graph-enriched representation
        """
        # ── 1. Mask missing features ──────────────────────────────
        # Always apply feature masking — uses default all-ones masks when none provided,
        # which lets the learned default values still participate in the computation.
        if feature_masks is None:
            feature_masks = {}
        fine_input = self.feature_mask_fine(
            fine_input, feature_masks.get("fine", torch.ones_like(fine_input))
        )
        medium_input = self.feature_mask_medium(
            medium_input, feature_masks.get("medium", torch.ones_like(medium_input))
        )
        coarse_input = self.feature_mask_coarse(
            coarse_input, feature_masks.get("coarse", torch.ones_like(coarse_input))
        )

        # ── 2. Multi-resolution temporal encoding ─────────────────
        temporal_state = self.multi_res_mamba(
            fine_input, medium_input, coarse_input
        )  # (B, d_fusion=896)

        # ── 3. Spatial encoding across stations ───────────────────
        spatial_state = self.graph_mamba(
            temporal_state,
            station_features=station_features,
            wind_dirs=wind_dirs,
        )  # (B, d_model=896)

        # ── 4. Regime discovery ───────────────────────────────────
        regime_posterior = self.hdp(spatial_state)  # (B, k_regimes)

        # ── 5. Particle filter latent summary ─────────────────────
        particle_state = self.dpf(
            spatial_state, regime_posterior
        )  # (B, d_latent=192)

        # ── 6. Multi-task predictions ─────────────────────────────
        predictions = self.heads(spatial_state)

        # ── 7. Dual density estimation ────────────────────────────
        # Direct bracket probabilities via softmax head
        bracket_probs = F.softmax(
            self.bracket_softmax(spatial_state), dim=-1
        )  # (B, n_brackets)

        # Flow matching conditioning vector
        # Concatenate: spatial_state (896) + regime_posterior (8) + particle_state (192) = 1096
        condition = torch.cat(
            [spatial_state, regime_posterior, particle_state], dim=-1
        )  # (B, d_condition)

        return {
            "predictions": predictions,
            "bracket_probs": bracket_probs,
            "regime_posterior": regime_posterior,
            "particle_state": particle_state,
            "condition": condition,
            "temporal_state": temporal_state,
            "spatial_state": spatial_state,
        }

    def compute_loss(
        self,
        outputs: dict[str, Tensor | dict],
        targets: dict[str, Tensor],
        target_temp: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Compute combined training loss.

        Parameters
        ----------
        outputs : dict
            Forward pass outputs.
        targets : dict[str, Tensor]
            Target values for multi-task heads.
        target_temp : Tensor or None
            Shape (B, 1) — target temperature for flow matching.

        Returns
        -------
        dict[str, Tensor]
            Contains 'total_loss' and all component losses.
        """
        result = {}

        # Multi-task loss (with GradNorm)
        task_result = self.heads.compute_loss(
            outputs["predictions"], targets
        )
        result.update(task_result)

        # Bracket cross-entropy
        if "bracket_target" in targets:
            bracket_loss = F.kl_div(
                outputs["bracket_probs"].log().clamp(min=-10),
                targets["bracket_target"],
                reduction="batchmean",
            )
            result["bracket_loss"] = bracket_loss
        else:
            bracket_loss = torch.tensor(0.0, device=outputs["bracket_probs"].device)

        # Flow matching loss
        if target_temp is not None:
            flow_result = self.flow_matching.training_loss(
                outputs["condition"], target_temp
            )
            result["flow_loss"] = flow_result["flow_loss"]
            result["consistency_loss"] = flow_result["consistency_loss"]
            flow_total = flow_result["loss"]
        else:
            flow_total = torch.tensor(0.0, device=outputs["bracket_probs"].device)

        # Combined loss
        result["total_loss"] = (
            task_result["total_loss"]
            + 0.5 * bracket_loss
            + 0.3 * flow_total
        )

        return result

    @torch.no_grad()
    def predict_brackets(
        self,
        fine_input: Tensor,
        medium_input: Tensor,
        coarse_input: Tensor,
        bracket_edges: list[float],
        station_features: Tensor | None = None,
        feature_masks: dict[str, Tensor] | None = None,
        wind_dirs: Tensor | None = None,
        n_samples: int = 5000,
    ) -> dict[str, Any]:
        """Full inference pipeline for Kalshi bracket trading.

        Returns bracket probabilities from both the direct softmax
        head and the flow matching density estimator.

        Parameters
        ----------
        fine_input, medium_input, coarse_input : Tensor
            Input features at each resolution.
        bracket_edges : list[float]
            Temperature edges defining Kalshi brackets.
        station_features, feature_masks, wind_dirs : optional
            Additional inputs (see forward()).
        n_samples : int
            Number of Monte Carlo samples for flow matching.

        Returns
        -------
        dict
            Contains:
              - 'softmax_probs': list of bracket probabilities (direct head)
              - 'flow_probs': list of bracket probabilities (flow matching)
              - 'ensemble_probs': averaged bracket probabilities
              - 'predictions': dict of all task predictions
              - 'regime_posterior': regime probabilities
        """
        outputs = self.forward(
            fine_input, medium_input, coarse_input,
            station_features=station_features,
            feature_masks=feature_masks,
            wind_dirs=wind_dirs,
        )

        # Direct softmax bracket probabilities (apply softmax here; heads return raw logits)
        bracket_probs = F.softmax(outputs["predictions"]["bracket_probs"], dim=-1)
        softmax_probs = bracket_probs[0].cpu().tolist()

        # Flow matching bracket probabilities
        flow_probs = self.flow_matching.bracket_probabilities(
            outputs["condition"], bracket_edges, n_samples=n_samples
        )

        # Ensemble average (equal weight for now)
        n_brackets = len(softmax_probs)
        ensemble_probs = [
            0.5 * softmax_probs[i] + 0.5 * flow_probs[i]
            for i in range(min(n_brackets, len(flow_probs)))
        ]

        return {
            "softmax_probs": softmax_probs,
            "flow_probs": flow_probs,
            "ensemble_probs": ensemble_probs,
            "predictions": {
                k: v[0].cpu().item() if v.dim() == 1 else v[0].cpu().tolist()
                for k, v in outputs["predictions"].items()
            },
            "regime_posterior": outputs["regime_posterior"][0].cpu().tolist(),
        }


# ──────────────────────────────────────────────────────────────────────
# Ensemble Wrapper
# ──────────────────────────────────────────────────────────────────────

class WeatherBrainV3Ensemble(nn.Module):
    """5-model ensemble of Weather Brain v3.1.

    Each member is initialized with a different random seed for
    diversity. Predictions are aggregated via averaging.

    Total: 5 x 112M = 560M parameters.

    Parameters
    ----------
    config : WB3Config or None
        Shared configuration for all ensemble members.
    """

    def __init__(self, config: WB3Config | None = None) -> None:
        super().__init__()
        if config is None:
            config = WB3Config()
        self.config = config

        self.members = nn.ModuleList([
            WeatherBrainV3(
                config=config,
                seed=config.ensemble_seed_base + i,
            )
            for i in range(config.n_ensemble)
        ])

        total_params = sum(p.numel() for p in self.parameters())
        log.info(
            f"WeatherBrainV3Ensemble: {total_params:,} total parameters "
            f"({config.n_ensemble} members)"
        )

    def forward(
        self,
        fine_input: Tensor,
        medium_input: Tensor,
        coarse_input: Tensor,
        **kwargs: Any,
    ) -> dict[str, Tensor]:
        """Forward pass through all ensemble members.

        Returns averaged predictions across all members.
        """
        all_outputs = [
            member(fine_input, medium_input, coarse_input, **kwargs)
            for member in self.members
        ]

        # Average bracket probabilities
        avg_brackets = torch.stack(
            [o["bracket_probs"] for o in all_outputs]
        ).mean(dim=0)

        # Average regime posterior
        avg_regime = torch.stack(
            [o["regime_posterior"] for o in all_outputs]
        ).mean(dim=0)

        # Average predictions
        avg_predictions = {}
        for key in all_outputs[0]["predictions"]:
            avg_predictions[key] = torch.stack(
                [o["predictions"][key] for o in all_outputs]
            ).mean(dim=0)

        # Concatenate conditions (for flow matching, use first member)
        condition = all_outputs[0]["condition"]

        return {
            "predictions": avg_predictions,
            "bracket_probs": avg_brackets,
            "regime_posterior": avg_regime,
            "condition": condition,
            "member_outputs": all_outputs,
        }

    def state_dict(self, *args, **kwargs):
        """Return state_dict without non-tensor entries.

        Ensemble seeds are stored separately as metadata when saving
        checkpoints (see save/load helpers) rather than polluting the
        state_dict with plain Python lists that break load_state_dict().
        """
        return super().state_dict(*args, **kwargs)

    def get_ensemble_seeds(self) -> list[int]:
        """Return ensemble seeds as separate metadata for checkpoint saving."""
        return [
            getattr(m, 'seed', self.config.ensemble_seed_base + i)
            for i, m in enumerate(self.members)
        ]

    @torch.no_grad()
    def predict_brackets(
        self,
        fine_input: Tensor,
        medium_input: Tensor,
        coarse_input: Tensor,
        bracket_edges: list[float],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Ensemble bracket prediction with uncertainty estimation.

        Returns averaged probabilities plus per-member spread
        for uncertainty estimation.
        """
        member_results = [
            member.predict_brackets(
                fine_input, medium_input, coarse_input,
                bracket_edges, **kwargs,
            )
            for member in self.members
        ]

        n_brackets = len(member_results[0]["ensemble_probs"])

        # Average across members
        avg_probs = [
            sum(r["ensemble_probs"][i] for r in member_results)
            / len(member_results)
            for i in range(n_brackets)
        ]

        # Spread (std) across members for uncertainty
        import statistics
        spread = [
            statistics.stdev(
                [r["ensemble_probs"][i] for r in member_results]
            ) if len(member_results) > 1 else 0.0
            for i in range(n_brackets)
        ]

        return {
            "ensemble_probs": avg_probs,
            "ensemble_spread": spread,
            "member_results": member_results,
            "n_members": len(member_results),
        }
