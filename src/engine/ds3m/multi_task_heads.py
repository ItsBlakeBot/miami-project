"""Multi-Task Prediction Heads with GradNorm for Weather Brain v3.1.

8-target multi-task prediction with adaptive loss weighting via GradNorm.

Targets:
  0. daily_max       — predicted daily maximum temperature (degF)
  1. daily_min       — predicted daily minimum temperature (degF)
  2. max_hour        — UTC decimal hour of daily max occurrence
  3. min_hour        — UTC decimal hour of daily min occurrence
  4. bracket_probs   — 6-bracket Kalshi probability distribution (softmax)
  5. next_hour_temp  — temperature prediction 1 hour ahead (degF)
  6. nwp_bias        — NWP model bias correction (degF)
  7. regime          — 8-class regime classification (softmax)

GradNorm (Chen et al. 2018) dynamically rebalances task losses by
monitoring gradient magnitudes. Tasks that are lagging get higher
weight, preventing dominant tasks from monopolizing training.

Reference: Chen et al. "GradNorm" (2018)
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from engine.ds3m.wb3_config import MultiTaskConfig

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Task names and indices
# ──────────────────────────────────────────────────────────────────────

TASK_NAMES = [
    "daily_max",
    "daily_min",
    "max_hour",
    "min_hour",
    "bracket_probs",
    "next_hour_temp",
    "nwp_bias",
    "regime",
    "bracket_probs_low",
    "daily_low_bracket",
]

N_TASKS = len(TASK_NAMES)


# ──────────────────────────────────────────────────────────────────────
# Multi-Task Prediction Heads
# ──────────────────────────────────────────────────────────────────────

class MultiTaskHeads(nn.Module):
    """8-target multi-task prediction with GradNorm adaptive weighting.

    Each head is a lightweight linear projection from the shared
    representation. The heavier lifting is done by the upstream
    temporal/spatial encoders; these heads just decode predictions.

    Parameters
    ----------
    config : MultiTaskConfig or None
        Configuration for the multi-task heads.
    """

    def __init__(self, config: MultiTaskConfig | None = None) -> None:
        super().__init__()
        if config is None:
            config = MultiTaskConfig()
        self.config = config
        d = config.d_input

        # ── Prediction heads ──────────────────────────────────────
        self.head_daily_max = nn.Linear(d, 1)
        self.head_daily_min = nn.Linear(d, 1)
        self.head_max_hour = nn.Linear(d, 1)   # UTC decimal hour [0, 24)
        self.head_min_hour = nn.Linear(d, 1)
        self.head_bracket_probs = nn.Linear(d, config.n_brackets)  # softmax (HIGH)
        self.head_bracket_probs_low = nn.Linear(d, config.n_brackets)  # softmax (LOW)
        self.head_next_hour_temp = nn.Linear(d, 1)
        self.head_nwp_bias = nn.Linear(d, 1)
        self.head_regime = nn.Linear(d, config.n_regimes)  # classification

        # ── Initialize regression head biases to target means ────
        # Prevents gradient explosion: initial predictions are in the
        # right ballpark (~85°F for daily_max) instead of ~0.
        # No target normalization needed — raw Fahrenheit throughout.
        with torch.no_grad():
            self.head_daily_max.bias.fill_(85.0)       # Miami avg daily high
            self.head_daily_min.bias.fill_(70.0)        # Miami avg daily low
            self.head_next_hour_temp.bias.fill_(80.0)   # Miami avg temp
            self.head_nwp_bias.bias.fill_(0.0)          # Bias centered at 0
            # max_hour/min_hour use sigmoid*24 → bias=0 gives ~12h, fine

        # ── GradNorm adaptive weights ─────────────────────────────
        # Stored as log-weights to ensure positivity via exp()
        self.log_weights = nn.Parameter(torch.zeros(N_TASKS))

        # GradNorm hyperparameter (restoring force strength)
        self.gradnorm_alpha = config.gradnorm_alpha

        # GradNorm delay: use equal weights for the first N epochs
        self.gradnorm_delay_epochs = getattr(config, "gradnorm_delay_epochs", 10)
        self._gradnorm_active = False

        # Running average of initial loss per task (for relative loss ratios)
        self.register_buffer(
            "initial_losses", torch.ones(N_TASKS)
        )
        self._initialized = False

        n_params = sum(p.numel() for p in self.parameters())
        log.info(
            f"MultiTaskHeads: {n_params:,} parameters, "
            f"{N_TASKS} tasks, d_input={d}"
        )

    @property
    def task_weights(self) -> Tensor:
        """Normalized task weights with safety clamp. Shape (N_TASKS,)."""
        w = torch.exp(self.log_weights)
        w = w / w.sum() * N_TASKS  # normalize to sum = N_TASKS
        # Safety clamp: no task gets more than 3x average weight
        # Prevents GradNorm from giving flow/bracket 10x weight and nuking training
        w = torch.clamp(w, min=0.1, max=3.0)
        return w

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """Compute all task predictions.

        Parameters
        ----------
        x : Tensor
            Shape (B, d_input) — shared representation from encoder.

        Returns
        -------
        dict[str, Tensor]
            Predictions keyed by task name.
        """
        return {
            "daily_max": self.head_daily_max(x).squeeze(-1),          # (B,)
            "daily_min": self.head_daily_min(x).squeeze(-1),          # (B,)
            "max_hour": torch.sigmoid(self.head_max_hour(x).squeeze(-1)) * 24.0,  # (B,) in [0, 24)
            "min_hour": torch.sigmoid(self.head_min_hour(x).squeeze(-1)) * 24.0,  # (B,)
            "bracket_probs": self.head_bracket_probs(x),  # (B, n_brackets) — raw logits (HIGH)
            "bracket_probs_low": self.head_bracket_probs_low(x),  # (B, n_brackets) — raw logits (LOW)
            "next_hour_temp": self.head_next_hour_temp(x).squeeze(-1),  # (B,)
            "nwp_bias": self.head_nwp_bias(x).squeeze(-1),            # (B,)
            "regime": self.head_regime(x),                              # (B, n_regimes) — logits
        }

    def physics_consistency_loss(self, nwp_bias_pred: Tensor, features: Tensor) -> Tensor:
        """Penalize physically implausible NWP biases.

        Constraints:
        1. Bias should be smooth in time (no wild jumps between consecutive hours)
        2. Bias magnitude should correlate with CAPE (convective days have larger biases)
        3. Bias sign should be consistent within a regime

        Parameters
        ----------
        nwp_bias_pred : Tensor
            Predicted NWP bias values.
        features : Tensor
            Input features (unused for now, placeholder for CAPE correlation).

        Returns
        -------
        Tensor
            Scalar physics consistency regularization loss.
        """
        # Temporal smoothness
        if nwp_bias_pred.dim() > 1 and nwp_bias_pred.size(1) > 1:
            temporal_smooth = (nwp_bias_pred[:, 1:] - nwp_bias_pred[:, :-1]).pow(2).mean()
        else:
            temporal_smooth = torch.tensor(0.0, device=nwp_bias_pred.device)

        # Magnitude constraint (bias shouldn't exceed 10 deg F typically)
        magnitude_penalty = F.relu(nwp_bias_pred.abs() - 10.0).pow(2).mean()

        return 0.01 * temporal_smooth + 0.005 * magnitude_penalty

    def compute_loss(
        self,
        predictions: dict[str, Tensor],
        targets: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        """Compute per-task losses and weighted total with GradNorm.

        Parameters
        ----------
        predictions : dict[str, Tensor]
            Predictions from forward().
        targets : dict[str, Tensor]
            Target values keyed by task name. Expected shapes:
              - daily_max, daily_min, max_hour, min_hour: (B,)
              - bracket_probs: (B, n_brackets)
              - next_hour_temp, nwp_bias: (B,)
              - regime: (B,) — integer class labels

        Returns
        -------
        dict[str, Tensor]
            Contains 'total_loss', per-task losses, and 'task_weights'.
        """
        losses = {}

        # Regression tasks: MSE loss (skip NaN targets)
        for name in ["daily_max", "daily_min", "max_hour", "min_hour",
                      "next_hour_temp", "nwp_bias"]:
            if name in targets:
                t = targets[name]
                p = predictions[name]
                valid = ~torch.isnan(t) & ~torch.isinf(t)
                if valid.any():
                    losses[name] = F.mse_loss(p[valid], t[valid])

        # Bracket probs (HIGH): handle both scalar index (-1=unknown) and distribution targets
        if "bracket_target" in targets:
            bracket_idx = targets["bracket_target"].long()
            valid = bracket_idx >= 0
            if valid.any():
                losses["bracket_probs"] = F.cross_entropy(
                    predictions["bracket_probs"][valid], bracket_idx[valid]
                )
        elif "bracket_probs" in targets:
            log_pred = F.log_softmax(predictions["bracket_probs"], dim=-1)
            losses["bracket_probs"] = F.kl_div(
                log_pred,
                targets["bracket_probs"],
                reduction="batchmean",
            )

        # Bracket probs (LOW): same pattern as HIGH
        if "bracket_target_low" in targets:
            bracket_idx_low = targets["bracket_target_low"].long()
            valid_low = bracket_idx_low >= 0
            if valid_low.any():
                losses["bracket_probs_low"] = F.cross_entropy(
                    predictions["bracket_probs_low"][valid_low], bracket_idx_low[valid_low]
                )

        # Regime: skip entirely when ALL targets are -1 (produces NaN otherwise)
        if "regime" in targets:
            regime_tgt = targets["regime"].long()
            if (regime_tgt >= 0).any():
                losses["regime"] = F.cross_entropy(
                    predictions["regime"], regime_tgt,
                    ignore_index=-1,
                )

        # ── Physics consistency regularization on NWP bias ─────────
        if "nwp_bias" in losses and "nwp_bias" in predictions:
            physics_loss = self.physics_consistency_loss(
                predictions["nwp_bias"],
                features=predictions.get("next_hour_temp", predictions["nwp_bias"]),
            )
            losses["nwp_bias"] = losses["nwp_bias"] + physics_loss

        # ── GradNorm weighted loss (with delayed activation) ──────
        # Before GradNorm activates, use equal weights.
        # GradNorm activates after self.gradnorm_delay_epochs epochs.
        weights = self.task_weights
        dev = weights.device
        dt = weights.dtype

        task_losses = torch.stack([
            losses.get(name, torch.tensor(0.0, device=dev, dtype=dt))
            for name in TASK_NAMES
        ])

        # Replace any NaN/Inf in individual task losses with 0
        task_losses = torch.nan_to_num(task_losses, nan=0.0, posinf=1e6, neginf=0.0)

        # Initialize baseline losses for GradNorm (clamp to avoid 0/0)
        if not self._initialized and task_losses.sum() > 0:
            self.initial_losses.copy_(task_losses.detach().clamp(min=1e-6))
            self._initialized = True

        # If GradNorm not yet active, use equal weights (sum of losses)
        if not getattr(self, '_gradnorm_active', False):
            total_loss = task_losses.sum()
        else:
            # GradNorm weighted sum
            safe_weights = torch.nan_to_num(weights, nan=1.0)
            total_loss = (safe_weights * task_losses).sum()

        result = {"total_loss": total_loss, "task_weights": weights.detach()}
        result["_raw_losses"] = losses  # non-detached, for GradNorm
        for name in TASK_NAMES:
            if name in losses:
                result[f"loss_{name}"] = losses[name].detach()  # detached, for logging

        return result

    def gradnorm_step(
        self,
        shared_params: list[nn.Parameter],
        task_losses: dict[str, Tensor],
    ) -> None:
        """GradNorm update step: adjust task weights based on gradient magnitudes.

        Call this after the main optimizer step to rebalance weights.

        Parameters
        ----------
        shared_params : list[nn.Parameter]
            Parameters of the shared encoder (used for gradient computation).
        task_losses : dict[str, Tensor]
            Per-task losses from compute_loss().
        """
        if not shared_params:
            return

        # Use non-detached losses from _raw_losses dict
        raw_losses = task_losses.get("_raw_losses", {})
        if not raw_losses:
            return

        if not hasattr(self, '_initial_losses'):
            self._initial_losses = {k: v.detach().clone() for k, v in raw_losses.items()}

        weights = self.task_weights
        n_tasks = N_TASKS

        # Compute gradient norms per task
        grad_norms = []
        for i, name in enumerate(TASK_NAMES):
            if name not in raw_losses:
                grad_norms.append(torch.tensor(1.0, device=weights.device))
                continue

            # Use non-detached loss from _raw_losses
            loss_i = raw_losses[name]
            if not loss_i.requires_grad:
                grad_norms.append(torch.tensor(1.0, device=weights.device))
                continue

            # Compute gradient of weighted loss w.r.t. shared params
            weighted_loss = weights[i] * loss_i
            grads = torch.autograd.grad(
                weighted_loss, shared_params[0],
                retain_graph=True, allow_unused=True,
            )
            if grads[0] is not None:
                grad_norms.append(grads[0].norm())
            else:
                grad_norms.append(torch.tensor(1.0, device=weights.device))

        grad_norms = torch.stack(grad_norms)

        # Compute relative training rates
        loss_ratios = torch.stack([
            task_losses.get(f"loss_{name}", self.initial_losses[i])
            / self.initial_losses[i].clamp(min=1e-6)
            for i, name in enumerate(TASK_NAMES)
        ])

        # Inverse training rate: tasks that are lagging get higher weight
        mean_ratio = loss_ratios.mean()
        inverse_rate = (loss_ratios / mean_ratio.clamp(min=1e-6)) ** self.gradnorm_alpha

        # Target gradient norm
        mean_grad = grad_norms.mean()
        target_grad = mean_grad * inverse_rate

        # GradNorm loss: minimize ||G_i - target_i||
        gradnorm_loss = (grad_norms - target_grad.detach()).abs().sum()

        # Update log_weights
        self.log_weights.grad = torch.autograd.grad(
            gradnorm_loss, self.log_weights, retain_graph=False,
        )[0]

        # Manual SGD step on log_weights
        with torch.no_grad():
            self.log_weights -= 0.01 * self.log_weights.grad
            self.log_weights.grad = None
