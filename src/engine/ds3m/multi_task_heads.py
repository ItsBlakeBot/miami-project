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
        self.head_bracket_probs = nn.Linear(d, config.n_brackets)  # softmax
        self.head_next_hour_temp = nn.Linear(d, 1)
        self.head_nwp_bias = nn.Linear(d, 1)
        self.head_regime = nn.Linear(d, config.n_regimes)  # classification

        # ── GradNorm adaptive weights ─────────────────────────────
        # Stored as log-weights to ensure positivity via exp()
        self.log_weights = nn.Parameter(torch.zeros(N_TASKS))

        # GradNorm hyperparameter (restoring force strength)
        self.gradnorm_alpha = config.gradnorm_alpha

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
        """Normalized task weights. Shape (N_TASKS,)."""
        w = torch.exp(self.log_weights)
        return w / w.sum() * N_TASKS  # normalize to sum = N_TASKS

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
            "bracket_probs": F.softmax(
                self.head_bracket_probs(x), dim=-1
            ),  # (B, n_brackets)
            "next_hour_temp": self.head_next_hour_temp(x).squeeze(-1),  # (B,)
            "nwp_bias": self.head_nwp_bias(x).squeeze(-1),            # (B,)
            "regime": self.head_regime(x),                              # (B, n_regimes) — logits
        }

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

        # Regression tasks: MSE loss
        for name in ["daily_max", "daily_min", "max_hour", "min_hour",
                      "next_hour_temp", "nwp_bias"]:
            if name in targets:
                losses[name] = F.mse_loss(predictions[name], targets[name])

        # Bracket probs: cross-entropy (KL divergence from target distribution)
        if "bracket_probs" in targets:
            losses["bracket_probs"] = F.kl_div(
                predictions["bracket_probs"].log().clamp(min=-10),
                targets["bracket_probs"],
                reduction="batchmean",
            )

        # Regime: cross-entropy classification
        if "regime" in targets:
            losses["regime"] = F.cross_entropy(
                predictions["regime"], targets["regime"].long()
            )

        # ── GradNorm weighted loss ────────────────────────────────
        weights = self.task_weights
        task_losses = torch.stack([
            losses.get(name, torch.tensor(0.0, device=weights.device))
            for name in TASK_NAMES
        ])

        # Initialize baseline losses on first call
        if not self._initialized and task_losses.sum() > 0:
            self.initial_losses.copy_(task_losses.detach())
            self._initialized = True

        # Weighted sum
        total_loss = (weights * task_losses).sum()

        result = {"total_loss": total_loss, "task_weights": weights.detach()}
        for name in TASK_NAMES:
            if name in losses:
                result[f"loss_{name}"] = losses[name].detach()

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

        weights = self.task_weights
        n_tasks = N_TASKS

        # Compute gradient norms per task
        grad_norms = []
        for i, name in enumerate(TASK_NAMES):
            loss_key = f"loss_{name}"
            if loss_key not in task_losses:
                grad_norms.append(torch.tensor(1.0, device=weights.device))
                continue

            # We need the loss WITH grad (not detached)
            loss_i = task_losses[loss_key]
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
