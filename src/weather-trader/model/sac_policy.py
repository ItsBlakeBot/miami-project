"""Soft Actor-Critic Policy Head for the Trading Brain.

Takes the rich state vector from the Graph-Mamba + MoE encoder and outputs:
    - Continuous: bet sizing (0-1 fraction per bracket per side)
    - Hybrid discrete: action type via Gumbel-softmax

SAC features:
    - Twin Q-networks (reduce overestimation)
    - Entropy regularization (exploration)
    - Off-policy (sample efficient, works with replay buffer)
    - Sortino-shaped reward (penalizes downside, not upside volatility)

The Hard Risk Manager sits OUTSIDE this module and vetoes unsafe actions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


LOG_STD_MIN = -10
LOG_STD_MAX = 2
EPSILON = 1e-6


@dataclass
class SACConfig:
    """SAC policy configuration."""
    d_state: int = 384                   # input state dimension
    n_brackets: int = 6
    hidden_dim: int = 512
    n_hidden_layers: int = 3

    # Action space
    # Per bracket: [size_yes, size_no] → 2 continuous values in [0, 1]
    # Total: n_brackets * 2 = 12 continuous actions
    # Plus route preference: 1 continuous per bracket (maker vs taker)
    action_dim_per_bracket: int = 3      # size_yes, size_no, route_pref

    # SAC hyperparameters
    gamma: float = 0.99                  # discount factor
    tau: float = 0.005                   # soft target update rate
    alpha: float = 0.2                   # entropy temperature (learnable)
    learn_alpha: bool = True
    target_entropy_ratio: float = 0.5    # target entropy as ratio of max

    # Reward shaping
    lambda_cost: float = 0.1            # transaction cost penalty
    lambda_drawdown: float = 0.5        # drawdown penalty
    lambda_sharpe: float = 0.2          # Sharpe bonus


# ---------------------------------------------------------------------------
# Actor (Policy Network)
# ---------------------------------------------------------------------------

class GaussianActor(nn.Module):
    """Squashed Gaussian policy for continuous bet sizing.

    For each bracket, outputs:
        - size_yes: fraction of bankroll to bet YES [0, 1]
        - size_no: fraction of bankroll to bet NO [0, 1]
        - route_pref: maker vs taker preference [-1, 1]
          (negative = maker, positive = taker)

    Uses tanh squashing for bounded outputs.
    """

    def __init__(self, cfg: SACConfig):
        super().__init__()
        self.cfg = cfg
        self.action_dim = cfg.n_brackets * cfg.action_dim_per_bracket

        # Shared trunk
        layers = []
        in_dim = cfg.d_state + cfg.n_brackets * cfg.d_state  # global + per-bracket
        for i in range(cfg.n_hidden_layers):
            out_dim = cfg.hidden_dim
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.SiLU(),
            ])
            in_dim = out_dim
        self.trunk = nn.Sequential(*layers)

        # Mean and log_std heads
        self.mean_head = nn.Linear(cfg.hidden_dim, self.action_dim)
        self.log_std_head = nn.Linear(cfg.hidden_dim, self.action_dim)

        # Per-bracket attention (so each bracket's action considers others)
        self.bracket_attn = nn.MultiheadAttention(
            embed_dim=cfg.d_state,
            num_heads=4,
            dropout=0.1,
            batch_first=True,
        )

    def forward(
        self,
        state: torch.Tensor,              # (B, d_state) global state
        bracket_states: torch.Tensor,      # (B, n_brackets, d_state) per-bracket
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns: action (B, action_dim), log_prob (B, 1)"""

        # Cross-bracket attention for coordinated actions
        bracket_attended, _ = self.bracket_attn(
            bracket_states, bracket_states, bracket_states
        )

        # Concatenate global + per-bracket states
        bracket_flat = bracket_attended.reshape(state.shape[0], -1)
        x = torch.cat([state, bracket_flat], dim=-1)

        # Trunk
        h = self.trunk(x)

        # Mean and std
        mean = self.mean_head(h)
        log_std = self.log_std_head(h)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Reparameterization trick with tanh squashing
        dist = Normal(mean, std)
        x_t = dist.rsample()                        # reparameterized sample
        action = torch.tanh(x_t)                     # squash to [-1, 1]

        # Log probability with tanh correction
        log_prob = dist.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + EPSILON)
        log_prob = log_prob.sum(dim=-1, keepdim=True)  # (B, 1)

        return action, log_prob

    def get_action(
        self,
        state: torch.Tensor,
        bracket_states: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Get action for inference (no log_prob needed)."""
        bracket_attended, _ = self.bracket_attn(
            bracket_states, bracket_states, bracket_states
        )
        bracket_flat = bracket_attended.reshape(state.shape[0], -1)
        x = torch.cat([state, bracket_flat], dim=-1)
        h = self.trunk(x)
        mean = self.mean_head(h)

        if deterministic:
            return torch.tanh(mean)

        log_std = torch.clamp(self.log_std_head(h), LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        x_t = dist.rsample()
        return torch.tanh(x_t)


# ---------------------------------------------------------------------------
# Critic (Twin Q-Networks)
# ---------------------------------------------------------------------------

class QNetwork(nn.Module):
    """Single Q-network: Q(state, action) → scalar value."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 512, n_layers: int = 3):
        super().__init__()
        layers = []
        in_dim = state_dim + action_dim
        for _ in range(n_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
            ])
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=-1))


class TwinQCritic(nn.Module):
    """Twin Q-networks for SAC (reduces overestimation bias)."""

    def __init__(self, cfg: SACConfig):
        super().__init__()
        state_dim = cfg.d_state + cfg.n_brackets * cfg.d_state
        action_dim = cfg.n_brackets * cfg.action_dim_per_bracket

        self.q1 = QNetwork(state_dim, action_dim, cfg.hidden_dim, cfg.n_hidden_layers)
        self.q2 = QNetwork(state_dim, action_dim, cfg.hidden_dim, cfg.n_hidden_layers)

    def forward(
        self,
        state: torch.Tensor,
        bracket_states: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bracket_flat = bracket_states.reshape(state.shape[0], -1)
        full_state = torch.cat([state, bracket_flat], dim=-1)
        return self.q1(full_state, action), self.q2(full_state, action)


# ---------------------------------------------------------------------------
# Action Decoder
# ---------------------------------------------------------------------------

class ActionDecoder:
    """Decodes continuous SAC actions into discrete Kalshi orders.

    SAC outputs per bracket: [size_yes, size_no, route_pref] in [-1, 1]

    Decoding:
        size_yes > threshold → BUY YES with size proportional to magnitude
        size_no > threshold → BUY NO with size proportional to magnitude
        both below threshold → HOLD
        route_pref < 0 → MAKER (limit order)
        route_pref > 0 → TAKER (market order)
    """

    def __init__(
        self,
        action_threshold: float = 0.1,
        max_contracts: int = 25,
        bankroll: float = 1000.0,
    ):
        self.action_threshold = action_threshold
        self.max_contracts = max_contracts
        self.bankroll = bankroll

    def decode(
        self,
        raw_action: torch.Tensor,      # (n_brackets * 3,) in [-1, 1]
        bracket_prices: list[float],    # current mid prices per bracket
    ) -> list[dict]:
        """Decode raw SAC action into trade orders."""
        n_brackets = len(bracket_prices)
        action = raw_action.detach().cpu().numpy()

        orders = []
        for i in range(n_brackets):
            base = i * 3
            size_yes = (action[base] + 1) / 2       # map [-1,1] → [0,1]
            size_no = (action[base + 1] + 1) / 2
            route_pref = action[base + 2]

            # Determine side
            if size_yes > self.action_threshold and size_yes > size_no:
                side = "yes"
                magnitude = size_yes
            elif size_no > self.action_threshold and size_no > size_yes:
                side = "no"
                magnitude = size_no
            else:
                continue  # HOLD

            # Size: magnitude * bankroll / price → contracts
            price = bracket_prices[i] if side == "yes" else (1.0 - bracket_prices[i])
            if price <= 0.01:
                continue
            dollars = self.bankroll * magnitude * 0.05  # max 5% per trade
            contracts = int(dollars / price)
            contracts = max(1, min(contracts, self.max_contracts))

            # Route
            route = "taker" if route_pref > 0 else "maker"

            orders.append({
                "bracket_idx": i,
                "side": side,
                "contracts": contracts,
                "route": route,
                "magnitude": float(magnitude),
                "route_preference": float(route_pref),
            })

        return orders


# ---------------------------------------------------------------------------
# Reward Function
# ---------------------------------------------------------------------------

def compute_reward(
    pnl: float,
    transaction_costs: float,
    drawdown: float,
    rolling_sharpe: float,
    cfg: SACConfig,
) -> float:
    """Sortino-shaped reward with penalty terms.

    r_t = PnL_after_costs
          - lambda_cost * transaction_costs
          - lambda_drawdown * max(0, drawdown)^2
          + lambda_sharpe * rolling_sharpe_bonus
    """
    reward = pnl - transaction_costs

    # Drawdown penalty (quadratic — small drawdowns OK, big drawdowns punished hard)
    if drawdown > 0:
        reward -= cfg.lambda_drawdown * drawdown ** 2

    # Sharpe bonus (reward consistency)
    if rolling_sharpe > 0:
        reward += cfg.lambda_sharpe * min(rolling_sharpe, 3.0)

    return reward


# ---------------------------------------------------------------------------
# Full SAC Agent
# ---------------------------------------------------------------------------

class SACAgent(nn.Module):
    """Complete SAC agent wrapping actor + critic + alpha."""

    def __init__(self, cfg: SACConfig):
        super().__init__()
        self.cfg = cfg

        # Networks
        self.actor = GaussianActor(cfg)
        self.critic = TwinQCritic(cfg)
        self.critic_target = TwinQCritic(cfg)

        # Copy initial weights to target
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Learnable entropy temperature
        action_dim = cfg.n_brackets * cfg.action_dim_per_bracket
        self.target_entropy = -action_dim * cfg.target_entropy_ratio

        if cfg.learn_alpha:
            self.log_alpha = nn.Parameter(torch.tensor(math.log(cfg.alpha)))
        else:
            self.log_alpha = torch.tensor(math.log(cfg.alpha))

        # Action decoder for inference
        self.decoder = ActionDecoder()

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def get_action(
        self,
        state: torch.Tensor,
        bracket_states: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Get action for inference."""
        return self.actor.get_action(state, bracket_states, deterministic)

    def soft_update_target(self):
        """Polyak averaging for target Q-network."""
        for param, target_param in zip(
            self.critic.parameters(), self.critic_target.parameters()
        ):
            target_param.data.copy_(
                self.cfg.tau * param.data + (1 - self.cfg.tau) * target_param.data
            )

    def count_parameters(self) -> dict:
        actor_params = sum(p.numel() for p in self.actor.parameters())
        critic_params = sum(p.numel() for p in self.critic.parameters())
        target_params = sum(p.numel() for p in self.critic_target.parameters())
        return {
            "actor": actor_params,
            "critic": critic_params,
            "critic_target": target_params,
            "total_trainable": actor_params + critic_params,
            "total_with_target": actor_params + critic_params + target_params,
        }
