"""Trading dataset for TradingBrainV2 training.

Loads frozen DS3M signals (from weather brain backtest) and pairs them
with historical Kalshi trade outcomes to create (state, action, reward)
tuples for Decision Mamba and CQL/SAC training.
"""

import logging
import os
import sqlite3

import numpy as np
import torch
from torch.utils.data import Dataset

log = logging.getLogger("trading_dataset")


class TradingDataset(Dataset):
    """Dataset for trading brain training.

    States: frozen DS3M signals (bracket_probs, regime_posterior, ensemble_std,
            spatial_state) from weather brain backtest.
    Actions: historical bracket selections (which bracket to trade).
    Rewards: P&L from settled Kalshi markets.

    Parameters
    ----------
    db_path : str
        Path to miami_collector.db.
    frozen_signals_path : str or None
        Path to ds3m_frozen_signals.pt. If None, looks in trained_weights/.
    """

    def __init__(self, db_path: str, frozen_signals_path: str = None):
        self.db_path = db_path

        # Load frozen signals from weather brain
        if frozen_signals_path is None:
            # Try common locations
            candidates = [
                "trained_weights/ds3m_frozen_signals.pt",
                os.path.join(os.path.dirname(db_path), "trained_weights", "ds3m_frozen_signals.pt"),
            ]
            for p in candidates:
                if os.path.exists(p):
                    frozen_signals_path = p
                    break

        if frozen_signals_path is None or not os.path.exists(frozen_signals_path):
            raise FileNotFoundError(
                f"Frozen signals not found. Run backtest phase first. "
                f"Tried: {frozen_signals_path or candidates}"
            )

        log.info(f"Loading frozen signals from {frozen_signals_path}")
        self.signals = torch.load(frozen_signals_path, map_location="cpu", weights_only=False)

        # Extract signal tensors
        self.bracket_probs = self.signals["bracket_probs"]      # (N, 6)
        self.regime_posterior = self.signals["regime_posterior"]  # (N, 8)
        self.ensemble_std = self.signals["ensemble_std"]         # (N, 6)
        self.spatial_state = self.signals["spatial_state"]       # (N, 896)

        n_samples = self.bracket_probs.shape[0]
        log.info(f"  Loaded {n_samples} frozen signal samples")

        # Build state tensor: concat all signals
        self.states = torch.cat([
            self.bracket_probs,
            self.regime_posterior,
            self.ensemble_std,
            self.spatial_state,
        ], dim=-1)  # (N, 6+8+6+896 = 916)

        log.info(f"  State dimension: {self.states.shape[-1]}")

        # Load Kalshi trade data for reward computation
        self._build_trade_data()

        log.info(f"  TradingDataset ready: {len(self)} samples, "
                 f"state_dim={self.states.shape[-1]}")

    def _build_trade_data(self):
        """Build action/reward pairs from Kalshi market settlements."""
        conn = sqlite3.connect(self.db_path)

        # Get settled HIGH bracket markets for Miami
        high_df = self._query_settlements(conn, pattern="KXHIGH%MIA%")
        low_df = self._query_settlements(conn, pattern="KXLOW%MIA%")
        conn.close()

        n_samples = self.states.shape[0]

        # For now, create synthetic behavioral prior:
        # Action = argmax of bracket_probs (what the weather brain thinks is most likely)
        # Reward = 1 if correct, -1 if wrong (will be replaced with real P&L)
        self.actions = self.bracket_probs.argmax(dim=-1).float().unsqueeze(-1)  # (N, 1)

        # Compute returns-to-go (cumulative future rewards, discounted)
        # Placeholder: use bracket confidence as proxy
        confidence = self.bracket_probs.max(dim=-1).values  # (N,)
        self.rewards = (confidence - 0.5) * 2  # Scale to roughly [-1, 1]
        self.rewards = self.rewards.unsqueeze(-1)  # (N, 1)

        # Returns-to-go: reverse cumsum of rewards
        rewards_np = self.rewards.squeeze(-1).numpy()
        gamma = 0.99
        rtg = np.zeros_like(rewards_np)
        rtg[-1] = rewards_np[-1]
        for i in range(len(rewards_np) - 2, -1, -1):
            rtg[i] = rewards_np[i] + gamma * rtg[i + 1]
        self.returns_to_go = torch.from_numpy(rtg).float().unsqueeze(-1)  # (N, 1)

        log.info(f"  Actions: {self.actions.shape}, Rewards: {self.rewards.shape}")

    def _query_settlements(self, conn, pattern):
        """Query settled market outcomes."""
        try:
            df = conn.execute(
                """SELECT date, ticker, yes_price, result
                   FROM kalshi_markets
                   WHERE ticker LIKE ? AND result IS NOT NULL
                   ORDER BY date""",
                (pattern,),
            ).fetchall()
            return df
        except Exception:
            return []

    def __len__(self):
        return self.states.shape[0]

    def __getitem__(self, idx):
        return {
            "states": self.states[idx],
            "candles": self.states[idx],  # alias for compatibility
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "returns_to_go": self.returns_to_go[idx],
        }
