"""DS3M Signal Protocol — interface contract between weather brain and trading brain.

The DS3M emits a frozen signal packet every inference cycle. The TSSM consumes
these signals alongside market data to make trading decisions. The weather model
NEVER sees trading feedback; the trading model NEVER modifies weather signals.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class DS3MSignal:
    """Frozen signal packet from a city's DS3M weather bot.

    This is the interface contract. Any city bot (KMIA, KFLL, KTPA, etc.)
    produces this exact structure. The TSSM consumes it as-is.
    """

    # Identity
    timestamp_utc: str                             # ISO format
    station: str                                   # "KMIA"
    target_date: str                               # "2026-03-26"
    market_type: str = "high"                      # "high" or "low"

    # Bracket probabilities (sum to 1.0)
    bracket_probs: list[float] = field(default_factory=list)   # [0.02, 0.08, 0.35, 0.40, 0.12, 0.03]
    bracket_ranges: list[list[float]] = field(default_factory=list)  # [[0,80],[80,85],[85,90],[90,95],[95,100],[100,150]]
    n_brackets: int = 6

    # Regime posterior
    regime_posterior: list[float] = field(default_factory=list)  # [0.1, 0.6, 0.05, 0.2, 0.05]
    regime_names: list[str] = field(default_factory=list)
    dominant_regime: str = ""
    dominant_regime_prob: float = 0.0

    # Mamba embedding (frozen, high-dimensional context)
    mamba_embedding: list[float] = field(default_factory=list)  # 384-dim from GraphMamba-3
    mamba_embedding_dim: int = 384

    # Kalman filter state
    filtered_temp_f: float = 0.0
    filtered_uncertainty: float = 0.0
    running_max_f: float = 0.0
    running_min_f: float = 0.0
    predicted_max_f: float = 0.0
    predicted_min_f: float = 0.0

    # Conformal prediction intervals
    prediction_interval_90: list[float] = field(default_factory=lambda: [0.0, 0.0])
    prediction_interval_95: list[float] = field(default_factory=lambda: [0.0, 0.0])

    # Metadata
    hours_to_settlement: float = 0.0
    n_obs_today: int = 0
    model_confidence: float = 0.0                   # ESS-based
    particle_ess: float = 0.0
    kf_innovation_magnitude: float = 0.0

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, s: str) -> "DS3MSignal":
        return cls(**json.loads(s))

    def to_feature_vector(self, pca_dim: int = 8) -> list[float]:
        """Extract a fixed-size feature vector for the TSSM encoder.

        Returns ~25 floats that summarize this signal, suitable for
        concatenation with market features in the Mamba input.
        """
        import math

        features = []

        # Bracket probs (n_brackets floats, typically 6)
        features.extend(self.bracket_probs[:self.n_brackets])
        while len(features) < self.n_brackets:
            features.append(0.0)

        # Regime posterior (top 5)
        regime = self.regime_posterior[:5]
        while len(regime) < 5:
            regime.append(0.0)
        features.extend(regime)

        # KF state
        features.append(self.filtered_temp_f / 120.0)       # normalize to ~[0,1]
        features.append(self.filtered_uncertainty / 10.0)
        features.append(self.running_max_f / 120.0)
        features.append(self.running_min_f / 120.0)
        features.append(self.predicted_max_f / 120.0)
        features.append(self.predicted_min_f / 120.0)

        # Time features
        features.append(self.hours_to_settlement / 24.0)
        hours_utc = float(self.timestamp_utc[11:13]) if len(self.timestamp_utc) > 13 else 12.0
        features.append(math.sin(2 * math.pi * hours_utc / 24.0))
        features.append(math.cos(2 * math.pi * hours_utc / 24.0))

        # Confidence
        features.append(self.model_confidence)
        features.append(min(1.0, self.particle_ess / 500.0))  # ESS ratio

        # Mamba embedding summary (PCA to pca_dim)
        # For now: take first pca_dim components or zero-pad
        emb = self.mamba_embedding[:pca_dim]
        while len(emb) < pca_dim:
            emb.append(0.0)
        features.extend(emb)

        return features

    @property
    def signal_dim(self) -> int:
        """Total feature dimension from to_feature_vector()."""
        return self.n_brackets + 5 + 6 + 3 + 2 + 8  # 30


@dataclass
class MarketState:
    """Per-bracket market state from Kalshi.

    One of these exists per bracket per city per timestep.
    """
    ticker: str
    bracket_idx: int                    # 0=lowest ... 5=highest
    bracket_floor_f: Optional[float] = None
    bracket_cap_f: Optional[float] = None

    # Price data
    mid_price: float = 0.0              # (best_bid + best_ask) / 2, in dollars [0,1]
    spread: float = 0.0                 # best_ask - best_bid
    last_trade_price: float = 0.0
    yes_bid: float = 0.0
    yes_ask: float = 0.0

    # Volume & flow
    volume_rate: float = 0.0            # trades per minute (rolling 5 min)
    flow_imbalance: float = 0.0         # (buy_vol - sell_vol) / total_vol, [-1, 1]

    # Price dynamics
    price_velocity_5m: float = 0.0      # 5-min price change
    price_velocity_30m: float = 0.0
    price_velocity_1h: float = 0.0

    # Book depth
    bid_depth: float = 0.0              # total contracts at top 5 bid levels
    ask_depth: float = 0.0
    book_asymmetry: float = 0.0         # (bid_depth - ask_depth) / (bid + ask + 1e-8)

    # Open interest
    open_interest: float = 0.0

    def to_feature_vector(self) -> list[float]:
        """20-dim market feature vector per bracket."""
        import math

        return [
            self.mid_price,
            self.spread,
            self.volume_rate / 10.0,              # normalize
            self.flow_imbalance,
            self.price_velocity_5m,
            self.price_velocity_30m,
            self.price_velocity_1h,
            self.bid_depth / 100.0,
            self.ask_depth / 100.0,
            self.book_asymmetry,
            self.open_interest / 1000.0,
            self.last_trade_price,
            # Bracket position features
            float(self.bracket_idx) / 5.0,        # normalized position
            (self.bracket_cap_f or 100.0) / 120.0,
            (self.bracket_floor_f or 60.0) / 120.0,
            # Derived
            1.0 if self.spread < 0.05 else 0.0,   # tight spread flag
            1.0 if self.volume_rate > 2.0 else 0.0,  # active flag
            min(1.0, self.spread / 0.20),          # spread normalized
            math.log1p(self.open_interest) / 10.0,
            self.yes_bid * self.yes_ask,           # bid*ask (volatility proxy)
        ]

    @staticmethod
    def feature_dim() -> int:
        return 20


@dataclass
class PortfolioState:
    """Current portfolio state fed to the TSSM."""
    # Per-bracket positions (positive = long YES, negative = short YES / long NO)
    positions: list[float] = field(default_factory=lambda: [0.0] * 6)
    # Per-bracket unrealized P&L
    unrealized_pnl: list[float] = field(default_factory=lambda: [0.0] * 6)
    # Per-bracket avg entry price
    avg_entry_prices: list[float] = field(default_factory=lambda: [0.0] * 6)

    # Aggregate
    total_exposure: float = 0.0        # sum of |position * price|
    total_unrealized_pnl: float = 0.0
    daily_realized_pnl: float = 0.0
    daily_trade_count: int = 0
    bankroll: float = 1000.0

    # Risk state
    max_drawdown_today: float = 0.0
    peak_equity_today: float = 0.0

    def to_feature_vector(self) -> list[float]:
        """18-dim portfolio state vector."""
        features = []
        # Per-bracket (12 dims)
        for i in range(6):
            features.append(self.positions[i] / 25.0)     # normalize by max position
            features.append(self.unrealized_pnl[i] / 50.0)
        # Aggregate (6 dims)
        features.append(self.total_exposure / 500.0)
        features.append(self.total_unrealized_pnl / 100.0)
        features.append(self.daily_realized_pnl / 100.0)
        features.append(float(self.daily_trade_count) / 30.0)
        features.append(self.max_drawdown_today / 100.0)
        features.append(self.bankroll / 2000.0)
        return features

    @staticmethod
    def feature_dim() -> int:
        return 18
