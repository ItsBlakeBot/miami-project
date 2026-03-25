"""Configuration for the weather trader.

The trader is city-agnostic. It reads bracket_estimates from city DBs,
compares to live Kalshi prices, and executes trades.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class KalshiConfig:
    """Kalshi API connection."""
    api_key_id: str = ""
    private_key_path: str = ""
    rest_base: str = "https://api.elections.kalshi.com/trade-api/v2"


@dataclass
class CityConfig:
    """One city the trader monitors."""
    station: str              # e.g., "KMIA"
    db_path: str              # path to city collector DB
    kalshi_high_series: str = ""  # e.g., "KXHIGHMIA"
    kalshi_low_series: str = ""   # e.g., "KXLOWTMIA"


@dataclass
class TradingParams:
    """Position sizing and risk management."""
    edge_threshold: float = 0.15
    kelly_fraction: float = 0.25
    max_contracts_per_trade: int = 50
    bankroll: float = 1000.0
    max_daily_loss: float = 100.0
    max_open_positions: int = 4
    max_consecutive_losses: int = 5
    cooldown_minutes: int = 60


@dataclass
class TraderConfig:
    """Top-level trader configuration."""

    # Trader's own DB (trades, outcomes, params)
    trader_db_path: str = ""

    # Mode: shadow (log only), paper (fake orders), live (real orders)
    mode: str = "shadow"

    # Poll interval
    poll_interval_sec: int = 300

    # Cities to monitor
    cities: list[CityConfig] = field(default_factory=list)

    # Sub-configs
    kalshi: KalshiConfig = field(default_factory=KalshiConfig)
    trading: TradingParams = field(default_factory=TradingParams)
