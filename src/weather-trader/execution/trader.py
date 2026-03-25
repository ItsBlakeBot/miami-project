"""Trader — reads bracket estimates from city DBs, compares to market prices.

The trader has zero weather knowledge. It sees:
  - City says: "T57 has 30% probability"
  - Market says: "T57 ask is 25c"
  - Edge = 30% - 25% = 5%
  - Kelly says: buy 8 contracts

That's it.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass

from src.config import TradingParams

log = logging.getLogger(__name__)


@dataclass
class CityEstimate:
    """A single bracket probability estimate from a city."""
    station: str
    ticker: str
    market_type: str
    target_date: str
    probability: float
    mu: float | None
    sigma: float | None
    settlement_floor: float | None = None
    settlement_ceil: float | None = None
    timestamp_utc: str = ""


@dataclass
class MarketPrice:
    """Live market price from Kalshi."""
    ticker: str
    yes_bid: float | None    # cents
    yes_ask: float | None    # cents
    last_price: float | None # cents


@dataclass
class TradeDecision:
    """A trade the bot wants to make."""
    station: str
    ticker: str
    market_type: str
    target_date: str
    side: str               # "yes" or "no"
    action: str = "buy"
    price_cents: int = 0
    contracts: int = 0
    our_probability: float = 0.0
    market_probability: float = 0.0
    edge: float = 0.0
    kelly_f: float = 0.0
    settlement_floor: float | None = None
    settlement_ceil: float | None = None


def kelly_size(prob: float, price_cents: int, bankroll: float,
               kelly_frac: float, max_contracts: int) -> tuple[int, float]:
    """Conservative fractional Kelly. Returns (contracts, raw_f*)."""
    price = price_cents / 100.0
    if price <= 0 or price >= 1.0 or prob <= 0 or prob >= 1.0:
        return 0, 0.0
    b = (1.0 / price) - 1.0
    q = 1.0 - prob
    f_star = (prob * b - q) / b
    if f_star <= 0:
        return 0, f_star
    position = bankroll * f_star * kelly_frac
    contracts = int(position / price)
    return max(0, min(contracts, max_contracts)), f_star


def read_city_estimates(city_db: sqlite3.Connection,
                        target_date: str) -> list[CityEstimate]:
    """Read latest bracket estimates from a city's DB."""
    city_db.row_factory = sqlite3.Row
    rows = city_db.execute(
        """SELECT be.station, be.ticker, be.market_type, be.target_date,
                  be.probability, be.mu, be.sigma,
                  ab.settlement_floor, ab.settlement_ceil,
                  be.timestamp_utc
           FROM bracket_estimates be
           LEFT JOIN active_brackets ab ON ab.ticker = be.ticker
           WHERE be.target_date = ?
             AND be.timestamp_utc = (
                 SELECT MAX(timestamp_utc) FROM bracket_estimates
                 WHERE target_date = ?
             )""",
        (target_date, target_date),
    ).fetchall()
    return [CityEstimate(
        station=r["station"], ticker=r["ticker"],
        market_type=r["market_type"], target_date=r["target_date"],
        probability=r["probability"], mu=r["mu"], sigma=r["sigma"],
        settlement_floor=r["settlement_floor"],
        settlement_ceil=r["settlement_ceil"],
        timestamp_utc=r["timestamp_utc"],
    ) for r in rows]


class Trader:
    """Compares city probability estimates to market prices."""

    def __init__(self, params: TradingParams):
        self.p = params

    def evaluate(self, estimates: list[CityEstimate],
                 prices: dict[str, MarketPrice]) -> list[TradeDecision]:
        """Find all brackets with edge above threshold."""
        decisions: list[TradeDecision] = []

        for est in estimates:
            price = prices.get(est.ticker)
            if not price:
                continue

            # --- YES side ---
            if price.yes_ask is not None and price.yes_ask > 0:
                mkt_prob = price.yes_ask / 100.0
                edge = est.probability - mkt_prob
                if edge > self.p.edge_threshold:
                    contracts, kelly_f = kelly_size(
                        est.probability, int(price.yes_ask),
                        self.p.bankroll, self.p.kelly_fraction,
                        self.p.max_contracts_per_trade,
                    )
                    if contracts > 0:
                        decisions.append(TradeDecision(
                            station=est.station, ticker=est.ticker,
                            market_type=est.market_type,
                            target_date=est.target_date,
                            side="yes", price_cents=int(price.yes_ask),
                            contracts=contracts,
                            our_probability=round(est.probability, 4),
                            market_probability=round(mkt_prob, 4),
                            edge=round(edge, 4), kelly_f=round(kelly_f, 4),
                            settlement_floor=est.settlement_floor,
                            settlement_ceil=est.settlement_ceil,
                        ))

            # --- NO side ---
            if price.yes_bid is not None and price.yes_bid > 0:
                no_price = 100 - price.yes_bid
                no_prob = 1.0 - est.probability
                mkt_no_prob = no_price / 100.0
                edge = no_prob - mkt_no_prob
                if edge > self.p.edge_threshold:
                    contracts, kelly_f = kelly_size(
                        no_prob, int(no_price),
                        self.p.bankroll, self.p.kelly_fraction,
                        self.p.max_contracts_per_trade,
                    )
                    if contracts > 0:
                        decisions.append(TradeDecision(
                            station=est.station, ticker=est.ticker,
                            market_type=est.market_type,
                            target_date=est.target_date,
                            side="no", price_cents=int(no_price),
                            contracts=contracts,
                            our_probability=round(no_prob, 4),
                            market_probability=round(mkt_no_prob, 4),
                            edge=round(edge, 4), kelly_f=round(kelly_f, 4),
                            settlement_floor=est.settlement_floor,
                            settlement_ceil=est.settlement_ceil,
                        ))

        decisions.sort(key=lambda d: d.edge, reverse=True)
        return decisions


def format_decisions(decisions: list[TradeDecision], mode: str = "shadow") -> str:
    """Human-readable trade summary."""
    if not decisions:
        return "  No trades with sufficient edge."
    lines = [
        f"  Mode: {mode.upper()}",
        f"  {'Ticker':<35s} {'Side':>4s} {'Price':>6s} {'Qty':>4s} {'Ours':>6s} {'Mkt':>6s} {'Edge':>6s}",
        f"  {'─' * 70}",
    ]
    for d in decisions:
        lines.append(
            f"  {d.ticker:<35s} {d.side:>4s} {d.price_cents:>5d}c {d.contracts:>4d} "
            f"{d.our_probability:>6.1%} {d.market_probability:>6.1%} {d.edge:>+5.1%}"
        )
    return "\n".join(lines)
