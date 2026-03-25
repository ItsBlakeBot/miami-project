"""T6.4: Cross-bracket arbitrage detection.

Monitors bracket price strips for probability sum violations.
In a complete bracket market, P(bracket_1) + P(bracket_2) + ... = 1.
When market-implied probabilities (mid prices) sum to != 1.00, there
may be an arbitrage opportunity (after fees).

Types of opportunities:
  - Sum of best-asks < $1.00 - total_fees: buy all brackets (guaranteed profit)
  - Sum of best-bids > $1.00 - total_fees: sell all brackets (guaranteed profit)
  - Individual bracket mispricing vs model: most common, handled by normal trading

Note: Pure arbitrage opportunities are thin and fast on Kalshi.
This module primarily serves as a diagnostic/monitor rather than
a trading strategy. The real alpha is in probabilistic edge detection.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone

log = logging.getLogger(__name__)


@dataclass
class BracketStrip:
    """A complete set of brackets for one market type on one day."""

    target_date: str
    market_type: str  # "high" or "low"
    brackets: list[dict] = field(default_factory=list)
    # Each bracket: {ticker, floor, ceiling, best_ask, best_bid, mid}

    @property
    def sum_of_asks(self) -> float:
        """Sum of best-ask prices (cents). Should be ≥ 100 in fair market."""
        return sum(b.get("best_ask", 0) for b in self.brackets if b.get("best_ask"))

    @property
    def sum_of_bids(self) -> float:
        """Sum of best-bid prices (cents). Should be ≤ 100 in fair market."""
        return sum(b.get("best_bid", 0) for b in self.brackets if b.get("best_bid"))

    @property
    def sum_of_mids(self) -> float:
        """Sum of mid prices (cents). Should be ~100."""
        return sum(b.get("mid", 0) for b in self.brackets if b.get("mid"))

    @property
    def n_brackets(self) -> int:
        return len(self.brackets)


@dataclass
class ArbitrageOpportunity:
    """A detected arbitrage or near-arbitrage."""

    type: str  # "buy_all", "sell_all", "relative_mispricing"
    target_date: str
    market_type: str
    profit_cents: float  # estimated profit after fees per full set
    sum_asks: float
    sum_bids: float
    fee_total: float
    n_brackets: int
    timestamp_utc: str

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "target_date": self.target_date,
            "market_type": self.market_type,
            "profit_cents": round(self.profit_cents, 2),
            "sum_asks": round(self.sum_asks, 2),
            "sum_bids": round(self.sum_bids, 2),
            "fee_total": round(self.fee_total, 2),
            "n_brackets": self.n_brackets,
            "timestamp_utc": self.timestamp_utc,
        }


def kalshi_fee_cents(price_cents: float) -> float:
    """Taker fee for one contract."""
    p = max(0.0, min(1.0, price_cents / 100.0))
    return round(0.07 * p * (1.0 - p) * 100.0, 2)


def scan_bracket_arbitrage(
    db: sqlite3.Connection,
    target_date: str,
    station: str = "KMIA",
) -> list[ArbitrageOpportunity]:
    """Scan bracket strips for arbitrage opportunities.

    Checks if the sum of bracket prices violates the no-arbitrage condition
    after accounting for Kalshi taker fees.

    Args:
        db: SQLite connection with row_factory=sqlite3.Row
        target_date: Climate day to check
        station: Trading station

    Returns:
        List of detected opportunities (may be empty).
    """
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    opportunities: list[ArbitrageOpportunity] = []

    for market_type in ("high", "low"):
        # Get latest market snapshots for this market type's brackets
        rows = db.execute(
            """SELECT ticker, yes_price_cents, no_price_cents
               FROM market_snapshots
               WHERE station = ? AND target_date = ? AND market_type = ?
               ORDER BY timestamp_utc DESC""",
            (station, target_date, market_type),
        ).fetchall()

        if not rows:
            continue

        # Deduplicate: latest per ticker
        seen: dict[str, dict] = {}
        for row in rows:
            ticker = row["ticker"]
            if ticker not in seen:
                yes_p = row["yes_price_cents"]
                no_p = row["no_price_cents"]
                if yes_p is not None and no_p is not None:
                    seen[ticker] = {
                        "ticker": ticker,
                        "best_ask": yes_p,  # cost to buy YES
                        "best_bid": 100.0 - no_p,  # implied YES bid
                        "mid": (yes_p + (100.0 - no_p)) / 2.0,
                    }

        if len(seen) < 2:
            continue

        strip = BracketStrip(
            target_date=target_date,
            market_type=market_type,
            brackets=list(seen.values()),
        )

        # Calculate total fees for buying/selling all brackets
        total_buy_fee = sum(kalshi_fee_cents(b["best_ask"]) for b in strip.brackets)
        total_sell_fee = sum(kalshi_fee_cents(b["best_bid"]) for b in strip.brackets)

        # Check: buy all brackets (sum of asks < 100 - fees)
        if strip.sum_of_asks < 100.0 - total_buy_fee:
            profit = 100.0 - strip.sum_of_asks - total_buy_fee
            opportunities.append(ArbitrageOpportunity(
                type="buy_all",
                target_date=target_date,
                market_type=market_type,
                profit_cents=profit,
                sum_asks=strip.sum_of_asks,
                sum_bids=strip.sum_of_bids,
                fee_total=total_buy_fee,
                n_brackets=strip.n_brackets,
                timestamp_utc=now_utc,
            ))
            log.info(
                "ARBITRAGE (buy_all): %s %s — sum_asks=%.1f¢, fees=%.1f¢, profit=%.1f¢",
                market_type, target_date, strip.sum_of_asks, total_buy_fee, profit,
            )

        # Check: sell all brackets (sum of bids > 100 + fees)
        if strip.sum_of_bids > 100.0 + total_sell_fee:
            profit = strip.sum_of_bids - 100.0 - total_sell_fee
            opportunities.append(ArbitrageOpportunity(
                type="sell_all",
                target_date=target_date,
                market_type=market_type,
                profit_cents=profit,
                sum_asks=strip.sum_of_asks,
                sum_bids=strip.sum_of_bids,
                fee_total=total_sell_fee,
                n_brackets=strip.n_brackets,
                timestamp_utc=now_utc,
            ))
            log.info(
                "ARBITRAGE (sell_all): %s %s — sum_bids=%.1f¢, fees=%.1f¢, profit=%.1f¢",
                market_type, target_date, strip.sum_of_bids, total_sell_fee, profit,
            )

        # Log near-misses for monitoring
        ask_excess = strip.sum_of_asks - 100.0
        bid_deficit = 100.0 - strip.sum_of_bids
        if abs(ask_excess) < 5.0 or abs(bid_deficit) < 5.0:
            log.debug(
                "Bracket strip %s %s: sum_asks=%.1f¢ (excess=%.1f¢), sum_bids=%.1f¢ (deficit=%.1f¢), %d brackets",
                market_type, target_date, strip.sum_of_asks, ask_excess,
                strip.sum_of_bids, bid_deficit, strip.n_brackets,
            )

    return opportunities
