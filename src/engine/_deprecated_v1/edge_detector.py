"""Compare model-implied fair probabilities to live market books."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# Kalshi fee rates: fee = ceil(rate * P * (1-P) * 100) cents per contract
# where P is the price in dollars (0-1). Max fee at P=0.50.
KALSHI_TAKER_RATE = 0.07
KALSHI_MAKER_RATE = 0.0175


def kalshi_fee_cents(price_cents: float, is_maker: bool = False) -> float:
    """Compute Kalshi fee for a single contract at the given price.

    Fee = ceil(rate * P * (1-P) * 100) where P = price_cents / 100.
    """
    p = max(0.0, min(1.0, price_cents / 100.0))
    rate = KALSHI_MAKER_RATE if is_maker else KALSHI_TAKER_RATE
    return math.ceil(rate * p * (1.0 - p) * 100.0)


@dataclass(frozen=True)
class MarketBook:
    ticker: str
    best_yes_bid_cents: int | None = None
    best_yes_ask_cents: int | None = None
    best_no_bid_cents: int | None = None
    best_no_ask_cents: int | None = None
    last_price_cents: int | None = None


@dataclass
class EdgeAssessment:
    ticker: str
    fair_yes_cents: float | None
    tradable_yes_edge_cents: float | None
    tradable_no_edge_cents: float | None
    yes_fee_cents: float | None = None
    no_fee_cents: float | None = None
    recommendation: str = "none"
    notes: list[str] = field(default_factory=list)


def assess_edge(
    *,
    ticker: str,
    fair_yes_probability: float | None,
    book: MarketBook,
    is_maker: bool = False,
    slippage_cents: float = 1.0,
) -> EdgeAssessment:
    if fair_yes_probability is None:
        return EdgeAssessment(ticker=ticker, fair_yes_cents=None, tradable_yes_edge_cents=None, tradable_no_edge_cents=None)

    fair_yes_cents = round(100.0 * fair_yes_probability, 3)
    yes_edge = None
    no_edge = None
    yes_fee = None
    no_fee = None
    notes: list[str] = []

    if book.best_yes_ask_cents is not None:
        yes_fee = kalshi_fee_cents(book.best_yes_ask_cents, is_maker)
        yes_edge = round(fair_yes_cents - (book.best_yes_ask_cents + yes_fee + slippage_cents), 3)
    if book.best_no_ask_cents is not None:
        fair_no = 100.0 - fair_yes_cents
        no_fee = kalshi_fee_cents(book.best_no_ask_cents, is_maker)
        no_edge = round(fair_no - (book.best_no_ask_cents + no_fee + slippage_cents), 3)

    recommendation = "none"
    if yes_edge is not None and yes_edge > max(no_edge or float('-inf'), 0.0):
        recommendation = "buy_yes"
        notes.append("yes side has positive tradable edge")
    elif no_edge is not None and no_edge > max(yes_edge or float('-inf'), 0.0):
        recommendation = "buy_no"
        notes.append("no side has positive tradable edge")

    return EdgeAssessment(
        ticker=ticker,
        fair_yes_cents=fair_yes_cents,
        tradable_yes_edge_cents=yes_edge,
        tradable_no_edge_cents=no_edge,
        yes_fee_cents=yes_fee,
        no_fee_cents=no_fee,
        recommendation=recommendation,
        notes=notes,
    )
