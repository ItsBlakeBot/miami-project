"""Thin trade-policy layer.

This stays intentionally downstream of weather inference.
"""

from __future__ import annotations

from dataclasses import dataclass

from .edge_detector import EdgeAssessment


@dataclass
class TradeDecision:
    ticker: str
    action: str
    reason: str
    edge_cents: float | None = None


def choose_trade(edge: EdgeAssessment, *, min_edge_cents: float = 4.0) -> TradeDecision:
    if edge.recommendation == "buy_yes" and (edge.tradable_yes_edge_cents or 0.0) >= min_edge_cents:
        return TradeDecision(edge.ticker, "buy_yes", "tradable yes edge exceeds threshold", edge.tradable_yes_edge_cents)
    if edge.recommendation == "buy_no" and (edge.tradable_no_edge_cents or 0.0) >= min_edge_cents:
        return TradeDecision(edge.ticker, "buy_no", "tradable no edge exceeds threshold", edge.tradable_no_edge_cents)
    return TradeDecision(edge.ticker, "hold", "no tradable edge above threshold")
