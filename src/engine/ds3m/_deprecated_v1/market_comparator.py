"""Layer 4: Compare DS3M probabilities vs market prices and production.

Computes per-bracket edge (fair value - price - fee) for the DS3M
system and optionally for the production system.  Also provides
post-settlement scoring (CRPS, Brier, accuracy) for model comparison.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

log = logging.getLogger(__name__)

# Same fee rate as production (engine.edge_detector)
KALSHI_TAKER_RATE = 0.07


def kalshi_fee_cents(price_cents: float) -> float:
    """Kalshi taker fee: ceil(0.07 * P * (1-P) * 100) cents."""
    p = max(0.0, min(1.0, price_cents / 100.0))
    return math.ceil(KALSHI_TAKER_RATE * p * (1.0 - p) * 100.0)


# ------------------------------------------------------------------
# Edge dataclass
# ------------------------------------------------------------------

@dataclass
class DS3MEdge:
    """Per-bracket edge comparison record."""

    ticker: str
    market_type: str
    ds3m_probability: float
    production_probability: float | None
    market_probability: float
    ds3m_edge_cents: float
    production_edge_cents: float | None
    conformal_probability: float | None = None

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "market_type": self.market_type,
            "ds3m_probability": round(self.ds3m_probability, 6),
            "production_probability": (
                round(self.production_probability, 6)
                if self.production_probability is not None
                else None
            ),
            "market_probability": round(self.market_probability, 6),
            "ds3m_edge_cents": round(self.ds3m_edge_cents, 2),
            "production_edge_cents": (
                round(self.production_edge_cents, 2)
                if self.production_edge_cents is not None
                else None
            ),
            "conformal_probability": (
                round(self.conformal_probability, 6)
                if self.conformal_probability is not None
                else None
            ),
        }


# ------------------------------------------------------------------
# Edge computation
# ------------------------------------------------------------------

def _edge_cents(fair_value_cents: float, price_cents: float) -> float:
    """Edge = fair_value - price - fee.  Positive means underpriced."""
    fee = kalshi_fee_cents(price_cents)
    return fair_value_cents - price_cents - fee


def compute_ds3m_edges(
    ds3m_probs: dict[str, float],
    market_prices: dict[str, float],
    production_probs: dict[str, float] | None = None,
    conformal_probs: dict[str, float] | None = None,
) -> list[DS3MEdge]:
    """Compute edge for each bracket: DS3M vs market, optionally vs production.

    Parameters
    ----------
    ds3m_probs      : ticker -> DS3M P(YES).
    market_prices   : ticker -> last_price_cents (0-100 scale).
    production_probs: ticker -> production P(YES), if available.
    conformal_probs : ticker -> conformally calibrated P(YES), if available.

    Returns
    -------
    List of DS3MEdge, one per ticker present in both ds3m_probs and
    market_prices.
    """
    edges: list[DS3MEdge] = []

    for ticker, ds3m_p in ds3m_probs.items():
        if ticker not in market_prices:
            continue

        price_cents = market_prices[ticker]
        market_p = price_cents / 100.0
        ds3m_fair = ds3m_p * 100.0

        ds3m_edge = _edge_cents(ds3m_fair, price_cents)

        prod_p = production_probs.get(ticker) if production_probs else None
        prod_edge: float | None = None
        if prod_p is not None:
            prod_fair = prod_p * 100.0
            prod_edge = _edge_cents(prod_fair, price_cents)

        # Infer market_type from ticker convention (e.g. contains "HIGH" / "LOW")
        market_type = _infer_market_type(ticker)

        conf_p = conformal_probs.get(ticker) if conformal_probs else None

        edges.append(DS3MEdge(
            ticker=ticker,
            market_type=market_type,
            ds3m_probability=ds3m_p,
            production_probability=prod_p,
            market_probability=market_p,
            ds3m_edge_cents=ds3m_edge,
            production_edge_cents=prod_edge,
            conformal_probability=conf_p,
        ))

    return edges


# ------------------------------------------------------------------
# Post-settlement scoring
# ------------------------------------------------------------------

def crps_binary(prob: float, outcome: bool) -> float:
    """CRPS for a binary outcome: (prob - indicator)^2."""
    indicator = 1.0 if outcome else 0.0
    return (prob - indicator) ** 2


def compute_comparison_metrics(
    ds3m_probs: dict[str, float],
    production_probs: dict[str, float],
    actual_outcomes: dict[str, bool],
) -> dict:
    """Post-settlement comparison: CRPS, Brier score, accuracy for both systems.

    Only tickers present in all three dicts are scored.
    """
    ds3m_crps: list[float] = []
    prod_crps: list[float] = []
    ds3m_correct = 0
    prod_correct = 0
    n = 0

    for ticker in actual_outcomes:
        if ticker not in ds3m_probs or ticker not in production_probs:
            continue

        outcome = actual_outcomes[ticker]
        dp = ds3m_probs[ticker]
        pp = production_probs[ticker]

        ds3m_crps.append(crps_binary(dp, outcome))
        prod_crps.append(crps_binary(pp, outcome))

        # Accuracy: did the system correctly predict the most likely outcome?
        ds3m_pred = dp >= 0.5
        prod_pred = pp >= 0.5
        if ds3m_pred == outcome:
            ds3m_correct += 1
        if prod_pred == outcome:
            prod_correct += 1

        n += 1

    if n == 0:
        log.warning("No overlapping tickers for comparison metrics")
        return {"n": 0}

    return {
        "n": n,
        "ds3m_mean_crps": round(sum(ds3m_crps) / n, 6),
        "production_mean_crps": round(sum(prod_crps) / n, 6),
        "ds3m_brier_score": round(sum(ds3m_crps) / n, 6),  # CRPS == Brier for binary
        "production_brier_score": round(sum(prod_crps) / n, 6),
        "ds3m_accuracy": round(ds3m_correct / n, 4),
        "production_accuracy": round(prod_correct / n, 4),
    }


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _infer_market_type(ticker: str) -> str:
    """Best-effort market type from ticker string."""
    t = ticker.upper()
    if "HIGH" in t:
        return "high"
    if "LOW" in t:
        return "low"
    return "unknown"
