"""Breeden-Litzenberger market-implied density extraction from Kalshi bracket prices.

Given a set of bracket contract prices, recovers the market-implied probability
distribution over temperature outcomes.  Compares with model probabilities to
identify highest-edge trading opportunities.

Uses only stdlib + math -- no numpy/scipy dependency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MarketImpliedDensity:
    mu_implied: float                          # market-implied mean temperature
    sigma_implied: float                       # market-implied std
    bracket_probs: dict[str, float]            # ticker -> market-implied P(YES)
    density_points: list[tuple[float, float]]  # (temperature, density) pairs


@dataclass
class TradeCandidate:
    ticker: str
    model_prob: float
    market_prob: float
    edge_cents: float       # (model_prob - market_prob) * 100
    confidence_score: float


# ---------------------------------------------------------------------------
# Core: extract market-implied density from bracket prices
# ---------------------------------------------------------------------------

def extract_market_density(
    brackets: list[tuple[str, float | None, float | None, str, float]],
) -> MarketImpliedDensity:
    """Recover the market-implied density from Kalshi bracket prices.

    Parameters
    ----------
    brackets : list of (ticker, floor_f, ceiling_f, market_type, yes_price_cents)
        Each tuple describes one bracket contract.
        - floor_f / ceiling_f are in Fahrenheit (None for directional tails).
        - market_type is ``"high"`` or ``"low"``.
        - yes_price_cents is the last YES price in cents (0-100).

    Returns
    -------
    MarketImpliedDensity
        Histogram-style density with implied moments.

    Notes
    -----
    Breeden-Litzenberger insight: the price of a bracket spanning [s1, s2] is
    the risk-neutral probability that the outcome lands in that interval.  For
    Kalshi YES contracts priced in cents, P(YES) ~ yes_price / 100.

    Directional contracts (over / under) give CDF tails directly:
        P(T < ceiling) = under_price / 100
        P(T >= floor)  = over_price  / 100
    """
    if not brackets:
        return MarketImpliedDensity(
            mu_implied=0.0, sigma_implied=0.0,
            bracket_probs={}, density_points=[],
        )

    # --- Step 1: normalise prices to probabilities --------------------------
    raw_probs: dict[str, float] = {}
    # Separate interior brackets from directional tails
    interior: list[tuple[str, float, float, float]] = []   # (ticker, lo, hi, prob)
    under_tails: list[tuple[str, float, float]] = []       # (ticker, ceiling, prob)
    over_tails: list[tuple[str, float, float]] = []        # (ticker, floor, prob)

    for ticker, floor_f, ceiling_f, _mtype, yes_cents in brackets:
        prob = max(0.0, min(1.0, yes_cents / 100.0))
        raw_probs[ticker] = prob

        is_under = floor_f is None and ceiling_f is not None
        is_over = ceiling_f is None and floor_f is not None

        if is_under:
            under_tails.append((ticker, ceiling_f, prob))  # type: ignore[arg-type]
        elif is_over:
            over_tails.append((ticker, floor_f, prob))      # type: ignore[arg-type]
        elif floor_f is not None and ceiling_f is not None:
            interior.append((ticker, floor_f, ceiling_f, prob))

    # --- Step 2: normalise so probabilities sum to 1 ------------------------
    total = sum(p for _, _, _, p in interior)
    # Add tail contributions if present
    for _, _, p in under_tails:
        total += p
    for _, _, p in over_tails:
        total += p

    if total > 0:
        scale = 1.0 / total
    else:
        scale = 1.0

    bracket_probs: dict[str, float] = {}
    for ticker, lo, hi, p in interior:
        bracket_probs[ticker] = p * scale
    for ticker, _, p in under_tails:
        bracket_probs[ticker] = p * scale
    for ticker, _, p in over_tails:
        bracket_probs[ticker] = p * scale

    # --- Step 3: build histogram density ------------------------------------
    # For each interior bracket, density = P / width at the bracket midpoint.
    density_points: list[tuple[float, float]] = []

    for _ticker, lo, hi, p in interior:
        width = hi - lo
        if width > 0:
            midpoint = (lo + hi) / 2.0
            density = (p * scale) / width
            density_points.append((midpoint, density))

    # For tails, we don't have a natural width — use a nominal 5 degF width
    # so the density is representable, anchored at the boundary.
    TAIL_WIDTH = 5.0
    for _ticker, ceiling, p in under_tails:
        midpoint = ceiling - TAIL_WIDTH / 2.0
        density = (p * scale) / TAIL_WIDTH
        density_points.append((midpoint, density))
    for _ticker, floor, p in over_tails:
        midpoint = floor + TAIL_WIDTH / 2.0
        density = (p * scale) / TAIL_WIDTH
        density_points.append((midpoint, density))

    density_points.sort(key=lambda pt: pt[0])

    # --- Step 4: compute implied moments from histogram ---------------------
    mu = 0.0
    for _ticker, lo, hi, p in interior:
        mu += ((lo + hi) / 2.0) * (p * scale)
    for _ticker, ceiling, p in under_tails:
        mu += (ceiling - TAIL_WIDTH / 2.0) * (p * scale)
    for _ticker, floor, p in over_tails:
        mu += (floor + TAIL_WIDTH / 2.0) * (p * scale)

    var = 0.0
    for _ticker, lo, hi, p in interior:
        midpoint = (lo + hi) / 2.0
        var += ((midpoint - mu) ** 2) * (p * scale)
    for _ticker, ceiling, p in under_tails:
        midpoint = ceiling - TAIL_WIDTH / 2.0
        var += ((midpoint - mu) ** 2) * (p * scale)
    for _ticker, floor, p in over_tails:
        midpoint = floor + TAIL_WIDTH / 2.0
        var += ((midpoint - mu) ** 2) * (p * scale)

    sigma = math.sqrt(max(0.0, var))

    return MarketImpliedDensity(
        mu_implied=round(mu, 2),
        sigma_implied=round(sigma, 2),
        bracket_probs=bracket_probs,
        density_points=density_points,
    )


# ---------------------------------------------------------------------------
# Model vs market divergence
# ---------------------------------------------------------------------------

def compute_model_market_divergence(
    model_probs: dict[str, float],
    market_probs: dict[str, float],
) -> dict[str, float]:
    """Per-bracket KL divergence between model and market probabilities.

    For a single binary contract with model probability p and market
    probability q, the KL divergence D_KL(p || q) is:

        p * ln(p/q) + (1-p) * ln((1-p)/(1-q))

    Parameters
    ----------
    model_probs : dict[str, float]
        ticker -> model P(YES), values in [0, 1].
    market_probs : dict[str, float]
        ticker -> market P(YES), values in [0, 1].

    Returns
    -------
    dict[str, float]
        ticker -> KL divergence, sorted descending by divergence.
        Largest divergence = biggest model/market disagreement = highest
        potential edge.
    """
    EPS = 1e-9  # clamp to avoid log(0)
    result: dict[str, float] = {}

    common_tickers = set(model_probs) & set(market_probs)
    for ticker in common_tickers:
        p = max(EPS, min(1.0 - EPS, model_probs[ticker]))
        q = max(EPS, min(1.0 - EPS, market_probs[ticker]))

        kl = p * math.log(p / q) + (1.0 - p) * math.log((1.0 - p) / (1.0 - q))
        result[ticker] = round(kl, 6)

    # Sort descending by divergence
    return dict(sorted(result.items(), key=lambda kv: kv[1], reverse=True))


# ---------------------------------------------------------------------------
# Trade ranking
# ---------------------------------------------------------------------------

def rank_trades_by_edge(
    estimates: list[dict],
) -> list[TradeCandidate]:
    """Rank brackets by edge-weighted confidence.

    Score = |model_probability - market_probability| * sqrt(volume_proxy)

    Parameters
    ----------
    estimates : list[dict]
        Each dict must contain at minimum:
            - ticker: str
            - model_probability: float  (0-1)
            - market_price: float | None  (cents, 0-100)
        Optional:
            - volume_proxy: float  (default 1.0, e.g. order book depth)

    Returns
    -------
    list[TradeCandidate]
        Ordered by confidence_score descending (best trades first).
    """
    candidates: list[TradeCandidate] = []

    for est in estimates:
        ticker = est.get("ticker")
        model_prob = est.get("model_probability")
        market_cents = est.get("market_price")

        if ticker is None or model_prob is None or market_cents is None:
            continue

        market_prob = market_cents / 100.0
        edge_cents = (model_prob - market_prob) * 100.0
        volume_proxy = est.get("volume_proxy", 1.0)

        confidence = abs(edge_cents) * math.sqrt(max(volume_proxy, 0.01))

        candidates.append(TradeCandidate(
            ticker=ticker,
            model_prob=round(model_prob, 4),
            market_prob=round(market_prob, 4),
            edge_cents=round(edge_cents, 2),
            confidence_score=round(confidence, 4),
        ))

    candidates.sort(key=lambda c: c.confidence_score, reverse=True)
    return candidates
