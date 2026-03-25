"""Bracket pricer with microstructure signals for DS3M.

Converts NSF density → bracket probabilities → fee-adjusted edge,
then applies microstructure signal multipliers for timing-aware execution.

Replaces: engine/bracket_pricer.py + engine/ds3m/market_comparator.py

Five exploitable market patterns:
  1. Morning anchoring bias — markets anchor to NWS consensus
  2. Favorite-longshot bias — tail brackets systematically mispriced
  3. Intraday herding — traders overshoot on hot/cold obs
  4. DST overnight trap — warm front at 12:30 AM = daily high nobody priced
  5. Model run timing arbitrage — HRRR available 45min before market reacts
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import torch
from torch import Tensor

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Bracket definition
# ──────────────────────────────────────────────────────────────────────

@dataclass
class Bracket:
    """A single Kalshi bracket contract."""
    ticker: str
    event_ticker: str
    market_type: str        # "high" or "low"
    floor_strike: float     # lower bound (°F), -inf for edge
    ceiling_strike: float   # upper bound (°F), +inf for edge
    directional: str        # "under" or "over"
    yes_price: float = 0.0  # current market YES price (0-1)
    no_price: float = 0.0   # current market NO price (0-1)
    yes_liquidity: float = 0.0  # $ available on YES side
    no_liquidity: float = 0.0   # $ available on NO side
    best_bid: float = 0.0
    best_ask: float = 0.0

    @property
    def label(self) -> str:
        if self.floor_strike < -900:
            return f"≤{int(self.ceiling_strike)}°F"
        elif self.ceiling_strike > 900:
            return f"≥{int(self.floor_strike)}°F"
        else:
            return f"{int(self.floor_strike)}-{int(self.ceiling_strike)}°F"


# ──────────────────────────────────────────────────────────────────────
# Fee model
# ──────────────────────────────────────────────────────────────────────

def kalshi_taker_fee(price_cents: float) -> float:
    """Kalshi taker fee: ceil(0.07 * P(YES) * P(NO) * 100) cents."""
    p = price_cents / 100.0
    return math.ceil(0.07 * p * (1 - p) * 100)


def kalshi_maker_fee(price_cents: float) -> float:
    """Kalshi maker fee (lower than taker)."""
    p = price_cents / 100.0
    return math.ceil(0.035 * p * (1 - p) * 100)


# ──────────────────────────────────────────────────────────────────────
# Trade signal
# ──────────────────────────────────────────────────────────────────────

@dataclass
class TradeSignal:
    """A scored trade opportunity."""
    bracket: Bracket
    side: str               # "BUY_YES" or "BUY_NO"
    model_prob: float       # DS3M probability
    market_prob: float      # implied from market price
    gross_edge_cents: float
    taker_fee_cents: float
    maker_fee_cents: float
    net_edge_taker: float   # edge after taker fee
    net_edge_maker: float   # edge after maker fee
    kelly_fraction: float   # 15% fractional Kelly
    microstructure_mult: float = 1.0  # signal multiplier
    adjusted_edge: float = 0.0  # edge * microstructure multiplier
    order_type: str = "maker"  # "maker" or "taker"
    regime_name: str = ""
    regime_sizing_mult: float = 1.0


# ──────────────────────────────────────────────────────────────────────
# Microstructure Signals
# ──────────────────────────────────────────────────────────────────────

class MicrostructureSignals:
    """Detects and scores the 5 exploitable market patterns.

    Each signal outputs a multiplier on the raw model edge:
      > 1.0 = edge is likely larger than model thinks (market slow to react)
      < 1.0 = edge is likely smaller (market already reacted)
      = 1.0 = no signal
    """

    def __init__(self):
        self._last_hrrr_run_time: datetime | None = None
        self._last_ecmwf_run_time: datetime | None = None
        self._obs_trend: list[float] = []  # recent obs temps

    def score_all(
        self,
        bracket: Bracket,
        model_prob: float,
        market_prob: float,
        now_utc: datetime,
        hour_local: float,
        is_dst: bool,
        latest_hrrr_shift: float | None = None,  # °F shift from prev HRRR
        hrrr_run_age_minutes: float = 60,
        wind_925_speed: float = 0,
        wind_925_dir: float = 0,
        obs_temp_trend_3h: float = 0,
        model_consensus_shift: float = 0,
    ) -> float:
        """Compute aggregate microstructure multiplier."""
        multipliers = []

        # 1. Morning anchoring bias
        m = self._morning_anchoring(hour_local, now_utc, model_consensus_shift)
        if m != 1.0:
            multipliers.append(m)

        # 2. Favorite-longshot bias
        m = self._favorite_longshot(bracket, model_prob, market_prob)
        if m != 1.0:
            multipliers.append(m)

        # 3. Intraday herding
        m = self._intraday_herding(obs_temp_trend_3h, model_prob, market_prob)
        if m != 1.0:
            multipliers.append(m)

        # 4. DST overnight trap
        m = self._dst_overnight_trap(is_dst, hour_local, wind_925_speed, wind_925_dir, bracket)
        if m != 1.0:
            multipliers.append(m)

        # 5. Model run timing arbitrage
        m = self._model_timing_arb(latest_hrrr_shift, hrrr_run_age_minutes)
        if m != 1.0:
            multipliers.append(m)

        if not multipliers:
            return 1.0
        # Geometric mean of multipliers (multiplicative composition)
        return float(np.exp(np.mean(np.log(multipliers))))

    def _morning_anchoring(
        self, hour_local: float, now_utc: datetime, consensus_shift: float
    ) -> float:
        """Markets anchor to NWS consensus at open. 00Z models create edge at 3AM.

        Strongest signal: 0-6 AM local, when 00Z ECMWF/GFS diverge from market.
        """
        if 0 <= hour_local <= 6 and abs(consensus_shift) > 2.0:
            # Market hasn't woken up yet — edge is real
            return 1.3
        elif 6 < hour_local <= 10 and abs(consensus_shift) > 2.0:
            # Market starting to catch up — still some edge
            return 1.15
        return 1.0

    def _favorite_longshot(
        self, bracket: Bracket, model_prob: float, market_prob: float
    ) -> float:
        """Tail brackets are systematically mispriced (academic finding on Kalshi).

        Low-prob outcomes (1-10¢) win more often than their price implies.
        YES buyers excited by 20:1 payout overpay → NO side has edge.
        """
        price_cents = market_prob * 100
        if price_cents <= 10:
            # Tail bracket — FLB says market overprices YES
            if model_prob < market_prob:
                return 1.25  # BUY_NO signal amplified
            else:
                return 0.85  # BUY_YES damped (you'd be joining the overpriced crowd)
        elif price_cents >= 90:
            # Near-certain bracket — opposite FLB
            if model_prob > market_prob:
                return 1.15
        return 1.0

    def _intraday_herding(
        self, obs_temp_trend_3h: float, model_prob: float, market_prob: float
    ) -> float:
        """When obs run hot/cold, traders herd into adjacent bracket.

        Often overshoots — fade the overreaction.
        Signal: obs trending strongly AND market has moved past model fair value.
        """
        if abs(obs_temp_trend_3h) > 3.0:
            # Strong obs trend — market probably overreacted
            # If model still disagrees with market direction, amplify
            edge_direction = model_prob - market_prob
            trend_direction = 1 if obs_temp_trend_3h > 0 else -1
            if edge_direction * trend_direction < 0:
                # Model fading the trend — microstructure supports this
                return 1.2
        return 1.0

    def _dst_overnight_trap(
        self, is_dst: bool, hour_local: float, wind_925_speed: float,
        wind_925_dir: float, bracket: Bracket,
    ) -> float:
        """During DST, CLI window extends to 12:59 AM EDT.

        Warm front arriving overnight → daily high at 1AM that nobody priced.
        Signal: evening hours + strong 925hPa southerly + upper bracket.
        """
        if not is_dst:
            return 1.0
        if hour_local >= 18 or hour_local <= 2:
            # Evening/overnight during DST
            if 150 <= wind_925_dir <= 240 and wind_925_speed > 20:
                # Strong southerly LLJ — overnight warm advection likely
                if bracket.ceiling_strike > 900 or bracket.floor_strike >= 85:
                    # Upper bracket — market probably underpriced
                    return 1.4
        return 1.0

    def _model_timing_arb(
        self, hrrr_shift: float | None, run_age_minutes: float,
    ) -> float:
        """HRRR updates hourly, available 45min after. Market lags 30-60 min.

        If fresh HRRR shows a big shift, the market hasn't absorbed it yet.
        """
        if hrrr_shift is not None and abs(hrrr_shift) > 2.0 and run_age_minutes < 30:
            # Very fresh HRRR with a big shift — market hasn't caught up
            return 1.3
        elif hrrr_shift is not None and abs(hrrr_shift) > 1.5 and run_age_minutes < 60:
            return 1.1
        return 1.0


# ──────────────────────────────────────────────────────────────────────
# Bracket Pricer
# ──────────────────────────────────────────────────────────────────────

class BracketPricerV2:
    """Full bracket pricing pipeline with microstructure awareness.

    Pipeline:
      1. NSF density → bracket P(YES) for all active brackets
      2. Compare model prob vs market price → raw edge
      3. Apply fee model (taker vs maker)
      4. Apply microstructure signals → adjusted edge
      5. Kelly fraction sizing with regime-conditioned caps
      6. Generate sorted trade signals
    """

    def __init__(
        self,
        min_edge_cents: float = 4.0,
        min_ev_cents: float = 2.0,
        min_price_cents: float = 3.0,
        max_price_cents: float = 80.0,
        kelly_fraction: float = 0.15,
        max_position_per_bracket: float = 0.05,  # 5% of bankroll
        taker_edge_threshold: float = 6.0,  # use taker if net_edge > this
    ):
        self.min_edge_cents = min_edge_cents
        self.min_ev_cents = min_ev_cents
        self.min_price_cents = min_price_cents
        self.max_price_cents = max_price_cents
        self.kelly_fraction = kelly_fraction
        self.max_position_per_bracket = max_position_per_bracket
        self.taker_edge_threshold = taker_edge_threshold
        self.microstructure = MicrostructureSignals()

    def price_all_brackets(
        self,
        brackets: list[Bracket],
        model_probs: dict[str, float],  # bracket_label → model P(YES)
        regime_name: str = "",
        regime_sizing_mult: float = 1.0,
        regime_min_edge: float | None = None,
        microstructure_context: dict | None = None,
    ) -> list[TradeSignal]:
        """Generate trade signals for all brackets.

        Returns: list of TradeSignal sorted by adjusted_edge (descending)
        """
        signals = []
        min_edge = regime_min_edge if regime_min_edge else self.min_edge_cents

        for bracket in brackets:
            label = bracket.label
            if label not in model_probs:
                continue

            model_p = model_probs[label]
            market_p = bracket.yes_price
            if market_p <= 0 or market_p >= 1:
                continue

            price_cents = market_p * 100
            if price_cents < self.min_price_cents or price_cents > self.max_price_cents:
                continue

            # Raw edge
            edge = model_p - market_p
            gross_edge_cents = abs(edge) * 100

            # Fees
            taker_fee = kalshi_taker_fee(price_cents)
            maker_fee = kalshi_maker_fee(price_cents)

            side = "BUY_YES" if edge > 0 else "BUY_NO"

            net_edge_taker = gross_edge_cents - taker_fee
            net_edge_maker = gross_edge_cents - maker_fee

            # Microstructure multiplier
            micro_mult = 1.0
            if microstructure_context:
                micro_mult = self.microstructure.score_all(
                    bracket, model_p, market_p, **microstructure_context
                )

            adjusted_edge = max(net_edge_taker, net_edge_maker) * micro_mult

            if adjusted_edge < min_edge:
                continue

            # EV check
            if side == "BUY_YES":
                ev = model_p * (100 - price_cents) - (1 - model_p) * price_cents
            else:
                ev = (1 - model_p) * price_cents - model_p * (100 - price_cents)

            if ev < self.min_ev_cents:
                continue

            # Kelly sizing
            kelly = self._kelly(model_p, market_p, side == "BUY_YES")
            kelly *= regime_sizing_mult  # regime adjustment

            # Order type: maker if edge is moderate, taker if large
            order_type = "taker" if net_edge_taker > self.taker_edge_threshold else "maker"

            signals.append(TradeSignal(
                bracket=bracket,
                side=side,
                model_prob=round(model_p, 4),
                market_prob=round(market_p, 4),
                gross_edge_cents=round(gross_edge_cents, 2),
                taker_fee_cents=taker_fee,
                maker_fee_cents=maker_fee,
                net_edge_taker=round(net_edge_taker, 2),
                net_edge_maker=round(net_edge_maker, 2),
                kelly_fraction=round(kelly, 4),
                microstructure_mult=round(micro_mult, 3),
                adjusted_edge=round(adjusted_edge, 2),
                order_type=order_type,
                regime_name=regime_name,
                regime_sizing_mult=round(regime_sizing_mult, 2),
            ))

        signals.sort(key=lambda s: s.adjusted_edge, reverse=True)
        return signals

    def _kelly(self, model_p: float, market_price: float, is_yes: bool) -> float:
        """Fractional Kelly criterion with position cap."""
        if is_yes:
            p = model_p
            b = (1 - market_price) / max(market_price, 0.01)
        else:
            p = 1 - model_p
            b = market_price / max(1 - market_price, 0.01)

        q = 1 - p
        kelly = (p * b - q) / max(b, 0.01)
        sized = max(0, kelly * self.kelly_fraction)
        return min(sized, self.max_position_per_bracket)
