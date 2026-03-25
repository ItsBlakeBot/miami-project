from __future__ import annotations

from dataclasses import dataclass


def kalshi_fee_cents(price_cents: float) -> float:
    """Fee approximation used by both weather-trader and city paper traders."""
    p = max(0.0, min(1.0, price_cents / 100.0))
    fee = 0.07 * p * (1.0 - p) * 100.0
    return round(fee, 2)


@dataclass
class TradeRecommendation:
    side: str
    contracts: int
    ask_price_cents: float
    probability: float
    edge_cents: float
    expected_value_cents: float
    sizing: dict


class TraderPolicy:
    """Reusable side-selection + sizing policy for weather city traders.

    Sizing is explicitly driven by:
    - regime confidence
    - adjusted EV-after-fees
    - market price / disagreement sanity controls
    """

    def __init__(
        self,
        *,
        bankroll_dollars: float = 250.0,
        kelly_fraction: float = 0.25,
        max_contracts_per_trade: int = 12,
        min_edge_cents: float = 2.0,
        min_ev_cents: float = 1.0,
        max_market_disagreement: float = 0.45,
        max_model_shift_from_market: float = 0.22,
    ):
        self.bankroll_dollars = bankroll_dollars
        self.kelly_fraction = kelly_fraction
        self.max_contracts_per_trade = max_contracts_per_trade
        self.min_edge_cents = min_edge_cents
        self.min_ev_cents = min_ev_cents
        self.max_market_disagreement = max_market_disagreement
        self.max_model_shift_from_market = max_model_shift_from_market

    @staticmethod
    def _kelly_contracts(
        *,
        prob_win: float,
        price_cents: float,
        bankroll_dollars: float,
        kelly_fraction: float,
        max_contracts: int,
    ) -> tuple[int, float]:
        price = price_cents / 100.0
        if price <= 0 or price >= 1.0 or prob_win <= 0 or prob_win >= 1.0:
            return 0, 0.0

        b = (1.0 / price) - 1.0
        q = 1.0 - prob_win
        f_star = (prob_win * b - q) / b
        if f_star <= 0:
            return 0, f_star

        position_dollars = bankroll_dollars * f_star * kelly_fraction
        contracts = int(position_dollars / price)
        return max(0, min(contracts, max_contracts)), f_star

    @staticmethod
    def _clip(x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    def _recommend_side(
        self,
        *,
        side: str,
        prob_yes: float,
        regime_confidence: float,
        ask_price_cents: float | None,
        ask_qty: int,
    ) -> TradeRecommendation | None:
        if ask_price_cents is None or ask_price_cents <= 0 or ask_qty <= 0:
            return None

        conf = self._clip(float(regime_confidence), 0.0, 1.0)
        p_model = self._clip(prob_yes if side == "yes" else (1.0 - prob_yes), 1e-4, 1.0 - 1e-4)
        p_market = self._clip(ask_price_cents / 100.0, 1e-4, 1.0 - 1e-4)

        fee = kalshi_fee_cents(ask_price_cents)
        raw_edge_c = p_model * 100.0 - ask_price_cents - fee

        # Market sanity: if model is far from price, trust model less unless confidence is strong.
        market_gap = abs(p_model - p_market)
        if market_gap > self.max_market_disagreement and conf < 0.75:
            return None

        raw_model_weight = 0.20 + 0.65 * conf
        disagreement_penalty = max(0.15, 1.0 - market_gap / max(self.max_market_disagreement, 1e-6))
        model_weight = self._clip(raw_model_weight * disagreement_penalty, 0.05, 0.90)

        p_adj = p_market + model_weight * (p_model - p_market)

        # Explicit max distance from market so we don't produce fantasy EV.
        max_shift = self.max_model_shift_from_market * (0.70 + 0.90 * conf)
        delta = self._clip(p_adj - p_market, -max_shift, max_shift)
        p_adj = self._clip(p_market + delta, 1e-4, 1.0 - 1e-4)

        edge_c = p_adj * 100.0 - ask_price_cents - fee
        ev_c = edge_c

        if edge_c < self.min_edge_cents or ev_c < self.min_ev_cents:
            return None

        base_contracts, f_star = self._kelly_contracts(
            prob_win=p_adj,
            price_cents=ask_price_cents,
            bankroll_dollars=self.bankroll_dollars,
            kelly_fraction=self.kelly_fraction,
            max_contracts=self.max_contracts_per_trade,
        )
        if base_contracts <= 0:
            return None

        confidence_scale = 0.35 + 0.65 * conf
        ev_scale = self._clip(ev_c / 6.0, 0.35, 1.75)
        agreement_scale = max(0.20, 1.0 - market_gap / max(self.max_market_disagreement, 1e-6))

        # Price-aware risk scale:
        # - de-emphasize near-0/near-100 asks where small belief errors are costly
        # - keep neutral prices close to full size
        distance_from_mid = abs(p_market - 0.5) / 0.5
        price_scale = 1.0 - 0.45 * self._clip(distance_from_mid, 0.0, 1.0)

        contracts = int(round(base_contracts * confidence_scale * ev_scale * agreement_scale * price_scale))
        contracts = max(1, min(contracts, self.max_contracts_per_trade, ask_qty))

        return TradeRecommendation(
            side=side,
            contracts=contracts,
            ask_price_cents=float(ask_price_cents),
            probability=round(float(p_adj), 4),
            edge_cents=round(float(edge_c), 3),
            expected_value_cents=round(float(ev_c), 3),
            sizing={
                "regime_confidence": round(conf, 4),
                "model_probability": round(float(p_model), 4),
                "market_probability": round(float(p_market), 4),
                "market_gap": round(float(market_gap), 4),
                "model_weight": round(float(model_weight), 4),
                "max_shift_from_market": round(float(max_shift), 4),
                "raw_edge_cents": round(float(raw_edge_c), 3),
                "kelly_f_star": round(float(f_star), 4),
                "base_contracts": int(base_contracts),
                "confidence_scale": round(float(confidence_scale), 4),
                "ev_scale": round(float(ev_scale), 4),
                "agreement_scale": round(float(agreement_scale), 4),
                "price_scale": round(float(price_scale), 4),
                "ask_qty": int(ask_qty),
            },
        )

    def recommend_trade(
        self,
        *,
        prob_yes: float,
        regime_confidence: float,
        yes_ask_cents: float | None,
        yes_ask_qty: int,
        no_ask_cents: float | None,
        no_ask_qty: int,
    ) -> TradeRecommendation | None:
        candidates: list[TradeRecommendation] = []

        yes_rec = self._recommend_side(
            side="yes",
            prob_yes=prob_yes,
            regime_confidence=regime_confidence,
            ask_price_cents=yes_ask_cents,
            ask_qty=yes_ask_qty,
        )
        if yes_rec:
            candidates.append(yes_rec)

        no_rec = self._recommend_side(
            side="no",
            prob_yes=prob_yes,
            regime_confidence=regime_confidence,
            ask_price_cents=no_ask_cents,
            ask_qty=no_ask_qty,
        )
        if no_rec:
            candidates.append(no_rec)

        if not candidates:
            return None

        candidates.sort(key=lambda c: c.expected_value_cents * c.contracts, reverse=True)
        return candidates[0]


class TradingMode:
    """Day-ahead vs intraday trading mode parameters.

    Day-ahead (D-1 evening / D-0 early morning):
      - Positions held to settlement → entry fee only (no exit fee)
      - Conservative Kelly (high uncertainty at longer lead times)
      - Edge source: speed of model ingestion

    Intraday (D-0 through settlement):
      - Round-trips pay entry + exit fees → need higher edge threshold
      - Increasing Kelly as forecast confidence grows through the day
      - Edge source: obs speed + truncated normal CDF convergence
    """

    @staticmethod
    def day_ahead_params(base_bankroll: float = 250.0) -> dict:
        """Conservative parameters for day-ahead positions (held to settlement)."""
        return {
            "bankroll_dollars": base_bankroll,
            "kelly_fraction": 0.12,           # conservative — high uncertainty
            "max_contracts_per_trade": 8,
            "min_edge_cents": 3.0,            # lower threshold — no exit fee
            "min_ev_cents": 2.0,
            "max_market_disagreement": 0.35,
            "max_model_shift_from_market": 0.15,
            "mode": "day_ahead",
        }

    @staticmethod
    def intraday_params(
        base_bankroll: float = 250.0,
        hours_to_settlement: float = 12.0,
    ) -> dict:
        """Adaptive parameters for intraday trading (round-trips pay double fees)."""
        # Kelly increases as settlement approaches and uncertainty shrinks
        # At 12h out: 0.12, at 6h: 0.18, at 2h: 0.25, at 0h: 0.30
        time_factor = max(0.0, min(1.0, 1.0 - hours_to_settlement / 14.0))
        kelly = 0.12 + 0.18 * time_factor

        # Edge threshold higher for intraday (pays entry + exit fees)
        # Double the fee drag means need double the edge
        min_edge = max(2.0, 5.5 - 2.5 * time_factor)

        return {
            "bankroll_dollars": base_bankroll,
            "kelly_fraction": round(kelly, 4),
            "max_contracts_per_trade": int(6 + 8 * time_factor),
            "min_edge_cents": round(min_edge, 3),
            "min_ev_cents": round(max(1.0, 3.0 - 1.5 * time_factor), 3),
            "max_market_disagreement": round(0.25 + 0.20 * time_factor, 4),
            "max_model_shift_from_market": round(0.10 + 0.15 * time_factor, 4),
            "mode": "intraday",
            "hours_to_settlement": round(hours_to_settlement, 2),
            "time_factor": round(time_factor, 4),
        }

    @staticmethod
    def select_mode(hours_to_settlement: float | None, base_bankroll: float = 250.0) -> dict:
        """Auto-select day-ahead or intraday based on time to settlement."""
        if hours_to_settlement is None or hours_to_settlement > 18.0:
            return TradingMode.day_ahead_params(base_bankroll)
        return TradingMode.intraday_params(base_bankroll, hours_to_settlement)


def adaptive_policy_params_from_performance(
    *,
    trade_count: int,
    brier: float | None,
    pnl_per_contract_cents: float | None,
    base_bankroll_dollars: float = 250.0,
    rolling_sharpe: float | None = None,
    max_drawdown_pct: float | None = None,
    calibration_sample_size: int | None = None,
) -> dict:
    """Self-adjusting policy knobs from recent trade quality.

    Enhancements (T6.2):
      - Baker-McHale shrinkage: reduce Kelly based on calibration sample size
      - Drawdown constraint: f ≤ 2 × S² × D_max (S=Sharpe, D_max=max drawdown)
      - Rolling Sharpe tracking for dynamic risk scaling

    Inputs should come from a settled recent window (for example rolling 48h)
    to avoid leaking future outcomes.
    """
    # Sparse data: stay conservative but still active.
    if trade_count < 8 or brier is None or pnl_per_contract_cents is None:
        return {
            "bankroll_dollars": float(base_bankroll_dollars),
            "kelly_fraction": 0.16,
            "max_contracts_per_trade": 8,
            "min_edge_cents": 3.5,
            "min_ev_cents": 2.0,
            "max_market_disagreement": 0.30,
            "max_model_shift_from_market": 0.12,
            "adaptive": {
                "mode": "cold_start",
                "trade_count": int(trade_count),
            },
        }

    brier_clamped = max(0.0, min(0.25, float(brier)))
    pnl_pc = float(pnl_per_contract_cents)

    quality = max(0.0, min(1.0, 1.0 - (brier_clamped / 0.25)))
    profit = max(0.0, min(1.0, (pnl_pc + 8.0) / 16.0))
    composite = 0.65 * quality + 0.35 * profit

    kelly_fraction = max(0.08, min(0.45, 0.10 + 0.32 * composite))

    # Baker-McHale shrinkage: reduce Kelly proportional to estimation error
    # With small calibration samples, our probability estimates are noisy
    # Shrinkage factor ≈ 1 - 1/N where N = calibration sample size
    if calibration_sample_size is not None and calibration_sample_size > 0:
        shrinkage = max(0.5, 1.0 - 1.0 / calibration_sample_size)
        kelly_fraction *= shrinkage

    # Drawdown constraint: f ≤ 2 × S² × D_max
    # Limits Kelly based on realized Sharpe ratio and acceptable max drawdown
    if rolling_sharpe is not None and rolling_sharpe > 0:
        max_acceptable_drawdown = 0.20  # 20% max drawdown
        drawdown_constrained_kelly = 2.0 * rolling_sharpe**2 * max_acceptable_drawdown
        kelly_fraction = min(kelly_fraction, drawdown_constrained_kelly)

    # Drawdown scaling: halve size after 10% drawdown, quarter after 20%
    if max_drawdown_pct is not None:
        if max_drawdown_pct >= 0.30:
            kelly_fraction = 0.0  # STOP trading
        elif max_drawdown_pct >= 0.20:
            kelly_fraction *= 0.25  # quarter size
        elif max_drawdown_pct >= 0.10:
            kelly_fraction *= 0.50  # half size

    kelly_fraction = max(0.04, min(0.45, kelly_fraction))

    max_contracts = int(round(6 + 8 * composite))
    min_edge = max(1.0, min(5.0, 4.6 - 3.4 * composite))
    min_ev = max(0.5, min(3.0, 2.8 - 2.1 * composite))
    max_disagreement = max(0.20, min(0.55, 0.22 + 0.30 * composite))
    max_shift = max(0.06, min(0.30, 0.08 + 0.18 * composite))

    return {
        "bankroll_dollars": float(base_bankroll_dollars),
        "kelly_fraction": round(kelly_fraction, 4),
        "max_contracts_per_trade": int(max_contracts),
        "min_edge_cents": round(min_edge, 3),
        "min_ev_cents": round(min_ev, 3),
        "max_market_disagreement": round(max_disagreement, 4),
        "max_model_shift_from_market": round(max_shift, 4),
        "adaptive": {
            "mode": "rolling",
            "trade_count": int(trade_count),
            "brier": round(brier_clamped, 6),
            "pnl_per_contract_cents": round(pnl_pc, 4),
            "quality": round(quality, 4),
            "profit": round(profit, 4),
            "composite": round(composite, 4),
            "rolling_sharpe": round(rolling_sharpe, 4) if rolling_sharpe is not None else None,
            "max_drawdown_pct": round(max_drawdown_pct, 4) if max_drawdown_pct is not None else None,
            "calibration_sample_size": calibration_sample_size,
        },
    }
