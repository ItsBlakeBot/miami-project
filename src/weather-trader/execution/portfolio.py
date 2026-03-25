"""Portfolio middleman between model outputs and execution.

Responsibilities:
- Consume settled-trade performance (per city/station)
- Convert performance into adaptive policy params (Brier + PnL/contract)
- Apply per-station concentration + re-entry pacing controls
- Provide a single recommendation interface for executors (paper/live)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from execution.trader_policy import (
    TradeRecommendation,
    TraderPolicy,
    adaptive_policy_params_from_performance,
)

UTC = timezone.utc


@dataclass
class SettledTradeRecord:
    station: str
    side: str
    winning_side: str
    estimated_probability: float
    realized_pnl_cents: float
    contracts: int
    settled_at_utc: datetime | None


@dataclass
class PortfolioProfile:
    station: str
    trader_policy: TraderPolicy
    max_open_trades_per_ticker: int
    min_minutes_between_entries: int
    adaptive_meta: dict


class PortfolioMiddleman:
    def __init__(
        self,
        *,
        base_bankroll_dollars: float = 250.0,
        adaptive_tuning: bool = True,
        adaptive_lookback_hours: int = 48,
        adaptive_min_settled_trades: int = 8,
        default_max_open_trades_per_ticker: int = 3,
        default_min_minutes_between_entries: int = 30,
    ):
        self.base_bankroll_dollars = float(base_bankroll_dollars)
        self.adaptive_tuning = bool(adaptive_tuning)
        self.adaptive_lookback_hours = max(12, int(adaptive_lookback_hours))
        self.adaptive_min_settled_trades = max(4, int(adaptive_min_settled_trades))
        self.default_max_open_trades_per_ticker = max(1, int(default_max_open_trades_per_ticker))
        self.default_min_minutes_between_entries = max(0, int(default_min_minutes_between_entries))

        self._profiles: dict[str, PortfolioProfile] = {}

    @staticmethod
    def parse_utc(ts: str | None) -> datetime | None:
        if not ts:
            return None
        for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ"):
            try:
                return datetime.strptime(ts, fmt).replace(tzinfo=UTC)
            except ValueError:
                continue
        return None

    def _recent_rows(
        self,
        rows: list[SettledTradeRecord],
        station: str,
        now_utc: datetime,
    ) -> list[SettledTradeRecord]:
        cutoff = now_utc - timedelta(hours=self.adaptive_lookback_hours)
        out: list[SettledTradeRecord] = []
        for r in rows:
            if r.station != station:
                continue
            if r.settled_at_utc is None or r.settled_at_utc >= cutoff:
                out.append(r)
        return out

    @staticmethod
    def _brier_and_pnl_per_contract(rows: list[SettledTradeRecord]) -> tuple[float | None, float | None]:
        if not rows:
            return None, None

        brier_vals: list[float] = []
        pnl_vals: list[float] = []
        for r in rows:
            y = 1.0 if r.side.lower() == r.winning_side.lower() else 0.0
            p = max(1e-6, min(1.0 - 1e-6, float(r.estimated_probability)))
            brier_vals.append((p - y) ** 2)

            contracts = max(1, int(r.contracts or 1))
            pnl_vals.append(float(r.realized_pnl_cents) / contracts)

        brier = sum(brier_vals) / len(brier_vals) if brier_vals else None
        pnl_pc = sum(pnl_vals) / len(pnl_vals) if pnl_vals else None
        return brier, pnl_pc

    def _build_profile(
        self,
        *,
        station: str,
        rows: list[SettledTradeRecord],
        now_utc: datetime,
    ) -> PortfolioProfile:
        if self.adaptive_tuning:
            recent = self._recent_rows(rows, station, now_utc)
            brier, pnl_pc = self._brier_and_pnl_per_contract(recent)
            effective_count = len(recent) if len(recent) >= self.adaptive_min_settled_trades else 0
            params = adaptive_policy_params_from_performance(
                trade_count=effective_count,
                brier=brier,
                pnl_per_contract_cents=pnl_pc,
                base_bankroll_dollars=self.base_bankroll_dollars,
            )
        else:
            recent = []
            params = {
                "bankroll_dollars": float(self.base_bankroll_dollars),
                "kelly_fraction": 0.16,
                "max_contracts_per_trade": 8,
                "min_edge_cents": 3.5,
                "min_ev_cents": 2.0,
                "max_market_disagreement": 0.30,
                "max_model_shift_from_market": 0.12,
                "adaptive": {
                    "mode": "disabled",
                    "trade_count": 0,
                },
            }

        trader_policy = TraderPolicy(
            bankroll_dollars=float(params["bankroll_dollars"]),
            kelly_fraction=float(params["kelly_fraction"]),
            max_contracts_per_trade=int(params["max_contracts_per_trade"]),
            min_edge_cents=float(params["min_edge_cents"]),
            min_ev_cents=float(params["min_ev_cents"]),
            max_market_disagreement=float(params["max_market_disagreement"]),
            max_model_shift_from_market=float(params["max_model_shift_from_market"]),
        )

        adaptive_meta = dict(params.get("adaptive") or {})
        adaptive_meta.update(
            {
                "lookback_hours": self.adaptive_lookback_hours,
                "min_settled_trades": self.adaptive_min_settled_trades,
                "raw_trade_count": len(recent),
            }
        )

        max_open = self.default_max_open_trades_per_ticker
        min_reentry = self.default_min_minutes_between_entries

        if adaptive_meta.get("mode") == "rolling":
            composite = float(adaptive_meta.get("composite") or 0.0)
            max_open = max(1, min(5, int(round(1 + 4 * composite))))
            min_reentry = max(10, int(round(120 - 90 * composite)))

        adaptive_meta["max_open_trades_per_ticker"] = max_open
        adaptive_meta["min_minutes_between_entries"] = min_reentry

        return PortfolioProfile(
            station=station,
            trader_policy=trader_policy,
            max_open_trades_per_ticker=max_open,
            min_minutes_between_entries=min_reentry,
            adaptive_meta=adaptive_meta,
        )

    def refresh(
        self,
        *,
        stations: list[str],
        settled_rows: list[SettledTradeRecord],
        now_utc: datetime | None = None,
    ) -> None:
        as_of = now_utc or datetime.now(UTC)
        for station in stations:
            self._profiles[station] = self._build_profile(
                station=station,
                rows=settled_rows,
                now_utc=as_of,
            )

    def profile_for(self, station: str) -> PortfolioProfile:
        profile = self._profiles.get(station)
        if profile is not None:
            return profile

        fallback = self._build_profile(
            station=station,
            rows=[],
            now_utc=datetime.now(UTC),
        )
        self._profiles[station] = fallback
        return fallback

    def recommend_trade(
        self,
        *,
        station: str,
        now_utc: datetime,
        open_count_for_ticker: int,
        last_entry_utc: datetime | None,
        prob_yes: float,
        regime_confidence: float,
        yes_ask_cents: float | None,
        yes_ask_qty: int,
        no_ask_cents: float | None,
        no_ask_qty: int,
    ) -> TradeRecommendation | None:
        profile = self.profile_for(station)

        if open_count_for_ticker >= profile.max_open_trades_per_ticker:
            return None

        if (
            last_entry_utc is not None
            and profile.min_minutes_between_entries > 0
            and (now_utc - last_entry_utc).total_seconds() / 60.0 < profile.min_minutes_between_entries
        ):
            return None

        return profile.trader_policy.recommend_trade(
            prob_yes=prob_yes,
            regime_confidence=regime_confidence,
            yes_ask_cents=yes_ask_cents,
            yes_ask_qty=yes_ask_qty,
            no_ask_cents=no_ask_cents,
            no_ask_qty=no_ask_qty,
        )
