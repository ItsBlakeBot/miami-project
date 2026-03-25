"""Ledger — trade logging, P/L tracking, and statistics.

Every signal snapshot and trade decision gets recorded here,
regardless of mode (shadow, paper, live).
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone

from src.execution.trader import TradeDecision

log = logging.getLogger(__name__)


class Ledger:
    """Tracks signal snapshots, trades, and outcomes in the trader DB."""

    def __init__(self, db: sqlite3.Connection):
        self._db = db

    def log_snapshot(self, snapshot_data: dict) -> int:
        """Log a signal snapshot. Returns the snapshot ID."""
        cols = [
            "timestamp_utc", "station", "target_date", "market_type",
            "hours_remaining", "consensus_f", "consensus_sigma",
            "raw_consensus_f", "n_models",
            "obs_current_f", "obs_trend_2hr", "obs_vs_consensus",
            "projected_extreme_f",
            "cape", "pw_mm", "outflow_risk", "cape_trend_1hr",
            "wind_dir_deg", "continental", "wind_shift",
            "dew_point_f", "evening_dew_mean_f", "estimated_floor_f", "dew_crash",
            "fawn_temp_f", "fawn_crash", "fawn_lead_minutes",
            "nearby_divergence_f", "nearby_crash_count",
            "pressure_hpa", "pressure_3hr_trend", "pressure_surge",
            "estimated_mu", "estimated_sigma",
            "active_flags", "adjustments",
            "running_high_f", "running_low_f",
        ]

        values = []
        for c in cols:
            v = snapshot_data.get(c)
            if isinstance(v, (list, dict)):
                v = json.dumps(v)
            elif isinstance(v, bool):
                v = int(v)
            values.append(v)

        placeholders = ", ".join(["?"] * len(cols))
        col_names = ", ".join(cols)

        cur = self._db.execute(
            f"INSERT INTO signal_snapshots ({col_names}) VALUES ({placeholders})",
            values,
        )
        self._db.commit()
        return cur.lastrowid

    def log_trade(
        self,
        decision: TradeDecision,
        snapshot_id: int | None,
        mode: str = "shadow",
        *,
        kalshi_order_id: str | None = None,
        order_status: str | None = None,
        fill_price_cents: int | None = None,
    ) -> int:
        """Log a trade decision. Returns the trade ID."""
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        if order_status is None:
            if mode == "shadow":
                order_status = "shadow"
            elif mode == "paper":
                order_status = "paper"
            else:
                order_status = "pending"

        cur = self._db.execute(
            """INSERT INTO trades (
                timestamp_utc, station, target_date, market_type,
                ticker, side, action, price_cents, contracts,
                our_probability, market_probability, edge, kelly_size,
                settlement_floor, settlement_ceil,
                signal_snapshot_id, mode, kalshi_order_id, order_status,
                fill_price_cents
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                now,
                "",  # station filled by caller
                "",  # target_date filled by caller
                decision.market_type,
                decision.ticker,
                decision.side,
                decision.action,
                decision.price_cents,
                decision.contracts,
                decision.our_probability,
                decision.market_probability,
                decision.edge,
                getattr(decision, 'kelly_f', getattr(decision, 'kelly_fraction', 0)),
                decision.settlement_floor,
                decision.settlement_ceil,
                snapshot_id,
                mode,
                kalshi_order_id,
                order_status,
                fill_price_cents,
            ),
        )
        self._db.commit()
        return cur.lastrowid

    def log_trade_with_context(
        self,
        decision: TradeDecision,
        snapshot_id: int | None,
        station: str,
        target_date: str,
        mode: str = "shadow",
        *,
        kalshi_order_id: str | None = None,
        order_status: str | None = None,
        fill_price_cents: int | None = None,
    ) -> int:
        """Log a trade decision with full context."""
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        if order_status is None:
            if mode == "shadow":
                order_status = "shadow"
            elif mode == "paper":
                order_status = "paper"
            else:
                order_status = "pending"

        cur = self._db.execute(
            """INSERT INTO trades (
                timestamp_utc, station, target_date, market_type,
                ticker, side, action, price_cents, contracts,
                our_probability, market_probability, edge, kelly_size,
                settlement_floor, settlement_ceil,
                signal_snapshot_id, mode, kalshi_order_id, order_status,
                fill_price_cents
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                now, station, target_date, decision.market_type,
                decision.ticker, decision.side, decision.action,
                decision.price_cents, decision.contracts,
                decision.our_probability, decision.market_probability,
                decision.edge, getattr(decision, 'kelly_f', getattr(decision, 'kelly_fraction', 0)),
                decision.settlement_floor, decision.settlement_ceil,
                snapshot_id, mode,
                kalshi_order_id, order_status, fill_price_cents,
            ),
        )
        self._db.commit()
        return cur.lastrowid

    def settle_trades(self, target_date: str, market_type: str,
                      cli_value_f: float) -> list[dict]:
        """Settle all trades for a date/market_type against CLI value.

        Determines winning side for each bracket and computes P/L.
        Returns list of outcome dicts.
        """
        trades = self._db.execute(
            """SELECT id, ticker, side, price_cents, contracts, market_type,
                      settlement_floor, settlement_ceil
               FROM trades
               WHERE target_date = ? AND market_type = ?
                 AND id NOT IN (SELECT trade_id FROM trade_outcomes)""",
            (target_date, market_type),
        ).fetchall()

        outcomes = []
        for t in trades:
            trade_id = t[0]
            ticker = t[1]
            side = t[2]
            price = t[3]
            contracts = t[4]
            settlement_floor = t[6]
            settlement_ceil = t[7]

            # Determine if bracket won YES.
            bracket_won_yes = self._did_bracket_win(
                ticker,
                cli_value_f,
                settlement_floor=settlement_floor,
                settlement_ceil=settlement_ceil,
            )

            if bracket_won_yes is None:
                log.warning("Could not determine winner for %s (cli=%.1f)", ticker, cli_value_f)
                continue

            winning_side = "yes" if bracket_won_yes else "no"
            our_side_won = (side == winning_side)

            if our_side_won:
                pnl_cents = 100 - price  # we bought at price, won $1
            else:
                pnl_cents = -price  # we lose what we paid

            total_pnl = pnl_cents * contracts

            self._db.execute(
                """INSERT INTO trade_outcomes (
                    trade_id, target_date, market_type, cli_value_f,
                    winning_side, our_side_won, pnl_cents, total_pnl_cents
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (trade_id, target_date, market_type, cli_value_f,
                 winning_side, int(our_side_won), pnl_cents, total_pnl),
            )

            outcomes.append({
                "trade_id": trade_id,
                "ticker": ticker,
                "side": side,
                "won": our_side_won,
                "pnl_cents": pnl_cents,
                "total_pnl_cents": total_pnl,
            })

        self._db.commit()
        return outcomes

    def _did_bracket_win(
        self,
        ticker: str,
        cli_value_f: float,
        *,
        settlement_floor: float | None = None,
        settlement_ceil: float | None = None,
    ) -> bool | None:
        """Determine if a bracket's YES side won given CLI settlement value.

        If settlement bounds are available, use them directly.
        Otherwise fall back to ticker parsing heuristics.
        """
        if settlement_floor is not None and settlement_ceil is not None:
            return settlement_floor <= cli_value_f < settlement_ceil

        cli_rounded = round(cli_value_f)
        upper = ticker.upper()

        # Parse: KXLOWTMIA-26MAR17-T57 or ...-B57.5
        import re
        match = re.search(r"-([TB])(\d+(?:\.\d+)?)$", upper)
        if not match:
            return None

        side_char = match.group(1)
        strike = float(match.group(2))

        if side_char == "T":
            # Legacy fallback when bounds are missing.
            if "LOW" in upper:
                return cli_rounded >= strike
            return cli_rounded < strike

        if side_char == "B":
            # B-bracket: 2°F window
            floor_val = int(strike - 0.5) if strike != int(strike) else int(strike)
            return cli_rounded in (floor_val, floor_val + 1)

        return None

    def get_stats(self, lookback_days: int = 30) -> dict:
        """Compute trading statistics over recent history."""
        rows = self._db.execute(
            """SELECT t.side, t.price_cents, t.contracts, t.mode,
                      o.our_side_won, o.pnl_cents, o.total_pnl_cents
               FROM trades t
               JOIN trade_outcomes o ON o.trade_id = t.id
               WHERE o.created_at >= datetime('now', ?)
               ORDER BY t.timestamp_utc ASC""",
            (f"-{lookback_days} days",),
        ).fetchall()

        if not rows:
            return {"trades": 0, "win_rate": 0, "total_pnl": 0, "max_drawdown": 0}

        total = len(rows)
        wins = sum(1 for r in rows if r[4])
        pnl_list = [r[6] for r in rows]
        total_pnl = sum(pnl_list)

        # Max drawdown
        peak = 0
        max_dd = 0
        running = 0
        for p in pnl_list:
            running += p
            peak = max(peak, running)
            dd = peak - running
            max_dd = max(max_dd, dd)

        return {
            "trades": total,
            "wins": wins,
            "win_rate": wins / total if total > 0 else 0,
            "total_pnl_cents": total_pnl,
            "total_pnl_dollars": total_pnl / 100.0,
            "max_drawdown_cents": max_dd,
            "avg_pnl_cents": total_pnl / total if total > 0 else 0,
        }
