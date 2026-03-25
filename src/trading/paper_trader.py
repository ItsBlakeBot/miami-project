from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from engine.replay_context import climate_target_date, is_stale, parse_utc
from pathlib import Path

from engine.edge_detector import kalshi_fee_cents

_WEATHER_TRADER_SRC = Path(__file__).resolve().parents[3] / "weather-trader" / "src"
if str(_WEATHER_TRADER_SRC) not in sys.path:
    sys.path.append(str(_WEATHER_TRADER_SRC))

from execution.portfolio import PortfolioMiddleman, SettledTradeRecord

log = logging.getLogger(__name__)


@dataclass
class EntryCandidate:
    ticker: str
    market_type: str
    target_date: str
    side: str
    ask_price_cents: float
    ask_qty: int
    probability: float
    edge_cents: float
    expected_value_cents: float


@dataclass
class ExitDecision:
    should_exit: bool
    reason: str
    fair_value_cents: float
    expected_value_cents: float


class PaperTradingPolicy:
    def __init__(
        self,
        min_edge_cents: float = 4.0,
        min_ev_cents: float = 2.0,
        max_price_cents: float = 80.0,
        min_price_cents: float = 3.0,
        deterioration_buffer_cents: float = 3.0,
        min_regime_confidence: float = 0.0,
        # Profit-taking: take profit when realized gain covers this fraction of remaining risk.
        # Higher = more patient (hold winners longer). Lower = greedier (take profits sooner).
        profit_take_risk_fraction: float = 0.6,
        # Time decay: as settlement approaches, the risk fraction requirement shrinks,
        # making the system MORE willing to hold. Far from settlement, it scales UP,
        # making it LESS willing to hold uncertain positions overnight.
        # Formula: effective_fraction = base_fraction * (1 + time_decay * hours_remaining / 24)
        profit_take_time_decay: float = 0.3,
    ):
        self.min_edge_cents = min_edge_cents
        self.min_ev_cents = min_ev_cents
        self.max_price_cents = max_price_cents
        self.min_price_cents = min_price_cents
        self.deterioration_buffer_cents = deterioration_buffer_cents
        self.min_regime_confidence = min_regime_confidence
        self.profit_take_risk_fraction = profit_take_risk_fraction
        self.profit_take_time_decay = profit_take_time_decay

    @staticmethod
    def fair_value_cents(probability: float, side: str) -> float:
        p = max(0.0, min(1.0, probability))
        return round((p if side == "yes" else (1.0 - p)) * 100.0, 2)

    def evaluate_entry(self, *, probability: float, side: str, ask_price_cents: float | None, ask_qty: int | None, regime_confidence: float | None = None) -> EntryCandidate | None:
        if ask_price_cents is None or ask_qty is None or ask_qty <= 0:
            return None
        if ask_price_cents > self.max_price_cents:
            return None
        if ask_price_cents < self.min_price_cents:
            return None  # sub-3¢ bids have terrible risk/reward and high relative fees
        if regime_confidence is not None and regime_confidence < self.min_regime_confidence:
            return None  # model too uncertain to trade
        fair = self.fair_value_cents(probability, side)
        fee = kalshi_fee_cents(ask_price_cents)
        edge = round(fair - ask_price_cents - fee, 2)
        ev = edge
        if edge < self.min_edge_cents or ev < self.min_ev_cents:
            return None
        return EntryCandidate(
            ticker="",
            market_type="",
            target_date="",
            side=side,
            ask_price_cents=ask_price_cents,
            ask_qty=ask_qty,
            probability=probability,
            edge_cents=edge,
            expected_value_cents=ev,
        )

    def evaluate_exit(self, *, side: str, entry_price_cents: float, probability: float, bid_price_cents: float | None, hours_to_settlement: float | None = None) -> ExitDecision:
        entry_fee = kalshi_fee_cents(entry_price_cents)
        p_win = probability if side == "yes" else (1.0 - probability)

        # EV of holding to settlement: binary payout, no exit fee at settlement.
        ev_hold = round(p_win * 100.0 - entry_price_cents - entry_fee, 2)

        # EV of selling now via limit order (maker fee) or market order (taker fee).
        # Compute both — use maker fee as the optimistic case since we should
        # always try limit orders first, falling back to taker if urgent.
        ev_sell = None
        realized_profit = None
        if bid_price_cents is not None:
            maker_fee = kalshi_fee_cents(bid_price_cents, is_maker=True)
            taker_fee = kalshi_fee_cents(bid_price_cents, is_maker=False)
            # Best case: limit order fills at maker fee
            realized_profit_maker = round(bid_price_cents - entry_price_cents - entry_fee - maker_fee, 2)
            # Worst case: cross the spread at taker fee
            realized_profit_taker = round(bid_price_cents - entry_price_cents - entry_fee - taker_fee, 2)
            # Use maker for profit-taking decisions (we have time to post limit),
            # but use taker for urgent exits (negative EV cuts).
            realized_profit = realized_profit_maker
            ev_sell = realized_profit

        should_exit = False
        reason = "hold"

        # Rule 1: Sell beats hold — pure EV comparison.
        if ev_sell is not None and ev_sell >= ev_hold:
            should_exit = True
            reason = "sell_beats_hold"

        # Rule 2: Conviction + time aware profit-taking.
        #
        # The profit threshold scales with TWO factors:
        #   a) Conviction: high p_win = higher threshold (hold for big payout)
        #   b) Time: more hours to settlement = LOWER threshold (take profit sooner,
        #      because overnight/multi-hour holds carry drift risk the model can't price)
        #
        # Formula:
        #   risk = sqrt(p_win * (1-p_win)) * 100   (binary outcome uncertainty)
        #   time_factor = 1 + time_decay * hours_remaining / 24
        #   threshold = risk * base_fraction / time_factor
        #
        # Examples (base_fraction=0.6, time_decay=0.3):
        #   p=0.9, 2h left:  risk=30, time=1.025, threshold=17.6¢ → hold 10¢ gain
        #   p=0.9, 18h left: risk=30, time=1.225, threshold=14.7¢ → hold 10¢ gain
        #   p=0.5, 2h left:  risk=50, time=1.025, threshold=29.3¢ → hold 20¢ gain
        #   p=0.5, 18h left: risk=50, time=1.225, threshold=24.5¢ → take 25¢ gain
        #
        # The time factor LOWERS the threshold far from settlement, making us
        # more willing to take profits when there's lots of time for things to go wrong.
        elif ev_sell is not None and realized_profit is not None and realized_profit > 0:
            risk_cents = max(1.0, (p_win * (1.0 - p_win)) ** 0.5 * 100.0)

            time_factor = 1.0
            if hours_to_settlement is not None:
                time_factor = 1.0 + self.profit_take_time_decay * hours_to_settlement / 24.0

            profit_threshold = risk_cents * self.profit_take_risk_fraction / time_factor
            if realized_profit > profit_threshold:
                should_exit = True
                reason = "profit_take"

        # Rule 3: Negative EV cut — use TAKER fee for urgency (crossing the spread).
        elif bid_price_cents is not None and ev_hold <= -abs(self.deterioration_buffer_cents):
            should_exit = True
            reason = "negative_ev_cut"

        fair = round(p_win * 100.0, 2)
        return ExitDecision(should_exit=should_exit, reason=reason, fair_value_cents=fair, expected_value_cents=ev_hold)


def _strike_label(row) -> str:
    """Derive a human-readable strike label like 'H/B77.5' or 'L/T73 under'."""
    mt = "H" if row["market_type"] == "high" else "L"
    ticker = row["ticker"]
    # Extract strike from ticker: ...B77.5 or ...T73
    for part in ticker.split("-"):
        if part.startswith("B") or part.startswith("T"):
            direction = ""
            if part.startswith("T"):
                sf = row["settlement_floor"]
                if sf is not None and sf < -10:
                    direction = " under"
                else:
                    direction = " over"
            return f"{mt}/{part}{direction}"
    return f"{mt}/?"


class PaperTrader:
    def __init__(
        self,
        db_path: str | Path,
        station: str = "KMIA",
        policy: PaperTradingPolicy | None = None,
        max_open_trades_per_ticker: int = 3,
        min_minutes_between_entries: int = 30,
        adaptive_tuning: bool = True,
        adaptive_lookback_hours: int = 48,
        adaptive_min_settled_trades: int = 8,
        max_quote_age_minutes: int = 10,
    ):
        self.db_path = str(db_path)
        self.station = station
        self.policy = policy or PaperTradingPolicy()
        self.portfolio = PortfolioMiddleman(
            adaptive_tuning=adaptive_tuning,
            adaptive_lookback_hours=adaptive_lookback_hours,
            adaptive_min_settled_trades=adaptive_min_settled_trades,
            default_max_open_trades_per_ticker=max_open_trades_per_ticker,
            default_min_minutes_between_entries=min_minutes_between_entries,
        )
        self.max_quote_age_minutes = max(1, int(max_quote_age_minutes))
        self._last_adaptive_meta: dict | None = None

    @staticmethod
    def _parse_utc(ts: str | None) -> datetime | None:
        return parse_utc(ts)

    @staticmethod
    def _current_target_date(now_utc: datetime | None = None, utc_hour_start: int = 5) -> str:
        now = now_utc or datetime.now(timezone.utc)
        return climate_target_date(now, boundary_hour_utc=utc_hour_start)

    @staticmethod
    def _climate_day_bounds(target_date: str) -> tuple[str, str]:
        # KMIA climate day uses fixed 05:00Z boundary.
        start = f"{target_date}T05:00:00Z"
        from datetime import datetime, timedelta
        next_day = (datetime.strptime(target_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
        end = f"{next_day}T05:00:00Z"
        return start, end

    def _is_stale_snapshot(self, snapshot_time: str | None, now_utc: datetime) -> bool:
        return is_stale(now_utc, snapshot_time, max_age_minutes=self.max_quote_age_minutes)

    def _running_extremes_nws(self, conn: sqlite3.Connection, target_date: str) -> tuple[float | None, float | None]:
        start, end = self._climate_day_bounds(target_date)
        row = conn.execute(
            """SELECT MAX(wethr_high_nws_f), MIN(wethr_low_nws_f)
               FROM observations
               WHERE station = ? AND timestamp_utc >= ? AND timestamp_utc < ?""",
            (self.station, start, end),
        ).fetchone()
        if not row:
            return None, None
        return row[0], row[1]

    @staticmethod
    def _is_confirmed_loser(
        *,
        market_type: str,
        side: str,
        settlement_floor: float | None,
        settlement_ceil: float | None,
        running_high_nws_f: float | None,
        running_low_nws_f: float | None,
    ) -> tuple[bool, str | None]:
        if settlement_floor is None or settlement_ceil is None:
            return False, None

        yes_busted = False
        if market_type == "high" and running_high_nws_f is not None:
            # High only goes up intraday; once running high breaches bracket ceiling,
            # YES on this bracket can no longer win.
            yes_busted = running_high_nws_f >= settlement_ceil
        elif market_type == "low" and running_low_nws_f is not None:
            # Low only goes down intraday; once running low falls below bracket floor,
            # YES on this bracket can no longer win.
            yes_busted = running_low_nws_f < settlement_floor

        if side == "yes" and yes_busted:
            return True, "confirmed_loser_bracket_busted"

        return False, None

    def _load_settled_trade_records(self, conn: sqlite3.Connection) -> list[SettledTradeRecord]:
        rows = conn.execute(
            """SELECT pt.station,
                      pt.side,
                      pts.winning_side,
                      pt.estimated_probability,
                      pts.realized_pnl_cents,
                      pt.contracts,
                      COALESCE(pts.settled_at, pts.created_at) AS settled_at
               FROM paper_trades pt
               JOIN paper_trade_settlements pts ON pts.trade_id = pt.id
               WHERE pt.station = ?
                 AND pt.estimated_probability IS NOT NULL
                 AND pt.contracts IS NOT NULL
                 AND pt.contracts > 0""",
            (self.station,),
        ).fetchall()

        out: list[SettledTradeRecord] = []
        for row in rows:
            out.append(
                SettledTradeRecord(
                    station=str(row["station"]),
                    side=str(row["side"]),
                    winning_side=str(row["winning_side"]),
                    estimated_probability=float(row["estimated_probability"]),
                    realized_pnl_cents=float(row["realized_pnl_cents"] or 0.0),
                    contracts=max(1, int(row["contracts"] or 1)),
                    settled_at_utc=self._parse_utc(row["settled_at"]),
                )
            )
        return out

    def _refresh_portfolio(self, conn: sqlite3.Connection, now_utc: datetime) -> None:
        settled = self._load_settled_trade_records(conn)
        self.portfolio.refresh(
            stations=[self.station],
            settled_rows=settled,
            now_utc=now_utc,
        )
        profile = self.portfolio.profile_for(self.station)
        self._last_adaptive_meta = dict(profile.adaptive_meta)

    def run_cycle(self) -> dict:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        now_utc = datetime.now(timezone.utc)
        try:
            self._resolve_settlements(conn)
            self._refresh_portfolio(conn, now_utc)

            # Apply auto-tuned exit params from the exit tuner (if available).
            # This reads the latest EMA-blended params from analysis_data/exit_tuner_state.json
            # and overwrites the policy defaults. The tuner runs daily post-settlement.
            try:
                from engine.exit_tuner import apply_tuned_params
                applied = apply_tuned_params(self.policy)
                log.debug("Exit tuner params applied: %s", applied)
            except Exception:
                pass  # tuner not yet initialized — use defaults

            entries = self._scan_entries(conn)
            exits = self._scan_exits(conn)
            conn.commit()
            return {
                "entries": entries,
                "exits": exits,
                "adaptive": self._last_adaptive_meta,
            }
        finally:
            conn.close()

    def _latest_estimates(self, conn: sqlite3.Connection, target_date: str) -> list[sqlite3.Row]:
        return conn.execute(
            """
            SELECT be.*,
                   ab.settlement_floor, ab.settlement_ceil,
                   ab.floor_strike, ab.cap_strike,
                   ms.snapshot_time,
                   ms.best_yes_bid_cents, ms.best_yes_ask_cents,
                   ms.best_no_bid_cents, ms.best_no_ask_cents,
                   COALESCE(ms.yes_ask_qty, 0) AS yes_ask_qty,
                   COALESCE(ms.no_ask_qty, 0) AS no_ask_qty
            FROM bracket_estimates be
            JOIN (
                SELECT ticker, MAX(timestamp_utc) AS max_ts
                FROM bracket_estimates
                WHERE station = ? AND target_date = ?
                GROUP BY ticker
            ) latest ON latest.ticker = be.ticker AND latest.max_ts = be.timestamp_utc
            JOIN active_brackets ab ON ab.ticker = be.ticker
            LEFT JOIN market_snapshots ms ON ms.id = (
                SELECT id FROM market_snapshots ms2
                WHERE ms2.ticker = be.ticker AND ms2.forecast_date = be.target_date
                ORDER BY ms2.id DESC LIMIT 1
            )
            WHERE be.station = ?
              AND be.target_date = ?
            """,
            (self.station, target_date, self.station, target_date),
        ).fetchall()

    def _scan_entries(self, conn: sqlite3.Connection) -> int:
        # Entry selection/sizing lives in weather-trader portfolio middleman so
        # paper/live execution can share one adaptive policy source.
        count = 0

        now_utc = datetime.now(timezone.utc)
        target_date = self._current_target_date(now_utc)

        profile = self.portfolio.profile_for(self.station)

        for row in self._latest_estimates(conn, target_date):
            row = dict(row)
            open_count = conn.execute(
                "SELECT COUNT(*) FROM paper_trades WHERE ticker=? AND status='open'",
                (row["ticker"],),
            ).fetchone()[0]

            last_entry_row = conn.execute(
                "SELECT entry_time FROM paper_trades WHERE ticker=? ORDER BY id DESC LIMIT 1",
                (row["ticker"],),
            ).fetchone()
            last_entry_dt = self._parse_utc(last_entry_row["entry_time"]) if last_entry_row else None

            if self._is_stale_snapshot(row.get("snapshot_time"), now_utc):
                continue

            prob_yes = float(row["probability"])
            regime_confidence = float(row["regime_confidence"] if row["regime_confidence"] is not None else 0.5)

            rec = self.portfolio.recommend_trade(
                station=self.station,
                now_utc=now_utc,
                open_count_for_ticker=int(open_count or 0),
                last_entry_utc=last_entry_dt,
                prob_yes=prob_yes,
                regime_confidence=regime_confidence,
                yes_ask_cents=row["best_yes_ask_cents"],
                yes_ask_qty=int(row["yes_ask_qty"] or 0),
                no_ask_cents=row["best_no_ask_cents"],
                no_ask_qty=int(row["no_ask_qty"] or 0),
            )
            if rec is None:
                continue

            label = _strike_label(row)
            conn.execute(
                """INSERT INTO paper_trades
                (ticker, station, target_date, market_type, side, contracts, entry_price_cents, entry_time, estimated_probability, expected_edge_cents, expected_value_cents, thesis_json, strike_label, status)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?, 'open')""",
                (
                    row["ticker"],
                    self.station,
                    row["target_date"],
                    row["market_type"],
                    rec.side,
                    rec.contracts,
                    rec.ask_price_cents,
                    now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    rec.probability,
                    rec.edge_cents,
                    rec.expected_value_cents * rec.contracts,
                    json.dumps(
                        {
                            "source": "weather_trader_portfolio_middleman_v1",
                            "regime_confidence": regime_confidence,
                            "portfolio": profile.adaptive_meta,
                            "sizing": rec.sizing,
                        }
                    ),
                    label,
                ),
            )
            count += 1

        return count

    def _scan_exits(self, conn: sqlite3.Connection) -> int:
        count = 0
        now_utc = datetime.now(timezone.utc)
        rows = conn.execute(
            """
            SELECT pt.*, be.probability,
                   ms.snapshot_time,
                   ms.best_yes_bid_cents, ms.best_no_bid_cents,
                   ab.settlement_floor, ab.settlement_ceil
            FROM paper_trades pt
            LEFT JOIN (
                SELECT be1.* FROM bracket_estimates be1
                JOIN (
                    SELECT ticker, MAX(timestamp_utc) AS max_ts FROM bracket_estimates GROUP BY ticker
                ) latest ON latest.ticker = be1.ticker AND latest.max_ts = be1.timestamp_utc
            ) be ON be.ticker = pt.ticker
            LEFT JOIN market_snapshots ms ON ms.id = (
                SELECT id FROM market_snapshots ms2
                WHERE ms2.ticker = pt.ticker AND ms2.forecast_date = pt.target_date
                ORDER BY ms2.id DESC LIMIT 1
            )
            LEFT JOIN active_brackets ab ON ab.ticker = pt.ticker
            WHERE pt.status='open'
            """
        ).fetchall()
        running_cache: dict[str, tuple[float | None, float | None]] = {}

        for row in rows:
            row = dict(row)
            bid = row["best_yes_bid_cents"] if row["side"] == "yes" else row["best_no_bid_cents"]

            if row["target_date"] not in running_cache:
                running_cache[row["target_date"]] = self._running_extremes_nws(conn, row["target_date"])
            running_high_nws_f, running_low_nws_f = running_cache[row["target_date"]]

            confirmed_loser, loser_reason = self._is_confirmed_loser(
                market_type=row["market_type"],
                side=row["side"],
                settlement_floor=row["settlement_floor"],
                settlement_ceil=row["settlement_ceil"],
                running_high_nws_f=running_high_nws_f,
                running_low_nws_f=running_low_nws_f,
            )

            prob = float(row["probability"]) if row["probability"] is not None else 0.0

            contracts = max(1, int(row["contracts"] or 1))

            if confirmed_loser:
                # Forced risk-off: bracket is mathematically busted for our side.
                # Close immediately to free capital / avoid stale open-loser state.
                forced_exit_price = float(bid) if bid is not None else 0.0
                entry_fee = kalshi_fee_cents(float(row["entry_price_cents"]))
                exit_fee = kalshi_fee_cents(float(forced_exit_price))
                pnl_per_contract = forced_exit_price - float(row["entry_price_cents"]) - entry_fee - exit_fee
                pnl = round(pnl_per_contract * contracts, 2)
                reason = loser_reason if bid is not None else f"{loser_reason}_no_bid_force_zero"
                conn.execute(
                    "INSERT INTO paper_trade_marks (trade_id, ticker, mark_time, mark_price_cents, estimated_probability, expected_value_cents, note) VALUES (?,?,?,?,?,?,?)",
                    (row["id"], row["ticker"], now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"), forced_exit_price, prob, pnl, reason),
                )
                conn.execute(
                    "UPDATE paper_trades SET status='closed', exit_price_cents=?, exit_time=?, exit_reason=?, realized_pnl_cents=? WHERE id=?",
                    (forced_exit_price, now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"), reason, pnl, row["id"]),
                )
                count += 1
                continue

            if self._is_stale_snapshot(row.get("snapshot_time"), now_utc):
                # Zero tolerance for stale data. Don't trade on it — no entries, no
                # exits, no decisions. Stale quotes are unreliable and any action
                # taken on them is a coin flip. Skip entirely and wait for fresh data.
                continue

            if row["probability"] is None:
                continue

            # Compute hours to settlement for profit-taking logic
            hours_to_settlement = None
            try:
                _, end_utc_str = self._climate_day_bounds(row["target_date"])
                end_utc = datetime.strptime(end_utc_str.replace("Z", ""), "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
                hours_to_settlement = max(0.0, (end_utc - now_utc).total_seconds() / 3600.0)
            except (ValueError, TypeError):
                pass

            decision = self.policy.evaluate_exit(
                side=row["side"],
                entry_price_cents=float(row["entry_price_cents"]),
                probability=prob,
                bid_price_cents=bid,
                hours_to_settlement=hours_to_settlement,
            )
            conn.execute(
                "INSERT INTO paper_trade_marks (trade_id, ticker, mark_time, mark_price_cents, estimated_probability, expected_value_cents, note) VALUES (?,?,?,?,?,?,?)",
                (
                    row["id"],
                    row["ticker"],
                    now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    bid,
                    prob,
                    decision.expected_value_cents * contracts,
                    decision.reason,
                ),
            )
            if decision.should_exit:
                if bid is None:
                    # Want to exit but no liquidity — log it
                    log.warning(
                        "EXIT BLOCKED (no bid): %s %s | EV(hold)=%.1f¢ | fair=%.1f¢",
                        row["ticker"], row["side"], decision.expected_value_cents, decision.fair_value_cents,
                    )
                    conn.execute(
                        "INSERT INTO paper_trade_marks (trade_id, ticker, mark_time, mark_price_cents, estimated_probability, expected_value_cents, note) VALUES (?,?,?,?,?,?,?)",
                        (
                            row["id"],
                            row["ticker"],
                            now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
                            None,
                            prob,
                            decision.expected_value_cents * contracts,
                            "exit_blocked_no_liquidity",
                        ),
                    )
                else:
                    entry_fee = kalshi_fee_cents(float(row["entry_price_cents"]))
                    exit_fee = kalshi_fee_cents(float(bid))
                    pnl_per_contract = float(bid) - float(row["entry_price_cents"]) - entry_fee - exit_fee
                    pnl = round(pnl_per_contract * contracts, 2)
                    log.info(
                        "EXIT: %s %s x%d | sell@%.0f¢ | pnl=%.1f¢ | reason=%s",
                        row["ticker"], row["side"], contracts, bid, pnl, decision.reason,
                    )
                    conn.execute("UPDATE paper_trades SET status='closed', exit_price_cents=?, exit_time=?, exit_reason=?, realized_pnl_cents=? WHERE id=?", (bid, now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"), decision.reason, pnl, row["id"]))
                    count += 1
        return count

    def _resolve_settlements(self, conn: sqlite3.Connection) -> int:
        count = 0
        rows = conn.execute(
            """
            SELECT pt.*, ms.winning_side, ms.settled_at
            FROM paper_trades pt
            JOIN market_settlements ms ON ms.ticker = pt.ticker
            WHERE pt.status='open'
            """
        ).fetchall()
        for row in rows:
            contracts = max(1, int(row["contracts"] or 1))
            settlement_price = 100.0 if str(row["side"]).lower() == str(row["winning_side"]).lower() else 0.0
            entry_fee = kalshi_fee_cents(float(row["entry_price_cents"]))
            pnl_per_contract = settlement_price - float(row["entry_price_cents"]) - entry_fee
            pnl = round(pnl_per_contract * contracts, 2)
            conn.execute("UPDATE paper_trades SET status='settled', exit_price_cents=?, exit_time=?, exit_reason='settlement', realized_pnl_cents=? WHERE id=?", (settlement_price, row["settled_at"], pnl, row["id"]))
            conn.execute("INSERT OR REPLACE INTO paper_trade_settlements (trade_id, ticker, winning_side, settlement_price_cents, settled_at, realized_pnl_cents) VALUES (?,?,?,?,?,?)", (row["id"], row["ticker"], row["winning_side"], settlement_price, row["settled_at"], pnl))
            count += 1
        return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper trading bot")
    parser.add_argument("--db", required=True)
    parser.add_argument("--station", default="KMIA")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval-seconds", type=int, default=300)
    parser.add_argument("--max-open-trades-per-ticker", type=int, default=3)
    parser.add_argument("--min-minutes-between-entries", type=int, default=30)
    parser.add_argument("--adaptive", action="store_true", default=True)
    parser.add_argument("--no-adaptive", action="store_true")
    parser.add_argument("--adaptive-lookback-hours", type=int, default=48)
    parser.add_argument("--adaptive-min-settled-trades", type=int, default=8)
    parser.add_argument("--max-quote-age-minutes", type=int, default=10)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-5s %(name)s: %(message)s")
    trader = PaperTrader(
        args.db,
        station=args.station,
        max_open_trades_per_ticker=args.max_open_trades_per_ticker,
        min_minutes_between_entries=args.min_minutes_between_entries,
        adaptive_tuning=(args.adaptive and not args.no_adaptive),
        adaptive_lookback_hours=args.adaptive_lookback_hours,
        adaptive_min_settled_trades=args.adaptive_min_settled_trades,
        max_quote_age_minutes=args.max_quote_age_minutes,
    )
    if args.watch:
        while True:
            print(json.dumps(trader.run_cycle(), indent=2))
            time.sleep(args.interval_seconds)
    else:
        print(json.dumps(trader.run_cycle(), indent=2))


if __name__ == "__main__":
    main()
