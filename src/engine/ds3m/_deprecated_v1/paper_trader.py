"""DS3M shadow paper trader.

Mirrors the production paper trader's entry/exit logic but reads from
ds3m_estimates and writes to ds3m_paper_trades / ds3m_paper_trade_settlements.
Uses the SAME market prices and fee calculations as production for a fair
side-by-side comparison.

This module is intentionally simpler than the production paper_trader.py —
no PortfolioMiddleman adaptive tuning, no weather-trader cross-project
imports.  The goal is a clean EV-based comparison: given DS3M's probability
estimates, would its trades have made money?
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from math import ceil
from pathlib import Path

log = logging.getLogger(__name__)

UTC = timezone.utc


def _kalshi_fee_cents(price_cents: float) -> float:
    """Kalshi taker fee: ceil(0.07 * P * (1-P) * 100) cents."""
    p = max(0.01, min(99.0, price_cents)) / 100.0
    return ceil(0.07 * p * (1.0 - p) * 100.0)


@dataclass
class DS3MTradeCandidate:
    ticker: str
    station: str
    target_date: str
    market_type: str
    side: str  # "yes" or "no"
    entry_price_cents: float
    ds3m_probability: float
    conformal_probability: float | None
    edge_cents: float
    ev_cents: float
    regime_posterior: str | None  # JSON
    ess: float | None
    strike_label: str = ""


# ---------------------------------------------------------------------------
# Entry / exit logic (mirrors production PaperTradingPolicy)
# ---------------------------------------------------------------------------

MAX_OPEN_PER_TICKER = 2
MIN_MINUTES_BETWEEN = 30
MAX_QUOTE_AGE_MINUTES = 10


def _load_tuned_params() -> dict:
    """Load DS3M-specific auto-tuned exit params. Falls back to defaults if unavailable.

    DS3M has its OWN tuner state (ds3m_exit_tuner_state.json), separate from
    production (exit_tuner_state.json). They learn independently from their
    own trade data.
    """
    defaults = {
        "min_edge_cents": 4.0,
        "min_ev_cents": 2.0,
        "max_price_cents": 80.0,
        "min_price_cents": 3.0,
        "profit_take_risk_fraction": 0.6,
        "profit_take_time_decay": 0.3,
        "deterioration_buffer_cents": 3.0,
    }
    try:
        from engine.exit_tuner import ExitTunerState, DS3M_STATE_PATH
        state = ExitTunerState.load(DS3M_STATE_PATH)
        if state.current_params:
            defaults.update({k: v for k, v in state.current_params.items() if k in defaults})
    except Exception:
        pass
    return defaults


def _evaluate_entry(
    ds3m_prob: float,
    side: str,
    ask_price_cents: float,
    params: dict | None = None,
) -> tuple[float, float] | None:
    """Returns (edge_cents, ev_cents) or None if no entry."""
    tp = params or _load_tuned_params()
    p = max(0.0, min(1.0, ds3m_prob))
    fair = (p if side == "yes" else (1.0 - p)) * 100.0
    fee = _kalshi_fee_cents(ask_price_cents)
    edge = fair - ask_price_cents - fee
    if ask_price_cents > tp.get("max_price_cents", 80.0):
        return None
    if ask_price_cents < tp.get("min_price_cents", 3.0):
        return None
    if edge < tp.get("min_edge_cents", 4.0) or edge < tp.get("min_ev_cents", 2.0):
        return None
    return (round(edge, 2), round(edge, 2))


def _evaluate_exit(
    side: str,
    entry_price_cents: float,
    ds3m_prob: float,
    bid_price_cents: float | None,
    hours_to_settlement: float | None = None,
    params: dict | None = None,
) -> tuple[bool, str]:
    """Returns (should_exit, reason). Uses same tuned params as production."""
    tp = params or _load_tuned_params()
    p_win = ds3m_prob if side == "yes" else (1.0 - ds3m_prob)
    entry_fee = _kalshi_fee_cents(entry_price_cents)
    ev_hold = p_win * 100.0 - entry_price_cents - entry_fee

    if bid_price_cents is None:
        return False, "hold"

    maker_fee = _kalshi_fee_cents(bid_price_cents)  # DS3M uses taker for simplicity
    realized_profit = bid_price_cents - entry_price_cents - entry_fee - maker_fee

    # Rule 1: sell beats hold
    if realized_profit >= ev_hold:
        return True, "sell_beats_hold"

    # Rule 2: conviction + time profit-taking (same formula as production)
    if realized_profit > 0:
        risk_cents = max(1.0, (p_win * (1.0 - p_win)) ** 0.5 * 100.0)
        risk_frac = tp.get("profit_take_risk_fraction", 0.6)
        time_decay = tp.get("profit_take_time_decay", 0.3)
        time_factor = 1.0
        if hours_to_settlement is not None:
            time_factor = 1.0 + time_decay * hours_to_settlement / 24.0
        threshold = risk_cents * risk_frac / time_factor
        if realized_profit > threshold:
            return True, "profit_take"

    # Rule 3: negative EV cut
    buffer = tp.get("deterioration_buffer_cents", 3.0)
    if ev_hold <= -abs(buffer):
        return True, "negative_ev_cut"

    return False, "hold"


def _strike_label(ticker: str, market_type: str) -> str:
    mt = "H" if market_type == "high" else "L"
    for part in ticker.split("-"):
        if part.startswith("B") or part.startswith("T"):
            return f"{mt}/{part}"
    return f"{mt}/?"


def _hours_to_settlement(target_date: str | None, now_utc: datetime) -> float | None:
    if not target_date:
        return None
    try:
        end_day = datetime.strptime(target_date, "%Y-%m-%d") + timedelta(days=1)
        settlement_utc = end_day.replace(hour=5, minute=0, second=0, microsecond=0, tzinfo=UTC)
        return max(0.0, (settlement_utc - now_utc).total_seconds() / 3600.0)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Core: run one cycle of DS3M paper trading
# ---------------------------------------------------------------------------

def run_ds3m_paper_trading(
    db_path: str | Path,
    station: str = "KMIA",
    target_date: str | None = None,
) -> dict:
    """Evaluate DS3M estimates for paper trade entry/exit.

    Reads from ds3m_estimates + market_snapshots.
    Writes to ds3m_paper_trades.
    Returns summary dict.
    """
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    now_utc = datetime.now(UTC)
    now_str = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

    if target_date is None:
        hour_lst = (now_utc.hour - 5) % 24
        if hour_lst < 5:
            target_date = (now_utc - timedelta(days=1)).strftime("%Y-%m-%d")
        else:
            target_date = (now_utc + timedelta(hours=-5)).strftime("%Y-%m-%d")

    entries = 0
    exits = 0
    skipped = 0

    try:
        # 1. Process exits for open DS3M trades
        open_trades = conn.execute(
            """SELECT id, ticker, side, entry_price_cents, market_type, entry_time,
                      target_date, contracts
               FROM ds3m_paper_trades
               WHERE station = ? AND status = 'open'""",
            (station,),
        ).fetchall()

        for trade in open_trades:
            ticker = trade["ticker"]

            # Get latest DS3M probability
            est = conn.execute(
                """SELECT probability FROM ds3m_estimates
                   WHERE ticker = ? ORDER BY timestamp_utc DESC LIMIT 1""",
                (ticker,),
            ).fetchone()
            if est is None:
                continue

            # Get latest bid
            ms = conn.execute(
                """SELECT best_yes_bid_cents, best_no_bid_cents, snapshot_time
                   FROM market_snapshots
                   WHERE ticker = ? ORDER BY id DESC LIMIT 1""",
                (ticker,),
            ).fetchone()

            bid = None
            if ms is not None:
                if trade["side"] == "yes":
                    bid = ms["best_yes_bid_cents"]
                else:
                    bid = ms["best_no_bid_cents"]

            hours_to_settlement = _hours_to_settlement(trade["target_date"], now_utc)
            should_exit, reason = _evaluate_exit(
                trade["side"],
                trade["entry_price_cents"],
                est["probability"],
                bid,
                hours_to_settlement=hours_to_settlement,
            )

            contracts = max(1, int(trade["contracts"] or 1))
            p_win = est["probability"] if trade["side"] == "yes" else (1.0 - est["probability"])
            entry_fee = _kalshi_fee_cents(trade["entry_price_cents"])
            ev_hold = p_win * 100.0 - trade["entry_price_cents"] - entry_fee
            conn.execute(
                """INSERT INTO ds3m_paper_trade_marks
                   (trade_id, ticker, mark_time, mark_price_cents, estimated_probability, expected_value_cents, note)
                   VALUES (?,?,?,?,?,?,?)""",
                (trade["id"], ticker, now_str, bid, est["probability"], round(ev_hold * contracts, 2), reason),
            )

            if should_exit and bid is not None:
                exit_price = float(bid)
                exit_fee = _kalshi_fee_cents(exit_price)
                pnl = round((exit_price - trade["entry_price_cents"] - entry_fee - exit_fee) * contracts, 2)

                conn.execute(
                    """UPDATE ds3m_paper_trades
                       SET status='closed', exit_price_cents=?, exit_time=?,
                           exit_reason=?, realized_pnl_cents=?
                       WHERE id=?""",
                    (exit_price, now_str, reason, pnl, trade["id"]),
                )
                exits += 1

        # 2. Process entries from latest DS3M estimates
        estimates = conn.execute(
            """SELECT de.ticker, de.market_type, de.probability, de.conformal_probability,
                      de.regime_posterior, de.ess,
                      ms.best_yes_ask_cents, ms.best_no_ask_cents,
                      ms.yes_ask_qty, ms.no_ask_qty, ms.snapshot_time
               FROM ds3m_estimates de
               JOIN (
                   SELECT ticker, MAX(timestamp_utc) AS max_ts
                   FROM ds3m_estimates
                   WHERE station = ? AND target_date = ?
                   GROUP BY ticker
               ) latest ON latest.ticker = de.ticker AND latest.max_ts = de.timestamp_utc
               LEFT JOIN (
                   SELECT ticker, best_yes_ask_cents, best_no_ask_cents,
                          yes_ask_qty, no_ask_qty, snapshot_time,
                          ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY id DESC) AS rn
                   FROM market_snapshots
                   WHERE forecast_date = ?
               ) ms ON ms.ticker = de.ticker AND ms.rn = 1
               WHERE de.station = ? AND de.target_date = ?""",
            (station, target_date, target_date, station, target_date),
        ).fetchall()

        for est in estimates:
            ticker = est["ticker"]
            prob = est["probability"]
            mtype = est["market_type"]

            # Check existing open trades
            n_open = conn.execute(
                "SELECT COUNT(*) FROM ds3m_paper_trades WHERE ticker=? AND status='open'",
                (ticker,),
            ).fetchone()[0]
            if n_open >= MAX_OPEN_PER_TICKER:
                skipped += 1
                continue

            # Check cooldown
            last_entry = conn.execute(
                """SELECT entry_time FROM ds3m_paper_trades
                   WHERE ticker=? ORDER BY id DESC LIMIT 1""",
                (ticker,),
            ).fetchone()
            if last_entry and last_entry["entry_time"]:
                try:
                    last_dt = datetime.fromisoformat(last_entry["entry_time"].replace("Z", "+00:00"))
                    if (now_utc - last_dt).total_seconds() < MIN_MINUTES_BETWEEN * 60:
                        skipped += 1
                        continue
                except (ValueError, TypeError):
                    pass

            # Check quote staleness
            if est["snapshot_time"]:
                try:
                    snap_dt = datetime.fromisoformat(est["snapshot_time"].replace("Z", "+00:00"))
                    age_min = (now_utc - snap_dt).total_seconds() / 60.0
                    if age_min > MAX_QUOTE_AGE_MINUTES:
                        skipped += 1
                        continue
                except (ValueError, TypeError):
                    pass

            # Evaluate YES side
            yes_ask = est["best_yes_ask_cents"]
            if yes_ask is not None:
                result = _evaluate_entry(prob, "yes", yes_ask)
                if result is not None:
                    edge, ev = result
                    _insert_trade(
                        conn, ticker, station, target_date, mtype, "yes",
                        yes_ask, prob, est["conformal_probability"],
                        edge, ev, est["regime_posterior"], est["ess"],
                        _strike_label(ticker, mtype), now_str,
                    )
                    entries += 1
                    continue  # don't also enter NO on same ticker

            # Evaluate NO side
            no_ask = est["best_no_ask_cents"]
            if no_ask is not None:
                result = _evaluate_entry(1.0 - prob, "no", no_ask)
                if result is not None:
                    edge, ev = result
                    _insert_trade(
                        conn, ticker, station, target_date, mtype, "no",
                        no_ask, prob, est["conformal_probability"],
                        edge, ev, est["regime_posterior"], est["ess"],
                        _strike_label(ticker, mtype), now_str,
                    )
                    entries += 1

        conn.commit()
    finally:
        conn.close()

    log.info(
        "DS3M paper trader: %d entries, %d exits, %d skipped for %s",
        entries, exits, skipped, target_date,
    )
    return {"entries": entries, "exits": exits, "skipped": skipped, "target_date": target_date}


def _insert_trade(
    conn, ticker, station, target_date, market_type, side,
    entry_price, ds3m_prob, conformal_prob, edge, ev,
    regime_posterior, ess, strike_label, now_str,
):
    conn.execute(
        """INSERT INTO ds3m_paper_trades
           (ticker, station, target_date, market_type, side, entry_price_cents,
            entry_time, ds3m_probability, conformal_probability,
            expected_edge_cents, expected_value_cents,
            regime_posterior, ess, strike_label)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (ticker, station, target_date, market_type, side,
         entry_price, now_str, ds3m_prob, conformal_prob,
         edge, ev, regime_posterior, ess, strike_label),
    )


# ---------------------------------------------------------------------------
# Settlement: settle DS3M trades after CLI arrives
# ---------------------------------------------------------------------------

def settle_ds3m_trades(
    db_path: str | Path,
    station: str = "KMIA",
    target_date: str | None = None,
) -> dict:
    """Settle open DS3M paper trades using market_settlements.

    Called from the post-settlement pipeline after CLI arrives.
    """
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    now_str = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    settled = 0

    try:
        # Find open DS3M trades for the target date
        if target_date:
            trades = conn.execute(
                """SELECT id, ticker, side, entry_price_cents
                   FROM ds3m_paper_trades
                   WHERE station = ? AND target_date = ? AND status = 'open'""",
                (station, target_date),
            ).fetchall()
        else:
            trades = conn.execute(
                """SELECT id, ticker, side, entry_price_cents
                   FROM ds3m_paper_trades
                   WHERE station = ? AND status = 'open'""",
                (station,),
            ).fetchall()

        for trade in trades:
            ticker = trade["ticker"]

            # Look up market settlement
            ms = conn.execute(
                """SELECT winning_side, settlement_price_cents
                   FROM market_settlements
                   WHERE ticker = ? ORDER BY id DESC LIMIT 1""",
                (ticker,),
            ).fetchone()
            if ms is None:
                continue

            winning_side = ms["winning_side"]
            entry_fee = _kalshi_fee_cents(trade["entry_price_cents"])

            if trade["side"] == winning_side:
                # Won: payout = 100 - entry - fee
                pnl = round(100.0 - trade["entry_price_cents"] - entry_fee, 2)
            else:
                # Lost: lose entry + fee
                pnl = round(-trade["entry_price_cents"] - entry_fee, 2)

            # Update trade
            conn.execute(
                """UPDATE ds3m_paper_trades
                   SET status='settled', exit_reason='settlement',
                       realized_pnl_cents=?, exit_time=?
                   WHERE id=?""",
                (pnl, now_str, trade["id"]),
            )

            # Write settlement record
            conn.execute(
                """INSERT INTO ds3m_paper_trade_settlements
                   (trade_id, ticker, winning_side, settlement_price_cents,
                    settled_at, realized_pnl_cents)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (trade["id"], ticker, winning_side,
                 ms["settlement_price_cents"], now_str, pnl),
            )
            settled += 1

        conn.commit()
    finally:
        conn.close()

    log.info("DS3M settlements: %d trades settled for %s %s", settled, station, target_date or "all")
    return {"settled": settled}


# ---------------------------------------------------------------------------
# Comparison report: DS3M vs production paper trading
# ---------------------------------------------------------------------------

def compare_paper_trading(
    db_path: str | Path,
    station: str = "KMIA",
    lookback_days: int = 30,
) -> dict:
    """Side-by-side comparison of DS3M vs production paper trading P&L."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    try:
        # Production stats
        prod = conn.execute(
            """SELECT COUNT(*) as n,
                      SUM(CASE WHEN realized_pnl_cents > 0 THEN 1 ELSE 0 END) as wins,
                      ROUND(AVG(realized_pnl_cents), 1) as avg_pnl,
                      ROUND(SUM(realized_pnl_cents), 1) as total_pnl,
                      ROUND(AVG(entry_price_cents), 1) as avg_entry
               FROM paper_trades
               WHERE station = ? AND status != 'open'
                 AND realized_pnl_cents IS NOT NULL
                 AND created_at >= datetime('now', ?)""",
            (station, f"-{lookback_days} days"),
        ).fetchone()

        # DS3M stats
        ds3m = conn.execute(
            """SELECT COUNT(*) as n,
                      SUM(CASE WHEN realized_pnl_cents > 0 THEN 1 ELSE 0 END) as wins,
                      ROUND(AVG(realized_pnl_cents), 1) as avg_pnl,
                      ROUND(SUM(realized_pnl_cents), 1) as total_pnl,
                      ROUND(AVG(entry_price_cents), 1) as avg_entry
               FROM ds3m_paper_trades
               WHERE station = ? AND status != 'open'
                 AND realized_pnl_cents IS NOT NULL
                 AND created_at >= datetime('now', ?)""",
            (station, f"-{lookback_days} days"),
        ).fetchone()

        return {
            "production": {
                "n_trades": prod["n"] or 0,
                "wins": prod["wins"] or 0,
                "avg_pnl_cents": prod["avg_pnl"],
                "total_pnl_cents": prod["total_pnl"],
                "avg_entry_cents": prod["avg_entry"],
                "win_rate": round((prod["wins"] or 0) / max(1, prod["n"] or 1), 3),
            },
            "ds3m": {
                "n_trades": ds3m["n"] or 0,
                "wins": ds3m["wins"] or 0,
                "avg_pnl_cents": ds3m["avg_pnl"],
                "total_pnl_cents": ds3m["total_pnl"],
                "avg_entry_cents": ds3m["avg_entry"],
                "win_rate": round((ds3m["wins"] or 0) / max(1, ds3m["n"] or 1), 3),
            },
            "lookback_days": lookback_days,
        }
    finally:
        conn.close()
