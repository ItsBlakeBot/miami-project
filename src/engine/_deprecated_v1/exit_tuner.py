"""Exit policy auto-tuner.

Learns optimal exit thresholds from historical trade mark data by analyzing
what WOULD have happened under different parameter settings. Runs after each
settlement day, evaluates counterfactual P&L under a grid of parameter
combinations, and recommends the best settings.

The tuner answers: "Given the trades we took, what exit parameters would
have maximized P&L?" It then smoothly adjusts live parameters toward the
optimum via EMA blending (no abrupt jumps).

Used by both the production paper trader and DS3M shadow paper trader.

Tunable parameters:
  - profit_take_risk_fraction: how much profit (relative to risk) triggers exit
  - profit_take_time_decay: how time-to-settlement scales the profit threshold
  - deterioration_buffer_cents: how deep into negative EV before cutting
  - min_edge_cents: minimum edge required for entry
  - min_ev_cents: minimum expected value for entry
  - min_price_cents: minimum ask price for entry
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

STATE_PATH = Path("analysis_data/exit_tuner_state.json")
DS3M_STATE_PATH = Path("analysis_data/ds3m_exit_tuner_state.json")


@dataclass
class TunerParams:
    """A single parameter set to evaluate."""
    profit_take_risk_fraction: float = 0.6
    profit_take_time_decay: float = 0.3
    deterioration_buffer_cents: float = 3.0
    min_edge_cents: float = 4.0
    min_ev_cents: float = 2.0
    min_price_cents: float = 3.0


@dataclass
class TunerResult:
    """Result of evaluating one parameter set against historical data."""
    params: TunerParams
    total_pnl_cents: float = 0.0
    n_trades: int = 0
    n_wins: int = 0
    avg_pnl_cents: float = 0.0
    win_rate: float = 0.0
    sharpe_ratio: float = 0.0  # PnL mean / PnL std (risk-adjusted)
    n_exits_profit_take: int = 0
    n_exits_sell_beats_hold: int = 0
    n_exits_negative_ev: int = 0
    n_held_to_settlement: int = 0


@dataclass
class ExitTunerState:
    """Persistent state for the exit tuner."""
    # Current recommended params (blended via EMA from best historical)
    current_params: dict = field(default_factory=lambda: asdict(TunerParams()))
    # History of evaluations
    evaluation_history: list[dict] = field(default_factory=list)
    # How fast to move toward new optimum (0 = never change, 1 = instant)
    ema_alpha: float = 0.15
    n_evaluations: int = 0
    last_evaluation_utc: str | None = None

    def save(self, path: Path | None = None) -> None:
        p = path or STATE_PATH
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path | None = None) -> ExitTunerState:
        p = path or STATE_PATH
        if not p.exists():
            return cls()
        try:
            data = json.loads(p.read_text())
            state = cls()
            state.current_params = data.get("current_params", asdict(TunerParams()))
            state.evaluation_history = data.get("evaluation_history", [])
            state.ema_alpha = data.get("ema_alpha", 0.15)
            state.n_evaluations = data.get("n_evaluations", 0)
            state.last_evaluation_utc = data.get("last_evaluation_utc")
            return state
        except (json.JSONDecodeError, KeyError):
            return cls()


# ---------------------------------------------------------------------------
# Counterfactual simulation: replay marks under different exit params
# ---------------------------------------------------------------------------

def _kalshi_fee_cents(price_cents: float, is_maker: bool = False) -> float:
    """Kalshi fee calculation."""
    p = max(0.0, min(1.0, price_cents / 100.0))
    rate = 0.0175 if is_maker else 0.07
    return math.ceil(rate * p * (1.0 - p) * 100.0)


def _simulate_exit_decision(
    params: TunerParams,
    entry_price_cents: float,
    side: str,
    probability: float,
    bid_price_cents: float | None,
    hours_to_settlement: float | None,
    liquidity_adjusted_bid: float | None = None,
) -> tuple[bool, str]:
    """Use liquidity-adjusted bid if available, otherwise fall back to raw bid."""
    effective_bid = liquidity_adjusted_bid if liquidity_adjusted_bid is not None else bid_price_cents
    """Simulate one exit decision under given params. Returns (should_exit, reason)."""
    entry_fee = _kalshi_fee_cents(entry_price_cents)
    p_win = probability if side == "yes" else (1.0 - probability)
    ev_hold = p_win * 100.0 - entry_price_cents - entry_fee

    if bid_price_cents is None:
        return False, "hold"

    maker_fee = _kalshi_fee_cents(bid_price_cents, is_maker=True)
    realized_profit = bid_price_cents - entry_price_cents - entry_fee - maker_fee

    # Rule 1: sell beats hold
    if realized_profit >= ev_hold:
        return True, "sell_beats_hold"

    # Rule 2: conviction + time profit-taking
    if realized_profit > 0:
        risk_cents = max(1.0, (p_win * (1.0 - p_win)) ** 0.5 * 100.0)
        time_factor = 1.0
        if hours_to_settlement is not None:
            time_factor = 1.0 + params.profit_take_time_decay * hours_to_settlement / 24.0
        threshold = risk_cents * params.profit_take_risk_fraction / time_factor
        if realized_profit > threshold:
            return True, "profit_take"

    # Rule 3: negative EV cut
    if ev_hold <= -abs(params.deterioration_buffer_cents):
        return True, "negative_ev_cut"

    return False, "hold"


def calculate_liquidity_adjusted_price(
    bid_price: float | None,
    bid_qty: int | None,
    total_depth: int | None,
    volume: int | None,
    target_liquidity_contracts: int = 500,
) -> float | None:
    """Return a more realistic executable price considering depth and volume.

    If depth at best price is low, we blend toward last_price or a conservative estimate.
    """
    if bid_price is None:
        return None
    if bid_qty is None or bid_qty >= target_liquidity_contracts:
        return bid_price  # good depth at top of book

    # Simple approximation: if depth is low, move price toward less favorable by 20-50%
    depth_ratio = min(1.0, (bid_qty or 0) / target_liquidity_contracts)
    adjustment = (1.0 - depth_ratio) * 0.35  # move 35% toward worse price
    adjusted = bid_price * (1.0 - adjustment) if bid_price > 50 else bid_price * (1.0 + adjustment)
    return round(adjusted, 1)


def replay_trade_marks(
    db: sqlite3.Connection,
    trade_id: int,
    params: TunerParams,
    settlement_outcome: dict,
    *,
    trades_table: str = "paper_trades",
    marks_table: str = "paper_trade_marks",
) -> dict:
    """Replay one trade's mark history under different exit params.

    Returns the counterfactual outcome: when would we have exited,
    at what price, and what would the P&L have been?
    """
    marks = db.execute(
        f"""SELECT mark_time, mark_price_cents, estimated_probability,
                  expected_value_cents, note
           FROM {marks_table}
           WHERE trade_id = ?
           ORDER BY mark_time""",
        (trade_id,),
    ).fetchall()

    trade = db.execute(
        f"SELECT * FROM {trades_table} WHERE id = ?", (trade_id,)
    ).fetchone()

    if not trade or not marks:
        return {"status": "no_data"}

    entry_price = float(trade["entry_price_cents"])
    side = trade["side"]
    entry_fee = _kalshi_fee_cents(entry_price)
    contracts = max(1, int(trade["contracts"] or 1))
    target_date = trade["target_date"]

    # Parse settlement end time for hours_to_settlement calculation
    try:
        from datetime import timedelta
        end_day = datetime.strptime(target_date, "%Y-%m-%d") + timedelta(days=1)
        settlement_utc = end_day.replace(hour=5, minute=0, tzinfo=timezone.utc)
    except (ValueError, TypeError):
        settlement_utc = None

    # Replay marks — find the first mark where the simulated exit triggers
    for mark in marks:
        bid = mark["mark_price_cents"]
        prob = mark["estimated_probability"]
        if bid is None or prob is None:
            continue

        # Liquidity-adjusted price (realistic executable price)
        liquidity_adjusted = calculate_liquidity_adjusted_price(
            bid, None, None, None, target_liquidity_contracts=500
        )
        should_exit, reason = _simulate_exit_decision(
            params, entry_price, side, prob, bid, hours_left, liquidity_adjusted_bid=liquidity_adjusted
        )

        hours_left = None
        if settlement_utc:
            try:
                mark_dt = datetime.fromisoformat(
                    mark["mark_time"].replace("Z", "+00:00")
                )
                hours_left = max(0.0, (settlement_utc - mark_dt).total_seconds() / 3600.0)
            except (ValueError, TypeError):
                pass

        should_exit, reason = _simulate_exit_decision(
            params, entry_price, side, prob, bid, hours_left,
        )

        if should_exit:
            maker_fee = _kalshi_fee_cents(bid, is_maker=True)
            pnl = round((bid - entry_price - entry_fee - maker_fee) * contracts, 2)
            return {
                "status": "exited",
                "exit_reason": reason,
                "exit_time": mark["mark_time"],
                "exit_price": bid,
                "pnl_cents": pnl,
                "hours_remaining": hours_left,
            }

    # Never exited → went to settlement
    winning_side = settlement_outcome.get("winning_side", "")
    won = str(side).lower() == str(winning_side).lower()
    settlement_price = 100.0 if won else 0.0
    pnl = round((settlement_price - entry_price - entry_fee) * contracts, 2)
    return {
        "status": "settlement",
        "exit_reason": "settlement",
        "pnl_cents": pnl,
        "won": won,
    }


# ---------------------------------------------------------------------------
# Grid search: evaluate multiple parameter sets
# ---------------------------------------------------------------------------

def _param_grid() -> list[TunerParams]:
    """Generate a grid of parameter combinations to evaluate."""
    grid = []
    for risk_frac in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
        for time_decay in [0.0, 0.15, 0.3, 0.5]:
            for buffer in [1.0, 2.0, 3.0, 5.0, 8.0]:
                grid.append(TunerParams(
                    profit_take_risk_fraction=risk_frac,
                    profit_take_time_decay=time_decay,
                    deterioration_buffer_cents=buffer,
                ))
    return grid


def evaluate_param_set(
    db: sqlite3.Connection,
    params: TunerParams,
    trade_ids: list[int],
    settlement_outcomes: dict[int, dict],
    *,
    trades_table: str = "paper_trades",
    marks_table: str = "paper_trade_marks",
) -> TunerResult:
    """Evaluate one parameter set across all historical trades."""
    pnls: list[float] = []
    n_profit_take = 0
    n_sell_beats = 0
    n_neg_ev = 0
    n_settlement = 0

    for trade_id in trade_ids:
        outcome = settlement_outcomes.get(trade_id, {})
        result = replay_trade_marks(
            db,
            trade_id,
            params,
            outcome,
            trades_table=trades_table,
            marks_table=marks_table,
        )

        if result.get("status") in ("exited", "settlement"):
            pnl = result.get("pnl_cents", 0.0)
            pnls.append(pnl)
            reason = result.get("exit_reason", "")
            if reason == "profit_take":
                n_profit_take += 1
            elif reason == "sell_beats_hold":
                n_sell_beats += 1
            elif reason == "negative_ev_cut":
                n_neg_ev += 1
            elif reason == "settlement":
                n_settlement += 1

    if not pnls:
        return TunerResult(params=params)

    total = sum(pnls)
    avg = total / len(pnls)
    wins = sum(1 for p in pnls if p > 0)
    std = (sum((p - avg) ** 2 for p in pnls) / max(1, len(pnls) - 1)) ** 0.5

    return TunerResult(
        params=params,
        total_pnl_cents=round(total, 1),
        n_trades=len(pnls),
        n_wins=wins,
        avg_pnl_cents=round(avg, 1),
        win_rate=round(wins / len(pnls), 3),
        sharpe_ratio=round(avg / max(0.01, std), 3),
        n_exits_profit_take=n_profit_take,
        n_exits_sell_beats_hold=n_sell_beats,
        n_exits_negative_ev=n_neg_ev,
        n_held_to_settlement=n_settlement,
    )


# ---------------------------------------------------------------------------
# Main tuning loop
# ---------------------------------------------------------------------------

def run_exit_tuning(
    db_path: str | Path,
    station: str = "KMIA",
    table_prefix: str = "",
    state_path: Path | None = None,
) -> dict:
    """Run the full exit tuning cycle.

    1. Load all settled trades with mark history
    2. Load settlement outcomes
    3. Evaluate a grid of parameter sets via counterfactual replay
    4. Find the best params by Sharpe ratio (risk-adjusted P&L)
    5. EMA-blend toward the best params
    6. Save state and return summary

    Args:
        table_prefix: "" for production paper_trades, "ds3m_" for DS3M
        state_path: where to save tuner state. Defaults to production or DS3M path.
    """
    if state_path is None:
        state_path = DS3M_STATE_PATH if table_prefix == "ds3m_" else STATE_PATH
    trades_table = f"{table_prefix}paper_trades"
    marks_table = f"{table_prefix}paper_trade_marks"
    settlements_table = "market_settlements"

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    try:
        marks_exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (marks_table,),
        ).fetchone()
        if not marks_exists:
            log.info("Exit tuner: marks table %s does not exist. Skipping.", marks_table)
            return {"status": "insufficient_data", "reason": "marks_table_missing", "n_trades": 0}

        # 1. Get all settled/closed trades that have mark history
        trade_rows = conn.execute(
            f"""SELECT pt.id, pt.ticker, pt.side, pt.entry_price_cents,
                       pt.realized_pnl_cents, pt.exit_reason
                FROM {trades_table} pt
                WHERE pt.station = ? AND pt.status != 'open'
                  AND pt.realized_pnl_cents IS NOT NULL
                  AND EXISTS (
                      SELECT 1 FROM {marks_table} pm WHERE pm.trade_id = pt.id
                  )""",
            (station,),
        ).fetchall()

        if len(trade_rows) < 5:
            log.info("Exit tuner: only %d trades with marks, need ≥5. Skipping.", len(trade_rows))
            return {"status": "insufficient_data", "n_trades": len(trade_rows)}

        trade_ids = [r["id"] for r in trade_rows]

        # 2. Load settlement outcomes per ticker
        settlement_outcomes: dict[int, dict] = {}
        for row in trade_rows:
            ms = conn.execute(
                "SELECT winning_side FROM market_settlements WHERE ticker = ? ORDER BY id DESC LIMIT 1",
                (row["ticker"],),
            ).fetchone()
            if ms:
                settlement_outcomes[row["id"]] = {"winning_side": ms["winning_side"]}

        # 3. Evaluate parameter grid
        grid = _param_grid()
        results: list[TunerResult] = []

        log.info("Exit tuner: evaluating %d param sets across %d trades (%d marks)...",
                 len(grid), len(trade_ids),
                 conn.execute(
                     f"SELECT COUNT(*) FROM {marks_table} WHERE trade_id IN ({','.join('?' * len(trade_ids))})",
                     trade_ids,
                 ).fetchone()[0])

        for params in grid:
            result = evaluate_param_set(
                conn,
                params,
                trade_ids,
                settlement_outcomes,
                trades_table=trades_table,
                marks_table=marks_table,
            )
            results.append(result)

        # 4. Find best by Sharpe ratio (risk-adjusted, not just raw P&L)
        # This aligns with the new "money bot" philosophy - we care about smooth, reliable profit
        results.sort(key=lambda r: r.sharpe_ratio, reverse=True)
        best = results[0]

        # Also find best by raw P&L for comparison
        best_pnl = max(results, key=lambda r: r.total_pnl_cents)

        # Current params (what we're actually running)
        state = ExitTunerState.load(state_path)
        current = TunerParams(**{k: v for k, v in state.current_params.items() if k in TunerParams.__dataclass_fields__})
        current_result = evaluate_param_set(
            conn,
            current,
            trade_ids,
            settlement_outcomes,
            trades_table=trades_table,
            marks_table=marks_table,
        )

        # 5. EMA blend toward best Sharpe params
        alpha = state.ema_alpha
        new_params = {}
        for field_name in TunerParams.__dataclass_fields__:
            current_val = getattr(current, field_name)
            best_val = getattr(best.params, field_name)
            blended = round(current_val * (1 - alpha) + best_val * alpha, 4)
            new_params[field_name] = blended

        # 6. Save state
        state.current_params = new_params
        state.n_evaluations += 1
        state.last_evaluation_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        state.evaluation_history.append({
            "timestamp": state.last_evaluation_utc,
            "n_trades": len(trade_ids),
            "current_pnl": current_result.total_pnl_cents,
            "best_sharpe_pnl": best.total_pnl_cents,
            "best_sharpe_ratio": best.sharpe_ratio,
            "best_raw_pnl": best_pnl.total_pnl_cents,
            "recommended_params": asdict(best.params),
            "blended_params": new_params,
        })
        # Keep last 30 evaluations
        state.evaluation_history = state.evaluation_history[-30:]
        state.save(state_path)

        log.info(
            "Exit tuner complete: current_pnl=%.1f¢, best_sharpe_pnl=%.1f¢ (sharpe=%.2f), "
            "best_raw_pnl=%.1f¢. Blended params: risk_frac=%.2f, time_decay=%.2f, buffer=%.1f¢",
            current_result.total_pnl_cents,
            best.total_pnl_cents,
            best.sharpe_ratio,
            best_pnl.total_pnl_cents,
            new_params["profit_take_risk_fraction"],
            new_params["profit_take_time_decay"],
            new_params["deterioration_buffer_cents"],
        )

        return {
            "status": "ok",
            "n_trades": len(trade_ids),
            "n_param_sets": len(grid),
            "current": {
                "params": asdict(current),
                "pnl": current_result.total_pnl_cents,
                "sharpe": current_result.sharpe_ratio,
                "win_rate": current_result.win_rate,
                "exits": {
                    "profit_take": current_result.n_exits_profit_take,
                    "sell_beats_hold": current_result.n_exits_sell_beats_hold,
                    "negative_ev_cut": current_result.n_exits_negative_ev,
                    "settlement": current_result.n_held_to_settlement,
                },
            },
            "best_sharpe": {
                "params": asdict(best.params),
                "pnl": best.total_pnl_cents,
                "sharpe": best.sharpe_ratio,
                "win_rate": best.win_rate,
                "exits": {
                    "profit_take": best.n_exits_profit_take,
                    "sell_beats_hold": best.n_exits_sell_beats_hold,
                    "negative_ev_cut": best.n_exits_negative_ev,
                    "settlement": best.n_held_to_settlement,
                },
            },
            "best_raw_pnl": {
                "params": asdict(best_pnl.params),
                "pnl": best_pnl.total_pnl_cents,
            },
            "blended_params": new_params,
            "improvement_cents": round(best.total_pnl_cents - current_result.total_pnl_cents, 1),
        }
    finally:
        conn.close()


def apply_tuned_params(policy) -> dict:
    """Apply the tuner's recommended params to a PaperTradingPolicy instance.

    Call this at the start of each trading cycle to use the latest tuned values.
    Returns the params that were applied.
    """
    state = ExitTunerState.load()
    params = state.current_params

    if "profit_take_risk_fraction" in params:
        policy.profit_take_risk_fraction = params["profit_take_risk_fraction"]
    if "profit_take_time_decay" in params:
        policy.profit_take_time_decay = params["profit_take_time_decay"]
    if "deterioration_buffer_cents" in params:
        policy.deterioration_buffer_cents = params["deterioration_buffer_cents"]
    if "min_edge_cents" in params:
        policy.min_edge_cents = params["min_edge_cents"]
    if "min_ev_cents" in params:
        policy.min_ev_cents = params["min_ev_cents"]
    if "min_price_cents" in params:
        policy.min_price_cents = params["min_price_cents"]

    return params


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Exit policy auto-tuner")
    parser.add_argument("--db", required=True, help="Path to collector DB")
    parser.add_argument("--station", default="KMIA")
    parser.add_argument("--ds3m", action="store_true", help="Tune DS3M paper trader instead of production")
    args = parser.parse_args()

    prefix = "ds3m_" if args.ds3m else ""
    result = run_exit_tuning(args.db, station=args.station, table_prefix=prefix)
    print(json.dumps(result, indent=2, default=str))
