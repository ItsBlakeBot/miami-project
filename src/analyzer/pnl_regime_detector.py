"""T7.2: Trading performance regime detection via BOCPD on P&L series.

Applies the system's own BOCPD (Bayesian Online Changepoint Detection) to
the daily P&L time series. Detects when the trading system enters a
degraded performance regime (e.g., model update changed error characteristics,
seasonal transition, market microstructure change).

When a P&L regime shift is detected:
  - Log the event
  - Recommend: reduce position sizes, widen edge thresholds
  - Feed into adaptive Kelly (T6.2) drawdown scaling

This is the bot watching itself for performance degradation — the same
probabilistic changepoint detector it uses for weather regime shifts.
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class PnLRegimeState:
    """Current P&L regime assessment."""

    regime: str  # "normal", "degraded", "recovering", "halted"
    changepoint_probability: float = 0.0
    run_length_since_change: int = 0
    rolling_sharpe: float | None = None
    rolling_win_rate: float | None = None
    cumulative_pnl_cents: float = 0.0
    n_settlement_days: int = 0
    recommended_size_multiplier: float = 1.0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "regime": self.regime,
            "changepoint_probability": round(self.changepoint_probability, 4),
            "run_length_since_change": self.run_length_since_change,
            "rolling_sharpe": round(self.rolling_sharpe, 3) if self.rolling_sharpe else None,
            "rolling_win_rate": round(self.rolling_win_rate, 3) if self.rolling_win_rate else None,
            "cumulative_pnl_cents": round(self.cumulative_pnl_cents, 2),
            "n_settlement_days": self.n_settlement_days,
            "recommended_size_multiplier": round(self.recommended_size_multiplier, 3),
            "notes": self.notes,
        }


def detect_pnl_regime(
    db: sqlite3.Connection,
    lookback_days: int = 30,
    reference_date: str | None = None,
    bocpd_hazard: float = 0.05,
) -> PnLRegimeState:
    """Run BOCPD on daily P&L to detect trading regime shifts.

    Args:
        db: SQLite connection with row_factory=sqlite3.Row
        lookback_days: Number of days to analyze
        reference_date: Reference date (default: today)
        bocpd_hazard: Prior probability of changepoint at each step

    Returns:
        PnLRegimeState with current regime assessment and recommendations.
    """
    from engine.bocpd import BOCPDConfig, GaussianMeanBOCPD

    if reference_date is None:
        ref = date.today()
    else:
        ref = date.fromisoformat(reference_date)

    start_date = (ref - timedelta(days=lookback_days)).isoformat()

    # Get daily P&L from paper trade settlements
    rows = db.execute(
        """SELECT substr(created_at, 1, 10) AS trade_date,
                  SUM(realized_pnl_cents) AS daily_pnl,
                  COUNT(*) AS n_trades,
                  SUM(CASE WHEN realized_pnl_cents > 0 THEN 1 ELSE 0 END) AS wins
           FROM paper_trade_settlements
           WHERE substr(created_at, 1, 10) >= ?
             AND realized_pnl_cents IS NOT NULL
           GROUP BY trade_date
           ORDER BY trade_date""",
        (start_date,),
    ).fetchall()

    state = PnLRegimeState(regime="normal")

    if len(rows) < 3:
        state.notes.append(f"Only {len(rows)} settlement days — insufficient for regime detection")
        state.n_settlement_days = len(rows)
        return state

    daily_pnl = [float(r["daily_pnl"]) for r in rows]
    daily_trades = [int(r["n_trades"]) for r in rows]
    daily_wins = [int(r["wins"]) for r in rows]
    state.n_settlement_days = len(rows)

    # Cumulative P&L
    state.cumulative_pnl_cents = sum(daily_pnl)

    # Rolling win rate
    total_trades = sum(daily_trades)
    total_wins = sum(daily_wins)
    if total_trades > 0:
        state.rolling_win_rate = total_wins / total_trades

    # Rolling Sharpe
    if len(daily_pnl) >= 5:
        mean_pnl = sum(daily_pnl) / len(daily_pnl)
        std_pnl = math.sqrt(sum((x - mean_pnl) ** 2 for x in daily_pnl) / len(daily_pnl))
        if std_pnl > 0:
            state.rolling_sharpe = round((mean_pnl / std_pnl) * math.sqrt(252), 3)

    # Run BOCPD on standardized daily P&L
    if len(daily_pnl) >= 5:
        pnl_arr = np.array(daily_pnl)
        mean = np.mean(pnl_arr)
        std = max(np.std(pnl_arr), 1.0)
        standardized = (pnl_arr - mean) / std

        bocpd = GaussianMeanBOCPD(BOCPDConfig(hazard=bocpd_hazard))
        for val in standardized:
            cp_prob = bocpd.update(float(val))

        state.changepoint_probability = float(cp_prob)
        posterior = bocpd.run_length_posterior
        if posterior.size > 0:
            state.run_length_since_change = int(np.argmax(posterior))

    # Determine regime
    if state.changepoint_probability > 0.7:
        state.regime = "degraded"
        state.recommended_size_multiplier = 0.50
        state.notes.append(f"High changepoint probability ({state.changepoint_probability:.2f}) — possible regime shift")
    elif state.rolling_sharpe is not None and state.rolling_sharpe < -0.5:
        state.regime = "degraded"
        state.recommended_size_multiplier = 0.50
        state.notes.append(f"Negative Sharpe ({state.rolling_sharpe:.2f}) — losing money")
    elif state.rolling_win_rate is not None and state.rolling_win_rate < 0.40:
        state.regime = "degraded"
        state.recommended_size_multiplier = 0.75
        state.notes.append(f"Low win rate ({state.rolling_win_rate:.1%}) — below breakeven")
    elif state.cumulative_pnl_cents < 0:
        state.regime = "recovering"
        state.recommended_size_multiplier = 0.85
        state.notes.append("Cumulative P&L negative — recovery mode")
    else:
        state.regime = "normal"
        state.recommended_size_multiplier = 1.0

    return state


def main() -> None:
    """Run P&L regime detection and print results."""
    import argparse

    parser = argparse.ArgumentParser(description="Detect P&L trading regime")
    parser.add_argument("--db", required=True)
    parser.add_argument("--lookback-days", type=int, default=30)
    parser.add_argument("--reference-date", default=None)
    parser.add_argument("--out", default="analysis_data/pnl_regime.json")
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row

    state = detect_pnl_regime(conn, args.lookback_days, args.reference_date)
    conn.close()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(state.to_dict(), indent=2))

    print(f"P&L Regime: {state.regime}")
    print(f"  Changepoint prob: {state.changepoint_probability:.3f}")
    print(f"  Run length: {state.run_length_since_change}")
    print(f"  Rolling Sharpe: {state.rolling_sharpe}")
    print(f"  Win rate: {state.rolling_win_rate}")
    print(f"  Cumulative P&L: {state.cumulative_pnl_cents:.0f}¢")
    print(f"  Size multiplier: {state.recommended_size_multiplier:.2f}")
    for n in state.notes:
        print(f"  Note: {n}")


if __name__ == "__main__":
    main()
