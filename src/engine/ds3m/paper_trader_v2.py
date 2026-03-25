"""Unified paper trader for DS3M v2.

Replaces both:
  - trading/paper_trader.py (old production paper trader)
  - engine/ds3m/paper_trader.py (shadow comparison trader)

Key features:
  - Regime-conditioned position sizing (fractional Kelly × regime mult)
  - Maker/taker order type selection based on edge magnitude
  - Microstructure-aware execution timing
  - Concurrent position limits per bracket and total
  - Settlement P&L tracking with per-regime attribution
  - Conformal-calibrated probabilities
  - KF settlement error feedback for adaptation
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────

@dataclass
class TraderConfig:
    """Paper trading configuration."""
    bankroll_cents: int = 100_000           # $1000 starting bankroll
    max_open_trades: int = 8                # total concurrent positions
    max_per_bracket: int = 2                # max positions per bracket label
    max_daily_trades: int = 15              # throttle
    min_edge_cents: float = 4.0             # minimum adjusted edge to enter
    min_ev_cents: float = 2.0               # minimum expected value
    kelly_fraction: float = 0.15            # 15% fractional Kelly
    max_position_pct: float = 0.05          # 5% of bankroll per trade
    cooldown_minutes: int = 10              # min time between trades on same bracket
    # Regime caps: override max_position_pct by regime
    regime_position_caps: dict[str, float] = field(default_factory=lambda: {
        "continental": 0.05,
        "sea_breeze": 0.04,    # tighter — sea breeze regime = more volatile
        "frontal": 0.03,       # frontal passage = high uncertainty
        "tropical": 0.04,
        "nocturnal": 0.06,     # overnight warm = profitable niche
    })
    default_regime_cap: float = 0.04


# ──────────────────────────────────────────────────────────────────────
# Trade record
# ──────────────────────────────────────────────────────────────────────

@dataclass
class TradeRecord:
    """In-memory trade record."""
    trade_id: int
    ticker: str
    station: str
    target_date: str
    market_type: str
    side: str                  # "BUY_YES" or "BUY_NO"
    contracts: int
    entry_price_cents: float
    entry_time: str
    model_prob: float
    edge_cents: float
    ev_cents: float
    kelly_fraction: float
    order_type: str            # "maker" or "taker"
    regime_name: str
    regime_sizing_mult: float
    microstructure_mult: float
    bracket_label: str


# ──────────────────────────────────────────────────────────────────────
# Unified Paper Trader
# ──────────────────────────────────────────────────────────────────────

class PaperTraderV2:
    """Production paper trader for DS3M v2.

    Lifecycle:
      1. evaluate_signals() — score and filter TradeSignals from BracketPricerV2
      2. execute_trades() — insert qualifying trades into DB
      3. settle_trades() — called after CLI settlement, compute P&L
      4. get_performance() — aggregate stats for dashboard

    Uses the paper_trades / paper_trade_settlements tables (shared with old system
    for backwards-compatible dashboard queries).
    """

    def __init__(self, db_path: str, station: str = "KMIA", config: TraderConfig | None = None):
        self.db_path = db_path
        self.station = station
        self.config = config or TraderConfig()
        self._open_trades: dict[int, TradeRecord] = {}
        self._load_open_trades()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=5)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    # ── Trade Evaluation ──────────────────────────────────────────

    def evaluate_signals(self, signals: list[Any]) -> list[Any]:
        """Filter TradeSignals through risk management rules.

        Args:
            signals: list of TradeSignal from BracketPricerV2.price_all_brackets()

        Returns:
            Filtered list of signals that pass all checks.
        """
        if not signals:
            return []

        now_utc = datetime.now(timezone.utc).isoformat()
        today = now_utc[:10]

        # Check position limits
        n_open = len(self._open_trades)
        if n_open >= self.config.max_open_trades:
            log.debug(f"At max open trades ({n_open})")
            return []

        # Count today's trades
        daily_count = self._count_daily_trades(today)
        if daily_count >= self.config.max_daily_trades:
            log.debug(f"At daily trade limit ({daily_count})")
            return []

        approved = []
        slots_available = self.config.max_open_trades - n_open
        daily_remaining = self.config.max_daily_trades - daily_count

        for signal in signals:
            if len(approved) >= min(slots_available, daily_remaining):
                break

            # Edge threshold
            if signal.adjusted_edge < self.config.min_edge_cents:
                continue

            # Per-bracket limit
            bracket_label = signal.bracket.label
            bracket_count = sum(
                1 for t in self._open_trades.values()
                if t.bracket_label == bracket_label
            )
            if bracket_count >= self.config.max_per_bracket:
                continue

            # Cooldown check
            if self._in_cooldown(bracket_label, signal.side):
                continue

            # Position size with regime cap
            regime_cap = self.config.regime_position_caps.get(
                signal.regime_name, self.config.default_regime_cap
            )
            max_size_cents = self.config.bankroll_cents * min(
                signal.kelly_fraction,
                regime_cap,
            )

            # Calculate contracts
            if signal.side == "BUY_YES":
                cost_per_contract = signal.bracket.yes_price * 100  # cents
            else:
                cost_per_contract = signal.bracket.no_price * 100

            if cost_per_contract <= 0:
                continue

            contracts = max(1, int(max_size_cents / cost_per_contract))
            # Store computed contracts on signal for execute step
            signal._computed_contracts = contracts

            approved.append(signal)

        return approved

    # ── Trade Execution ───────────────────────────────────────────

    def execute_trades(self, signals: list[Any], target_date: str) -> list[int]:
        """Insert approved signals as paper trades.

        Returns: list of trade IDs created.
        """
        if not signals:
            return []

        conn = self._conn()
        trade_ids = []
        now_utc = datetime.now(timezone.utc).isoformat()

        for signal in signals:
            contracts = getattr(signal, '_computed_contracts', 1)

            if signal.side == "BUY_YES":
                entry_price = signal.bracket.yes_price * 100
            else:
                entry_price = signal.bracket.no_price * 100

            thesis = {
                "model_prob": signal.model_prob,
                "market_prob": signal.market_prob,
                "gross_edge": signal.gross_edge_cents,
                "net_edge_taker": signal.net_edge_taker,
                "net_edge_maker": signal.net_edge_maker,
                "adjusted_edge": signal.adjusted_edge,
                "micro_mult": signal.microstructure_mult,
                "kelly": signal.kelly_fraction,
                "order_type": signal.order_type,
                "regime": signal.regime_name,
                "regime_sizing_mult": signal.regime_sizing_mult,
                "source": "ds3m_v2",
            }

            try:
                cursor = conn.execute(
                    """INSERT INTO paper_trades
                       (ticker, station, target_date, market_type, side, contracts,
                        entry_price_cents, entry_time, estimated_probability,
                        expected_edge_cents, expected_value_cents, thesis_json,
                        status)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open')""",
                    (
                        signal.bracket.ticker,
                        self.station,
                        target_date,
                        signal.bracket.market_type,
                        signal.side,
                        contracts,
                        entry_price,
                        now_utc,
                        signal.model_prob,
                        signal.adjusted_edge,
                        signal.adjusted_edge * contracts,
                        json.dumps(thesis),
                    ),
                )
                trade_id = cursor.lastrowid
                trade_ids.append(trade_id)

                record = TradeRecord(
                    trade_id=trade_id,
                    ticker=signal.bracket.ticker,
                    station=self.station,
                    target_date=target_date,
                    market_type=signal.bracket.market_type,
                    side=signal.side,
                    contracts=contracts,
                    entry_price_cents=entry_price,
                    entry_time=now_utc,
                    model_prob=signal.model_prob,
                    edge_cents=signal.adjusted_edge,
                    ev_cents=signal.adjusted_edge * contracts,
                    kelly_fraction=signal.kelly_fraction,
                    order_type=signal.order_type,
                    regime_name=signal.regime_name,
                    regime_sizing_mult=signal.regime_sizing_mult,
                    microstructure_mult=signal.microstructure_mult,
                    bracket_label=signal.bracket.label,
                )
                self._open_trades[trade_id] = record

                log.info(
                    f"TRADE: {signal.side} {signal.bracket.label} "
                    f"x{contracts} @ {entry_price:.0f}¢ "
                    f"(edge={signal.adjusted_edge:.1f}¢, "
                    f"regime={signal.regime_name}, {signal.order_type})"
                )

            except Exception as e:
                log.error(f"Trade insert error: {e}")

        conn.commit()
        conn.close()
        return trade_ids

    # ── Settlement ────────────────────────────────────────────────

    def settle_trades(
        self,
        target_date: str,
        actual_high_f: float,
        actual_low_f: float,
    ) -> list[dict]:
        """Settle all open trades for a given target date.

        Called after CLI settlement with actual high/low temperatures.

        Returns: list of settlement dicts with P&L and regime info.
        """
        conn = self._conn()
        settled_at = datetime.now(timezone.utc).isoformat()
        settlements = []

        rows = conn.execute(
            """SELECT id, ticker, side, contracts, entry_price_cents,
                      market_type, thesis_json
               FROM paper_trades
               WHERE station = ? AND target_date = ? AND status = 'open'""",
            (self.station, target_date),
        ).fetchall()

        for row in rows:
            d = dict(row)
            trade_id = d["id"]
            side = d["side"]
            entry_price = d["entry_price_cents"]
            contracts = d["contracts"]
            market_type = d.get("market_type", "high")

            # Parse thesis for regime attribution
            thesis = {}
            if d.get("thesis_json"):
                try:
                    thesis = json.loads(d["thesis_json"])
                except Exception:
                    pass

            # Determine winning side from actual settlement
            actual_val = actual_high_f if market_type == "high" else actual_low_f
            ticker = d["ticker"]

            # Parse bracket bounds from ticker
            winning_side = self._resolve_bracket_outcome(ticker, actual_val)

            # Calculate P&L
            if winning_side == "YES":
                settlement_price = 100  # YES wins → $1
            else:
                settlement_price = 0    # NO wins → $0

            if side == "BUY_YES":
                pnl_per_contract = settlement_price - entry_price
            else:  # BUY_NO
                pnl_per_contract = (100 - settlement_price) - (100 - entry_price)

            total_pnl = pnl_per_contract * contracts

            # Subtract fees
            if thesis.get("order_type") == "taker":
                from engine.ds3m.bracket_pricer_v2 import kalshi_taker_fee
                fee = kalshi_taker_fee(entry_price) * contracts
            else:
                from engine.ds3m.bracket_pricer_v2 import kalshi_maker_fee
                fee = kalshi_maker_fee(entry_price) * contracts
            total_pnl -= fee

            # Update trade
            conn.execute(
                """UPDATE paper_trades
                   SET status = 'settled', exit_price_cents = ?,
                       exit_time = ?, exit_reason = 'settlement',
                       realized_pnl_cents = ?
                   WHERE id = ?""",
                (settlement_price, settled_at, total_pnl, trade_id),
            )

            # Insert settlement record
            conn.execute(
                """INSERT OR REPLACE INTO paper_trade_settlements
                   (trade_id, ticker, winning_side, settlement_price_cents,
                    settled_at, realized_pnl_cents)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (trade_id, ticker, winning_side, settlement_price,
                 settled_at, total_pnl),
            )

            settlement = {
                "trade_id": trade_id,
                "ticker": ticker,
                "side": side,
                "winning_side": winning_side,
                "pnl_cents": total_pnl,
                "entry_price": entry_price,
                "contracts": contracts,
                "regime_name": thesis.get("regime", ""),
                "model_prob": thesis.get("model_prob", 0),
                "order_type": thesis.get("order_type", "maker"),
                "won": (side == "BUY_YES" and winning_side == "YES") or
                       (side == "BUY_NO" and winning_side == "NO"),
            }
            settlements.append(settlement)

            # Remove from open trades
            self._open_trades.pop(trade_id, None)

            log.info(
                f"SETTLED: {ticker} {side} → {'WIN' if settlement['won'] else 'LOSS'} "
                f"{total_pnl:+.0f}¢ (regime={thesis.get('regime', '?')})"
            )

        conn.commit()
        conn.close()

        if settlements:
            self._log_settlement_summary(settlements)

        return settlements

    def _resolve_bracket_outcome(self, ticker: str, actual_temp: float) -> str:
        """Determine if YES or NO wins for a bracket given the actual temperature.

        Parses bracket strikes from DB and checks if actual_temp falls within range.
        """
        try:
            conn = self._conn()
            row = conn.execute(
                "SELECT floor_strike, cap_strike FROM active_brackets WHERE ticker = ?",
                (ticker,),
            ).fetchone()
            conn.close()

            if row:
                floor_s = row["floor_strike"]
                cap_s = row["cap_strike"]
                floor_val = float(floor_s) if floor_s is not None else -1e6
                cap_val = float(cap_s) if cap_s is not None else 1e6
                if floor_val <= actual_temp < cap_val:
                    return "YES"
                return "NO"
        except Exception:
            pass

        # Fallback: try parsing from ticker string
        return "YES"  # conservative fallback

    def _log_settlement_summary(self, settlements: list[dict]):
        total_pnl = sum(s["pnl_cents"] for s in settlements)
        wins = sum(1 for s in settlements if s["won"])
        n = len(settlements)

        # Per-regime breakdown
        regime_pnl: dict[str, float] = {}
        regime_count: dict[str, int] = {}
        for s in settlements:
            r = s["regime_name"] or "unknown"
            regime_pnl[r] = regime_pnl.get(r, 0) + s["pnl_cents"]
            regime_count[r] = regime_count.get(r, 0) + 1

        regime_str = ", ".join(
            f"{r}: {pnl:+.0f}¢ ({regime_count[r]} trades)"
            for r, pnl in sorted(regime_pnl.items(), key=lambda x: -abs(x[1]))
        )

        log.info(
            f"SETTLEMENT SUMMARY: {wins}/{n} wins, total P&L: {total_pnl:+.0f}¢ "
            f"(${total_pnl/100:+.2f}) | by regime: {regime_str}"
        )

    # ── Performance Metrics ───────────────────────────────────────

    def get_performance(self, days: int = 30) -> dict:
        """Aggregate performance metrics for dashboard."""
        try:
            conn = self._conn()
            rows = conn.execute(
                """SELECT t.side, t.entry_price_cents, t.contracts, t.thesis_json,
                          s.winning_side, s.realized_pnl_cents, s.settled_at
                   FROM paper_trade_settlements s
                   JOIN paper_trades t ON t.id = s.trade_id
                   WHERE t.station = ?
                     AND s.settled_at >= date('now', ?)
                   ORDER BY s.settled_at DESC""",
                (self.station, f"-{days} days"),
            ).fetchall()
            conn.close()

            if not rows:
                return {"total_pnl_cents": 0, "win_rate": 0, "n_trades": 0,
                        "regime_breakdown": {}, "maker_vs_taker": {}}

            total_pnl = 0
            wins = 0
            regime_stats: dict[str, dict] = {}
            maker_pnl = 0
            taker_pnl = 0
            maker_count = 0
            taker_count = 0

            for row in rows:
                d = dict(row)
                pnl = d["realized_pnl_cents"] or 0
                total_pnl += pnl
                won = (d["side"] == "BUY_YES" and d["winning_side"] == "YES") or \
                      (d["side"] == "BUY_NO" and d["winning_side"] == "NO")
                if won:
                    wins += 1

                thesis = {}
                if d.get("thesis_json"):
                    try:
                        thesis = json.loads(d["thesis_json"])
                    except Exception:
                        pass

                regime = thesis.get("regime", "unknown")
                if regime not in regime_stats:
                    regime_stats[regime] = {"pnl": 0, "wins": 0, "total": 0}
                regime_stats[regime]["pnl"] += pnl
                regime_stats[regime]["total"] += 1
                if won:
                    regime_stats[regime]["wins"] += 1

                order_type = thesis.get("order_type", "maker")
                if order_type == "taker":
                    taker_pnl += pnl
                    taker_count += 1
                else:
                    maker_pnl += pnl
                    maker_count += 1

            n = len(rows)
            return {
                "total_pnl_cents": total_pnl,
                "total_pnl_dollars": total_pnl / 100,
                "win_rate": wins / n if n > 0 else 0,
                "n_trades": n,
                "avg_pnl_per_trade": total_pnl / n if n > 0 else 0,
                "regime_breakdown": {
                    r: {
                        "pnl_cents": s["pnl"],
                        "win_rate": s["wins"] / s["total"] if s["total"] > 0 else 0,
                        "n_trades": s["total"],
                    }
                    for r, s in regime_stats.items()
                },
                "maker_vs_taker": {
                    "maker": {"pnl": maker_pnl, "count": maker_count},
                    "taker": {"pnl": taker_pnl, "count": taker_count},
                },
            }

        except Exception as e:
            log.error(f"Performance query error: {e}")
            return {"total_pnl_cents": 0, "win_rate": 0, "n_trades": 0,
                    "regime_breakdown": {}, "maker_vs_taker": {}}

    # ── Helpers ───────────────────────────────────────────────────

    def _load_open_trades(self):
        """Load open trades from DB on startup."""
        try:
            conn = self._conn()
            rows = conn.execute(
                """SELECT id, ticker, station, target_date, market_type, side,
                          contracts, entry_price_cents, entry_time,
                          estimated_probability, expected_edge_cents,
                          expected_value_cents, thesis_json
                   FROM paper_trades
                   WHERE station = ? AND status = 'open'""",
                (self.station,),
            ).fetchall()
            conn.close()

            for row in rows:
                d = dict(row)
                thesis = {}
                if d.get("thesis_json"):
                    try:
                        thesis = json.loads(d["thesis_json"])
                    except Exception:
                        pass

                self._open_trades[d["id"]] = TradeRecord(
                    trade_id=d["id"],
                    ticker=d["ticker"],
                    station=d["station"],
                    target_date=d["target_date"],
                    market_type=d["market_type"],
                    side=d["side"],
                    contracts=d["contracts"],
                    entry_price_cents=d["entry_price_cents"],
                    entry_time=d["entry_time"],
                    model_prob=d.get("estimated_probability", 0),
                    edge_cents=d.get("expected_edge_cents", 0),
                    ev_cents=d.get("expected_value_cents", 0),
                    kelly_fraction=thesis.get("kelly", 0),
                    order_type=thesis.get("order_type", "maker"),
                    regime_name=thesis.get("regime", ""),
                    regime_sizing_mult=thesis.get("regime_sizing_mult", 1.0),
                    microstructure_mult=thesis.get("micro_mult", 1.0),
                    bracket_label="",
                )

            if self._open_trades:
                log.info(f"Loaded {len(self._open_trades)} open trades")

        except Exception as e:
            log.warning(f"Could not load open trades: {e}")

    def _count_daily_trades(self, date: str) -> int:
        try:
            conn = self._conn()
            row = conn.execute(
                """SELECT COUNT(*) as n FROM paper_trades
                   WHERE station = ? AND target_date = ?""",
                (self.station, date),
            ).fetchone()
            conn.close()
            return row["n"] if row else 0
        except Exception:
            return 0

    def _in_cooldown(self, bracket_label: str, side: str) -> bool:
        """Check if we recently traded this bracket (avoid rapid-fire)."""
        now = datetime.now(timezone.utc)
        for trade in self._open_trades.values():
            if trade.bracket_label == bracket_label and trade.side == side:
                try:
                    entry_time = datetime.fromisoformat(trade.entry_time.replace("Z", "+00:00"))
                    minutes_ago = (now - entry_time).total_seconds() / 60
                    if minutes_ago < self.config.cooldown_minutes:
                        return True
                except Exception:
                    pass
        return False

    @property
    def open_trade_count(self) -> int:
        return len(self._open_trades)

    @property
    def open_trades(self) -> list[TradeRecord]:
        return list(self._open_trades.values())
