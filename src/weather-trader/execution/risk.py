"""Risk management — pre-trade safety checks + portfolio-level risk controls.

Every trade passes through these gates before execution.
Any gate failure = trade blocked.

T6.3 enhancements:
  - Cross-station correlation-aware exposure limits
  - Portfolio-level heat (total capital at risk) limits
  - Drawdown-triggered scaling
  - Per-station concentration limits
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timedelta, timezone

from src.config import TraderConfig
from src.execution.trader import TradeDecision

log = logging.getLogger(__name__)

# Station clusters for correlation-aware sizing.
# Stations in the same cluster have correlated forecast errors (~70%+ ρ).
# Combined exposure within a cluster is treated as concentrated risk.
STATION_CLUSTERS: dict[str, list[str]] = {
    "se_florida": ["KMIA", "KFLL"],
    "ne_corridor": ["KNYC", "KPHL", "KBOS", "KDCA"],
    "texas": ["KAUS", "KSAT", "KHOU", "KDFW"],
    "upper_midwest": ["KMDW", "KMSP"],
    "mountain": ["KDEN", "KPHX", "KLAS"],
    "west_coast": ["KLAX", "KSFO", "KSEA"],
}

# Reverse lookup: station → cluster name
_STATION_TO_CLUSTER: dict[str, str] = {}
for cluster_name, stations in STATION_CLUSTERS.items():
    for s in stations:
        _STATION_TO_CLUSTER[s] = cluster_name


class RiskManager:
    """Pre-trade risk gates with portfolio-level controls."""

    def __init__(self, cfg: TraderConfig, trader_db: sqlite3.Connection):
        self.cfg = cfg
        self._db = trader_db
        self._consecutive_losses = 0
        self._cooldown_until: datetime | None = None
        # Portfolio-level limits (T6.3)
        self.max_portfolio_heat_pct = 0.25  # 25% of bankroll at risk at any time
        self.max_per_station_pct = 0.10     # 10% per station-day
        self.max_per_cluster_pct = 0.15     # 15% per cluster (correlated stations)
        self.daily_loss_circuit_breaker_pct = 0.03  # 3% daily loss → halve all sizes
        # Drawdown state
        self.peak_bankroll: float | None = None
        self.current_drawdown_pct: float = 0.0

    def check(self, decision: TradeDecision) -> tuple[bool, str]:
        """Check all risk gates. Returns (allowed, reason)."""
        tp = self.cfg.trading

        # Gate 1: Cooldown
        if self._cooldown_until and datetime.now(timezone.utc) < self._cooldown_until:
            remaining = (self._cooldown_until - datetime.now(timezone.utc)).total_seconds() / 60
            return False, f"COOLDOWN: {remaining:.0f}min left"
        elif self._cooldown_until:
            self._cooldown_until = None
            self._consecutive_losses = 0

        # Gate 2: Daily loss limit
        today_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        row = self._db.execute(
            """SELECT COALESCE(SUM(CASE WHEN total_pnl_cents < 0 THEN total_pnl_cents ELSE 0 END), 0)
               FROM trade_outcomes
               WHERE substr(created_at, 1, 10) = ?""",
            (today_utc,),
        ).fetchone()
        daily_loss = abs(row[0] / 100.0) if row else 0.0
        if daily_loss >= tp.max_daily_loss:
            return False, f"DAILY_LOSS: ${daily_loss:.2f} >= ${tp.max_daily_loss:.2f}"

        # Gate 3: Max positions per market type
        row = self._db.execute(
            """SELECT COUNT(DISTINCT ticker) FROM trades
               WHERE market_type = ? AND target_date = ?
                 AND order_status IN ('pending', 'filled') AND mode = 'live'""",
            (decision.market_type, decision.target_date),
        ).fetchone()
        if (row[0] or 0) >= tp.max_open_positions:
            return False, f"MAX_POSITIONS: {row[0]} >= {tp.max_open_positions}"

        # Gate 4: Edge sanity
        if decision.edge > 0.60:
            return False, f"EDGE_SANITY: {decision.edge:.1%} — likely data issue"

        # Gate 5: Price sanity
        if decision.price_cents < 2 or decision.price_cents > 95:
            return False, f"PRICE_SANITY: {decision.price_cents}c extreme"

        # Gate 6: Portfolio heat (T6.3) — total capital at risk across all stations
        total_at_risk = self._total_capital_at_risk()
        bankroll = tp.bankroll if hasattr(tp, 'bankroll') else 250.0
        if bankroll > 0 and total_at_risk / bankroll > self.max_portfolio_heat_pct:
            return False, f"PORTFOLIO_HEAT: ${total_at_risk:.2f} ({total_at_risk/bankroll:.0%}) > {self.max_portfolio_heat_pct:.0%}"

        # Gate 7: Per-station concentration (T6.3)
        station = getattr(decision, 'station', None)
        if station and bankroll > 0:
            station_exposure = self._station_exposure(station)
            if station_exposure / bankroll > self.max_per_station_pct:
                return False, f"STATION_CONCENTRATION: {station} ${station_exposure:.2f} > {self.max_per_station_pct:.0%}"

            # Gate 8: Per-cluster concentration (T6.3)
            cluster = _STATION_TO_CLUSTER.get(station)
            if cluster:
                cluster_exposure = self._cluster_exposure(cluster)
                if cluster_exposure / bankroll > self.max_per_cluster_pct:
                    return False, f"CLUSTER_CONCENTRATION: {cluster} ${cluster_exposure:.2f} > {self.max_per_cluster_pct:.0%}"

        # Gate 9: Drawdown scaling (T6.3)
        if self.current_drawdown_pct >= 0.30:
            return False, "DRAWDOWN_HALT: ≥30% drawdown — trading stopped"

        return True, "OK"

    def _total_capital_at_risk(self) -> float:
        """Total dollars at risk across all open positions."""
        try:
            row = self._db.execute(
                """SELECT COALESCE(SUM(contracts * price_cents / 100.0), 0)
                   FROM trades
                   WHERE order_status IN ('pending', 'filled') AND mode = 'live'"""
            ).fetchone()
            return float(row[0]) if row else 0.0
        except Exception:
            return 0.0

    def _station_exposure(self, station: str) -> float:
        """Dollars at risk for a specific station."""
        try:
            row = self._db.execute(
                """SELECT COALESCE(SUM(contracts * price_cents / 100.0), 0)
                   FROM trades
                   WHERE station = ? AND order_status IN ('pending', 'filled') AND mode = 'live'""",
                (station,),
            ).fetchone()
            return float(row[0]) if row else 0.0
        except Exception:
            return 0.0

    def _cluster_exposure(self, cluster: str) -> float:
        """Dollars at risk across all stations in a cluster."""
        stations = STATION_CLUSTERS.get(cluster, [])
        if not stations:
            return 0.0
        placeholders = ",".join("?" * len(stations))
        try:
            row = self._db.execute(
                f"""SELECT COALESCE(SUM(contracts * price_cents / 100.0), 0)
                    FROM trades
                    WHERE station IN ({placeholders})
                      AND order_status IN ('pending', 'filled') AND mode = 'live'""",
                stations,
            ).fetchone()
            return float(row[0]) if row else 0.0
        except Exception:
            return 0.0

    def update_drawdown(self, current_bankroll: float) -> None:
        """Update drawdown state from current bankroll value.

        Call this after each settlement cycle with the updated bankroll.
        """
        if self.peak_bankroll is None or current_bankroll > self.peak_bankroll:
            self.peak_bankroll = current_bankroll

        if self.peak_bankroll > 0:
            self.current_drawdown_pct = (self.peak_bankroll - current_bankroll) / self.peak_bankroll
        else:
            self.current_drawdown_pct = 0.0

        if self.current_drawdown_pct >= 0.30:
            log.critical("DRAWDOWN HALT: %.1f%% from peak — trading stopped", self.current_drawdown_pct * 100)
        elif self.current_drawdown_pct >= 0.20:
            log.warning("DRAWDOWN SEVERE: %.1f%% — quarter-size positions", self.current_drawdown_pct * 100)
        elif self.current_drawdown_pct >= 0.10:
            log.warning("DRAWDOWN WARNING: %.1f%% — half-size positions", self.current_drawdown_pct * 100)

    def drawdown_size_multiplier(self) -> float:
        """Position size multiplier based on current drawdown.

        Returns:
            1.0 = normal, 0.5 = half-size, 0.25 = quarter-size, 0.0 = stopped
        """
        if self.current_drawdown_pct >= 0.30:
            return 0.0
        elif self.current_drawdown_pct >= 0.20:
            return 0.25
        elif self.current_drawdown_pct >= 0.10:
            return 0.50
        return 1.0

    def record_outcome(self, won: bool) -> None:
        tp = self.cfg.trading
        if won:
            self._consecutive_losses = 0
        else:
            self._consecutive_losses += 1
            if self._consecutive_losses >= tp.max_consecutive_losses:
                self._cooldown_until = datetime.now(timezone.utc) + timedelta(minutes=tp.cooldown_minutes)
                log.warning("Cooldown: %d consecutive losses, pausing %d min",
                            self._consecutive_losses, tp.cooldown_minutes)
