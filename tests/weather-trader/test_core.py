import sqlite3
import unittest

from src.config import TraderConfig
from src.execution.kalshi_client import KalshiClient
from src.execution.risk import RiskManager
from src.execution.trader import TradeDecision
from src.ledger.ledger import Ledger


class WeatherTraderTests(unittest.TestCase):
    def test_kalshi_balance_non_live_is_none(self):
        cfg = TraderConfig()
        client = KalshiClient(cfg.kalshi, mode="shadow")

        async def _run():
            self.assertIsNone(await client.get_balance())

        import asyncio
        asyncio.run(_run())

    def test_risk_max_positions_uses_decision_target_date(self):
        conn = sqlite3.connect(":memory:")
        conn.executescript(
            """
            CREATE TABLE trades (
                id INTEGER PRIMARY KEY,
                market_type TEXT,
                target_date TEXT,
                ticker TEXT,
                order_status TEXT,
                mode TEXT
            );
            CREATE TABLE trade_outcomes (
                id INTEGER PRIMARY KEY,
                total_pnl_cents INTEGER,
                created_at TEXT
            );
            """
        )
        # Existing open position for 2026-03-20 should block another on same date.
        conn.execute(
            "INSERT INTO trades (market_type, target_date, ticker, order_status, mode) VALUES (?,?,?,?,?)",
            ("high", "2026-03-20", "T1", "pending", "live"),
        )
        conn.commit()

        cfg = TraderConfig()
        cfg.trading.max_open_positions = 1
        risk = RiskManager(cfg, conn)
        decision = TradeDecision(
            station="KMIA",
            ticker="T2",
            market_type="high",
            target_date="2026-03-20",
            side="yes",
            price_cents=40,
            contracts=1,
            edge=0.05,
        )
        allowed, reason = risk.check(decision)
        self.assertFalse(allowed)
        self.assertIn("MAX_POSITIONS", reason)

    def test_settlement_prefers_explicit_bounds(self):
        conn = sqlite3.connect(":memory:")
        conn.executescript(
            """
            CREATE TABLE signal_snapshots (id INTEGER PRIMARY KEY);
            CREATE TABLE trades (
                id INTEGER PRIMARY KEY,
                timestamp_utc TEXT,
                station TEXT,
                target_date TEXT,
                market_type TEXT,
                ticker TEXT,
                side TEXT,
                action TEXT,
                price_cents INTEGER,
                contracts INTEGER,
                our_probability REAL,
                market_probability REAL,
                edge REAL,
                kelly_size REAL,
                settlement_floor REAL,
                settlement_ceil REAL,
                signal_snapshot_id INTEGER,
                mode TEXT,
                kalshi_order_id TEXT,
                order_status TEXT,
                fill_price_cents INTEGER,
                created_at TEXT
            );
            CREATE TABLE trade_outcomes (
                id INTEGER PRIMARY KEY,
                trade_id INTEGER,
                target_date TEXT,
                market_type TEXT,
                cli_value_f REAL,
                winning_side TEXT,
                our_side_won INTEGER,
                pnl_cents INTEGER,
                total_pnl_cents INTEGER,
                created_at TEXT
            );
            """
        )
        conn.execute(
            """INSERT INTO trades (
                timestamp_utc, station, target_date, market_type, ticker, side, action,
                price_cents, contracts, our_probability, market_probability, edge, kelly_size,
                settlement_floor, settlement_ceil, signal_snapshot_id, mode, order_status
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                "2026-03-20T01:00:00Z", "KMIA", "2026-03-20", "high", "KXHIGHMIA-...-T80",
                "yes", "buy", 40, 1, 0.6, 0.4, 0.2, 0.1,
                79.0, 200.0, None, "shadow", "shadow",
            ),
        )
        conn.commit()

        ledger = Ledger(conn)
        outcomes = ledger.settle_trades("2026-03-20", "high", 79.5)
        self.assertEqual(len(outcomes), 1)
        self.assertTrue(outcomes[0]["won"])


if __name__ == "__main__":
    unittest.main()
