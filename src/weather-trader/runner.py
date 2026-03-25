"""Runner — main loop for the weather trader.

Architecture:
  1. Discover Kalshi brackets → push to city DBs (active_brackets)
  2. Read city probability estimates (bracket_estimates)
  3. Compare estimates to live Kalshi prices
  4. Edge > threshold → Kelly size → place order (or log in shadow/paper)
  5. Track in ledger

The trader has ZERO weather knowledge. Cities are the weather brains.
The trader is purely: probabilities vs prices → edge → execute.

Usage:
    python3 -m src.runner --city KMIA:/path/to/miami_collector.db --mode shadow --once
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.config import CityConfig, TraderConfig
from src.execution.bracket_publisher import discover_and_publish
from src.execution.kalshi_client import KalshiClient
from src.execution.risk import RiskManager
from src.execution.trader import (
    MarketPrice, Trader,
    format_decisions, read_city_estimates,
)
from src.ledger.db import init_trader_db
from src.ledger.ledger import Ledger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _current_target_date(utc_hour_start: int = 5) -> str:
    """Determine the current climate day (midnight LST = 05:00Z for EST)."""
    now = datetime.now(timezone.utc)
    d = now.date() - timedelta(days=1) if now.hour < utc_hour_start else now.date()
    return d.strftime("%Y-%m-%d")


def _read_market_prices_from_city_db(
    city_db: sqlite3.Connection, target_date: str,
) -> dict[str, MarketPrice]:
    """Read latest Kalshi prices from the city's market_snapshots table.

    Temporary: until we move the Kalshi WS feed to the trader,
    we read cached prices from the city collector's DB.
    """
    city_db.row_factory = sqlite3.Row
    rows = city_db.execute(
        """SELECT ms.ticker, ms.best_yes_bid_cents, ms.best_yes_ask_cents, ms.last_price_cents
           FROM market_snapshots ms
           JOIN (
               SELECT ticker, MAX(id) AS max_id
               FROM market_snapshots
               WHERE forecast_date = ?
               GROUP BY ticker
           ) latest ON latest.max_id = ms.id""",
        (target_date,),
    ).fetchall()

    prices = {}
    for r in rows:
        prices[r["ticker"]] = MarketPrice(
            ticker=r["ticker"],
            yes_bid=r["best_yes_bid_cents"],
            yes_ask=r["best_yes_ask_cents"],
            last_price=r["last_price_cents"],
        )
    return prices


def _publish_brackets_from_snapshots(
    city_db: sqlite3.Connection, station: str, target_date: str,
) -> int:
    """Discover brackets from market_snapshots and publish to active_brackets.

    Temporary bridge: until the trader has its own Kalshi WS feed,
    we derive bracket definitions from the city's cached market data.
    """
    city_db.row_factory = sqlite3.Row
    rows = city_db.execute(
        """SELECT ms.ticker, ms.market_type, ms.floor_strike, ms.cap_strike
           FROM market_snapshots ms
           JOIN (
               SELECT ticker, MAX(id) AS max_id
               FROM market_snapshots
               WHERE forecast_date = ?
               GROUP BY ticker
           ) latest ON latest.max_id = ms.id""",
        (target_date,),
    ).fetchall()

    markets = [dict(r) for r in rows]
    dbs = {station: city_db}
    return discover_and_publish(markets, dbs, target_date)


async def run_cycle(
    cfg: TraderConfig,
    city_conns: dict[str, sqlite3.Connection],
    trader: Trader,
    risk: RiskManager,
    ledger: Ledger,
    kalshi: KalshiClient,
) -> None:
    """Run one complete poll → evaluate → trade cycle."""
    now = datetime.now(timezone.utc)
    target_date = _current_target_date()
    log.info("─── Cycle: %s | target: %s | mode: %s ───",
             now.strftime("%H:%M:%SZ"), target_date, cfg.mode.upper())

    for city in cfg.cities:
        db = city_conns.get(city.station)
        if not db:
            continue

        # Step 1: Publish bracket definitions to city DB
        n_brackets = _publish_brackets_from_snapshots(db, city.station, target_date)
        log.info("[%s] Published %d brackets", city.station, n_brackets)

        # Step 2: Read city's probability estimates
        estimates = read_city_estimates(db, target_date)
        if not estimates:
            log.info("[%s] No bracket estimates found — city engine may not have run yet", city.station)
            continue
        log.info("[%s] Read %d estimates (latest: %s)",
                 city.station, len(estimates), estimates[0].timestamp_utc)

        # Step 3: Get market prices
        prices = _read_market_prices_from_city_db(db, target_date)
        log.info("[%s] Got %d market prices", city.station, len(prices))

        # Step 4: Find trades with edge
        decisions = trader.evaluate(estimates, prices)
        log.info("[%s] Trade decisions:\n%s",
                 city.station, format_decisions(decisions, cfg.mode))

        # Step 5: Execute
        for d in decisions:
            allowed, reason = risk.check(d)
            if not allowed:
                log.info("  ❌ BLOCKED: %s — %s", d.ticker, reason)
                continue

            result = await kalshi.place_order(d)
            if not result.success:
                log.warning("  ⚠️ ORDER FAILED: %s — %s", d.ticker, result.message)

            # Log to ledger
            ledger.log_trade_with_context(
                d, snapshot_id=None,
                station=city.station, target_date=target_date,
                mode=cfg.mode,
                kalshi_order_id=result.order_id,
                order_status=("rejected" if not result.success else result.status),
                fill_price_cents=result.fill_price_cents,
            )

    log.info("─── Cycle complete ───\n")


async def main(args: argparse.Namespace) -> None:
    # Build config
    cfg = TraderConfig(
        trader_db_path=args.trader_db or str(Path.cwd() / "trader.db"),
        mode=args.mode,
        poll_interval_sec=args.interval,
    )

    # Parse city args: "KMIA:/path/to/db"
    for city_str in args.city:
        parts = city_str.split(":", 1)
        if len(parts) != 2:
            log.error("Invalid city format: %s (expected STATION:/path/to/db)", city_str)
            sys.exit(1)
        cfg.cities.append(CityConfig(station=parts[0], db_path=parts[1]))

    log.info("Weather Trader starting")
    log.info("  Mode:      %s", cfg.mode.upper())
    log.info("  Trader DB: %s", cfg.trader_db_path)
    log.info("  Cities:    %s", [c.station for c in cfg.cities])

    # Open connections
    trader_conn = init_trader_db(cfg.trader_db_path)
    city_conns: dict[str, sqlite3.Connection] = {}
    for city in cfg.cities:
        city_conns[city.station] = sqlite3.connect(city.db_path)
        log.info("  Connected to %s: %s", city.station, city.db_path)

    # Initialize components
    trader_mod = Trader(cfg.trading)
    risk = RiskManager(cfg, trader_conn)
    ledger_mod = Ledger(trader_conn)
    kalshi = KalshiClient(cfg.kalshi, cfg.mode)
    await kalshi.open()

    try:
        if args.once:
            await run_cycle(cfg, city_conns, trader_mod, risk, ledger_mod, kalshi)
        else:
            while True:
                try:
                    await run_cycle(cfg, city_conns, trader_mod, risk, ledger_mod, kalshi)
                except Exception:
                    log.exception("Cycle error — retrying next interval")
                await asyncio.sleep(cfg.poll_interval_sec)
    finally:
        await kalshi.close()
        for conn in city_conns.values():
            conn.close()
        trader_conn.close()


def cli() -> None:
    parser = argparse.ArgumentParser(description="Weather Trader")
    parser.add_argument(
        "--city", action="append", required=True,
        help="City in STATION:/path/to/db format (can repeat for multi-city)",
    )
    parser.add_argument("--trader-db", default=None, help="Trader DB path")
    parser.add_argument("--mode", choices=["shadow", "paper", "live"], default="shadow")
    parser.add_argument("--interval", type=int, default=300, help="Poll interval (seconds)")
    parser.add_argument("--once", action="store_true", help="Single cycle then exit")

    args = parser.parse_args()
    asyncio.run(main(args))


if __name__ == "__main__":
    cli()
