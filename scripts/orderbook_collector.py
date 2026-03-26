#!/usr/bin/env python3
"""Kalshi orderbook depth collector — snapshots every 60 seconds for active weather markets.

Designed for 24/7 operation on the Mac Mini under launchd.
Stores snapshots in miami_collector.db, table kalshi_orderbook_snapshots.

Auth: RSA-PSS signing using the same credentials as the main collector
  - KALSHI_API_KEY_ID from /Users/blakebot/.credentials/.env
  - Private key from /Users/blakebot/.credentials/kalshi-prod-private.pem
  - Config from /Users/blakebot/blakebot/miami-project/config.toml
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import logging.handlers
import os
import signal
import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import aiohttp
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from dotenv import dotenv_values

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_DIR = Path("/Users/blakebot/blakebot/miami-project")
CONFIG_PATH = PROJECT_DIR / "config.toml"
DB_PATH = PROJECT_DIR / "miami_collector.db"
LOG_DIR = PROJECT_DIR / "logs"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_DIR.mkdir(exist_ok=True)

_fmt = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)

_sh = logging.StreamHandler(sys.stdout)
_sh.setFormatter(_fmt)

_fh = logging.handlers.RotatingFileHandler(
    LOG_DIR / "orderbook_collector.log",
    maxBytes=10 * 1024 * 1024,  # 10 MB
    backupCount=5,
)
_fh.setFormatter(_fmt)

logging.root.setLevel(logging.INFO)
logging.root.addHandler(_sh)
logging.root.addHandler(_fh)

log = logging.getLogger("orderbook_collector")

# ---------------------------------------------------------------------------
# Kalshi RSA-PSS auth (standalone, no project imports needed)
# ---------------------------------------------------------------------------
REST_BASE = "https://api.elections.kalshi.com/trade-api/v2"
REST_PATH_PREFIX = "/trade-api/v2"


def _load_private_key(key_path: str | Path) -> rsa.RSAPrivateKey:
    data = Path(key_path).read_bytes()
    key = serialization.load_pem_private_key(data, password=None)
    assert isinstance(key, rsa.RSAPrivateKey)
    return key


def _sign_request(private_key: rsa.RSAPrivateKey, timestamp_ms: int, method: str, path: str) -> str:
    message = str(timestamp_ms) + method.upper() + path
    signature = private_key.sign(
        message.encode("utf-8"),
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH,
        ),
        hashes.SHA256(),
    )
    return base64.b64encode(signature).decode("utf-8")


def _make_auth_headers(key_id: str, private_key: rsa.RSAPrivateKey, method: str, path: str) -> dict[str, str]:
    ts_ms = int(time.time() * 1000)
    full_path = REST_PATH_PREFIX + path if not path.startswith(REST_PATH_PREFIX) else path
    sig = _sign_request(private_key, ts_ms, method, full_path)
    return {
        "KALSHI-ACCESS-KEY": key_id,
        "KALSHI-ACCESS-SIGNATURE": sig,
        "KALSHI-ACCESS-TIMESTAMP": str(ts_ms),
        "Content-Type": "application/json",
    }


# ---------------------------------------------------------------------------
# Config loading (lightweight, no project import dependency)
# ---------------------------------------------------------------------------
def _load_config() -> dict:
    """Load minimal config needed for the collector."""
    import tomllib

    with open(CONFIG_PATH, "rb") as f:
        raw = tomllib.load(f)

    env_path = raw.get("credentials", {}).get("env_path", "")
    env = dotenv_values(env_path) if env_path else {}

    kalshi_key_path = raw.get("credentials", {}).get(
        "kalshi_key_path",
        env.get("KALSHI_PRIVATE_KEY_PATH", ""),
    )

    return {
        "api_key_id": env.get("KALSHI_API_KEY_ID", ""),
        "private_key_path": kalshi_key_path,
        "rest_base": raw.get("kalshi", {}).get("rest_base", REST_BASE),
        "high_series": raw.get("station", {}).get("kalshi_high_series", "KXHIGHMIA"),
        "low_series": raw.get("station", {}).get("kalshi_low_series", "KXLOWTMIA"),
    }


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS kalshi_orderbook_snapshots (
    ticker          TEXT    NOT NULL,
    timestamp_utc   TEXT    NOT NULL,
    yes_best_bid    REAL,
    yes_best_bid_size REAL,
    yes_total_bid_depth REAL,
    no_best_bid     REAL,
    no_best_bid_size REAL,
    no_total_bid_depth REAL,
    spread_cents    REAL,
    mid_price       REAL,
    levels_json     TEXT,
    PRIMARY KEY (ticker, timestamp_utc)
);
"""

CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_ob_snap_ts ON kalshi_orderbook_snapshots(timestamp_utc);
"""


def _init_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(CREATE_TABLE_SQL)
    conn.execute(CREATE_INDEX_SQL)
    conn.commit()
    return conn


def _insert_snapshots(conn: sqlite3.Connection, rows: list[tuple]) -> int:
    if not rows:
        return 0
    conn.executemany(
        """INSERT OR IGNORE INTO kalshi_orderbook_snapshots
           (ticker, timestamp_utc, yes_best_bid, yes_best_bid_size,
            yes_total_bid_depth, no_best_bid, no_best_bid_size,
            no_total_bid_depth, spread_cents, mid_price, levels_json)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    conn.commit()
    return len(rows)


# ---------------------------------------------------------------------------
# Kalshi REST client
# ---------------------------------------------------------------------------
class KalshiClient:
    def __init__(self, cfg: dict):
        self._cfg = cfg
        self._key_id = cfg["api_key_id"]
        self._private_key = _load_private_key(cfg["private_key_path"])
        self._rest_base = cfg["rest_base"]
        self._session: aiohttp.ClientSession | None = None

    async def open(self) -> None:
        timeout = aiohttp.ClientTimeout(total=30)
        self._session = aiohttp.ClientSession(timeout=timeout)

    async def close(self) -> None:
        if self._session:
            await self._session.close()
            self._session = None

    def _headers(self, method: str, path: str) -> dict[str, str]:
        return _make_auth_headers(self._key_id, self._private_key, method, path)

    async def _get(self, path: str, params: dict | None = None) -> dict | None:
        assert self._session, "Client not opened"
        url = f"{self._rest_base}{path}"
        headers = self._headers("GET", path)
        try:
            async with self._session.get(url, headers=headers, params=params) as resp:
                if resp.status == 429:
                    retry_after = float(resp.headers.get("Retry-After", "5"))
                    log.warning("Rate limited on %s — sleeping %.1fs", path, retry_after)
                    await asyncio.sleep(retry_after)
                    return None
                if resp.status == 404:
                    return None
                if resp.status != 200:
                    log.warning("HTTP %d on %s", resp.status, path)
                    return None
                return await resp.json()
        except asyncio.TimeoutError:
            log.warning("Timeout on %s", path)
            return None
        except Exception:
            log.exception("Request error on %s", path)
            return None

    async def discover_active_markets(self) -> list[dict]:
        """Find all weather markets closing within the next 48 hours."""
        now = datetime.now(timezone.utc)
        cutoff = now + timedelta(hours=48)
        markets: list[dict] = []

        for series in [self._cfg["high_series"], self._cfg["low_series"]]:
            if not series:
                continue
            cursor = None
            while True:
                params: dict = {
                    "series_ticker": series,
                    "status": "open",
                    "limit": 200,
                }
                if cursor:
                    params["cursor"] = cursor

                data = await self._get("/markets", params)
                if not data:
                    break

                batch = data.get("markets", [])
                for m in batch:
                    # Filter: only markets closing within 48h
                    close_time_str = m.get("close_time") or m.get("expiration_time") or ""
                    if not close_time_str:
                        # Include if we can't determine close time
                        markets.append(m)
                        continue
                    try:
                        close_time = datetime.fromisoformat(close_time_str.replace("Z", "+00:00"))
                        if close_time <= cutoff:
                            markets.append(m)
                    except ValueError:
                        markets.append(m)

                cursor = data.get("cursor")
                if not cursor or len(batch) < 200:
                    break

        return markets

    async def get_orderbook(self, ticker: str, depth: int = 0) -> dict | None:
        """Fetch full orderbook for a single market."""
        return await self._get(f"/markets/{ticker}/orderbook", {"depth": depth})


# ---------------------------------------------------------------------------
# Snapshot processing
# ---------------------------------------------------------------------------
def _process_orderbook(ticker: str, data: dict, ts_utc: str) -> tuple | None:
    """Convert raw orderbook JSON into a DB row tuple."""
    ob = data.get("orderbook", data)
    yes_levels = ob.get("yes", [])
    no_levels = ob.get("no", [])

    # Best bids and depths
    yes_best_bid = None
    yes_best_bid_size = None
    yes_total_depth = 0.0
    if yes_levels:
        # yes_levels: [[price_cents, quantity], ...]
        best = max(yes_levels, key=lambda x: x[0])
        yes_best_bid = best[0]
        yes_best_bid_size = best[1]
        yes_total_depth = sum(q for _, q in yes_levels)

    no_best_bid = None
    no_best_bid_size = None
    no_total_depth = 0.0
    if no_levels:
        best = max(no_levels, key=lambda x: x[0])
        no_best_bid = best[0]
        no_best_bid_size = best[1]
        no_total_depth = sum(q for _, q in no_levels)

    # Derived: spread and mid price
    spread = None
    mid_price = None
    if yes_best_bid is not None and no_best_bid is not None:
        # Spread = 100 - yes_best_bid - no_best_bid
        # If positive, there's a gap; if negative, bids overlap
        spread = 100.0 - yes_best_bid - no_best_bid
        # Mid = (yes_best_bid + (100 - no_best_bid)) / 2
        yes_implied_ask = 100.0 - no_best_bid
        mid_price = (yes_best_bid + yes_implied_ask) / 2.0

    # Compact JSON of full depth
    levels_json = json.dumps({"yes": yes_levels, "no": no_levels}, separators=(",", ":"))

    return (
        ticker,
        ts_utc,
        yes_best_bid,
        yes_best_bid_size,
        yes_total_depth if yes_levels else None,
        no_best_bid,
        no_best_bid_size,
        no_total_depth if no_levels else None,
        spread,
        mid_price,
        levels_json,
    )


# ---------------------------------------------------------------------------
# Main collector loop
# ---------------------------------------------------------------------------
class OrderbookCollector:
    def __init__(self):
        self._cfg = _load_config()
        self._client = KalshiClient(self._cfg)
        self._db: sqlite3.Connection | None = None
        self._running = False

        # Stats
        self._total_snapshots = 0
        self._total_errors = 0
        self._total_cycles = 0
        self._last_stats_time = time.monotonic()
        self._current_tickers: list[str] = []

    async def start(self) -> None:
        log.info("Starting orderbook collector")
        log.info("DB: %s", DB_PATH)
        log.info("Series: %s, %s", self._cfg["high_series"], self._cfg["low_series"])

        self._db = _init_db(DB_PATH)
        await self._client.open()
        self._running = True

        # Set up signal handlers for graceful shutdown
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(self._shutdown(s)))

        try:
            await self._run_loop()
        finally:
            await self._cleanup()

    async def _shutdown(self, sig: signal.Signals) -> None:
        log.info("Received %s, shutting down gracefully...", sig.name)
        self._running = False

    async def _cleanup(self) -> None:
        await self._client.close()
        if self._db:
            self._db.close()
            self._db = None
        log.info(
            "Collector stopped. Total cycles=%d, snapshots=%d, errors=%d",
            self._total_cycles, self._total_snapshots, self._total_errors,
        )

    async def _run_loop(self) -> None:
        # Discover markets immediately on first run
        market_refresh_interval = 600  # Re-discover markets every 10 minutes
        last_market_refresh = 0.0

        while self._running:
            cycle_start = time.monotonic()

            # Refresh market list periodically
            if time.monotonic() - last_market_refresh >= market_refresh_interval:
                await self._refresh_markets()
                last_market_refresh = time.monotonic()

            if self._current_tickers:
                await self._snapshot_cycle()
            else:
                log.info("No active markets found, waiting...")

            # Log stats every 10 minutes
            if time.monotonic() - self._last_stats_time >= 600:
                self._log_stats()
                self._last_stats_time = time.monotonic()

            # Sleep until next 60-second mark
            elapsed = time.monotonic() - cycle_start
            sleep_time = max(1.0, 60.0 - elapsed)
            try:
                await asyncio.sleep(sleep_time)
            except asyncio.CancelledError:
                break

    async def _refresh_markets(self) -> None:
        log.info("Refreshing active weather markets...")
        try:
            markets = await self._client.discover_active_markets()
            self._current_tickers = [m.get("ticker", "") for m in markets if m.get("ticker")]
            log.info("Found %d active markets closing within 48h", len(self._current_tickers))
            if self._current_tickers:
                log.debug("Tickers: %s", ", ".join(self._current_tickers[:10]))
        except Exception:
            log.exception("Failed to refresh markets")
            self._total_errors += 1

    async def _snapshot_cycle(self) -> None:
        self._total_cycles += 1
        ts_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        rows: list[tuple] = []
        cycle_errors = 0

        # Fetch orderbooks with small delays between requests to stay under rate limit
        for i, ticker in enumerate(self._current_tickers):
            try:
                data = await self._client.get_orderbook(ticker)
                if data is None:
                    cycle_errors += 1
                    continue

                row = _process_orderbook(ticker, data, ts_utc)
                if row:
                    rows.append(row)

            except Exception:
                log.exception("Error fetching orderbook for %s", ticker)
                cycle_errors += 1

            # Small delay between requests: ~50ms keeps us well under 20 req/sec
            if i < len(self._current_tickers) - 1:
                await asyncio.sleep(0.05)

        # Batch insert
        if rows and self._db:
            try:
                inserted = _insert_snapshots(self._db, rows)
                self._total_snapshots += inserted
            except Exception:
                log.exception("DB insert error")
                cycle_errors += 1

        self._total_errors += cycle_errors

        if cycle_errors > 0:
            log.warning(
                "Cycle %d: %d/%d snapshots, %d errors",
                self._total_cycles, len(rows), len(self._current_tickers), cycle_errors,
            )
        else:
            log.debug(
                "Cycle %d: %d snapshots stored", self._total_cycles, len(rows),
            )

    def _log_stats(self) -> None:
        log.info(
            "STATS — cycles=%d | snapshots=%d | errors=%d | active_markets=%d",
            self._total_cycles,
            self._total_snapshots,
            self._total_errors,
            len(self._current_tickers),
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    collector = OrderbookCollector()
    try:
        asyncio.run(collector.start())
    except KeyboardInterrupt:
        log.info("Interrupted by user")


if __name__ == "__main__":
    main()
