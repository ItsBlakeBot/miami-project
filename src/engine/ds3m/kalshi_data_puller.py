#!/usr/bin/env python3
"""
Kalshi Historical Weather Data Puller
=====================================
Pulls all historical weather market data from Kalshi's free public API
for Miami and surrounding SE US cities. Stores tick-by-tick trades,
hourly candlesticks, and market metadata in SQLite.

No authentication required for historical endpoints.
"""

import json
import os
import re
import sqlite3
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DB_DIR = Path(__file__).resolve().parent.parent.parent.parent  # miami-project root
DB_PATH = DB_DIR / "miami_collector.db"

# Try both known API base URLs
API_BASES = [
    "https://api.elections.kalshi.com/trade-api/v2",
    "https://trading-api.kalshi.com/trade-api/v2",
]

# Rate limit: be conservative (10 req/sec -> 100ms between requests)
MIN_REQUEST_INTERVAL = 0.10  # seconds

# Retry config
MAX_RETRIES = 5
RETRY_BACKOFF_BASE = 2.0  # exponential backoff

# Pagination
PAGE_LIMIT = 1000  # max allowed by API

# ---------------------------------------------------------------------------
# City / ticker search patterns
# ---------------------------------------------------------------------------

# Known Kalshi weather ticker prefixes and city codes
# Format examples: KXHIGHNY, KXHIGHMI, KXLOWNY, HIGHLAX, etc.
WEATHER_PREFIXES = ["KXHIGH", "KXLOW", "HIGH", "LOW", "KHIGH", "KLOW"]

# City codes to search for (SE US focus, plus major cities for reference)
CITY_CODES = {
    # Primary - Florida
    "MIA": "Miami",
    "MI": "Miami",
    "KMIA": "Miami (KMIA)",
    "FLL": "Fort Lauderdale",
    "FL": "Fort Lauderdale",
    "TPA": "Tampa",
    "TP": "Tampa",
    "ORL": "Orlando",
    "OR": "Orlando",
    "MCO": "Orlando (MCO)",
    "JAX": "Jacksonville",
    "JA": "Jacksonville",
    "EYW": "Key West",
    "KW": "Key West",
    "PBI": "Palm Beach",
    "SRQ": "Sarasota",
    "RSW": "Fort Myers",
    # SE US
    "ATL": "Atlanta",
    "AT": "Atlanta",
    "CLT": "Charlotte",
    "CH": "Charlotte",
    "RDU": "Raleigh",
    "SAV": "Savannah",
    "CHS": "Charleston",
    "BNA": "Nashville",
    "NA": "Nashville",
    "MEM": "Memphis",
    "MSY": "New Orleans",
    "NO": "New Orleans",
    "HSV": "Huntsville",
    "BHM": "Birmingham",
    # Other major cities (useful reference data)
    "NY": "New York",
    "NYC": "New York",
    "CH": "Chicago",
    "CHI": "Chicago",
    "LA": "Los Angeles",
    "LAX": "Los Angeles",
    "HOU": "Houston",
    "HO": "Houston",
    "DFW": "Dallas",
    "DA": "Dallas",
    "PHX": "Phoenix",
    "PH": "Phoenix",
    "DEN": "Denver",
    "DE": "Denver",
    "SF": "San Francisco",
    "SEA": "Seattle",
    "BOS": "Boston",
    "BO": "Boston",
    "DC": "Washington DC",
    "WA": "Washington DC",
    "PHL": "Philadelphia",
    "PHI": "Philadelphia",
    "DET": "Detroit",
    "MSP": "Minneapolis",
    "STL": "St. Louis",
    "AUS": "Austin",
}

# Additional direct search terms
SEARCH_TERMS = [
    "KXHIGH", "KXLOW",
    "HIGHTEMP", "LOWTEMP",
    "TEMP", "WEATHER",
]

# ---------------------------------------------------------------------------
# HTTP helpers (using only stdlib - no requests dependency needed)
# ---------------------------------------------------------------------------

_last_request_time = 0.0
_working_base_url = None


def _rate_limit():
    """Enforce rate limit between requests."""
    global _last_request_time
    now = time.time()
    elapsed = now - _last_request_time
    if elapsed < MIN_REQUEST_INTERVAL:
        time.sleep(MIN_REQUEST_INTERVAL - elapsed)
    _last_request_time = time.time()


def _make_request(path: str, params: dict = None, base_url: str = None) -> dict | None:
    """Make a GET request to the Kalshi API with retries and rate limiting."""
    global _working_base_url

    bases = [base_url] if base_url else (
        [_working_base_url] if _working_base_url else API_BASES
    )

    for base in bases:
        url = f"{base}{path}"
        if params:
            query = urllib.parse.urlencode(
                {k: v for k, v in params.items() if v is not None}
            )
            url = f"{url}?{query}"

        for attempt in range(MAX_RETRIES):
            _rate_limit()
            try:
                req = urllib.request.Request(url, headers={
                    "User-Agent": "MiamiWeatherTrader/1.0",
                    "Accept": "application/json",
                })
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read().decode())
                    _working_base_url = base
                    return data
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    wait = RETRY_BACKOFF_BASE ** (attempt + 1)
                    print(f"  [429] Rate limited. Waiting {wait:.1f}s...")
                    time.sleep(wait)
                    continue
                elif e.code == 404:
                    return None
                elif e.code >= 500:
                    wait = RETRY_BACKOFF_BASE ** attempt
                    print(f"  [{e.code}] Server error. Retry in {wait:.1f}s...")
                    time.sleep(wait)
                    continue
                else:
                    # Try next base URL
                    break
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                if attempt < MAX_RETRIES - 1:
                    wait = RETRY_BACKOFF_BASE ** attempt
                    time.sleep(wait)
                    continue
                break

    return None


# ---------------------------------------------------------------------------
# Database setup
# ---------------------------------------------------------------------------

def init_db(db_path: Path) -> sqlite3.Connection:
    """Create SQLite database and tables."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS kalshi_markets (
            ticker TEXT PRIMARY KEY,
            event_ticker TEXT,
            title TEXT,
            subtitle TEXT,
            open_time TEXT,
            close_time TEXT,
            settlement_time TEXT,
            result TEXT,
            yes_sub_title TEXT,
            no_sub_title TEXT,
            volume INTEGER,
            open_interest INTEGER,
            status TEXT,
            category TEXT,
            series_ticker TEXT,
            strike_value REAL,
            floor_strike REAL,
            cap_strike REAL,
            raw_json TEXT,
            discovered_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS kalshi_trades (
            trade_id TEXT PRIMARY KEY,
            ticker TEXT NOT NULL,
            yes_price REAL,
            no_price REAL,
            count INTEGER,
            taker_side TEXT,
            created_time TEXT,
            FOREIGN KEY (ticker) REFERENCES kalshi_markets(ticker)
        );

        CREATE TABLE IF NOT EXISTS kalshi_candles (
            ticker TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            period INTEGER,
            PRIMARY KEY (ticker, timestamp, period),
            FOREIGN KEY (ticker) REFERENCES kalshi_markets(ticker)
        );

        CREATE INDEX IF NOT EXISTS idx_trades_ticker ON kalshi_trades(ticker);
        CREATE INDEX IF NOT EXISTS idx_trades_time ON kalshi_trades(created_time);
        CREATE INDEX IF NOT EXISTS idx_candles_ticker ON kalshi_candles(ticker);
        CREATE INDEX IF NOT EXISTS idx_markets_event ON kalshi_markets(event_ticker);
        CREATE INDEX IF NOT EXISTS idx_markets_category ON kalshi_markets(category);
    """)
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Market discovery
# ---------------------------------------------------------------------------

def discover_weather_markets(conn: sqlite3.Connection) -> list[str]:
    """
    Discover all weather-related markets by searching the events and markets
    endpoints with various weather-related terms.
    """
    print("\n" + "=" * 70)
    print("PHASE 1: DISCOVERING WEATHER MARKETS")
    print("=" * 70)

    discovered_tickers = set()
    discovered_events = set()

    # Strategy 1: Search events endpoint
    print("\n[1/3] Searching events for weather tickers...")
    event_search_terms = [
        "temperature", "weather", "high temp", "low temp",
        "degrees", "fahrenheit",
    ]
    for term in event_search_terms:
        print(f"  Searching events: '{term}'...")
        data = _make_request("/events", params={
            "series_ticker": "",
            "status": "",
            "limit": PAGE_LIMIT,
            "with_nested_markets": "true",
        })
        if not data:
            # Try without nested markets
            data = _make_request("/events", params={"limit": PAGE_LIMIT})
        if data and "events" in data:
            for event in data["events"]:
                eticker = event.get("event_ticker", "")
                title = event.get("title", "").upper()
                cat = event.get("category", "")
                # Check if weather-related
                is_weather = any(w in title for w in [
                    "TEMPERATURE", "WEATHER", "HIGH", "LOW", "DEGREES",
                    "FAHRENHEIT", "TEMP"
                ]) or any(w in eticker.upper() for w in [
                    "KXHIGH", "KXLOW", "HIGH", "LOW", "TEMP"
                ])
                if is_weather:
                    discovered_events.add(eticker)
                    # Get markets for this event
                    markets = event.get("markets", [])
                    for m in markets:
                        discovered_tickers.add(m.get("ticker", ""))
            print(f"    Found {len(discovered_events)} weather events so far")

    # Strategy 2: Direct ticker pattern search via markets endpoint
    print("\n[2/3] Searching markets by ticker patterns...")
    # Build search patterns
    patterns = set()
    for prefix in WEATHER_PREFIXES:
        patterns.add(prefix)
        for code in CITY_CODES:
            patterns.add(f"{prefix}{code}")

    # Also search by series_ticker patterns
    series_patterns = set()
    for prefix in ["KXHIGH", "KXLOW", "KHIGH", "KLOW"]:
        series_patterns.add(prefix)
        for code in ["MIA", "MI", "FLL", "TPA", "ORL", "JAX", "EYW",
                      "ATL", "CLT", "NY", "CHI", "LA", "HOU", "DFW",
                      "PHX", "DEN", "SF", "SEA", "BOS", "DC", "PHL",
                      "DET", "MSP", "STL", "AUS", "NO", "MEM", "BNA",
                      "SAV", "CHS"]:
            series_patterns.add(f"{prefix}{code}")

    searched = set()
    for pattern in sorted(series_patterns):
        if pattern in searched:
            continue
        searched.add(pattern)

        # Try as series_ticker
        data = _make_request("/markets", params={
            "series_ticker": pattern,
            "limit": PAGE_LIMIT,
        })
        if data and "markets" in data and len(data["markets"]) > 0:
            for m in data["markets"]:
                ticker = m.get("ticker", "")
                if ticker:
                    discovered_tickers.add(ticker)
                    _store_market(conn, m)
            print(f"    series={pattern}: found {len(data['markets'])} markets")

            # Paginate if needed
            cursor = data.get("cursor")
            while cursor:
                data = _make_request("/markets", params={
                    "series_ticker": pattern,
                    "limit": PAGE_LIMIT,
                    "cursor": cursor,
                })
                if not data or "markets" not in data or not data["markets"]:
                    break
                for m in data["markets"]:
                    ticker = m.get("ticker", "")
                    if ticker:
                        discovered_tickers.add(ticker)
                        _store_market(conn, m)
                cursor = data.get("cursor")

        # Try as event_ticker
        data = _make_request("/markets", params={
            "event_ticker": pattern,
            "limit": PAGE_LIMIT,
        })
        if data and "markets" in data and len(data["markets"]) > 0:
            new_count = 0
            for m in data["markets"]:
                ticker = m.get("ticker", "")
                if ticker and ticker not in discovered_tickers:
                    discovered_tickers.add(ticker)
                    _store_market(conn, m)
                    new_count += 1
            if new_count > 0:
                print(f"    event={pattern}: found {new_count} new markets")

            cursor = data.get("cursor")
            while cursor:
                data = _make_request("/markets", params={
                    "event_ticker": pattern,
                    "limit": PAGE_LIMIT,
                    "cursor": cursor,
                })
                if not data or "markets" not in data or not data["markets"]:
                    break
                for m in data["markets"]:
                    ticker = m.get("ticker", "")
                    if ticker and ticker not in discovered_tickers:
                        discovered_tickers.add(ticker)
                        _store_market(conn, m)
                cursor = data.get("cursor")

    # Strategy 3: Search via ticker prefix patterns
    print("\n[3/3] Searching by direct ticker prefixes...")
    ticker_prefixes = set()
    for prefix in ["KXHIGH", "KXLOW"]:
        for code in ["MIA", "MI", "FLL", "TPA", "ORL", "JAX", "EYW",
                      "ATL", "CLT", "NY", "CHI", "LA", "HOU", "DFW",
                      "PHX", "DEN", "SF", "SEA", "BOS", "DC", "PHL",
                      "DET", "MSP", "STL", "AUS", "NO", "MEM", "BNA",
                      "SAV", "CHS"]:
            ticker_prefixes.add(f"{prefix}{code}")

    for prefix in sorted(ticker_prefixes):
        if prefix in searched:
            continue
        searched.add(prefix)
        data = _make_request("/markets", params={
            "ticker": prefix,
            "limit": PAGE_LIMIT,
        })
        if data and "markets" in data and len(data["markets"]) > 0:
            new = 0
            for m in data["markets"]:
                t = m.get("ticker", "")
                if t and t not in discovered_tickers:
                    discovered_tickers.add(t)
                    _store_market(conn, m)
                    new += 1
            if new:
                print(f"    ticker={prefix}: {new} new markets")

    # Also try the historical markets endpoint
    print("\n  Checking historical/markets endpoint...")
    data = _make_request("/historical/markets", params={"limit": PAGE_LIMIT})
    if data and "markets" in data:
        hist_weather = 0
        for m in data["markets"]:
            ticker = m.get("ticker", "")
            title = (m.get("title", "") + " " + ticker).upper()
            if any(w in title for w in ["KXHIGH", "KXLOW", "TEMPERATURE", "TEMP", "HIGH", "LOW"]):
                if ticker not in discovered_tickers:
                    discovered_tickers.add(ticker)
                    _store_market(conn, m)
                    hist_weather += 1
        if hist_weather:
            print(f"    Found {hist_weather} weather markets from historical endpoint")
        # Paginate
        cursor = data.get("cursor")
        page = 1
        while cursor:
            page += 1
            data = _make_request("/historical/markets", params={
                "limit": PAGE_LIMIT,
                "cursor": cursor,
            })
            if not data or "markets" not in data or not data["markets"]:
                break
            hw = 0
            for m in data["markets"]:
                ticker = m.get("ticker", "")
                title = (m.get("title", "") + " " + ticker).upper()
                if any(w in title for w in ["KXHIGH", "KXLOW", "TEMPERATURE", "TEMP"]):
                    if ticker not in discovered_tickers:
                        discovered_tickers.add(ticker)
                        _store_market(conn, m)
                        hw += 1
            if hw:
                print(f"    Page {page}: {hw} new weather markets")
            cursor = data.get("cursor")
            if page > 100:  # safety limit
                break

    conn.commit()

    # Filter to only weather tickers
    weather_tickers = sorted([
        t for t in discovered_tickers
        if t and any(w in t.upper() for w in ["KXHIGH", "KXLOW", "HIGH", "LOW", "TEMP"])
    ])

    print(f"\n  TOTAL WEATHER MARKETS DISCOVERED: {len(weather_tickers)}")
    if weather_tickers:
        # Show breakdown by prefix pattern
        prefixes = {}
        for t in weather_tickers:
            # Extract prefix up to the date part
            match = re.match(r'^([A-Z]+)', t)
            if match:
                p = match.group(1)
                prefixes[p] = prefixes.get(p, 0) + 1
        for p, c in sorted(prefixes.items(), key=lambda x: -x[1]):
            print(f"    {p}: {c} markets")

    return weather_tickers


def _store_market(conn: sqlite3.Connection, m: dict):
    """Store or update a market record."""
    ticker = m.get("ticker", "")
    if not ticker:
        return
    conn.execute("""
        INSERT OR REPLACE INTO kalshi_markets
        (ticker, event_ticker, title, subtitle, open_time, close_time,
         settlement_time, result, yes_sub_title, no_sub_title, volume,
         open_interest, status, category, series_ticker, strike_value,
         floor_strike, cap_strike, raw_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        ticker,
        m.get("event_ticker", ""),
        m.get("title", ""),
        m.get("subtitle", ""),
        m.get("open_time", ""),
        m.get("close_time", m.get("expected_expiration_time", "")),
        m.get("settlement_time", m.get("expiration_time", "")),
        m.get("result", ""),
        m.get("yes_sub_title", ""),
        m.get("no_sub_title", ""),
        m.get("volume", 0),
        m.get("open_interest", 0),
        m.get("status", ""),
        m.get("category", ""),
        m.get("series_ticker", ""),
        m.get("strike_value", None),
        m.get("floor_strike", None),
        m.get("cap_strike", None),
        json.dumps(m),
    ))


# ---------------------------------------------------------------------------
# Trade pulling
# ---------------------------------------------------------------------------

def pull_trades_for_ticker(conn: sqlite3.Connection, ticker: str) -> int:
    """Pull all trades for a single ticker. Returns count of new trades."""
    # Check what we already have
    row = conn.execute(
        "SELECT MAX(created_time) FROM kalshi_trades WHERE ticker = ?",
        (ticker,)
    ).fetchone()
    min_ts = None
    if row and row[0]:
        # Start from last known trade time (will get some dupes, but safe)
        min_ts = row[0]

    total_new = 0
    cursor = None
    page = 0

    while True:
        page += 1
        params = {
            "ticker": ticker,
            "limit": PAGE_LIMIT,
        }
        if min_ts:
            # Convert ISO to epoch seconds if needed
            params["min_ts"] = min_ts
        if cursor:
            params["cursor"] = cursor

        data = _make_request("/markets/trades", params=params)
        if not data:
            # Try alternative endpoint
            data = _make_request(f"/markets/{ticker}/trades", params={
                "limit": PAGE_LIMIT,
                "cursor": cursor,
                "min_ts": min_ts,
            })
        if not data:
            break

        trades = data.get("trades", [])
        if not trades:
            break

        new_count = 0
        for t in trades:
            trade_id = t.get("trade_id", "")
            if not trade_id:
                continue
            try:
                conn.execute("""
                    INSERT OR IGNORE INTO kalshi_trades
                    (trade_id, ticker, yes_price, no_price, count,
                     taker_side, created_time)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    trade_id,
                    ticker,
                    t.get("yes_price", t.get("yes_price_dollars")),
                    t.get("no_price", t.get("no_price_dollars")),
                    t.get("count", t.get("count_fp", 1)),
                    t.get("taker_side", ""),
                    t.get("created_time", ""),
                ))
                new_count += 1
            except sqlite3.IntegrityError:
                pass

        total_new += new_count

        cursor = data.get("cursor")
        if not cursor or len(trades) < PAGE_LIMIT:
            break

        if page % 10 == 0:
            conn.commit()

    conn.commit()
    return total_new


def pull_all_trades(conn: sqlite3.Connection, tickers: list[str]):
    """Pull trades for all discovered tickers."""
    print("\n" + "=" * 70)
    print("PHASE 2: PULLING TRADES (tick-by-tick)")
    print("=" * 70)

    total_trades = 0
    for i, ticker in enumerate(tickers, 1):
        print(f"\n  [{i}/{len(tickers)}] {ticker}...", end="", flush=True)
        count = pull_trades_for_ticker(conn, ticker)
        total_trades += count
        print(f" {count} trades", end="", flush=True)
        if count > 0:
            print(f"  (running total: {total_trades})", end="")
        print()

    print(f"\n  TOTAL TRADES PULLED: {total_trades}")
    return total_trades


# ---------------------------------------------------------------------------
# Candlestick pulling
# ---------------------------------------------------------------------------

def pull_candles_for_ticker(conn: sqlite3.Connection, ticker: str,
                            period: int = 60) -> int:
    """Pull candlesticks for a ticker. period: 1=1min, 60=1hr, 1440=1day."""
    data = _make_request(f"/markets/{ticker}/candlesticks", params={
        "period_interval": period,
    })
    if not data:
        return 0

    candles = data.get("candlesticks", [])
    count = 0
    for c in candles:
        try:
            conn.execute("""
                INSERT OR REPLACE INTO kalshi_candles
                (ticker, timestamp, open, high, low, close, volume, period)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                ticker,
                c.get("end_period_ts", c.get("timestamp", c.get("t", ""))),
                c.get("open", c.get("price", {}).get("open")),
                c.get("high", c.get("price", {}).get("high")),
                c.get("low", c.get("price", {}).get("low")),
                c.get("close", c.get("price", {}).get("close")),
                c.get("volume", 0),
                period,
            ))
            count += 1
        except (sqlite3.IntegrityError, TypeError):
            pass

    conn.commit()
    return count


def pull_all_candles(conn: sqlite3.Connection, tickers: list[str]):
    """Pull hourly candlesticks for all tickers."""
    print("\n" + "=" * 70)
    print("PHASE 3: PULLING CANDLESTICKS (1-hour)")
    print("=" * 70)

    total_candles = 0
    for i, ticker in enumerate(tickers, 1):
        print(f"  [{i}/{len(tickers)}] {ticker}...", end="", flush=True)
        count = pull_candles_for_ticker(conn, ticker, period=60)
        total_candles += count
        if count > 0:
            print(f" {count} candles")
        else:
            print(" (no candles)")

    print(f"\n  TOTAL CANDLES PULLED: {total_candles}")
    return total_candles


# ---------------------------------------------------------------------------
# Market metadata enrichment
# ---------------------------------------------------------------------------

def enrich_market_metadata(conn: sqlite3.Connection, tickers: list[str]):
    """Fetch full market details for each ticker."""
    print("\n" + "=" * 70)
    print("PHASE 4: ENRICHING MARKET METADATA")
    print("=" * 70)

    enriched = 0
    for i, ticker in enumerate(tickers, 1):
        if i % 50 == 0:
            print(f"  [{i}/{len(tickers)}] enriched {enriched} markets...")

        data = _make_request(f"/markets/{ticker}")
        if data and "market" in data:
            m = data["market"]
            _store_market(conn, m)
            enriched += 1
        elif data:
            # Sometimes the response is the market itself
            if "ticker" in data:
                _store_market(conn, data)
                enriched += 1

    conn.commit()
    print(f"\n  ENRICHED: {enriched} markets")
    return enriched


# ---------------------------------------------------------------------------
# Summary / stats
# ---------------------------------------------------------------------------

def print_summary(conn: sqlite3.Connection):
    """Print a summary of what's in the database."""
    print("\n" + "=" * 70)
    print("DATABASE SUMMARY")
    print("=" * 70)

    row = conn.execute("SELECT COUNT(*) FROM kalshi_markets").fetchone()
    print(f"\n  Total markets: {row[0]}")

    row = conn.execute("SELECT COUNT(*) FROM kalshi_trades").fetchone()
    print(f"  Total trades:  {row[0]}")

    row = conn.execute("SELECT COUNT(*) FROM kalshi_candles").fetchone()
    print(f"  Total candles: {row[0]}")

    # Markets by series/prefix
    print("\n  Markets by ticker prefix:")
    rows = conn.execute("""
        SELECT
            CASE
                WHEN ticker LIKE 'KXHIGH%' THEN 'KXHIGH*'
                WHEN ticker LIKE 'KXLOW%' THEN 'KXLOW*'
                WHEN ticker LIKE 'HIGH%' THEN 'HIGH*'
                WHEN ticker LIKE 'LOW%' THEN 'LOW*'
                ELSE 'OTHER'
            END as prefix,
            COUNT(*) as cnt
        FROM kalshi_markets
        GROUP BY prefix
        ORDER BY cnt DESC
    """).fetchall()
    for prefix, cnt in rows:
        print(f"    {prefix}: {cnt}")

    # Markets by event_ticker (top 20)
    print("\n  Top 20 event tickers:")
    rows = conn.execute("""
        SELECT event_ticker, COUNT(*) as cnt
        FROM kalshi_markets
        WHERE event_ticker != ''
        GROUP BY event_ticker
        ORDER BY cnt DESC
        LIMIT 20
    """).fetchall()
    for et, cnt in rows:
        print(f"    {et}: {cnt} markets")

    # Trade date range
    row = conn.execute("""
        SELECT MIN(created_time), MAX(created_time)
        FROM kalshi_trades
    """).fetchone()
    if row and row[0]:
        print(f"\n  Trade date range: {row[0][:19]} to {row[1][:19]}")

    # Markets with most trades
    print("\n  Top 10 markets by trade count:")
    rows = conn.execute("""
        SELECT ticker, COUNT(*) as cnt
        FROM kalshi_trades
        GROUP BY ticker
        ORDER BY cnt DESC
        LIMIT 10
    """).fetchall()
    for t, cnt in rows:
        print(f"    {t}: {cnt} trades")

    # Florida-specific stats
    print("\n  Florida market stats:")
    for code, name in [("MIA", "Miami"), ("MI", "Miami-2"), ("FLL", "Fort Lauderdale"),
                        ("TPA", "Tampa"), ("ORL", "Orlando"), ("JAX", "Jacksonville"),
                        ("EYW", "Key West")]:
        row = conn.execute(
            "SELECT COUNT(*) FROM kalshi_markets WHERE ticker LIKE ?",
            (f"%{code}%",)
        ).fetchone()
        if row[0] > 0:
            row2 = conn.execute(
                "SELECT COUNT(*) FROM kalshi_trades WHERE ticker LIKE ?",
                (f"%{code}%",)
            ).fetchone()
            print(f"    {name} ({code}): {row[0]} markets, {row2[0]} trades")

    print(f"\n  Database file: {DB_PATH}")
    print(f"  Database size: {DB_PATH.stat().st_size / 1024 / 1024:.1f} MB")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("KALSHI HISTORICAL WEATHER DATA PULLER")
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Database: {DB_PATH}")
    print("=" * 70)

    # Test API connectivity
    print("\nTesting API connectivity...")
    for base in API_BASES:
        print(f"  Trying {base}...", end="", flush=True)
        data = _make_request("/markets", params={"limit": 1}, base_url=base)
        if data:
            print(" OK!")
            global _working_base_url
            _working_base_url = base
            break
        else:
            print(" FAILED")
    else:
        print("\nERROR: Could not connect to any Kalshi API endpoint!")
        sys.exit(1)

    print(f"\nUsing API base: {_working_base_url}")

    # Initialize database
    conn = init_db(DB_PATH)
    print(f"Database initialized: {DB_PATH}")

    try:
        # Phase 1: Discover markets
        tickers = discover_weather_markets(conn)

        if not tickers:
            print("\nNo weather markets found! Trying broader search...")
            # Fallback: just get all markets and filter
            data = _make_request("/markets", params={
                "limit": PAGE_LIMIT,
                "status": "settled",
            })
            if data and "markets" in data:
                for m in data["markets"]:
                    t = m.get("ticker", "")
                    title = (m.get("title", "") + t).upper()
                    if any(w in title for w in ["TEMP", "HIGH", "LOW", "WEATHER", "DEGREE"]):
                        tickers.append(t)
                        _store_market(conn, m)
                conn.commit()
                print(f"  Fallback found {len(tickers)} markets")

        if not tickers:
            print("\nStill no markets found. Printing sample of available markets...")
            data = _make_request("/markets", params={"limit": 20})
            if data and "markets" in data:
                for m in data["markets"]:
                    print(f"  {m.get('ticker', '?')}: {m.get('title', '?')}")
            print("\nExiting - no weather markets to pull.")
            return

        # Phase 2: Pull all trades
        pull_all_trades(conn, tickers)

        # Phase 3: Pull candlesticks
        pull_all_candles(conn, tickers)

        # Phase 4: Enrich metadata
        enrich_market_metadata(conn, tickers)

        # Summary
        print_summary(conn)

    finally:
        conn.close()

    print(f"\nCompleted: {datetime.now().isoformat()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
