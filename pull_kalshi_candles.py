#!/usr/bin/env python3
"""
Pull hourly candlestick data from Kalshi API for all weather markets.
Uses the series endpoint: GET /trade-api/v2/series/{series}/markets/{ticker}/candlesticks

Batches DB writes to minimize contention with other processes using the DB.
"""

import sqlite3
import requests
import time
import sys
from datetime import datetime, timezone

DB_PATH = "/Users/blakebot/blakebot/miami-project/miami_collector.db"
BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
HEADERS = {"Accept": "application/json"}

# Rate limiting
MIN_DELAY = 0.22  # ~4.5 req/s
HOURS_PER_CHUNK = 4999  # API max is 5000 candles per request
WRITE_BATCH_SIZE = 200  # Write to DB every N markets


def extract_series_ticker(event_ticker: str) -> str:
    parts = event_ticker.split("-")
    return parts[0] if parts else ""


def create_tables(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS kalshi_candles_hourly (
            ticker TEXT,
            timestamp TEXT,
            yes_bid_open REAL, yes_bid_high REAL, yes_bid_low REAL, yes_bid_close REAL,
            yes_ask_open REAL, yes_ask_high REAL, yes_ask_low REAL, yes_ask_close REAL,
            price_open REAL, price_high REAL, price_low REAL, price_close REAL, price_mean REAL,
            volume REAL,
            open_interest REAL,
            PRIMARY KEY (ticker, timestamp)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS kalshi_forecast_percentiles (
            event_ticker TEXT,
            timestamp TEXT,
            p5 REAL, p10 REAL, p25 REAL, p50 REAL, p75 REAL, p90 REAL, p95 REAL,
            PRIMARY KEY (event_ticker, timestamp)
        )
    """)
    conn.commit()


def parse_dollars(val):
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def flush_candles(conn, pending_rows):
    """Write accumulated candle rows to DB with retry."""
    if not pending_rows:
        return 0
    for attempt in range(10):
        try:
            conn.executemany("""
                INSERT OR IGNORE INTO kalshi_candles_hourly
                (ticker, timestamp,
                 yes_bid_open, yes_bid_high, yes_bid_low, yes_bid_close,
                 yes_ask_open, yes_ask_high, yes_ask_low, yes_ask_close,
                 price_open, price_high, price_low, price_close, price_mean,
                 volume, open_interest)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, pending_rows)
            conn.commit()
            return len(pending_rows)
        except sqlite3.OperationalError as e:
            if "locked" in str(e) and attempt < 9:
                wait = min(0.5 * (2 ** attempt), 30)
                print(f"  DB locked on flush ({len(pending_rows)} rows), retry {attempt+1}/10 in {wait:.1f}s", flush=True)
                time.sleep(wait)
            else:
                raise
    return 0


def fetch_market_candles(ticker, series_ticker, open_time, close_time, session, stats):
    """Fetch all hourly candles for a single market. Returns list of row tuples."""
    now_ts = int(time.time())

    if open_time:
        try:
            start_ts = int(datetime.fromisoformat(open_time.replace("Z", "+00:00")).timestamp())
        except Exception:
            start_ts = now_ts - (30 * 24 * 3600)
    else:
        start_ts = now_ts - (30 * 24 * 3600)

    if close_time:
        try:
            end_ts = int(datetime.fromisoformat(close_time.replace("Z", "+00:00")).timestamp())
        except Exception:
            end_ts = now_ts
    else:
        end_ts = now_ts

    end_ts = min(end_ts, now_ts)
    if start_ts >= end_ts:
        return []

    all_rows = []
    chunk_start = start_ts
    backoff = 3

    while chunk_start < end_ts:
        chunk_end = min(chunk_start + HOURS_PER_CHUNK * 3600, end_ts)

        url = f"{BASE_URL}/series/{series_ticker}/markets/{ticker}/candlesticks"
        params = {"period_interval": 60, "start_ts": chunk_start, "end_ts": chunk_end}

        try:
            r = session.get(url, headers=HEADERS, params=params, timeout=30)
        except requests.exceptions.RequestException as e:
            stats["errors"] += 1
            return all_rows
        stats["requests"] += 1

        if r.status_code == 200:
            backoff = 3
            candles = r.json().get("candlesticks", [])
            for c in candles:
                ts = c.get("end_period_ts")
                if ts is None:
                    continue
                ts_str = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")
                price = c.get("price", {})
                yes_bid = c.get("yes_bid", {})
                yes_ask = c.get("yes_ask", {})
                all_rows.append((
                    ticker, ts_str,
                    parse_dollars(yes_bid.get("open_dollars")),
                    parse_dollars(yes_bid.get("high_dollars")),
                    parse_dollars(yes_bid.get("low_dollars")),
                    parse_dollars(yes_bid.get("close_dollars")),
                    parse_dollars(yes_ask.get("open_dollars")),
                    parse_dollars(yes_ask.get("high_dollars")),
                    parse_dollars(yes_ask.get("low_dollars")),
                    parse_dollars(yes_ask.get("close_dollars")),
                    parse_dollars(price.get("open_dollars")),
                    parse_dollars(price.get("high_dollars")),
                    parse_dollars(price.get("low_dollars")),
                    parse_dollars(price.get("close_dollars")),
                    parse_dollars(price.get("mean_dollars")),
                    parse_dollars(c.get("volume_fp")),
                    parse_dollars(c.get("open_interest_fp")),
                ))
            chunk_start = chunk_end

        elif r.status_code == 404:
            stats["skipped_404"] += 1
            return all_rows

        elif r.status_code == 429:
            stats["rate_limited"] += 1
            print(f"  429 rate limited, backing off {backoff}s", flush=True)
            time.sleep(backoff)
            backoff = min(backoff * 2, 60)
            continue

        elif r.status_code == 400:
            stats["skipped_400"] += 1
            return all_rows

        else:
            stats["errors"] += 1
            return all_rows

        time.sleep(MIN_DELAY)

    return all_rows


def main():
    test_mode = "--test" in sys.argv

    # Read-only connection for initial queries
    conn_read = sqlite3.connect(DB_PATH, timeout=30)
    conn_read.execute("PRAGMA busy_timeout=30000")

    cur = conn_read.cursor()

    # Ensure tables exist
    conn_write = sqlite3.connect(DB_PATH, timeout=30)
    conn_write.execute("PRAGMA busy_timeout=30000")
    conn_write.execute("PRAGMA journal_mode=WAL")
    create_tables(conn_write)
    conn_write.close()

    # Get all weather markets ordered by close_time DESC
    cur.execute("""
        SELECT ticker, event_ticker, open_time, close_time
        FROM kalshi_markets
        WHERE ticker LIKE '%HIGH%' OR ticker LIKE '%LOW%'
        ORDER BY close_time DESC
    """)
    markets = cur.fetchall()
    total = len(markets)

    # Get already-fetched tickers
    cur.execute("SELECT DISTINCT ticker FROM kalshi_candles_hourly")
    already_fetched = set(r[0] for r in cur.fetchall())
    conn_read.close()

    if test_mode:
        markets = markets[:5]
        print(f"TEST MODE: processing {len(markets)} of {total} markets", flush=True)
    else:
        print(f"Processing {total} weather markets, {len(already_fetched)} already fetched", flush=True)

    session = requests.Session()
    stats = {
        "requests": 0,
        "candles_inserted": 0,
        "markets_with_data": 0,
        "skipped_404": 0,
        "skipped_400": 0,
        "skipped_existing": 0,
        "rate_limited": 0,
        "errors": 0,
    }

    start_time = time.time()
    pending_rows = []
    markets_since_flush = 0

    for i, (ticker, event_ticker, open_time, close_time) in enumerate(markets):
        series_ticker = extract_series_ticker(event_ticker)
        if not series_ticker:
            continue

        if ticker in already_fetched:
            stats["skipped_existing"] += 1
            continue

        rows = fetch_market_candles(ticker, series_ticker, open_time, close_time, session, stats)
        if rows:
            pending_rows.extend(rows)
            stats["markets_with_data"] += 1

        markets_since_flush += 1

        # Flush to DB periodically
        if markets_since_flush >= WRITE_BATCH_SIZE or (i + 1) == len(markets):
            if pending_rows:
                conn_w = sqlite3.connect(DB_PATH, timeout=30)
                conn_w.execute("PRAGMA busy_timeout=30000")
                conn_w.execute("PRAGMA journal_mode=WAL")
                written = flush_candles(conn_w, pending_rows)
                conn_w.close()
                stats["candles_inserted"] += written
                pending_rows = []
            markets_since_flush = 0

        if (i + 1) % 100 == 0 or (test_mode and i == len(markets) - 1):
            elapsed = time.time() - start_time
            rate = stats["requests"] / elapsed if elapsed > 0 else 0
            print(
                f"[{i+1}/{len(markets)}] "
                f"candles={stats['candles_inserted']:,}+{len(pending_rows)} "
                f"w_data={stats['markets_with_data']} "
                f"404={stats['skipped_404']} "
                f"400={stats['skipped_400']} "
                f"429={stats['rate_limited']} "
                f"err={stats['errors']} "
                f"skip={stats['skipped_existing']} "
                f"rate={rate:.1f}/s "
                f"{elapsed:.0f}s",
                flush=True,
            )

    # Final flush
    if pending_rows:
        conn_w = sqlite3.connect(DB_PATH, timeout=30)
        conn_w.execute("PRAGMA busy_timeout=30000")
        conn_w.execute("PRAGMA journal_mode=WAL")
        written = flush_candles(conn_w, pending_rows)
        conn_w.close()
        stats["candles_inserted"] += written

    # Forecast percentiles (tested: returns 404 for all weather events, but try a few)
    print(f"\n--- Forecast Percentiles ---", flush=True)
    conn_r2 = sqlite3.connect(DB_PATH, timeout=30)
    conn_r2.execute("PRAGMA busy_timeout=30000")
    cur2 = conn_r2.cursor()
    cur2.execute("""
        SELECT DISTINCT event_ticker
        FROM kalshi_markets
        WHERE ticker LIKE '%HIGH%' OR ticker LIKE '%LOW%'
        ORDER BY event_ticker DESC
    """)
    event_tickers = [r[0] for r in cur2.fetchall()]
    conn_r2.close()

    if test_mode:
        event_tickers = event_tickers[:5]

    print(f"Trying {len(event_tickers)} event tickers...", flush=True)
    forecast_found = 0
    forecast_404 = 0

    for i, evt in enumerate(event_tickers):
        url = f"{BASE_URL}/events/{evt}/forecast/percentile_history"
        try:
            r = session.get(url, headers=HEADERS, timeout=30)
        except requests.exceptions.RequestException:
            continue
        stats["requests"] += 1

        if r.status_code == 200:
            data = r.json()
            history = data.get("history", data.get("percentile_history", []))
            if history:
                forecast_found += 1
                print(f"  Found {len(history)} percentile records for {evt}", flush=True)
                rows = []
                for entry in history:
                    ts = entry.get("timestamp", entry.get("ts", ""))
                    rows.append((evt, ts,
                                 entry.get("p5"), entry.get("p10"), entry.get("p25"),
                                 entry.get("p50"), entry.get("p75"), entry.get("p90"),
                                 entry.get("p95")))
                if rows:
                    conn_w = sqlite3.connect(DB_PATH, timeout=30)
                    conn_w.execute("PRAGMA busy_timeout=30000")
                    conn_w.execute("PRAGMA journal_mode=WAL")
                    conn_w.executemany("""
                        INSERT OR IGNORE INTO kalshi_forecast_percentiles
                        (event_ticker, timestamp, p5, p10, p25, p50, p75, p90, p95)
                        VALUES (?,?,?,?,?,?,?,?,?)
                    """, rows)
                    conn_w.commit()
                    conn_w.close()
        else:
            forecast_404 += 1

        if forecast_404 > 20 and forecast_found == 0:
            print(f"  No forecast data after {forecast_404} attempts, stopping.", flush=True)
            break

        time.sleep(MIN_DELAY)
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(event_tickers)}] found={forecast_found} 404s={forecast_404}", flush=True)

    elapsed = time.time() - start_time
    print(f"\n=== DONE ===", flush=True)
    print(f"Elapsed: {elapsed:.0f}s", flush=True)
    print(f"Total requests: {stats['requests']}", flush=True)
    print(f"Candles inserted: {stats['candles_inserted']:,}", flush=True)
    print(f"Markets with data: {stats['markets_with_data']}", flush=True)
    print(f"Skipped existing: {stats['skipped_existing']}", flush=True)
    print(f"Skipped 404: {stats['skipped_404']}", flush=True)
    print(f"Skipped 400: {stats['skipped_400']}", flush=True)
    print(f"Rate limited 429: {stats['rate_limited']}", flush=True)
    print(f"Errors: {stats['errors']}", flush=True)
    print(f"Forecast events with data: {forecast_found}", flush=True)


if __name__ == "__main__":
    main()
