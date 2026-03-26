#!/usr/bin/env python3
"""
Pull hourly candlestick data from Kalshi API for all weather markets.
Uses the series endpoint: GET /trade-api/v2/series/{series}/markets/{ticker}/candlesticks
"""

import sqlite3
import requests
import time
import re
import sys
from datetime import datetime

DB_PATH = "/Users/blakebot/blakebot/miami-project/miami_collector.db"
BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
HEADERS = {"Accept": "application/json"}

# Rate limiting: target 5 req/s => 0.2s between requests
MIN_DELAY = 0.22
MAX_CANDLES_PER_REQUEST = 4999  # API limit is 5000
HOURS_PER_CHUNK = MAX_CANDLES_PER_REQUEST  # 1 candle per hour


def extract_series_ticker(event_ticker: str) -> str:
    """Extract series ticker from event ticker.
    KXHIGHDEN-26MAR27 -> KXHIGHDEN
    KXLOWMIA-25DEC01 -> KXLOWMIA
    """
    parts = event_ticker.split("-")
    if parts:
        return parts[0]
    return ""


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
    """Parse dollar string like '0.1000' to float, return None if missing."""
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def fetch_candles(ticker, series_ticker, start_ts, end_ts, session):
    """Fetch candlesticks for a ticker within a time range."""
    url = f"{BASE_URL}/series/{series_ticker}/markets/{ticker}/candlesticks"
    params = {
        "period_interval": 60,
        "start_ts": start_ts,
        "end_ts": end_ts,
    }
    r = session.get(url, headers=HEADERS, params=params, timeout=30)
    return r


def process_market(ticker, series_ticker, open_time, close_time, conn, session, stats):
    """Fetch all hourly candles for a single market."""
    # Determine time range from market open to close (or now if still active)
    now_ts = int(time.time())

    if open_time:
        try:
            start_ts = int(datetime.fromisoformat(open_time.replace("Z", "+00:00")).timestamp())
        except Exception:
            start_ts = now_ts - (30 * 24 * 3600)  # default 30 days back
    else:
        start_ts = now_ts - (30 * 24 * 3600)

    if close_time:
        try:
            end_ts = int(datetime.fromisoformat(close_time.replace("Z", "+00:00")).timestamp())
        except Exception:
            end_ts = now_ts
    else:
        end_ts = now_ts

    # Don't go past now
    end_ts = min(end_ts, now_ts)

    if start_ts >= end_ts:
        return

    total_candles = 0
    chunk_start = start_ts
    backoff = 3

    while chunk_start < end_ts:
        chunk_end = min(chunk_start + HOURS_PER_CHUNK * 3600, end_ts)

        r = fetch_candles(ticker, series_ticker, chunk_start, chunk_end, session)
        stats["requests"] += 1

        if r.status_code == 200:
            backoff = 3  # reset backoff on success
            data = r.json()
            candles = data.get("candlesticks", [])

            rows = []
            for c in candles:
                ts = c.get("end_period_ts")
                if ts is None:
                    continue
                ts_str = datetime.utcfromtimestamp(ts).isoformat() + "Z"

                price = c.get("price", {})
                yes_bid = c.get("yes_bid", {})
                yes_ask = c.get("yes_ask", {})

                rows.append((
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

            if rows:
                conn.executemany("""
                    INSERT OR IGNORE INTO kalshi_candles_hourly
                    (ticker, timestamp,
                     yes_bid_open, yes_bid_high, yes_bid_low, yes_bid_close,
                     yes_ask_open, yes_ask_high, yes_ask_low, yes_ask_close,
                     price_open, price_high, price_low, price_close, price_mean,
                     volume, open_interest)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, rows)
                conn.commit()
                total_candles += len(rows)

            chunk_start = chunk_end

        elif r.status_code == 404:
            stats["skipped_404"] += 1
            return  # No data for this market

        elif r.status_code == 429:
            stats["rate_limited"] += 1
            print(f"  429 rate limited, backing off {backoff}s")
            time.sleep(backoff)
            backoff = min(backoff * 2, 60)
            continue  # retry same chunk

        elif r.status_code == 400:
            # Bad request - skip this market
            stats["skipped_400"] += 1
            return

        else:
            print(f"  Unexpected {r.status_code} for {ticker}: {r.text[:200]}")
            stats["errors"] += 1
            return

        time.sleep(MIN_DELAY)

    stats["candles_inserted"] += total_candles
    if total_candles > 0:
        stats["markets_with_data"] += 1


def fetch_forecast_percentiles(event_ticker, session, conn, stats):
    """Try to fetch forecast percentile history for an event."""
    url = f"{BASE_URL}/events/{event_ticker}/forecast/percentile_history"
    r = session.get(url, headers=HEADERS, timeout=30)
    stats["requests"] += 1

    if r.status_code == 200:
        data = r.json()
        history = data.get("history", data.get("percentile_history", []))
        if not history:
            return 0

        rows = []
        for entry in history:
            ts = entry.get("timestamp", entry.get("ts", ""))
            rows.append((
                event_ticker, ts,
                entry.get("p5"), entry.get("p10"), entry.get("p25"),
                entry.get("p50"), entry.get("p75"), entry.get("p90"),
                entry.get("p95"),
            ))

        if rows:
            conn.executemany("""
                INSERT OR IGNORE INTO kalshi_forecast_percentiles
                (event_ticker, timestamp, p5, p10, p25, p50, p75, p90, p95)
                VALUES (?,?,?,?,?,?,?,?,?)
            """, rows)
            conn.commit()
        return len(rows)

    elif r.status_code == 429:
        stats["rate_limited"] += 1
        time.sleep(3)
        return -1  # signal retry

    return 0


def main():
    test_mode = "--test" in sys.argv

    conn = sqlite3.connect(DB_PATH)
    create_tables(conn)

    # Get all weather markets ordered by close_time DESC
    cur = conn.cursor()
    cur.execute("""
        SELECT ticker, event_ticker, open_time, close_time
        FROM kalshi_markets
        WHERE ticker LIKE '%HIGH%' OR ticker LIKE '%LOW%'
        ORDER BY close_time DESC
    """)
    markets = cur.fetchall()
    total = len(markets)

    if test_mode:
        markets = markets[:5]
        print(f"TEST MODE: processing {len(markets)} of {total} markets")
    else:
        print(f"Processing {total} weather markets")

    session = requests.Session()
    stats = {
        "requests": 0,
        "candles_inserted": 0,
        "markets_with_data": 0,
        "skipped_404": 0,
        "skipped_400": 0,
        "rate_limited": 0,
        "errors": 0,
    }

    start_time = time.time()

    for i, (ticker, event_ticker, open_time, close_time) in enumerate(markets):
        series_ticker = extract_series_ticker(event_ticker)
        if not series_ticker:
            continue

        process_market(ticker, series_ticker, open_time, close_time, conn, session, stats)

        if (i + 1) % 100 == 0 or (test_mode and i == len(markets) - 1):
            elapsed = time.time() - start_time
            rate = stats["requests"] / elapsed if elapsed > 0 else 0
            print(
                f"[{i+1}/{len(markets)}] "
                f"candles={stats['candles_inserted']:,} "
                f"markets_w_data={stats['markets_with_data']} "
                f"404s={stats['skipped_404']} "
                f"400s={stats['skipped_400']} "
                f"429s={stats['rate_limited']} "
                f"errors={stats['errors']} "
                f"reqs={stats['requests']} "
                f"rate={rate:.1f}/s "
                f"elapsed={elapsed:.0f}s"
            )

    # Now try forecast percentiles for distinct event tickers
    cur.execute("""
        SELECT DISTINCT event_ticker
        FROM kalshi_markets
        WHERE ticker LIKE '%HIGH%' OR ticker LIKE '%LOW%'
        ORDER BY event_ticker DESC
    """)
    event_tickers = [r[0] for r in cur.fetchall()]

    if test_mode:
        event_tickers = event_tickers[:5]

    print(f"\n--- Forecast Percentiles ---")
    print(f"Trying {len(event_tickers)} event tickers...")

    forecast_found = 0
    forecast_404 = 0
    for i, evt in enumerate(event_tickers):
        result = fetch_forecast_percentiles(evt, session, conn, stats)
        if result > 0:
            forecast_found += 1
            print(f"  Found {result} percentile records for {evt}")
        elif result == 0:
            forecast_404 += 1
        # result == -1 means retry (429), but we'll just move on

        if forecast_404 > 20 and forecast_found == 0:
            print(f"  No forecast data found after {forecast_404} attempts, stopping.")
            break

        time.sleep(MIN_DELAY)

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(event_tickers)}] found={forecast_found} 404s={forecast_404}")

    elapsed = time.time() - start_time
    print(f"\n=== DONE ===")
    print(f"Elapsed: {elapsed:.0f}s")
    print(f"Total requests: {stats['requests']}")
    print(f"Candles inserted: {stats['candles_inserted']:,}")
    print(f"Markets with candle data: {stats['markets_with_data']}")
    print(f"Skipped (404): {stats['skipped_404']}")
    print(f"Skipped (400): {stats['skipped_400']}")
    print(f"Rate limited (429): {stats['rate_limited']}")
    print(f"Errors: {stats['errors']}")
    print(f"Forecast events with data: {forecast_found}")

    conn.close()


if __name__ == "__main__":
    main()
