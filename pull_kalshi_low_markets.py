#!/usr/bin/env python3
"""
Pull all KXLOWTMIA (and other city LOW) markets from Kalshi API into kalshi_markets table.
Targets both KXLOWTMIA (bracket style) and KXLOWMIA (binary style) series.
"""

import json
import sqlite3
import time
import urllib.parse
import urllib.request
from pathlib import Path

DB_PATH = Path(__file__).parent / "miami_collector.db"
BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
MIN_DELAY = 0.12  # ~8 req/s

# All LOW weather series to pull
LOW_SERIES = [
    "KXLOWTMIA",   # Miami LOW bracket (primary target)
    "KXLOWMIA",    # Miami LOW binary
    "KXLOWTNYC",
    "KXLOWNYC",
    "KXLOWTCHI",
    "KXLOWCHI",
    "KXLOWTDEN",
    "KXLOWDEN",
    "KXLOWTAUS",
    "KXLOWAUS",
    "KXLOWTLAX",
    "KXLOWLAX",
    "KXLOWTPHIL",
    "KXLOWPHIL",
]

_last_req = 0.0


def _get(path, params=None):
    global _last_req
    now = time.time()
    wait = MIN_DELAY - (now - _last_req)
    if wait > 0:
        time.sleep(wait)
    _last_req = time.time()

    url = BASE_URL + path
    if params:
        url += "?" + urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})

    for attempt in range(5):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "MiamiWeatherTrader/1.0", "Accept": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=20) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(2 ** (attempt + 1))
                continue
            elif e.code == 404:
                return None
            else:
                time.sleep(2 ** attempt)
                continue
        except Exception:
            time.sleep(2 ** attempt)
    return None


def pull_series(conn, series_ticker):
    """Pull all markets for a series and upsert into kalshi_markets."""
    print(f"  Pulling {series_ticker}...", end="", flush=True)
    cursor = None
    total = 0
    inserted = 0

    while True:
        params = {"series_ticker": series_ticker, "limit": 1000}
        if cursor:
            params["cursor"] = cursor

        data = _get("/markets", params)
        if not data:
            break

        markets = data.get("markets", [])
        if not markets:
            break

        rows = []
        for m in markets:
            ticker = m.get("ticker", "")
            if not ticker:
                continue

            # Extract strike values
            floor_strike = m.get("floor_strike")
            cap_strike = m.get("cap_strike")

            # Extract result: normalize to 'yes'/'no'/''
            result = m.get("result", "")
            if result is None:
                result = ""

            rows.append((
                ticker,
                m.get("event_ticker", ""),
                m.get("title", ""),
                m.get("yes_sub_title", ""),
                m.get("open_time", ""),
                m.get("close_time", ""),
                m.get("expiration_time", ""),
                result,
                m.get("yes_sub_title", ""),
                m.get("no_sub_title", ""),
                int(float(m.get("volume_fp", 0) or 0)),
                int(float(m.get("open_interest_fp", 0) or 0)),
                m.get("status", ""),
                "temperature",
                series_ticker,
                floor_strike,
                floor_strike,   # floor_strike used as strike_value too
                cap_strike,
                json.dumps(m),
            ))

        if rows:
            conn.executemany("""
                INSERT OR REPLACE INTO kalshi_markets
                (ticker, event_ticker, title, subtitle, open_time, close_time,
                 settlement_time, result, yes_sub_title, no_sub_title,
                 volume, open_interest, status, category, series_ticker,
                 strike_value, floor_strike, cap_strike, raw_json)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, rows)
            conn.commit()
            inserted += len(rows)
            total += len(markets)

        cursor = data.get("cursor")
        if not cursor:
            break

    print(f" {total} markets, {inserted} upserted")
    return inserted


def main():
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    print("=" * 60)
    print("  Pulling Kalshi LOW temperature markets")
    print("=" * 60)

    total = 0
    for series in LOW_SERIES:
        n = pull_series(conn, series)
        total += n

    print(f"\nTotal upserted: {total}")

    # Verify - check how many settled LOW MIA markets we got
    cur = conn.execute(
        "SELECT COUNT(*) FROM kalshi_markets WHERE event_ticker LIKE '%LOWTMIA%' OR event_ticker LIKE '%LOWMIA%'"
    )
    print(f"MIA LOW markets in DB: {cur.fetchone()[0]}")

    # Check settled ones with result='yes'
    cur = conn.execute(
        "SELECT COUNT(DISTINCT event_ticker) FROM kalshi_markets "
        "WHERE (event_ticker LIKE '%LOWTMIA%' OR event_ticker LIKE '%LOWMIA%') "
        "AND result='yes'"
    )
    print(f"MIA LOW settled events (result=yes): {cur.fetchone()[0]}")

    conn.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
