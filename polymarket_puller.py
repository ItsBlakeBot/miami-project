#!/usr/bin/env python3
"""
Polymarket Weather Temperature Market Data Puller

Discovers temperature prediction markets via the Gamma API,
then pulls hourly price history from the CLOB API.
Stores everything in the miami_collector.db SQLite database.
"""

import json
import re
import sqlite3
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone

DB_PATH = "/Users/blakebot/blakebot/miami-project/miami_collector.db"
GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"
RATE_LIMIT_DELAY = 0.55  # ~2 requests/sec with margin
CHUNK_DAYS = 14  # API max range is ~15 days
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"


def log(msg):
    print(msg)
    sys.stdout.flush()


def api_get(url, retries=3):
    """GET request with retries and rate limiting."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url)
            req.add_header("Accept", "application/json")
            req.add_header("User-Agent", UA)
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
            time.sleep(RATE_LIMIT_DELAY)
            return data
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = 5 * (attempt + 1)
                log(f"  [rate limited] waiting {wait}s...")
                time.sleep(wait)
            elif e.code == 400:
                # Bad request - not retryable (e.g. invalid token)
                return None
            elif attempt < retries - 1:
                time.sleep(2 ** (attempt + 1))
            else:
                return None
        except (urllib.error.URLError, json.JSONDecodeError) as e:
            if attempt < retries - 1:
                time.sleep(2 ** (attempt + 1))
            else:
                return None


def init_db(conn):
    """Create tables if they don't exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS polymarket_weather_markets (
            market_id TEXT PRIMARY KEY,
            question TEXT,
            description TEXT,
            city TEXT,
            market_type TEXT,
            resolution_date TEXT,
            outcome TEXT,
            volume REAL,
            liquidity REAL,
            event_id TEXT,
            event_title TEXT,
            condition_id TEXT,
            clob_token_yes TEXT,
            clob_token_no TEXT,
            created_at TEXT,
            closed INTEGER DEFAULT 0
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS polymarket_price_history (
            market_id TEXT,
            timestamp TEXT,
            yes_price REAL,
            no_price REAL,
            PRIMARY KEY (market_id, timestamp)
        )
    """)
    conn.commit()


# --- City extraction ---
KNOWN_CITIES = [
    "Miami", "NYC", "New York City", "New York", "Chicago", "Dallas",
    "Atlanta", "Houston", "Los Angeles", "San Francisco", "Denver",
    "Seattle", "Boston", "Phoenix", "Detroit", "Minneapolis",
    "Philadelphia", "Washington", "London", "Paris", "Tokyo", "Berlin",
    "Sydney", "Melbourne", "Toronto", "Montreal", "Vancouver",
    "Mexico City", "Sao Paulo", "São Paulo", "Buenos Aires",
    "Mumbai", "Delhi", "Shanghai", "Beijing", "Seoul", "Taipei",
    "Hong Kong", "Singapore", "Bangkok", "Jakarta", "Shenzhen",
    "Guangzhou", "Osaka", "Rome", "Madrid", "Barcelona", "Amsterdam",
    "Stockholm", "Oslo", "Copenhagen", "Helsinki", "Zurich", "Vienna",
    "Warsaw", "Prague", "Dublin", "Lisbon", "Istanbul", "Cairo",
    "Lagos", "Johannesburg", "Nairobi", "Cape Town", "Auckland",
    "Lima", "Bogota", "Santiago", "Riyadh", "Dubai", "Tel Aviv",
    "Athens", "Brussels", "Hanoi", "Kuala Lumpur", "Ankara",
    "Wellington", "Lucknow", "Munich",
]


def extract_city(question, event_title=""):
    """Extract city name from market question or event title."""
    text = event_title or question
    for city in sorted(KNOWN_CITIES, key=len, reverse=True):
        if city.lower() in text.lower():
            return city
    m = re.search(r"in\s+([A-Z][a-zA-Z\s]+?)\s+(?:on|be)", text)
    if m:
        return m.group(1).strip()
    return "Unknown"


def extract_market_type(question, event_title=""):
    """Determine if this is a high or low temperature market."""
    text = (question + " " + event_title).lower()
    if "highest" in text or "high" in text:
        return "high"
    elif "lowest" in text or "low" in text:
        return "low"
    return "temperature"


def is_temperature_market(question, description="", event_title=""):
    """Filter to only temperature markets."""
    text = (question + " " + description + " " + event_title).lower()
    temp_keywords = ["temperature", "temp", "°f", "°c", "degrees"]
    exclude_keywords = ["hurricane", "snow", "rain", "wind", "tornado",
                        "storm", "flood", "wildfire", "earthquake"]
    has_temp = any(kw in text for kw in temp_keywords)
    has_exclude = any(kw in text for kw in exclude_keywords)
    return has_temp and not has_exclude


def discover_markets(conn):
    """Discover all temperature markets from the Gamma API."""
    log("=" * 60)
    log("PHASE 1: Discovering temperature markets")
    log("=" * 60)

    all_markets = []
    offset = 0
    page_size = 100
    total_events = 0

    while True:
        url = f"{GAMMA_API}/events?tag_slug=temperature&limit={page_size}&offset={offset}"
        log(f"  Fetching events offset={offset}...")
        events = api_get(url)

        if events is None or len(events) == 0:
            break

        total_events += len(events)

        for event in events:
            event_id = str(event.get("id", ""))
            event_title = event.get("title", "")
            markets = event.get("markets", [])

            for mkt in markets:
                question = mkt.get("question", "")
                description = mkt.get("description", "")

                if not is_temperature_market(question, description, event_title):
                    continue

                clob_ids = json.loads(mkt.get("clobTokenIds", "[]"))
                if len(clob_ids) < 2:
                    continue

                market_data = {
                    "market_id": str(mkt.get("id", "")),
                    "question": question,
                    "description": description[:500] if description else "",
                    "city": extract_city(question, event_title),
                    "market_type": extract_market_type(question, event_title),
                    "resolution_date": mkt.get("endDate", ""),
                    "outcome": mkt.get("outcomePrices", ""),
                    "volume": float(mkt.get("volume", 0) or 0),
                    "liquidity": float(mkt.get("liquidity", 0) or 0),
                    "event_id": event_id,
                    "event_title": event_title,
                    "condition_id": mkt.get("conditionId", ""),
                    "clob_token_yes": clob_ids[0],
                    "clob_token_no": clob_ids[1],
                    "created_at": mkt.get("createdAt", ""),
                    "closed": 1 if mkt.get("closed") else 0,
                }
                all_markets.append(market_data)

        if len(events) < page_size:
            break
        offset += page_size

    log(f"\n  Total events scanned: {total_events}")
    log(f"  Temperature markets found: {len(all_markets)}")

    # Store in DB
    for m in all_markets:
        try:
            conn.execute("""
                INSERT INTO polymarket_weather_markets
                (market_id, question, description, city, market_type,
                 resolution_date, outcome, volume, liquidity,
                 event_id, event_title, condition_id,
                 clob_token_yes, clob_token_no, created_at, closed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(market_id) DO UPDATE SET
                    outcome = excluded.outcome,
                    volume = excluded.volume,
                    liquidity = excluded.liquidity,
                    closed = excluded.closed
            """, (
                m["market_id"], m["question"], m["description"],
                m["city"], m["market_type"], m["resolution_date"],
                m["outcome"], m["volume"], m["liquidity"],
                m["event_id"], m["event_title"], m["condition_id"],
                m["clob_token_yes"], m["clob_token_no"],
                m["created_at"], m["closed"],
            ))
        except sqlite3.Error as e:
            log(f"  DB error for market {m['market_id']}: {e}")

    conn.commit()
    log(f"  Saved to DB: {len(all_markets)} markets")

    # Summary by city
    cities = {}
    for m in all_markets:
        c = m["city"]
        cities[c] = cities.get(c, 0) + 1
    log("\n  Markets by city:")
    for city, count in sorted(cities.items(), key=lambda x: -x[1])[:20]:
        log(f"    {city}: {count}")

    return all_markets


def fetch_token_history_chunked(token, start_ts, end_ts):
    """Fetch price history for a token, chunking into 14-day windows."""
    history = {}
    chunk_size = CHUNK_DAYS * 86400

    chunk_start = start_ts
    while chunk_start < end_ts:
        chunk_end = min(chunk_start + chunk_size, end_ts)
        url = (
            f"{CLOB_API}/prices-history"
            f"?market={token}"
            f"&startTs={chunk_start}&endTs={chunk_end}"
            f"&interval=1h&fidelity=1"
        )
        data = api_get(url)
        if data and "history" in data:
            for pt in data["history"]:
                ts_str = datetime.fromtimestamp(pt["t"], tz=timezone.utc).isoformat()
                history[ts_str] = pt["p"]
        chunk_start = chunk_end

    return history


def pull_price_history(conn, markets):
    """Pull hourly price history for markets that have trading volume."""
    log("\n" + "=" * 60)
    log("PHASE 2: Pulling price history")
    log("=" * 60)

    # Only pull for markets with volume > 0 (saves massive API calls)
    markets_with_vol = [m for m in markets if m["volume"] > 0]
    log(f"  Markets with volume > 0: {len(markets_with_vol)} / {len(markets)}")

    # Sort by volume descending so we get the most important ones first
    markets_with_vol.sort(key=lambda x: -x["volume"])

    now_ts = int(time.time())
    total = len(markets_with_vol)
    markets_with_data = 0
    total_points = 0
    skipped = 0
    api_calls = 0

    for i, mkt in enumerate(markets_with_vol):
        market_id = mkt["market_id"]
        token_yes = mkt["clob_token_yes"]

        # Parse creation time for start timestamp
        created = mkt.get("created_at", "")
        if created:
            try:
                dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
                start_ts = int(dt.timestamp())
            except (ValueError, TypeError):
                start_ts = now_ts - 14 * 86400
        else:
            start_ts = now_ts - 14 * 86400

        # For closed markets, use resolution date as end (don't scan to now)
        end_ts = now_ts
        resolution = mkt.get("resolution_date", "")
        if mkt.get("closed") and resolution:
            try:
                res_dt = datetime.fromisoformat(resolution.replace("Z", "+00:00"))
                end_ts = min(int(res_dt.timestamp()) + 86400, now_ts)  # +1 day buffer
            except (ValueError, TypeError):
                pass

        # Check if we already have data for this market
        existing = conn.execute(
            "SELECT MAX(timestamp) FROM polymarket_price_history WHERE market_id = ?",
            (market_id,)
        ).fetchone()
        if existing and existing[0]:
            try:
                last_ts = int(datetime.fromisoformat(
                    existing[0].replace("Z", "+00:00")
                ).timestamp())
                if last_ts > end_ts - 3600:
                    skipped += 1
                    continue
                start_ts = max(start_ts, last_ts + 1)
            except (ValueError, TypeError):
                pass

        if (i + 1) % 200 == 0 or i == 0:
            log(f"\n  [{i+1}/{total}] {mkt['question'][:65]}...")
            log(f"    vol=${mkt['volume']:.0f}, city={mkt['city']}")

        # Pull YES token history only (NO = 1 - YES)
        yes_history = fetch_token_history_chunked(token_yes, start_ts, end_ts)

        # Count API calls for this market
        time_span = end_ts - start_ts
        chunks = max(1, (time_span + CHUNK_DAYS * 86400 - 1) // (CHUNK_DAYS * 86400))
        api_calls += chunks

        if yes_history:
            markets_with_data += 1
            points_inserted = 0
            for ts_str, yes_p in yes_history.items():
                no_p = round(1.0 - yes_p, 6) if yes_p is not None else None
                try:
                    conn.execute("""
                        INSERT OR REPLACE INTO polymarket_price_history
                        (market_id, timestamp, yes_price, no_price)
                        VALUES (?, ?, ?, ?)
                    """, (market_id, ts_str, yes_p, no_p))
                    points_inserted += 1
                except sqlite3.Error:
                    pass
            total_points += points_inserted

        # Commit and report every 100 markets
        if (i + 1) % 100 == 0:
            conn.commit()
            log(f"    === Progress: {i+1}/{total} processed, "
                f"{markets_with_data} with data, {total_points} points, "
                f"~{api_calls} API calls ===")

    conn.commit()
    log(f"\n  Markets with price data: {markets_with_data}/{total}")
    log(f"  Total price points stored: {total_points}")
    log(f"  Skipped (already up to date): {skipped}")
    log(f"  Approximate API calls made: {api_calls}")


def main():
    log("Polymarket Temperature Market Data Puller")
    log(f"Database: {DB_PATH}")
    log(f"Started: {datetime.now().isoformat()}")
    log("")

    conn = sqlite3.connect(DB_PATH)
    init_db(conn)

    # Phase 1: Discover markets
    markets = discover_markets(conn)

    if not markets:
        log("\nNo temperature markets found!")
        conn.close()
        return

    # Phase 2: Pull price history
    pull_price_history(conn, markets)

    # Final summary
    log("\n" + "=" * 60)
    log("FINAL SUMMARY")
    log("=" * 60)
    row = conn.execute("SELECT COUNT(*) FROM polymarket_weather_markets").fetchone()
    log(f"  Total markets in DB: {row[0]}")
    row = conn.execute("SELECT COUNT(*) FROM polymarket_price_history").fetchone()
    log(f"  Total price history rows: {row[0]}")
    row = conn.execute("SELECT COUNT(DISTINCT market_id) FROM polymarket_price_history").fetchone()
    log(f"  Markets with price data: {row[0]}")

    row = conn.execute("""
        SELECT city, COUNT(*) as cnt
        FROM polymarket_weather_markets
        GROUP BY city ORDER BY cnt DESC LIMIT 10
    """).fetchall()
    log("  Top cities:")
    for city, cnt in row:
        log(f"    {city}: {cnt} markets")

    row = conn.execute("""
        SELECT city, COUNT(DISTINCT ph.market_id) as cnt, SUM(1) as pts
        FROM polymarket_price_history ph
        JOIN polymarket_weather_markets m ON ph.market_id = m.market_id
        GROUP BY city ORDER BY pts DESC LIMIT 10
    """).fetchall()
    log("  Top cities by price data:")
    for city, cnt, pts in row:
        log(f"    {city}: {cnt} markets, {pts} price points")

    conn.close()
    log(f"\nCompleted: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
