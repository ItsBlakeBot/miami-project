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
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone

DB_PATH = "/Users/blakebot/blakebot/miami-project/miami_collector.db"
GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"
RATE_LIMIT_DELAY = 0.55  # ~2 requests/sec with margin


def api_get(url, retries=3):
    """GET request with retries and rate limiting."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url)
            req.add_header("Accept", "application/json")
            req.add_header("User-Agent", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36")
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
            time.sleep(RATE_LIMIT_DELAY)
            return data
        except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError) as e:
            if attempt < retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"  [retry {attempt+1}] {e} — waiting {wait}s")
                time.sleep(wait)
            else:
                print(f"  [FAILED] {url[:100]}... : {e}")
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
    "Miami", "NYC", "New York", "Chicago", "Dallas", "Atlanta", "Houston",
    "Los Angeles", "LA", "San Francisco", "Denver", "Seattle", "Boston",
    "Phoenix", "Detroit", "Minneapolis", "Philadelphia", "Washington",
    "London", "Paris", "Tokyo", "Berlin", "Sydney", "Melbourne",
    "Toronto", "Montreal", "Vancouver", "Mexico City", "Sao Paulo",
    "São Paulo", "Buenos Aires", "Mumbai", "Delhi", "Shanghai",
    "Beijing", "Seoul", "Taipei", "Hong Kong", "Singapore",
    "Bangkok", "Jakarta", "Shenzhen", "Guangzhou", "Osaka",
    "Rome", "Madrid", "Barcelona", "Amsterdam", "Stockholm",
    "Oslo", "Copenhagen", "Helsinki", "Zurich", "Vienna",
    "Warsaw", "Prague", "Dublin", "Lisbon", "Istanbul",
    "Cairo", "Lagos", "Johannesburg", "Nairobi", "Cape Town",
    "Auckland", "Lima", "Bogota", "Santiago", "Riyadh",
    "Dubai", "Tel Aviv", "Athens", "Brussels", "Hanoi",
    "Kuala Lumpur",
]


def extract_city(question, event_title=""):
    """Extract city name from market question or event title."""
    text = event_title or question
    for city in sorted(KNOWN_CITIES, key=len, reverse=True):
        if city.lower() in text.lower():
            return city
    # Try pattern: "in <City> on"
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
    """Filter to only temperature markets (not hurricanes, snow, etc)."""
    text = (question + " " + description + " " + event_title).lower()
    temp_keywords = ["temperature", "temp", "°f", "°c", "degrees"]
    exclude_keywords = ["hurricane", "snow", "rain", "wind", "tornado",
                        "storm", "flood", "wildfire", "earthquake"]
    has_temp = any(kw in text for kw in temp_keywords)
    has_exclude = any(kw in text for kw in exclude_keywords)
    return has_temp and not has_exclude


def discover_markets(conn):
    """Discover all temperature markets from the Gamma API."""
    print("=" * 60)
    print("PHASE 1: Discovering temperature markets")
    print("=" * 60)

    all_markets = []
    offset = 0
    page_size = 100
    total_events = 0
    total_skipped = 0

    while True:
        url = f"{GAMMA_API}/events?tag_slug=temperature&limit={page_size}&offset={offset}"
        print(f"\n  Fetching events offset={offset}...")
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
                    total_skipped += 1
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

    print(f"\n  Total events scanned: {total_events}")
    print(f"  Non-temperature skipped: {total_skipped}")
    print(f"  Temperature markets found: {len(all_markets)}")

    # Store in DB
    inserted = 0
    updated = 0
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
            inserted += 1
        except sqlite3.Error as e:
            print(f"  DB error for market {m['market_id']}: {e}")

    conn.commit()
    print(f"  Saved to DB: {inserted} markets")

    # Summary by city
    cities = {}
    for m in all_markets:
        c = m["city"]
        cities[c] = cities.get(c, 0) + 1
    print("\n  Markets by city:")
    for city, count in sorted(cities.items(), key=lambda x: -x[1])[:20]:
        print(f"    {city}: {count}")

    return all_markets


CHUNK_DAYS = 14  # API max range is ~15 days, use 14 for safety


def fetch_token_history(token, start_ts, end_ts):
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
    """Pull hourly price history for each market."""
    print("\n" + "=" * 60)
    print("PHASE 2: Pulling price history")
    print("=" * 60)

    now_ts = int(time.time())
    total = len(markets)
    markets_with_data = 0
    total_points = 0
    skipped = 0

    for i, mkt in enumerate(markets):
        market_id = mkt["market_id"]
        token_yes = mkt["clob_token_yes"]
        token_no = mkt["clob_token_no"]

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

        # Check if we already have recent data for this market
        existing = conn.execute(
            "SELECT MAX(timestamp) FROM polymarket_price_history WHERE market_id = ?",
            (market_id,)
        ).fetchone()
        if existing and existing[0]:
            try:
                last_ts = int(datetime.fromisoformat(existing[0].replace("Z", "+00:00")).timestamp())
                if last_ts > now_ts - 3600:
                    skipped += 1
                    continue
                start_ts = max(start_ts, last_ts + 1)
            except (ValueError, TypeError):
                pass

        if (i + 1) % 100 == 0 or i == 0:
            print(f"\n  [{i+1}/{total}] Pulling: {mkt['question'][:60]}...")

        # Pull YES and NO token history with chunking
        yes_history = fetch_token_history(token_yes, start_ts, now_ts)
        no_history = fetch_token_history(token_no, start_ts, now_ts)

        # Merge YES and NO history
        all_timestamps = sorted(set(list(yes_history.keys()) + list(no_history.keys())))

        if all_timestamps:
            markets_with_data += 1
            points_inserted = 0
            for ts_str in all_timestamps:
                yes_p = yes_history.get(ts_str)
                no_p = no_history.get(ts_str)
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

        # Commit every 100 markets
        if (i + 1) % 100 == 0:
            conn.commit()
            print(f"    Progress: {markets_with_data} with data, {total_points} total points, {skipped} skipped")

    conn.commit()
    print(f"\n  Markets with price data: {markets_with_data}/{total}")
    print(f"  Total price points stored: {total_points}")
    print(f"  Skipped (already up to date): {skipped}")


def main():
    print("Polymarket Temperature Market Data Puller")
    print(f"Database: {DB_PATH}")
    print(f"Started: {datetime.now().isoformat()}")
    print()

    conn = sqlite3.connect(DB_PATH)
    init_db(conn)

    # Phase 1: Discover markets
    markets = discover_markets(conn)

    if not markets:
        print("\nNo temperature markets found!")
        conn.close()
        return

    # Phase 2: Pull price history
    pull_price_history(conn, markets)

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    row = conn.execute("SELECT COUNT(*) FROM polymarket_weather_markets").fetchone()
    print(f"  Total markets in DB: {row[0]}")
    row = conn.execute("SELECT COUNT(*) FROM polymarket_price_history").fetchone()
    print(f"  Total price history rows: {row[0]}")
    row = conn.execute("SELECT COUNT(DISTINCT market_id) FROM polymarket_price_history").fetchone()
    print(f"  Markets with price data: {row[0]}")
    row = conn.execute("""
        SELECT city, COUNT(*) as cnt
        FROM polymarket_weather_markets
        GROUP BY city ORDER BY cnt DESC LIMIT 10
    """).fetchall()
    print("  Top cities:")
    for city, cnt in row:
        print(f"    {city}: {cnt} markets")

    conn.close()
    print(f"\nCompleted: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
