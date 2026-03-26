#!/usr/bin/env python3
"""
Pull HRRR sub-hourly (15-minute resolution) forecast data from Open-Meteo
and store in miami_collector.db hrrr_subhourly table.

Open-Meteo provides minutely_15 HRRR data via:
  Historical: https://historical-forecast-api.open-meteo.com/v1/forecast
  Recent:     https://api.open-meteo.com/v1/forecast

Pulls 2024-01-01 to present in 90-day chunks.
Each day has 96 data points (every 15 minutes).
"""

import sqlite3
import json
import time
import sys
import urllib.request
import urllib.error
from datetime import datetime, timedelta, date

DB_PATH = "/Users/blakebot/blakebot/miami-project/miami_collector.db"

# Miami / KMIA grid point
LAT = 25.79
LON = -80.29

HISTORICAL_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
RECENT_URL = "https://api.open-meteo.com/v1/forecast"

MINUTELY_15_VARS = [
    "temperature_2m",
    "dew_point_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "surface_pressure",
    "cape",
    "cloud_cover",
    "visibility",
]

# Map Open-Meteo variable names to our DB column names
VAR_TO_COL = {
    "temperature_2m": "temperature_2m",
    "dew_point_2m": "dewpoint_2m",
    "wind_speed_10m": "wind_speed_10m",
    "wind_direction_10m": "wind_direction_10m",
    "surface_pressure": "surface_pressure",
    "cape": "cape",
    "cloud_cover": "cloud_cover",
    "visibility": "visibility",
}


def create_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS hrrr_subhourly (
            station TEXT NOT NULL DEFAULT 'KMIA',
            model_run_utc TEXT,
            valid_time_utc TEXT NOT NULL,
            forecast_minute INTEGER NOT NULL,
            temperature_2m REAL,
            dewpoint_2m REAL,
            wind_speed_10m REAL,
            wind_direction_10m REAL,
            surface_pressure REAL,
            cape REAL,
            cloud_cover REAL,
            visibility REAL,
            PRIMARY KEY (station, valid_time_utc)
        )
    """)
    conn.commit()


def fetch_chunk(start_date: str, end_date: str, use_recent: bool = False) -> dict:
    """Fetch one date range from Open-Meteo minutely_15."""
    base_url = RECENT_URL if use_recent else HISTORICAL_URL
    params = {
        "latitude": LAT,
        "longitude": LON,
        "minutely_15": ",".join(MINUTELY_15_VARS),
        "models": "ncep_hrrr_conus",
        "timezone": "GMT",
    }
    if use_recent:
        params["past_days"] = 5
        params["forecast_days"] = 0
    else:
        params["start_date"] = start_date
        params["end_date"] = end_date

    query = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"{base_url}?{query}"

    for attempt in range(3):
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode())
            if "error" in data and data["error"]:
                print(f"  API error: {data.get('reason', 'unknown')}")
                return None
            return data
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = 30 * (attempt + 1)
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"  HTTP error {e.code}: {e.reason}")
                if attempt < 2:
                    time.sleep(5)
        except Exception as e:
            print(f"  Error: {e}")
            if attempt < 2:
                time.sleep(5)
    return None


def insert_data(conn, data: dict) -> int:
    """Insert Open-Meteo minutely_15 response into hrrr_subhourly. Returns rows inserted."""
    m15 = data.get("minutely_15", {})
    times = m15.get("time", [])
    if not times:
        return 0

    cols = [
        "station", "model_run_utc", "valid_time_utc", "forecast_minute",
        "temperature_2m", "dewpoint_2m", "wind_speed_10m",
        "wind_direction_10m", "surface_pressure", "cape",
        "cloud_cover", "visibility",
    ]
    placeholders = ",".join(["?"] * len(cols))
    col_str = ",".join(cols)

    inserted = 0
    for i, t in enumerate(times):
        # t is like "2024-01-01T00:15"
        valid_time_utc = t.replace("T", " ") + ":00"

        # Parse the minute to get forecast_minute (0, 15, 30, 45)
        minute = int(t.split(":")[-1])

        # model_run_utc is the enclosing hour (minute 0 -> same hour, else same hour)
        # For sub-hourly, the model run is the top of the hour
        hour_part = t[:14] + "00"  # e.g. "2024-01-01T00:00"
        model_run_utc = hour_part.replace("T", " ") + ":00"

        row = {
            "station": "KMIA",
            "model_run_utc": model_run_utc,
            "valid_time_utc": valid_time_utc,
            "forecast_minute": minute,
        }
        for var_name, col_name in VAR_TO_COL.items():
            vals = m15.get(var_name, [])
            row[col_name] = vals[i] if i < len(vals) else None

        # Skip rows where all weather values are None
        weather_vals = [row.get(c) for c in cols[4:]]
        if all(v is None for v in weather_vals):
            continue

        values = [row.get(c) for c in cols]
        try:
            conn.execute(
                f"INSERT OR IGNORE INTO hrrr_subhourly ({col_str}) VALUES ({placeholders})",
                values,
            )
            inserted += 1
        except sqlite3.Error as e:
            print(f"  DB error: {e}")

    conn.commit()
    return inserted


def get_existing_date_range(conn):
    """Get min/max dates already in the table."""
    try:
        row = conn.execute(
            "SELECT MIN(valid_time_utc), MAX(valid_time_utc), COUNT(*) FROM hrrr_subhourly WHERE station='KMIA'"
        ).fetchone()
        return row
    except sqlite3.OperationalError:
        return None, None, 0


def get_existing_dates(conn) -> set:
    """Get set of dates that already have data."""
    try:
        rows = conn.execute(
            "SELECT DISTINCT substr(valid_time_utc, 1, 10) FROM hrrr_subhourly WHERE station='KMIA'"
        ).fetchall()
        return {r[0] for r in rows}
    except sqlite3.OperationalError:
        return set()


def main():
    conn = sqlite3.connect(DB_PATH)
    create_table(conn)

    # Date range: 2024-01-01 to today
    start = date(2024, 1, 1)
    end = date.today()

    existing_dates = get_existing_dates(conn)
    info = get_existing_date_range(conn)
    print(f"Existing sub-hourly data: {info[2]} rows, {len(existing_dates)} days")
    if info[0]:
        print(f"  Range: {info[0]} to {info[1]}")

    # Process in 90-day chunks
    chunk_days = 90
    current = start
    total_inserted = 0
    chunks_processed = 0

    while current <= end:
        chunk_end = min(current + timedelta(days=chunk_days - 1), end)
        start_str = current.strftime("%Y-%m-%d")
        end_str = chunk_end.strftime("%Y-%m-%d")

        # Check if this chunk is fully covered (all days present with 96 points each)
        chunk_dates = set()
        d = current
        while d <= chunk_end:
            chunk_dates.add(d.strftime("%Y-%m-%d"))
            d += timedelta(days=1)

        missing = chunk_dates - existing_dates
        if not missing:
            print(f"  [{start_str} to {end_str}] fully covered, skipping")
            current = chunk_end + timedelta(days=1)
            continue

        # Use historical API for dates more than 5 days ago, recent for last 5 days
        days_ago = (date.today() - chunk_end).days
        use_recent = days_ago < 2

        if use_recent:
            print(f"Fetching recent data [{start_str} to {end_str}] via forecast API...")
            data = fetch_chunk("", "", use_recent=True)
        else:
            print(f"Fetching [{start_str} to {end_str}] ({len(missing)} missing days)...")
            data = fetch_chunk(start_str, end_str)

        if data:
            n = insert_data(conn, data)
            total_inserted += n
            print(f"  Inserted {n} rows")
        else:
            print(f"  FAILED to fetch chunk")

        chunks_processed += 1
        current = chunk_end + timedelta(days=1)

        # Rate limit: 1 req/sec for Open-Meteo free tier
        if chunks_processed % 5 == 0:
            print("  Pausing 5s for rate limiting...")
            time.sleep(5)
        else:
            time.sleep(1)

    # Also fetch recent data to fill the last few days
    print("\nFetching recent data via forecast API...")
    data = fetch_chunk("", "", use_recent=True)
    if data:
        n = insert_data(conn, data)
        total_inserted += n
        print(f"  Inserted {n} recent rows")

    # Summary
    total = conn.execute("SELECT COUNT(*) FROM hrrr_subhourly WHERE station='KMIA'").fetchone()[0]
    min_dt = conn.execute("SELECT MIN(valid_time_utc) FROM hrrr_subhourly WHERE station='KMIA'").fetchone()[0]
    max_dt = conn.execute("SELECT MAX(valid_time_utc) FROM hrrr_subhourly WHERE station='KMIA'").fetchone()[0]
    print(f"\n=== Summary ===")
    print(f"Total rows in hrrr_subhourly: {total}")
    print(f"Date range: {min_dt} to {max_dt}")
    print(f"Rows inserted this run: {total_inserted}")
    print(f"Expected rows (2+ years * 96/day): ~{(end - start).days * 96}")

    conn.close()


if __name__ == "__main__":
    main()
