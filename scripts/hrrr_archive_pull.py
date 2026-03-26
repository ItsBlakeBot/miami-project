#!/usr/bin/env python3
"""
Pull HRRR historical analysis data from Open-Meteo Historical Forecast API
and store in miami_collector.db hrrr_archive table.

Open-Meteo provides hourly HRRR data going back to ~2020 via:
https://historical-forecast-api.open-meteo.com/v1/forecast

For recent data (last few days), uses:
https://api.open-meteo.com/v1/forecast
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

HOURLY_VARS = [
    "temperature_2m",
    "dew_point_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "surface_pressure",
    "cape",
    "total_column_integrated_water_vapour",
    "cloud_cover",
    "shortwave_radiation",
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
    "total_column_integrated_water_vapour": "precipitable_water",
    "cloud_cover": "cloud_cover",
    "shortwave_radiation": "shortwave_radiation",
    "visibility": "visibility",
}


def create_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS hrrr_archive (
            station TEXT NOT NULL DEFAULT 'KMIA',
            model_run_utc TEXT NOT NULL,
            forecast_hour INTEGER NOT NULL DEFAULT 0,
            temperature_2m REAL,
            dewpoint_2m REAL,
            wind_speed_10m REAL,
            wind_direction_10m REAL,
            surface_pressure REAL,
            cape REAL,
            precipitable_water REAL,
            cloud_cover REAL,
            shortwave_radiation REAL,
            visibility REAL,
            PRIMARY KEY (station, model_run_utc, forecast_hour)
        )
    """)
    conn.commit()


def fetch_chunk(start_date: str, end_date: str, use_recent: bool = False) -> dict:
    """Fetch one date range from Open-Meteo."""
    base_url = RECENT_URL if use_recent else HISTORICAL_URL
    params = {
        "latitude": LAT,
        "longitude": LON,
        "hourly": ",".join(HOURLY_VARS),
        "models": "ncep_hrrr_conus",
        "timezone": "GMT",
    }
    if use_recent:
        # For recent API, use past_days
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
    """Insert Open-Meteo response data into hrrr_archive. Returns rows inserted."""
    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    if not times:
        return 0

    rows = []
    for i, t in enumerate(times):
        # Convert "2024-01-01T00:00" to ISO format with seconds
        model_run_utc = t.replace("T", " ") + ":00"
        row = {
            "station": "KMIA",
            "model_run_utc": model_run_utc,
            "forecast_hour": 0,
        }
        for var_name, col_name in VAR_TO_COL.items():
            vals = hourly.get(var_name, [])
            row[col_name] = vals[i] if i < len(vals) else None
        rows.append(row)

    cols = [
        "station", "model_run_utc", "forecast_hour",
        "temperature_2m", "dewpoint_2m", "wind_speed_10m",
        "wind_direction_10m", "surface_pressure", "cape",
        "precipitable_water", "cloud_cover", "shortwave_radiation", "visibility",
    ]
    placeholders = ",".join(["?"] * len(cols))
    col_str = ",".join(cols)

    inserted = 0
    for row in rows:
        # Skip rows where all weather values are None
        weather_vals = [row.get(c) for c in cols[3:]]
        if all(v is None for v in weather_vals):
            continue
        values = [row.get(c) for c in cols]
        try:
            conn.execute(
                f"INSERT OR REPLACE INTO hrrr_archive ({col_str}) VALUES ({placeholders})",
                values,
            )
            inserted += 1
        except sqlite3.Error as e:
            print(f"  DB error: {e}")
    conn.commit()
    return inserted


def get_existing_coverage(conn) -> set:
    """Get set of dates that already have data."""
    try:
        rows = conn.execute(
            "SELECT DISTINCT substr(model_run_utc, 1, 10) FROM hrrr_archive WHERE station='KMIA'"
        ).fetchall()
        return {r[0] for r in rows}
    except sqlite3.OperationalError:
        return set()


def main():
    conn = sqlite3.connect(DB_PATH)
    create_table(conn)

    # Determine date range
    # Open-Meteo HRRR historical goes back to ~2020-08
    start = date(2020, 8, 1)
    end = date.today() - timedelta(days=2)  # historical API lags ~2 days

    existing = get_existing_coverage(conn)
    print(f"Existing coverage: {len(existing)} days already in DB")

    # Process in 3-month chunks (90 days) to avoid huge responses
    chunk_days = 90
    current = start
    total_inserted = 0
    chunks_processed = 0

    while current <= end:
        chunk_end = min(current + timedelta(days=chunk_days - 1), end)
        start_str = current.strftime("%Y-%m-%d")
        end_str = chunk_end.strftime("%Y-%m-%d")

        # Check if this chunk is fully covered
        chunk_dates = set()
        d = current
        while d <= chunk_end:
            chunk_dates.add(d.strftime("%Y-%m-%d"))
            d += timedelta(days=1)

        missing = chunk_dates - existing
        if not missing:
            print(f"  [{start_str} to {end_str}] fully covered, skipping")
            current = chunk_end + timedelta(days=1)
            continue

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

        # Rate limit: be nice to Open-Meteo (free tier)
        if chunks_processed % 5 == 0:
            print("  Pausing 5s for rate limiting...")
            time.sleep(5)
        else:
            time.sleep(1)

    # Also fetch recent data via the forecast API
    print("\nFetching recent data via forecast API...")
    data = fetch_chunk("", "", use_recent=True)
    if data:
        n = insert_data(conn, data)
        total_inserted += n
        print(f"  Inserted {n} recent rows")

    # Summary
    total = conn.execute("SELECT COUNT(*) FROM hrrr_archive WHERE station='KMIA'").fetchone()[0]
    min_date = conn.execute("SELECT MIN(model_run_utc) FROM hrrr_archive WHERE station='KMIA'").fetchone()[0]
    max_date = conn.execute("SELECT MAX(model_run_utc) FROM hrrr_archive WHERE station='KMIA'").fetchone()[0]
    print(f"\n=== Summary ===")
    print(f"Total rows in hrrr_archive: {total}")
    print(f"Date range: {min_date} to {max_date}")
    print(f"Rows inserted this run: {total_inserted}")

    conn.close()


if __name__ == "__main__":
    main()
