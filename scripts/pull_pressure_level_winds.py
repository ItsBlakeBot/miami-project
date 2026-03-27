#!/usr/bin/env python3
"""
Pull 925/850 hPa pressure level wind and temperature data from Open-Meteo
historical forecast API for Miami.

Stores hourly data in pressure_level_winds table.
Coverage: 2020-01-01 to 2026-03-27.
Chunks requests by 3-month intervals to stay within API limits.
"""

import sqlite3
import time
from datetime import datetime, timedelta

import requests

DB_PATH = "/Users/blakebot/blakebot/miami-project/miami_collector.db"

LATITUDE = 25.79
LONGITUDE = -80.29

# Historical forecast API for past data
HIST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
# Regular forecast API for recent/future data
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

HOURLY_VARS = (
    "temperature_925hPa,windspeed_925hPa,winddirection_925hPa,"
    "temperature_850hPa,windspeed_850hPa,winddirection_850hPa"
)

START_DATE = datetime(2020, 1, 1)
END_DATE = datetime(2026, 3, 27)

# Chunk size in days (3 months)
CHUNK_DAYS = 90


def create_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS pressure_level_winds (
            timestamp_utc TEXT PRIMARY KEY,
            temp_925_c REAL,
            wind_speed_925_ms REAL,
            wind_dir_925 REAL,
            temp_850_c REAL,
            wind_speed_850_ms REAL,
            wind_dir_850 REAL
        )
    """)
    conn.commit()


def get_existing_timestamps(conn):
    cursor = conn.execute("SELECT timestamp_utc FROM pressure_level_winds")
    return {row[0] for row in cursor.fetchall()}


def fetch_chunk(session, start_date, end_date, use_forecast=False):
    """Fetch one chunk of pressure level data from Open-Meteo."""
    url = FORECAST_URL if use_forecast else HIST_URL
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": HOURLY_VARS,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "wind_speed_unit": "ms",
        "timezone": "UTC",
    }

    r = session.get(url, params=params, timeout=120)
    if r.status_code != 200:
        print(f"  HTTP {r.status_code}: {r.text[:200]}")
        return []

    data = r.json()
    if "error" in data:
        print(f"  API error: {data['error']}")
        return []

    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    if not times:
        return []

    rows = []
    for i, t in enumerate(times):
        # Convert "2024-01-01T00:00" to "2024-01-01T00:00:00Z"
        ts = t + ":00Z" if len(t) == 16 else t + "Z"

        row = (
            ts,
            hourly.get("temperature_925hPa", [None] * len(times))[i],
            hourly.get("windspeed_925hPa", [None] * len(times))[i],
            hourly.get("winddirection_925hPa", [None] * len(times))[i],
            hourly.get("temperature_850hPa", [None] * len(times))[i],
            hourly.get("windspeed_850hPa", [None] * len(times))[i],
            hourly.get("winddirection_850hPa", [None] * len(times))[i],
        )
        rows.append(row)

    return rows


def main():
    conn = sqlite3.connect(DB_PATH)
    create_table(conn)

    existing = get_existing_timestamps(conn)
    print(f"Existing rows in pressure_level_winds: {len(existing)}")

    session = requests.Session()
    session.headers.update({
        "User-Agent": "MiamiWeatherResearch/1.0 (academic research)",
    })

    total_inserted = 0
    total_skipped = 0
    t0 = time.time()

    # The historical forecast API may not have the very latest data.
    # For recent dates, we'll use the regular forecast API.
    # Cutoff: data older than 5 days ago uses historical, newer uses forecast.
    forecast_cutoff = datetime.utcnow() - timedelta(days=5)

    current = START_DATE
    while current < END_DATE:
        chunk_end = min(current + timedelta(days=CHUNK_DAYS), END_DATE)
        use_forecast = current >= forecast_cutoff

        chunk_t0 = time.time()
        rows = fetch_chunk(session, current, chunk_end, use_forecast=use_forecast)
        fetch_time = time.time() - chunk_t0

        # Filter out existing
        new_rows = [r for r in rows if r[0] not in existing]
        skipped = len(rows) - len(new_rows)

        if new_rows:
            conn.executemany(
                "INSERT OR REPLACE INTO pressure_level_winds VALUES (?,?,?,?,?,?,?)",
                new_rows,
            )
            conn.commit()
            # Add to existing set
            for r in new_rows:
                existing.add(r[0])

        total_inserted += len(new_rows)
        total_skipped += skipped

        src = "forecast" if use_forecast else "historical"
        print(
            f"  {current.strftime('%Y-%m-%d')} -> {chunk_end.strftime('%Y-%m-%d')} [{src}]: "
            f"{len(rows)} hours fetched in {fetch_time:.1f}s, "
            f"{len(new_rows)} inserted, {skipped} skipped"
        )

        current = chunk_end + timedelta(days=1)
        time.sleep(0.3)  # Be polite to the API

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s ({elapsed/60:.1f}m)")
    print(f"Total inserted: {total_inserted}")
    print(f"Total skipped: {total_skipped}")

    # Final stats
    cursor = conn.execute("SELECT COUNT(*) FROM pressure_level_winds")
    print(f"\nTotal rows in pressure_level_winds: {cursor.fetchone()[0]}")

    cursor = conn.execute(
        "SELECT MIN(timestamp_utc), MAX(timestamp_utc) FROM pressure_level_winds"
    )
    row = cursor.fetchone()
    print(f"Date range: {row[0]} to {row[1]}")

    # Sample data
    cursor = conn.execute(
        "SELECT * FROM pressure_level_winds ORDER BY timestamp_utc LIMIT 3"
    )
    print("\nSample rows:")
    for row in cursor.fetchall():
        print(f"  {row}")

    conn.close()


if __name__ == "__main__":
    import functools
    print = functools.partial(print, flush=True)
    main()
