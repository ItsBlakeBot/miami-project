"""Pull historical NWP forecast archives from Open-Meteo for DS3M training.

Fetches archived model forecasts (GFS, ECMWF, best_match) for 5 cities
from 2020-01-01 to 2026-03-26 and stores them in nwp_forecast_archive table.

Chunks requests into 90-day windows with 1s rate limiting.

Usage:
    python nwp_forecast_puller.py
"""

from __future__ import annotations

import logging
import sqlite3
import time
from datetime import datetime, timedelta

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DB_PATH = "/Users/blakebot/blakebot/miami-project/miami_collector.db"
API_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"

START_DATE = "2020-01-01"
END_DATE = "2026-03-26"

CITIES = {
    "Miami":   {"lat": 25.79, "lon": -80.29},
    "NYC":     {"lat": 40.78, "lon": -73.97},
    "Chicago": {"lat": 41.79, "lon": -87.75},
    "Denver":  {"lat": 39.86, "lon": -104.67},
    "Austin":  {"lat": 30.19, "lon": -97.67},
}

HOURLY_VARS = (
    "temperature_2m,dewpoint_2m,windspeed_10m,winddirection_10m,"
    "cape,surface_pressure,cloudcover,shortwave_radiation"
)

MODELS = ["gfs_seamless", "ecmwf_ifs", "best_match"]


def ensure_table(conn: sqlite3.Connection):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS nwp_forecast_archive (
            city TEXT NOT NULL,
            model TEXT NOT NULL,
            timestamp_utc TEXT NOT NULL,
            temperature_2m REAL,
            dewpoint_2m REAL,
            windspeed_10m REAL,
            winddirection_10m REAL,
            cape REAL,
            surface_pressure REAL,
            cloudcover REAL,
            shortwave_radiation REAL,
            PRIMARY KEY (city, model, timestamp_utc)
        )
    """)
    conn.commit()


def get_existing_count(conn: sqlite3.Connection, city: str, model: str) -> int:
    row = conn.execute(
        "SELECT COUNT(*) FROM nwp_forecast_archive WHERE city=? AND model=?",
        (city, model)
    ).fetchone()
    return row[0] if row else 0


def fetch_chunk(lat: float, lon: float, start: str, end: str, model: str) -> dict | None:
    """Fetch one 90-day chunk from Open-Meteo Historical Forecast API."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": HOURLY_VARS,
        "timezone": "UTC",
    }
    # best_match uses default (no model param), others specify model
    if model != "best_match":
        params["models"] = model

    try:
        resp = requests.get(API_URL, params=params, timeout=120)
        if resp.status_code == 429:
            log.warning("  Rate limited, waiting 30s...")
            time.sleep(30)
            resp = requests.get(API_URL, params=params, timeout=120)
        if resp.status_code != 200:
            log.warning(f"  HTTP {resp.status_code}: {resp.text[:200]}")
            return None
        return resp.json()
    except Exception as e:
        log.error(f"  Request error: {e}")
        return None


def parse_hourly(data: dict) -> list[dict]:
    """Parse Open-Meteo hourly response into records."""
    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    records = []

    for i, ts in enumerate(times):
        def val(key):
            arr = hourly.get(key, [])
            return arr[i] if i < len(arr) and arr[i] is not None else None

        records.append({
            "timestamp_utc": ts.replace("T", " "),
            "temperature_2m": val("temperature_2m"),
            "dewpoint_2m": val("dewpoint_2m"),
            "windspeed_10m": val("windspeed_10m") or val("wind_speed_10m"),
            "winddirection_10m": val("winddirection_10m") or val("wind_direction_10m"),
            "cape": val("cape"),
            "surface_pressure": val("surface_pressure"),
            "cloudcover": val("cloudcover") or val("cloud_cover"),
            "shortwave_radiation": val("shortwave_radiation"),
        })

    return records


def store_records(conn: sqlite3.Connection, city: str, model: str, records: list[dict]) -> int:
    inserted = 0
    for r in records:
        try:
            conn.execute(
                """INSERT OR IGNORE INTO nwp_forecast_archive
                   (city, model, timestamp_utc, temperature_2m, dewpoint_2m,
                    windspeed_10m, winddirection_10m, cape, surface_pressure,
                    cloudcover, shortwave_radiation)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (city, model, r["timestamp_utc"], r["temperature_2m"],
                 r["dewpoint_2m"], r["windspeed_10m"], r["winddirection_10m"],
                 r["cape"], r["surface_pressure"], r["cloudcover"],
                 r["shortwave_radiation"])
            )
            inserted += 1
        except sqlite3.IntegrityError:
            pass
    conn.commit()
    return inserted


def generate_chunks(start: str, end: str, chunk_days: int = 90):
    """Generate (start_date, end_date) tuples for chunked requests."""
    sd = datetime.strptime(start, "%Y-%m-%d")
    ed = datetime.strptime(end, "%Y-%m-%d")
    while sd < ed:
        ce = min(sd + timedelta(days=chunk_days - 1), ed)
        yield sd.strftime("%Y-%m-%d"), ce.strftime("%Y-%m-%d")
        sd = ce + timedelta(days=1)


def main():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    ensure_table(conn)

    total_all = 0

    for city_name, coords in CITIES.items():
        for model in MODELS:
            existing = get_existing_count(conn, city_name, model)
            log.info(f"\n{'='*60}")
            log.info(f"City: {city_name} | Model: {model} | Existing rows: {existing:,}")
            log.info(f"{'='*60}")

            city_model_total = 0
            chunks = list(generate_chunks(START_DATE, END_DATE))

            for chunk_idx, (cs, ce) in enumerate(chunks):
                log.info(f"  Chunk {chunk_idx+1}/{len(chunks)}: {cs} -> {ce}")

                data = fetch_chunk(coords["lat"], coords["lon"], cs, ce, model)
                if data is None:
                    log.warning(f"  Skipping chunk {cs} -> {ce}")
                    time.sleep(2)
                    continue

                records = parse_hourly(data)
                if not records:
                    log.warning(f"  No records in chunk")
                    time.sleep(1)
                    continue

                inserted = store_records(conn, city_name, model, records)
                city_model_total += inserted
                log.info(f"    -> {inserted} new rows ({len(records)} total in chunk)")

                time.sleep(1.0)  # rate limit

            log.info(f"  {city_name}/{model}: {city_model_total:,} new rows inserted")
            total_all += city_model_total

    conn.close()
    log.info(f"\n{'='*60}")
    log.info(f"NWP FORECAST ARCHIVE COMPLETE: {total_all:,} total new rows")
    log.info(f"{'='*60}")


if __name__ == "__main__":
    main()
