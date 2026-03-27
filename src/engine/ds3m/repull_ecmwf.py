"""Re-pull ECMWF IFS forecast data for Miami using correct model name 'ecmwf_ifs'.

The old model name 'ecmwf_ifs04' returned all NULLs from the Open-Meteo API.
This script pulls Miami-only data with the correct 'ecmwf_ifs' model name.
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

LAT = 25.79
LON = -80.29
CITY = "Miami"
MODEL = "ecmwf_ifs"

HOURLY_VARS = (
    "temperature_2m,dewpoint_2m,windspeed_10m,winddirection_10m,"
    "cape,surface_pressure,cloudcover,shortwave_radiation"
)


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


def fetch_chunk(start: str, end: str) -> dict | None:
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": start,
        "end_date": end,
        "hourly": HOURLY_VARS,
        "timezone": "UTC",
        "models": MODEL,
    }
    for attempt in range(3):
        try:
            resp = requests.get(API_URL, params=params, timeout=120)
            if resp.status_code == 429:
                wait = 30 * (attempt + 1)
                log.warning(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            if resp.status_code != 200:
                log.warning(f"  HTTP {resp.status_code}: {resp.text[:200]}")
                return None
            return resp.json()
        except Exception as e:
            log.error(f"  Request error (attempt {attempt+1}): {e}")
            time.sleep(5)
    return None


def parse_hourly(data: dict) -> list[dict]:
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


def store_records(conn: sqlite3.Connection, records: list[dict]) -> int:
    inserted = 0
    for r in records:
        try:
            conn.execute(
                """INSERT OR IGNORE INTO nwp_forecast_archive
                   (city, model, timestamp_utc, temperature_2m, dewpoint_2m,
                    windspeed_10m, winddirection_10m, cape, surface_pressure,
                    cloudcover, shortwave_radiation)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (CITY, MODEL, r["timestamp_utc"], r["temperature_2m"],
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
    sd = datetime.strptime(start, "%Y-%m-%d")
    ed = datetime.strptime(end, "%Y-%m-%d")
    while sd < ed:
        ce = min(sd + timedelta(days=chunk_days - 1), ed)
        yield sd.strftime("%Y-%m-%d"), ce.strftime("%Y-%m-%d")
        sd = ce + timedelta(days=1)


def main():
    conn = sqlite3.connect(DB_PATH, timeout=60)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=60000")
    ensure_table(conn)

    existing = conn.execute(
        "SELECT COUNT(*) FROM nwp_forecast_archive WHERE city=? AND model=?",
        (CITY, MODEL)
    ).fetchone()[0]
    log.info(f"City: {CITY} | Model: {MODEL} | Existing rows: {existing:,}")

    total = 0
    chunks = list(generate_chunks(START_DATE, END_DATE))
    log.info(f"Total chunks to fetch: {len(chunks)}")

    for chunk_idx, (cs, ce) in enumerate(chunks):
        log.info(f"  Chunk {chunk_idx+1}/{len(chunks)}: {cs} -> {ce}")

        data = fetch_chunk(cs, ce)
        if data is None:
            log.warning(f"  Skipping chunk {cs} -> {ce}")
            time.sleep(2)
            continue

        records = parse_hourly(data)
        if not records:
            log.warning(f"  No records in chunk")
            time.sleep(1)
            continue

        non_null = sum(1 for r in records if r["temperature_2m"] is not None)
        inserted = store_records(conn, records)
        total += inserted
        log.info(f"    -> {inserted} rows ({non_null}/{len(records)} with data)")

        time.sleep(1.0)

    conn.close()
    log.info(f"\nCOMPLETE: {total:,} total rows inserted for {CITY}/{MODEL}")


if __name__ == "__main__":
    main()
