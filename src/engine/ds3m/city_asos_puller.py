"""Pull IEM ASOS observations for new cities (NYC, Chicago, Denver, Austin).

Backfills city_asos_backfill table with hourly METAR data from IEM
for primary + nearby stations per city. Uses same approach as enriched_backfill.py.

Usage:
    python city_asos_puller.py
"""

from __future__ import annotations

import csv
import io
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
IEM_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"

START_DATE = "2020-01-01"
END_DATE = "2026-03-26"

# Primary + nearby stations per city
CITY_STATIONS = {
    "NYC": ["JFK", "LGA", "EWR"],
    "Chicago": ["MDW", "ORD", "RFD"],
    "Denver": ["DEN", "APA", "BJC"],
    "Austin": ["AUS", "GTU", "SAT"],
}


def ensure_table(conn: sqlite3.Connection):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS city_asos_backfill (
            station TEXT NOT NULL,
            city TEXT NOT NULL,
            timestamp_utc TEXT NOT NULL,
            temperature_f REAL,
            dewpoint_f REAL,
            wind_direction REAL,
            wind_speed_kt REAL,
            visibility_mi REAL,
            sky_cover TEXT,
            pressure_mb REAL,
            humidity_pct REAL,
            precip_in REAL,
            PRIMARY KEY (station, timestamp_utc)
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_city_asos_city
        ON city_asos_backfill(city, timestamp_utc)
    """)
    conn.commit()


def get_existing_count(conn: sqlite3.Connection, station: str) -> int:
    row = conn.execute(
        "SELECT COUNT(*) FROM city_asos_backfill WHERE station=?",
        (station,)
    ).fetchone()
    return row[0] if row else 0


def fetch_asos_chunk(station: str, start: str, end: str) -> list[dict]:
    """Fetch ASOS data for one station for a date range from IEM."""
    sd = datetime.strptime(start, "%Y-%m-%d")
    ed = datetime.strptime(end, "%Y-%m-%d")

    params = {
        "station": station,
        "data": "tmpf,dwpf,relh,drct,sknt,vsby,skyc1,mslp,p01i",
        "year1": sd.year, "month1": sd.month, "day1": sd.day,
        "year2": ed.year, "month2": ed.month, "day2": ed.day,
        "tz": "UTC",
        "format": "onlycomma",
        "latlon": "no",
        "elev": "no",
        "missing": "empty",
        "trace": "empty",
        "direct": "no",
        "report_type": "3",
    }

    try:
        resp = requests.get(IEM_URL, params=params, timeout=300)
        if resp.status_code != 200:
            log.warning(f"  HTTP {resp.status_code} for {station}")
            return []
        text = resp.text
    except Exception as e:
        log.error(f"  Request error for {station}: {e}")
        return []

    records = []
    reader = csv.DictReader(io.StringIO(text))
    for row in reader:
        try:
            ts = row.get("valid", "").strip()
            if not ts:
                continue

            def safe_float(key):
                v = row.get(key, "").strip()
                if not v or v == "M":
                    return None
                try:
                    return float(v)
                except ValueError:
                    return None

            records.append({
                "timestamp_utc": ts,
                "temperature_f": safe_float("tmpf"),
                "dewpoint_f": safe_float("dwpf"),
                "wind_direction": safe_float("drct"),
                "wind_speed_kt": safe_float("sknt"),
                "visibility_mi": safe_float("vsby"),
                "sky_cover": row.get("skyc1", "").strip() or None,
                "pressure_mb": safe_float("mslp"),
                "humidity_pct": safe_float("relh"),
                "precip_in": safe_float("p01i"),
            })
        except Exception:
            continue

    return records


def store_records(conn: sqlite3.Connection, station: str, city: str, records: list[dict]) -> int:
    inserted = 0
    for r in records:
        try:
            conn.execute(
                """INSERT OR IGNORE INTO city_asos_backfill
                   (station, city, timestamp_utc, temperature_f, dewpoint_f,
                    wind_direction, wind_speed_kt, visibility_mi, sky_cover,
                    pressure_mb, humidity_pct, precip_in)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (station, city, r["timestamp_utc"], r["temperature_f"],
                 r["dewpoint_f"], r["wind_direction"], r["wind_speed_kt"],
                 r["visibility_mi"], r["sky_cover"], r["pressure_mb"],
                 r["humidity_pct"], r["precip_in"])
            )
            inserted += 1
        except sqlite3.IntegrityError:
            pass
    conn.commit()
    return inserted


def generate_year_chunks(start: str, end: str):
    """Generate yearly chunks to keep IEM requests manageable."""
    sd = datetime.strptime(start, "%Y-%m-%d")
    ed = datetime.strptime(end, "%Y-%m-%d")
    while sd < ed:
        ce = min(datetime(sd.year, 12, 31), ed)
        yield sd.strftime("%Y-%m-%d"), ce.strftime("%Y-%m-%d")
        sd = datetime(sd.year + 1, 1, 1)


def main():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    ensure_table(conn)

    total_all = 0

    for city, stations in CITY_STATIONS.items():
        log.info(f"\n{'='*60}")
        log.info(f"CITY: {city} | Stations: {stations}")
        log.info(f"{'='*60}")

        for station in stations:
            existing = get_existing_count(conn, station)
            log.info(f"\n  Station: {station} | Existing rows: {existing:,}")

            station_total = 0
            chunks = list(generate_year_chunks(START_DATE, END_DATE))

            for chunk_idx, (cs, ce) in enumerate(chunks):
                log.info(f"    Chunk {chunk_idx+1}/{len(chunks)}: {cs} -> {ce}")

                records = fetch_asos_chunk(station, cs, ce)
                if not records:
                    log.warning(f"    No data for {station} {cs}->{ce}")
                    time.sleep(2)
                    continue

                inserted = store_records(conn, station, city, records)
                station_total += inserted
                log.info(f"      -> {inserted} new rows ({len(records)} total in chunk)")

                time.sleep(2.0)  # rate limit - IEM is generous but be respectful

            log.info(f"  {station} ({city}): {station_total:,} new rows")
            total_all += station_total

    conn.close()
    log.info(f"\n{'='*60}")
    log.info(f"CITY ASOS BACKFILL COMPLETE: {total_all:,} total new rows")
    log.info(f"{'='*60}")


if __name__ == "__main__":
    main()
