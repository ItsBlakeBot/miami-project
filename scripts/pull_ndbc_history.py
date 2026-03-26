#!/usr/bin/env python3
"""Pull historical NDBC buoy data (stdmet) for 2020-2026 and store in buoy_observations."""

import gzip
import io
import sqlite3
import sys
import time
from datetime import datetime, timezone

import requests

DB_PATH = "/Users/blakebot/blakebot/miami-project/miami_collector.db"

# (station, city, description)
BUOYS = [
    # SE Florida buoys
    ("VAKF1", "Miami", "Virginia Key"),
    ("FWYF1", "Miami", "Fowey Rocks"),
    ("LKWF1", "Miami", "Lake Worth"),
    ("SMKF1", "Miami", "Sombrero Key"),
    ("LONF1", "Miami", "Long Key"),
    ("SPGF1", "Miami", "Settlement Point Grand Bahama"),
    # NYC buoys
    ("44025", "NYC", "Long Island"),
    ("44065", "NYC", "New York Harbor"),
    # Chicago buoy
    ("45007", "Chicago", "South Lake Michigan"),
]

YEARS = range(2020, 2027)  # 2020-2026

# NDBC missing value sentinels
MISSING_VALUES = {999, 999.0, 99.0, 9999, 9999.0}


def create_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS buoy_observations (
            station TEXT NOT NULL,
            city TEXT,
            timestamp_utc TEXT NOT NULL,
            wind_dir REAL,
            wind_speed_ms REAL,
            gust_ms REAL,
            pressure_hpa REAL,
            air_temp_c REAL,
            water_temp_c REAL,
            dewpoint_c REAL,
            wave_height_m REAL,
            visibility_mi REAL,
            PRIMARY KEY (station, timestamp_utc)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_buoy_obs_city ON buoy_observations(city)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_buoy_obs_ts ON buoy_observations(timestamp_utc)")
    conn.commit()


def parse_val(s: str) -> float | None:
    """Parse a value, returning None for NDBC missing sentinels."""
    if s == "MM" or s == "":
        return None
    try:
        v = float(s)
        if v in MISSING_VALUES:
            return None
        return v
    except ValueError:
        return None


def fetch_and_parse(station: str, year: int) -> list[tuple]:
    """Download one year of historical stdmet data for a station."""
    station_lower = station.lower()
    url = (
        f"https://www.ndbc.noaa.gov/view_text_file.php?"
        f"filename={station_lower}h{year}.txt.gz&dir=data/historical/stdmet/"
    )
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 404:
            return []
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  WARNING: {station} {year} fetch error: {e}")
        return []

    # The response may be gzipped or plain text
    try:
        text = gzip.decompress(resp.content).decode("utf-8", errors="replace")
    except (gzip.BadGzipFile, OSError):
        text = resp.text

    lines = text.strip().split("\n")
    if len(lines) < 3:
        return []

    # Parse header - first line has column names (starts with #)
    header_line = lines[0].replace("#", "").strip()
    headers = header_line.split()

    # Second line is units, skip it
    rows = []
    for line in lines[2:]:
        parts = line.split()
        if len(parts) < len(headers):
            continue
        row = dict(zip(headers, parts))

        # Parse timestamp
        try:
            yr = int(row.get("YY", row.get("YYYY", 0)))
            mo = int(row["MM"])
            dd = int(row["DD"])
            hh = int(row["hh"])
            mm_val = int(row.get("mm", 0))
            ts = datetime(yr, mo, dd, hh, mm_val, tzinfo=timezone.utc)
            ts_str = ts.strftime("%Y-%m-%dT%H:%M:%SZ")
        except (ValueError, KeyError):
            continue

        rows.append((
            ts_str,
            parse_val(row.get("WDIR", "")),
            parse_val(row.get("WSPD", "")),
            parse_val(row.get("GST", "")),
            parse_val(row.get("PRES", "")),
            parse_val(row.get("ATMP", "")),
            parse_val(row.get("WTMP", "")),
            parse_val(row.get("DEWP", "")),
            parse_val(row.get("WVHT", "")),
            parse_val(row.get("VIS", "")),
        ))

    return rows


def main():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    create_table(conn)

    total_inserted = 0
    total_skipped = 0

    for station, city, desc in BUOYS:
        print(f"\n=== {station} ({desc}, {city}) ===")
        station_total = 0

        for year in YEARS:
            rows = fetch_and_parse(station, year)
            if not rows:
                print(f"  {year}: no data")
                continue

            inserted = 0
            for ts_str, wdir, wspd, gst, pres, atmp, wtmp, dewp, wvht, vis in rows:
                try:
                    conn.execute(
                        """INSERT OR IGNORE INTO buoy_observations
                           (station, city, timestamp_utc, wind_dir, wind_speed_ms,
                            gust_ms, pressure_hpa, air_temp_c, water_temp_c,
                            dewpoint_c, wave_height_m, visibility_mi)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (station, city, ts_str, wdir, wspd, gst, pres, atmp, wtmp, dewp, wvht, vis),
                    )
                    if conn.total_changes > total_inserted + total_skipped + inserted:
                        inserted += 1
                except sqlite3.IntegrityError:
                    pass

            conn.commit()
            print(f"  {year}: {len(rows)} rows parsed, {inserted} new inserted")
            station_total += inserted
            total_inserted += inserted

            # Be polite to NDBC servers
            time.sleep(0.5)

        print(f"  Station total: {station_total} new rows")

    # Final summary
    cursor = conn.execute("SELECT COUNT(*) FROM buoy_observations")
    total_rows = cursor.fetchone()[0]
    cursor = conn.execute("SELECT COUNT(DISTINCT station) FROM buoy_observations")
    n_stations = cursor.fetchone()[0]

    print(f"\n{'='*60}")
    print(f"DONE. {total_inserted} new rows inserted.")
    print(f"buoy_observations table: {total_rows} total rows, {n_stations} stations")

    # Show per-station counts
    cursor = conn.execute(
        "SELECT station, city, COUNT(*), MIN(timestamp_utc), MAX(timestamp_utc) "
        "FROM buoy_observations GROUP BY station ORDER BY station"
    )
    for row in cursor:
        print(f"  {row[0]} ({row[1]}): {row[2]} rows  [{row[3]} .. {row[4]}]")

    conn.close()


if __name__ == "__main__":
    main()
