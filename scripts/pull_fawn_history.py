#!/usr/bin/env python3
"""Pull historical FAWN soil/weather data from fawnpub CSV files and store in fawn_soil_history."""

import csv
import io
import sqlite3
import time

import requests

DB_PATH = "/Users/blakebot/blakebot/miami-project/miami_collector.db"

# FAWN station directories on the public server
# Format: (station_id, directory_name)
# Note: station IDs in FAWN don't match what was expected
# 440 = Homestead, 420 = Fort Lauderdale, 450 = Immokalee
FAWN_STATIONS = [
    ("440", "Homestead_440", "Homestead"),
    ("420", "Fort%20Lauderdale_420", "Fort Lauderdale"),
    ("450", "Immokalee_450", "Immokalee"),
    ("330", "Lake%20Alfred_330", "Lake Alfred"),  # 330 is Lake Alfred, not Fort Lauderdale
]

YEARS = range(2021, 2027)  # Data starts at 2021 based on directory listing

BASE_URL = "https://fawn.ifas.ufl.edu/data/fawnpub/daily_summaries/BY_STATION"


def create_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS fawn_soil_history (
            station_id TEXT NOT NULL,
            date TEXT NOT NULL,
            soil_temp_4in_f REAL,
            soil_temp_24in_f REAL,
            rainfall_in REAL,
            solar_radiation REAL,
            humidity_pct REAL,
            et_in REAL,
            PRIMARY KEY (station_id, date)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_fawn_soil_date ON fawn_soil_history(date)")
    conn.commit()


def parse_fawn_csv(text: str, station_id: str) -> list[tuple]:
    """Parse FAWN daily summary CSV.

    Expected columns:
    StationID, Date Time, Soil Temp (C) Avg, Soil Temp (C) Min, Soil Temp (C) Max,
    Temp @ 60cm (C) Avg, ... Relative Humidity (%) Avg, ...
    Rainfall Amount (in) Sum, ... Solar Radiation (w/m2) Avg,
    Solar Radiation (MJ/m2) Sum, ETo Grass (mm) Avg, ...
    """
    rows = []
    reader = csv.DictReader(io.StringIO(text))

    for row in reader:
        try:
            # Date
            date_str = row.get("Date Time", "").strip()
            if not date_str:
                continue
            # Format: "2024-01-01 00:00:00"
            date_val = date_str[:10]

            def safe_float(key):
                v = row.get(key, "").strip()
                if not v or v == "NA" or v == "":
                    return None
                try:
                    return float(v)
                except ValueError:
                    return None

            # Soil temp is in C, convert to F
            soil_c = safe_float("Soil Temp (C) Avg")
            soil_4in_f = (soil_c * 9/5 + 32) if soil_c is not None else None

            # No 24in soil temp in this dataset; use None
            soil_24in_f = None

            rainfall_in = safe_float("Rainfall Amount (in) Sum")
            solar_rad = safe_float("Solar Radiation (w/m2) Avg")
            humidity = safe_float("Relative Humidity (%) Avg")

            # ET in mm, convert to inches
            et_mm = safe_float("ETo Grass (mm) Avg")
            et_in = (et_mm / 25.4) if et_mm is not None else None

            rows.append((
                station_id, date_val, soil_4in_f, soil_24in_f,
                rainfall_in, solar_rad, humidity, et_in,
            ))
        except Exception:
            continue

    return rows


def main():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    create_table(conn)

    total_inserted = 0

    for station_id, dir_name, display_name in FAWN_STATIONS:
        print(f"\n=== FAWN Station {station_id} ({display_name}) ===")
        station_total = 0

        for year in YEARS:
            url = f"{BASE_URL}/{dir_name}/{year}.csv"
            print(f"  Fetching {year}...", end=" ", flush=True)

            try:
                resp = requests.get(url, timeout=30)
                if resp.status_code == 404:
                    print("not found")
                    continue
                resp.raise_for_status()
            except requests.RequestException as e:
                print(f"error: {e}")
                continue

            rows = parse_fawn_csv(resp.text, station_id)
            inserted = 0
            for row in rows:
                try:
                    conn.execute(
                        """INSERT OR IGNORE INTO fawn_soil_history
                           (station_id, date, soil_temp_4in_f, soil_temp_24in_f,
                            rainfall_in, solar_radiation, humidity_pct, et_in)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                        row,
                    )
                    inserted += 1
                except sqlite3.IntegrityError:
                    pass

            conn.commit()
            print(f"{len(rows)} parsed, {inserted} inserted")
            station_total += inserted
            total_inserted += inserted
            time.sleep(0.5)

        print(f"  Station total: {station_total} rows")

    # Summary
    cursor = conn.execute("SELECT COUNT(*) FROM fawn_soil_history")
    total_rows = cursor.fetchone()[0]
    print(f"\n{'='*60}")
    print(f"DONE. {total_inserted} new rows inserted into fawn_soil_history.")
    print(f"Table total: {total_rows} rows")

    cursor = conn.execute(
        "SELECT station_id, COUNT(*), MIN(date), MAX(date) "
        "FROM fawn_soil_history GROUP BY station_id"
    )
    for row in cursor:
        print(f"  Station {row[0]}: {row[1]} rows [{row[2]} .. {row[3]}]")

    conn.close()


if __name__ == "__main__":
    main()
