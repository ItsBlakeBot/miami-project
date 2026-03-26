#!/usr/bin/env python3
"""Pull historical MOS forecast data from IEM dl.php and store in mos_forecasts."""

import csv
import io
import sqlite3
import time
from datetime import datetime, timedelta

import requests

DB_PATH = "/Users/blakebot/blakebot/miami-project/miami_collector.db"

# MOS stations (4-letter ICAO)
MOS_STATIONS = ["KMIA", "KNYC", "KMDW", "KDEN", "KAUS"]

# Pull in monthly chunks: 2020-2026
START_YEAR = 2020
END_YEAR = 2026
END_MONTH = 3  # through March 2026


def create_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS mos_forecasts (
            station TEXT NOT NULL,
            model TEXT NOT NULL,
            runtime_utc TEXT NOT NULL,
            valid_time_utc TEXT NOT NULL,
            max_temp_f REAL,
            min_temp_f REAL,
            temp_f REAL,
            wind_speed_kt REAL,
            wind_dir REAL,
            pop_pct REAL,
            cloud_cover TEXT,
            PRIMARY KEY (station, model, runtime_utc, valid_time_utc)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_mos_station ON mos_forecasts(station)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_mos_valid ON mos_forecasts(valid_time_utc)")
    conn.commit()


def fetch_iem_mos(station: str, y1: int, m1: int, d1: int, y2: int, m2: int, d2: int) -> str | None:
    """Fetch MOS CSV data from IEM dl.php."""
    url = "https://mesonet.agron.iastate.edu/mos/dl.php"
    params = {
        "station": station,
        "model": "GFS",
        "year1": str(y1),
        "month1": str(m1),
        "day1": str(d1),
        "year2": str(y2),
        "month2": str(m2),
        "day2": str(d2),
    }
    try:
        resp = requests.get(url, params=params, timeout=120)
        if resp.status_code != 200:
            print(f"HTTP {resp.status_code}")
            return None
        if len(resp.text) < 50:
            return None
        return resp.text
    except requests.RequestException as e:
        print(f"error: {e}")
        return None


def _float_or_none(s: str) -> float | None:
    if not s or s.strip() in ("", "M", "9999", "999"):
        return None
    try:
        return float(s.strip())
    except ValueError:
        return None


def parse_mos_csv(text: str) -> list[tuple]:
    """Parse IEM MOS CSV.

    Columns: runtime, ftime, model, n_x, tmp, dpt, cld, wdr, wsp, p06, p12,
             q06, q12, t06_1, t06_2, t12_1, t12_2, cig, vis, obv, poz, pos,
             typ, station, t06, t12
    """
    rows = []
    reader = csv.DictReader(io.StringIO(text))

    for row in reader:
        try:
            station = row.get("station", "").strip()
            model = row.get("model", "GFS").strip()
            runtime = row.get("runtime", "").strip()
            ftime = row.get("ftime", "").strip()

            if not runtime or not ftime or not station:
                continue

            tmp = _float_or_none(row.get("tmp", ""))
            n_x = _float_or_none(row.get("n_x", ""))
            wsp = _float_or_none(row.get("wsp", ""))
            wdr = _float_or_none(row.get("wdr", ""))
            cld = row.get("cld", "").strip() or None

            # POP - use p12 preferentially, fall back to p06
            pop = _float_or_none(row.get("p12", ""))
            if pop is None:
                pop = _float_or_none(row.get("p06", ""))

            # n_x is max or min temp depending on valid time
            max_temp = None
            min_temp = None
            if n_x is not None:
                # In MOS, n_x at 00Z valid times is typically max, at 12Z is min
                # But more reliably: if n_x > tmp, it's max; if n_x < tmp, it's min
                if tmp is not None:
                    if n_x >= tmp:
                        max_temp = n_x
                    else:
                        min_temp = n_x
                else:
                    # Can't determine, store in max
                    max_temp = n_x

            rows.append((
                station, model, runtime, ftime,
                max_temp, min_temp, tmp,
                wsp, wdr, pop, cld,
            ))
        except Exception:
            continue

    return rows


def month_ranges(start_year: int, end_year: int, end_month: int):
    """Generate (y1, m1, d1, y2, m2, d2) tuples for each month."""
    for year in range(start_year, end_year + 1):
        max_month = end_month if year == end_year else 12
        for month in range(1, max_month + 1):
            # End of month
            if month == 12:
                next_y, next_m = year + 1, 1
            else:
                next_y, next_m = year, month + 1
            # Last day of this month
            last_day = (datetime(next_y, next_m, 1) - timedelta(days=1)).day
            yield year, month, 1, year, month, last_day


def main():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    create_table(conn)

    total_inserted = 0

    for station in MOS_STATIONS:
        print(f"\n=== MOS {station} ===")
        station_total = 0

        for y1, m1, d1, y2, m2, d2 in month_ranges(START_YEAR, END_YEAR, END_MONTH):
            label = f"{y1}-{m1:02d}"
            print(f"  {label}...", end=" ", flush=True)

            text = fetch_iem_mos(station, y1, m1, d1, y2, m2, d2)
            if not text:
                print("no data")
                time.sleep(1)
                continue

            rows = parse_mos_csv(text)
            inserted = 0
            for row in rows:
                try:
                    conn.execute(
                        """INSERT OR IGNORE INTO mos_forecasts
                           (station, model, runtime_utc, valid_time_utc,
                            max_temp_f, min_temp_f, temp_f,
                            wind_speed_kt, wind_dir, pop_pct, cloud_cover)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        row,
                    )
                    inserted += 1
                except sqlite3.IntegrityError:
                    pass

            conn.commit()
            print(f"{len(rows)} parsed, {inserted} new")
            station_total += inserted
            total_inserted += inserted

            time.sleep(1)  # Be polite to IEM

        print(f"  Station total: {station_total} rows")

    # Summary
    cursor = conn.execute("SELECT COUNT(*) FROM mos_forecasts")
    total_rows = cursor.fetchone()[0]
    print(f"\n{'='*60}")
    print(f"DONE. {total_inserted} new rows inserted into mos_forecasts.")
    print(f"Table total: {total_rows} rows")

    cursor = conn.execute(
        "SELECT station, model, COUNT(*), MIN(runtime_utc), MAX(runtime_utc) "
        "FROM mos_forecasts GROUP BY station, model"
    )
    for row in cursor:
        print(f"  {row[0]} ({row[1]}): {row[2]} rows [{row[3]} .. {row[4]}]")

    conn.close()


if __name__ == "__main__":
    main()
