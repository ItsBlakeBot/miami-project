"""
Backfill CLI timing data from IEM (Iowa Environmental Mesonet).
Pulls daily CLI (Climate Summary) data including time of high/low temperature occurrence.

Source: https://mesonet.agron.iastate.edu/geojson/cli.py?dt=YYYY-MM-DD&fmt=csv
"""

import sqlite3
import csv
import io
import re
import time
import datetime
import requests

DB_PATH = "/Users/blakebot/blakebot/miami-project/miami_collector.db"

STATIONS = ["KMIA", "KNYC", "KMDW", "KDEN", "KAUS"]

# UTC offsets for NWS CLI (standard time, no DST)
UTC_OFFSETS = {
    "KMIA": 5,   # EST = UTC-5
    "KNYC": 5,   # EST = UTC-5
    "KMDW": 6,   # CST = UTC-6
    "KDEN": 7,   # MST = UTC-7
    "KAUS": 6,   # CST = UTC-6
}

START_DATE = datetime.date(2020, 1, 1)
END_DATE = datetime.date(2026, 3, 26)

IEM_URL = "https://mesonet.agron.iastate.edu/geojson/cli.py"


def create_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cli_timing (
            station TEXT NOT NULL,
            date TEXT NOT NULL,
            high_temp_f REAL,
            high_time_lst TEXT,
            high_hour_utc REAL,
            low_temp_f REAL,
            low_time_lst TEXT,
            low_hour_utc REAL,
            precip_in REAL,
            snow_in REAL,
            PRIMARY KEY (station, date)
        )
    """)
    conn.commit()


def parse_cli_time(time_str):
    """
    Parse CLI time string like '124 PM', '437 AM', '1159 PM', 'MM', 'M', ''
    Returns 24h HHMM string like '1324', '0437', '2359', or None if missing.
    """
    if not time_str or time_str.strip() in ('M', 'MM', ''):
        return None

    time_str = time_str.strip()
    # Match patterns like "124 PM", "1200 AM", "959 AM"
    m = re.match(r'^(\d{1,4})\s*(AM|PM)$', time_str, re.IGNORECASE)
    if not m:
        return None

    num_part = m.group(1)
    ampm = m.group(2).upper()

    # Pad to 3 or 4 digits
    if len(num_part) <= 2:
        # e.g. "12 PM" -> hour=12, minute=0
        hour = int(num_part)
        minute = 0
    elif len(num_part) == 3:
        # e.g. "124 PM" -> hour=1, minute=24
        hour = int(num_part[0])
        minute = int(num_part[1:])
    else:
        # e.g. "1159 PM" -> hour=11, minute=59
        hour = int(num_part[:2])
        minute = int(num_part[2:])

    # Convert to 24h
    if ampm == 'AM':
        if hour == 12:
            hour = 0
    else:  # PM
        if hour != 12:
            hour += 12

    return f"{hour:02d}{minute:02d}"


def local_time_to_utc_hour(time_lst, utc_offset):
    """
    Convert local standard time HHMM string to UTC decimal hour.
    utc_offset is the positive offset (e.g. 5 for EST).
    """
    if time_lst is None:
        return None
    hour = int(time_lst[:2])
    minute = int(time_lst[2:])
    utc_hour = hour + utc_offset + minute / 60.0
    # Wrap around 24
    if utc_hour >= 24:
        utc_hour -= 24
    return round(utc_hour, 2)


def parse_float(val):
    """Parse a float value, returning None for missing/trace values."""
    if val is None:
        return None
    val = str(val).strip()
    if val in ('M', 'MM', '', 'T'):
        # T = trace for precip, treat as 0.0
        if val == 'T':
            return 0.0001  # trace amount
        return None
    try:
        return float(val)
    except ValueError:
        return None


def fetch_day(session, dt):
    """Fetch CLI data for a single day. Returns list of dicts for target stations."""
    url = f"{IEM_URL}?dt={dt.isoformat()}&fmt=csv"
    resp = session.get(url, timeout=30)
    if resp.status_code != 200:
        return []

    rows = []
    reader = csv.DictReader(io.StringIO(resp.text))
    for row in reader:
        station = row.get('station', '').strip()
        if station in STATIONS:
            rows.append(row)
    return rows


def process_row(row):
    """Convert a CSV row dict into a cli_timing record tuple."""
    station = row['station'].strip()
    date = row['valid'].strip()

    high_temp = parse_float(row.get('high'))
    low_temp = parse_float(row.get('low'))
    precip = parse_float(row.get('precip'))
    snow = parse_float(row.get('snow'))

    high_time_raw = row.get('high_time', '').strip()
    low_time_raw = row.get('low_time', '').strip()

    high_time_lst = parse_cli_time(high_time_raw)
    low_time_lst = parse_cli_time(low_time_raw)

    utc_offset = UTC_OFFSETS[station]
    high_hour_utc = local_time_to_utc_hour(high_time_lst, utc_offset)
    low_hour_utc = local_time_to_utc_hour(low_time_lst, utc_offset)

    return (
        station, date,
        high_temp, high_time_lst, high_hour_utc,
        low_temp, low_time_lst, low_hour_utc,
        precip, snow
    )


def get_existing_dates(conn):
    """Get set of (station, date) pairs already in the table."""
    cursor = conn.execute("SELECT station, date FROM cli_timing")
    return set(cursor.fetchall())


def main():
    conn = sqlite3.connect(DB_PATH)
    create_table(conn)

    existing = get_existing_dates(conn)
    print(f"Already have {len(existing)} records in cli_timing")

    # Generate all dates
    total_days = (END_DATE - START_DATE).days + 1
    all_dates = [START_DATE + datetime.timedelta(days=i) for i in range(total_days)]

    # Figure out which dates we still need (any station missing)
    dates_needed = set()
    for dt in all_dates:
        date_str = dt.isoformat()
        for station in STATIONS:
            if (station, date_str) not in existing:
                dates_needed.add(dt)
                break

    dates_needed = sorted(dates_needed)
    print(f"Need to fetch {len(dates_needed)} days out of {total_days} total")

    session = requests.Session()
    session.headers['User-Agent'] = 'miami-weather-research/1.0 (blakebot)'

    inserted = 0
    errors = 0
    batch_rows = []
    batch_size = 50

    for i, dt in enumerate(dates_needed):
        try:
            rows = fetch_day(session, dt)
            date_str = dt.isoformat()

            for row in rows:
                station = row['station'].strip()
                if (station, date_str) not in existing:
                    record = process_row(row)
                    batch_rows.append(record)
                    existing.add((station, date_str))

            if len(batch_rows) >= batch_size:
                conn.executemany("""
                    INSERT OR REPLACE INTO cli_timing
                    (station, date, high_temp_f, high_time_lst, high_hour_utc,
                     low_temp_f, low_time_lst, low_hour_utc, precip_in, snow_in)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, batch_rows)
                conn.commit()
                inserted += len(batch_rows)
                batch_rows = []

        except Exception as e:
            errors += 1
            if errors <= 10:
                print(f"  Error on {dt}: {e}")

        # Progress
        if (i + 1) % 100 == 0:
            # Flush batch
            if batch_rows:
                conn.executemany("""
                    INSERT OR REPLACE INTO cli_timing
                    (station, date, high_temp_f, high_time_lst, high_hour_utc,
                     low_temp_f, low_time_lst, low_hour_utc, precip_in, snow_in)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, batch_rows)
                conn.commit()
                inserted += len(batch_rows)
                batch_rows = []
            print(f"  Progress: {i+1}/{len(dates_needed)} days fetched, {inserted} rows inserted, {errors} errors")

        # Rate limit: ~10 requests/sec
        time.sleep(0.1)

    # Final flush
    if batch_rows:
        conn.executemany("""
            INSERT OR REPLACE INTO cli_timing
            (station, date, high_temp_f, high_time_lst, high_hour_utc,
             low_temp_f, low_time_lst, low_hour_utc, precip_in, snow_in)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, batch_rows)
        conn.commit()
        inserted += len(batch_rows)

    print(f"\nDone! Inserted {inserted} total rows, {errors} errors")

    # Summary
    cursor = conn.execute("""
        SELECT station, COUNT(*) as cnt,
               MIN(date) as min_date, MAX(date) as max_date,
               SUM(CASE WHEN high_time_lst IS NOT NULL THEN 1 ELSE 0 END) as has_high_time,
               SUM(CASE WHEN low_time_lst IS NOT NULL THEN 1 ELSE 0 END) as has_low_time
        FROM cli_timing
        GROUP BY station
        ORDER BY station
    """)
    print("\nStation Summary:")
    print(f"{'Station':<8} {'Count':>6} {'From':>12} {'To':>12} {'HasHighT':>9} {'HasLowT':>8}")
    for row in cursor:
        print(f"{row[0]:<8} {row[1]:>6} {row[2]:>12} {row[3]:>12} {row[4]:>9} {row[5]:>8}")

    # Sample data
    print("\nSample KMIA records (last 5):")
    cursor = conn.execute("""
        SELECT * FROM cli_timing
        WHERE station = 'KMIA'
        ORDER BY date DESC
        LIMIT 5
    """)
    for row in cursor:
        print(f"  {row}")

    conn.close()


if __name__ == "__main__":
    main()
