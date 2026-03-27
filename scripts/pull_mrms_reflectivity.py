#!/usr/bin/env python3
"""
Pull MRMS radar reflectivity proxy data from Open-Meteo archive API.

Since actual MRMS GRIB2 data requires downloading huge national-coverage files,
we use Open-Meteo's precipitation data (which is itself derived from radar/ERA5)
as a proxy for radar reflectivity near Miami.

We derive max_reflectivity_dbz from precipitation rate using the standard
Z-R relationship: Z = 300 * R^1.4 (Marshall-Palmer), then dBZ = 10*log10(Z).

Coverage: 2020-01-01 to 2026-03-26, hourly, for Miami (25.79, -80.29).
"""

import sqlite3
import math
import requests
import time
from datetime import datetime, timedelta

DB_PATH = "/Users/blakebot/blakebot/miami-project/miami_collector.db"
LAT, LON = 25.79, -80.29


def create_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS mrms_reflectivity (
            timestamp_utc TEXT PRIMARY KEY,
            max_reflectivity_dbz REAL,
            precip_rate_mmhr REAL
        )
    """)
    conn.commit()


def precip_to_dbz(precip_mm_hr):
    """Convert precipitation rate (mm/hr) to approximate dBZ using Marshall-Palmer Z-R.

    Z = 300 * R^1.4  (Z in mm^6/m^3, R in mm/hr)
    dBZ = 10 * log10(Z)

    Typical scale:
      0 mm/hr  -> 0 dBZ (no echo)
      0.5 mm/hr -> ~23 dBZ (light rain)
      2.5 mm/hr -> ~33 dBZ (moderate rain)
      10 mm/hr  -> ~43 dBZ (heavy rain)
      50 mm/hr  -> ~55 dBZ (very heavy / thunderstorm)
    """
    if precip_mm_hr is None or precip_mm_hr <= 0:
        return 0.0
    z = 300.0 * (precip_mm_hr ** 1.4)
    dbz = 10.0 * math.log10(z)
    return round(dbz, 1)


def fetch_chunk(start_date, end_date):
    """Fetch one chunk from Open-Meteo archive API."""
    params = {
        "latitude": LAT,
        "longitude": LON,
        "hourly": "precipitation,rain,snowfall,weather_code",
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "UTC",
    }
    r = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params=params,
        timeout=60,
    )
    r.raise_for_status()
    return r.json()


def weather_code_dbz_boost(wc, base_dbz, precip_rate):
    """Boost reflectivity estimate for convective weather codes and high precip rates.

    Hourly precip averages can underestimate peak reflectivity in convective
    cells. In Miami, high precip rates almost always mean convective storms
    where instantaneous reflectivity peaks much higher than the hourly average.

    ERA5 reanalysis typically uses WMO codes 61/63/65 (rain) rather than
    80-82 (showers) or 95-99 (thunderstorms), so we also boost based on
    raw precipitation rate.
    """
    if wc is None:
        wc = 0
    else:
        wc = int(wc)

    boosted = base_dbz

    # Boost by WMO code
    if wc in (95, 96, 99):
        boosted = max(boosted, 50.0) + min(15.0, base_dbz * 0.2)
    elif wc in (80, 81, 82):
        boosted = max(boosted, 40.0) + min(8.0, base_dbz * 0.1)
    elif wc == 65:  # Heavy rain (ERA5 equivalent of convective)
        boosted = max(boosted, 42.0) + min(10.0, base_dbz * 0.15)
    elif wc == 63:  # Moderate rain
        boosted = max(boosted, 32.0) + min(5.0, base_dbz * 0.05)

    # Additional boost by raw precip rate (captures convective intensity
    # even when WMO code underreports)
    p = precip_rate or 0.0
    if p >= 20.0:
        # 20+ mm/hr is almost certainly a strong thunderstorm in Miami
        # Peak instantaneous reflectivity in such storms typically 50-60+ dBZ
        boosted = max(boosted, 50.0 + min(10.0, (p - 20) * 0.3))
    elif p >= 10.0:
        boosted = max(boosted, 45.0 + min(5.0, (p - 10) * 0.3))
    elif p >= 5.0:
        boosted = max(boosted, 38.0 + min(5.0, (p - 5) * 0.3))

    return round(boosted, 1)


def main():
    conn = sqlite3.connect(DB_PATH)
    create_table(conn)

    existing = conn.execute("SELECT COUNT(*) FROM mrms_reflectivity").fetchone()[0]
    print(f"Existing rows: {existing}")

    start = datetime(2020, 1, 1)
    end = datetime(2026, 3, 26)

    chunk_days = 365
    total_inserted = 0
    current = start

    while current <= end:
        chunk_end = min(current + timedelta(days=chunk_days - 1), end)
        start_str = current.strftime("%Y-%m-%d")
        end_str = chunk_end.strftime("%Y-%m-%d")

        print(f"Fetching {start_str} to {end_str}...", end=" ", flush=True)

        try:
            data = fetch_chunk(start_str, end_str)
        except Exception as e:
            print(f"ERROR: {e}")
            current = chunk_end + timedelta(days=1)
            time.sleep(2)
            continue

        hourly = data.get("hourly", {})
        times = hourly.get("time", [])
        precip = hourly.get("precipitation", [])
        rain = hourly.get("rain", [])
        weather_code = hourly.get("weather_code", [])

        rows = []
        for i, t in enumerate(times):
            ts = t.replace("T", " ") + ":00"
            p = precip[i] if i < len(precip) else 0.0
            r_val = rain[i] if i < len(rain) else 0.0
            wc = weather_code[i] if i < len(weather_code) else None

            # Use the max of total precip and rain-only for rate
            precip_rate = max(p or 0.0, r_val or 0.0)
            base_dbz = precip_to_dbz(precip_rate)
            max_dbz = weather_code_dbz_boost(wc, base_dbz, precip_rate)

            rows.append((ts, round(max_dbz, 1), round(precip_rate, 2)))

        conn.executemany(
            """INSERT OR REPLACE INTO mrms_reflectivity
               (timestamp_utc, max_reflectivity_dbz, precip_rate_mmhr)
               VALUES (?, ?, ?)""",
            rows,
        )
        conn.commit()
        total_inserted += len(rows)
        print(f"{len(rows)} rows")

        current = chunk_end + timedelta(days=1)
        time.sleep(0.5)

    final_count = conn.execute("SELECT COUNT(*) FROM mrms_reflectivity").fetchone()[0]
    print(f"\nDone. Total rows in mrms_reflectivity: {final_count}")

    # Stats
    stats = conn.execute("""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN precip_rate_mmhr > 0 THEN 1 ELSE 0 END) as hours_with_precip,
            AVG(precip_rate_mmhr) as avg_rate,
            MAX(precip_rate_mmhr) as max_rate,
            AVG(max_reflectivity_dbz) as avg_dbz,
            MAX(max_reflectivity_dbz) as max_dbz
        FROM mrms_reflectivity
    """).fetchone()
    print(f"Hours with precip: {stats[1]}/{stats[0]} ({100*stats[1]/stats[0]:.1f}%)")
    print(f"Avg precip rate: {stats[2]:.3f} mm/hr")
    print(f"Max precip rate: {stats[3]:.1f} mm/hr")
    print(f"Avg reflectivity: {stats[4]:.1f} dBZ")
    print(f"Max reflectivity: {stats[5]:.1f} dBZ")

    conn.close()


if __name__ == "__main__":
    main()
