#!/usr/bin/env python3
"""
Pull GLM lightning proxy data from Open-Meteo archive API.

Since direct GLM/NLDN lightning flash counts aren't available via simple APIs,
we use Open-Meteo archive data to build a lightning proxy:
- weather_code: WMO codes 95/96/99 = thunderstorm (direct lightning indicator)
- precipitation rate: heavy convective precip correlates with lightning
- lightning_potential: Open-Meteo's derived field (ERA5-based, may be sparse)

We derive:
- flash_count: estimated from weather_code (thunderstorm codes)
- flash_density: estimated from precip intensity during thunderstorms
- lightning_potential: from CAPE-like convective parameters or weather_code

Coverage: 2020-01-01 to 2026-03-26, hourly, for Miami (25.79, -80.29).
"""

import sqlite3
import requests
import time
import sys
from datetime import datetime, timedelta

DB_PATH = "/Users/blakebot/blakebot/miami-project/miami_collector.db"
LAT, LON = 25.79, -80.29

# ERA5/Open-Meteo archive uses WMO codes for drizzle/rain (not thunderstorm/shower codes).
# WMO codes from ERA5 reanalysis:
#   0 = clear, 1-3 = partly cloudy/overcast
#   51/53/55 = drizzle (slight/moderate/dense)
#   61/63/65 = rain (slight/moderate/heavy)
#   71/73/75 = snow
#   80/81/82 = rain showers (if available)
#   95/96/99 = thunderstorm (if available, rare in ERA5)
#
# In Miami, heavy rain (WMO 65, >10mm/hr) almost always means convective storms
# with lightning. We use precip intensity + WMO code to estimate lightning.


def create_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS lightning_glm (
            timestamp_utc TEXT PRIMARY KEY,
            flash_count INTEGER,
            flash_density REAL,
            lightning_potential REAL
        )
    """)
    conn.commit()


def estimate_lightning(weather_code, precip_mm, rain_mm):
    """Estimate lightning metrics from weather code and precipitation.

    In Miami's subtropical climate, most heavy rainfall is convective (thunderstorms).
    ERA5 reanalysis doesn't reliably flag thunderstorm codes, so we infer lightning
    from precipitation intensity and weather code severity.

    Returns (flash_count, flash_density, lightning_potential).
    """
    if weather_code is None:
        return (0, 0.0, 0.0)

    wc = int(weather_code)
    p = precip_mm or 0.0

    # Lightning potential: 0-1 scale based on precip intensity and code
    # In Miami, precip > 10 mm/hr is almost always convective (thunderstorm)
    if wc in (95, 96, 99):
        # Explicit thunderstorm codes (rare in ERA5 but handle if present)
        lightning_potential = 0.9 + min(0.1, p * 0.005)
    elif wc == 65 and p >= 10.0:
        # Heavy rain, high rate -> very likely thunderstorm
        lightning_potential = 0.85 + min(0.1, (p - 10) * 0.005)
    elif wc == 65:
        # Heavy rain code, moderate rate
        lightning_potential = 0.6 + min(0.2, p * 0.02)
    elif wc == 63 and p >= 5.0:
        # Moderate rain, decent rate -> possible thunderstorm
        lightning_potential = 0.4 + min(0.3, (p - 5) * 0.03)
    elif wc == 63:
        # Moderate rain
        lightning_potential = 0.2 + min(0.2, p * 0.02)
    elif wc in (80, 81, 82):
        # Shower codes if present
        lightning_potential = 0.3 + min(0.5, p * 0.03)
    elif p >= 15.0:
        # Very heavy precip regardless of code
        lightning_potential = 0.7
    elif p >= 5.0:
        lightning_potential = 0.15
    else:
        lightning_potential = 0.0

    # Flash count estimate (per hour, ~30km radius around Miami)
    # Based on literature: ~0.5-3 flashes per mm of convective precip per hour
    # in tropical/subtropical environments
    if lightning_potential >= 0.6:
        # Strong convective activity
        flash_count = int(5 + p * 2.5)
    elif lightning_potential >= 0.3:
        # Moderate convective activity
        flash_count = int(1 + p * 1.0)
    elif lightning_potential >= 0.15:
        # Weak/marginal
        flash_count = int(p * 0.3)
    else:
        flash_count = 0

    # Flash density: flashes per km^2 per hour (~2800 km^2 area for 30km radius)
    area_km2 = 2800.0
    flash_density = flash_count / area_km2

    return (flash_count, round(flash_density, 6), round(lightning_potential, 3))


def fetch_chunk(start_date, end_date):
    """Fetch one chunk from Open-Meteo archive API."""
    params = {
        "latitude": LAT,
        "longitude": LON,
        "hourly": "precipitation,rain,weather_code",
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


def main():
    conn = sqlite3.connect(DB_PATH)
    create_table(conn)

    # Check existing data
    existing = conn.execute("SELECT COUNT(*) FROM lightning_glm").fetchone()[0]
    print(f"Existing rows: {existing}")

    # Date range
    start = datetime(2020, 1, 1)
    end = datetime(2026, 3, 26)

    # Fetch in ~1-year chunks to avoid API limits
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
            p = precip[i] if i < len(precip) else None
            r_val = rain[i] if i < len(rain) else None
            wc = weather_code[i] if i < len(weather_code) else None

            fc, fd, lp = estimate_lightning(wc, p, r_val)
            rows.append((ts, fc, fd, lp))

        conn.executemany(
            """INSERT OR REPLACE INTO lightning_glm
               (timestamp_utc, flash_count, flash_density, lightning_potential)
               VALUES (?, ?, ?, ?)""",
            rows,
        )
        conn.commit()
        total_inserted += len(rows)
        print(f"{len(rows)} rows")

        current = chunk_end + timedelta(days=1)
        time.sleep(0.5)  # Be nice to the API

    final_count = conn.execute("SELECT COUNT(*) FROM lightning_glm").fetchone()[0]
    print(f"\nDone. Total rows in lightning_glm: {final_count}")

    # Show some stats
    stats = conn.execute("""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN flash_count > 0 THEN 1 ELSE 0 END) as hours_with_lightning,
            AVG(lightning_potential) as avg_potential,
            MAX(flash_count) as max_flashes
        FROM lightning_glm
    """).fetchone()
    print(f"Hours with lightning: {stats[1]}/{stats[0]}")
    print(f"Avg lightning potential: {stats[2]:.4f}")
    print(f"Max flash count: {stats[3]}")

    conn.close()


if __name__ == "__main__":
    main()
