#!/usr/bin/env python3
"""
Pull upper air sounding data from IEM RAOB archive.
Much faster than UWyo — serves CSV directly.

Stores data in miami_collector.db in the upper_air_soundings table (same schema as pull_upper_air.py).
Pulls from 4 Florida stations: MFL (Miami), TBW (Tampa), JAX (Jacksonville), EYW (Key West).

Coverage: 2020-01-01 to 2026-03-27, both 00Z and 12Z.
"""

import csv
import io
import math
import sqlite3
import sys
import time
from datetime import datetime

import requests

DB_PATH = "/Users/blakebot/blakebot/miami-project/miami_collector.db"

IEM_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/raob.py"

# ICAO code -> (WMO ID, station_name, city)
STATIONS = {
    "KMFL": ("72202", "MFL", "Miami"),
    "KTBW": ("72210", "TBW", "Tampa"),
    "KJAX": ("72206", "JAX", "Jacksonville"),
    "KEYW": ("72201", "EYW", "Key West"),
}

# Date range
START_DATE = "2020-01-01"
END_DATE = "2026-03-27"

# Chunk by year to avoid giant responses
YEAR_RANGES = [
    ("2020-01-01", "2021-01-01"),
    ("2021-01-01", "2022-01-01"),
    ("2022-01-01", "2023-01-01"),
    ("2023-01-01", "2024-01-01"),
    ("2024-01-01", "2025-01-01"),
    ("2025-01-01", "2026-01-01"),
    ("2026-01-01", "2026-03-28"),  # inclusive end via next day
]

TARGET_LEVELS = [925.0, 850.0, 700.0, 500.0]


def create_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS upper_air_soundings (
            station_id TEXT NOT NULL,
            station_name TEXT,
            city TEXT,
            timestamp_utc TEXT NOT NULL,
            sfc_temp_c REAL,
            sfc_dewpoint_c REAL,
            sfc_wind_dir REAL,
            sfc_wind_kt REAL,
            sfc_pressure_hpa REAL,
            t925_temp_c REAL,
            t925_dewpoint_c REAL,
            t925_wind_dir REAL,
            t925_wind_kt REAL,
            t925_height_m REAL,
            t850_temp_c REAL,
            t850_dewpoint_c REAL,
            t850_wind_dir REAL,
            t850_wind_kt REAL,
            t850_height_m REAL,
            t700_temp_c REAL,
            t700_dewpoint_c REAL,
            t700_wind_dir REAL,
            t700_wind_kt REAL,
            t700_height_m REAL,
            t500_temp_c REAL,
            t500_dewpoint_c REAL,
            t500_wind_dir REAL,
            t500_wind_kt REAL,
            t500_height_m REAL,
            cape REAL,
            cin REAL,
            lifted_index REAL,
            precipitable_water_mm REAL,
            k_index REAL,
            total_totals REAL,
            PRIMARY KEY (station_id, timestamp_utc)
        )
    """)
    conn.commit()


def safe_float(val):
    """Convert a value to float, returning None if missing ('M' or empty)."""
    if val is None or val == "M" or val == "" or val == "m":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def interpolate_level(rows, target_pres):
    """Find or interpolate data at a target pressure level using ln(p)."""
    above = None  # lower pressure (higher altitude)
    below = None  # higher pressure (lower altitude)

    for row in rows:
        if abs(row["pres"] - target_pres) < 1.5:
            return row
        if row["pres"] > target_pres:
            if below is None or row["pres"] < below["pres"]:
                below = row
        elif row["pres"] < target_pres:
            if above is None or row["pres"] > above["pres"]:
                above = row

    if above is None or below is None:
        return None

    lnp_below = math.log(below["pres"])
    lnp_above = math.log(above["pres"])
    lnp_target = math.log(target_pres)
    if abs(lnp_below - lnp_above) < 1e-10:
        return below

    w = (lnp_below - lnp_target) / (lnp_below - lnp_above)

    result = {"pres": target_pres}
    for key in ["hght_m", "temp", "dwpt", "speed_kt", "mixr"]:
        v_below = below.get(key)
        v_above = above.get(key)
        if v_below is not None and v_above is not None:
            result[key] = v_below + w * (v_above - v_below)
        elif v_below is not None:
            result[key] = v_below
        elif v_above is not None:
            result[key] = v_above
        else:
            result[key] = None
    # Wind direction: use nearest
    result["drct"] = below.get("drct") if below.get("drct") is not None else above.get("drct")
    return result


def saturation_mixing_ratio(temp_c, pres_hpa):
    es = 6.112 * math.exp(17.67 * temp_c / (temp_c + 243.5))
    if es >= pres_hpa:
        return 1000.0
    return 621.97 * es / (pres_hpa - es)


def theta(temp_c, pres_hpa):
    return (temp_c + 273.15) * (1000.0 / pres_hpa) ** 0.286


def theta_e(temp_c, dwpt_c, pres_hpa):
    t_k = temp_c + 273.15
    if t_k < 1.0:
        t_k = 1.0
    w = saturation_mixing_ratio(dwpt_c, pres_hpa) / 1000.0
    th = theta(temp_c, pres_hpa)
    exponent = (2675.0 * w) / t_k
    if exponent > 500:
        exponent = 500
    return th * math.exp(exponent)


def lcl_temp(temp_c, dwpt_c):
    return dwpt_c - (0.001296 * dwpt_c + 0.1963) * (temp_c - dwpt_c)


def lcl_pressure(temp_c, dwpt_c, pres_hpa):
    t_lcl = lcl_temp(temp_c, dwpt_c) + 273.15
    t_k = temp_c + 273.15
    return pres_hpa * (t_lcl / t_k) ** (1.0 / 0.286)


def moist_adiabat_temp(target_the, pres_hpa, iterations=50):
    t_lo = -100.0
    t_hi = min(60.0, pres_hpa * 0.15 - 30)
    if t_hi < t_lo + 1:
        t_hi = t_lo + 1
    try:
        the_lo = theta_e(t_lo, t_lo, pres_hpa)
        the_hi = theta_e(t_hi, t_hi, pres_hpa)
        if target_the < the_lo:
            return t_lo
        if target_the > the_hi:
            return t_hi
        for _ in range(iterations):
            t_mid = (t_lo + t_hi) / 2.0
            the_mid = theta_e(t_mid, t_mid, pres_hpa)
            if abs(the_mid - target_the) < 0.01:
                return t_mid
            if the_mid < target_the:
                t_lo = t_mid
            else:
                t_hi = t_mid
        return (t_lo + t_hi) / 2.0
    except (OverflowError, ValueError, ZeroDivisionError):
        return -999.0


def compute_derived(rows):
    """Compute CAPE, CIN, LI, PW, K-Index, Total Totals from sounding rows."""
    result = {
        "cape": None, "cin": None, "lifted_index": None,
        "precipitable_water_mm": None, "k_index": None, "total_totals": None,
    }

    if len(rows) < 5:
        return result

    t850 = interpolate_level(rows, 850.0)
    t700 = interpolate_level(rows, 700.0)
    t500 = interpolate_level(rows, 500.0)

    # K-Index
    if t850 and t700 and t500:
        try:
            if all(v.get("temp") is not None for v in [t850, t700, t500]) and \
               t850.get("dwpt") is not None and t700.get("dwpt") is not None:
                result["k_index"] = round(
                    (t850["temp"] - t500["temp"])
                    + t850["dwpt"]
                    - (t700["temp"] - t700["dwpt"]),
                    1,
                )
        except (TypeError, KeyError):
            pass

    # Total Totals
    if t850 and t500:
        try:
            if t850.get("temp") is not None and t850.get("dwpt") is not None and t500.get("temp") is not None:
                result["total_totals"] = round(
                    (t850["temp"] - t500["temp"]) + (t850["dwpt"] - t500["temp"]),
                    1,
                )
        except (TypeError, KeyError):
            pass

    # Precipitable Water
    try:
        pw = 0.0
        pw_valid = False
        for i in range(len(rows) - 1):
            r1, r2 = rows[i], rows[i + 1]
            if r2["pres"] < 300:
                break
            m1 = r1.get("mixr")
            m2 = r2.get("mixr")
            if m1 is not None and m2 is not None:
                avg_mixr = (m1 + m2) / 2.0
                dp = abs(r1["pres"] - r2["pres"])
                pw += (avg_mixr / 1000.0) * (dp * 100.0) / 9.81
                pw_valid = True
        if pw_valid:
            result["precipitable_water_mm"] = round(pw, 2)
    except (TypeError, KeyError):
        pass

    # CAPE, CIN, Lifted Index
    sfc = rows[0]
    try:
        sfc_t = sfc.get("temp")
        sfc_td = sfc.get("dwpt")
        sfc_p = sfc["pres"]
        if sfc_t is None or sfc_td is None:
            return result

        p_lcl = lcl_pressure(sfc_t, sfc_td, sfc_p)
        t_lcl_c = lcl_temp(sfc_t, sfc_td)
        parcel_the = theta_e(t_lcl_c, t_lcl_c, p_lcl)

        cape, cin = 0.0, 0.0
        for i in range(len(rows) - 1):
            r, r2 = rows[i], rows[i + 1]
            if r.get("temp") is None or r2.get("temp") is None:
                continue
            if r.get("hght_m") is None or r2.get("hght_m") is None:
                continue
            p = (r["pres"] + r2["pres"]) / 2.0
            env_t = (r["temp"] + r2["temp"]) / 2.0
            dz = abs(r2["hght_m"] - r["hght_m"])
            if p > sfc_p or p < 100:
                continue
            if p >= p_lcl:
                avg_z = (r["hght_m"] + r2["hght_m"]) / 2.0
                parcel_t = sfc_t - 9.8 / 1004.0 * (avg_z - rows[0].get("hght_m", 0))
            else:
                parcel_t = moist_adiabat_temp(parcel_the, p)
            env_tv = env_t + 273.15
            parcel_tv = parcel_t + 273.15
            if env_tv < 100:
                continue
            buoyancy = 9.81 * (parcel_tv - env_tv) / env_tv * dz
            if buoyancy > 0:
                cape += buoyancy
            elif cape < 1.0:
                cin += buoyancy

        result["cape"] = round(max(cape, 0), 1)
        result["cin"] = round(min(cin, 0), 1)

        if t500 and t500.get("temp") is not None:
            parcel_t_500 = moist_adiabat_temp(parcel_the, 500.0)
            result["lifted_index"] = round(t500["temp"] - parcel_t_500, 1)

    except (TypeError, KeyError, ValueError, ZeroDivisionError, OverflowError):
        pass

    return result


def process_sounding(raw_rows):
    """
    Process a list of IEM CSV rows (all for the same station+time) into
    the sounding record dict matching the DB schema.
    """
    # Parse into structured rows
    parsed = []
    for r in raw_rows:
        pres = safe_float(r["pressure_mb"])
        if pres is None:
            continue
        hght = safe_float(r["height_m"])
        temp = safe_float(r["tmpc"])
        dwpt = safe_float(r["dwpc"])
        drct = safe_float(r["drct"])
        speed = safe_float(r["speed_kts"])

        # Compute mixing ratio from temp/dwpt if available
        mixr = None
        if temp is not None and dwpt is not None and pres > 0:
            try:
                mixr = saturation_mixing_ratio(dwpt, pres)
            except (OverflowError, ValueError, ZeroDivisionError):
                pass

        parsed.append({
            "pres": pres,
            "hght_m": hght,
            "temp": temp,
            "dwpt": dwpt,
            "drct": drct,
            "speed_kt": speed,
            "mixr": mixr,
        })

    if len(parsed) < 5:
        return None

    # Sort by pressure descending (surface first)
    parsed.sort(key=lambda r: r["pres"], reverse=True)

    result = {}

    # Surface data (highest pressure)
    sfc = parsed[0]
    result["sfc_temp_c"] = sfc["temp"]
    result["sfc_dewpoint_c"] = sfc["dwpt"]
    result["sfc_wind_dir"] = sfc["drct"]
    result["sfc_wind_kt"] = sfc["speed_kt"]
    result["sfc_pressure_hpa"] = sfc["pres"]

    # Extract standard levels
    for level in TARGET_LEVELS:
        prefix = f"t{int(level)}"
        val = interpolate_level(parsed, level)
        if val:
            result[f"{prefix}_temp_c"] = round(val["temp"], 2) if val.get("temp") is not None else None
            result[f"{prefix}_dewpoint_c"] = round(val["dwpt"], 2) if val.get("dwpt") is not None else None
            result[f"{prefix}_wind_dir"] = round(val["drct"], 1) if val.get("drct") is not None else None
            result[f"{prefix}_wind_kt"] = round(val["speed_kt"], 1) if val.get("speed_kt") is not None else None
            result[f"{prefix}_height_m"] = round(val["hght_m"], 1) if val.get("hght_m") is not None else None

    # Compute derived indices
    result.update(compute_derived(parsed))

    return result


def fetch_station_chunk(session, icao, start_date, end_date, max_retries=3):
    """Fetch all soundings for a station in a date range from IEM RAOB."""
    params = {
        "station": icao,
        "sts": f"{start_date}T00:00:00Z",
        "ets": f"{end_date}T00:00:00Z",
        "format": "csv",
    }

    for attempt in range(max_retries):
        try:
            r = session.get(IEM_URL, params=params, timeout=120)
            if r.status_code != 200:
                print(f"  HTTP {r.status_code} for {icao} {start_date}-{end_date}")
                if attempt < max_retries - 1:
                    time.sleep(5 * (attempt + 1))
                    continue
                return {}
            break
        except requests.RequestException as e:
            print(f"  Request error for {icao} {start_date}-{end_date}: {e}")
            if attempt < max_retries - 1:
                print(f"  Retrying in {5 * (attempt + 1)}s...")
                time.sleep(5 * (attempt + 1))
                # Create a fresh session for retry
                session.close()
                session = requests.Session()
                session.headers.update({
                    "User-Agent": "MiamiWeatherResearch/1.0 (academic research)",
                })
            else:
                print(f"  Failed after {max_retries} attempts")
                return {}

    reader = csv.DictReader(io.StringIO(r.text))

    # Group rows by valid time
    soundings = {}
    for row in reader:
        ts = row["validUTC"]
        if ts not in soundings:
            soundings[ts] = []
        soundings[ts].append(row)

    return soundings


def main():
    conn = sqlite3.connect(DB_PATH)
    create_table(conn)

    # Get existing keys
    cursor = conn.execute("SELECT station_id, timestamp_utc FROM upper_air_soundings")
    existing = {(row[0], row[1]) for row in cursor.fetchall()}
    print(f"Existing soundings in DB: {len(existing)}")

    session = requests.Session()
    session.headers.update({
        "User-Agent": "MiamiWeatherResearch/1.0 (academic research)",
    })

    total_inserted = 0
    total_skipped = 0
    total_failed = 0
    t0 = time.time()

    for icao, (wmo_id, stn_name, city) in STATIONS.items():
        station_inserted = 0
        station_skipped = 0
        station_failed = 0

        for start, end in YEAR_RANGES:
            chunk_t0 = time.time()
            soundings = fetch_station_chunk(session, icao, start, end)
            fetch_time = time.time() - chunk_t0

            batch = []
            for ts_str, raw_rows in soundings.items():
                # Convert IEM timestamp format to our format
                # IEM: "2024-01-01 00:00:00" -> "2024-01-01T00:00:00Z"
                ts_utc = ts_str.replace(" ", "T") + "Z"

                # Check if already exists
                if (wmo_id, ts_utc) in existing:
                    station_skipped += 1
                    continue

                data = process_sounding(raw_rows)
                if data is None:
                    station_failed += 1
                    continue

                row = (
                    wmo_id, stn_name, city, ts_utc,
                    data.get("sfc_temp_c"), data.get("sfc_dewpoint_c"),
                    data.get("sfc_wind_dir"), data.get("sfc_wind_kt"),
                    data.get("sfc_pressure_hpa"),
                    data.get("t925_temp_c"), data.get("t925_dewpoint_c"),
                    data.get("t925_wind_dir"), data.get("t925_wind_kt"),
                    data.get("t925_height_m"),
                    data.get("t850_temp_c"), data.get("t850_dewpoint_c"),
                    data.get("t850_wind_dir"), data.get("t850_wind_kt"),
                    data.get("t850_height_m"),
                    data.get("t700_temp_c"), data.get("t700_dewpoint_c"),
                    data.get("t700_wind_dir"), data.get("t700_wind_kt"),
                    data.get("t700_height_m"),
                    data.get("t500_temp_c"), data.get("t500_dewpoint_c"),
                    data.get("t500_wind_dir"), data.get("t500_wind_kt"),
                    data.get("t500_height_m"),
                    data.get("cape"), data.get("cin"),
                    data.get("lifted_index"), data.get("precipitable_water_mm"),
                    data.get("k_index"), data.get("total_totals"),
                )
                batch.append(row)

            if batch:
                conn.executemany(
                    """INSERT OR REPLACE INTO upper_air_soundings VALUES
                    (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    batch,
                )
                conn.commit()

            station_inserted += len(batch)
            print(
                f"  {icao} ({city}) {start} -> {end}: "
                f"{len(soundings)} soundings fetched in {fetch_time:.1f}s, "
                f"{len(batch)} inserted, {station_skipped} skipped"
            )

            # Delay between requests to be polite and avoid disconnects
            time.sleep(2.0)

        total_inserted += station_inserted
        total_skipped += station_skipped
        total_failed += station_failed
        print(
            f"  {icao} TOTAL: inserted={station_inserted} skipped={station_skipped} "
            f"failed={station_failed}"
        )
        print()

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s ({elapsed/60:.1f}m)")
    print(f"Total inserted: {total_inserted}")
    print(f"Total skipped (already existed): {total_skipped}")
    print(f"Total failed (too few levels): {total_failed}")

    # Final stats
    cursor = conn.execute("SELECT COUNT(*) FROM upper_air_soundings")
    print(f"\nTotal rows in upper_air_soundings: {cursor.fetchone()[0]}")

    cursor = conn.execute(
        "SELECT station_id, station_name, city, COUNT(*) as cnt "
        "FROM upper_air_soundings GROUP BY station_id ORDER BY station_id"
    )
    print("\nPer-station counts:")
    for row in cursor.fetchall():
        print(f"  {row[0]} ({row[1]}/{row[2]}): {row[3]} soundings")

    # Date range check
    cursor = conn.execute(
        "SELECT MIN(timestamp_utc), MAX(timestamp_utc) FROM upper_air_soundings "
        "WHERE station_id IN ('72201','72202','72206','72210')"
    )
    row = cursor.fetchone()
    print(f"\nFL stations date range: {row[0]} to {row[1]}")

    conn.close()


if __name__ == "__main__":
    import functools
    print = functools.partial(print, flush=True)
    main()
