#!/usr/bin/env python3
"""
Pull upper air sounding data from University of Wyoming archive.
Stores data in miami_collector.db in the upper_air_soundings table.

Uses TEXT:AVIATION format from https://weather.uwyo.edu/wsgi/sounding
Rate limited to 1 request per 2 seconds.
"""

import math
import re
import sqlite3
import sys
import time
from datetime import datetime, timedelta

import requests

DB_PATH = "/Users/blakebot/blakebot/miami-project/miami_collector.db"

STATIONS = {
    "72202": ("MFL", "Miami"),
    "72210": ("TBW", "Tampa"),
    "72206": ("JAX", "Jacksonville"),
    "72201": ("EYW", "Key West"),
    "72501": ("OKX", "NYC"),
    "72440": ("ILX", "Chicago"),
    "72469": ("DNR", "Denver"),
    "72261": ("DRT", "Austin"),  # Del Rio TX, proxy for Austin
}

START_DATE = datetime(2020, 1, 1)
END_DATE = datetime(2026, 3, 26)
HOURS = [0, 12]

BASE_URL = "https://weather.uwyo.edu/wsgi/sounding"
RATE_LIMIT = 2.0  # seconds between requests

# Standard pressure levels to extract (hPa)
TARGET_LEVELS = [925.0, 850.0, 700.0, 500.0]

FT_TO_M = 0.3048


def create_table(conn: sqlite3.Connection):
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


def get_existing_keys(conn: sqlite3.Connection) -> set:
    """Get set of (station_id, timestamp_utc) already in the database."""
    cursor = conn.execute(
        "SELECT station_id, timestamp_utc FROM upper_air_soundings"
    )
    return {(row[0], row[1]) for row in cursor.fetchall()}


def saturation_mixing_ratio(temp_c, pres_hpa):
    """Saturation mixing ratio in g/kg using Bolton's formula."""
    es = 6.112 * math.exp(17.67 * temp_c / (temp_c + 243.5))
    if es >= pres_hpa:
        return 1000.0
    return 621.97 * es / (pres_hpa - es)


def theta(temp_c, pres_hpa):
    """Potential temperature in K."""
    return (temp_c + 273.15) * (1000.0 / pres_hpa) ** 0.286


def theta_e(temp_c, dwpt_c, pres_hpa):
    """Approximate equivalent potential temperature."""
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
    """Approximate LCL temperature in C."""
    return dwpt_c - (0.001296 * dwpt_c + 0.1963) * (temp_c - dwpt_c)


def lcl_pressure(temp_c, dwpt_c, pres_hpa):
    """Approximate LCL pressure in hPa."""
    t_lcl = lcl_temp(temp_c, dwpt_c) + 273.15
    t_k = temp_c + 273.15
    return pres_hpa * (t_lcl / t_k) ** (1.0 / 0.286)


def moist_adiabat_temp(target_the, pres_hpa, iterations=50):
    """
    Find temperature on a moist adiabat given equivalent potential temperature.
    Uses bisection method for robustness.
    """
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


def interpolate_level(rows, target_pres):
    """Find or interpolate data at a target pressure level using ln(p)."""
    above = None  # lower pressure
    below = None  # higher pressure

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
    return {
        "pres": target_pres,
        "hght_m": below["hght_m"] + w * (above["hght_m"] - below["hght_m"]),
        "temp": below["temp"] + w * (above["temp"] - below["temp"]),
        "dwpt": below["dwpt"] + w * (above["dwpt"] - below["dwpt"]),
        "drct": below["drct"],
        "speed_kt": below["speed_kt"] + w * (above["speed_kt"] - below["speed_kt"]),
        "mixr": below["mixr"] + w * (above["mixr"] - below["mixr"]),
    }


def parse_aviation_sounding(html: str):
    """
    Parse TEXT:AVIATION sounding HTML from UWyo.
    Returns dict with surface/level data and derived indices.
    """
    pre_match = re.search(r"<PRE>(.*?)</PRE>", html, re.DOTALL)
    if not pre_match:
        return None

    pre_text = pre_match.group(1)
    rows = []

    for line in pre_text.split("\n"):
        line = line.strip()
        if not line or line.startswith("-") or line.startswith("PRES") or line.startswith("hPa"):
            continue
        parts = line.split()
        if len(parts) < 8:
            continue
        try:
            pres = float(parts[0])
            hght_ft = float(parts[1])
            temp = float(parts[2])
            dwpt = float(parts[3])
            rh = float(parts[4])
            mixr = float(parts[5])
            drct = float(parts[6])
            speed_kt = float(parts[7])
            rows.append({
                "pres": pres,
                "hght_m": hght_ft * FT_TO_M,
                "temp": temp,
                "dwpt": dwpt,
                "rh": rh,
                "mixr": mixr,
                "drct": drct,
                "speed_kt": speed_kt,
            })
        except (ValueError, IndexError):
            continue

    if len(rows) < 5:
        return None

    # Sort by pressure descending (surface first)
    rows.sort(key=lambda r: r["pres"], reverse=True)

    result = {}

    # Surface data (highest pressure)
    sfc = rows[0]
    result["sfc_temp_c"] = sfc["temp"]
    result["sfc_dewpoint_c"] = sfc["dwpt"]
    result["sfc_wind_dir"] = sfc["drct"]
    result["sfc_wind_kt"] = sfc["speed_kt"]
    result["sfc_pressure_hpa"] = sfc["pres"]

    # Extract standard levels
    for level in TARGET_LEVELS:
        prefix = f"t{int(level)}"
        val = interpolate_level(rows, level)
        if val:
            result[f"{prefix}_temp_c"] = round(val["temp"], 2)
            result[f"{prefix}_dewpoint_c"] = round(val["dwpt"], 2)
            result[f"{prefix}_wind_dir"] = round(val["drct"], 1)
            result[f"{prefix}_wind_kt"] = round(val["speed_kt"], 1)
            result[f"{prefix}_height_m"] = round(val["hght_m"], 1)

    # Compute derived indices
    result.update(compute_derived(rows))

    return result


def compute_derived(rows):
    """Compute CAPE, CIN, LI, PW, K-Index, Total Totals."""
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
            result["total_totals"] = round(
                (t850["temp"] - t500["temp"]) + (t850["dwpt"] - t500["temp"]),
                1,
            )
        except (TypeError, KeyError):
            pass

    # Precipitable Water
    try:
        pw = 0.0
        for i in range(len(rows) - 1):
            r1, r2 = rows[i], rows[i + 1]
            if r2["pres"] < 300:
                break
            avg_mixr = (r1["mixr"] + r2["mixr"]) / 2.0
            dp = abs(r1["pres"] - r2["pres"])
            pw += (avg_mixr / 1000.0) * (dp * 100.0) / 9.81
        result["precipitable_water_mm"] = round(pw, 2)
    except (TypeError, KeyError):
        pass

    # CAPE, CIN, Lifted Index
    sfc = rows[0]
    try:
        sfc_t, sfc_td, sfc_p = sfc["temp"], sfc["dwpt"], sfc["pres"]
        p_lcl = lcl_pressure(sfc_t, sfc_td, sfc_p)
        t_lcl_c = lcl_temp(sfc_t, sfc_td)
        parcel_the = theta_e(t_lcl_c, t_lcl_c, p_lcl)

        cape, cin = 0.0, 0.0
        for i in range(len(rows) - 1):
            r, r2 = rows[i], rows[i + 1]
            p = (r["pres"] + r2["pres"]) / 2.0
            env_t = (r["temp"] + r2["temp"]) / 2.0
            dz = abs(r2["hght_m"] - r["hght_m"])
            if p > sfc_p or p < 100:
                continue
            if p >= p_lcl:
                avg_z = (r["hght_m"] + r2["hght_m"]) / 2.0
                parcel_t = sfc_t - 9.8 / 1004.0 * (avg_z - rows[0]["hght_m"])
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

        if t500:
            parcel_t_500 = moist_adiabat_temp(parcel_the, 500.0)
            result["lifted_index"] = round(t500["temp"] - parcel_t_500, 1)

    except (TypeError, KeyError, ValueError, ZeroDivisionError, OverflowError):
        pass

    return result


def fetch_sounding(station_id: str, dt: datetime, session: requests.Session):
    """Fetch a single sounding from UWyo. Returns parsed dict or None."""
    datetime_str = dt.strftime("%Y-%m-%d %H:%M:%S")
    params = {
        "datetime": datetime_str,
        "id": station_id,
        "src": "UNKNOWN",
        "type": "TEXT:AVIATION",
    }
    try:
        r = session.get(BASE_URL, params=params, timeout=60)
        if r.status_code != 200:
            return None
        if "<PRE>" not in r.text:
            return None
        return parse_aviation_sounding(r.text)
    except requests.RequestException:
        return None


def main():
    conn = sqlite3.connect(DB_PATH)
    create_table(conn)

    existing = get_existing_keys(conn)
    print(f"Found {len(existing)} existing soundings in database")

    # Build task list
    tasks = []
    current = START_DATE
    while current <= END_DATE:
        for hour in HOURS:
            dt = current.replace(hour=hour, minute=0, second=0)
            ts = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            for station_id in STATIONS:
                if (station_id, ts) not in existing:
                    tasks.append((station_id, dt, ts))
        current += timedelta(days=1)

    total = len(tasks)
    print(f"Total soundings to fetch: {total}")
    if total == 0:
        print("Nothing to do.")
        conn.close()
        return

    session = requests.Session()
    session.headers.update({
        "User-Agent": "MiamiWeatherResearch/1.0 (academic research)",
    })

    fetched = 0
    inserted = 0
    errors = 0
    batch = []
    start_time = time.time()

    for i, (station_id, dt, ts) in enumerate(tasks):
        stn_name, city = STATIONS[station_id]

        data = fetch_sounding(station_id, dt, session)
        fetched += 1

        if data:
            row = (
                station_id, stn_name, city, ts,
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
            inserted += 1
        else:
            errors += 1

        # Batch insert every 50
        if len(batch) >= 50:
            conn.executemany(
                """INSERT OR REPLACE INTO upper_air_soundings VALUES
                (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                batch,
            )
            conn.commit()
            batch = []

        if fetched % 100 == 0:
            elapsed = time.time() - start_time
            rate = fetched / elapsed if elapsed > 0 else 0
            eta_s = (total - fetched) / rate if rate > 0 else 0
            eta_h = eta_s / 3600
            print(
                f"[{fetched}/{total}] inserted={inserted} errors={errors} "
                f"rate={rate:.1f}/s ETA={eta_h:.1f}h "
                f"last={city}/{stn_name} {ts}"
            )

        time.sleep(RATE_LIMIT)

    # Final batch
    if batch:
        conn.executemany(
            """INSERT OR REPLACE INTO upper_air_soundings VALUES
            (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            batch,
        )
        conn.commit()

    elapsed = time.time() - start_time
    print(f"\nDone in {elapsed/3600:.1f}h. Fetched={fetched} Inserted={inserted} Errors/Missing={errors}")

    cursor = conn.execute("SELECT COUNT(*) FROM upper_air_soundings")
    print(f"Total rows in upper_air_soundings: {cursor.fetchone()[0]}")
    conn.close()


if __name__ == "__main__":
    main()
