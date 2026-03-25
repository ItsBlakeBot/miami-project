"""Enriched historical data backfill for DS3M pre-training.

Pulls from multiple free archives to populate as many of the 33 training
features as possible:

  1. IEM ASOS: temp, dewpoint, wind dir/speed/gust, pressure, sky cover,
     visibility, precipitation (hourly, 2023-present, 10 stations)
  2. Open-Meteo Historical Forecast: CAPE, CIN, lifted index, cloud cover,
     shortwave radiation (hourly, 2018-present)
  3. NDBC: SST from Virginia Key (VAKF1) buoy (6-min, 2015-present)

All timestamps stored in UTC (Z time) to avoid DST ambiguity.

Usage:
  python -m engine.ds3m.enriched_backfill --db miami_collector.db
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import sqlite3
import time
from collections import defaultdict
from datetime import datetime, timedelta

import aiohttp
import asyncio
import numpy as np

log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Station mapping (ICAO → IEM 3-letter)
# ──────────────────────────────────────────────────────────────────────

STATION_MAP = {
    "KMIA": "MIA", "KOPF": "OPF", "KTMB": "TMB", "KHWO": "HWO",
    "KFLL": "FLL", "KHST": "HST", "KFXE": "FXE", "KPMP": "PMP",
    "KBCT": "BCT", "KPBI": "PBI", "KX51": "X51", "KIMM": "IMM",
    "KPHK": "PHK", "KOBE": "OBE", "KSUA": "SUA", "KMTH": "MTH",
    "KNQX": "NQX", "KEYW": "EYW", "KAPF": "APF", "KRSW": "RSW",
    "KFMY": "FMY", "KPGD": "PGD", "KFPR": "FPR", "KSEF": "SEF",
    "KVRB": "VRB", "KMLB": "MLB", "KBOW": "BOW", "KSRQ": "SRQ",
    "KLAL": "LAL", "KMCO": "MCO", "KSPG": "SPG", "KORL": "ORL",
    "KTPA": "TPA", "KPIE": "PIE", "KSFB": "SFB",
}
SE_FLORIDA_STATIONS = list(STATION_MAP.keys())

IEM_ASOS_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
OPENMETEO_HIST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
OPENMETEO_ERA5_URL = "https://archive-api.open-meteo.com/v1/archive"
NDBC_HIST_URL = "https://www.ndbc.noaa.gov/data/historical/stdmet"

KMIA_LAT, KMIA_LON = 25.7959, -80.2870


# ──────────────────────────────────────────────────────────────────────
# DB Schema
# ──────────────────────────────────────────────────────────────────────

def ensure_enriched_tables(conn: sqlite3.Connection):
    """Create tables for enriched backfill data."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS enriched_asos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            station TEXT NOT NULL,
            timestamp_utc TEXT NOT NULL,
            temp_f REAL,
            dewpoint_f REAL,
            wind_dir_deg REAL,
            wind_speed_kt REAL,
            wind_gust_kt REAL,
            pressure_hpa REAL,
            altimeter_inhg REAL,
            visibility_mi REAL,
            sky_cover_1 TEXT,
            sky_cover_2 TEXT,
            sky_cover_3 TEXT,
            sky_height_1 REAL,
            sky_height_2 REAL,
            sky_height_3 REAL,
            rel_humidity REAL,
            precip_1h_in REAL,
            feels_like_f REAL,
            UNIQUE(station, timestamp_utc)
        );
        CREATE INDEX IF NOT EXISTS idx_enriched_asos
            ON enriched_asos(station, timestamp_utc);

        CREATE TABLE IF NOT EXISTS enriched_atmosphere (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp_utc TEXT NOT NULL,
            cape_jkg REAL,
            cin_jkg REAL,
            lifted_index REAL,
            cloud_cover_pct REAL,
            cloud_cover_low_pct REAL,
            cloud_cover_mid_pct REAL,
            cloud_cover_high_pct REAL,
            shortwave_rad_wm2 REAL,
            direct_rad_wm2 REAL,
            pressure_msl_hpa REAL,
            soil_temp_0_7cm_c REAL,
            source TEXT DEFAULT 'openmeteo',
            UNIQUE(timestamp_utc, source)
        );
        CREATE INDEX IF NOT EXISTS idx_enriched_atmos
            ON enriched_atmosphere(timestamp_utc);

        CREATE TABLE IF NOT EXISTS enriched_sst (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            station TEXT NOT NULL,
            timestamp_utc TEXT NOT NULL,
            water_temp_f REAL,
            air_temp_f REAL,
            wind_dir_deg REAL,
            wind_speed_kt REAL,
            pressure_hpa REAL,
            UNIQUE(station, timestamp_utc)
        );
        CREATE INDEX IF NOT EXISTS idx_enriched_sst
            ON enriched_sst(station, timestamp_utc);
    """)
    conn.commit()


# ──────────────────────────────────────────────────────────────────────
# 1. IEM ASOS Enriched Pull
# ──────────────────────────────────────────────────────────────────────

async def fetch_enriched_asos(
    iem_id: str,
    start_date: str = "2023-01-01",
    end_date: str | None = None,
) -> list[dict]:
    """Fetch full ASOS observations from IEM in UTC."""
    if end_date is None:
        end_date = datetime.utcnow().strftime("%Y-%m-%d")

    sd = datetime.strptime(start_date, "%Y-%m-%d")
    ed = datetime.strptime(end_date, "%Y-%m-%d")

    params = {
        "station": iem_id,
        "data": "tmpf,dwpf,relh,drct,sknt,gust,alti,mslp,vsby,skyc1,skyc2,skyc3,skyl1,skyl2,skyl3,p01i,feel",
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

    async with aiohttp.ClientSession() as session:
        async with session.get(IEM_ASOS_URL, params=params,
                               timeout=aiohttp.ClientTimeout(total=180)) as resp:
            if resp.status != 200:
                log.error(f"IEM ASOS fetch failed for {iem_id}: HTTP {resp.status}")
                return []
            text = await resp.text()

    records = []
    reader = csv.DictReader(io.StringIO(text))
    for row in reader:
        try:
            ts = row.get("valid", "").strip()
            if not ts:
                continue

            def safe_float(key):
                v = row.get(key, "").strip()
                return float(v) if v else None

            records.append({
                "timestamp_utc": ts,
                "temp_f": safe_float("tmpf"),
                "dewpoint_f": safe_float("dwpf"),
                "wind_dir_deg": safe_float("drct"),
                "wind_speed_kt": safe_float("sknt"),
                "wind_gust_kt": safe_float("gust"),
                "pressure_hpa": safe_float("mslp"),
                "altimeter_inhg": safe_float("alti"),
                "visibility_mi": safe_float("vsby"),
                "sky_cover_1": row.get("skyc1", "").strip() or None,
                "sky_cover_2": row.get("skyc2", "").strip() or None,
                "sky_cover_3": row.get("skyc3", "").strip() or None,
                "sky_height_1": safe_float("skyl1"),
                "sky_height_2": safe_float("skyl2"),
                "sky_height_3": safe_float("skyl3"),
                "rel_humidity": safe_float("relh"),
                "precip_1h_in": safe_float("p01i"),
                "feels_like_f": safe_float("feel"),
            })
        except Exception:
            continue

    return records


async def backfill_asos_station(icao: str, db_path: str, start_date: str = "2023-01-01") -> int:
    """Pull enriched ASOS for one station."""
    iem_id = STATION_MAP.get(icao, icao.lstrip("K"))
    log.info(f"Fetching enriched ASOS for {icao} (IEM: {iem_id}) from {start_date} in UTC...")

    records = await fetch_enriched_asos(iem_id, start_date)
    if not records:
        log.warning(f"No ASOS data for {icao}")
        return 0

    log.info(f"  {icao}: {len(records)} records received")

    conn = sqlite3.connect(db_path, timeout=10)
    ensure_enriched_tables(conn)

    inserted = 0
    for r in records:
        try:
            conn.execute(
                """INSERT OR IGNORE INTO enriched_asos
                   (station, timestamp_utc, temp_f, dewpoint_f, wind_dir_deg,
                    wind_speed_kt, wind_gust_kt, pressure_hpa, altimeter_inhg,
                    visibility_mi, sky_cover_1, sky_cover_2, sky_cover_3,
                    sky_height_1, sky_height_2, sky_height_3,
                    rel_humidity, precip_1h_in, feels_like_f)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (icao, r["timestamp_utc"], r["temp_f"], r["dewpoint_f"],
                 r["wind_dir_deg"], r["wind_speed_kt"], r["wind_gust_kt"],
                 r["pressure_hpa"], r["altimeter_inhg"], r["visibility_mi"],
                 r["sky_cover_1"], r["sky_cover_2"], r["sky_cover_3"],
                 r["sky_height_1"], r["sky_height_2"], r["sky_height_3"],
                 r["rel_humidity"], r["precip_1h_in"], r["feels_like_f"]),
            )
            inserted += 1
        except sqlite3.IntegrityError:
            pass

    conn.commit()
    conn.close()
    log.info(f"  {icao}: {inserted} records stored")
    return inserted


# ──────────────────────────────────────────────────────────────────────
# 2. Open-Meteo CAPE / Atmospheric Data
# ──────────────────────────────────────────────────────────────────────

async def fetch_openmeteo_atmosphere(
    start_date: str = "2023-01-01",
    end_date: str | None = None,
) -> list[dict]:
    """Fetch CAPE, CIN, cloud cover, radiation from Open-Meteo.

    Uses the Historical Forecast API for CAPE/CIN (2018+)
    and ERA5 archive for cloud/radiation (extends further back).
    """
    if end_date is None:
        end_date = datetime.utcnow().strftime("%Y-%m-%d")

    records = []

    # ── Historical Forecast API: CAPE, CIN, Lifted Index ──
    log.info(f"Fetching Open-Meteo Historical Forecast (CAPE/CIN) {start_date} → {end_date}...")

    # API has a 1-year max per request, so chunk
    sd = datetime.strptime(start_date, "%Y-%m-%d")
    ed = datetime.strptime(end_date, "%Y-%m-%d")

    async with aiohttp.ClientSession() as session:
        chunk_start = sd
        while chunk_start < ed:
            chunk_end = min(chunk_start + timedelta(days=90), ed)

            params = {
                "latitude": KMIA_LAT,
                "longitude": KMIA_LON,
                "start_date": chunk_start.strftime("%Y-%m-%d"),
                "end_date": chunk_end.strftime("%Y-%m-%d"),
                "hourly": "cape,lifted_index,convective_inhibition,cloud_cover,cloud_cover_low,cloud_cover_mid,cloud_cover_high,shortwave_radiation,direct_radiation,pressure_msl,soil_temperature_0cm",
                "timezone": "UTC",
            }

            try:
                async with session.get(OPENMETEO_HIST_URL, params=params,
                                       timeout=aiohttp.ClientTimeout(total=60)) as resp:
                    if resp.status != 200:
                        log.warning(f"Open-Meteo forecast API error: HTTP {resp.status}")
                        chunk_start = chunk_end + timedelta(days=1)
                        continue
                    data = await resp.json()
            except Exception as e:
                log.warning(f"Open-Meteo fetch error: {e}")
                chunk_start = chunk_end + timedelta(days=1)
                continue

            hourly = data.get("hourly", {})
            times = hourly.get("time", [])

            for i, ts in enumerate(times):
                def val(key):
                    arr = hourly.get(key, [])
                    return arr[i] if i < len(arr) and arr[i] is not None else None

                records.append({
                    "timestamp_utc": ts.replace("T", " "),
                    "cape_jkg": val("cape"),
                    "cin_jkg": val("convective_inhibition"),
                    "lifted_index": val("lifted_index"),
                    "cloud_cover_pct": val("cloud_cover"),
                    "cloud_cover_low_pct": val("cloud_cover_low"),
                    "cloud_cover_mid_pct": val("cloud_cover_mid"),
                    "cloud_cover_high_pct": val("cloud_cover_high"),
                    "shortwave_rad_wm2": val("shortwave_radiation"),
                    "direct_rad_wm2": val("direct_radiation"),
                    "pressure_msl_hpa": val("pressure_msl"),
                    "soil_temp_0_7cm_c": val("soil_temperature_0cm"),
                })

            n_chunk = len(times)
            log.info(f"  {chunk_start.strftime('%Y-%m-%d')} → {chunk_end.strftime('%Y-%m-%d')}: {n_chunk} hours")
            chunk_start = chunk_end + timedelta(days=1)
            await asyncio.sleep(0.5)  # rate limit

    return records


async def backfill_atmosphere(db_path: str, start_date: str = "2023-01-01") -> int:
    """Pull and store atmospheric data from Open-Meteo."""
    records = await fetch_openmeteo_atmosphere(start_date)
    if not records:
        return 0

    conn = sqlite3.connect(db_path, timeout=10)
    ensure_enriched_tables(conn)

    inserted = 0
    for r in records:
        try:
            conn.execute(
                """INSERT OR IGNORE INTO enriched_atmosphere
                   (timestamp_utc, cape_jkg, cin_jkg, lifted_index,
                    cloud_cover_pct, cloud_cover_low_pct, cloud_cover_mid_pct,
                    cloud_cover_high_pct, shortwave_rad_wm2, direct_rad_wm2,
                    pressure_msl_hpa, soil_temp_0_7cm_c, source)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,'openmeteo_hist')""",
                (r["timestamp_utc"], r["cape_jkg"], r["cin_jkg"],
                 r["lifted_index"], r["cloud_cover_pct"],
                 r["cloud_cover_low_pct"], r["cloud_cover_mid_pct"],
                 r["cloud_cover_high_pct"], r["shortwave_rad_wm2"],
                 r["direct_rad_wm2"], r["pressure_msl_hpa"],
                 r["soil_temp_0_7cm_c"]),
            )
            inserted += 1
        except sqlite3.IntegrityError:
            pass

    conn.commit()
    conn.close()
    log.info(f"Atmosphere: {inserted}/{len(records)} records stored")
    return inserted


# ──────────────────────────────────────────────────────────────────────
# 3. NDBC Buoy SST
# ──────────────────────────────────────────────────────────────────────

async def fetch_ndbc_historical(station: str = "VAKF1", year: int = 2024) -> list[dict]:
    """Fetch NDBC historical standard meteorological data for a year.

    NDBC archives are gzipped text files per station per year.
    """
    url = f"{NDBC_HIST_URL}/{station.lower()}h{year}.txt.gz"
    log.info(f"  Fetching NDBC {station} year {year}...")

    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                if resp.status != 200:
                    log.debug(f"  NDBC {station} {year}: HTTP {resp.status}")
                    return []
                raw = await resp.read()
        except Exception as e:
            log.debug(f"  NDBC {station} {year}: {e}")
            return []

    import gzip
    try:
        text = gzip.decompress(raw).decode("utf-8", errors="replace")
    except Exception:
        text = raw.decode("utf-8", errors="replace")

    records = []
    lines = text.strip().split("\n")
    if len(lines) < 3:
        return []

    # Skip header lines (first 2 lines are headers)
    for line in lines[2:]:
        parts = line.split()
        if len(parts) < 14:
            continue
        try:
            yr, mo, dy, hr, mn = parts[0], parts[1], parts[2], parts[3], parts[4]
            ts = f"{yr}-{mo}-{dy} {hr}:{mn}"

            def safe(idx, missing="99.0"):
                v = parts[idx] if idx < len(parts) else missing
                return float(v) if v != "999" and v != "999.0" and v != "99.0" and v != "9999.0" and v != "MM" else None

            wdir = safe(5)
            wspd = safe(6)  # m/s
            atmp = safe(13)  # °C
            wtmp = safe(14)  # °C
            pres = safe(12)  # hPa

            # Convert
            wtmp_f = wtmp * 9 / 5 + 32 if wtmp is not None else None
            atmp_f = atmp * 9 / 5 + 32 if atmp is not None else None
            wspd_kt = wspd * 1.944 if wspd is not None else None  # m/s → kt

            if wtmp_f is not None or atmp_f is not None:
                records.append({
                    "timestamp_utc": ts,
                    "water_temp_f": wtmp_f,
                    "air_temp_f": atmp_f,
                    "wind_dir_deg": wdir,
                    "wind_speed_kt": wspd_kt,
                    "pressure_hpa": pres,
                })
        except (ValueError, IndexError):
            continue

    return records


async def backfill_sst(
    db_path: str,
    station: str = "VAKF1",
    start_year: int = 2023,
    end_year: int | None = None,
) -> int:
    """Pull NDBC buoy data for training."""
    if end_year is None:
        end_year = datetime.utcnow().year

    conn = sqlite3.connect(db_path, timeout=10)
    ensure_enriched_tables(conn)

    total = 0
    for year in range(start_year, end_year + 1):
        records = await fetch_ndbc_historical(station, year)
        if not records:
            continue

        inserted = 0
        for r in records:
            try:
                conn.execute(
                    """INSERT OR IGNORE INTO enriched_sst
                       (station, timestamp_utc, water_temp_f, air_temp_f,
                        wind_dir_deg, wind_speed_kt, pressure_hpa)
                       VALUES (?,?,?,?,?,?,?)""",
                    (station, r["timestamp_utc"], r["water_temp_f"],
                     r["air_temp_f"], r["wind_dir_deg"],
                     r["wind_speed_kt"], r["pressure_hpa"]),
                )
                inserted += 1
            except sqlite3.IntegrityError:
                pass

        conn.commit()
        log.info(f"  {station} {year}: {inserted}/{len(records)} records stored")
        total += inserted
        await asyncio.sleep(1.0)

    conn.close()
    log.info(f"SST total: {total} records")
    return total


# ──────────────────────────────────────────────────────────────────────
# Main orchestrator
# ──────────────────────────────────────────────────────────────────────

async def run_all_backfills(
    db_path: str,
    start_date: str = "2023-01-01",
    stations: list[str] | None = None,
):
    """Run all enriched backfills."""
    if stations is None:
        stations = SE_FLORIDA_STATIONS

    log.info("=" * 60)
    log.info("ENRICHED BACKFILL — pulling all available historical data")
    log.info("=" * 60)

    # 1. IEM ASOS (full obs)
    log.info("\n── Phase 1: IEM ASOS (10 stations, full fields, UTC) ──")
    asos_total = 0
    for station in stations:
        n = await backfill_asos_station(station, db_path, start_date)
        asos_total += n
        await asyncio.sleep(2.0)  # rate limit
    log.info(f"ASOS total: {asos_total} records")

    # 2. Open-Meteo atmosphere (CAPE, CIN, cloud, radiation)
    log.info("\n── Phase 2: Open-Meteo Atmospheric (CAPE/CIN/cloud/rad) ──")
    atmos_total = await backfill_atmosphere(db_path, start_date)

    # 3. NDBC buoy SST
    log.info("\n── Phase 3: NDBC Buoy SST (VAKF1 Virginia Key) ──")
    sst_total = await backfill_sst(db_path, "VAKF1", int(start_date[:4]))

    log.info("\n" + "=" * 60)
    log.info(f"ENRICHED BACKFILL COMPLETE")
    log.info(f"  ASOS:       {asos_total:>8,} records ({len(stations)} stations)")
    log.info(f"  Atmosphere: {atmos_total:>8,} records (CAPE/CIN/cloud)")
    log.info(f"  SST:        {sst_total:>8,} records (VAKF1)")
    log.info("=" * 60)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-5s %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Enriched historical data backfill")
    parser.add_argument("--db", default="miami_collector.db")
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--stations", nargs="+", default=None)
    parser.add_argument("--only", choices=["asos", "atmos", "sst"], default=None,
                        help="Run only one phase")
    args = parser.parse_args()

    if args.only == "asos":
        asyncio.run(_run_asos_only(args.db, args.start, args.stations))
    elif args.only == "atmos":
        asyncio.run(backfill_atmosphere(args.db, args.start))
    elif args.only == "sst":
        asyncio.run(backfill_sst(args.db, "VAKF1", int(args.start[:4])))
    else:
        asyncio.run(run_all_backfills(args.db, args.start, args.stations))


async def _run_asos_only(db_path, start_date, stations):
    stations = stations or SE_FLORIDA_STATIONS
    for s in stations:
        await backfill_asos_station(s, db_path, start_date)
        await asyncio.sleep(2.0)


if __name__ == "__main__":
    main()
