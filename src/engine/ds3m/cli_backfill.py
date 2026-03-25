"""Backfill CLI-equivalent daily high/low from IEM for training.

IEM provides hourly ASOS observations from which we compute daily max/min.
These serve as settlement proxies for training DS3M when we only have
11 actual Kalshi CLI settlements.

Data source: Iowa Environmental Mesonet ASOS
URL: https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py

IMPORTANT: IEM uses 3-letter ICAO identifiers (MIA not KMIA) for ASOS stations.
We map back to 4-letter ICAO for consistency with our DB.

This gives us ~1100 daily highs/lows per station (3 years) → ~11K
settlement-equivalent training targets across all 10 SE Florida stations.
"""

from __future__ import annotations

import argparse
import csv
import io
import logging
import sqlite3
from collections import defaultdict
from datetime import datetime, timedelta

import aiohttp
import asyncio

log = logging.getLogger(__name__)

IEM_ASOS_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"

# Map 4-letter ICAO → 3-letter IEM station ID
# 35 Florida ASOS stations within 350km of KMIA
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

ALL_STATIONS = list(STATION_MAP.keys())
SE_FLORIDA_STATIONS = ALL_STATIONS  # backward compat


async def fetch_hourly_temps(
    iem_id: str,
    start_date: str = "2023-01-01",
    end_date: str | None = None,
) -> list[tuple[str, float]]:
    """Fetch hourly temperature observations from IEM ASOS.

    Returns list of (timestamp, temp_f) tuples.
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    sd = datetime.strptime(start_date, "%Y-%m-%d")
    ed = datetime.strptime(end_date, "%Y-%m-%d")

    params = {
        "station": iem_id,
        "data": "tmpf",
        "year1": sd.year, "month1": sd.month, "day1": sd.day,
        "year2": ed.year, "month2": ed.month, "day2": ed.day,
        "tz": "America/New_York",  # CLI uses EST/EDT
        "format": "onlycomma",
        "latlon": "no",
        "elev": "no",
        "missing": "empty",
        "trace": "empty",
        "direct": "no",
        "report_type": "3",  # hourly ASOS
    }

    async with aiohttp.ClientSession() as session:
        async with session.get(IEM_ASOS_URL, params=params,
                               timeout=aiohttp.ClientTimeout(total=120)) as resp:
            if resp.status != 200:
                log.error(f"IEM fetch failed for {iem_id}: HTTP {resp.status}")
                return []
            text = await resp.text()

    records = []
    reader = csv.DictReader(io.StringIO(text))
    for row in reader:
        try:
            ts = row.get("valid", "")
            tmpf = row.get("tmpf", "")
            if not ts or not tmpf:
                continue
            records.append((ts.strip(), float(tmpf)))
        except (ValueError, KeyError):
            continue

    return records


def compute_daily_extremes(
    obs: list[tuple[str, float]],
    utc: bool = False,
) -> list[dict]:
    """Compute daily max/min from hourly observations.

    When utc=True, groups by NWS CLI climate day:
      - Climate day = 05:00 UTC to 04:59 UTC (midnight-to-midnight EST)
      - An observation at 03:00 UTC belongs to the PREVIOUS calendar date's climate day
    When utc=False (legacy), groups by local date string.
    """
    daily: dict[str, list[float]] = defaultdict(list)

    for ts, temp in obs:
        if utc:
            # Parse UTC timestamp → assign to CLI climate day
            # CLI day boundary is 05:00 UTC
            # Obs before 05:00 UTC belongs to previous day's climate day
            try:
                dt = datetime.strptime(ts[:16], "%Y-%m-%d %H:%M")
            except ValueError:
                continue

            if dt.hour < 5:
                # Before 05Z → belongs to previous day's climate day
                climate_date = (dt - timedelta(days=1)).strftime("%Y-%m-%d")
            else:
                climate_date = dt.strftime("%Y-%m-%d")
            daily[climate_date].append(temp)
        else:
            # Legacy: group by date in timestamp string
            date = ts[:10]
            daily[date].append(temp)

    results = []
    for date in sorted(daily.keys()):
        temps = daily[date]
        if len(temps) < 12:  # need at least half a day of obs
            continue
        results.append({
            "date": date,
            "max_tmpf": max(temps),
            "min_tmpf": min(temps),
            "n_obs": len(temps),
        })

    return results


def ensure_cli_table(conn: sqlite3.Connection):
    """Create the CLI backfill table if it doesn't exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cli_daily_backfill (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            station TEXT NOT NULL,
            date TEXT NOT NULL,
            max_tmpf REAL,
            min_tmpf REAL,
            n_obs INTEGER,
            source TEXT DEFAULT 'iem_hourly',
            created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
            UNIQUE(station, date)
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_cli_backfill_station
        ON cli_daily_backfill(station, date)
    """)
    conn.commit()


async def backfill_station(
    icao: str,
    db_path: str,
    start_date: str = "2023-01-01",
) -> int:
    """Fetch and store daily climate data for one station."""
    iem_id = STATION_MAP.get(icao, icao.lstrip("K"))
    log.info(f"Fetching hourly ASOS for {icao} (IEM: {iem_id}) from {start_date}...")

    obs = await fetch_hourly_temps(iem_id, start_date)
    if not obs:
        log.warning(f"No data returned for {icao}")
        return 0

    log.info(f"  {icao}: {len(obs)} hourly obs received")
    daily = compute_daily_extremes(obs)
    log.info(f"  {icao}: {len(daily)} daily summaries computed")

    conn = sqlite3.connect(db_path, timeout=10)
    ensure_cli_table(conn)

    inserted = 0
    for d in daily:
        try:
            conn.execute(
                """INSERT OR IGNORE INTO cli_daily_backfill
                   (station, date, max_tmpf, min_tmpf, n_obs)
                   VALUES (?, ?, ?, ?, ?)""",
                (icao, d["date"], d["max_tmpf"], d["min_tmpf"], d["n_obs"]),
            )
            inserted += 1
        except sqlite3.IntegrityError:
            pass

    conn.commit()
    conn.close()
    log.info(f"  {icao}: {inserted} records stored")
    return inserted


async def backfill_all(
    db_path: str,
    stations: list[str] | None = None,
    start_date: str = "2023-01-01",
) -> int:
    """Backfill daily climate data for all SE Florida stations."""
    if stations is None:
        stations = SE_FLORIDA_STATIONS

    total = 0
    for station in stations:
        n = await backfill_station(station, db_path, start_date)
        total += n
        # Rate limit: be nice to IEM
        await asyncio.sleep(2.0)

    log.info(f"CLI backfill complete: {total} total records across {len(stations)} stations")
    return total


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)-20s %(levelname)-5s %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Backfill IEM daily climate data")
    parser.add_argument("--db", default="miami_collector.db")
    parser.add_argument("--start", default="2023-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--stations", nargs="+", default=None)
    args = parser.parse_args()

    asyncio.run(backfill_all(args.db, args.stations, args.start))


if __name__ == "__main__":
    main()
