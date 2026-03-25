"""Backfill historical 5-min ASOS data from IEM for DS3M training.

Fetches data for KMIA and SE Florida cluster stations. Stores in the
observations and nearby_observations tables (same schema as live data).
Does NOT classify regimes — that requires atmospheric context not in IEM.

Usage:
    python -m engine.ds3m.backfill --db miami_collector.db --years 3
"""

from __future__ import annotations

import calendar
import csv
import io
import logging
import math
import sqlite3
import statistics
import urllib.request
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

log = logging.getLogger(__name__)

# IEM ASOS download URL (extended field set vs sigma_climatology.py)
_IEM_URL = (
    "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?"
    "station={station}&data=tmpf&data=dwpf&data=drct&data=sknt&data=gust_sknt"
    "&data=alti&data=skyc1&data=skyc2&data=skyc3&data=p01i"
    "&year1={y1}&month1={m1}&day1=1"
    "&year2={y2}&month2={m2}&day2={last_day}"
    "&tz=America%2FNew_York&format=comma&latlon=no&elev=no"
    "&missing=M&trace=T&direct=no&report_type=3"
)

# Station mapping: IEM accepts 4-letter ICAO codes in the URL but returns
# 3-letter codes in the CSV output. Use ICAO for the URL.
_IEM_STATIONS = {
    "KMIA": "KMIA",  # Primary trading station (IEM returns "MIA")
    "KFLL": "KFLL",
    "KOPF": "KOPF",
    "KTMB": "KTMB",
    "KHWO": "KHWO",
    "KHST": "KHST",
    "KFXE": "KFXE",
    "KPMP": "KPMP",
    "KBCT": "KBCT",
    "KPBI": "KPBI",
}

# Station metadata for nearby_observations table
_STATION_META: dict[str, dict[str, Any]] = {
    "KFLL": {"name": "Fort Lauderdale", "network": "FL_ASOS", "lat": 26.0726, "lon": -80.1527, "dist_mi": 20.5, "elev_m": 3.0},
    "KOPF": {"name": "Miami/Opa Locka", "network": "FL_ASOS", "lat": 25.907, "lon": -80.278, "dist_mi": 8.5, "elev_m": 2.4},
    "KTMB": {"name": "Kendall-Tamiami", "network": "FL_ASOS", "lat": 25.648, "lon": -80.433, "dist_mi": 14.0, "elev_m": 2.4},
    "KHWO": {"name": "Hollywood/N. Perry", "network": "FL_ASOS", "lat": 26.001, "lon": -80.240, "dist_mi": 15.0, "elev_m": 2.1},
    "KHST": {"name": "Homestead AFB", "network": "FL_ASOS", "lat": 25.489, "lon": -80.384, "dist_mi": 24.0, "elev_m": 1.5},
    "KFXE": {"name": "Fort Lauderdale Exec", "network": "FL_ASOS", "lat": 26.197, "lon": -80.171, "dist_mi": 28.0, "elev_m": 4.0},
    "KPMP": {"name": "Pompano Beach", "network": "FL_ASOS", "lat": 26.247, "lon": -80.111, "dist_mi": 33.0, "elev_m": 5.0},
    "KBCT": {"name": "Boca Raton", "network": "FL_ASOS", "lat": 26.379, "lon": -80.108, "dist_mi": 42.0, "elev_m": 4.0},
    "KPBI": {"name": "West Palm Beach", "network": "FL_ASOS", "lat": 26.683, "lon": -80.096, "dist_mi": 62.0, "elev_m": 6.0},
}

# Sky cover code to percentage mapping
_SKY_PCT = {"CLR": 0.0, "FEW": 25.0, "SCT": 50.0, "BKN": 75.0, "OVC": 100.0}

# EST offset from UTC (IEM returns America/New_York timestamps)
_EST_OFFSET = timedelta(hours=-5)
_EDT_OFFSET = timedelta(hours=-4)

# Column indices are determined dynamically from the header row.
# IEM sometimes omits requested fields (e.g., gust_sknt may be missing).


def _is_dst(dt_local: datetime) -> bool:
    """Rough check: EDT is second Sunday of March to first Sunday of November."""
    year = dt_local.year
    # Second Sunday of March
    mar1 = datetime(year, 3, 1)
    dst_start = mar1 + timedelta(days=(6 - mar1.weekday()) % 7 + 7)
    # First Sunday of November
    nov1 = datetime(year, 11, 1)
    dst_end = nov1 + timedelta(days=(6 - nov1.weekday()) % 7)
    return dst_start <= dt_local.replace(tzinfo=None) < dst_end


def _parse_float(val: str) -> float | None:
    """Parse IEM numeric field. 'M' = missing, 'T' = trace (0.001)."""
    val = val.strip()
    if val in ("M", "", "null"):
        return None
    if val == "T":
        return 0.001
    try:
        return float(val)
    except ValueError:
        return None


def _local_to_utc(dt_local: datetime) -> datetime:
    """Convert IEM local-time timestamp (America/New_York) to UTC."""
    offset = _EDT_OFFSET if _is_dst(dt_local) else _EST_OFFSET
    return dt_local - offset


def _sky_code_to_pct(code: str | None) -> float | None:
    """Convert sky cover code to percentage."""
    if code is None:
        return None
    return _SKY_PCT.get(code.strip().upper())


def _highest_sky_cover(sky1: str | None, sky2: str | None, sky3: str | None) -> tuple[str | None, float | None]:
    """Return the highest sky cover code and its percentage."""
    best_code = None
    best_pct: float | None = None
    for code in (sky1, sky2, sky3):
        pct = _sky_code_to_pct(code)
        if pct is not None:
            if best_pct is None or pct > best_pct:
                best_pct = pct
                best_code = code.strip().upper() if code else None
    return best_code, best_pct


def fetch_station_month(iem_station: str, year: int, month: int) -> list[dict]:
    """Fetch one month of 5-min ASOS data from IEM.

    Returns list of dicts with parsed fields. Handles 'M' as None
    and 'T' (trace precip) as 0.001.
    """
    last_day = calendar.monthrange(year, month)[1]
    url = _IEM_URL.format(
        station=iem_station,
        y1=year, m1=month,
        y2=year, m2=month,
        last_day=last_day,
    )

    log.debug("Fetching IEM: %s %d-%02d", iem_station, year, month)
    resp = urllib.request.urlopen(url, timeout=60)
    text = resp.read().decode()

    rows: list[dict] = []
    reader = csv.reader(io.StringIO(text))
    col_map: dict[str, int] = {}

    for row in reader:
        if not row or row[0].startswith("#"):
            continue
        # Parse header row to build column index map
        if row[0] == "station":
            col_map = {col.strip(): i for i, col in enumerate(row)}
            continue
        if not col_map or len(row) < 3:
            continue

        def _get(name: str) -> str | None:
            idx = col_map.get(name)
            if idx is None or idx >= len(row):
                return None
            v = row[idx].strip()
            return v if v not in ("M", "") else None

        try:
            valid_str = row[col_map.get("valid", 1)].strip()
            dt_local = datetime.strptime(valid_str, "%Y-%m-%d %H:%M")
        except (ValueError, IndexError):
            continue

        dt_utc = _local_to_utc(dt_local)

        tmpf = _parse_float(_get("tmpf") or "M")
        dwpf = _parse_float(_get("dwpf") or "M")
        drct = _parse_float(_get("drct") or "M")
        sknt = _parse_float(_get("sknt") or "M")
        gust = _parse_float(_get("gust_sknt") or "M")
        alti = _parse_float(_get("alti") or "M")
        p01i = _parse_float(_get("p01i") or "M")

        sky1 = _get("skyc1")
        sky2 = _get("skyc2")
        sky3 = _get("skyc3")

        # Unit conversions
        wind_speed_mph = round(sknt * 1.15078, 2) if sknt is not None else None
        wind_gust_mph = round(gust * 1.15078, 2) if gust is not None else None
        pressure_hpa = round(alti * 33.8639, 2) if alti is not None else None
        precip_mm = round(p01i * 25.4, 3) if p01i is not None else None

        sky_code, sky_pct = _highest_sky_cover(sky1, sky2, sky3)

        # LST date: use the local date for climate-day grouping
        lst_date = dt_local.strftime("%Y-%m-%d")

        # IEM returns 3-letter code (MIA) but we want 4-letter ICAO (KMIA)
        raw_station = _get("station") or row[0].strip()
        icao = f"K{raw_station}" if len(raw_station) == 3 and not raw_station.startswith("K") else raw_station

        rows.append({
            "station": icao,
            "timestamp_utc": dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "lst_date": lst_date,
            "temperature_f": tmpf,
            "dew_point_f": dwpf,
            "wind_dir_deg": drct,
            "wind_speed_mph": wind_speed_mph,
            "wind_gust_mph": wind_gust_mph,
            "pressure_hpa": pressure_hpa,
            "sky_cover_code": sky_code,
            "sky_cover_pct": sky_pct,
            "precip_1hr_in": p01i,
            "precip_mm": precip_mm,
        })

    log.info("Fetched %d rows for %s %d-%02d", len(rows), iem_station, year, month)
    return rows


def backfill_station(db_path: str, icao_code: str, years_back: int = 3) -> int:
    """Fetch all months for a station and insert into the DB.

    KMIA data goes into the observations table with source='iem_backfill'.
    All other stations go into nearby_observations.

    Returns row count inserted.
    """
    iem_station = _IEM_STATIONS.get(icao_code)
    if iem_station is None:
        log.error("Unknown station %s — not in _IEM_STATIONS", icao_code)
        return 0

    now = datetime.now()
    start_year = now.year - years_back
    end_year = now.year
    end_month = now.month

    is_kmia = icao_code == "KMIA"
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    total = 0

    for year in range(start_year, end_year + 1):
        m_start = 1
        m_end = 12
        if year == end_year:
            m_end = end_month

        for month in range(m_start, m_end + 1):
            try:
                rows = fetch_station_month(iem_station, year, month)
            except Exception:
                log.warning("Failed to fetch %s %d-%02d", iem_station, year, month)
                continue

            if not rows:
                continue

            inserted = 0
            if is_kmia:
                inserted = _insert_kmia_obs(conn, icao_code, rows)
            else:
                inserted = _insert_nearby_obs(conn, icao_code, rows)

            total += inserted
            log.info(
                "Inserted %d/%d rows for %s %d-%02d",
                inserted, len(rows), icao_code, year, month,
            )

    conn.close()
    log.info("Backfill complete for %s: %d total rows", icao_code, total)
    return total


def _insert_kmia_obs(conn: sqlite3.Connection, icao: str, rows: list[dict]) -> int:
    """Insert KMIA data into the observations table."""
    sql = """
        INSERT OR IGNORE INTO observations (
            station, timestamp_utc, lst_date,
            temperature_f, dew_point_f,
            wind_speed_mph, wind_heading_deg, wind_gust_mph,
            pressure_hpa, sky_cover_code, sky_cover_pct,
            precipitation_last_hour_mm,
            source
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    inserted = 0
    batch: list[tuple] = []
    for r in rows:
        if r["temperature_f"] is None:
            continue
        batch.append((
            icao,
            r["timestamp_utc"],
            r["lst_date"],
            r["temperature_f"],
            r["dew_point_f"],
            r["wind_speed_mph"],
            r["wind_dir_deg"],
            r["wind_gust_mph"],
            r["pressure_hpa"],
            r["sky_cover_code"],
            r["sky_cover_pct"],
            r["precip_mm"],
            "iem_backfill",
        ))

    if batch:
        cur = conn.executemany(sql, batch)
        conn.commit()
        inserted = cur.rowcount if cur.rowcount > 0 else len(batch)

    return inserted


def _insert_nearby_obs(conn: sqlite3.Connection, icao: str, rows: list[dict]) -> int:
    """Insert non-KMIA station data into nearby_observations."""
    meta = _STATION_META.get(icao, {})
    sql = """
        INSERT OR IGNORE INTO nearby_observations (
            stid, name, network, latitude, longitude,
            distance_mi, elevation_m,
            timestamp_utc, lst_date,
            air_temp_f, dew_point_f,
            wind_speed_mph, wind_direction_deg, wind_gust_mph,
            pressure_slp_hpa, sky_cover_code
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    inserted = 0
    batch: list[tuple] = []
    for r in rows:
        if r["temperature_f"] is None:
            continue
        batch.append((
            icao,
            meta.get("name", icao),
            meta.get("network", "FL_ASOS"),
            meta.get("lat"),
            meta.get("lon"),
            meta.get("dist_mi"),
            meta.get("elev_m"),
            r["timestamp_utc"],
            r["lst_date"],
            r["temperature_f"],
            r["dew_point_f"],
            r["wind_speed_mph"],
            r["wind_dir_deg"],
            r["wind_gust_mph"],
            r["pressure_hpa"],
            r["sky_cover_code"],
        ))

    if batch:
        cur = conn.executemany(sql, batch)
        conn.commit()
        inserted = cur.rowcount if cur.rowcount > 0 else len(batch)

    return inserted


def backfill_all(db_path: str, years_back: int = 3) -> dict[str, int]:
    """Backfill all stations in _IEM_STATIONS.

    Returns {station: rows_inserted}.
    """
    counts: dict[str, int] = {}
    for icao in _IEM_STATIONS:
        count = backfill_station(db_path, icao, years_back)
        counts[icao] = count
    return counts


def compute_remaining_move_stats(
    db_path: str,
    station: str = "KMIA",
    years_back: int = 3,
) -> dict[tuple[int, int], dict[str, float]]:
    """Compute per-(month, hour) remaining-move statistics from backfilled data.

    Groups by (month, hour_utc) for seasonal awareness.
    Computes separate stats for HIGH (remaining upside) and LOW (remaining
    downside) — model scoring must be separate for highs vs lows.

    Returns:
        {(month, hour): {
            high_mean, high_std, high_q50, high_q75, high_q90,
            low_mean, low_std, low_q50, low_q75, low_q90,
            sample_count,
        }}
    """
    conn = sqlite3.connect(db_path)
    cutoff = datetime.now() - timedelta(days=years_back * 365)
    cutoff_str = cutoff.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Fetch all observations for this station
    sql = """
        SELECT lst_date, timestamp_utc, temperature_f
        FROM observations
        WHERE station = ? AND timestamp_utc >= ? AND temperature_f IS NOT NULL
        ORDER BY timestamp_utc
    """
    cur = conn.execute(sql, (station, cutoff_str))
    rows_raw = cur.fetchall()
    conn.close()

    if not rows_raw:
        log.warning("No observations found for %s; cannot compute remaining-move stats", station)
        return {}

    # Group by climate day
    day_obs: dict[str, list[tuple[int, int, float]]] = defaultdict(list)
    for lst_date, ts_utc, temp_f in rows_raw:
        try:
            dt = datetime.strptime(ts_utc, "%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            continue
        month = dt.month
        hour = dt.hour
        day_obs[lst_date].append((month, hour, temp_f))

    # Compute remaining move per (month, hour)
    high_remaining: dict[tuple[int, int], list[float]] = defaultdict(list)
    low_remaining: dict[tuple[int, int], list[float]] = defaultdict(list)

    n_days = 0
    for day_key, obs_list in day_obs.items():
        if len(obs_list) < 18:  # skip incomplete days
            continue
        n_days += 1

        final_high = max(t for _, _, t in obs_list)
        final_low = min(t for _, _, t in obs_list)

        # Running max/min keyed by hour
        running_max = float("-inf")
        running_min = float("inf")

        for month, hour, temp in sorted(obs_list, key=lambda x: x[1]):
            running_max = max(running_max, temp)
            running_min = min(running_min, temp)
            high_remaining[(month, hour)].append(final_high - running_max)
            low_remaining[(month, hour)].append(running_min - final_low)

    # Aggregate stats
    result: dict[tuple[int, int], dict[str, float]] = {}
    for key in sorted(set(high_remaining.keys()) | set(low_remaining.keys())):
        h_vals = high_remaining.get(key, [])
        l_vals = low_remaining.get(key, [])

        entry: dict[str, float] = {"sample_count": float(max(len(h_vals), len(l_vals)))}

        if len(h_vals) > 2:
            h_sorted = sorted(h_vals)
            entry["high_mean"] = round(statistics.mean(h_vals), 4)
            entry["high_std"] = round(statistics.stdev(h_vals), 4)
            entry["high_q50"] = round(_quantile(h_sorted, 0.50), 4)
            entry["high_q75"] = round(_quantile(h_sorted, 0.75), 4)
            entry["high_q90"] = round(_quantile(h_sorted, 0.90), 4)
        else:
            entry.update({"high_mean": 3.0, "high_std": 3.0, "high_q50": 2.0, "high_q75": 4.0, "high_q90": 6.0})

        if len(l_vals) > 2:
            l_sorted = sorted(l_vals)
            entry["low_mean"] = round(statistics.mean(l_vals), 4)
            entry["low_std"] = round(statistics.stdev(l_vals), 4)
            entry["low_q50"] = round(_quantile(l_sorted, 0.50), 4)
            entry["low_q75"] = round(_quantile(l_sorted, 0.75), 4)
            entry["low_q90"] = round(_quantile(l_sorted, 0.90), 4)
        else:
            entry.update({"low_mean": 3.0, "low_std": 3.0, "low_q50": 2.0, "low_q75": 4.0, "low_q90": 6.0})

        result[key] = entry

    log.info(
        "Computed remaining-move stats from %d days, %d (month,hour) cells",
        n_days, len(result),
    )
    return result


def _quantile(sorted_vals: list[float], q: float) -> float:
    """Simple quantile on a pre-sorted list."""
    if not sorted_vals:
        return 0.0
    idx = q * (len(sorted_vals) - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return sorted_vals[lo]
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Backfill IEM ASOS data for DS3M training",
    )
    parser.add_argument("--db", required=True, help="Path to SQLite database")
    parser.add_argument("--years", type=int, default=3, help="Years of history to fetch")
    parser.add_argument("--station", default=None, help="Single station ICAO (e.g. KMIA) or all")
    parser.add_argument(
        "--compute-stats", action="store_true",
        help="After backfill, compute remaining-move stats and print summary",
    )
    args = parser.parse_args()

    if args.station:
        count = backfill_station(args.db, args.station, args.years)
        print(f"Backfilled {count} rows for {args.station}")
    else:
        counts = backfill_all(args.db, args.years)
        for stn, n in counts.items():
            print(f"  {stn}: {n} rows")
        print(f"Total: {sum(counts.values())} rows")

    if args.compute_stats:
        stats = compute_remaining_move_stats(args.db, "KMIA", args.years)
        print(f"\nRemaining-move stats: {len(stats)} (month,hour) cells")
        for (m, h), s in sorted(stats.items())[:5]:
            print(
                f"  ({m:02d}, {h:02d}): high_mean={s['high_mean']:.1f} "
                f"high_std={s['high_std']:.1f}  low_mean={s['low_mean']:.1f} "
                f"low_std={s['low_std']:.1f}  n={int(s['sample_count'])}"
            )
        if len(stats) > 5:
            print(f"  ... ({len(stats) - 5} more cells)")
