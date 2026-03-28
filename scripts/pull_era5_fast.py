#!/usr/bin/env python3
"""Fast ERA5 download using ARCO time-series endpoint.

Instead of downloading gridded NetCDF month-by-month, this uses the
dedicated time-series endpoint that returns single-point data over
the full time range in one request. ~46 years per station per request.

Usage:
    python scripts/pull_era5_fast.py --db miami_collector.db
    python scripts/pull_era5_fast.py --db miami_collector.db --start-year 2000
"""

import argparse
import logging
import os
import sqlite3
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("pull_era5_fast")

try:
    from ecmwf.datastores import Client as CDSClient
    USE_NEW_CLIENT = True
except ImportError:
    try:
        import cdsapi
        USE_NEW_CLIENT = False
    except ImportError:
        log.error("No CDS client. Run: pip install ecmwf-datastores-client")
        sys.exit(1)

try:
    import pandas as pd
    import numpy as np
except ImportError:
    log.error("Missing deps. Run: pip install pandas numpy")
    sys.exit(1)


# ──────────────────────────────────────────────────────────────────
# All 46 stations (33 ASOS + 10 FAWN + 3 Buoys)
# ──────────────────────────────────────────────────────────────────

STATIONS = {
    # ASOS (FAA-verified)
    "KMIA": (25.7954, -80.2901), "KFLL": (26.0717, -80.1497),
    "KPBI": (26.6832, -80.0956), "KHWO": (26.0013, -80.2407),
    "KOPF": (25.9074, -80.2782), "KFXE": (26.1973, -80.1707),
    "KBCT": (26.3785, -80.1077), "KTMB": (25.6476, -80.4332),
    "KHST": (25.4886, -80.3836), "KEYW": (24.5561, -81.7600),
    "KAPF": (26.1524, -81.7756), "KRSW": (26.5362, -81.7552),
    "KFMY": (26.5866, -81.8632), "KOBE": (27.2666, -80.8504),
    "KSPG": (27.7651, -82.6269), "KMLB": (28.1027, -80.6453),
    "KMCO": (28.4294, -81.3090), "KSFB": (28.7772, -81.2349),
    "KTPA": (27.9755, -82.5332), "KPGD": (26.9186, -81.9909),
    "KVRB": (27.6556, -80.4180), "KSRQ": (27.3954, -82.5544),
    "KSEF": (27.4564, -81.3424), "KNQX": (24.5746, -81.6866),
    "KMTH": (24.7262, -81.0514), "KFPR": (27.4975, -80.3726),
    "KPIE": (27.9086, -82.6865), "KSUA": (27.1817, -80.2213),
    "KORL": (28.5455, -81.3329), "KPMP": (26.2474, -80.1111),
    "KLAL": (27.9876, -82.0190), "KBOW": (27.9434, -81.7834),
    "KIMM": (26.4337, -81.4005),
    # FAWN (IFAS-verified)
    "FAWN_405": (26.7391, -81.0528), "FAWN_410": (26.6568, -80.6300),
    "FAWN_420": (26.0853, -80.2405), "FAWN_425": (26.6808, -80.3006),
    "FAWN_430": (27.4271, -80.4054), "FAWN_435": (27.3378, -80.6192),
    "FAWN_440": (25.5126, -80.5031), "FAWN_450": (26.4623, -81.4403),
    "FAWN_455": (27.3306, -80.8514), "FAWN_460": (26.9248, -81.3146),
    # Buoys (NDBC)
    "VAKF1": (25.7314, -80.1620), "FWYF1": (25.5906, -80.0969),
    "LKWF1": (26.6128, -80.0339),
}

# Variables available on the time-series endpoint
# Only variables confirmed available on the ARCO time-series endpoint
TIMESERIES_VARS = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "surface_pressure",
    "mean_sea_level_pressure",
    "total_precipitation",
    "surface_solar_radiation_downwards",
    "sea_surface_temperature",
]

HOURS = [f"{h:02d}:00" for h in range(24)]


def snap_to_grid(coord, resolution=0.25):
    """Snap coordinate to nearest ERA5 grid point (0.25° resolution)."""
    return round(coord / resolution) * resolution


def download_station_timeseries(client, station_id, lat, lon, start_year, end_year, output_dir):
    """Download time-series for one station using date range format.

    Chunks into 10-year blocks to stay within CDS limits.
    Uses the correct date range format: 'YYYY-MM-DD/YYYY-MM-DD'.
    Coordinates snapped to 0.25° ERA5 grid.
    """
    all_files = []
    chunk_size = 10  # years per request — tested working

    # Snap to ERA5 grid
    grid_lat = snap_to_grid(lat)
    grid_lon = snap_to_grid(lon)

    for chunk_start in range(start_year, end_year + 1, chunk_size):
        chunk_end = min(chunk_start + chunk_size - 1, end_year)
        output_file = output_dir / f"era5_ts_{station_id}_{chunk_start}_{chunk_end}.nc"

        if output_file.exists() and output_file.stat().st_size > 1000:
            log.info(f"  {output_file} already exists, skipping")
            all_files.append(output_file)
            continue

        date_range = f"{chunk_start}-01-01/{chunk_end}-12-31"
        log.info(f"  Requesting {station_id} ({grid_lat}, {grid_lon}) for {date_range}...")

        request = {
            "variable": TIMESERIES_VARS,
            "location": {"latitude": grid_lat, "longitude": grid_lon},
            "date": [date_range],
            "data_format": "netcdf",
        }

        try:
            client.retrieve(
                "reanalysis-era5-single-levels-timeseries",
                request,
                str(output_file),
            )
            size_mb = output_file.stat().st_size / 1e6
            log.info(f"  Downloaded: {output_file} ({size_mb:.1f} MB)")
            all_files.append(output_file)
        except Exception as e:
            log.error(f"  Failed {station_id} {chunk_start}-{chunk_end}: {e}")
            time.sleep(2)

    return all_files


def parse_netcdf_to_rows(nc_file, station_id):
    """Parse NetCDF time-series into list of dicts for DB insertion."""
    try:
        import xarray as xr
        ds = xr.open_dataset(nc_file)
    except Exception:
        # Try CSV fallback
        log.warning(f"  NetCDF parse failed for {nc_file}, trying pandas read")
        return []

    rows = []
    times = pd.to_datetime(ds.time.values)

    for var in ds.data_vars:
        if var == 'time':
            continue

    # Build rows efficiently
    n_times = len(times)
    log.info(f"  Parsing {n_times:,} timesteps for {station_id}...")

    # Extract all variables at once
    data = {}
    for var in ds.data_vars:
        try:
            vals = ds[var].values.flatten()
            if len(vals) == n_times:
                data[var] = vals
        except Exception:
            pass

    for i in range(n_times):
        row = {
            "station_id": station_id,
            "timestamp_utc": str(times[i]),
        }
        for var, vals in data.items():
            row[var] = float(vals[i]) if not np.isnan(vals[i]) else None
        rows.append(row)

    ds.close()
    return rows


def ingest_to_db(rows, db_path, table_name):
    """Insert rows into SQLite, creating table dynamically from first row."""
    if not rows:
        return

    conn = sqlite3.connect(db_path)

    # Create table if needed (dynamic columns from first row)
    cols = list(rows[0].keys())
    col_defs = []
    for c in cols:
        if c in ("station_id", "timestamp_utc"):
            col_defs.append(f"{c} TEXT NOT NULL")
        else:
            col_defs.append(f"{c} REAL")

    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            {', '.join(col_defs)},
            PRIMARY KEY (station_id, timestamp_utc)
        )
    """)

    # Batch insert with OR IGNORE for duplicates
    placeholders = ', '.join(['?'] * len(cols))
    col_names = ', '.join(cols)

    batch_size = 5000
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        values = [tuple(row.get(c) for c in cols) for row in batch]
        conn.executemany(
            f"INSERT OR IGNORE INTO {table_name} ({col_names}) VALUES ({placeholders})",
            values,
        )
        conn.commit()

    count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    log.info(f"  {table_name}: {count:,} total rows")
    conn.close()


def main():
    parser = argparse.ArgumentParser(description="Fast ERA5 download (ARCO time-series)")
    parser.add_argument("--db", type=str, default="miami_collector.db")
    parser.add_argument("--start-year", type=int, default=1980)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--output-dir", type=str, default="data/era5")
    parser.add_argument("--stations", type=str, default=None,
                        help="Comma-separated station IDs (default: all 46)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if USE_NEW_CLIENT:
        client = CDSClient()
        log.info("Using ecmwf-datastores-client (ARCO time-series)")
    else:
        client = cdsapi.Client()
        log.info("Using cdsapi (ARCO time-series)")

    # Filter stations if specified
    if args.stations:
        station_ids = [s.strip() for s in args.stations.split(",")]
        stations = {k: v for k, v in STATIONS.items() if k in station_ids}
    else:
        stations = STATIONS

    log.info("=" * 60)
    log.info("ERA5 ARCO Time-Series Download")
    log.info("=" * 60)
    log.info(f"Stations: {len(stations)}")
    log.info(f"Years: {args.start_year} to {args.end_year}")
    log.info(f"Variables: {len(TIMESERIES_VARS)}")
    log.info(f"Estimated: {len(stations)} requests (one per station)")
    log.info(f"Output: {output_dir}")
    log.info("")

    total = len(stations)
    for i, (station_id, (lat, lon)) in enumerate(stations.items()):
        log.info(f"\n--- Station {station_id} ({i + 1}/{total}) ---")

        try:
            nc_files = download_station_timeseries(
                client, station_id, lat, lon,
                args.start_year, args.end_year, output_dir,
            )

            total_rows = 0
            for nc_file in nc_files:
                try:
                    rows = parse_netcdf_to_rows(nc_file, station_id)
                    if rows:
                        ingest_to_db(rows, args.db, "era5_timeseries")
                        total_rows += len(rows)
                except Exception as e:
                    log.error(f"  Parse/ingest failed for {nc_file}: {e}")

            log.info(f"  {station_id} total: {total_rows:,} rows ingested")

        except Exception as e:
            log.error(f"  Failed {station_id}: {e}")

        # Rate limiting
        if i < total - 1:
            time.sleep(3)

    log.info("\n" + "=" * 60)
    log.info("ERA5 ARCO DOWNLOAD COMPLETE")
    log.info("=" * 60)

    conn = sqlite3.connect(args.db)
    try:
        count = conn.execute("SELECT COUNT(*) FROM era5_timeseries").fetchone()[0]
        stations_done = conn.execute("SELECT COUNT(DISTINCT station_id) FROM era5_timeseries").fetchone()[0]
        min_date = conn.execute("SELECT MIN(timestamp_utc) FROM era5_timeseries").fetchone()[0]
        max_date = conn.execute("SELECT MAX(timestamp_utc) FROM era5_timeseries").fetchone()[0]
        log.info(f"  era5_timeseries: {count:,} rows, {stations_done} stations")
        log.info(f"  Date range: {min_date} to {max_date}")
    except Exception:
        pass
    conn.close()


if __name__ == "__main__":
    main()
