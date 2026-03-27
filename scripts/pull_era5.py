#!/usr/bin/env python3
"""Download ERA5 reanalysis data for South Florida weather trading.

Downloads hourly single-level and pressure-level ERA5 data for a spatial
box covering all 20 South FL stations used in the model. Data goes back
to 1980, providing ~336,000 station-days vs our current 1,826.

Setup:
    1. pip install cdsapi netcdf4 xarray
    2. Create ~/.cdsapirc with your CDS API key:
       url: https://cds.climate.copernicus.eu/api
       key: YOUR-API-KEY-HERE
    3. Run: python scripts/pull_era5.py

The script downloads year-by-year to avoid CDS request size limits,
then ingests into miami_collector.db as era5_surface and era5_pressure tables.

Spatial box: 24°N to 28°N, -83°W to -79°W (all of South FL + Gulf Stream)
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
log = logging.getLogger("pull_era5")

try:
    from ecmwf.datastores import Client as CDSClient
    USE_NEW_CLIENT = True
except ImportError:
    try:
        import cdsapi
        USE_NEW_CLIENT = False
    except ImportError:
        log.error("No CDS client installed. Run: pip install ecmwf-datastores-client")
        log.error("Or fallback: pip install cdsapi")
        sys.exit(1)

try:
    import xarray as xr
    import numpy as np
    import pandas as pd
except ImportError:
    log.error("Missing deps. Run: pip install xarray netcdf4 numpy pandas")
    sys.exit(1)


# ──────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────

# Spatial box covering South FL + surrounding ocean
# Covers all 20 model stations plus Gulf Stream SST zone
AREA = [28, -83, 24, -79]  # [North, West, South, East]

# Station coordinates for nearest-grid-point extraction
STATIONS = {
    "KMIA": (25.793, -80.291),
    "KFLL": (26.073, -80.153),
    "KPBI": (26.683, -80.096),
    "KHWO": (26.001, -80.241),
    "KOPF": (25.907, -80.279),
    "KFXE": (26.197, -80.170),
    "KBCT": (26.379, -80.108),
    "KTMB": (25.648, -80.433),
    "KHST": (25.487, -80.383),
    "KEYW": (24.556, -81.760),
    "KAPF": (26.152, -81.775),
    "KRSW": (26.536, -81.755),
    "KFMY": (26.586, -81.863),
    "KOBE": (26.269, -81.443),
    "KSPG": (27.765, -82.627),
    "KMLB": (28.102, -80.645),
    "KMCO": (28.430, -81.309),
    "KDAB": (29.180, -81.058),
    "KSFB": (28.778, -81.244),
    "KTLH": (30.397, -84.350),
}

# ERA5 single-level variables (maps to our model features)
SINGLE_LEVEL_VARS = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "surface_pressure",
    "mean_sea_level_pressure",
    "total_precipitation",
    "surface_net_solar_radiation",
    "total_cloud_cover",
    "convective_available_potential_energy",
    "boundary_layer_height",
    "soil_temperature_level_1",
]

# ERA5 pressure-level variables (maps to upper air features)
PRESSURE_LEVEL_VARS = [
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "relative_humidity",
    "geopotential",
]

PRESSURE_LEVELS = ["925", "850", "700", "500"]

# Hours to download (all 24)
HOURS = [f"{h:02d}:00" for h in range(24)]

# Output directory
OUTPUT_DIR = Path("data/era5")


def download_single_levels(client, year, output_dir):
    """Download ERA5 single-level hourly data for one year."""
    output_file = output_dir / f"era5_single_{year}.nc"

    if output_file.exists():
        log.info(f"  {output_file} already exists, skipping")
        return output_file

    log.info(f"  Requesting ERA5 single levels for {year}...")

    request = {
        "product_type": ["reanalysis"],
        "variable": SINGLE_LEVEL_VARS,
        "year": [str(year)],
        "month": [f"{m:02d}" for m in range(1, 13)],
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": HOURS,
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": AREA,
    }

    client.retrieve(
        "reanalysis-era5-single-levels",
        request,
        str(output_file),
    )
    log.info(f"  Downloaded: {output_file} ({output_file.stat().st_size / 1e6:.1f} MB)")
    return output_file


def download_pressure_levels(client, year, output_dir):
    """Download ERA5 pressure-level hourly data for one year."""
    output_file = output_dir / f"era5_pressure_{year}.nc"

    if output_file.exists():
        log.info(f"  {output_file} already exists, skipping")
        return output_file

    log.info(f"  Requesting ERA5 pressure levels for {year}...")

    request = {
        "product_type": ["reanalysis"],
        "variable": PRESSURE_LEVEL_VARS,
        "year": [str(year)],
        "month": [f"{m:02d}" for m in range(1, 13)],
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": HOURS,
        "pressure_level": PRESSURE_LEVELS,
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": AREA,
    }

    client.retrieve(
        "reanalysis-era5-pressure-levels",
        request,
        str(output_file),
    )
    log.info(f"  Downloaded: {output_file} ({output_file.stat().st_size / 1e6:.1f} MB)")
    return output_file


def extract_station_data(nc_file, stations, is_pressure=False):
    """Extract data at station locations from ERA5 NetCDF file."""
    ds = xr.open_dataset(nc_file)

    rows = []
    for station_id, (lat, lon) in stations.items():
        # Select nearest grid point to station
        point = ds.sel(latitude=lat, longitude=lon % 360 if lon < 0 else lon,
                       method="nearest")

        # Handle longitude convention (ERA5 uses 0-360 sometimes)
        if len(point.dims) == 0 or 'time' not in point.dims:
            # Try with negative longitude
            point = ds.sel(latitude=lat, longitude=lon, method="nearest")

        for t_idx in range(len(point.time)):
            row = {
                "station_id": station_id,
                "timestamp_utc": str(pd.Timestamp(point.time.values[t_idx])),
            }

            for var in point.data_vars:
                if is_pressure:
                    # For pressure levels, flatten: var_925, var_850, etc
                    if 'pressure_level' in point[var].dims:
                        for plev in point.pressure_level.values:
                            key = f"{var}_{int(plev)}"
                            try:
                                row[key] = float(point[var].sel(
                                    pressure_level=plev).isel(time=t_idx).values)
                            except Exception:
                                row[key] = None
                    else:
                        try:
                            row[var] = float(point[var].isel(time=t_idx).values)
                        except Exception:
                            row[var] = None
                else:
                    try:
                        row[var] = float(point[var].isel(time=t_idx).values)
                    except Exception:
                        row[var] = None

            rows.append(row)

    ds.close()
    return pd.DataFrame(rows)


def ingest_to_db(df, db_path, table_name):
    """Insert DataFrame into SQLite database."""
    conn = sqlite3.connect(db_path)

    df.to_sql(table_name, conn, if_exists="append", index=False)

    count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    log.info(f"  {table_name}: {count:,} total rows")

    conn.close()


def main():
    parser = argparse.ArgumentParser(description="Download ERA5 data for Miami weather trading")
    parser.add_argument("--db", type=str, default="miami_collector.db",
                        help="Path to SQLite database")
    parser.add_argument("--start-year", type=int, default=1980,
                        help="Start year (default: 1980)")
    parser.add_argument("--end-year", type=int, default=2026,
                        help="End year (default: 2026)")
    parser.add_argument("--single-levels-only", action="store_true",
                        help="Skip pressure level downloads")
    parser.add_argument("--output-dir", type=str, default="data/era5",
                        help="Directory for NetCDF downloads")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if USE_NEW_CLIENT:
        client = CDSClient()
        log.info("Using ecmwf-datastores-client (new)")
    else:
        client = cdsapi.Client()
        log.info("Using cdsapi (legacy)")

    log.info("=" * 60)
    log.info("ERA5 Data Download for Miami Weather Trading")
    log.info("=" * 60)
    log.info(f"Years: {args.start_year} to {args.end_year}")
    log.info(f"Area: {AREA} (South FL + Gulf Stream)")
    log.info(f"Stations: {len(STATIONS)}")
    log.info(f"Single-level vars: {len(SINGLE_LEVEL_VARS)}")
    if not args.single_levels_only:
        log.info(f"Pressure-level vars: {len(PRESSURE_LEVEL_VARS)} × {len(PRESSURE_LEVELS)} levels")
    log.info(f"Output: {output_dir}")
    log.info(f"Database: {args.db}")
    log.info("")

    # Create tables if needed
    conn = sqlite3.connect(args.db)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS era5_surface (
            station_id TEXT NOT NULL,
            timestamp_utc TEXT NOT NULL,
            t2m REAL,
            d2m REAL,
            u10 REAL,
            v10 REAL,
            sp REAL,
            msl REAL,
            tp REAL,
            ssr REAL,
            tcc REAL,
            cape REAL,
            blh REAL,
            stl1 REAL,
            PRIMARY KEY (station_id, timestamp_utc)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS era5_pressure (
            station_id TEXT NOT NULL,
            timestamp_utc TEXT NOT NULL,
            t_925 REAL, t_850 REAL, t_700 REAL, t_500 REAL,
            u_925 REAL, u_850 REAL, u_700 REAL, u_500 REAL,
            v_925 REAL, v_850 REAL, v_700 REAL, v_500 REAL,
            r_925 REAL, r_850 REAL, r_700 REAL, r_500 REAL,
            z_925 REAL, z_850 REAL, z_700 REAL, z_500 REAL,
            PRIMARY KEY (station_id, timestamp_utc)
        )
    """)
    conn.close()

    total_years = args.end_year - args.start_year + 1
    for i, year in enumerate(range(args.start_year, args.end_year + 1)):
        log.info(f"\n--- Year {year} ({i + 1}/{total_years}) ---")

        # Single levels
        try:
            nc_file = download_single_levels(client, year, output_dir)
            log.info(f"  Extracting station data from {nc_file}...")
            df = extract_station_data(nc_file, STATIONS, is_pressure=False)
            log.info(f"  Extracted {len(df):,} rows")
            ingest_to_db(df, args.db, "era5_surface")
        except Exception as e:
            log.error(f"  Single levels failed for {year}: {e}")

        # Pressure levels
        if not args.single_levels_only:
            try:
                nc_file = download_pressure_levels(client, year, output_dir)
                log.info(f"  Extracting pressure data from {nc_file}...")
                df = extract_station_data(nc_file, STATIONS, is_pressure=True)
                log.info(f"  Extracted {len(df):,} rows")
                ingest_to_db(df, args.db, "era5_pressure")
            except Exception as e:
                log.error(f"  Pressure levels failed for {year}: {e}")

        # Rate limiting — CDS has request quotas
        if i < total_years - 1:
            log.info("  Waiting 5s before next request...")
            time.sleep(5)

    log.info("\n" + "=" * 60)
    log.info("ERA5 DOWNLOAD COMPLETE")
    log.info("=" * 60)

    # Final stats
    conn = sqlite3.connect(args.db)
    for table in ["era5_surface", "era5_pressure"]:
        try:
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            min_date = conn.execute(f"SELECT MIN(timestamp_utc) FROM {table}").fetchone()[0]
            max_date = conn.execute(f"SELECT MAX(timestamp_utc) FROM {table}").fetchone()[0]
            log.info(f"  {table}: {count:,} rows ({min_date} to {max_date})")
        except Exception:
            pass
    conn.close()


if __name__ == "__main__":
    main()
