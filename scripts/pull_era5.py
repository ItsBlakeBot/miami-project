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

# Spatial box covering all FL stations + ocean (for grid download mode)
AREA = [31, -85, 24, -79]  # [North, West, South, East]

# All 46 observation locations: 33 ASOS + 10 FAWN + 3 Buoys
# Coordinates verified from FAA AIP (ASOS), FAWN/IFAS (FAWN), NDBC (buoys)
STATIONS = {
    # ── 33 ASOS stations (FAA-verified coordinates) ──────────────
    "KMIA": (25.7954, -80.2901),   # Miami International
    "KFLL": (26.0717, -80.1497),   # Fort Lauderdale-Hollywood
    "KPBI": (26.6832, -80.0956),   # Palm Beach International
    "KHWO": (26.0013, -80.2407),   # North Perry
    "KOPF": (25.9074, -80.2782),   # Opa-locka Executive
    "KFXE": (26.1973, -80.1707),   # Fort Lauderdale Executive
    "KBCT": (26.3785, -80.1077),   # Boca Raton
    "KTMB": (25.6476, -80.4332),   # Kendall-Tamiami Executive
    "KHST": (25.4886, -80.3836),   # Homestead ARB
    "KEYW": (24.5561, -81.7600),   # Key West International
    "KAPF": (26.1524, -81.7756),   # Naples Municipal
    "KRSW": (26.5362, -81.7552),   # Southwest Florida International
    "KFMY": (26.5866, -81.8632),   # Page Field (Fort Myers)
    "KOBE": (27.2666, -80.8504),   # Okeechobee County
    "KSPG": (27.7651, -82.6269),   # Albert Whitted (St Pete)
    "KMLB": (28.1027, -80.6453),   # Melbourne Orlando International
    "KMCO": (28.4294, -81.3090),   # Orlando International
    "KSFB": (28.7772, -81.2349),   # Orlando Sanford
    "KTPA": (27.9755, -82.5332),   # Tampa International
    "KPGD": (26.9186, -81.9909),   # Punta Gorda
    "KVRB": (27.6556, -80.4180),   # Vero Beach Regional
    "KSRQ": (27.3954, -82.5544),   # Sarasota-Bradenton
    "KSEF": (27.4564, -81.3424),   # Sebring Regional
    "KNQX": (24.5746, -81.6866),   # NAS Key West
    "KMTH": (24.7262, -81.0514),   # Marathon (Florida Keys)
    "KFPR": (27.4975, -80.3726),   # Fort Pierce / St Lucie County
    "KPIE": (27.9086, -82.6865),   # St Pete-Clearwater
    "KSUA": (27.1817, -80.2213),   # Witham Field (Stuart)
    "KORL": (28.5455, -81.3329),   # Orlando Executive
    "KPMP": (26.2474, -80.1111),   # Pompano Beach Airpark
    "KLAL": (27.9876, -82.0190),   # Lakeland Linder
    "KBOW": (27.9434, -81.7834),   # Bartow Executive
    "KIMM": (26.4337, -81.4005),   # Immokalee Regional

    # ── 10 FAWN agricultural stations (IFAS-verified coordinates) ─
    "FAWN_405": (26.7391, -81.0528),   # Clewiston
    "FAWN_410": (26.6568, -80.6300),   # Belle Glade
    "FAWN_420": (26.0853, -80.2405),   # Ft. Lauderdale (TREC)
    "FAWN_425": (26.6808, -80.3006),   # Wellington
    "FAWN_430": (27.4271, -80.4054),   # Ft. Pierce (Indian River)
    "FAWN_435": (27.3378, -80.6192),   # St. Lucie West
    "FAWN_440": (25.5126, -80.5031),   # Homestead (TREC)
    "FAWN_450": (26.4623, -81.4403),   # Immokalee (SWFREC)
    "FAWN_455": (27.3306, -80.8514),   # Okeechobee
    "FAWN_460": (26.9248, -81.3146),   # Palmdale

    # ── 3 NDBC/NOS buoy stations (coastal marine) ────────────────
    "VAKF1": (25.7314, -80.1620),   # Virginia Key, Biscayne Bay
    "FWYF1": (25.5906, -80.0969),   # Fowey Rocks (offshore Miami)
    "LKWF1": (26.6128, -80.0339),   # Lake Worth Pier
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
