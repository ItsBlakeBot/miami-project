"""RTMA-RU (Real-Time Mesoscale Analysis - Rapid Update) via Herbie + AWS S3.

2.5km analyzed surface fields every 15 minutes, assimilating all real-time
observations. The best "what IS the temperature right now" product available.

Fields ingested:
  - TMP 2m — analyzed 2m temperature (K -> C)
  - DPT 2m — analyzed 2m dewpoint (K -> C)
  - WIND 10m — analyzed 10m wind speed (m/s)
  - WDIR 10m — analyzed 10m wind direction (deg)
  - PRES surface — analyzed surface pressure (Pa -> hPa)
  - GUST 10m — analyzed 10m wind gust (m/s)
  - TCDC — total cloud cover (%)
  - VIS — visibility (m)

Extracts both the nearest grid point to KMIA and a 5x5 spatial grid for
context (urban heat island gradient, land-sea breeze structure).

Data source: AWS S3 `noaa-rtma-pds` (free, no auth)
Latency: ~20 minutes (RTMA-RU for 12:15 UTC available ~12:35 UTC)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

# KMIA coordinates
KMIA_LAT = 25.7959
KMIA_LON = -80.2870

# 5x5 grid extraction radius (grid points in each direction from center)
GRID_RADIUS = 2


def _k_to_c(kelvin: float | None) -> float | None:
    """Convert Kelvin to Celsius."""
    if kelvin is None:
        return None
    return round(kelvin - 273.15, 2)


def _pa_to_hpa(pascal: float | None) -> float | None:
    """Convert Pascals to hectopascals."""
    if pascal is None:
        return None
    return round(pascal / 100.0, 2)


@dataclass
class RTMAObservation:
    """Single RTMA-RU grid point observation."""

    timestamp_utc: str  # ISO format, analysis valid time
    lat: float
    lon: float  # stored as negative (Western hemisphere convention)
    temperature_2m: float | None = None  # Celsius
    dewpoint_2m: float | None = None  # Celsius
    wind_speed_10m: float | None = None  # m/s
    wind_direction_10m: float | None = None  # degrees
    surface_pressure: float | None = None  # hPa
    wind_gust_10m: float | None = None  # m/s
    cloud_cover_pct: float | None = None  # %
    visibility_m: float | None = None  # meters


def _find_nearest_index(
    lats: np.ndarray, lons: np.ndarray, target_lat: float, target_lon: float
) -> tuple[int, int]:
    """Find nearest grid point index for 2D lat/lon arrays.

    RTMA uses 0-360 longitude convention internally.
    """
    # Convert target lon to 0-360 if negative
    target_lon_360 = target_lon % 360
    dist = (lats - target_lat) ** 2 + (lons - target_lon_360) ** 2
    return np.unravel_index(np.argmin(dist), dist.shape)


def _extract_field(
    herbie_obj,
    search: str,
    idx: tuple[int, int],
) -> float | None:
    """Extract a single field value at the given grid index."""
    try:
        ds = herbie_obj.xarray(search)
        var_name = list(ds.data_vars)[0]
        val = float(ds[var_name].values[idx])
        if np.isnan(val):
            return None
        return round(val, 4)
    except Exception as e:
        log.debug("RTMA-RU field extraction failed for %s: %s", search, e)
        return None


def _extract_grid(
    herbie_obj,
    search: str,
    center_idx: tuple[int, int],
    radius: int,
    grid_shape: tuple[int, int],
) -> np.ndarray | None:
    """Extract a (2*radius+1)^2 grid of values around center_idx."""
    try:
        ds = herbie_obj.xarray(search)
        var_name = list(ds.data_vars)[0]
        y, x = center_idx
        y_lo = max(0, y - radius)
        y_hi = min(grid_shape[0], y + radius + 1)
        x_lo = max(0, x - radius)
        x_hi = min(grid_shape[1], x + radius + 1)
        return ds[var_name].values[y_lo:y_hi, x_lo:x_hi]
    except Exception as e:
        log.debug("RTMA-RU grid extraction failed for %s: %s", search, e)
        return None


def _merge_herbie_datasets(ds_list: list) -> dict[str, Any]:
    """Merge list of xarray Datasets from Herbie's combined download.

    Herbie returns a list of Datasets when using a combined regex search.
    This merges all variables into a single dict for easy access.
    """
    merged: dict[str, Any] = {}
    for ds in ds_list:
        for var_name in ds.data_vars:
            merged[var_name] = ds[var_name]
        # Grab coordinates from first dataset
        if "latitude" not in merged:
            merged["latitude"] = ds["latitude"]
            merged["longitude"] = ds["longitude"]
    return merged


# Variable name mapping: Herbie/cfgrib short names -> our field keys
_VAR_MAP = {
    "t2m": "tmp",       # TMP 2m above ground
    "d2m": "dpt",       # DPT 2m above ground
    "si10": "wind",     # WIND 10m above ground
    "wdir10": "wdir",   # WDIR 10m above ground
    "sp": "pres",       # PRES surface
    "i10fg": "gust",    # GUST 10m above ground
    "tcc": "tcdc",      # TCDC entire atmosphere
    "vis": "vis",       # VIS surface
}


def fetch_rtma_ru(
    lat: float = KMIA_LAT,
    lon: float = KMIA_LON,
    analysis_time: datetime | None = None,
    grid_radius: int = GRID_RADIUS,
) -> list[RTMAObservation] | None:
    """Fetch RTMA-RU analysis for a single time step.

    Returns a list of RTMAObservation: the center point plus surrounding
    grid points (5x5 by default = 25 observations per time step).

    Uses a combined download (single regex) to fetch all fields at once,
    which is significantly faster than individual field downloads.

    Args:
        lat: Target latitude
        lon: Target longitude (negative for west)
        analysis_time: Analysis valid time (UTC). Default: ~35 min ago
            to account for RTMA-RU latency.
        grid_radius: Number of grid points in each direction from center.
            Set to 0 for center point only.

    Returns:
        List of RTMAObservation or None if data unavailable.
    """
    try:
        from herbie import Herbie
    except ImportError:
        log.warning("herbie-data not installed; RTMA-RU adapter disabled")
        return None

    if analysis_time is None:
        # Default to ~35 minutes ago (RTMA-RU has ~20 min latency, add buffer)
        analysis_time = datetime.now(timezone.utc) - timedelta(minutes=35)
        # Round down to nearest 15 minutes
        minute = (analysis_time.minute // 15) * 15
        analysis_time = analysis_time.replace(minute=minute, second=0, microsecond=0)

    date_str = analysis_time.strftime("%Y-%m-%d %H:%M")
    valid_time_str = analysis_time.strftime("%Y-%m-%dT%H:%M:%SZ")

    try:
        H = Herbie(date_str, model="rtma_ru", product="anl", verbose=False)
    except Exception as e:
        log.warning("RTMA-RU not available for %s: %s", date_str, e)
        return None

    # Download all fields in one combined call (much faster than individual)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            result = H.xarray("TMP|DPT|WIND|WDIR|PRES|GUST|TCDC|VIS")
        except Exception as e:
            log.warning("RTMA-RU download failed for %s: %s", date_str, e)
            return None

    # Herbie returns list of datasets for combined regex
    if isinstance(result, list):
        merged = _merge_herbie_datasets(result)
    else:
        merged = {v: result[v] for v in result.data_vars}
        merged["latitude"] = result["latitude"]
        merged["longitude"] = result["longitude"]

    lats = merged["latitude"].values
    lons = merged["longitude"].values
    grid_shape = lats.shape

    center_idx = _find_nearest_index(lats, lons, lat, lon)

    # Extract grid region
    y, x = center_idx
    y_lo = max(0, y - grid_radius)
    y_hi = min(grid_shape[0], y + grid_radius + 1)
    x_lo = max(0, x - grid_radius)
    x_hi = min(grid_shape[1], x + grid_radius + 1)

    grid_lats = lats[y_lo:y_hi, x_lo:x_hi]
    grid_lons = lons[y_lo:y_hi, x_lo:x_hi]

    # Extract grid values for each field
    field_grids: dict[str, np.ndarray | None] = {}
    for cfgrib_name, our_key in _VAR_MAP.items():
        if cfgrib_name in merged:
            try:
                field_grids[our_key] = merged[cfgrib_name].values[y_lo:y_hi, x_lo:x_hi]
            except Exception:
                field_grids[our_key] = None
        else:
            field_grids[our_key] = None

    # Build observations for each grid point
    observations: list[RTMAObservation] = []
    ny, nx = grid_lats.shape

    for iy in range(ny):
        for ix in range(nx):
            pt_lat = float(grid_lats[iy, ix])
            # Convert from 0-360 back to -180..180
            pt_lon_360 = float(grid_lons[iy, ix])
            pt_lon = pt_lon_360 - 360.0 if pt_lon_360 > 180 else pt_lon_360

            def _get(key: str, _iy=iy, _ix=ix) -> float | None:
                grid = field_grids.get(key)
                if grid is None:
                    return None
                try:
                    val = float(grid[_iy, _ix])
                    return None if np.isnan(val) else round(val, 4)
                except (IndexError, ValueError):
                    return None

            observations.append(
                RTMAObservation(
                    timestamp_utc=valid_time_str,
                    lat=round(pt_lat, 4),
                    lon=round(pt_lon, 4),
                    temperature_2m=_k_to_c(_get("tmp")),
                    dewpoint_2m=_k_to_c(_get("dpt")),
                    wind_speed_10m=_get("wind"),
                    wind_direction_10m=_get("wdir"),
                    surface_pressure=_pa_to_hpa(_get("pres")),
                    wind_gust_10m=_get("gust"),
                    cloud_cover_pct=_get("tcdc"),
                    visibility_m=_get("vis"),
                )
            )

    return observations


def fetch_rtma_ru_range(
    start: datetime,
    end: datetime,
    lat: float = KMIA_LAT,
    lon: float = KMIA_LON,
    grid_radius: int = GRID_RADIUS,
    center_only: bool = False,
) -> list[RTMAObservation]:
    """Fetch RTMA-RU for a range of times (for backfill).

    Iterates every 15 minutes from start to end.

    Args:
        start: Start time (UTC, will be rounded down to 15-min)
        end: End time (UTC)
        lat, lon: Target coordinates
        grid_radius: Grid size (0 for center only)
        center_only: If True, only return center point (faster for backfill)
    """
    # Round start down to nearest 15 min
    start = start.replace(
        minute=(start.minute // 15) * 15, second=0, microsecond=0
    )

    all_obs: list[RTMAObservation] = []
    current = start
    step = timedelta(minutes=15)
    total_steps = int((end - start) / step) + 1
    completed = 0

    while current <= end:
        radius = 0 if center_only else grid_radius
        try:
            obs = fetch_rtma_ru(lat, lon, analysis_time=current, grid_radius=radius)
            if obs:
                all_obs.extend(obs)
                completed += 1
                if completed % 20 == 0:
                    pct = completed / total_steps * 100
                    log.info(
                        "RTMA-RU backfill: %d/%d (%.0f%%) — %d obs so far",
                        completed, total_steps, pct, len(all_obs),
                    )
        except Exception as e:
            log.debug("RTMA-RU backfill skip %s: %s", current.isoformat(), e)

        current += step

    log.info(
        "RTMA-RU backfill complete: %d time steps, %d observations",
        completed, len(all_obs),
    )
    return all_obs


def main() -> None:
    """Test RTMA-RU fetch for KMIA."""
    import sys

    logging.basicConfig(level=logging.INFO)

    print(f"Fetching RTMA-RU analysis for KMIA ({KMIA_LAT}, {KMIA_LON})...")

    obs_list = fetch_rtma_ru()
    if not obs_list:
        print("No data available")
        sys.exit(1)

    print(f"Got {len(obs_list)} grid points")

    # Find center point (closest to KMIA)
    center = min(
        obs_list,
        key=lambda o: (o.lat - KMIA_LAT) ** 2 + (o.lon - KMIA_LON) ** 2,
    )
    temp_f = center.temperature_2m * 9 / 5 + 32 if center.temperature_2m is not None else None
    dp_f = center.dewpoint_2m * 9 / 5 + 32 if center.dewpoint_2m is not None else None

    print(f"\nCenter point ({center.lat}, {center.lon}):")
    print(f"  Valid time:  {center.timestamp_utc}")
    print(f"  Temperature: {center.temperature_2m}°C ({temp_f:.1f}°F)" if temp_f else "  Temperature: N/A")
    print(f"  Dewpoint:    {center.dewpoint_2m}°C ({dp_f:.1f}°F)" if dp_f else "  Dewpoint: N/A")
    print(f"  Wind:        {center.wind_speed_10m} m/s @ {center.wind_direction_10m}°")
    print(f"  Gust:        {center.wind_gust_10m} m/s")
    print(f"  Pressure:    {center.surface_pressure} hPa")
    print(f"  Cloud cover: {center.cloud_cover_pct}%")
    print(f"  Visibility:  {center.visibility_m} m")

    # Spatial summary
    temps = [o.temperature_2m for o in obs_list if o.temperature_2m is not None]
    if temps:
        print(f"\nSpatial grid ({len(obs_list)} points):")
        print(f"  Temp range: {min(temps):.2f} to {max(temps):.2f}°C")
        print(f"  Temp spread: {max(temps) - min(temps):.2f}°C")


if __name__ == "__main__":
    main()
