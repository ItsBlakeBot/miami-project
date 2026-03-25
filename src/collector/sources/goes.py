"""GOES-19 satellite product adapter — live cloud cover and stability indices.

Provides ACTUAL OBSERVATIONS (not model forecasts) of:
  - Cloud fraction from Clear Sky Mask (ABI-L2-ACMC) — 2km, 5-min CONUS
  - CAPE from Derived Stability Index (ABI-L2-DSIC) — 10km, 5-min CONUS
  - Lifted Index from DSI — 10km, 5-min CONUS
  - Downward shortwave radiation (ABI-L2-DSRC) — optional, 0.25deg, 5-min

GOES-19 replaced GOES-16 as GOES-East on April 7, 2025.
Data is free on AWS S3: `noaa-goes19` bucket (us-east-1, no auth).

These are satellite-derived measurements, not NWP model output.
They beat HRRR for current conditions because they show what IS happening,
not what a model predicts.

Usage:
    obs = fetch_goes_observations(lat=25.7959, lon=-80.2870)
    print(obs.cloud_fraction, obs.cape_jkg, obs.lifted_index)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class GOESObs:
    """GOES-19 satellite-derived observations at a single point."""

    station: str
    timestamp_utc: str
    lat: float
    lon: float

    # Cloud (from ABI-L2-ACMC Clear Sky Mask)
    cloud_fraction: float | None = None  # 0.0 = clear, 1.0 = overcast
    cloud_mask_raw: int | None = None    # 0=clear, 1=prob_clear, 2=prob_cloudy, 3=cloudy

    # Stability (from ABI-L2-DSIC Derived Stability Index)
    cape_jkg: float | None = None        # Convective Available Potential Energy
    lifted_index: float | None = None    # Lifted Index (°C, negative = unstable)
    k_index: float | None = None         # K-Index
    total_totals: float | None = None    # Total Totals index
    showalter_index: float | None = None # Showalter Index

    # Radiation (from ABI-L2-DSRC)
    dswrf_wm2: float | None = None       # Downward shortwave radiation


def _goes_fixed_grid_to_latlon(ds) -> tuple[np.ndarray, np.ndarray]:
    """Convert GOES fixed grid (x, y) to latitude/longitude arrays.

    Uses the goes_imager_projection variable which contains the
    geostationary projection parameters.
    """
    proj = ds["goes_imager_projection"]
    H = float(proj.attrs["perspective_point_height"]) + float(proj.attrs["semi_major_axis"])
    r_eq = float(proj.attrs["semi_major_axis"])
    r_pol = float(proj.attrs["semi_minor_axis"])
    lon_0 = float(proj.attrs["longitude_of_projection_origin"])
    e = float(proj.attrs.get("inverse_flattening", 298.257222096))

    x = ds["x"].values
    y = ds["y"].values
    xx, yy = np.meshgrid(x, y)

    sin_x = np.sin(xx)
    cos_x = np.cos(xx)
    sin_y = np.sin(yy)
    cos_y = np.cos(yy)

    a = sin_x**2 + cos_x**2 * (cos_y**2 + (r_eq / r_pol)**2 * sin_y**2)
    b = -2 * H * cos_x * cos_y
    c = H**2 - r_eq**2

    discriminant = b**2 - 4 * a * c
    mask = discriminant >= 0

    rs = np.full_like(a, np.nan)
    rs[mask] = (-b[mask] - np.sqrt(discriminant[mask])) / (2 * a[mask])

    sx = rs * cos_x * cos_y
    sy = -rs * sin_x
    sz = rs * cos_x * sin_y

    lat = np.degrees(np.arctan((r_eq / r_pol)**2 * sz / np.sqrt((H - sx)**2 + sy**2)))
    lon = lon_0 - np.degrees(np.arctan(sy / (H - sx)))

    return lat, lon


def _extract_nearest(ds, var_name: str, lat: float, lon: float) -> float | None:
    """Extract nearest-neighbor value from GOES xarray Dataset.

    GOES uses a fixed grid geostationary projection. We convert to lat/lon
    and find the nearest grid point.
    """
    try:
        if var_name not in ds:
            return None

        da = ds[var_name]

        # Check if latitude/longitude are already available as coords
        if "latitude" in ds.coords and "longitude" in ds.coords:
            lats = ds["latitude"].values
            lons = ds["longitude"].values
        elif "goes_imager_projection" in ds:
            lats, lons = _goes_fixed_grid_to_latlon(ds)
        else:
            return None

        if lats.ndim == 2:
            dist = (lats - lat) ** 2 + (lons - lon) ** 2
            valid = ~np.isnan(dist)
            if not valid.any():
                return None
            dist[~valid] = 1e20
            idx = np.unravel_index(np.argmin(dist), dist.shape)
            val = float(da.values[idx])
        else:
            return None

        if np.isnan(val) or val < -9000:
            return None
        return round(val, 4)
    except Exception as e:
        log.debug("GOES extraction failed for %s: %s", var_name, e)
        return None


def fetch_goes_observations(
    lat: float = 25.7959,
    lon: float = -80.2870,
    station: str = "KMIA",
) -> GOESObs | None:
    """Fetch latest GOES-19 satellite observations for a point.

    Attempts to fetch Clear Sky Mask and Derived Stability Index products.
    Each product is fetched independently — partial data is OK.

    Returns:
        GOESObs with available satellite-derived measurements, or None on failure.
    """
    try:
        from goes2go import GOES
    except ImportError:
        log.warning("goes2go not installed; GOES adapter disabled")
        return None

    now_utc = datetime.now(timezone.utc)
    timestamp = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

    obs = GOESObs(
        station=station,
        timestamp_utc=timestamp,
        lat=lat,
        lon=lon,
    )

    # 1. Clear Sky Mask (ABI-L2-ACM, CONUS domain)
    try:
        G = GOES(satellite=19, product="ABI-L2-ACMC", domain="C")
        ds = G.latest()
        if ds is not None:
            # BCM = Binary Cloud Mask (0=clear, 1=cloudy)
            # ACM = 4-level (0=clear, 1=prob_clear, 2=prob_cloudy, 3=cloudy)
            for var in ["BCM", "ACM", "Cloud_Mask"]:
                val = _extract_nearest(ds, var, lat, lon)
                if val is not None:
                    obs.cloud_mask_raw = int(val)
                    if var == "BCM":
                        # Binary: 0=clear→0.0, 1=cloudy→1.0
                        obs.cloud_fraction = 1.0 if val >= 0.5 else 0.0
                    else:
                        # 4-level ACM: 0=clear→0.0, 1=prob_clear→0.33,
                        # 2=prob_cloudy→0.67, 3=cloudy→1.0
                        obs.cloud_fraction = round(min(1.0, max(0.0, val / 3.0)), 3)
                    break
    except Exception as e:
        log.debug("GOES Clear Sky Mask unavailable: %s", e)

    # 2. Derived Stability Index (ABI-L2-DSI, CONUS domain)
    try:
        G = GOES(satellite=19, product="ABI-L2-DSIC", domain="C")
        ds = G.latest()
        if ds is not None:
            obs.cape_jkg = _extract_nearest(ds, "CAPE", lat, lon)
            obs.lifted_index = _extract_nearest(ds, "LI", lat, lon)
            obs.k_index = _extract_nearest(ds, "KI", lat, lon)
            obs.total_totals = _extract_nearest(ds, "TT", lat, lon)
            obs.showalter_index = _extract_nearest(ds, "SI", lat, lon)
    except Exception as e:
        log.debug("GOES DSI unavailable: %s", e)

    # 3. Downward Shortwave Radiation (ABI-L2-DSR, CONUS domain)
    try:
        G = GOES(satellite=19, product="ABI-L2-DSRC", domain="C")
        ds = G.latest()
        if ds is not None:
            obs.dswrf_wm2 = _extract_nearest(ds, "DSR", lat, lon)
    except Exception as e:
        log.debug("GOES DSR unavailable: %s", e)

    # Return obs if we got anything
    has_data = any([
        obs.cloud_fraction is not None,
        obs.cape_jkg is not None,
        obs.lifted_index is not None,
        obs.dswrf_wm2 is not None,
    ])

    return obs if has_data else None


def main() -> None:
    """Test GOES fetch for KMIA."""
    import sys
    logging.basicConfig(level=logging.INFO)

    lat, lon = 25.7959, -80.2870
    print(f"Fetching GOES-19 satellite obs for ({lat}, {lon})...")

    obs = fetch_goes_observations(lat, lon, station="KMIA")
    if obs is None:
        print("No satellite data available")
        sys.exit(1)

    print(f"  Timestamp: {obs.timestamp_utc}")
    print(f"  Cloud fraction: {obs.cloud_fraction}")
    print(f"  Cloud mask raw: {obs.cloud_mask_raw}")
    print(f"  CAPE: {obs.cape_jkg} J/kg")
    print(f"  Lifted Index: {obs.lifted_index} °C")
    print(f"  K-Index: {obs.k_index}")
    print(f"  Total Totals: {obs.total_totals}")
    print(f"  DSWRF: {obs.dswrf_wm2} W/m²")


if __name__ == "__main__":
    main()
