"""HRRR (High-Resolution Rapid Refresh) adapter via Herbie + AWS S3.

Replaces Open-Meteo atmospheric data with free, higher-resolution HRRR fields.
3km grid, hourly model cycles, available ~1h after initialization.

Fields ingested:
  - CAPE (surface-based) — convective available potential energy
  - PWAT (entire atmosphere) — precipitable water
  - TCDC (entire atmosphere) — total cloud cover percentage
  - DSWRF (surface) — downward shortwave radiation
  - HPBL (surface) — planetary boundary layer height
  - TMP at 925, 850, 700, 500 hPa — upper-air temperature
  - UGRD/VGRD at 850 hPa — 850mb wind components

Data source: AWS S3 `noaa-hrrr-bdp-pds` (free, no auth)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class HRRRAtmosphericObs:
    """Atmospheric fields extracted from a single HRRR analysis/forecast."""

    station: str
    model_init_utc: str  # ISO format
    valid_time_utc: str
    forecast_hour: int

    # Convection
    cape_jkg: float | None = None       # Surface-based CAPE (J/kg)
    pwat_mm: float | None = None        # Precipitable water (kg/m² = mm)

    # Cloud & radiation
    cloud_cover_pct: float | None = None  # Total cloud cover (%)
    dswrf_wm2: float | None = None        # Downward shortwave radiation (W/m²)

    # Boundary layer
    hpbl_m: float | None = None          # PBL height (m)

    # Upper-air temperatures (K → °C in storage)
    temp_925_c: float | None = None
    temp_850_c: float | None = None
    temp_700_c: float | None = None
    temp_500_c: float | None = None

    # 850 hPa wind
    wind_u_850_ms: float | None = None   # U-component (m/s)
    wind_v_850_ms: float | None = None   # V-component (m/s)

    @property
    def wind_speed_850_ms(self) -> float | None:
        if self.wind_u_850_ms is not None and self.wind_v_850_ms is not None:
            return math.sqrt(self.wind_u_850_ms**2 + self.wind_v_850_ms**2)
        return None

    @property
    def wind_dir_850(self) -> float | None:
        if self.wind_u_850_ms is not None and self.wind_v_850_ms is not None:
            return (270 - math.degrees(math.atan2(self.wind_v_850_ms, self.wind_u_850_ms))) % 360
        return None


def _k_to_c(kelvin: float | None) -> float | None:
    """Convert Kelvin to Celsius."""
    if kelvin is None:
        return None
    return round(kelvin - 273.15, 2)


def _extract_point(ds, var_name: str, lat: float, lon: float) -> float | None:
    """Extract nearest-neighbor value from xarray Dataset at (lat, lon).

    HRRR uses a Lambert Conformal grid with 2D lat/lon coordinates
    (dims are y, x — not lat, lon). We find the nearest grid point
    by computing distance to all grid points.
    """
    import numpy as np

    try:
        if var_name not in ds:
            var_name = list(ds.data_vars)[0] if ds.data_vars else None
            if var_name is None:
                return None

        da = ds[var_name]

        # HRRR has 2D latitude/longitude coordinates on (y, x) dims
        if "latitude" in ds.coords and "longitude" in ds.coords:
            lats = ds["latitude"].values
            lons = ds["longitude"].values

            if lats.ndim == 2:
                # 2D grid — find nearest point by distance
                dist = (lats - lat) ** 2 + (lons - lon) ** 2
                idx = np.unravel_index(np.argmin(dist), dist.shape)
                val = float(da.values[idx])
            else:
                # 1D coords — use sel
                val = float(da.sel(latitude=lat, longitude=lon, method="nearest").values)
        else:
            val = float(da.values.flat[0])

        if np.isnan(val):
            return None
        return round(val, 4)
    except Exception as e:
        log.debug("Point extraction failed for %s: %s", var_name, e)
        return None


def fetch_hrrr_atmospheric(
    lat: float,
    lon: float,
    station: str = "KMIA",
    model_run_utc: datetime | None = None,
    forecast_hour: int = 0,
) -> HRRRAtmosphericObs | None:
    """Fetch HRRR atmospheric fields for a single point.

    Uses Herbie to download specific GRIB2 fields from AWS S3 and extract
    the nearest grid point value.

    Args:
        lat: Station latitude
        lon: Station longitude (negative for west)
        station: Station identifier for metadata
        model_run_utc: HRRR model initialization time. Default: most recent.
        forecast_hour: Forecast lead time (0 = analysis, 1-18 for standard runs)

    Returns:
        HRRRAtmosphericObs or None if data unavailable.
    """
    try:
        from herbie import Herbie
    except ImportError:
        log.warning("herbie-data not installed; HRRR adapter disabled")
        return None

    if model_run_utc is None:
        # Default to 2 hours ago (HRRR takes ~1h to be available)
        model_run_utc = datetime.now(timezone.utc) - timedelta(hours=2)
        # Round down to nearest hour
        model_run_utc = model_run_utc.replace(minute=0, second=0, microsecond=0)

    date_str = model_run_utc.strftime("%Y-%m-%d %H:%M")

    # HRRR field search strings (from GRIB2 inventory)
    fields = {
        "cape": ":CAPE:surface",
        "pwat": ":PWAT:entire atmosphere",
        "tcdc": ":TCDC:entire atmosphere",
        "dswrf": ":DSWRF:surface",
        "hpbl": ":HPBL:surface",
        "tmp_925": ":TMP:925 mb",
        "tmp_850": ":TMP:850 mb",
        "tmp_700": ":TMP:700 mb",
        "tmp_500": ":TMP:500 mb",
        "ugrd_850": ":UGRD:850 mb",
        "vgrd_850": ":VGRD:850 mb",
    }

    results: dict[str, float | None] = {}

    # Herbie's xarray output preserves the HRRR grid's coordinate convention.
    # For CONUS HRRR this is typically negative longitudes (matching the input).
    # Do NOT convert to 0-360 — _extract_point handles whatever convention Herbie uses.
    for key, search in fields.items():
        try:
            product = "prs" if "mb" in search else "sfc"
            H = Herbie(
                date_str,
                model="hrrr",
                product=product,
                fxx=forecast_hour,
                verbose=False,
            )
            ds = H.xarray(search)
            results[key] = _extract_point(ds, list(ds.data_vars)[0], lat, lon)
        except Exception as e:
            log.debug("HRRR field %s unavailable: %s", key, e)
            results[key] = None

    valid_time = model_run_utc + timedelta(hours=forecast_hour)

    return HRRRAtmosphericObs(
        station=station,
        model_init_utc=model_run_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        valid_time_utc=valid_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        forecast_hour=forecast_hour,
        cape_jkg=results.get("cape"),
        pwat_mm=results.get("pwat"),
        cloud_cover_pct=results.get("tcdc"),
        dswrf_wm2=results.get("dswrf"),
        hpbl_m=results.get("hpbl"),
        temp_925_c=_k_to_c(results.get("tmp_925")),
        temp_850_c=_k_to_c(results.get("tmp_850")),
        temp_700_c=_k_to_c(results.get("tmp_700")),
        temp_500_c=_k_to_c(results.get("tmp_500")),
        wind_u_850_ms=results.get("ugrd_850"),
        wind_v_850_ms=results.get("vgrd_850"),
    )


def main() -> None:
    """Test HRRR fetch for KMIA."""
    import sys
    logging.basicConfig(level=logging.INFO)

    lat, lon = 25.7959, -80.2870
    print(f"Fetching HRRR atmospheric data for ({lat}, {lon})...")

    obs = fetch_hrrr_atmospheric(lat, lon, station="KMIA")
    if obs is None:
        print("No data available")
        sys.exit(1)

    print(f"  Model init: {obs.model_init_utc}")
    print(f"  Valid time: {obs.valid_time_utc}")
    print(f"  CAPE: {obs.cape_jkg} J/kg")
    print(f"  PWAT: {obs.pwat_mm} mm")
    print(f"  Cloud cover: {obs.cloud_cover_pct}%")
    print(f"  DSWRF: {obs.dswrf_wm2} W/m²")
    print(f"  HPBL: {obs.hpbl_m} m")
    print(f"  T925: {obs.temp_925_c}°C")
    print(f"  T850: {obs.temp_850_c}°C")
    print(f"  T700: {obs.temp_700_c}°C")
    print(f"  T500: {obs.temp_500_c}°C")
    print(f"  Wind 850: {obs.wind_speed_850_ms:.1f} m/s @ {obs.wind_dir_850:.0f}°" if obs.wind_speed_850_ms else "  Wind 850: N/A")


if __name__ == "__main__":
    main()
