"""Herbie-based NWP forecast collector for GFS, ECMWF IFS, ECMWF AIFS, and NBM.

Replaces Open-Meteo paid API calls with free GRIB2 data from:
  - GFS deterministic: AWS S3 `noaa-gfs-bdp-pds` (NOAA, free)
  - ECMWF IFS/AIFS: ECMWF Open Data (CC-BY-4.0, free commercial use since Oct 2025)
  - NBM: AWS S3 `noaa-nbm-pds` (NOAA, free)

Each function extracts hourly 2m temperature at a single lat/lon point,
returning ModelForecast objects compatible with the existing DB schema.

Data source: Herbie library (https://herbie.readthedocs.io)
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Any

from collector.types import ModelForecast

log = logging.getLogger(__name__)

UTC = timezone.utc


def _k_to_f(kelvin: float | None) -> float | None:
    """Convert Kelvin to Fahrenheit."""
    if kelvin is None:
        return None
    return round((kelvin - 273.15) * 9.0 / 5.0 + 32.0, 2)


def _c_to_f(celsius: float | None) -> float | None:
    """Convert Celsius to Fahrenheit."""
    if celsius is None:
        return None
    return round(celsius * 9.0 / 5.0 + 32.0, 2)


def _extract_point_value(ds: Any, lat: float, lon: float) -> float | None:
    """Extract nearest-neighbor value from xarray Dataset at (lat, lon).

    Handles both regular lat/lon grids (ECMWF, GFS 0.25) and
    projected grids (Lambert Conformal like HRRR/NBM).
    """
    import numpy as np

    try:
        if not ds.data_vars:
            return None

        var_name = list(ds.data_vars)[0]
        da = ds[var_name]

        # Regular lat/lon grid (ECMWF, GFS)
        if "latitude" in ds.dims or "lat" in ds.dims:
            lat_dim = "latitude" if "latitude" in ds.dims else "lat"
            lon_dim = "longitude" if "longitude" in ds.dims else "lon"
            # Handle ECMWF 0-360 longitude convention
            ds_lon = lon
            if ds[lon_dim].values.min() >= 0 and lon < 0:
                ds_lon = lon + 360.0
            val = float(da.sel(**{lat_dim: lat, lon_dim: ds_lon}, method="nearest").values)
        elif "latitude" in ds.coords and "longitude" in ds.coords:
            # 2D coordinate arrays (projected grids like NBM)
            lats = ds["latitude"].values
            lons = ds["longitude"].values
            if lats.ndim == 2:
                dist = (lats - lat) ** 2 + (lons - lon) ** 2
                idx = np.unravel_index(np.argmin(dist), dist.shape)
                val = float(da.values[idx])
            else:
                val = float(da.sel(latitude=lat, longitude=lon, method="nearest").values)
        else:
            val = float(da.values.flat[0])

        if np.isnan(val):
            return None
        return val
    except Exception as e:
        log.debug("Point extraction failed: %s", e)
        return None


def _latest_available_run(
    model_cycles: list[int],
    availability_lag_hours: float,
    now_utc: datetime | None = None,
) -> datetime:
    """Find the most recent model run whose data should be available."""
    now = now_utc or datetime.now(UTC)
    for hours_back in range(0, 48):
        candidate = now - timedelta(hours=hours_back)
        candidate = candidate.replace(minute=0, second=0, microsecond=0)
        if candidate.hour in model_cycles:
            ready_at = candidate + timedelta(hours=availability_lag_hours)
            if ready_at <= now:
                return candidate
    # Fallback: 24h ago at 00Z
    fallback = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return fallback


def fetch_gfs_forecasts(
    lat: float,
    lon: float,
    station: str,
    target_date: str,
) -> list[ModelForecast]:
    """Fetch GFS 0.25-degree hourly 2m temperature forecasts via Herbie.

    Source: AWS S3 `noaa-gfs-bdp-pds` (free, no auth).
    """
    try:
        from herbie import Herbie
    except ImportError:
        log.warning("herbie-data not installed; GFS Herbie adapter disabled")
        return []

    model_run = _latest_available_run([0, 6, 12, 18], 4.5)
    run_time_str = model_run.strftime("%Y-%m-%d %H:%M")
    now_utc = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    forecasts: list[ModelForecast] = []

    # Fetch hourly 2m temperature for forecast hours 0-36
    for fhour in range(0, 37):
        try:
            H = Herbie(
                run_time_str,
                model="gfs",
                product="pgrb2.0p25",
                fxx=fhour,
                verbose=False,
            )
            ds = H.xarray(":TMP:2 m above ground")
            val_k = _extract_point_value(ds, lat, lon)
            val_f = _k_to_f(val_k)
            ds.close()

            if val_f is None:
                continue

            valid_dt = model_run + timedelta(hours=fhour)
            valid_str = valid_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            valid_date = valid_str[:10]

            # Only keep forecasts for the target date
            if valid_date != target_date:
                continue

            forecasts.append(ModelForecast(
                station=station,
                forecast_date=target_date,
                model="GFS-Herbie",
                source="herbie",
                valid_time=valid_str,
                raw_temperature_f=val_f,
                run_time=model_run.strftime("%Y-%m-%d %H:%M:%S"),
                fetch_time_utc=now_utc,
            ))
        except Exception as e:
            log.debug("GFS fhour=%d failed: %s", fhour, e)
            continue

    # Compute daily high/low from hourly temps
    if forecasts:
        temps = [f.raw_temperature_f for f in forecasts if f.raw_temperature_f is not None]
        if temps:
            forecasts.append(ModelForecast(
                station=station,
                forecast_date=target_date,
                model="GFS-Herbie",
                source="herbie",
                forecast_high_f=max(temps),
                forecast_low_f=min(temps),
                run_time=model_run.strftime("%Y-%m-%d %H:%M:%S"),
                fetch_time_utc=now_utc,
            ))

    log.info("GFS-Herbie: %d forecasts for %s (run %s)", len(forecasts), target_date, run_time_str)
    return forecasts


def fetch_ecmwf_ifs_forecasts(
    lat: float,
    lon: float,
    station: str,
    target_date: str,
) -> list[ModelForecast]:
    """Fetch ECMWF IFS 0.25-degree hourly 2m temperature via ECMWF Open Data.

    Source: ECMWF Open Data (CC-BY-4.0, free commercial use since Oct 2025).
    """
    try:
        from herbie import Herbie
    except ImportError:
        log.warning("herbie-data not installed; ECMWF Herbie adapter disabled")
        return []

    model_run = _latest_available_run([0, 6, 12, 18], 5.0)
    run_time_str = model_run.strftime("%Y-%m-%d %H:%M")
    now_utc = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    forecasts: list[ModelForecast] = []

    for fhour in range(0, 37):
        try:
            H = Herbie(
                run_time_str,
                model="ifs",
                product="oper",
                fxx=fhour,
                verbose=False,
            )
            ds = H.xarray(":2t:")
            val_k = _extract_point_value(ds, lat, lon)
            val_f = _k_to_f(val_k)
            ds.close()

            if val_f is None:
                continue

            valid_dt = model_run + timedelta(hours=fhour)
            valid_str = valid_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            valid_date = valid_str[:10]

            if valid_date != target_date:
                continue

            forecasts.append(ModelForecast(
                station=station,
                forecast_date=target_date,
                model="ECMWF-IFS-Herbie",
                source="herbie",
                valid_time=valid_str,
                raw_temperature_f=val_f,
                run_time=model_run.strftime("%Y-%m-%d %H:%M:%S"),
                fetch_time_utc=now_utc,
            ))
        except Exception as e:
            log.debug("ECMWF IFS fhour=%d failed: %s", fhour, e)
            continue

    if forecasts:
        temps = [f.raw_temperature_f for f in forecasts if f.raw_temperature_f is not None]
        if temps:
            forecasts.append(ModelForecast(
                station=station,
                forecast_date=target_date,
                model="ECMWF-IFS-Herbie",
                source="herbie",
                forecast_high_f=max(temps),
                forecast_low_f=min(temps),
                run_time=model_run.strftime("%Y-%m-%d %H:%M:%S"),
                fetch_time_utc=now_utc,
            ))

    log.info("ECMWF-IFS-Herbie: %d forecasts for %s (run %s)", len(forecasts), target_date, run_time_str)
    return forecasts


def fetch_nbm_forecasts(
    lat: float,
    lon: float,
    station: str,
    target_date: str,
) -> list[ModelForecast]:
    """Fetch NBM (National Blend of Models) hourly 2m temperature via Herbie.

    NBM is arguably the best calibrated single-model US temperature forecast.
    It blends NWS and non-NWS models into probabilistic guidance.

    Source: AWS S3 `noaa-nbm-pds` (free, no auth).
    """
    try:
        from herbie import Herbie
    except ImportError:
        log.warning("herbie-data not installed; NBM Herbie adapter disabled")
        return []

    # NBM runs every hour but full 36h forecasts only at 01/07/13/19Z
    model_run = _latest_available_run([1, 7, 13, 19], 2.5)
    run_time_str = model_run.strftime("%Y-%m-%d %H:%M")
    now_utc = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    forecasts: list[ModelForecast] = []

    for fhour in range(1, 37):
        try:
            H = Herbie(
                run_time_str,
                model="nbm",
                product="co",
                fxx=fhour,
                verbose=False,
            )
            ds = H.xarray(":TMP:2 m above ground")
            val_k = _extract_point_value(ds, lat, lon)
            val_f = _k_to_f(val_k)
            ds.close()

            if val_f is None:
                continue

            valid_dt = model_run + timedelta(hours=fhour)
            valid_str = valid_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            valid_date = valid_str[:10]

            if valid_date != target_date:
                continue

            forecasts.append(ModelForecast(
                station=station,
                forecast_date=target_date,
                model="NBM-Herbie",
                source="herbie",
                valid_time=valid_str,
                raw_temperature_f=val_f,
                run_time=model_run.strftime("%Y-%m-%d %H:%M:%S"),
                fetch_time_utc=now_utc,
            ))
        except Exception as e:
            log.debug("NBM fhour=%d failed: %s", fhour, e)
            continue

    if forecasts:
        temps = [f.raw_temperature_f for f in forecasts if f.raw_temperature_f is not None]
        if temps:
            forecasts.append(ModelForecast(
                station=station,
                forecast_date=target_date,
                model="NBM-Herbie",
                source="herbie",
                forecast_high_f=max(temps),
                forecast_low_f=min(temps),
                run_time=model_run.strftime("%Y-%m-%d %H:%M:%S"),
                fetch_time_utc=now_utc,
            ))

    log.info("NBM-Herbie: %d forecasts for %s (run %s)", len(forecasts), target_date, run_time_str)
    return forecasts
