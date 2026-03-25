"""Open-Meteo client — deterministic, ensemble, and pressure level forecasts."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import aiohttp

from collector.config import Config
from collector.types import AtmosphericData, ModelForecast, PressureLevelData

log = logging.getLogger(__name__)

# (display_name, api_param)
DETERMINISTIC_MODELS: list[tuple[str, str]] = [
    ("GFS-GraphCast", "gfs_graphcast025"),
    ("GFS-HRRR", "gfs_hrrr"),
    ("GFS-Global", "gfs_seamless"),
    ("ECMWF-IFS", "ecmwf_ifs025"),
    # ECMWF-AIFS queried separately via /v1/ecmwf endpoint (see get_ecmwf_aifs)
    ("ICON-Global", "icon_seamless"),
    ("GEM-Global", "gem_seamless"),
    ("JMA-GSM", "jma_seamless"),
    ("UKMO-Global", "ukmo_seamless"),
    ("MetNo-Nordic", "metno_seamless"),
    ("KNMI-Harmonie", "knmi_seamless"),
]

# (display_name, api_param, member_count)
ENSEMBLE_MODELS: list[tuple[str, str, int]] = [
    ("GFS-Ensemble", "gfs025", 31),
    ("ECMWF-IFS-Ensemble", "ecmwf_ifs025", 51),
]

PRESSURE_LEVELS = [925, 850, 700, 500]
PRESSURE_VARS = (
    [f"wind_speed_{l}hPa" for l in PRESSURE_LEVELS]
    + [f"wind_direction_{l}hPa" for l in PRESSURE_LEVELS]
    + [f"temperature_{l}hPa" for l in PRESSURE_LEVELS]
    + [f"geopotential_height_{l}hPa" for l in [925, 850, 700]]
    + [f"relative_humidity_{l}hPa" for l in [850, 700]]
)

# Atmospheric parameters for the hourly atmospheric data call.
# CAPE, PW, radiation, BL height, soil, and lifted index have been migrated
# to free sources (HRRR via Herbie, GOES-19, FAWN). Only precipitation
# fields remain — these are not easily replaced from free model data.
ATMOSPHERIC_VARS = [
    "precipitation",
    "rain",
    "showers",
    "precipitation_probability",
    # MIGRATED TO FREE SOURCES (2026-03-22):
    # "shortwave_radiation",     → HRRR DSWRF + GOES-19 DSRC
    # "direct_radiation",        → not used in inference
    # "diffuse_radiation",       → not used in inference
    # "cape",                    → HRRR CAPE + GOES-19 DSI
    # "lifted_index",            → GOES-19 DSI
    # "boundary_layer_height",   → HRRR HPBL
    # "total_column_integrated_water_vapour",  → HRRR PWAT
    # "soil_temperature_0cm",    → FAWN soil temp
    # "soil_moisture_0_to_1cm",  → FAWN soil moisture
]


# Model run cycles (UTC hours when each model initializes).
# Used to infer run_time from fetch_time when the API doesn't return it.
MODEL_RUN_CYCLES: dict[str, list[int]] = {
    "GFS-GraphCast": [0, 6, 12, 18],
    "GFS-HRRR": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],  # hourly
    "GFS-Global": [0, 6, 12, 18],
    "ECMWF-IFS": [0, 6, 12, 18],
    "ECMWF-AIFS": [0, 6, 12, 18],
    "ICON-Global": [0, 6, 12, 18],
    "GEM-Global": [0, 6, 12, 18],
    "JMA-GSM": [0, 12],
    "UKMO-Global": [0, 6, 12, 18],
    "MetNo-Nordic": [0, 6, 12, 18],
    "KNMI-Harmonie": [0, 6, 12, 18],
    "GFS-Ensemble": [0, 6, 12, 18],
    "ECMWF-IFS-Ensemble": [0, 6, 12, 18],
}

# Model data availability lag (hours after run_time before data is available).
# E.g., GFS 06Z run typically available by ~10Z (4h lag).
MODEL_AVAILABILITY_LAG: dict[str, float] = {
    "GFS-GraphCast": 6.0,
    "GFS-HRRR": 1.5,
    "GFS-Global": 4.5,
    "ECMWF-IFS": 5.0,
    "ECMWF-AIFS": 5.0,
    "ICON-Global": 4.0,
    "GEM-Global": 5.0,
    "JMA-GSM": 5.0,
    "UKMO-Global": 5.0,
    "MetNo-Nordic": 3.0,
    "KNMI-Harmonie": 3.0,
    "GFS-Ensemble": 5.0,
    "ECMWF-IFS-Ensemble": 6.0,
}


def _infer_run_time(model: str, fetch_utc: datetime) -> str | None:
    """Infer the most recent model run_time from the fetch timestamp.

    Each model runs at fixed UTC hours. The data available at fetch time
    corresponds to the most recent run that has had enough time to process
    (accounting for the availability lag).
    """
    cycles = MODEL_RUN_CYCLES.get(model)
    if not cycles:
        return None

    lag = MODEL_AVAILABILITY_LAG.get(model, 4.0)
    # The latest run whose data would be available by fetch_utc
    available_at = fetch_utc.hour + fetch_utc.minute / 60.0

    best_run = None
    best_day_offset = None
    for cycle_hour in sorted(cycles, reverse=True):
        # Run at cycle_hour, available at cycle_hour + lag
        ready_at = cycle_hour + lag
        if ready_at <= available_at:
            best_run = cycle_hour
            best_day_offset = 0
            break
    else:
        # Must be from previous day's last run
        best_run = max(cycles)
        best_day_offset = -1

    from datetime import timedelta
    run_date = fetch_utc.date() + timedelta(days=best_day_offset)
    return f"{run_date} {best_run:02d}:00:00"


class OpenMeteo:
    """Open-Meteo forecast client."""

    def __init__(self, cfg: Config):
        self._cfg = cfg
        self._session: aiohttp.ClientSession | None = None

    async def open(self) -> None:
        self._session = aiohttp.ClientSession()

    async def close(self) -> None:
        if self._session:
            await self._session.close()
            self._session = None

    @property
    def session(self) -> aiohttp.ClientSession:
        assert self._session, "OpenMeteo not opened"
        return self._session

    async def get_deterministic(
        self, station: str, date_str: str
    ) -> list[ModelForecast]:
        """Fetch deterministic model forecasts (daily high/low + hourly temps)."""
        lat = self._cfg.station.lat
        lon = self._cfg.station.lon
        models_param = ",".join(m[1] for m in DETERMINISTIC_MODELS)
        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": "temperature_2m_max,temperature_2m_min",
            "hourly": "temperature_2m",
            "temperature_unit": "fahrenheit",
            "models": models_param,
            "start_date": date_str,
            "end_date": date_str,
            "timezone": "auto",
            "apikey": self._cfg.openmeteo.api_key,
        }

        forecasts: list[ModelForecast] = []
        fetch_dt = datetime.now(timezone.utc)
        now_utc = fetch_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        try:
            async with self.session.get(
                self._cfg.openmeteo.base_url, params=params
            ) as resp:
                if resp.status != 200:
                    log.warning("OpenMeteo deterministic: HTTP %d", resp.status)
                    return forecasts
                data = await resp.json()
        except Exception:
            log.exception("OpenMeteo deterministic fetch error")
            return forecasts

        # Parse daily high/low per model
        daily = data.get("daily", {})
        for display_name, api_param in DETERMINISTIC_MODELS:
            high_key = f"temperature_2m_max_{api_param}"
            low_key = f"temperature_2m_min_{api_param}"
            highs = daily.get(high_key, daily.get("temperature_2m_max", []))
            lows = daily.get(low_key, daily.get("temperature_2m_min", []))

            run_time = _infer_run_time(display_name, fetch_dt)

            if highs or lows:
                forecasts.append(ModelForecast(
                    station=station,
                    forecast_date=date_str,
                    model=display_name,
                    source="openmeteo",
                    forecast_high_f=highs[0] if highs else None,
                    forecast_low_f=lows[0] if lows else None,
                    run_time=run_time,
                    fetch_time_utc=now_utc,
                ))

        # Parse hourly temps per model
        hourly = data.get("hourly", {})
        times = hourly.get("time", [])
        for display_name, api_param in DETERMINISTIC_MODELS:
            temp_key = f"temperature_2m_{api_param}"
            temps = hourly.get(temp_key, [])
            run_time = _infer_run_time(display_name, fetch_dt)
            for i, (t, val) in enumerate(zip(times, temps)):
                if val is not None:
                    forecasts.append(ModelForecast(
                        station=station,
                        forecast_date=date_str,
                        model=display_name,
                        source="openmeteo",
                        valid_time=t,
                        raw_temperature_f=val,
                        run_time=run_time,
                        fetch_time_utc=now_utc,
                    ))

        return forecasts

    async def get_ecmwf_aifs(
        self, station: str, date_str: str
    ) -> list[ModelForecast]:
        """Fetch ECMWF-AIFS forecasts via the dedicated ECMWF endpoint."""
        lat = self._cfg.station.lat
        lon = self._cfg.station.lon
        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        display_name = "ECMWF-AIFS"

        # ECMWF endpoint: /v1/ecmwf with model=aifs_single
        ecmwf_url = self._cfg.openmeteo.base_url.replace("/v1/forecast", "/v1/ecmwf")
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": "temperature_2m_max,temperature_2m_min",
            "hourly": "temperature_2m",
            "temperature_unit": "fahrenheit",
            "model": "aifs_single",
            "start_date": date_str,
            "end_date": date_str,
            "timezone": "auto",
            "apikey": self._cfg.openmeteo.api_key,
        }

        forecasts: list[ModelForecast] = []
        fetch_dt = datetime.now(timezone.utc)
        now_utc = fetch_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        run_time = _infer_run_time(display_name, fetch_dt)

        try:
            async with self.session.get(ecmwf_url, params=params) as resp:
                if resp.status != 200:
                    log.warning("OpenMeteo ECMWF-AIFS: HTTP %d", resp.status)
                    return forecasts
                data = await resp.json()
        except Exception:
            log.exception("OpenMeteo ECMWF-AIFS fetch error")
            return forecasts

        # Parse daily
        daily = data.get("daily", {})
        highs = daily.get("temperature_2m_max", [])
        lows = daily.get("temperature_2m_min", [])
        if highs or lows:
            forecasts.append(ModelForecast(
                station=station,
                forecast_date=date_str,
                model=display_name,
                source="openmeteo",
                forecast_high_f=highs[0] if highs else None,
                forecast_low_f=lows[0] if lows else None,
                run_time=run_time,
                fetch_time_utc=now_utc,
            ))

        # Parse hourly
        hourly = data.get("hourly", {})
        times = hourly.get("time", [])
        temps = hourly.get("temperature_2m", [])
        for t, val in zip(times, temps):
            if val is not None:
                forecasts.append(ModelForecast(
                    station=station,
                    forecast_date=date_str,
                    model=display_name,
                    source="openmeteo",
                    valid_time=t,
                    raw_temperature_f=val,
                    run_time=run_time,
                    fetch_time_utc=now_utc,
                ))

        return forecasts

    async def get_ensemble(
        self, station: str, date_str: str
    ) -> list[ModelForecast]:
        """Fetch ensemble forecasts, return one aggregate row per ensemble family.

        Instead of storing individual members (31 GFS + 51 ECMWF = 82 rows per
        fetch), computes aggregate stats (mean, std, percentiles) and stores a
        single row per ensemble with the stats in source_record_json.
        """
        import statistics

        lat = self._cfg.station.lat
        lon = self._cfg.station.lon
        fetch_dt = datetime.now(timezone.utc)
        now_utc = fetch_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        forecasts: list[ModelForecast] = []

        for display_name, api_param, member_count in ENSEMBLE_MODELS:
            run_time = _infer_run_time(display_name, fetch_dt)

            params = {
                "latitude": lat,
                "longitude": lon,
                "daily": "temperature_2m_max,temperature_2m_min",
                "temperature_unit": "fahrenheit",
                "models": api_param,
                "start_date": date_str,
                "end_date": date_str,
                "timezone": "auto",
            }

            try:
                async with self.session.get(
                    self._cfg.openmeteo.ensemble_url, params=params
                ) as resp:
                    if resp.status != 200:
                        log.warning("OpenMeteo ensemble %s: HTTP %d", display_name, resp.status)
                        continue
                    data = await resp.json()
            except Exception:
                log.exception("OpenMeteo ensemble %s fetch error", display_name)
                continue

            daily = data.get("daily", {})

            # Collect all member values
            member_highs: list[float] = []
            member_lows: list[float] = []
            for member_idx in range(member_count):
                member_suffix = f"member{member_idx + 1:02d}"
                highs = daily.get(f"temperature_2m_max_{member_suffix}", [])
                lows = daily.get(f"temperature_2m_min_{member_suffix}", [])
                if highs and highs[0] is not None:
                    member_highs.append(highs[0])
                if lows and lows[0] is not None:
                    member_lows.append(lows[0])

            if not member_highs and not member_lows:
                continue

            def _percentile(vals: list[float], p: float) -> float:
                """Compute percentile using sorted interpolation."""
                s = sorted(vals)
                k = (len(s) - 1) * p / 100
                f_idx = int(k)
                c_idx = min(f_idx + 1, len(s) - 1)
                frac = k - f_idx
                return round(s[f_idx] + frac * (s[c_idx] - s[f_idx]), 2)

            def _stats(vals: list[float]) -> dict:
                if not vals:
                    return {}
                return {
                    "mean": round(statistics.mean(vals), 2),
                    "std": round(statistics.stdev(vals), 2) if len(vals) > 1 else 0.0,
                    "min": round(min(vals), 2),
                    "max": round(max(vals), 2),
                    "p10": _percentile(vals, 10),
                    "p25": _percentile(vals, 25),
                    "p50": _percentile(vals, 50),
                    "p75": _percentile(vals, 75),
                    "p90": _percentile(vals, 90),
                    "n_members": len(vals),
                }

            ensemble_stats = {
                "high": _stats(member_highs),
                "low": _stats(member_lows),
            }

            forecasts.append(ModelForecast(
                station=station,
                forecast_date=date_str,
                model=display_name,
                source="openmeteo",
                forecast_high_f=ensemble_stats["high"].get("mean"),
                forecast_low_f=ensemble_stats["low"].get("mean"),
                run_time=run_time,
                fetch_time_utc=now_utc,
                source_record_json=json.dumps({"ensemble_stats": ensemble_stats}),
            ))

        return forecasts

    async def get_pressure_levels(
        self, station: str
    ) -> list[PressureLevelData]:
        """Fetch GFS pressure level data (925/850/700/500 hPa)."""
        lat = self._cfg.station.lat
        lon = self._cfg.station.lon
        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": ",".join(PRESSURE_VARS),
            "models": "gfs_seamless",
            "timezone": "auto",
            "apikey": self._cfg.openmeteo.api_key,
        }

        levels: list[PressureLevelData] = []
        try:
            async with self.session.get(
                self._cfg.openmeteo.base_url, params=params
            ) as resp:
                if resp.status != 200:
                    log.warning("OpenMeteo pressure levels: HTTP %d", resp.status)
                    return levels
                data = await resp.json()
        except Exception:
            log.exception("OpenMeteo pressure levels fetch error")
            return levels

        hourly = data.get("hourly", {})
        times = hourly.get("time", [])

        for i, t in enumerate(times):
            def _get(var: str) -> float | None:
                vals = hourly.get(var, [])
                return vals[i] if i < len(vals) and vals[i] is not None else None

            levels.append(PressureLevelData(
                station=station,
                valid_time_utc=t,
                model="gfs_seamless",
                fetch_time_utc=now_utc,
                temp_925_c=_get("temperature_925hPa"),
                wind_speed_925=_get("wind_speed_925hPa"),
                wind_dir_925=_get("wind_direction_925hPa"),
                geopotential_925=_get("geopotential_height_925hPa"),
                temp_850_c=_get("temperature_850hPa"),
                wind_speed_850=_get("wind_speed_850hPa"),
                wind_dir_850=_get("wind_direction_850hPa"),
                geopotential_850=_get("geopotential_height_850hPa"),
                rh_850=_get("relative_humidity_850hPa"),
                rh_700=_get("relative_humidity_700hPa"),
                temp_700_c=_get("temperature_700hPa"),
                wind_speed_700=_get("wind_speed_700hPa"),
                wind_dir_700=_get("wind_direction_700hPa"),
                geopotential_700=_get("geopotential_height_700hPa"),
                temp_500_c=_get("temperature_500hPa"),
                wind_speed_500=_get("wind_speed_500hPa"),
                wind_dir_500=_get("wind_direction_500hPa"),
            ))

        return levels

    async def get_atmospheric(
        self, station: str
    ) -> list[AtmosphericData]:
        """Fetch atmospheric parameters (radiation, CAPE, BL height, PW, soil temp)."""
        lat = self._cfg.station.lat
        lon = self._cfg.station.lon
        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": ",".join(ATMOSPHERIC_VARS),
            "models": "best_match",
            "timezone": "auto",
            "apikey": self._cfg.openmeteo.api_key,
        }

        results: list[AtmosphericData] = []
        try:
            async with self.session.get(
                self._cfg.openmeteo.base_url, params=params
            ) as resp:
                if resp.status != 200:
                    log.warning("OpenMeteo atmospheric: HTTP %d", resp.status)
                    return results
                data = await resp.json()
        except Exception:
            log.exception("OpenMeteo atmospheric fetch error")
            return results

        hourly = data.get("hourly", {})
        times = hourly.get("time", [])

        for i, t in enumerate(times):
            def _get(var: str, _i=i) -> float | None:
                vals = hourly.get(var, [])
                return vals[_i] if _i < len(vals) and vals[_i] is not None else None

            results.append(AtmosphericData(
                station=station,
                valid_time_utc=t,
                model="best_match",
                fetch_time_utc=now_utc,
                shortwave_radiation=_get("shortwave_radiation"),
                direct_radiation=_get("direct_radiation"),
                diffuse_radiation=_get("diffuse_radiation"),
                cape=_get("cape"),
                lifted_index=_get("lifted_index"),
                boundary_layer_height=_get("boundary_layer_height"),
                precipitable_water_mm=_get("total_column_integrated_water_vapour"),
                soil_temperature_0_7cm=_get("soil_temperature_0cm"),
                soil_moisture_0_1cm=_get("soil_moisture_0_to_1cm"),
                precipitation_mm=_get("precipitation"),
                rain_mm=_get("rain"),
                showers_mm=_get("showers"),
                precipitation_probability=_get("precipitation_probability"),
            ))

        return results
