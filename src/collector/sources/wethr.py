"""Wethr REST client — model forecasts + observations."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone

import aiohttp

from collector.config import Config
from collector.types import ModelForecast, Observation

log = logging.getLogger(__name__)


def _to_float(val) -> float | None:
    """Flexible float extraction from various payload formats."""
    if val is None:
        return None
    if isinstance(val, dict):
        val = val.get("value")
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _first_float(d: dict, *keys: str) -> float | None:
    for k in keys:
        v = _to_float(d.get(k))
        if v is not None:
            return v
    return None


class WethrREST:
    """Wethr REST API client for forecasts and observations."""

    def __init__(self, cfg: Config):
        self._cfg = cfg
        self._base = cfg.wethr.rest_base
        self._headers = {
            "Authorization": f"Bearer {cfg.wethr.api_key}",
            "Accept": "application/json",
        }
        self._session: aiohttp.ClientSession | None = None

    async def open(self) -> None:
        self._session = aiohttp.ClientSession(headers=self._headers)

    async def close(self) -> None:
        if self._session:
            await self._session.close()
            self._session = None

    @property
    def session(self) -> aiohttp.ClientSession:
        assert self._session, "WethrREST not opened"
        return self._session

    async def get_forecasts(
        self, station: str, date_str: str
    ) -> list[ModelForecast]:
        """Fetch NWP model forecasts for a given date (YYYY-MM-DD LST)."""
        offset = self._cfg.station.utc_offset_hours
        # Build 24-hour UTC window from LST midnight
        lst_midnight = datetime.strptime(date_str, "%Y-%m-%d").replace(
            tzinfo=timezone(timedelta(hours=offset))
        )
        utc_start = lst_midnight.astimezone(timezone.utc)
        utc_end = utc_start + timedelta(hours=24)

        params = {
            "location_name": station,
            "start_valid_time": utc_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end_valid_time": utc_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }

        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        forecasts: list[ModelForecast] = []

        try:
            async with self.session.get(
                f"{self._base}/forecasts.php", params=params
            ) as resp:
                if resp.status != 200:
                    log.warning("Wethr forecasts %s: HTTP %d", station, resp.status)
                    return forecasts
                data = await resp.json()
        except Exception:
            log.exception("Wethr forecasts fetch error")
            return forecasts

        if not isinstance(data, list):
            data = data.get("data", data.get("forecasts", []))

        # Collect hourly temps per model for daily high/low derivation
        model_hourly: dict[str, list[float]] = {}

        for record in data:
            model = record.get("model", record.get("model_name", "unknown"))
            high = _to_float(record.get("forecast_high_f"))
            low = _to_float(record.get("forecast_low_f"))
            # Wethr API returns hourly point temp as "temperature_f" (or _k/_c)
            raw_temp = _to_float(record.get("raw_temperature_f"))
            if raw_temp is None:
                raw_temp = _first_float(record, "temperature_f", "temperature_fahrenheit")
                if raw_temp is None:
                    temp_k = _to_float(record.get("temperature_k"))
                    if temp_k is not None:
                        raw_temp = round(temp_k * 9 / 5 - 459.67, 2)
            valid_time = record.get("valid_time")
            run_time = record.get("run_time", record.get("model_run_time"))

            # Track hourly temps for daily high/low computation
            if raw_temp is not None:
                model_hourly.setdefault(model, []).append(raw_temp)

            forecasts.append(ModelForecast(
                station=station,
                forecast_date=date_str,
                model=model,
                source="wethr",
                forecast_high_f=high,
                forecast_low_f=low,
                run_time=run_time,
                valid_time=valid_time,
                raw_temperature_f=raw_temp,
                fetch_time_utc=now_utc,
                source_record_json=json.dumps(record),
            ))

        # Derive daily high/low from hourly data per model+run.
        # Group by model+run, filter valid_times to climate day (05Z-05Z),
        # take max as high, min as low. Only need 4+ hours for coverage.
        from datetime import timedelta as _td
        climate_start = f"{date_str}T05:00:00"
        next_day = (datetime.strptime(date_str, "%Y-%m-%d") + _td(days=1)).strftime("%Y-%m-%d")
        climate_end = f"{next_day}T05:00:00"

        model_run_temps: dict[str, list[float]] = {}
        for f in forecasts:
            if f.raw_temperature_f is not None and f.valid_time:
                vt = f.valid_time
                if climate_start <= vt < climate_end:
                    key = f"{f.model}|{f.run_time or 'latest'}"
                    model_run_temps.setdefault(key, []).append(f.raw_temperature_f)

        for key, temps in model_run_temps.items():
            model_name = key.split("|")[0]
            run_time = key.split("|")[1] if "|" in key else None
            if run_time == "latest":
                run_time = None
            if len(temps) >= 4:
                derived_high = round(max(temps), 1)
                derived_low = round(min(temps), 1)
                forecasts.append(ModelForecast(
                    station=station,
                    forecast_date=date_str,
                    model=model_name,
                    source="wethr",
                    forecast_high_f=derived_high,
                    forecast_low_f=derived_low,
                    run_time=run_time,
                    fetch_time_utc=now_utc,
                ))

        return forecasts

    async def get_nws_forecasts(
        self, station: str, date_str: str
    ) -> list[ModelForecast]:
        """Fetch NWS forecast versions for a given date."""
        params = {"station": station, "date": date_str}
        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        forecasts: list[ModelForecast] = []

        try:
            async with self.session.get(
                f"{self._base}/nws_forecasts.php", params=params
            ) as resp:
                if resp.status != 200:
                    return forecasts
                data = await resp.json()
        except Exception:
            log.exception("Wethr NWS forecasts fetch error")
            return forecasts

        if not isinstance(data, list):
            data = data.get("data", data.get("forecasts", []))

        for record in data:
            issued = record.get("issued_at", "")
            high = _to_float(record.get("high_f"))
            low = _to_float(record.get("low_f"))
            forecasts.append(ModelForecast(
                station=station,
                forecast_date=date_str,
                model=f"NWS-{issued[:13]}" if issued else "NWS",
                source="nws",
                forecast_high_f=high,
                forecast_low_f=low,
                run_time=issued,
                fetch_time_utc=now_utc,
                source_record_json=json.dumps(record),
            ))

        return forecasts

    async def get_observations(self, station: str) -> list[Observation]:
        """Fetch recent observations via REST (fallback to SSE)."""
        params = {"station": station}
        obs_list: list[Observation] = []

        try:
            async with self.session.get(
                f"{self._base}/observations.php", params=params
            ) as resp:
                if resp.status != 200:
                    return obs_list
                data = await resp.json()
        except Exception:
            log.exception("Wethr observations fetch error")
            return obs_list

        if not isinstance(data, list):
            data = data.get("data", data.get("observations", []))

        offset = self._cfg.station.utc_offset_hours
        for record in data:
            ts = record.get("observation_time_utc", record.get("timestamp_utc", ""))
            if not ts:
                continue

            # Compute LST date
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                lst = dt + timedelta(hours=offset)
                lst_date = lst.strftime("%Y-%m-%d")
            except ValueError:
                lst_date = ""

            obs_list.append(Observation(
                station=station,
                timestamp_utc=ts,
                lst_date=lst_date,
                temperature_f=_first_float(record, "temperature_f", "temperature_fahrenheit", "value_f"),
                dew_point_f=_first_float(record, "dew_point_f", "dewpoint_f"),
                relative_humidity=_to_float(record.get("relative_humidity")),
                wind_speed_mph=_first_float(record, "wind_speed_mph", "wind_mph"),
                wind_direction=record.get("wind_direction", record.get("wind_cardinal")),
                wind_gust_mph=_first_float(record, "wind_gust_mph", "gust_mph"),
                wind_heading_deg=_first_float(record, "wind_heading_deg", "wind_direction_degrees"),
                visibility_miles=_to_float(record.get("visibility_miles")),
                sky_cover_pct=_to_float(record.get("sky_cover_pct")),
                sky_cover_code=record.get("sky_cover_code", record.get("sky_cover")),
                pressure_hpa=_to_float(record.get("pressure_hpa")),
                wethr_high_f=_to_float(record.get("wethr_high_f")),
                wethr_low_f=_to_float(record.get("wethr_low_f")),
                source="wethr_rest",
            ))

        return obs_list
