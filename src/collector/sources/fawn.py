"""FAWN (Florida Automated Weather Network) client — 15-min actual sensor data.

No authentication required. Supports multiple FAWN stations within a configurable
radius of the trading station. Provides actual sensor readings for solar radiation,
precipitation, and soil temperature — ground truth vs model forecast estimates.

Endpoints used:
  - /controller.php/today/obs/{station_id};json  — today's 15-min obs (per station)
  - /data/complete.json                          — all stations with metadata + latest obs
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import aiohttp

from collector.config import Config
from collector.types import FAWNObservation

log = logging.getLogger(__name__)

BASE_URL = "https://fawn.ifas.ufl.edu"

# Homestead — only FAWN station in Dade County, ~30km S of KMIA
DEFAULT_STATION_ID = "440"
DEFAULT_STATION_NAME = "Homestead"


def _to_float(val) -> float | None:
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _c_to_f(celsius: float | None) -> float | None:
    if celsius is None:
        return None
    return round(celsius * 9.0 / 5.0 + 32.0, 1)


def _kmh_to_mph(kmh: float | None) -> float | None:
    if kmh is None:
        return None
    return round(kmh * 0.621371, 1)


def _in_to_mm(inches: float | None) -> float | None:
    if inches is None:
        return None
    return round(inches * 25.4, 2)


# FAWN stations within ~200km of KMIA (South/Central Florida)
# Full list: 47 stations across Florida. These are the ones within spatial
# correlation range for KMIA trading. IDs from https://fawn.ifas.ufl.edu/
SOUTH_FLORIDA_FAWN_STATIONS: dict[str, str] = {
    "440": "Homestead",         # ~35km S
    "420": "Ft. Lauderdale",    # ~33km N
    "425": "Wellington",        # ~100km N
    "410": "Belle Glade",       # ~100km NW
    "450": "Immokalee",         # ~135km NW
    "405": "Clewiston",         # ~125km NW
    "430": "Ft. Pierce",        # ~183km N
    "435": "St. Lucie West",    # ~175km N
    "460": "Palmdale",          # ~150km NW
    "455": "Okeechobee",        # ~175km NW
}


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance in km between two (lat, lon) points."""
    import math
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


class FAWNClient:
    """FAWN client for 15-min actual sensor observations.

    Supports multiple FAWN stations. When station_ids is provided (list),
    fetches from all of them. Otherwise falls back to single station_id
    from config.
    """

    def __init__(self, cfg: Config, station_ids: list[str] | None = None):
        self._cfg = cfg
        if station_ids is not None:
            self._station_ids = station_ids
        else:
            self._station_ids = [cfg.fawn.station_id]
        self._station_name = cfg.fawn.station_name
        self._station_id = cfg.fawn.station_id  # primary station for backward compat
        self._session: aiohttp.ClientSession | None = None

    async def open(self) -> None:
        self._session = aiohttp.ClientSession()

    async def close(self) -> None:
        if self._session:
            await self._session.close()
            self._session = None

    @property
    def session(self) -> aiohttp.ClientSession:
        assert self._session, "FAWNClient not opened"
        return self._session

    async def get_today_obs(self) -> list[FAWNObservation]:
        """Fetch today's 15-min observations from Homestead station."""
        url = f"{BASE_URL}/controller.php/today/obs/{self._station_id};json"

        try:
            async with self.session.get(url) as resp:
                if resp.status != 200:
                    log.warning("FAWN today obs: HTTP %d", resp.status)
                    return []
                data = await resp.json(content_type=None)
        except Exception:
            log.exception("FAWN today obs fetch error")
            return []

        if not isinstance(data, list):
            log.warning("FAWN today obs: unexpected format")
            return []

        results: list[FAWNObservation] = []
        for rec in data:
            end_time = rec.get("endTime", "")
            if not end_time:
                continue

            # Convert local timestamp to UTC
            ts_utc = self._to_utc(end_time)
            if not ts_utc:
                continue

            # Parse all fields — temps in C, convert to F; wind in km/h, convert to mph
            t2m_c = _to_float(rec.get("t2m"))
            dp_c = _to_float(rec.get("dp"))
            tsoil_c = _to_float(rec.get("tsoil"))
            ws_kmh = _to_float(rec.get("ws"))
            wsmax_kmh = _to_float(rec.get("wsmax"))
            rain_in = _to_float(rec.get("rain"))

            results.append(FAWNObservation(
                station_id=self._station_id,
                station_name=self._station_name,
                timestamp_utc=ts_utc,
                air_temp_f=_c_to_f(t2m_c),
                air_temp_c=round(t2m_c, 2) if t2m_c is not None else None,
                dew_point_f=_c_to_f(dp_c),
                relative_humidity=_to_float(rec.get("rh")),
                wind_speed_mph=_kmh_to_mph(ws_kmh),
                wind_gust_mph=_kmh_to_mph(wsmax_kmh),
                wind_direction_deg=_to_float(rec.get("wdir")),
                solar_radiation_wm2=_to_float(rec.get("rfd")),
                soil_temp_c=round(tsoil_c, 2) if tsoil_c is not None else None,
                soil_temp_f=_c_to_f(tsoil_c),
                rain_mm=_in_to_mm(rain_in),
                rain_in=round(rain_in, 3) if rain_in is not None else None,
            ))

        return results

    async def get_all_stations_obs(self) -> list[FAWNObservation]:
        """Fetch today's 15-min observations from ALL configured stations.

        Returns observations from every station in self._station_ids.
        Failed stations are silently skipped (logged at WARNING).
        """
        all_obs: list[FAWNObservation] = []
        for sid in self._station_ids:
            sname = SOUTH_FLORIDA_FAWN_STATIONS.get(sid, f"FAWN-{sid}")
            url = f"{BASE_URL}/controller.php/today/obs/{sid};json"
            try:
                async with self.session.get(url) as resp:
                    if resp.status != 200:
                        log.warning("FAWN station %s: HTTP %d", sid, resp.status)
                        continue
                    data = await resp.json(content_type=None)
            except Exception:
                log.exception("FAWN station %s fetch error", sid)
                continue

            if not isinstance(data, list):
                continue

            for rec in data:
                end_time = rec.get("endTime", "")
                if not end_time:
                    continue
                ts_utc = self._to_utc(end_time)
                if not ts_utc:
                    continue

                t2m_c = _to_float(rec.get("t2m"))
                dp_c = _to_float(rec.get("dp"))
                tsoil_c = _to_float(rec.get("tsoil"))
                ws_kmh = _to_float(rec.get("ws"))
                wsmax_kmh = _to_float(rec.get("wsmax"))
                rain_in = _to_float(rec.get("rain"))

                all_obs.append(FAWNObservation(
                    station_id=sid,
                    station_name=sname,
                    timestamp_utc=ts_utc,
                    air_temp_f=_c_to_f(t2m_c),
                    air_temp_c=round(t2m_c, 2) if t2m_c is not None else None,
                    dew_point_f=_c_to_f(dp_c),
                    relative_humidity=_to_float(rec.get("rh")),
                    wind_speed_mph=_kmh_to_mph(ws_kmh),
                    wind_gust_mph=_kmh_to_mph(wsmax_kmh),
                    wind_direction_deg=_to_float(rec.get("wdir")),
                    solar_radiation_wm2=_to_float(rec.get("rfd")),
                    soil_temp_c=round(tsoil_c, 2) if tsoil_c is not None else None,
                    soil_temp_f=_c_to_f(tsoil_c),
                    rain_mm=_in_to_mm(rain_in),
                    rain_in=round(rain_in, 3) if rain_in is not None else None,
                ))

        return all_obs

    @staticmethod
    def _to_utc(local_iso: str) -> str | None:
        """Convert FAWN local ISO timestamp (with offset) to UTC ISO string.
        FAWN returns: '2026-03-16T10:00:00-04:00'
        """
        try:
            dt = datetime.fromisoformat(local_iso)
            dt_utc = dt.astimezone(timezone.utc)
            return dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
        except (ValueError, TypeError):
            return None
