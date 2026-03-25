"""Iowa Environmental Mesonet (IEM) client — 5-min ASOS obs from nearby stations.

No authentication required. Single API call returns latest obs for all
FL ASOS stations; we filter to those within radius of KMIA.

Serves as a free, zero-auth alternative to Synoptic for spatial lead
indicators (temperature/wind systems approaching Miami).

Endpoints used:
  - /api/1/currents.json?network=FL_ASOS  — latest obs for all FL stations
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone

import aiohttp

from collector.config import Config

log = logging.getLogger(__name__)

BASE_URL = "https://mesonet.agron.iastate.edu"


@dataclass
class IEMObservation:
    """A single observation from a nearby ASOS station via IEM."""
    stid: str       # e.g. "FLL" (no K prefix in IEM)
    icao: str       # e.g. "KFLL" (with K prefix)
    name: str
    network: str    # e.g. "FL_ASOS", "FL_RWIS", "FL_DCP"
    latitude: float
    longitude: float
    distance_mi: float
    bearing_deg: float
    elevation_m: float | None
    timestamp_utc: str
    air_temp_f: float | None
    dew_point_f: float | None
    wind_speed_mph: float | None
    wind_direction_deg: float | None
    wind_gust_mph: float | None
    pressure_slp_hpa: float | None
    sky_cover_code: str | None


def _to_float(val) -> float | None:
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _knots_to_mph(knots: float | None) -> float | None:
    if knots is None:
        return None
    return round(knots * 1.15078, 1)


def _haversine_mi(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in miles."""
    R = 3958.8  # Earth radius in miles
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


def _initial_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial bearing (degrees 0-360) from point 1 to point 2."""
    lat1_r, lat2_r = math.radians(lat1), math.radians(lat2)
    dlon_r = math.radians(lon2 - lon1)
    x = math.sin(dlon_r) * math.cos(lat2_r)
    y = (math.cos(lat1_r) * math.sin(lat2_r) -
         math.sin(lat1_r) * math.cos(lat2_r) * math.cos(dlon_r))
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def _local_to_utc(local_str: str, utc_offset: int) -> str:
    """Convert IEM local time string to UTC ISO format.
    IEM returns: '2026-03-14T11:53:00' (local time, no TZ info).
    """
    if not local_str:
        return ""
    try:
        from datetime import timedelta
        dt = datetime.strptime(local_str, "%Y-%m-%dT%H:%M:%S")
        # IEM returns in station's local time; subtract offset to get UTC
        dt_utc = dt - timedelta(hours=utc_offset)
        return dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    except (ValueError, TypeError):
        return local_str


ADDITIONAL_NETWORKS = ["FL_RWIS", "FL_DCP"]


class IEMClient:
    """Iowa Environmental Mesonet client for nearby ASOS station observations."""

    def __init__(self, cfg: Config, radius_mi: float = 35.0):
        self._cfg = cfg
        self._center_lat = cfg.station.lat
        self._center_lon = cfg.station.lon
        self._radius_mi = radius_mi
        self._utc_offset = cfg.station.utc_offset_hours
        self._session: aiohttp.ClientSession | None = None
        self._networks = ["FL_ASOS"] + ADDITIONAL_NETWORKS

    async def open(self) -> None:
        self._session = aiohttp.ClientSession()

    async def close(self) -> None:
        if self._session:
            await self._session.close()
            self._session = None

    @property
    def session(self) -> aiohttp.ClientSession:
        assert self._session, "IEMClient not opened"
        return self._session

    async def get_latest_nearby(self) -> list[IEMObservation]:
        """Fetch latest obs for FL stations across multiple networks, filter to radius.

        Returns list of IEMObservation sorted by distance from KMIA.
        Excludes KMIA itself (we have direct obs for that).
        """
        all_stations: list[dict] = []
        for network in self._networks:
            try:
                async with self.session.get(
                    f"{BASE_URL}/api/1/currents.json",
                    params={"network": network},
                ) as resp:
                    if resp.status != 200:
                        log.warning("IEM currents %s: HTTP %d", network, resp.status)
                        continue
                    data = await resp.json()
                    for s in data.get("data", []):
                        s["_network"] = network
                        all_stations.append(s)
            except Exception:
                log.exception("IEM currents %s fetch error", network)

        if not all_stations:
            return []

        # Deduplicate by station ID (ASOS takes priority)
        seen: set[str] = set()
        results: list[IEMObservation] = []
        for s in all_stations:
            lat = _to_float(s.get("lat"))
            lon = _to_float(s.get("lon"))
            if lat is None or lon is None:
                continue

            dist = _haversine_mi(self._center_lat, self._center_lon, lat, lon)
            bearing = _initial_bearing(self._center_lat, self._center_lon, lat, lon)
            if dist > self._radius_mi:
                continue

            stid = s.get("station", "")

            # Skip KMIA itself — we have direct obs
            if stid == "MIA":
                continue

            # Skip duplicates (first seen wins — ASOS networks come first)
            if stid in seen:
                continue
            seen.add(stid)

            # IEM wind is in knots, convert to mph
            wind_mph = _knots_to_mph(_to_float(s.get("sknt")))
            gust_mph = _knots_to_mph(_to_float(s.get("gust_sknt")))

            # Convert local timestamp to UTC
            ts_utc = _local_to_utc(
                s.get("local_valid", ""), self._utc_offset
            )

            # Build ICAO from IEM station ID
            icao = f"K{stid}" if len(stid) == 3 else stid

            # Sky cover: IEM provides skyc1..skyc4 (cloud layer codes)
            sky_code = s.get("skyc1")
            if sky_code == "M":  # IEM uses "M" for missing
                sky_code = None

            obs = IEMObservation(
                stid=stid,
                icao=icao,
                name=s.get("name", ""),
                network=s.get("_network", "FL_ASOS"),
                latitude=lat,
                longitude=lon,
                distance_mi=round(dist, 1),
                bearing_deg=round(bearing, 1),
                elevation_m=_to_float(s.get("elevation")),
                timestamp_utc=ts_utc,
                air_temp_f=_to_float(s.get("tmpf")),
                dew_point_f=_to_float(s.get("dwpf")),
                wind_speed_mph=wind_mph,
                wind_direction_deg=_to_float(s.get("drct")),
                wind_gust_mph=gust_mph,
                pressure_slp_hpa=_to_float(s.get("mslp")),
                sky_cover_code=sky_code,
            )
            results.append(obs)

        results.sort(key=lambda x: x.distance_mi)
        return results
