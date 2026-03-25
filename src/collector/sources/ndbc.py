"""NDBC buoy client — sea surface temperature from C-MAN stations."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import aiohttp

from collector.config import Config
from collector.types import SSTObservation

log = logging.getLogger(__name__)

# NDBC C-MAN stations near Miami for SST
# Format: (station_id, name, lat, lon, distance_mi_from_kmia)
NDBC_STATIONS = [
    ("41122", "Hollywood Beach", 26.001, -80.096, 18.0),
    ("VAKF1", "Virginia Key", 25.731, -80.162, 8.5),
    ("MNBF1", "Manatee Bay", 25.239, -80.422, 25.0),
]


def _c_to_f(c: float) -> float:
    return c * 9.0 / 5.0 + 32.0


class NDBCClient:
    """Fetch real-time sea surface temperature from NDBC buoys."""

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
        assert self._session, "NDBCClient not opened"
        return self._session

    async def get_latest_sst(self) -> list[SSTObservation]:
        """Fetch latest observations from all configured NDBC stations."""
        results: list[SSTObservation] = []
        for station_id, name, lat, lon, dist in NDBC_STATIONS:
            obs = await self._fetch_station(station_id, name, dist)
            if obs:
                results.append(obs)
        return results

    async def _fetch_station(
        self, station_id: str, name: str, distance_mi: float
    ) -> SSTObservation | None:
        """Parse NDBC realtime2 text file for latest observation.

        Format: fixed-width columns, first data row after header rows.
        Columns: #YY MM DD hh mm WDIR WSPD GST WVHT DPD APD MWD PRES ATMP WTMP ...
        Missing values are 99.0 or 999.0.
        """
        url = f"https://www.ndbc.noaa.gov/data/realtime2/{station_id}.txt"
        try:
            async with self.session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status != 200:
                    log.warning("NDBC %s: HTTP %d", station_id, resp.status)
                    return None
                text = await resp.text()
        except Exception:
            log.exception("NDBC %s fetch error", station_id)
            return None

        lines = text.strip().split("\n")
        if len(lines) < 3:
            return None

        # Line 0: header names, Line 1: units, Line 2+: data (most recent first)
        headers = lines[0].replace("#", "").split()
        data_line = lines[2].split()

        if len(data_line) < len(headers):
            return None

        row = dict(zip(headers, data_line))

        # Parse timestamp
        try:
            yr = int(row.get("YY", 0))
            mo = int(row.get("MM", 0))
            dd = int(row.get("DD", 0))
            hh = int(row.get("hh", 0))
            mm = int(row.get("mm", 0))
            ts = datetime(yr, mo, dd, hh, mm, tzinfo=timezone.utc)
            ts_str = ts.strftime("%Y-%m-%dT%H:%M:%S+00:00")
        except (ValueError, KeyError):
            log.warning("NDBC %s: bad timestamp", station_id)
            return None

        def _val(key: str, missing: float = 999.0) -> float | None:
            try:
                v = float(row.get(key, ""))
                return None if v >= missing else v
            except (ValueError, TypeError):
                return None

        water_temp_c = _val("WTMP")
        air_temp_c = _val("ATMP")
        wind_speed = _val("WSPD")
        wind_dir = _val("WDIR")
        pressure = _val("PRES", missing=9999.0)

        if water_temp_c is None:
            return None

        return SSTObservation(
            station_id=station_id,
            name=name,
            timestamp_utc=ts_str,
            water_temp_c=water_temp_c,
            water_temp_f=_c_to_f(water_temp_c),
            air_temp_c=air_temp_c,
            wind_speed_mps=wind_speed,
            wind_dir_deg=wind_dir,
            pressure_hpa=pressure,
            distance_mi=distance_mi,
        )
