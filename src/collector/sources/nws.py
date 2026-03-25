"""NWS API client — observations + CLI report parsing."""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone

import aiohttp

from collector.config import Config
from collector.types import EventSettlement, Observation

log = logging.getLogger(__name__)


def _c_to_f(celsius: float) -> float:
    return celsius * 9.0 / 5.0 + 32.0


def _kmh_to_mph(kmh: float) -> float:
    return kmh * 0.621371


def _m_to_miles(meters: float) -> float:
    return meters / 1609.34


def _nws_val(v) -> float | None:
    """Extract numeric value from NWS API response (may be dict or float)."""
    if v is None:
        return None
    if isinstance(v, dict):
        v = v.get("value")
    if v is None:
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


class NWS:
    """NWS weather.gov API client."""

    def __init__(self, cfg: Config):
        self._cfg = cfg
        self._base = cfg.nws.base_url
        self._headers = {
            "User-Agent": cfg.nws.user_agent,
            "Accept": "application/geo+json",
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
        assert self._session, "NWS not opened"
        return self._session

    async def get_latest_observation(self, station: str) -> Observation | None:
        """Fetch the latest ASOS observation for a station."""
        url = f"{self._base}/stations/{station}/observations/latest"

        try:
            async with self.session.get(url) as resp:
                if resp.status != 200:
                    log.warning("NWS obs %s: HTTP %d", station, resp.status)
                    return None
                data = await resp.json()
        except Exception:
            log.exception("NWS observation fetch error")
            return None

        props = data.get("properties", {})
        ts = props.get("timestamp", "")

        # Convert units
        temp_c = _nws_val(props.get("temperature"))
        temp_f = _c_to_f(temp_c) if temp_c is not None else None

        dp_c = _nws_val(props.get("dewpoint"))
        dp_f = _c_to_f(dp_c) if dp_c is not None else None

        wind_kmh = _nws_val(props.get("windSpeed"))
        wind_mph = _kmh_to_mph(wind_kmh) if wind_kmh is not None else None

        gust_kmh = _nws_val(props.get("windGust"))
        gust_mph = _kmh_to_mph(gust_kmh) if gust_kmh is not None else None

        wind_dir = _nws_val(props.get("windDirection"))

        vis_m = _nws_val(props.get("visibility"))
        vis_mi = _m_to_miles(vis_m) if vis_m is not None else None

        pressure = _nws_val(props.get("barometricPressure"))
        pressure_hpa = pressure / 100.0 if pressure is not None else None  # Pa to hPa

        humidity = _nws_val(props.get("relativeHumidity"))

        # Precipitation (mm, last hour)
        precip_mm = _nws_val(props.get("precipitationLastHour"))

        # Sky cover from cloud layers
        sky_pct = None
        sky_code = None
        cloud_layers = props.get("cloudLayers", [])
        if cloud_layers:
            max_cover = 0.0
            from collector.types import SKY_COVER_CODE_TO_PCT
            for layer in cloud_layers:
                amount = layer.get("amount", "")
                pct = SKY_COVER_CODE_TO_PCT.get(str(amount).upper(), 0.0)
                if pct > max_cover:
                    max_cover = pct
                    sky_code = str(amount).upper()
            sky_pct = max_cover

        # Compute LST date
        offset = self._cfg.station.utc_offset_hours
        lst_date = ""
        if ts:
            try:
                from datetime import timedelta
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                lst = dt + timedelta(hours=offset)
                lst_date = lst.strftime("%Y-%m-%d")
            except ValueError:
                pass

        return Observation(
            station=station,
            timestamp_utc=ts,
            lst_date=lst_date,
            temperature_f=temp_f,
            dew_point_f=dp_f,
            relative_humidity=humidity,
            wind_speed_mph=wind_mph,
            wind_direction=str(int(wind_dir)) if wind_dir is not None else None,
            wind_gust_mph=gust_mph,
            wind_heading_deg=wind_dir,
            visibility_miles=vis_mi,
            sky_cover_pct=sky_pct,
            sky_cover_code=sky_code,
            pressure_hpa=pressure_hpa,
            precipitation_last_hour_mm=precip_mm,
            source="nws",
        )

    async def get_cli_report(
        self, cli_location: str, station: str
    ) -> list[EventSettlement]:
        """Fetch and parse CLI report. Returns settlements for the climate day."""
        url = f"{self._base}/products/types/CLI/locations/{cli_location}"
        settlements: list[EventSettlement] = []

        try:
            async with self.session.get(url) as resp:
                if resp.status != 200:
                    return settlements
                data = await resp.json()
        except Exception:
            log.exception("NWS CLI product list error")
            return settlements

        products = data.get("@graph", [])
        if not products:
            return settlements

        # Fetch the full CLI product text
        product_url = products[0].get("@id", "")
        if not product_url:
            return settlements

        try:
            async with self.session.get(product_url) as resp:
                if resp.status != 200:
                    return settlements
                product = await resp.json()
        except Exception:
            log.exception("NWS CLI product fetch error")
            return settlements

        text = product.get("productText", "")
        if not text:
            return settlements

        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        climate_date = _parse_cli_date(text)
        max_temp = _parse_cli_temp(text, "MAXIMUM")
        min_temp = _parse_cli_temp(text, "MINIMUM")

        if climate_date and max_temp is not None:
            settlements.append(EventSettlement(
                station=station,
                settlement_date=climate_date,
                market_type="high",
                actual_value_f=max_temp,
                settlement_source="cli",
                raw_text=text[:2000],
                received_at=now_utc,
            ))

        if climate_date and min_temp is not None:
            settlements.append(EventSettlement(
                station=station,
                settlement_date=climate_date,
                market_type="low",
                actual_value_f=min_temp,
                settlement_source="cli",
                raw_text=text[:2000],
                received_at=now_utc,
            ))

        return settlements

    async def get_gridpoint_mixing_height(self) -> dict | None:
        """Fetch NWS gridpoint mixing height and transport wind from MFL office.

        Returns dict with lists of {valid_time, value} entries, or None on error.
        """
        office = self._cfg.station.nws_office
        gx = self._cfg.station.nws_grid_x
        gy = self._cfg.station.nws_grid_y
        url = f"{self._base}/gridpoints/{office}/{gx},{gy}"

        try:
            async with self.session.get(url) as resp:
                if resp.status != 200:
                    log.warning("NWS gridpoint: HTTP %d", resp.status)
                    return None
                data = await resp.json()
        except Exception:
            log.exception("NWS gridpoint fetch error")
            return None

        props = data.get("properties", {})
        result = {}

        for field in ("mixingHeight", "transportWindSpeed", "transportWindDirection", "skyCover"):
            values = props.get(field, {}).get("values", [])
            parsed = []
            for entry in values:
                vt = entry.get("validTime", "")
                val = entry.get("value")
                if vt and val is not None:
                    # validTime format: "2026-03-14T12:00:00+00:00/PT1H"
                    ts = vt.split("/")[0]
                    parsed.append({"valid_time": ts, "value": float(val)})
            result[field] = parsed

        return result


def _parse_cli_date(text: str) -> str | None:
    """Extract the climate day from CLI report 'FOR' line.
    e.g. 'CLIMATE REPORT FOR MIAMI FL' with date on next line,
    or 'THE MIAMI CLIMATE SUMMARY FOR 03/13/2026'.
    Returns YYYY-MM-DD or None."""
    # Pattern: "FOR MM/DD/YYYY" or "FOR MONTH DD YYYY"
    m = re.search(r"FOR\s+(\d{1,2})/(\d{1,2})/(\d{4})", text)
    if m:
        month, day, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return f"{year:04d}-{month:02d}-{day:02d}"

    # Try "FOR MARCH 13 2026" style
    months = {
        "JANUARY": 1, "FEBRUARY": 2, "MARCH": 3, "APRIL": 4,
        "MAY": 5, "JUNE": 6, "JULY": 7, "AUGUST": 8,
        "SEPTEMBER": 9, "OCTOBER": 10, "NOVEMBER": 11, "DECEMBER": 12,
    }
    m = re.search(
        r"FOR\s+([A-Z]+)\s+(\d{1,2})[\s,]+(\d{4})", text, re.IGNORECASE
    )
    if m:
        month_name = m.group(1).upper()
        if month_name in months:
            month = months[month_name]
            day = int(m.group(2))
            year = int(m.group(3))
            return f"{year:04d}-{month:02d}-{day:02d}"

    # Fallback: date line like "REPORT FOR  03/13/26"
    m = re.search(r"(\d{1,2})/(\d{1,2})/(\d{2,4})", text[:500])
    if m:
        month, day = int(m.group(1)), int(m.group(2))
        year = int(m.group(3))
        if year < 100:
            year += 2000
        return f"{year:04d}-{month:02d}-{day:02d}"

    return None


def _parse_cli_temp(text: str, keyword: str) -> float | None:
    """Extract temperature from CLI TEMPERATURE section.
    Looks for lines like 'MAXIMUM    88' or 'MINIMUM    72'."""
    # Find the TEMPERATURE section
    temp_section = re.search(r"TEMPERATURE.*?(?=PRECIPITATION|HEATING|$)", text, re.DOTALL)
    if not temp_section:
        return None

    section = temp_section.group(0)
    # Look for the keyword followed by a number
    m = re.search(rf"{keyword}\s+(-?\d+)", section)
    if m:
        return float(m.group(1))
    return None
