"""Wethr SSE client — persistent real-time 1-min observations + events."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Callable

import aiohttp

from collector.config import Config
from collector.types import Observation, SKY_COVER_CODE_TO_PCT

log = logging.getLogger(__name__)

EVENT_OBSERVATION = "observation"
EVENT_NEW_HIGH = "new_high"
EVENT_NEW_LOW = "new_low"
EVENT_DSM = "dsm"
EVENT_CLI = "cli"
EVENT_HEARTBEAT = "heartbeat"
EVENT_CONNECTED = "connected"
EVENT_DISPLACED = "displaced"

HEARTBEAT_TIMEOUT = 90  # seconds (heartbeats sent every 30s per Wethr docs)
BACKOFF_INITIAL = 1.0
BACKOFF_MAX = 30.0
BACKOFF_MULTIPLIER = 2.0


def _to_float(val) -> float | None:
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


def _extract_nws_envelope(payload: dict, key: str) -> float | None:
    """Extract NWS-format integer envelope value (e.g. wethr_high.nws.value_f)."""
    envelope = payload.get(key)
    if isinstance(envelope, dict):
        nws = envelope.get("nws")
        if isinstance(nws, dict):
            return _to_float(nws.get("value_f"))
    return None


def _normalize_utc_timestamp(ts: str) -> str:
    """Normalize a UTC timestamp to ISO 8601 format with +00:00 suffix.
    Handles '2026-03-14 16:45:00' → '2026-03-14T16:45:00+00:00'."""
    if not ts:
        return ts
    # Already has timezone info
    if "+" in ts or ts.endswith("Z"):
        return ts.replace("Z", "+00:00")
    # Replace space with T and add UTC offset
    ts = ts.replace(" ", "T")
    if "T" in ts:
        return ts + "+00:00"
    return ts


def _parse_observation(
    station: str, payload: dict, utc_offset: int
) -> Observation:
    """Parse a Wethr SSE observation payload into an Observation."""
    ts = payload.get("observation_time_utc", payload.get("timestamp_utc", ""))
    ts = _normalize_utc_timestamp(ts)

    # Compute LST date
    lst_date = ""
    if ts:
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            lst = dt + timedelta(hours=utc_offset)
            lst_date = lst.strftime("%Y-%m-%d")
        except ValueError:
            pass

    sky_code = payload.get("sky_cover_code", payload.get("sky_cover"))
    sky_pct = _to_float(payload.get("sky_cover_pct"))
    if sky_pct is None and sky_code:
        sky_pct = SKY_COVER_CODE_TO_PCT.get(str(sky_code).upper())

    # NWS-format envelope is what Kalshi uses for settlement
    nws_high = _extract_nws_envelope(payload, "wethr_high")
    nws_low = _extract_nws_envelope(payload, "wethr_low")

    # Populate wethr_high_f/wethr_low_f from NWS envelope (authoritative for Kalshi)
    # Fall back to raw wethr_high_f/wethr_low_f if NWS envelope not present
    high_f = nws_high if nws_high is not None else _to_float(payload.get("wethr_high_f"))
    low_f = nws_low if nws_low is not None else _to_float(payload.get("wethr_low_f"))

    return Observation(
        station=station,
        timestamp_utc=ts,
        lst_date=lst_date,
        temperature_f=_first_float(payload, "temperature_f", "temperature_fahrenheit", "value_f"),
        dew_point_f=_first_float(payload, "dew_point_f", "dewpoint_f"),
        relative_humidity=_to_float(payload.get("relative_humidity")),
        wind_speed_mph=_first_float(payload, "wind_speed_mph", "wind_mph"),
        wind_direction=payload.get("wind_direction", payload.get("wind_cardinal")),
        wind_gust_mph=_first_float(payload, "wind_gust_mph", "gust_mph"),
        wind_heading_deg=_first_float(payload, "wind_heading_deg", "wind_direction_degrees", "wind_direction_deg", "wind_direction"),
        visibility_miles=_to_float(payload.get("visibility_miles")),
        sky_cover_pct=sky_pct,
        sky_cover_code=sky_code,
        pressure_hpa=_to_float(payload.get("pressure_hpa")),
        wethr_high_f=high_f,
        wethr_low_f=low_f,
        wethr_high_nws_f=nws_high,
        wethr_low_nws_f=nws_low,
        source="wethr_1min",
    )


class WethrStream:
    """Persistent SSE connection to Wethr for real-time 1-min observations."""

    def __init__(
        self,
        cfg: Config,
        on_observation: Callable[[Observation], None] | None = None,
        on_event: Callable[[str, str, dict], None] | None = None,
    ):
        self._cfg = cfg
        self._station = cfg.station.code
        self._utc_offset = cfg.station.utc_offset_hours
        self._on_observation = on_observation
        self._on_event = on_event  # (station, event_type, payload)
        self._running = False

    def _build_url(self) -> str:
        return (
            f"{self._cfg.wethr.sse_base}/stream"
            f"?stations={self._station}"
            f"&api_key={self._cfg.wethr.api_key}"
        )

    async def start(self) -> None:
        """Run the SSE listener with auto-reconnect. Runs forever."""
        self._running = True
        backoff = BACKOFF_INITIAL

        while self._running:
            try:
                await self._connect()
                backoff = BACKOFF_INITIAL  # Reset on clean disconnect
            except asyncio.CancelledError:
                self._running = False
                return
            except Exception:
                log.exception("WethrStream connection error, reconnecting in %.1fs", backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * BACKOFF_MULTIPLIER, BACKOFF_MAX)

    def stop(self) -> None:
        self._running = False

    async def _connect(self) -> None:
        url = self._build_url()
        log.info("WethrStream connecting to %s", self._station)

        timeout = aiohttp.ClientTimeout(total=None, sock_read=HEARTBEAT_TIMEOUT)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    log.warning("WethrStream HTTP %d", resp.status)
                    return

                log.info("WethrStream connected for %s", self._station)
                event_type = ""
                data_lines: list[str] = []

                async for line_bytes in resp.content:
                    line = line_bytes.decode("utf-8", errors="replace").rstrip("\n\r")

                    if line.startswith("event:"):
                        event_type = line[6:].strip()
                        data_lines = []
                    elif line.startswith("data:"):
                        data_lines.append(line[5:].strip())
                    elif line == "" and event_type:
                        # End of event
                        raw = "\n".join(data_lines)
                        await self._dispatch(event_type, raw)
                        event_type = ""
                        data_lines = []

    async def _dispatch(self, event_type: str, raw_data: str) -> None:
        if event_type == EVENT_HEARTBEAT:
            log.debug("WethrStream heartbeat received")
            return

        try:
            payload = json.loads(raw_data) if raw_data else {}
        except json.JSONDecodeError:
            log.warning("WethrStream bad JSON for %s: %s", event_type, raw_data[:200])
            return

        station = payload.get("station", self._station)

        if event_type == EVENT_CONNECTED:
            hint = payload.get("refreshHint")
            tier = payload.get("tier")
            log.info("WethrStream connected: tier=%s refreshHint=%s", tier, hint)
            if self._on_event:
                self._on_event(station, event_type, payload)
            return

        if event_type == EVENT_DISPLACED:
            log.warning("WethrStream displaced — another connection opened for this account")
            if self._on_event:
                self._on_event(station, event_type, payload)
            return

        if event_type == EVENT_OBSERVATION:
            obs = _parse_observation(station, payload, self._utc_offset)
            if self._on_observation:
                self._on_observation(obs)
            return

        # All non-observation events get logged once via on_event
        if self._on_event:
            self._on_event(station, event_type, payload)

        if event_type not in (
            EVENT_NEW_HIGH, EVENT_NEW_LOW, EVENT_CLI, EVENT_DSM,
            EVENT_CONNECTED, EVENT_DISPLACED,
        ):
            log.info("WethrStream unknown event: %s", event_type)
