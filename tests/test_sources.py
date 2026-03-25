"""Tests for source clients — response parsing with fixture data."""

from types import SimpleNamespace

import pytest

from collector.sources.kalshi_rest import _parse_ticker
from collector.sources.nws import _parse_cli_date, _parse_cli_temp
from collector.sources.openmeteo import OpenMeteo
from collector.sources.wethr_stream import _extract_nws_envelope, _parse_observation
from collector.types import SKY_COVER_CODE_TO_PCT


# ---------------------------------------------------------------------------
# CLI Parsing
# ---------------------------------------------------------------------------

CLI_SAMPLE = """
CLIMATE REPORT FOR MIAMI FL

THE MIAMI CLIMATE SUMMARY FOR 03/13/2026

                  TEMPERATURE (F)

                         YESTERDAY

MAXIMUM                       87
MINIMUM                       72

                  PRECIPITATION (INCHES)

YESTERDAY                    0.00
"""


def test_parse_cli_date():
    assert _parse_cli_date(CLI_SAMPLE) == "2026-03-13"


def test_parse_cli_date_word_format():
    text = "CLIMATE SUMMARY FOR MARCH 13 2026\nTEMPERATURE\nMAXIMUM 88"
    assert _parse_cli_date(text) == "2026-03-13"


def test_parse_cli_max_temp():
    assert _parse_cli_temp(CLI_SAMPLE, "MAXIMUM") == 87.0


def test_parse_cli_min_temp():
    assert _parse_cli_temp(CLI_SAMPLE, "MINIMUM") == 72.0


def test_parse_cli_missing_section():
    assert _parse_cli_temp("no temperature section here", "MAXIMUM") is None


# ---------------------------------------------------------------------------
# Kalshi Ticker Parsing
# ---------------------------------------------------------------------------

def test_parse_ticker_high_above():
    result = _parse_ticker("KXHIGHMIA-26MAR15-T82")
    assert result["market_type"] == "high"
    assert result["floor_strike"] == 82.0
    assert result["cap_strike"] is None


def test_parse_ticker_low_below():
    result = _parse_ticker("KXLOWTMIA-26MAR15-B72.5")
    assert result["market_type"] == "low"
    assert result["cap_strike"] == 72.5
    assert result["floor_strike"] is None


def test_parse_ticker_unknown():
    result = _parse_ticker("UNKNOWN-TICKER")
    assert result["market_type"] == ""


# ---------------------------------------------------------------------------
# Open-Meteo request wiring
# ---------------------------------------------------------------------------


class _DummyResponse:
    def __init__(self, status=200, payload=None):
        self.status = status
        self._payload = payload or {"daily": {}, "hourly": {}}

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def json(self):
        return self._payload


class _DummySession:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    def get(self, url, params=None):
        self.calls.append((url, params or {}))
        return _DummyResponse()


@pytest.mark.asyncio
async def test_ecmwf_aifs_uses_model_param_not_models():
    cfg = SimpleNamespace(
        station=SimpleNamespace(lat=25.7959, lon=-80.2870),
        openmeteo=SimpleNamespace(
            base_url="https://customer-api.open-meteo.com/v1/forecast",
            api_key="test-key",
        ),
    )
    client = OpenMeteo(cfg)
    session = _DummySession()
    client._session = session

    await client.get_ecmwf_aifs("KMIA", "2026-03-18")

    assert len(session.calls) == 1
    _, params = session.calls[0]
    assert params["model"] == "aifs_single"
    assert "models" not in params


# ---------------------------------------------------------------------------
# Wethr SSE Parsing
# ---------------------------------------------------------------------------

def test_parse_observation():
    payload = {
        "station": "KMIA",
        "observation_time_utc": "2026-03-14T15:30:00Z",
        "temperature_f": 84.2,
        "dew_point_f": 72.1,
        "wind_speed_mph": 12.0,
        "wind_direction": "SE",
        "wind_heading_deg": 135,
        "sky_cover_code": "SCT",
        "pressure_hpa": 1015.2,
        "wethr_high": {"nws": {"value_f": 85}},
        "wethr_low": {"nws": {"value_f": 71}},
    }
    obs = _parse_observation("KMIA", payload, utc_offset=-5)

    assert obs.station == "KMIA"
    assert obs.temperature_f == 84.2
    assert obs.dew_point_f == 72.1
    assert obs.wind_speed_mph == 12.0
    assert obs.sky_cover_pct == 50.0  # SCT
    assert obs.wethr_high_nws_f == 85.0
    assert obs.wethr_low_nws_f == 71.0
    assert obs.lst_date == "2026-03-14"
    assert obs.source == "wethr_1min"


def test_parse_observation_alternative_fields():
    payload = {
        "station": "KMIA",
        "timestamp_utc": "2026-03-14T15:30:00Z",
        "temperature_fahrenheit": 84.2,
        "dewpoint_f": 72.1,
        "wind_mph": 12.0,
        "gust_mph": 18.0,
        "wind_cardinal": "SE",
    }
    obs = _parse_observation("KMIA", payload, utc_offset=-5)

    assert obs.temperature_f == 84.2
    assert obs.dew_point_f == 72.1
    assert obs.wind_speed_mph == 12.0
    assert obs.wind_gust_mph == 18.0


def test_extract_nws_envelope():
    payload = {"wethr_high": {"nws": {"value_f": 88}}}
    assert _extract_nws_envelope(payload, "wethr_high") == 88.0


def test_extract_nws_envelope_missing():
    assert _extract_nws_envelope({}, "wethr_high") is None
    assert _extract_nws_envelope({"wethr_high": {}}, "wethr_high") is None


# ---------------------------------------------------------------------------
# Sky Cover Mapping
# ---------------------------------------------------------------------------

def test_sky_cover_codes():
    assert SKY_COVER_CODE_TO_PCT["CLR"] == 0.0
    assert SKY_COVER_CODE_TO_PCT["FEW"] == 25.0
    assert SKY_COVER_CODE_TO_PCT["SCT"] == 50.0
    assert SKY_COVER_CODE_TO_PCT["BKN"] == 75.0
    assert SKY_COVER_CODE_TO_PCT["OVC"] == 100.0
    assert SKY_COVER_CODE_TO_PCT["VV"] == 100.0
