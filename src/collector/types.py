"""Shared data types for the collector."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelForecast:
    station: str
    forecast_date: str  # YYYY-MM-DD LST
    model: str
    source: str  # "wethr", "openmeteo", "nws"
    forecast_high_f: float | None = None
    forecast_low_f: float | None = None
    run_time: str | None = None  # Model init time ISO
    valid_time: str | None = None  # Hourly valid time ISO
    raw_temperature_f: float | None = None  # Hourly point forecast
    run_age_hours: float | None = None
    fetch_time_utc: str | None = None
    source_record_json: str | None = None


@dataclass
class Observation:
    station: str
    timestamp_utc: str
    lst_date: str  # YYYY-MM-DD
    temperature_f: float | None = None
    dew_point_f: float | None = None
    relative_humidity: float | None = None
    wind_speed_mph: float | None = None
    wind_direction: str | None = None
    wind_gust_mph: float | None = None
    wind_heading_deg: float | None = None
    visibility_miles: float | None = None
    sky_cover_pct: float | None = None
    sky_cover_code: str | None = None
    pressure_hpa: float | None = None
    precipitation_last_hour_mm: float | None = None  # NWS precip in last hour
    wethr_high_f: float | None = None  # Running high (raw)
    wethr_low_f: float | None = None  # Running low (raw)
    wethr_high_nws_f: float | None = None  # NWS-format envelope integer
    wethr_low_nws_f: float | None = None  # NWS-format envelope integer
    source: str = ""  # "wethr_1min", "nws"


@dataclass
class PressureLevelData:
    station: str
    valid_time_utc: str  # ISO timestamp
    model: str  # e.g. "gfs_seamless"
    fetch_time_utc: str | None = None
    # 925 hPa
    temp_925_c: float | None = None
    wind_speed_925: float | None = None  # m/s
    wind_dir_925: float | None = None  # degrees
    geopotential_925: float | None = None  # meters
    # 850 hPa
    temp_850_c: float | None = None
    wind_speed_850: float | None = None
    wind_dir_850: float | None = None
    geopotential_850: float | None = None
    # 700 hPa
    temp_700_c: float | None = None
    wind_speed_700: float | None = None
    wind_dir_700: float | None = None
    geopotential_700: float | None = None
    # Relative humidity at key levels
    rh_850: float | None = None  # percent
    rh_700: float | None = None  # percent
    # 500 hPa
    temp_500_c: float | None = None
    wind_speed_500: float | None = None
    wind_dir_500: float | None = None


@dataclass
class MarketSnapshot:
    ticker: str
    event_ticker: str
    series_ticker: str
    market_type: str  # "high" or "low"
    forecast_date: str | None = None  # YYYY-MM-DD
    floor_strike: float | None = None
    cap_strike: float | None = None
    best_yes_bid_cents: float | None = None
    best_yes_ask_cents: float | None = None
    best_no_bid_cents: float | None = None
    best_no_ask_cents: float | None = None
    last_price_cents: float | None = None
    volume: int | None = None
    # Liquidity fields
    yes_bid_qty: int | None = None      # Contracts at best yes bid
    yes_ask_qty: int | None = None      # Contracts at best yes ask
    no_bid_qty: int | None = None       # Contracts at best no bid
    no_ask_qty: int | None = None       # Contracts at best no ask
    total_yes_depth: int | None = None  # Total contracts across all yes bid levels
    total_no_depth: int | None = None   # Total contracts across all no bid levels
    spread_cents: int | None = None     # Yes ask - yes bid
    num_yes_levels: int | None = None   # Number of price levels on yes side
    num_no_levels: int | None = None    # Number of price levels on no side
    snapshot_time: str = ""  # ISO timestamp


@dataclass
class EventSettlement:
    station: str
    settlement_date: str  # Climate day (CLI "FOR" date), NOT issue date
    market_type: str  # "high" or "low"
    actual_value_f: float | None = None
    settlement_source: str = ""  # "cli", "dsm", "manual"
    raw_text: str = ""
    received_at: str = ""


@dataclass
class AtmosphericData:
    """Hourly atmospheric parameters from Open-Meteo (non-pressure-level)."""
    station: str
    valid_time_utc: str  # ISO timestamp
    model: str  # e.g. "gfs_seamless"
    fetch_time_utc: str | None = None
    shortwave_radiation: float | None = None  # W/m^2
    direct_radiation: float | None = None  # W/m^2
    diffuse_radiation: float | None = None  # W/m^2
    cape: float | None = None  # J/kg
    lifted_index: float | None = None
    boundary_layer_height: float | None = None  # meters
    precipitable_water_mm: float | None = None  # mm (total column)
    soil_temperature_0_7cm: float | None = None  # Celsius
    soil_moisture_0_1cm: float | None = None  # m³/m³ volumetric
    precipitation_mm: float | None = None  # mm (hourly total)
    rain_mm: float | None = None  # mm (hourly liquid only)
    showers_mm: float | None = None  # mm (hourly convective)
    precipitation_probability: float | None = None  # percent


@dataclass
class SSTObservation:
    """Sea surface temperature from NDBC buoy."""
    station_id: str  # e.g. "FWYF1"
    name: str
    timestamp_utc: str
    water_temp_c: float | None = None
    water_temp_f: float | None = None
    air_temp_c: float | None = None
    wind_speed_mps: float | None = None
    wind_dir_deg: float | None = None
    pressure_hpa: float | None = None
    distance_mi: float | None = None


@dataclass
class FAWNObservation:
    """15-min actual sensor observation from FAWN (Homestead station)."""
    station_id: str  # e.g. "440"
    station_name: str  # e.g. "Homestead"
    timestamp_utc: str
    air_temp_f: float | None = None
    air_temp_c: float | None = None
    dew_point_f: float | None = None
    relative_humidity: float | None = None
    wind_speed_mph: float | None = None
    wind_gust_mph: float | None = None
    wind_direction_deg: float | None = None
    solar_radiation_wm2: float | None = None  # Actual sensor W/m²
    soil_temp_c: float | None = None
    soil_temp_f: float | None = None
    rain_mm: float | None = None  # 15-min accumulation
    rain_in: float | None = None


SKY_COVER_CODE_TO_PCT: dict[str, float] = {
    "CLR": 0.0,
    "SKC": 0.0,
    "FEW": 25.0,
    "SCT": 50.0,
    "BKN": 75.0,
    "OVC": 100.0,
    "VV": 100.0,
}
