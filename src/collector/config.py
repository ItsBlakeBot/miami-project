"""Configuration loading from TOML + .env overlay."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import dotenv_values


@dataclass(frozen=True)
class StationConfig:
    code: str
    label: str
    lat: float
    lon: float
    nws_office: str
    nws_grid_x: int
    nws_grid_y: int
    tz_standard: str
    utc_offset_hours: int
    dsm_zulu: str
    elevation_m: float
    kalshi_high_series: str
    kalshi_low_series: str
    nws_cli_location: str


@dataclass(frozen=True)
class WethrConfig:
    rest_base: str
    sse_base: str
    api_key: str
    forecast_poll_secs: int = 300
    cache_ttl_secs: int = 300


@dataclass(frozen=True)
class OpenMeteoConfig:
    base_url: str
    ensemble_url: str
    api_key: str
    det_poll_secs: int = 300
    ens_poll_secs: int = 3600
    plev_poll_secs: int = 3600


@dataclass(frozen=True)
class NWSConfig:
    base_url: str
    user_agent: str
    obs_poll_secs: int = 150
    cli_poll_secs: int = 1800


@dataclass(frozen=True)
class KalshiConfig:
    rest_base: str
    ws_url: str
    api_key_id: str
    private_key_path: str
    market_discovery_secs: int = 600
    ws_ping_secs: int = 20
    ws_ping_timeout_secs: int = 10


@dataclass(frozen=True)
class SynopticConfig:
    token: str
    radius_mi: float = 30.0
    poll_secs: int = 300  # 5 minutes
    within_minutes: int = 30
    max_stations: int = 50


@dataclass(frozen=True)
class IEMConfig:
    network: str = "FL_ASOS"
    radius_mi: float = 35.0
    poll_secs: int = 300  # 5 minutes


@dataclass(frozen=True)
class FAWNConfig:
    station_id: str = "440"  # Homestead — primary station (backward compat)
    station_name: str = "Homestead"
    poll_secs: int = 900  # 15 minutes (matches FAWN reporting interval)
    station_ids: tuple[str, ...] = ("440",)  # multi-station support


@dataclass(frozen=True)
class NDBCConfig:
    poll_secs: int = 1800  # 30 minutes (NDBC updates hourly)


@dataclass(frozen=True)
class Config:
    station: StationConfig
    wethr: WethrConfig
    openmeteo: OpenMeteoConfig
    nws: NWSConfig
    kalshi: KalshiConfig
    synoptic: SynopticConfig
    iem: IEMConfig
    fawn: FAWNConfig
    ndbc: NDBCConfig


def load_config(config_path: str | Path = "config.toml") -> Config:
    """Load config from TOML file with .env credential overlay."""
    path = Path(config_path)
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    # Load credentials from .env
    env_path = raw.get("credentials", {}).get("env_path", "")
    env = dotenv_values(env_path) if env_path else {}

    kalshi_key_path = raw.get("credentials", {}).get(
        "kalshi_key_path",
        env.get("KALSHI_PRIVATE_KEY_PATH", ""),
    )

    station = StationConfig(**raw["station"])

    wethr = WethrConfig(
        rest_base=raw["wethr"]["rest_base"],
        sse_base=raw["wethr"]["sse_base"],
        api_key=env.get("WETHR_API_KEY", ""),
        forecast_poll_secs=raw["wethr"].get("forecast_poll_secs", 300),
        cache_ttl_secs=raw["wethr"].get("cache_ttl_secs", 300),
    )

    openmeteo = OpenMeteoConfig(
        base_url=raw["openmeteo"]["base_url"],
        ensemble_url=raw["openmeteo"]["ensemble_url"],
        api_key=env.get("OPENMETEO_API_KEY", ""),
        det_poll_secs=raw["openmeteo"].get("det_poll_secs", 300),
        ens_poll_secs=raw["openmeteo"].get("ens_poll_secs", 3600),
        plev_poll_secs=raw["openmeteo"].get("plev_poll_secs", 3600),
    )

    nws = NWSConfig(
        base_url=raw["nws"]["base_url"],
        user_agent=raw["nws"]["user_agent"],
        obs_poll_secs=raw["nws"].get("obs_poll_secs", 150),
        cli_poll_secs=raw["nws"].get("cli_poll_secs", 1800),
    )

    kalshi = KalshiConfig(
        rest_base=raw["kalshi"]["rest_base"],
        ws_url=raw["kalshi"]["ws_url"],
        api_key_id=env.get("KALSHI_API_KEY_ID", ""),
        private_key_path=kalshi_key_path,
        market_discovery_secs=raw["kalshi"].get("market_discovery_secs", 600),
        ws_ping_secs=raw["kalshi"].get("ws_ping_secs", 20),
        ws_ping_timeout_secs=raw["kalshi"].get("ws_ping_timeout_secs", 10),
    )

    syn_raw = raw.get("synoptic", {})
    synoptic = SynopticConfig(
        token=env.get("SYNOPTIC_TOKEN", ""),
        radius_mi=syn_raw.get("radius_mi", 30.0),
        poll_secs=syn_raw.get("poll_secs", 300),
        within_minutes=syn_raw.get("within_minutes", 30),
        max_stations=syn_raw.get("max_stations", 50),
    )

    iem_raw = raw.get("iem", {})
    iem = IEMConfig(
        network=iem_raw.get("network", "FL_ASOS"),
        radius_mi=iem_raw.get("radius_mi", 35.0),
        poll_secs=iem_raw.get("poll_secs", 300),
    )

    fawn_raw = raw.get("fawn", {})
    fawn_station_ids = fawn_raw.get("station_ids", [fawn_raw.get("station_id", "440")])
    fawn = FAWNConfig(
        station_id=fawn_raw.get("station_id", "440"),
        station_name=fawn_raw.get("station_name", "Homestead"),
        poll_secs=fawn_raw.get("poll_secs", 900),
        station_ids=tuple(fawn_station_ids),
    )

    ndbc_raw = raw.get("ndbc", {})
    ndbc = NDBCConfig(
        poll_secs=ndbc_raw.get("poll_secs", 1800),
    )

    return Config(
        station=station,
        wethr=wethr,
        openmeteo=openmeteo,
        nws=nws,
        kalshi=kalshi,
        synoptic=synoptic,
        iem=iem,
        fawn=fawn,
        ndbc=ndbc,
    )
