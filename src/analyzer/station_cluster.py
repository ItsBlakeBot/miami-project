"""Simple station-cluster summarization for local transfer features."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean


@dataclass
class StationReading:
    station: str
    temp_f: float
    dew_f: float | None = None
    wind_dir_deg: float | None = None
    wind_speed_mph: float | None = None


@dataclass
class ClusterSummary:
    name: str
    station_count: int
    mean_temp_f: float | None
    min_temp_f: float | None
    max_temp_f: float | None


DEFAULT_CLUSTERS: dict[str, set[str]] = {
    "immediate_airports": {"KMIA", "KOPF", "KTMB", "KHWO"},
    "coastal": {"KFLL", "KPMP", "KBCT"},
    "inland_south": {"KHST", "KTMB", "KHWO", "KK70"},
}


def summarize_cluster(readings: list[StationReading], stations: set[str], *, name: str) -> ClusterSummary:
    cluster_values = [reading.temp_f for reading in readings if reading.station in stations]
    if not cluster_values:
        return ClusterSummary(name=name, station_count=0, mean_temp_f=None, min_temp_f=None, max_temp_f=None)
    return ClusterSummary(
        name=name,
        station_count=len(cluster_values),
        mean_temp_f=round(mean(cluster_values), 3),
        min_temp_f=min(cluster_values),
        max_temp_f=max(cluster_values),
    )


def summarize_default_clusters(readings: list[StationReading]) -> dict[str, ClusterSummary]:
    return {name: summarize_cluster(readings, stations, name=name) for name, stations in DEFAULT_CLUSTERS.items()}
