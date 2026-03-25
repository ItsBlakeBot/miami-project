"""Source registry for modular baseline construction.

This module defines the canonical interfaces for forecast-source snapshots so the
baseline layer can combine sources without hard-coding provider-specific logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable, Protocol

UTC = timezone.utc


@dataclass(frozen=True)
class ForecastSourceSnapshot:
    """Single-source forecast state at an evaluation time."""

    source_name: str
    family_name: str
    issued_at_utc: datetime
    target_date: str
    forecast_high_f: float | None = None
    forecast_low_f: float | None = None
    quantiles_high: dict[float, float] = field(default_factory=dict)
    quantiles_low: dict[float, float] = field(default_factory=dict)
    tracking_error_high: float | None = None
    tracking_error_low: float | None = None
    metadata: dict[str, str | float | int | bool] = field(default_factory=dict)

    @property
    def source_key(self) -> str:
        return f"{self.family_name}:{self.source_name}"

    def age_hours(self, now_utc: datetime) -> float:
        if now_utc.tzinfo is None:
            now_utc = now_utc.replace(tzinfo=UTC)
        return max(0.0, (now_utc.astimezone(UTC) - self.issued_at_utc.astimezone(UTC)).total_seconds() / 3600.0)


class ForecastSource(Protocol):
    """Interface for future source adapters."""

    name: str
    family: str

    def snapshot(self, target_date: str, eval_time_utc: datetime) -> ForecastSourceSnapshot | None:
        ...


@dataclass
class SourceFamilySummary:
    family_name: str
    snapshots: list[ForecastSourceSnapshot]

    @property
    def source_count(self) -> int:
        return len(self.snapshots)


class SourceRegistry:
    """Registry for baseline source adapters.

    The registry itself is intentionally simple: registration and snapshot
    collection. Weighting, bias correction, and quantile combination live in
    later layers.
    """

    def __init__(self) -> None:
        self._sources: dict[str, ForecastSource] = {}

    def register(self, source: ForecastSource) -> None:
        key = f"{source.family}:{source.name}"
        self._sources[key] = source

    def register_many(self, sources: Iterable[ForecastSource]) -> None:
        for source in sources:
            self.register(source)

    def collect(self, target_date: str, eval_time_utc: datetime) -> list[ForecastSourceSnapshot]:
        snapshots: list[ForecastSourceSnapshot] = []
        for source in self._sources.values():
            snapshot = source.snapshot(target_date, eval_time_utc)
            if snapshot is not None:
                snapshots.append(snapshot)
        return snapshots

    def latest_by_source_key(self, snapshots: Iterable[ForecastSourceSnapshot]) -> list[ForecastSourceSnapshot]:
        latest: dict[str, ForecastSourceSnapshot] = {}
        for snapshot in sorted(snapshots, key=lambda s: s.issued_at_utc, reverse=True):
            latest.setdefault(snapshot.source_key, snapshot)
        return list(latest.values())

    def group_by_family(self, snapshots: Iterable[ForecastSourceSnapshot]) -> list[SourceFamilySummary]:
        grouped: dict[str, list[ForecastSourceSnapshot]] = {}
        for snapshot in snapshots:
            grouped.setdefault(snapshot.family_name, []).append(snapshot)
        return [SourceFamilySummary(family, items) for family, items in sorted(grouped.items())]
