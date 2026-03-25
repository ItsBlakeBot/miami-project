"""Analyzer data types — scoring, verification, and analysis dataclasses.

Extracted from the former signals.signals module. These types are used by
the analyzer layer for model scoring, ensemble statistics, and bias tracking.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EnsembleStats:
    """Summary statistics for an ensemble (ECMWF or GFS)."""
    name: str
    mean: float
    std: float
    min_val: float
    max_val: float
    p10: float
    p25: float
    p50: float  # median
    p75: float
    p90: float
    n_members: int
    members: list[float] = field(default_factory=list)


@dataclass
class ModelForecastSummary:
    """A single model's high/low forecast for a date."""
    model: str
    source: str
    high_f: float | None
    low_f: float | None
    run_time: str | None = None


@dataclass
class ModelBias:
    """Per-model bias over recent settlements."""
    model: str
    high_bias: float | None  # positive = warm bias
    low_bias: float | None
    high_mae: float | None
    low_mae: float | None
    n_days: int = 0


@dataclass
class BracketPrice:
    """Single Kalshi bracket with current pricing."""
    ticker: str
    floor_strike: float | None
    cap_strike: float | None
    yes_bid: float | None  # cents
    yes_ask: float | None
    mid: float | None  # implied probability (cents / 100)
    volume: int | None = None
    spread: float | None = None
