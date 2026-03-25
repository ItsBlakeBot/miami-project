"""Shared replay/live time guardrails.

Phase mapping:
- T0.2 canonical replay harness hardening

Centralizes timestamp parsing, climate-day target-date boundaries, and quote
freshness checks so replay and trading code paths share one implementation.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

UTC = timezone.utc


def parse_utc(ts: str | None) -> datetime | None:
    if not ts:
        return None

    raw = str(ts).strip()
    for fmt in (
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M",
    ):
        try:
            return datetime.strptime(raw, fmt).replace(tzinfo=UTC)
        except ValueError:
            continue

    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt.astimezone(UTC)
    except ValueError:
        return None


def to_utc_iso(dt: datetime) -> str:
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def climate_target_date(now_utc: datetime, boundary_hour_utc: int = 5) -> str:
    """Return climate-day target date for a UTC timestamp.

    Example: with boundary_hour_utc=5,
    - 2026-03-21T04:59Z -> 2026-03-20
    - 2026-03-21T05:00Z -> 2026-03-21
    """
    d: date = now_utc.date()
    if now_utc.hour < boundary_hour_utc:
        d = d - timedelta(days=1)
    return d.strftime("%Y-%m-%d")


def age_minutes(now_utc: datetime, ts: str | None) -> float | None:
    dt = parse_utc(ts)
    if dt is None:
        return None
    return round((now_utc.astimezone(UTC) - dt).total_seconds() / 60.0, 2)


def freshness_status(
    now_utc: datetime,
    ts: str | None,
    *,
    stale_after_minutes: int,
    future_tolerance_minutes: float = 1.0,
) -> tuple[str, float | None]:
    """Classify a timestamp into a replay-safe freshness status.

    Returns (status, age_minutes), where status is one of:
    - missing: timestamp is null/empty
    - unparseable: timestamp string exists but could not be parsed
    - future: timestamp is ahead of now beyond tolerance
    - stale: timestamp is older than stale_after_minutes
    - fresh: timestamp is parseable, non-future, and within freshness window
    """
    if ts is None or str(ts).strip() == "":
        return "missing", None

    age = age_minutes(now_utc, ts)
    if age is None:
        return "unparseable", None
    if age < -abs(float(future_tolerance_minutes)):
        return "future", age
    if age > float(stale_after_minutes):
        return "stale", age
    return "fresh", age


def is_stale(now_utc: datetime, ts: str | None, *, max_age_minutes: int) -> bool:
    """Quote freshness guard.

    Returns True when timestamp is missing/unparseable OR older than max_age.
    Future timestamps are treated as non-stale but can be diagnosed by callers
    via freshness_status(...).
    """
    status, _age = freshness_status(
        now_utc,
        ts,
        stale_after_minutes=max_age_minutes,
    )
    return status in {"missing", "unparseable", "stale"}
