"""Climate-day clock utilities.

Kalshi weather markets settle from NWS climate reports using local standard time
(LST), not wall-clock DST. For KMIA that means the climate day always runs on a
fixed UTC-5 clock.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone

LST = timezone(timedelta(hours=-5), name="LST")
UTC = timezone.utc


def parse_utc_timestamp(value: str | datetime) -> datetime:
    """Parse an ISO timestamp into an aware UTC datetime.

    Accepts:
    - datetime objects (aware or naive)
    - strings ending in Z
    - strings with explicit offsets
    - strings without offsets (assumed UTC)
    """
    if isinstance(value, datetime):
        dt = value
    else:
        raw = value.strip()
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        dt = datetime.fromisoformat(raw)

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def to_lst(value: str | datetime) -> datetime:
    """Convert a UTC timestamp into fixed local standard time."""
    return parse_utc_timestamp(value).astimezone(LST)


def climate_day_bounds_utc(target_date: str | date) -> tuple[datetime, datetime]:
    """Return climate-day bounds in UTC for a target LST date.

    Climate day: 00:00 LST to 00:00 LST next day.
    For UTC-5 fixed LST that is 05:00Z to 05:00Z next day.
    """
    if isinstance(target_date, str):
        day = date.fromisoformat(target_date)
    else:
        day = target_date

    start_lst = datetime.combine(day, time.min, tzinfo=LST)
    end_lst = start_lst + timedelta(days=1)
    return start_lst.astimezone(UTC), end_lst.astimezone(UTC)


def climate_day_bounds_iso(target_date: str | date) -> tuple[str, str]:
    """Return UTC ISO strings for the target climate day."""
    start, end = climate_day_bounds_utc(target_date)
    return _iso_z(start), _iso_z(end)


def climate_date_for_utc(value: str | datetime) -> str:
    """Return the LST climate date for a UTC timestamp."""
    return to_lst(value).date().isoformat()


def hours_remaining_to_settlement(value: str | datetime, target_date: str | date) -> float:
    """Hours remaining until the end of the target climate day."""
    now_utc = parse_utc_timestamp(value)
    _, end_utc = climate_day_bounds_utc(target_date)
    return (end_utc - now_utc).total_seconds() / 3600.0


def checkpoint_utc(target_date: str | date, lst_hour: int, minute: int = 0) -> datetime:
    """Return a UTC checkpoint datetime for an hour expressed in LST."""
    if not 0 <= lst_hour <= 23:
        raise ValueError(f"lst_hour must be 0..23, got {lst_hour}")
    if not 0 <= minute <= 59:
        raise ValueError(f"minute must be 0..59, got {minute}")

    if isinstance(target_date, str):
        day = date.fromisoformat(target_date)
    else:
        day = target_date

    lst_dt = datetime.combine(day, time(hour=lst_hour, minute=minute), tzinfo=LST)
    return lst_dt.astimezone(UTC)


def _iso_z(value: datetime) -> str:
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
