from __future__ import annotations

from datetime import datetime, timezone

from engine.replay_context import (
    age_minutes,
    climate_target_date,
    freshness_status,
    is_stale,
    parse_utc,
)

UTC = timezone.utc


def test_parse_utc_supports_z_and_offset() -> None:
    a = parse_utc("2026-03-21T10:00:00Z")
    b = parse_utc("2026-03-21T10:00:00+00:00")

    assert a is not None and b is not None
    assert a == b
    assert a.tzinfo is not None


def test_climate_target_date_boundary() -> None:
    before = datetime(2026, 3, 21, 4, 59, tzinfo=UTC)
    after = datetime(2026, 3, 21, 5, 0, tzinfo=UTC)

    assert climate_target_date(before, boundary_hour_utc=5) == "2026-03-20"
    assert climate_target_date(after, boundary_hour_utc=5) == "2026-03-21"


def test_is_stale_and_age_minutes_behavior() -> None:
    now = datetime(2026, 3, 21, 10, 30, tzinfo=UTC)

    assert age_minutes(now, "2026-03-21T10:00:00Z") == 30.0
    assert is_stale(now, "2026-03-21T10:00:00Z", max_age_minutes=20) is True
    assert is_stale(now, "2026-03-21T10:20:00Z", max_age_minutes=20) is False
    assert is_stale(now, None, max_age_minutes=20) is True


def test_freshness_status_categories() -> None:
    now = datetime(2026, 3, 21, 10, 30, tzinfo=UTC)

    status, age = freshness_status(now, None, stale_after_minutes=20)
    assert status == "missing"
    assert age is None

    status, age = freshness_status(now, "not-a-date", stale_after_minutes=20)
    assert status == "unparseable"
    assert age is None

    status, age = freshness_status(now, "2026-03-21T10:40:00Z", stale_after_minutes=20)
    assert status == "future"
    assert age is not None and age < 0

    status, age = freshness_status(now, "2026-03-21T10:05:00Z", stale_after_minutes=20)
    assert status == "stale"
    assert age == 25.0

    status, age = freshness_status(now, "2026-03-21T10:25:00Z", stale_after_minutes=20)
    assert status == "fresh"
    assert age == 5.0
