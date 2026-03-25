from datetime import datetime, timezone

from engine.climate_clock import (
    checkpoint_utc,
    climate_date_for_utc,
    climate_day_bounds_iso,
    hours_remaining_to_settlement,
    to_lst,
)


def test_climate_day_bounds_are_fixed_to_lst_not_dst():
    start, end = climate_day_bounds_iso("2026-03-17")
    assert start == "2026-03-17T05:00:00Z"
    assert end == "2026-03-18T05:00:00Z"


def test_to_lst_uses_fixed_utc_minus_five():
    dt = to_lst("2026-03-18T03:00:00Z")
    assert dt.isoformat() == "2026-03-17T22:00:00-05:00"


def test_climate_date_for_utc_maps_late_evening_correctly():
    assert climate_date_for_utc("2026-03-18T03:00:00Z") == "2026-03-17"


def test_checkpoint_utc_converts_lst_hour_to_utc():
    dt = checkpoint_utc("2026-03-17", 22)
    assert dt.isoformat() == "2026-03-18T03:00:00+00:00"


def test_hours_remaining_to_settlement():
    ts = datetime(2026, 3, 18, 3, 0, tzinfo=timezone.utc)  # 10 PM LST on 3/17
    remaining = hours_remaining_to_settlement(ts, "2026-03-17")
    assert remaining == 2.0
