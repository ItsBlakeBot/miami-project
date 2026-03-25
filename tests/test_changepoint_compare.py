from __future__ import annotations

from datetime import datetime, timedelta, timezone

from analyzer.changepoint_replay_compare import FirePoint, _event_times, _score

UTC = timezone.utc


def test_event_times_detects_shifts_with_spacing() -> None:
    base = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)
    rows = []
    # steady
    for i in range(3):
        rows.append(
            {
                "timestamp_utc": (base + timedelta(minutes=5 * i)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "temperature_f": 80.0,
                "dew_point_f": 70.0,
                "pressure_hpa": 1014.0,
                "wind_heading_deg": 90.0,
            }
        )
    # abrupt temp jump
    rows.append(
        {
            "timestamp_utc": (base + timedelta(minutes=20)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "temperature_f": 83.0,
            "dew_point_f": 70.0,
            "pressure_hpa": 1014.0,
            "wind_heading_deg": 90.0,
        }
    )
    # another jump within spacing window (should be deduped)
    rows.append(
        {
            "timestamp_utc": (base + timedelta(minutes=25)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "temperature_f": 86.0,
            "dew_point_f": 70.0,
            "pressure_hpa": 1014.0,
            "wind_heading_deg": 90.0,
        }
    )

    events = _event_times(rows, min_spacing_minutes=30.0)
    assert len(events) == 1


def test_score_matches_and_false_positive_proxy() -> None:
    t0 = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)
    events = [t0, t0 + timedelta(minutes=180)]

    fires = [
        FirePoint(ts=t0 + timedelta(minutes=15), layer=2),
        FirePoint(ts=t0 + timedelta(minutes=195), layer=3),
        FirePoint(ts=t0 + timedelta(minutes=500), layer=2),
    ]

    matched, missed, latencies, fp = _score(events, fires, window_minutes=120.0)
    assert matched == 2
    assert missed == 0
    assert latencies == [15.0, 15.0]
    assert fp == 1
