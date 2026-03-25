from analyzer.review_labels import extract_review_labels


SAMPLE_REVIEW = """# KMIA Daily Review — 2026-03-17

## Structured Summary

- **Station:** KMIA
- **Time basis:** LST (UTC-5, fixed year-round for Kalshi/NWS climate settlement)
- **Climate day:** 12:00 AM LST → 11:59 PM LST
- **CLI settlement high:** 71.0°F at 1:10 AM LST
- **CLI settlement low:** 56.0°F at 11:52 PM LST
- **Obs high:** 71.6°F
- **Obs low:** 55.4°F
- **Primary regime:** `advective_cooling`
- **Background regime:** `radiative_cooling`
- **Path class:** `cold_grind`
- **Core day type:** post-frontal continental day

## Signal-to-Adjustment Notes

- **`high_locked_early`**
- **`postfrontal_continental_confirmed`**
- **`low_endpoint_locked`**

## Best Trade Lessons from the Day

### Best high-side idea

- **YES `T73` in the opening part of the climate day**

### Best core trade of the day

- **YES `T57` around midday**
"""


def test_extract_review_labels():
    record = extract_review_labels(SAMPLE_REVIEW)
    assert record.target_date == "2026-03-17"
    assert record.station == "KMIA"
    assert record.primary_regime == "advective_cooling"
    assert record.background_regime == "radiative_cooling"
    assert record.path_class == "cold_grind"
    assert record.cli_high_f == 71.0
    assert record.cli_low_f == 56.0
    assert record.obs_low_f == 55.4
    assert "high_locked_early" in record.signal_labels
    assert "low_endpoint_locked" in record.clamp_labels
    assert record.best_high_expression == "YES `T73` in the opening part of the climate day"
