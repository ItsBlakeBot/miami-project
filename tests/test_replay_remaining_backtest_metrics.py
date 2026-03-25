from __future__ import annotations

from analyzer.replay_remaining_backtest import (
    _approx_crps_from_quantiles,
    _confidence_bucket,
    _remaining_quantiles,
    _reliability_bins,
    _ttl_bucket,
)


def test_ttl_bucket_boundaries() -> None:
    assert _ttl_bucket(0.0) == "0-3h"
    assert _ttl_bucket(2.99) == "0-3h"
    assert _ttl_bucket(3.0) == "3-6h"
    assert _ttl_bucket(6.0) == "6-12h"
    assert _ttl_bucket(12.0) == "12-24h"
    assert _ttl_bucket(24.0) == "24h+"


def test_reliability_bins_basic_shape() -> None:
    pairs = [(0.1, 0), (0.15, 0), (0.75, 1), (0.8, 1)]
    bins = _reliability_bins(pairs, bins=5)

    assert len(bins) >= 2
    first = bins[0]
    assert "mean_prob" in first
    assert "empirical_yes" in first
    assert "calibration_gap" in first


def test_remaining_quantiles_transform_high_and_low() -> None:
    quantiles = {0.1: 79.0, 0.5: 80.0, 0.9: 82.0}

    high = _remaining_quantiles("high", quantiles, running_high_f=78.0, running_low_f=None)
    low = _remaining_quantiles("low", quantiles, running_high_f=None, running_low_f=81.0)

    assert high[0.1] == 1.0
    assert high[0.5] == 2.0
    assert high[0.9] == 4.0

    # Low reverses quantiles: remaining move grows as final low gets colder.
    assert low[0.1] == 0.0
    assert low[0.5] == 1.0
    assert low[0.9] == 2.0


def test_crps_and_confidence_bucket_helpers() -> None:
    crps = _approx_crps_from_quantiles({0.1: 1.0, 0.5: 2.0, 0.9: 3.0}, actual=2.0)
    assert crps is not None
    assert crps >= 0.0

    assert _confidence_bucket(0.2) == "low"
    assert _confidence_bucket(0.6) == "medium"
    assert _confidence_bucket(0.9) == "high"
