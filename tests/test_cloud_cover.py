"""Tests for cloud cover estimation module."""

from __future__ import annotations

import math

import pytest

from engine.cloud_cover import (
    SKY_COVER_FRACTION,
    clearness_to_cloud_fraction,
    expected_clear_sky_radiation,
    fawn_clearness_index,
    metar_sky_to_fraction,
)


class TestMETARSkyToFraction:
    def test_clear_sky(self):
        assert metar_sky_to_fraction("CLR") == 0.0
        assert metar_sky_to_fraction("SKC") == 0.0

    def test_few(self):
        assert metar_sky_to_fraction("FEW") == 0.125

    def test_scattered(self):
        assert metar_sky_to_fraction("SCT") == 0.375

    def test_broken(self):
        assert metar_sky_to_fraction("BKN") == 0.75

    def test_overcast(self):
        assert metar_sky_to_fraction("OVC") == 1.0

    def test_case_insensitive(self):
        assert metar_sky_to_fraction("bkn") == 0.75
        assert metar_sky_to_fraction("Ovc") == 1.0

    def test_none_returns_none(self):
        assert metar_sky_to_fraction(None) is None

    def test_unknown_code_returns_none(self):
        assert metar_sky_to_fraction("ZZZZZ") is None

    def test_vertical_visibility(self):
        assert metar_sky_to_fraction("VV") == 1.0


class TestExpectedClearSkyRadiation:
    def test_solar_noon_positive(self):
        """At solar noon in Miami, radiation should be substantial."""
        # Solar noon at KMIA ≈ 17:00 UTC
        rad = expected_clear_sky_radiation(hour_utc=17.0, day_of_year=80, lat_deg=25.8)
        assert rad > 500.0  # should be >500 W/m² at noon in spring

    def test_night_is_zero(self):
        """At midnight, radiation should be zero."""
        rad = expected_clear_sky_radiation(hour_utc=5.0, day_of_year=80, lat_deg=25.8)
        assert rad == 0.0

    def test_summer_higher_than_winter(self):
        """Summer solstice should have higher noon radiation than winter."""
        summer = expected_clear_sky_radiation(hour_utc=17.0, day_of_year=172, lat_deg=25.8)
        winter = expected_clear_sky_radiation(hour_utc=17.0, day_of_year=355, lat_deg=25.8)
        assert summer > winter


class TestFAWNClearnessIndex:
    def test_clear_sky_near_one(self):
        """High solar radiation relative to expected → clearness near 1.0."""
        # At noon spring, expected is ~900 W/m². If measured is 850:
        ci = fawn_clearness_index(850.0, hour_utc=17.0, day_of_year=80)
        assert ci is not None
        assert 0.8 < ci < 1.1

    def test_overcast_low_clearness(self):
        """Low solar radiation → low clearness index."""
        ci = fawn_clearness_index(100.0, hour_utc=17.0, day_of_year=80)
        assert ci is not None
        assert ci < 0.3

    def test_night_returns_none(self):
        """Can't compute clearness at night (expected radiation too low)."""
        ci = fawn_clearness_index(0.0, hour_utc=5.0, day_of_year=80)
        assert ci is None

    def test_capped_at_1_2(self):
        """Clearness index should be capped at 1.2 (reflection edge cases)."""
        ci = fawn_clearness_index(2000.0, hour_utc=17.0, day_of_year=80)
        assert ci is not None
        assert ci <= 1.2


class TestClearnessToCloudFraction:
    def test_clear_sky(self):
        assert clearness_to_cloud_fraction(1.0) == 0.0

    def test_overcast(self):
        assert clearness_to_cloud_fraction(0.0) == 1.0

    def test_partial_clouds(self):
        frac = clearness_to_cloud_fraction(0.5)
        assert frac is not None
        assert 0.4 < frac < 0.6

    def test_none_returns_none(self):
        assert clearness_to_cloud_fraction(None) is None

    def test_clamped(self):
        assert clearness_to_cloud_fraction(1.5) == 0.0  # clamped
        assert clearness_to_cloud_fraction(-0.5) == 1.0  # clamped
