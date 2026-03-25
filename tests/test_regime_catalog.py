"""Tests for live regime catalog and inference."""

from engine.regime_catalog import (
    LiveRegimeState,
    RegimeType,
    infer_live_regime,
)


class TestRegimeInference:
    def test_convective_outflow_detected(self):
        """High CAPE + PW in afternoon → outflow regime."""
        state = infer_live_regime(
            cape=2500.0, pw_mm=42.0, hour_lst=19.0,
        )
        assert state.primary == RegimeType.CONVECTIVE_OUTFLOW
        assert state.sigma_multiplier > 1.3

    def test_frontal_passage_detected(self):
        """NW wind + changepoint + dew crash → frontal passage."""
        state = infer_live_regime(
            wind_dir_deg=320.0,
            bocpd_changepoint_prob=0.8,
            dew_crash_active=True,
        )
        assert state.primary == RegimeType.FRONTAL_PASSAGE

    def test_cloud_suppression_detected(self):
        """Heavy cloud cover → cloud suppression."""
        state = infer_live_regime(cloud_fraction=0.85)
        assert state.primary == RegimeType.CLOUD_SUPPRESSION
        assert state.mu_bias_high_f < 0  # highs suppressed
        assert state.mu_bias_low_f > 0   # lows elevated (less radiational cooling)

    def test_marine_stable_detected(self):
        """SE winds, low CAPE → marine stable."""
        state = infer_live_regime(
            wind_dir_deg=140.0, cape=300.0,
        )
        assert state.primary == RegimeType.MARINE_STABLE
        assert state.sigma_multiplier < 1.0  # narrow uncertainty

    def test_inland_heating_detected(self):
        """Clear sky, daytime, low CAPE → inland heating."""
        state = infer_live_regime(
            cloud_fraction=0.05, cape=200.0, hour_lst=14.0,
        )
        assert state.primary == RegimeType.INLAND_HEATING

    def test_transition_default(self):
        """No strong signals → transition regime."""
        state = infer_live_regime()
        assert state.primary == RegimeType.TRANSITION

    def test_probabilities_sum_to_one(self):
        """Regime probabilities should sum to ~1.0."""
        state = infer_live_regime(cape=1000.0, pw_mm=30.0, hour_lst=15.0)
        total = sum(state.probabilities.values())
        assert abs(total - 1.0) < 0.01

    def test_confidence_is_bounded(self):
        """Confidence should be between 0 and 1."""
        state = infer_live_regime(cape=3000.0, pw_mm=50.0, hour_lst=20.0)
        assert 0.0 <= state.confidence <= 1.0

    def test_to_dict(self):
        """Serialization should work."""
        state = infer_live_regime(cape=500.0)
        d = state.to_dict()
        assert "primary" in d
        assert "confidence" in d
        assert "probabilities" in d
