"""Tests for EMOS calibration module."""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from engine.emos import (
    EMOSCoefficients,
    EMOSFitConfig,
    EMOSState,
    EMOSTrainingSample,
    _SIGMA_FLOOR,
    crps_normal,
    crps_normal_vectorized,
    fit_emos,
)


# ---------------------------------------------------------------------------
# CRPS tests
# ---------------------------------------------------------------------------
class TestCRPSNormal:
    """Test closed-form CRPS for Normal distribution."""

    def test_crps_at_mean_is_positive(self):
        """CRPS should be positive even when obs equals mean."""
        c = crps_normal(mu=70.0, sigma=2.0, obs=70.0)
        assert c > 0.0

    def test_crps_increases_with_error(self):
        """CRPS should increase as obs moves away from mean."""
        c_close = crps_normal(mu=70.0, sigma=2.0, obs=71.0)
        c_far = crps_normal(mu=70.0, sigma=2.0, obs=75.0)
        assert c_far > c_close

    def test_crps_decreases_with_sharpness(self):
        """Narrower sigma should give lower CRPS when obs is at mean."""
        c_wide = crps_normal(mu=70.0, sigma=4.0, obs=70.0)
        c_narrow = crps_normal(mu=70.0, sigma=1.0, obs=70.0)
        assert c_narrow < c_wide

    def test_crps_known_value(self):
        """Check CRPS against known analytical result.

        For N(0,1) at y=0: CRPS = 1/sqrt(pi) ≈ 0.56419
        Formula: sigma * [z*(2*Phi(z)-1) + 2*phi(z) - 1/sqrt(pi)]
        At z=0: sigma * [0 + 2*phi(0) - 1/sqrt(pi)]
               = 1 * [2 * 0.3989 - 0.5642] ≈ 0.2337
        """
        # The formula gives a specific value
        c = crps_normal(mu=0.0, sigma=1.0, obs=0.0)
        # 2*phi(0) - 1/sqrt(pi) = 2*0.39894 - 0.56419 = 0.23369
        assert abs(c - 0.2337) < 0.001

    def test_crps_symmetric(self):
        """CRPS should be same for obs above and below mean by same amount."""
        c_above = crps_normal(mu=70.0, sigma=2.0, obs=73.0)
        c_below = crps_normal(mu=70.0, sigma=2.0, obs=67.0)
        assert abs(c_above - c_below) < 1e-10

    def test_crps_sigma_floor(self):
        """CRPS should still work with very small sigma (floored)."""
        c = crps_normal(mu=70.0, sigma=0.001, obs=70.0)
        assert c > 0.0
        assert math.isfinite(c)

    def test_crps_vectorized_matches_scalar(self):
        """Vectorized CRPS should match scalar version."""
        mus = np.array([70.0, 72.0, 68.0])
        sigmas = np.array([2.0, 3.0, 1.5])
        obs = np.array([71.0, 70.0, 69.0])

        vec_result = crps_normal_vectorized(mus, sigmas, obs)

        for i in range(3):
            scalar = crps_normal(mus[i], sigmas[i], obs[i])
            assert abs(vec_result[i] - scalar) < 1e-10


# ---------------------------------------------------------------------------
# EMOSCoefficients tests
# ---------------------------------------------------------------------------
class TestEMOSCoefficients:
    """Test EMOS coefficient prediction and serialization."""

    def test_identity_mapping(self):
        """Default coefficients (a=0, b=1) should pass through consensus."""
        coeff = EMOSCoefficients(a=0.0, b=1.0, c=1.0, d=0.5)
        mu, sigma = coeff.predict(consensus_f=72.0, spread=2.0)
        assert mu == 72.0
        assert sigma == math.sqrt(1.0 + 0.5 * 4.0)  # sqrt(3.0)

    def test_bias_correction(self):
        """Non-zero intercept should shift mu."""
        coeff = EMOSCoefficients(a=1.5, b=1.0, c=1.0, d=0.0)
        mu, sigma = coeff.predict(consensus_f=70.0)
        assert mu == 71.5

    def test_slope_scaling(self):
        """Non-unity slope should scale consensus."""
        coeff = EMOSCoefficients(a=0.0, b=0.9, c=1.0, d=0.0)
        mu, sigma = coeff.predict(consensus_f=80.0)
        assert abs(mu - 72.0) < 1e-10

    def test_sigma_floor_enforced(self):
        """Sigma should never go below floor."""
        coeff = EMOSCoefficients(a=0.0, b=1.0, c=0.0, d=0.0)
        _, sigma = coeff.predict(consensus_f=70.0, spread=0.0)
        assert sigma >= _SIGMA_FLOOR

    def test_serialization_round_trip(self):
        """Coefficients should survive JSON round-trip."""
        coeff = EMOSCoefficients(
            a=0.5, b=0.95, c=1.2, d=0.3,
            market_type="high",
            n_training_samples=50,
            mean_crps=0.85,
        )
        d = coeff.to_dict()
        restored = EMOSCoefficients.from_dict(d)
        assert abs(restored.a - coeff.a) < 1e-6
        assert abs(restored.b - coeff.b) < 1e-6
        assert abs(restored.c - coeff.c) < 1e-6
        assert abs(restored.d - coeff.d) < 1e-6
        assert restored.market_type == "high"
        assert restored.n_training_samples == 50

    def test_no_spread_uses_default(self):
        """When spread is None, sigma should use default spread=1.0."""
        coeff = EMOSCoefficients(a=0.0, b=1.0, c=1.0, d=2.0)
        _, sigma_with = coeff.predict(consensus_f=70.0, spread=1.0)
        _, sigma_without = coeff.predict(consensus_f=70.0, spread=None)
        assert abs(sigma_with - sigma_without) < 1e-10


# ---------------------------------------------------------------------------
# EMOS fitting tests
# ---------------------------------------------------------------------------
def _make_synthetic_samples(
    n: int = 50,
    true_bias: float = 1.5,
    true_noise: float = 1.8,
    market_type: str = "high",
) -> list[EMOSTrainingSample]:
    """Create synthetic EMOS training data.

    Consensus has a bias of `true_bias` and noise of `true_noise`.
    EMOS should learn to correct the bias and estimate the noise level.
    """
    rng = np.random.default_rng(42)
    samples = []
    for i in range(n):
        # Simulate remaining moves (not absolute temps)
        actual_remaining = max(0.0, abs(rng.normal(3.0, 2.0)))
        # Predicted remaining = actual + bias + noise
        predicted_remaining = max(0.0, actual_remaining + true_bias + rng.normal(0, true_noise))
        spread = abs(rng.normal(2.0, 0.5))

        samples.append(
            EMOSTrainingSample(
                target_date=f"2026-03-{(i % 28) + 1:02d}",
                market_type=market_type,
                eval_utc="",
                hours_to_settlement=12.0,
                source_forecasts={"test:model1": predicted_remaining},
                predicted_remaining_f=predicted_remaining,
                ensemble_spread=spread,
                consensus_f=70.0 + predicted_remaining,  # for legacy compat
                consensus_sigma=spread,
                actual_remaining_f=actual_remaining,
                observation_f=70.0 + actual_remaining,  # for legacy compat
            )
        )
    return samples


class TestEMOSFit:
    """Test EMOS coefficient fitting."""

    def test_fit_corrects_bias(self):
        """EMOS should learn to correct systematic bias in remaining moves."""
        samples = _make_synthetic_samples(n=60, true_bias=2.0)
        coeff = fit_emos(samples, "high")

        assert coeff is not None
        # With 2°F positive bias in predicted remaining: predicted=5 → actual≈3
        # EMOS should produce calibrated remaining closer to 3 than to 5
        cal_rem, _ = coeff.predict_remaining(predicted_remaining=5.0)
        assert abs(cal_rem - 3.0) < abs(5.0 - 3.0)

    def test_fit_returns_none_insufficient_data(self):
        """Should return None with too few samples."""
        samples = _make_synthetic_samples(n=10)
        config = EMOSFitConfig(min_training_samples=25)
        coeff = fit_emos(samples, "high", config)
        assert coeff is None

    def test_fit_produces_valid_crps(self):
        """Fitted EMOS should have finite, positive mean CRPS."""
        samples = _make_synthetic_samples(n=60)
        coeff = fit_emos(samples, "high")
        assert coeff is not None
        assert coeff.mean_crps > 0.0
        assert math.isfinite(coeff.mean_crps)

    def test_fit_improves_over_raw_consensus(self):
        """EMOS-calibrated CRPS should beat raw consensus CRPS."""
        samples = _make_synthetic_samples(n=80, true_bias=1.5, true_noise=2.0)
        coeff = fit_emos(samples, "high")
        assert coeff is not None

        # Compute CRPS for raw remaining vs EMOS-calibrated remaining
        raw_crps_vals = []
        emos_crps_vals = []
        for s in samples:
            if s.predicted_remaining_f is None or s.actual_remaining_f is None:
                continue

            # Raw: use predicted remaining as mu
            raw_sigma = max(s.consensus_sigma or 1.0, _SIGMA_FLOOR)
            raw_crps_vals.append(
                crps_normal(s.predicted_remaining_f, raw_sigma, s.actual_remaining_f)
            )

            # EMOS: calibrated remaining
            cal_rem, sigma = coeff.predict_remaining(s.predicted_remaining_f, s.ensemble_spread)
            emos_crps_vals.append(crps_normal(cal_rem, sigma, s.actual_remaining_f))

        raw_mean = sum(raw_crps_vals) / len(raw_crps_vals)
        emos_mean = sum(emos_crps_vals) / len(emos_crps_vals)

        # EMOS should improve (lower CRPS)
        assert emos_mean < raw_mean, (
            f"EMOS CRPS ({emos_mean:.3f}) should be lower than "
            f"raw remaining CRPS ({raw_mean:.3f})"
        )

    def test_fit_respects_market_type_filter(self):
        """Fit should only use samples matching the requested market type."""
        high_samples = _make_synthetic_samples(n=40, market_type="high")
        low_samples = _make_synthetic_samples(n=40, market_type="low")
        all_samples = high_samples + low_samples

        coeff = fit_emos(all_samples, "high")
        assert coeff is not None
        assert coeff.n_training_samples == 40  # only high samples

    def test_fit_metadata_populated(self):
        """Fitted coefficients should have metadata."""
        samples = _make_synthetic_samples(n=50)
        coeff = fit_emos(samples, "high", fit_utc="2026-03-22T00:00:00Z")
        assert coeff is not None
        assert coeff.market_type == "high"
        assert coeff.n_training_samples == 50
        assert coeff.fit_utc == "2026-03-22T00:00:00Z"

    def test_slope_stays_reasonable(self):
        """Slope b should stay within bounds (not flip sign)."""
        samples = _make_synthetic_samples(n=60)
        config = EMOSFitConfig(b_bounds=(0.3, 1.7))
        coeff = fit_emos(samples, "high", config)
        assert coeff is not None
        assert 0.3 <= coeff.b <= 1.7


# ---------------------------------------------------------------------------
# EMOSState tests
# ---------------------------------------------------------------------------
class TestEMOSState:
    """Test EMOS state persistence."""

    def test_save_and_load(self):
        """State should survive save/load cycle."""
        state = EMOSState(
            high=EMOSCoefficients(a=0.5, b=0.95, c=1.2, d=0.3, market_type="high"),
            low=EMOSCoefficients(a=-0.3, b=1.02, c=0.8, d=0.4, market_type="low"),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "emos_state.json"
            state.save(path)
            loaded = EMOSState.load(path)

            assert loaded.high is not None
            assert loaded.low is not None
            assert abs(loaded.high.a - 0.5) < 1e-6
            assert abs(loaded.low.a - (-0.3)) < 1e-6

    def test_load_missing_file(self):
        """Loading non-existent file should return empty state."""
        state = EMOSState.load("/tmp/nonexistent_emos.json")
        assert state.high is None
        assert state.low is None

    def test_calibrate_returns_none_without_coefficients(self):
        """Calibrate should return None if no coefficients loaded."""
        state = EMOSState()
        result = state.calibrate("high", consensus_f=72.0)
        assert result is None

    def test_calibrate_with_coefficients(self):
        """Calibrate should return (mu, sigma) when coefficients exist."""
        state = EMOSState(
            high=EMOSCoefficients(a=0.5, b=0.95, c=1.0, d=0.5),
        )
        result = state.calibrate("high", consensus_f=72.0, spread=2.0)
        assert result is not None
        mu, sigma = result
        expected_mu = 0.5 + 0.95 * 72.0
        assert abs(mu - expected_mu) < 1e-6
        assert sigma > 0.0

    def test_load_corrupt_file(self):
        """Loading corrupt JSON should return empty state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "corrupt.json"
            path.write_text("not valid json{{{")
            state = EMOSState.load(path)
            assert state.high is None
            assert state.low is None
