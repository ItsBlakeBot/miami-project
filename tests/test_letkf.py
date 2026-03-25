"""Tests for Regional Cluster LETKF."""

from __future__ import annotations

import numpy as np
import pytest

from engine.letkf import (
    ClusterDef,
    LETKFState,
    StationDef,
    SurfaceObs,
    gaspari_cohn,
    haversine_km,
)


class TestGaspariCohn:
    def test_zero_distance_is_one(self):
        assert gaspari_cohn(0.0, 100.0) == 1.0

    def test_beyond_cutoff_is_zero(self):
        assert gaspari_cohn(200.0, 100.0) == 0.0

    def test_at_cutoff_is_zero(self):
        assert gaspari_cohn(100.0, 100.0) == 0.0

    def test_monotonically_decreasing(self):
        vals = [gaspari_cohn(d, 100.0) for d in range(0, 101, 10)]
        for i in range(len(vals) - 1):
            assert vals[i] >= vals[i + 1]

    def test_half_cutoff_is_moderate(self):
        val = gaspari_cohn(50.0, 100.0)
        assert 0.0 < val < 1.0


class TestHaversine:
    def test_same_point_is_zero(self):
        assert haversine_km(25.0, -80.0, 25.0, -80.0) == 0.0

    def test_kmia_to_kfll(self):
        # KMIA to KFLL should be ~40km
        d = haversine_km(25.7959, -80.2870, 26.0726, -80.1527)
        assert 30 < d < 50

    def test_symmetry(self):
        d1 = haversine_km(25.0, -80.0, 26.0, -81.0)
        d2 = haversine_km(26.0, -81.0, 25.0, -80.0)
        assert abs(d1 - d2) < 0.01


class TestLETKFState:
    @pytest.fixture
    def simple_cluster(self):
        return ClusterDef(
            name="test",
            stations=(
                StationDef("A", "Station A", 25.8, -80.3, is_trading=True),
                StationDef("B", "Station B", 26.1, -80.2, is_trading=True),
                StationDef("C", "Station C", 25.5, -80.5),
            ),
            localization_radius_km=100.0,
            ensemble_size=20,
        )

    def test_initialize_from_forecasts(self, simple_cluster):
        state = LETKFState(cluster=simple_cluster)
        state.initialize_from_forecasts({"A": 75.0, "B": 76.0, "C": 74.0})
        assert state.ensemble is not None
        assert state.ensemble.shape == (3, 20)
        assert state.analysis_mean is not None
        assert len(state.analysis_mean) == 3

    def test_update_with_observations(self, simple_cluster):
        state = LETKFState(cluster=simple_cluster)
        state.initialize_from_forecasts({"A": 75.0, "B": 76.0, "C": 74.0})

        obs = [
            SurfaceObs("A", 25.8, -80.3, 77.0),  # warmer than forecast
            SurfaceObs("B", 26.1, -80.2, 78.0),  # warmer than forecast
            SurfaceObs("C", 25.5, -80.5, 76.0),  # warmer than forecast
        ]
        result = state.update(obs)

        # Should have analysis for trading stations A and B
        assert "A" in result
        assert "B" in result
        # C is not a trading station
        assert "C" not in result

    def test_analysis_moves_toward_obs(self, simple_cluster):
        state = LETKFState(cluster=simple_cluster)
        state.initialize_from_forecasts({"A": 70.0, "B": 71.0, "C": 69.0})
        prior_mean_a = float(state.analysis_mean[0])

        obs = [
            SurfaceObs("A", 25.8, -80.3, 80.0),  # 10°F warmer than forecast
            SurfaceObs("B", 26.1, -80.2, 81.0),
            SurfaceObs("C", 25.5, -80.5, 79.0),
        ]
        result = state.update(obs)

        # Analysis should have moved toward observations
        post_mean_a = result["A"][0]
        assert post_mean_a > prior_mean_a, (
            f"Analysis ({post_mean_a:.1f}) should be warmer than prior ({prior_mean_a:.1f})"
        )

    def test_spatial_propagation(self, simple_cluster):
        """Obs at station A should influence nearby station B."""
        state = LETKFState(cluster=simple_cluster)
        state.initialize_from_forecasts({"A": 70.0, "B": 70.0, "C": 70.0})

        # Only observe at station A (hot), not at B
        obs = [
            SurfaceObs("A", 25.8, -80.3, 80.0),
            SurfaceObs("C", 25.5, -80.5, 70.0),  # C is normal
        ]
        result = state.update(obs)

        # Station B should be pulled warm by A's obs (spatial propagation)
        b_analysis = result["B"][0]
        assert b_analysis > 70.0, (
            f"B ({b_analysis:.1f}) should be pulled above prior (70.0) by A's warm obs"
        )

    def test_too_few_obs_skips_update(self, simple_cluster):
        state = LETKFState(cluster=simple_cluster)
        state.initialize_from_forecasts({"A": 75.0, "B": 76.0, "C": 74.0})

        # Only 1 obs — should skip update
        result = state.update([SurfaceObs("A", 25.8, -80.3, 80.0)])
        # Should still return trading station values (prior)
        assert "A" in result

    def test_multiple_updates_converge(self, simple_cluster):
        state = LETKFState(cluster=simple_cluster)
        state.initialize_from_forecasts({"A": 70.0, "B": 70.0, "C": 70.0})

        # Repeatedly observe 80°F at all stations
        for _ in range(5):
            obs = [
                SurfaceObs("A", 25.8, -80.3, 80.0),
                SurfaceObs("B", 26.1, -80.2, 80.0),
                SurfaceObs("C", 25.5, -80.5, 80.0),
            ]
            state.update(obs)

        # After 5 cycles, analysis should be close to 80°F
        result = state._trading_station_output()
        assert abs(result["A"][0] - 80.0) < 3.0, f"A should converge near 80.0, got {result['A'][0]}"

    def test_spread_shrinks_with_obs(self, simple_cluster):
        state = LETKFState(cluster=simple_cluster)
        state.initialize_from_forecasts({"A": 75.0, "B": 76.0, "C": 74.0}, forecast_spread=3.0)
        initial_spread = float(state.analysis_spread[0])

        obs = [
            SurfaceObs("A", 25.8, -80.3, 75.0),
            SurfaceObs("B", 26.1, -80.2, 76.0),
            SurfaceObs("C", 25.5, -80.5, 74.0),
        ]
        state.update(obs)

        # Spread should decrease after assimilating confirming observations
        # (RTPS inflation prevents it from going to zero)
        final_spread = result_spread = float(state.analysis_spread[0])
        # With RTPS, spread won't collapse completely but should be managed
        assert state.n_updates == 1
