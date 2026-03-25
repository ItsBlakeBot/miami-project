"""Tests for BOA (Bernstein Online Aggregation) with sleeping experts."""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import pytest

from engine.boa import (
    BOAConfig,
    BOAManager,
    BOAState,
    SourceState,
)


# ---------------------------------------------------------------------------
# SourceState tests
# ---------------------------------------------------------------------------
class TestSourceState:
    def test_serialization_round_trip(self):
        state = SourceState(
            source_key="openmeteo:GFS-Global",
            weight=0.85,
            cumulative_loss=12.5,
            cumulative_squared_loss=25.3,
            n_updates=10,
            last_crps=1.1,
        )
        d = state.to_dict()
        restored = SourceState.from_dict(d)
        assert restored.source_key == "openmeteo:GFS-Global"
        assert abs(restored.weight - 0.85) < 1e-6
        assert restored.n_updates == 10


# ---------------------------------------------------------------------------
# BOAState tests
# ---------------------------------------------------------------------------
class TestBOAState:
    def test_empty_weights_returns_empty(self):
        state = BOAState(market_type="high")
        assert state.get_weights() == {}

    def test_single_source_gets_weight_1(self):
        state = BOAState(market_type="high")
        state.sources["a"] = SourceState(source_key="a", weight=1.0)
        weights = state.get_weights()
        assert abs(weights["a"] - 1.0) < 1e-10

    def test_equal_sources_equal_weights(self):
        state = BOAState(market_type="high")
        state.sources["a"] = SourceState(source_key="a", weight=1.0)
        state.sources["b"] = SourceState(source_key="b", weight=1.0)
        weights = state.get_weights()
        assert abs(weights["a"] - 0.5) < 1e-10
        assert abs(weights["b"] - 0.5) < 1e-10

    def test_weights_sum_to_one(self):
        state = BOAState(market_type="high")
        state.sources["a"] = SourceState(source_key="a", weight=2.0)
        state.sources["b"] = SourceState(source_key="b", weight=1.0)
        state.sources["c"] = SourceState(source_key="c", weight=0.5)
        weights = state.get_weights()
        assert abs(sum(weights.values()) - 1.0) < 1e-10

    def test_update_increases_good_source_weight(self):
        """A source with lower CRPS should gain weight over time."""
        config = BOAConfig(learning_rate=0.5, forget_rate=0.0)
        state = BOAState(market_type="high", config=config)

        # Simulate 10 rounds: source A is accurate, source B is biased
        for i in range(10):
            true_temp = 75.0
            forecasts = {
                "good_source": (75.0 + 0.5, 1.5),  # slightly off but close
                "bad_source": (78.0, 1.5),  # consistently 3°F too high
            }
            state.update(forecasts, true_temp)

        weights = state.get_weights()
        assert weights["good_source"] > weights["bad_source"], (
            f"Good source ({weights['good_source']:.4f}) should have higher "
            f"weight than bad source ({weights['bad_source']:.4f})"
        )

    def test_update_tracks_cumulative_loss(self):
        state = BOAState(market_type="high")
        forecasts = {"a": (70.0, 2.0), "b": (72.0, 2.0)}
        state.update(forecasts, 71.0)

        assert state.sources["a"].n_updates == 1
        assert state.sources["b"].n_updates == 1
        assert state.sources["a"].cumulative_loss > 0
        assert state.sources["b"].cumulative_loss > 0

    def test_new_sources_auto_initialized(self):
        """Sources not seen before should be auto-created on first update."""
        state = BOAState(market_type="high")
        forecasts = {"new_source": (70.0, 2.0)}
        state.update(forecasts, 71.0)
        assert "new_source" in state.sources
        assert state.sources["new_source"].n_updates == 1

    def test_forget_rate_decays_history(self):
        """Non-zero forget rate should reduce impact of old losses."""
        config = BOAConfig(forget_rate=0.1)
        state = BOAState(market_type="high", config=config)

        # First update: bad forecast
        state.update({"a": (80.0, 2.0)}, 70.0)
        loss_after_1 = state.sources["a"].cumulative_loss

        # Second update: also bad, but some of the first loss is forgotten
        state.update({"a": (80.0, 2.0)}, 70.0)
        loss_after_2 = state.sources["a"].cumulative_loss

        # With forget_rate=0.1, cumulative should be less than 2x the single loss
        # because the first loss was decayed by 10%
        assert loss_after_2 < loss_after_1 * 2.0

    def test_serialization_round_trip(self):
        config = BOAConfig(learning_rate=0.3, forget_rate=0.05)
        state = BOAState(market_type="low", config=config)
        state.update({"a": (70.0, 2.0), "b": (72.0, 1.5)}, 71.0)

        d = state.to_dict()
        restored = BOAState.from_dict(d)

        assert restored.market_type == "low"
        assert restored.total_updates == 1
        assert abs(restored.config.learning_rate - 0.3) < 1e-10
        assert "a" in restored.sources
        assert "b" in restored.sources


# ---------------------------------------------------------------------------
# BOAManager tests
# ---------------------------------------------------------------------------
class TestBOAManager:
    def test_save_and_load(self):
        manager = BOAManager()
        manager.update("high", {"src1": (72.0, 2.0)}, 73.0)
        manager.update("low", {"src1": (65.0, 2.0)}, 64.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "boa_state.json"
            manager.save(path)
            loaded = BOAManager.load(path)

            assert loaded.high.total_updates == 1
            assert loaded.low.total_updates == 1
            assert "src1" in loaded.high.sources

    def test_load_missing_file(self):
        manager = BOAManager.load("/tmp/nonexistent_boa.json")
        assert manager.high.total_updates == 0
        assert manager.low.total_updates == 0

    def test_load_corrupt_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "corrupt.json"
            path.write_text("not valid json{{{")
            manager = BOAManager.load(path)
            assert manager.high.total_updates == 0

    def test_get_weights_empty(self):
        manager = BOAManager()
        assert manager.get_weights("high") == {}

    def test_summary(self):
        manager = BOAManager()
        manager.update("high", {"a": (72.0, 2.0), "b": (71.0, 2.0)}, 71.5)
        summary = manager.summary()
        assert summary["high"]["n_sources"] == 2
        assert summary["high"]["total_updates"] == 1

    def test_multi_day_convergence(self):
        """Over many updates, better source should dominate weights."""
        manager = BOAManager()
        config = BOAConfig(learning_rate=0.3, forget_rate=0.0)
        manager.high.config = config

        import random
        random.seed(42)

        for _ in range(30):
            true_temp = 75.0 + random.gauss(0, 3)
            forecasts = {
                "accurate": (true_temp + random.gauss(0, 1.0), 1.5),
                "biased": (true_temp + 3.0 + random.gauss(0, 1.0), 1.5),
                "noisy": (true_temp + random.gauss(0, 4.0), 3.0),
            }
            manager.update("high", forecasts, true_temp)

        weights = manager.get_weights("high")
        # Accurate source should have highest weight
        assert weights["accurate"] > weights["biased"]
        assert weights["accurate"] > weights["noisy"]

    def test_adaptation_to_regime_change(self):
        """BOA should adapt when source quality changes mid-stream."""
        config = BOAConfig(learning_rate=0.5, forget_rate=0.05)
        manager = BOAManager(
            high=BOAState(market_type="high", config=config),
        )

        import random
        random.seed(123)

        # Phase 1: source A is better (20 updates)
        for _ in range(20):
            true_temp = 75.0
            manager.update(
                "high",
                {
                    "A": (true_temp + random.gauss(0, 0.5), 1.0),
                    "B": (true_temp + 3.0, 1.0),
                },
                true_temp,
            )

        w_phase1 = manager.get_weights("high")
        assert w_phase1["A"] > w_phase1["B"]

        # Phase 2: source B becomes better, A degrades (20 updates)
        for _ in range(20):
            true_temp = 75.0
            manager.update(
                "high",
                {
                    "A": (true_temp + 3.0, 1.0),
                    "B": (true_temp + random.gauss(0, 0.5), 1.0),
                },
                true_temp,
            )

        w_phase2 = manager.get_weights("high")
        # B should now have higher weight than A
        assert w_phase2["B"] > w_phase2["A"], (
            f"After regime change, B ({w_phase2['B']:.4f}) should dominate "
            f"A ({w_phase2['A']:.4f})"
        )


# ---------------------------------------------------------------------------
# Sleeping expert tests
# ---------------------------------------------------------------------------
class TestSleepingExperts:
    """Test that sleeping expert protocol handles heterogeneous update frequencies."""

    def test_sleeping_source_weight_unchanged(self):
        """A sleeping source's weight should not change when it has no new forecast."""
        config = BOAConfig(learning_rate=0.5, forget_rate=0.0)
        state = BOAState(market_type="high", config=config)

        # Round 1: both awake
        state.update(
            {"hrrr": (72.0, 2.0), "gfs": (71.0, 2.0)},
            71.5,
            awake_sources={"hrrr", "gfs"},
        )
        w1_gfs = state.get_weights()["gfs"]
        gfs_loss_1 = state.sources["gfs"].cumulative_loss

        # Round 2: only HRRR awake (GFS hasn't refreshed)
        state.update(
            {"hrrr": (73.0, 2.0), "gfs": (71.0, 2.0)},
            72.0,
            awake_sources={"hrrr"},
        )
        gfs_loss_2 = state.sources["gfs"].cumulative_loss

        # GFS cumulative loss should NOT have increased (it was sleeping)
        assert gfs_loss_2 == gfs_loss_1

    def test_sleeping_source_n_updates_not_incremented(self):
        """Sleeping source's n_updates should not increase."""
        state = BOAState(market_type="high")

        # Round 1: both awake
        state.update(
            {"hrrr": (72.0, 2.0), "gfs": (71.0, 2.0)},
            71.5,
            awake_sources={"hrrr", "gfs"},
        )
        assert state.sources["gfs"].n_updates == 1

        # Round 2: only HRRR awake
        state.update(
            {"hrrr": (73.0, 2.0), "gfs": (71.0, 2.0)},
            72.0,
            awake_sources={"hrrr"},
        )
        assert state.sources["gfs"].n_updates == 1  # unchanged
        assert state.sources["hrrr"].n_updates == 2  # incremented

    def test_frequent_updater_no_unfair_advantage(self):
        """A model updating 6x more often should NOT dominate if equally skilled.

        Both models have the same forecast quality (same CRPS when awake).
        HRRR updates every round, GFS every 6th round.
        If BOA is fair, both should have similar weights.
        """
        import random
        random.seed(42)

        config = BOAConfig(learning_rate=0.3, forget_rate=0.0)
        state = BOAState(market_type="high", config=config)

        for i in range(60):
            true_temp = 75.0 + random.gauss(0, 2)
            # Both models are equally accurate when they have fresh data
            hrrr_fcst = true_temp + random.gauss(0, 1.5)
            gfs_fcst = true_temp + random.gauss(0, 1.5)

            forecasts = {"hrrr": (hrrr_fcst, 2.0), "gfs": (gfs_fcst, 2.0)}

            # GFS only awake every 6th round
            if i % 6 == 0:
                awake = {"hrrr", "gfs"}
            else:
                awake = {"hrrr"}

            state.update(forecasts, true_temp, awake_sources=awake)

        weights = state.get_weights()
        # With equal skill, weights should be roughly similar
        # (not HRRR dominating 6:1 due to update frequency)
        ratio = weights["hrrr"] / weights["gfs"] if weights["gfs"] > 0 else float("inf")
        assert ratio < 3.0, (
            f"HRRR/GFS weight ratio ({ratio:.1f}) should be <3.0 for "
            f"equally-skilled models. hrrr={weights['hrrr']:.4f}, gfs={weights['gfs']:.4f}"
        )

    def test_genuinely_better_model_still_wins(self):
        """A model that is genuinely better should still accumulate more weight."""
        import random
        random.seed(99)

        config = BOAConfig(learning_rate=0.3, forget_rate=0.0)
        state = BOAState(market_type="high", config=config)

        for i in range(60):
            true_temp = 75.0
            # Good model: low error. Bad model: 3°F bias.
            # Both update at the same frequency to isolate skill effect.
            forecasts = {
                "good": (true_temp + random.gauss(0, 0.5), 1.0),
                "bad": (true_temp + 3.0 + random.gauss(0, 0.5), 1.0),
            }
            state.update(forecasts, true_temp, awake_sources={"good", "bad"})

        weights = state.get_weights()
        assert weights["good"] > weights["bad"]

    def test_none_awake_sources_means_all_awake(self):
        """Passing awake_sources=None should treat all sources as awake (backward compat)."""
        state = BOAState(market_type="high")
        state.update({"a": (70.0, 2.0), "b": (72.0, 2.0)}, 71.0, awake_sources=None)
        assert state.sources["a"].n_updates == 1
        assert state.sources["b"].n_updates == 1

    def test_round_count_tracks_all_rounds(self):
        """n_rounds_total should increment for all sources, including sleeping ones."""
        state = BOAState(market_type="high")

        # Round 1: both awake
        state.update({"a": (70.0, 2.0), "b": (72.0, 2.0)}, 71.0, awake_sources={"a", "b"})
        # Round 2: only a awake
        state.update({"a": (70.0, 2.0), "b": (72.0, 2.0)}, 71.0, awake_sources={"a"})

        assert state.sources["a"].n_rounds_total == 2
        assert state.sources["b"].n_rounds_total == 2
        assert state.sources["a"].n_updates == 2
        assert state.sources["b"].n_updates == 1  # slept through round 2
