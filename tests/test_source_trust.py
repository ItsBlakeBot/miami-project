from __future__ import annotations

import json
from datetime import datetime, timezone

from engine.baseline_engine import BaselineEngine, BaselineEngineConfig
from engine.source_registry import ForecastSourceSnapshot
from engine.source_trust import SourceTrustConfig, derive_family_multipliers, load_source_trust_priors

UTC = timezone.utc


def test_derive_family_multipliers_prefers_lower_mae_with_support() -> None:
    summary = {
        "metrics_by_family": {
            "openmeteo": {"n": 500, "mae": 2.5},
            "wethr": {"n": 500, "mae": 5.0},
        }
    }
    priors = derive_family_multipliers(summary, SourceTrustConfig(min_family_samples=80, clip_low=0.7, clip_high=1.3))

    assert priors.global_mae is not None
    assert priors.family_multipliers["openmeteo"] > 1.0
    assert priors.family_multipliers["wethr"] < 1.0


def test_load_source_trust_priors_handles_missing_file(tmp_path) -> None:
    priors = load_source_trust_priors(tmp_path / "does_not_exist.json")
    assert priors.family_multipliers == {}
    assert priors.global_mae is None


def test_load_source_trust_priors_applies_step_cap_and_persists_state(tmp_path) -> None:
    summary_path = tmp_path / "source_trust_backfill.json"
    summary_path.write_text(
        json.dumps(
            {
                "metrics_by_family": {
                    "openmeteo": {"n": 1000, "mae": 1.0},
                    "wethr": {"n": 1000, "mae": 6.0},
                }
            }
        ),
        encoding="utf-8",
    )

    state_path = tmp_path / "state.json"
    state_path.write_text(
        json.dumps({"family_multipliers": {"openmeteo": 1.0, "wethr": 1.0}}),
        encoding="utf-8",
    )

    priors = load_source_trust_priors(
        summary_path,
        cfg=SourceTrustConfig(
            min_family_samples=80,
            clip_low=0.7,
            clip_high=1.3,
            max_step_per_refresh=0.05,
            state_path=str(state_path),
        ),
    )

    assert priors.family_multipliers["openmeteo"] <= 1.05 + 1e-9
    assert priors.family_multipliers["wethr"] >= 0.95 - 1e-9

    persisted = json.loads(state_path.read_text(encoding="utf-8"))
    assert persisted["family_multipliers"]["openmeteo"] == priors.family_multipliers["openmeteo"]


def test_baseline_engine_applies_source_trust_family_multipliers(tmp_path) -> None:
    trust_path = tmp_path / "source_trust_backfill.json"
    trust_path.write_text(
        json.dumps(
            {
                "metrics_by_family": {
                    "openmeteo": {"n": 600, "mae": 2.0},
                    "wethr": {"n": 600, "mae": 5.0},
                }
            }
        ),
        encoding="utf-8",
    )

    engine = BaselineEngine(
        BaselineEngineConfig(
            station="KMIA",
            bias_store_path=str(tmp_path / "bias.json"),
            source_trust_enabled=True,
            source_trust_path=str(trust_path),
            source_trust_state_path=str(tmp_path / "source_trust_state.json"),
            freshness_scale_hours=12.0,
            tracking_scale=6.0,
        )
    )

    now = datetime(2026, 3, 19, 12, 0, tzinfo=UTC)
    snapshots = [
        ForecastSourceSnapshot(
            source_name="gfs",
            family_name="openmeteo",
            issued_at_utc=datetime(2026, 3, 19, 9, 0, tzinfo=UTC),
            target_date="2026-03-19",
            forecast_high_f=82.0,
            quantiles_high={0.1: 80.0, 0.5: 82.0, 0.9: 84.0},
            tracking_error_high=1.0,
        ),
        ForecastSourceSnapshot(
            source_name="ecmwf",
            family_name="wethr",
            issued_at_utc=datetime(2026, 3, 19, 9, 0, tzinfo=UTC),
            target_date="2026-03-19",
            forecast_high_f=82.0,
            quantiles_high={0.1: 80.0, 0.5: 82.0, 0.9: 84.0},
            tracking_error_high=1.0,
        ),
    ]

    belief = engine.build_baseline(
        snapshots,
        market_type="high",
        target_date="2026-03-19",
        eval_time_utc=now,
    )

    assert belief.source_weights["openmeteo:gfs"] > belief.source_weights["wethr:ecmwf"]
    assert any(str(n).startswith("source_trust=") for n in belief.notes)


def test_sparse_window_guardrail_caps_single_source_authority(tmp_path) -> None:
    engine = BaselineEngine(
        BaselineEngineConfig(
            station="KMIA",
            bias_store_path=str(tmp_path / "bias.json"),
            source_trust_enabled=False,
            sparse_fresh_hours=6.0,
            sparse_min_sources=2,
            sparse_min_families=2,
            sparse_max_single_source_weight=0.72,
            sparse_max_model_trust=0.4,
        )
    )

    now = datetime(2026, 3, 19, 12, 0, tzinfo=UTC)
    snapshots = [
        ForecastSourceSnapshot(
            source_name="a",
            family_name="openmeteo",
            issued_at_utc=datetime(2026, 3, 19, 11, 50, tzinfo=UTC),
            target_date="2026-03-19",
            forecast_high_f=82.0,
            quantiles_high={0.1: 81.0, 0.5: 82.0, 0.9: 83.0},
            tracking_error_high=0.2,
        ),
        ForecastSourceSnapshot(
            source_name="b",
            family_name="openmeteo",
            issued_at_utc=datetime(2026, 3, 19, 9, 0, tzinfo=UTC),
            target_date="2026-03-19",
            forecast_high_f=81.8,
            quantiles_high={0.1: 80.8, 0.5: 81.8, 0.9: 82.8},
            tracking_error_high=6.0,
        ),
    ]

    belief = engine.build_baseline(
        snapshots,
        market_type="high",
        target_date="2026-03-19",
        eval_time_utc=now,
    )

    assert max(belief.source_weights.values()) <= 0.72 + 1e-6
    assert belief.model_trust <= 0.4
    assert any("sparse_window_guardrail" in str(n) for n in belief.notes)
