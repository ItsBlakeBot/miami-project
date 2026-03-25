from datetime import datetime, timezone

from engine.baseline_engine import BaselineEngine, BaselineEngineConfig
from engine.source_registry import ForecastSourceSnapshot

UTC = timezone.utc


def test_baseline_engine_combines_corrected_sources(tmp_path):
    engine = BaselineEngine(
        BaselineEngineConfig(
            station="KMIA",
            bias_store_path=str(tmp_path / "bias.json"),
            freshness_scale_hours=12.0,
        )
    )

    snapshots = [
        ForecastSourceSnapshot(
            source_name="nbm",
            family_name="guidance",
            issued_at_utc=datetime(2026, 3, 19, 9, 0, tzinfo=UTC),
            target_date="2026-03-19",
            forecast_high_f=82.0,
            quantiles_high={0.1: 80.0, 0.5: 82.0, 0.9: 84.0},
            tracking_error_high=0.8,
        ),
        ForecastSourceSnapshot(
            source_name="gfs",
            family_name="guidance",
            issued_at_utc=datetime(2026, 3, 19, 6, 0, tzinfo=UTC),
            target_date="2026-03-19",
            forecast_high_f=84.0,
            quantiles_high={0.1: 82.0, 0.5: 84.0, 0.9: 86.0},
            tracking_error_high=1.6,
        ),
    ]

    belief = engine.build_baseline(
        snapshots,
        market_type="high",
        target_date="2026-03-19",
        eval_time_utc=datetime(2026, 3, 19, 10, 0, tzinfo=UTC),
    )

    assert belief.market_type == "high"
    assert belief.n_models == 2
    assert belief.distribution.p50 is not None
    assert belief.model_trust > 0
    assert set(belief.source_weights) == {"guidance:nbm", "guidance:gfs"}
