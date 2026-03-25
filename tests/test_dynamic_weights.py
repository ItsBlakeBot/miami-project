from __future__ import annotations

from engine.dynamic_weights import DynamicWeightContext, DynamicWeightingConfig, compute_dynamic_weight


def test_late_day_tightens_freshness_decay() -> None:
    cfg = DynamicWeightingConfig(freshness_scale_hours=6.0, horizon_scale_hours=8.0)

    early = compute_dynamic_weight(
        DynamicWeightContext(age_hours=3.0, tracking_error_f=1.0, hours_to_settlement=12.0),
        cfg,
    )
    late = compute_dynamic_weight(
        DynamicWeightContext(age_hours=3.0, tracking_error_f=1.0, hours_to_settlement=1.0),
        cfg,
    )

    assert late.effective_freshness_scale_hours < early.effective_freshness_scale_hours
    assert late.freshness_weight < early.freshness_weight
    assert late.final_weight < early.final_weight


def test_break_penalty_applies_for_old_runs_when_break_prob_high() -> None:
    cfg = DynamicWeightingConfig(old_run_hours_threshold=1.5, break_probability_old_run_penalty=0.7)

    base = compute_dynamic_weight(
        DynamicWeightContext(age_hours=2.0, tracking_error_f=1.0, break_probability=0.2),
        cfg,
    )
    shocked = compute_dynamic_weight(
        DynamicWeightContext(age_hours=2.0, tracking_error_f=1.0, break_probability=0.9),
        cfg,
    )

    assert base.break_multiplier == 1.0
    assert shocked.break_multiplier == 0.7
    assert shocked.final_weight < base.final_weight


def test_family_multiplier_applies() -> None:
    cfg = DynamicWeightingConfig(family_multipliers={"nws": 1.2})

    boosted = compute_dynamic_weight(
        DynamicWeightContext(age_hours=1.0, tracking_error_f=1.0, source_family="nws"),
        cfg,
    )
    plain = compute_dynamic_weight(
        DynamicWeightContext(age_hours=1.0, tracking_error_f=1.0, source_family="other"),
        cfg,
    )

    assert boosted.family_multiplier == 1.2
    assert plain.family_multiplier == 1.0
    assert boosted.final_weight > plain.final_weight
