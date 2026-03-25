from __future__ import annotations

from analyzer.canonical_replay_bundle import _to_markdown


def test_to_markdown_includes_key_sections() -> None:
    bundle = {
        "station": "KMIA",
        "date_window": {"start": "2026-03-01", "end": "2026-03-03", "resolved_dates": 3, "dates": []},
        "replay_summary": {
            "brier_all_estimates": 0.12,
            "logloss_all_estimates": 0.45,
            "trade_brier": 0.2,
            "sharpness_mean_abs_p_minus_0_5": 0.3,
            "remaining_target_metrics": {
                "high": {"n": 8, "mae": 1.1, "rmse": 1.4, "crps": 0.8},
                "low": {"n": 8, "mae": 0.9, "rmse": 1.2, "crps": 0.7},
                "crps_method": "quantile-grid approximation",
            },
            "trades": 5,
            "contracts": 9,
            "pnl_cents": 123.0,
            "expected_value_cents_total": 150.0,
            "expected_minus_realized_cents": 27.0,
            "trade_quality_cuts": {
                "by_market_type": [
                    {"market_type": "high", "n": 3, "win_rate": 0.67, "avg_expected_value_cents": 12.0, "avg_realized_pnl_cents": 9.0}
                ]
            },
            "regime_cuts": [
                {"market_type": "high", "regime": "heat", "n": 4, "remaining_mae": 1.0, "remaining_rmse": 1.2, "remaining_crps": 0.7}
            ],
            "worst_day_diagnostics": {
                "highest_brier_day": {"date": "2026-03-02", "brier": 0.3},
                "lowest_realized_pnl_day": {"date": "2026-03-01", "realized_pnl_cents": -10.0},
            },
            "ttl_cuts": {"6-12h": {"n": 10, "brier": 0.2, "logloss": 0.5, "remaining_targets": {"remaining_high_mae": 1.0, "remaining_low_mae": 0.8}}},
        },
        "changepoint_compare": {
            "counts": {"events": 2},
            "bocpd_enabled": {"total_fires": 10},
            "cusum_only": {"total_fires": 12},
            "delta": {"false_positive_proxy": -1},
        },
    }

    md = _to_markdown(bundle)
    assert "# Canonical Replay Bundle" in md
    assert "## Core metrics" in md
    assert "## Remaining-target metrics" in md
    assert "## Trading outcomes" in md
    assert "## Trade-quality cuts" in md
    assert "## Regime cuts" in md
    assert "## Worst-day diagnostics" in md
    assert "## TTL cuts" in md
    assert "## Changepoint compare" in md
