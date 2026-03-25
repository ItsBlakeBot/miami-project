"""Baseline engine for model-first probabilistic belief construction.

This layer sits between raw source snapshots and the residual/regime stack.
It applies slow bias correction, computes per-source weights from freshness and
tracking quality, and combines source quantiles into a single baseline
predictive distribution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .bias_memory import JsonBiasMemory, bias_key
from .dynamic_weights import DynamicWeightContext, DynamicWeightingConfig, compute_dynamic_weight
from .quantile_combiner import PredictiveDistribution, combine_quantiles
from .source_registry import ForecastSourceSnapshot
from .source_trust import SourceTrustConfig, load_source_trust_priors

UTC = timezone.utc


@dataclass(frozen=True)
class BaselineEngineConfig:
    station: str = "KMIA"
    mae_floor: float = 0.75
    freshness_scale_hours: float = 6.0
    tracking_scale: float = 4.0
    min_weight: float = 0.01
    horizon_scale_hours: float = 8.0
    max_urgency_freshness_tightening: float = 0.45
    bias_alpha: float = 0.15
    bias_store_path: str = "analysis_data/bias_memory.json"
    source_trust_enabled: bool = True
    source_trust_path: str = "analysis_data/source_trust_backfill.json"
    source_trust_min_family_samples: int = 80
    source_trust_clip_low: float = 0.7
    source_trust_clip_high: float = 1.3
    source_trust_max_step_per_refresh: float = 0.08
    source_trust_state_path: str | None = "analysis_data/source_trust_state.json"

    # BOA weight overrides: when BOA has enough data, its per-family weights
    # replace the static source trust multipliers. Keyed by family name.
    # None = use source trust (T1.4); dict = use BOA weights (T1.5).
    boa_family_overrides: dict[str, float] | None = None

    # Sparse-window guardrails.
    sparse_fresh_hours: float = 2.5
    sparse_min_sources: int = 2
    sparse_min_families: int = 2
    sparse_max_single_source_weight: float = 0.72
    sparse_max_model_trust: float = 0.4


@dataclass(frozen=True)
class WeightedSourceSnapshot:
    source_key: str
    family_name: str
    issued_at_utc: datetime
    market_value_f: float | None
    tracking_error_f: float | None
    bias_applied_f: float = 0.0
    weight: float = 0.0
    weight_breakdown: dict[str, float | str] = field(default_factory=dict)


@dataclass
class BaselineBelief:
    station: str
    market_type: str
    target_date: str
    distribution: PredictiveDistribution
    model_trust: float
    n_models: int
    freshest_run_utc: str | None = None
    source_weights: dict[str, float] = field(default_factory=dict)
    tracking_errors: dict[str, float] = field(default_factory=dict)
    bias_applied_f: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


class BaselineEngine:
    def __init__(self, config: BaselineEngineConfig | None = None) -> None:
        self.config = config or BaselineEngineConfig()
        self.bias_memory = JsonBiasMemory(Path(self.config.bias_store_path))

        family_multipliers: dict[str, float] = {}
        self.source_trust_global_mae: float | None = None
        self.source_trust_source: str = "disabled"

        if self.config.boa_family_overrides is not None:
            # BOA weights override source trust when available (T1.5 > T1.4)
            family_multipliers = dict(self.config.boa_family_overrides)
            self.source_trust_global_mae = None
            self.source_trust_source = "boa"
        elif self.config.source_trust_enabled:
            priors = load_source_trust_priors(
                self.config.source_trust_path,
                cfg=SourceTrustConfig(
                    min_family_samples=self.config.source_trust_min_family_samples,
                    clip_low=self.config.source_trust_clip_low,
                    clip_high=self.config.source_trust_clip_high,
                    max_step_per_refresh=self.config.source_trust_max_step_per_refresh,
                    state_path=self.config.source_trust_state_path,
                ),
            )
            family_multipliers = dict(priors.family_multipliers)
            self.source_trust_global_mae = priors.global_mae
            self.source_trust_source = priors.source

        self.weighting_cfg = DynamicWeightingConfig(
            freshness_scale_hours=self.config.freshness_scale_hours,
            tracking_scale=self.config.tracking_scale,
            min_weight=self.config.min_weight,
            horizon_scale_hours=self.config.horizon_scale_hours,
            max_urgency_freshness_tightening=self.config.max_urgency_freshness_tightening,
            family_multipliers=family_multipliers,
        )

    def _correct_snapshot(self, snapshot: ForecastSourceSnapshot, market_type: str) -> tuple[ForecastSourceSnapshot, float]:
        value = snapshot.forecast_high_f if market_type == "high" else snapshot.forecast_low_f
        key = bias_key(
            station=self.config.station,
            family=snapshot.family_name,
            market_type=market_type,
        )
        corrected_value = self.bias_memory.corrected(key, value, alpha=self.config.bias_alpha)
        bias_applied = 0.0 if value is None or corrected_value is None else corrected_value - value

        quantiles = snapshot.quantiles_high if market_type == "high" else snapshot.quantiles_low
        corrected_quantiles = {}
        if quantiles:
            for q, v in quantiles.items():
                corrected = self.bias_memory.corrected(key, v, alpha=self.config.bias_alpha)
                if corrected is not None:
                    corrected_quantiles[q] = round(corrected, 3)

        if market_type == "high":
            corrected_snapshot = ForecastSourceSnapshot(
                source_name=snapshot.source_name,
                family_name=snapshot.family_name,
                issued_at_utc=snapshot.issued_at_utc,
                target_date=snapshot.target_date,
                forecast_high_f=corrected_value,
                forecast_low_f=snapshot.forecast_low_f,
                quantiles_high=corrected_quantiles,
                quantiles_low=snapshot.quantiles_low,
                tracking_error_high=snapshot.tracking_error_high,
                tracking_error_low=snapshot.tracking_error_low,
                metadata=dict(snapshot.metadata),
            )
        else:
            corrected_snapshot = ForecastSourceSnapshot(
                source_name=snapshot.source_name,
                family_name=snapshot.family_name,
                issued_at_utc=snapshot.issued_at_utc,
                target_date=snapshot.target_date,
                forecast_high_f=snapshot.forecast_high_f,
                forecast_low_f=corrected_value,
                quantiles_high=snapshot.quantiles_high,
                quantiles_low=corrected_quantiles,
                tracking_error_high=snapshot.tracking_error_high,
                tracking_error_low=snapshot.tracking_error_low,
                metadata=dict(snapshot.metadata),
            )
        return corrected_snapshot, round(bias_applied, 3)

    def _source_weight(
        self,
        snapshot: ForecastSourceSnapshot,
        market_type: str,
        now_utc: datetime,
        *,
        hours_to_settlement: float | None = None,
        regime_confidence: float | None = None,
        break_probability: float | None = None,
    ) -> tuple[float, dict[str, float | str]]:
        tracking_error = snapshot.tracking_error_high if market_type == "high" else snapshot.tracking_error_low
        breakdown = compute_dynamic_weight(
            DynamicWeightContext(
                age_hours=snapshot.age_hours(now_utc),
                tracking_error_f=tracking_error,
                hours_to_settlement=hours_to_settlement,
                regime_confidence=regime_confidence,
                break_probability=break_probability,
                source_family=snapshot.family_name,
            ),
            self.weighting_cfg,
        )
        return breakdown.final_weight, {
            "freshness_weight": breakdown.freshness_weight,
            "tracking_weight": breakdown.tracking_weight,
            "family_multiplier": breakdown.family_multiplier,
            "break_multiplier": breakdown.break_multiplier,
            "regime_multiplier": breakdown.regime_multiplier,
            "effective_freshness_scale_hours": breakdown.effective_freshness_scale_hours,
        }

    def _model_trust(self, weighted: list[WeightedSourceSnapshot]) -> float:
        if not weighted:
            return 0.0
        total = sum(item.weight for item in weighted)
        if total <= 0:
            return 0.0
        freshness_proxy = min(1.0, max(item.weight for item in weighted))
        mean_weight = total / len(weighted)
        trust = 0.55 * freshness_proxy + 0.45 * min(1.0, mean_weight)
        return round(max(0.0, min(1.0, trust)), 3)

    def _cap_single_source_authority(self, weights: dict[str, float], cap: float) -> dict[str, float]:
        if not weights:
            return {}
        total = sum(max(0.0, float(v)) for v in weights.values())
        if total <= 0:
            return dict(weights)

        normalized = {k: max(0.0, float(v)) / total for k, v in weights.items()}
        top_key = max(normalized, key=normalized.get)
        top_val = normalized[top_key]

        cap = max(0.5, min(0.99, float(cap)))
        if top_val <= cap or len(normalized) <= 1:
            return dict(weights)

        overflow = top_val - cap
        normalized[top_key] = cap

        other_total = sum(v for k, v in normalized.items() if k != top_key)
        if other_total <= 1e-9:
            return dict(weights)

        scale = (1.0 - cap) / other_total
        for k in list(normalized.keys()):
            if k == top_key:
                continue
            normalized[k] = normalized[k] * scale

        return {k: round(v * total, 8) for k, v in normalized.items()}

    def build_baseline(
        self,
        snapshots: list[ForecastSourceSnapshot],
        *,
        market_type: str,
        target_date: str,
        eval_time_utc: datetime | None = None,
        hours_to_settlement: float | None = None,
        regime_confidence: float | None = None,
        break_probability: float | None = None,
    ) -> BaselineBelief:
        now_utc = eval_time_utc.astimezone(UTC) if eval_time_utc else datetime.now(tz=UTC)

        corrected: list[ForecastSourceSnapshot] = []
        weighted: list[WeightedSourceSnapshot] = []
        source_weights: dict[str, float] = {}
        tracking_errors: dict[str, float] = {}
        bias_applied_f: dict[str, float] = {}

        for snapshot in snapshots:
            if snapshot.target_date != target_date:
                continue
            corrected_snapshot, bias_applied = self._correct_snapshot(snapshot, market_type)
            weight, weight_breakdown = self._source_weight(
                corrected_snapshot,
                market_type,
                now_utc,
                hours_to_settlement=hours_to_settlement,
                regime_confidence=regime_confidence,
                break_probability=break_probability,
            )
            corrected.append(corrected_snapshot)
            source_weights[corrected_snapshot.source_key] = weight
            bias_applied_f[corrected_snapshot.source_key] = bias_applied
            tracking_error = corrected_snapshot.tracking_error_high if market_type == "high" else corrected_snapshot.tracking_error_low
            if tracking_error is not None:
                tracking_errors[corrected_snapshot.source_key] = round(tracking_error, 3)
            weighted.append(
                WeightedSourceSnapshot(
                    source_key=corrected_snapshot.source_key,
                    family_name=corrected_snapshot.family_name,
                    issued_at_utc=corrected_snapshot.issued_at_utc,
                    market_value_f=corrected_snapshot.forecast_high_f if market_type == "high" else corrected_snapshot.forecast_low_f,
                    tracking_error_f=tracking_error,
                    bias_applied_f=bias_applied,
                    weight=weight,
                    weight_breakdown=weight_breakdown,
                )
            )

        fresh_sources = [
            s for s in corrected
            if s.issued_at_utc is not None and s.age_hours(now_utc) <= max(0.1, self.config.sparse_fresh_hours)
        ]
        fresh_families = {s.family_name for s in fresh_sources}
        sparse_window = (
            len(fresh_sources) < max(1, int(self.config.sparse_min_sources))
            or len(fresh_families) < max(1, int(self.config.sparse_min_families))
        )
        if sparse_window:
            source_weights = self._cap_single_source_authority(
                source_weights,
                cap=self.config.sparse_max_single_source_weight,
            )

        distribution = combine_quantiles(
            corrected,
            market_type=market_type,
            target_date=target_date,
            weights=source_weights,
        )
        freshest = max((s.issued_at_utc for s in corrected), default=None)
        model_trust = self._model_trust(weighted)
        if sparse_window:
            model_trust = min(model_trust, float(self.config.sparse_max_model_trust))

        notes = [
            "baseline built from dynamic freshness/tracking-weighted corrected sources",
            f"n_sources={len(corrected)}",
        ]
        if hours_to_settlement is not None:
            notes.append(f"hours_to_settlement={round(hours_to_settlement, 2)}")
        if regime_confidence is not None:
            notes.append(f"regime_confidence={round(regime_confidence, 3)}")
        if break_probability is not None:
            notes.append(f"break_probability={round(break_probability, 3)}")
        if sparse_window:
            notes.append(
                "sparse_window_guardrail="
                f"fresh_sources={len(fresh_sources)},fresh_families={len(fresh_families)},"
                f"single_source_cap={self.config.sparse_max_single_source_weight},"
                f"max_model_trust={self.config.sparse_max_model_trust}"
            )

        if self.config.source_trust_enabled:
            notes.append(f"source_trust={self.source_trust_source}")
            if self.source_trust_global_mae is not None:
                notes.append(f"source_trust_global_mae={self.source_trust_global_mae}")

        return BaselineBelief(
            station=self.config.station,
            market_type=market_type,
            target_date=target_date,
            distribution=distribution,
            model_trust=model_trust,
            n_models=len(corrected),
            freshest_run_utc=freshest.isoformat() if freshest else None,
            source_weights={k: round(v, 6) for k, v in distribution.source_weights.items()} if distribution.source_weights else source_weights,
            tracking_errors=tracking_errors,
            bias_applied_f=bias_applied_f,
            notes=notes,
        )
