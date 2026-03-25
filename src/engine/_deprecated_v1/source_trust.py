"""Source/family trust priors derived from historical backfill artifacts.

Phase mapping:
- T0.4 source-trust backfill bootstrap (artifact consumption)
- T1.4 adaptive source trust multipliers
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SourceTrustConfig:
    min_family_samples: int = 80
    clip_low: float = 0.7
    clip_high: float = 1.3
    max_step_per_refresh: float = 0.08
    state_path: str | None = "analysis_data/source_trust_state.json"


@dataclass(frozen=True)
class SourceTrustPriors:
    family_multipliers: dict[str, float]
    global_mae: float | None
    source: str


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def derive_family_multipliers(summary: dict, cfg: SourceTrustConfig | None = None) -> SourceTrustPriors:
    cfg = cfg or SourceTrustConfig()

    family_metrics = summary.get("metrics_by_family") or {}
    if not isinstance(family_metrics, dict) or not family_metrics:
        return SourceTrustPriors(family_multipliers={}, global_mae=None, source="empty")

    weighted_abs_sum = 0.0
    weighted_n = 0
    for metrics in family_metrics.values():
        try:
            n = int(metrics.get("n") or 0)
            mae = metrics.get("mae")
            if mae is None or n <= 0:
                continue
            weighted_abs_sum += float(mae) * n
            weighted_n += n
        except Exception:
            continue

    global_mae = (weighted_abs_sum / weighted_n) if weighted_n > 0 else None
    if global_mae is None or global_mae <= 1e-9:
        return SourceTrustPriors(family_multipliers={}, global_mae=None, source="invalid")

    multipliers: dict[str, float] = {}

    for family, metrics in family_metrics.items():
        try:
            n = int(metrics.get("n") or 0)
            mae = metrics.get("mae")
            if mae is None or float(mae) <= 1e-9:
                multipliers[family] = 1.0
                continue

            # Better-than-global MAE -> >1 multiplier. Worse -> <1.
            raw = float(global_mae) / float(mae)

            # Shrink toward 1.0 for sparse families.
            support = _clamp(n / max(1.0, float(cfg.min_family_samples * 3)), 0.0, 1.0)
            shrunk = 1.0 + (raw - 1.0) * support

            multipliers[family] = round(_clamp(shrunk, cfg.clip_low, cfg.clip_high), 6)
        except Exception:
            multipliers[family] = 1.0

    return SourceTrustPriors(
        family_multipliers=multipliers,
        global_mae=round(float(global_mae), 6),
        source="summary",
    )


def _load_state_multipliers(path: str | None) -> dict[str, float]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
    vals = raw.get("family_multipliers") if isinstance(raw, dict) else None
    if not isinstance(vals, dict):
        return {}
    out: dict[str, float] = {}
    for k, v in vals.items():
        try:
            out[str(k)] = float(v)
        except Exception:
            continue
    return out


def _persist_state_multipliers(path: str | None, multipliers: dict[str, float]) -> None:
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "family_multipliers": {k: round(float(v), 6) for k, v in multipliers.items()},
    }
    p.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _apply_step_cap(
    multipliers: dict[str, float],
    prev: dict[str, float],
    max_step: float,
    clip_low: float,
    clip_high: float,
) -> dict[str, float]:
    step = max(0.0, float(max_step))
    if step <= 0:
        return {k: round(_clamp(float(v), clip_low, clip_high), 6) for k, v in multipliers.items()}

    out: dict[str, float] = {}
    for family, value in multipliers.items():
        target = float(value)
        base = float(prev.get(family, 1.0))
        lo = base - step
        hi = base + step
        capped = _clamp(target, lo, hi)
        out[family] = round(_clamp(capped, clip_low, clip_high), 6)
    return out


def load_source_trust_priors(path: str | Path, cfg: SourceTrustConfig | None = None) -> SourceTrustPriors:
    cfg = cfg or SourceTrustConfig()

    p = Path(path)
    if not p.exists():
        return SourceTrustPriors(family_multipliers={}, global_mae=None, source="missing")

    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return SourceTrustPriors(family_multipliers={}, global_mae=None, source="unparseable")

    priors = derive_family_multipliers(raw, cfg=cfg)

    prev = _load_state_multipliers(cfg.state_path)
    capped = _apply_step_cap(
        priors.family_multipliers,
        prev,
        max_step=cfg.max_step_per_refresh,
        clip_low=cfg.clip_low,
        clip_high=cfg.clip_high,
    )
    _persist_state_multipliers(cfg.state_path, capped)

    src = str(p)
    if cfg.state_path:
        src = f"{src}|state={cfg.state_path}"

    return SourceTrustPriors(
        family_multipliers=capped,
        global_mae=priors.global_mae,
        source=src,
    )
