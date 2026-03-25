"""Autonomous regime discovery from the DIMMPF particle filter.

DS3M auto-discovers, auto-activates, and auto-tunes regimes. The ONLY
human intervention required is assigning a meaningful name to newly
discovered regimes (they activate immediately with an auto-generated
placeholder name like "regime_7_hot_humid").

Discovery flow:
  1. Each DS3M cycle logs avg_log_likelihood across particles
  2. If avg_log_lik < threshold for N consecutive cycles → regime gap
  3. Snapshot the current atmospheric context
  4. AUTO-ACTIVATE a new regime with conservative parameters
  5. Log for human to assign a proper name later
  6. Auto-tune the new regime's parameters from subsequent data

Auto-tuning (runs every cycle):
  - Transition matrix: updated from soft regime counts (MLE + Dirichlet)
  - Per-regime drift/sigma: updated from observation innovations
  - Observation R: Sage-Husa adaptive estimation
  - Regime merge: if two regimes consistently co-occur with similar
    dynamics, propose merge (logged, auto-executed after 7 days of evidence)
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .config import DS3MConfig

log = logging.getLogger(__name__)

REGIME_GAPS_PATH = Path("analysis_data/ds3m_regime_gaps.json")


@dataclass
class DiscoveredRegime:
    """A regime discovered and auto-activated by DS3M."""

    regime_index: int  # index in the dynamics arrays
    auto_name: str  # placeholder name (e.g. "regime_7_hot_humid")
    human_name: str | None = None  # assigned by human later
    detected_utc: str = ""
    target_date: str = ""
    station: str = "KMIA"

    # Detection context
    consecutive_low_lik_cycles: int = 0
    avg_log_likelihood: float = 0.0
    atmospheric_snapshot: dict = field(default_factory=dict)

    # Parameters at activation (auto-tuned afterward)
    initial_sigma_multiplier: float = 1.3
    initial_mu_bias_high_f: float = 0.0
    initial_mu_bias_low_f: float = 0.0

    # Performance tracking since activation
    n_cycles_active: int = 0  # how many cycles this regime was dominant
    avg_innovation_high: float = 0.0
    avg_innovation_low: float = 0.0
    activation_status: str = "active"  # active, merged, retired


@dataclass
class RegimeDiscoveryState:
    """Tracks low-likelihood streaks and discovered regimes."""

    consecutive_low_cycles: int = 0
    recent_log_likelihoods: list[float] = field(default_factory=list)
    max_recent: int = 20

    # All discovered regimes (including merged/retired)
    discovered_regimes: list[dict] = field(default_factory=list)

    # Regime co-occurrence tracking for merge detection
    # Key: "i,j" → count of consecutive cycles where both had >20% posterior
    co_occurrence_counts: dict[str, int] = field(default_factory=dict)
    merge_threshold_cycles: int = 100  # auto-merge after this many co-occurrences

    def to_dict(self) -> dict:
        return {
            "consecutive_low_cycles": self.consecutive_low_cycles,
            "recent_log_likelihoods": self.recent_log_likelihoods[-self.max_recent:],
            "discovered_regimes": self.discovered_regimes,
            "co_occurrence_counts": self.co_occurrence_counts,
        }

    @classmethod
    def from_dict(cls, d: dict) -> RegimeDiscoveryState:
        state = cls()
        state.consecutive_low_cycles = d.get("consecutive_low_cycles", 0)
        state.recent_log_likelihoods = d.get("recent_log_likelihoods", [])
        state.discovered_regimes = d.get("discovered_regimes", [])
        state.co_occurrence_counts = d.get("co_occurrence_counts", {})
        return state


def check_for_regime_gap(
    discovery_state: RegimeDiscoveryState,
    avg_log_likelihood: float,
    regime_posterior: np.ndarray,
    regime_names: list[str],
    config: DS3MConfig,
    *,
    station: str = "KMIA",
    target_date: str = "",
    atmospheric_context: dict | None = None,
) -> dict | None:
    """Check for regime gap and auto-activate a new regime if detected.

    Returns a dict describing the new regime if one was created, else None.
    The returned dict contains the parameters to pass to
    regime_dynamics.add_regime().
    """
    discovery_state.recent_log_likelihoods.append(avg_log_likelihood)
    if len(discovery_state.recent_log_likelihoods) > discovery_state.max_recent:
        discovery_state.recent_log_likelihoods = discovery_state.recent_log_likelihoods[
            -discovery_state.max_recent:
        ]

    # Check if likelihood is below threshold
    if avg_log_likelihood < config.likelihood_gap_threshold:
        discovery_state.consecutive_low_cycles += 1
    else:
        discovery_state.consecutive_low_cycles = 0
        return None

    # Not enough consecutive cycles yet
    if discovery_state.consecutive_low_cycles < config.gap_consecutive_cycles:
        return None

    # Already at max regimes
    current_k = len(regime_names)
    if current_k >= config.max_regimes:
        log.warning("Regime gap detected but at max_regimes=%d", config.max_regimes)
        discovery_state.consecutive_low_cycles = 0
        return None

    # Auto-generate name
    auto_name = _auto_name(current_k, atmospheric_context)
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    gap_liks = discovery_state.recent_log_likelihoods[-config.gap_consecutive_cycles:]
    avg_lik = float(np.mean(gap_liks))

    # Estimate initial parameters from the atmospheric context
    mu_high, mu_low, sigma_mult = _estimate_initial_params(atmospheric_context)

    discovered = DiscoveredRegime(
        regime_index=current_k,
        auto_name=auto_name,
        detected_utc=now_utc,
        target_date=target_date,
        station=station,
        consecutive_low_lik_cycles=discovery_state.consecutive_low_cycles,
        avg_log_likelihood=round(avg_lik, 3),
        atmospheric_snapshot=atmospheric_context or {},
        initial_sigma_multiplier=sigma_mult,
        initial_mu_bias_high_f=mu_high,
        initial_mu_bias_low_f=mu_low,
    )

    discovery_state.discovered_regimes.append(asdict(discovered))
    discovery_state.consecutive_low_cycles = 0

    log.info(
        "DS3M AUTO-ACTIVATED regime '%s' (K=%d→%d, avg_lik=%.2f). "
        "Human should assign a name when convenient.",
        auto_name, current_k, current_k + 1, avg_lik,
    )

    # Persist for human naming
    _save_discoveries(discovery_state.discovered_regimes)

    return {
        "name": auto_name,
        "drift_high": mu_high,
        "sigma_high": sigma_mult,
        "drift_low": mu_low,
        "sigma_low": sigma_mult,
    }


def check_for_regime_merge(
    discovery_state: RegimeDiscoveryState,
    regime_posterior: np.ndarray,
    regime_names: list[str],
    dynamics_dict: dict,
) -> dict | None:
    """Check if two regimes should be merged (consistently co-occurring).

    Auto-merges after merge_threshold_cycles of co-occurrence.
    Returns updated dynamics_dict if merge happened, else None.
    """
    k = len(regime_posterior)
    if k <= 3:
        return None  # don't merge below 3 regimes

    # Track co-occurrence: two regimes both above 20% posterior
    active_regimes = [i for i in range(k) if regime_posterior[i] > 0.20]
    if len(active_regimes) >= 2:
        for i in range(len(active_regimes)):
            for j in range(i + 1, len(active_regimes)):
                key = f"{active_regimes[i]},{active_regimes[j]}"
                discovery_state.co_occurrence_counts[key] = (
                    discovery_state.co_occurrence_counts.get(key, 0) + 1
                )
    else:
        # Reset co-occurrence when regimes are distinct
        discovery_state.co_occurrence_counts = {
            k: v for k, v in discovery_state.co_occurrence_counts.items()
            if v > discovery_state.merge_threshold_cycles // 2
        }

    # Check for merge candidates
    for key, count in list(discovery_state.co_occurrence_counts.items()):
        if count < discovery_state.merge_threshold_cycles:
            continue

        i, j = map(int, key.split(","))
        if i >= k or j >= k:
            continue

        # Check if dynamics are similar enough to merge
        drift_h = dynamics_dict.get("drift_high", [])
        sigma_h = dynamics_dict.get("sigma_high", [])
        if i < len(drift_h) and j < len(drift_h):
            drift_diff = abs(drift_h[i] - drift_h[j])
            sigma_diff = abs(sigma_h[i] - sigma_h[j])
            if drift_diff > 0.5 or sigma_diff > 0.5:
                # Dynamics are different — don't merge despite co-occurrence
                continue

        # Auto-merge: keep regime i, remove regime j
        log.info(
            "DS3M AUTO-MERGING regime '%s' into '%s' after %d co-occurrence cycles",
            regime_names[j] if j < len(regime_names) else f"regime_{j}",
            regime_names[i] if i < len(regime_names) else f"regime_{i}",
            count,
        )

        dynamics_dict = _merge_regimes(dynamics_dict, keep=i, remove=j)
        del discovery_state.co_occurrence_counts[key]
        return dynamics_dict

    return None


def auto_tune_regime_params(
    dynamics_dict: dict,
    regime_posterior: np.ndarray,
    innovation_high: float | None,
    innovation_low: float | None,
    alpha: float = 0.05,
) -> dict:
    """Online auto-tuning of per-regime drift and sigma from innovations.

    Called every DS3M cycle. Uses regime-weighted EMA to smoothly
    adapt parameters without human intervention.

    Args:
        dynamics_dict: current dynamics as dict
        regime_posterior: (K,) posterior from current cycle
        innovation_high: obs_remaining - predicted_remaining for HIGH
        innovation_low: same for LOW
        alpha: EMA learning rate (0.05 = 5% new info per cycle)
    """
    drift_high = dynamics_dict.get("drift_high", [])
    drift_low = dynamics_dict.get("drift_low", [])
    sigma_high = dynamics_dict.get("sigma_high", [])
    sigma_low = dynamics_dict.get("sigma_low", [])
    k = len(drift_high)

    for i in range(k):
        w = float(regime_posterior[i]) if i < len(regime_posterior) else 0.0
        if w < 0.05:
            continue  # regime not active enough to learn from

        effective_alpha = alpha * w  # scale learning by regime confidence

        if innovation_high is not None and i < len(drift_high):
            # Drift: EMA toward observed innovation
            drift_high[i] += effective_alpha * (innovation_high - drift_high[i])
            # Sigma: EMA toward observed |innovation|
            obs_sigma = abs(innovation_high)
            sigma_high[i] += effective_alpha * (obs_sigma - sigma_high[i])
            sigma_high[i] = max(0.25, min(5.0, sigma_high[i]))  # safety bounds

        if innovation_low is not None and i < len(drift_low):
            drift_low[i] += effective_alpha * (innovation_low - drift_low[i])
            obs_sigma = abs(innovation_low)
            sigma_low[i] += effective_alpha * (obs_sigma - sigma_low[i])
            sigma_low[i] = max(0.25, min(5.0, sigma_low[i]))

    dynamics_dict["drift_high"] = drift_high
    dynamics_dict["drift_low"] = drift_low
    dynamics_dict["sigma_high"] = sigma_high
    dynamics_dict["sigma_low"] = sigma_low
    return dynamics_dict


def _estimate_initial_params(ctx: dict | None) -> tuple[float, float, float]:
    """Estimate initial mu_bias and sigma for a new regime from atmos context."""
    if not ctx:
        return 0.0, 0.0, 1.3

    mu_high = 0.0
    mu_low = 0.0
    sigma = 1.3

    cape = ctx.get("cape")
    if cape is not None and cape > 1500:
        sigma = 1.6
        mu_low = -1.5  # convective outflow potential

    cloud = ctx.get("cloud_fraction")
    if cloud is not None and cloud > 0.7:
        sigma = 0.9
        mu_high = -1.0  # cloud suppression
        mu_low = 0.5

    wind = ctx.get("wind_speed_mph")
    wind_dir = ctx.get("wind_dir_deg")
    if wind_dir is not None and (wind_dir > 270 or wind_dir < 30):
        sigma = 1.4
        mu_low = -1.0  # continental air = colder lows

    return round(mu_high, 2), round(mu_low, 2), round(sigma, 2)


def _auto_name(k_index: int, ctx: dict | None) -> str:
    """Generate placeholder name: regime_{index}_{descriptor}."""
    descriptor = "unknown"
    if ctx:
        parts = []
        temp = ctx.get("temperature_f")
        if temp is not None:
            parts.append("hot" if temp > 85 else "cool" if temp < 65 else "mild")
        dew = ctx.get("dew_point_f")
        if dew is not None:
            parts.append("humid" if dew > 72 else "dry" if dew < 55 else "moderate")
        wind = ctx.get("wind_speed_mph")
        if wind is not None and wind > 15:
            parts.append("windy")
        cape = ctx.get("cape")
        if cape is not None and cape > 1000:
            parts.append("unstable")
        if parts:
            descriptor = "_".join(parts[:3])

    return f"regime_{k_index}_{descriptor}"


def _merge_regimes(dynamics_dict: dict, keep: int, remove: int) -> dict:
    """Merge regime `remove` into regime `keep`."""
    k = dynamics_dict.get("k_regimes", 0)
    if remove >= k or keep >= k:
        return dynamics_dict

    names = dynamics_dict.get("regime_names", [])
    trans = dynamics_dict.get("transition_matrix", [])
    drift_h = dynamics_dict.get("drift_high", [])
    drift_l = dynamics_dict.get("drift_low", [])
    sigma_h = dynamics_dict.get("sigma_high", [])
    sigma_l = dynamics_dict.get("sigma_low", [])

    # Average parameters of kept and removed
    if keep < len(drift_h) and remove < len(drift_h):
        drift_h[keep] = (drift_h[keep] + drift_h[remove]) / 2
        drift_l[keep] = (drift_l[keep] + drift_l[remove]) / 2
        sigma_h[keep] = (sigma_h[keep] + sigma_h[remove]) / 2
        sigma_l[keep] = (sigma_l[keep] + sigma_l[remove]) / 2

    # Remove regime from all arrays
    for arr in [names, drift_h, drift_l, sigma_h, sigma_l]:
        if remove < len(arr):
            arr.pop(remove)

    # Remove row and column from transition matrix
    if remove < len(trans):
        trans.pop(remove)
        for row in trans:
            if remove < len(row):
                row.pop(remove)
        # Renormalize rows
        for row in trans:
            total = sum(row)
            if total > 0:
                for i in range(len(row)):
                    row[i] /= total

    dynamics_dict.update({
        "k_regimes": k - 1,
        "regime_names": names,
        "transition_matrix": trans,
        "drift_high": drift_h,
        "drift_low": drift_l,
        "sigma_high": sigma_h,
        "sigma_low": sigma_l,
    })
    return dynamics_dict


def _save_discoveries(discoveries: list[dict]) -> None:
    """Write discovered regimes to JSON for human naming."""
    try:
        REGIME_GAPS_PATH.parent.mkdir(parents=True, exist_ok=True)
        REGIME_GAPS_PATH.write_text(json.dumps(discoveries, indent=2))
    except Exception:
        log.exception("Failed to save regime discoveries")


def get_unnamed_regimes(discovery_state: RegimeDiscoveryState) -> list[dict]:
    """Return discovered regimes that haven't been named by a human yet."""
    return [
        r for r in discovery_state.discovered_regimes
        if r.get("human_name") is None and r.get("activation_status") == "active"
    ]


def assign_human_name(
    discovery_state: RegimeDiscoveryState,
    regime_index: int,
    human_name: str,
) -> None:
    """The ONLY human intervention: assign a meaningful name to a regime."""
    for r in discovery_state.discovered_regimes:
        if r.get("regime_index") == regime_index:
            r["human_name"] = human_name
            log.info("Human named regime_%d as '%s'", regime_index, human_name)
            _save_discoveries(discovery_state.discovered_regimes)
            return
    log.warning("Regime index %d not found in discoveries", regime_index)
