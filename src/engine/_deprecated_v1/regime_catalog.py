"""T4.1-T4.2: Live regime catalog and inference.

Defines a compact, interpretable set of weather regimes for Miami (KMIA).
Each regime conditions the inference pipeline differently:
  - Sigma scaling (wider uncertainty in volatile regimes)
  - Mu bias (shift forecast in known-biased regimes)
  - Signal gating (which signals matter in each regime)
  - Position sizing via regime confidence

Regime detection uses a combination of:
  1. Deterministic rules from signal engine state
  2. BOCPD changepoint probability
  3. LETKF analysis spread (ensemble disagreement)
  4. Atmospheric context (CAPE, PW, cloud cover, wind direction)

The catalog is designed to be:
  - Small (6 regimes) — avoids overfitting with limited data
  - Interpretable — each regime has clear meteorological meaning
  - Extensible — HDP-HMM (T5.1) can propose new regimes via shadow lane
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum


class RegimeType(Enum):
    """Compact live regime catalog for SE Florida (KMIA/KFLL)."""

    MARINE_STABLE = "marine_stable"
    # Typical trade-wind pattern. Light E/SE winds, marine layer moderates temps.
    # Highs: predictable, near climatology. Lows: bounded by dew point floor.
    # Sigma: narrow. Confidence: high.

    INLAND_HEATING = "inland_heating"
    # Light winds, clear skies, strong solar heating.
    # Highs: can exceed climatology. Lows: radiational cooling possible.
    # Sigma: moderate. Key signal: FAWN solar radiation, cloud cover.

    CLOUD_SUPPRESSION = "cloud_suppression"
    # Persistent cloud cover limits daytime heating.
    # Highs: suppressed below model forecasts. Lows: elevated (no radiational cooling).
    # Sigma: narrow (cloud ceiling creates strong constraint).
    # Key signal: METAR BKN/OVC, GOES cloud fraction, FAWN solar < 400 W/m².

    CONVECTIVE_OUTFLOW = "convective_outflow"
    # Afternoon/evening thunderstorms produce cold outflow boundaries.
    # Lows: can crash 5-10°F in minutes. Highs: interrupted by storms.
    # Sigma: WIDE. Key signals: CAPE > 1500, PW > 35mm, MRMS/radar proximity.
    # This is the highest-risk regime for Miami low markets.

    FRONTAL_PASSAGE = "frontal_passage"
    # Cold front or boundary passage. Wind shift (NW/N/NE), dew crash, pressure rise.
    # Lows: can drop well below climatology. Highs: suppressed.
    # Sigma: wide during transition, narrows after passage.
    # Key signals: wind direction continental (250-360°), BOCPD changepoint.

    TRANSITION = "transition"
    # Mixed/uncertain regime. No clear dominant pattern.
    # Default when no other regime has high confidence.
    # Sigma: moderate-wide. Conservative position sizing.


@dataclass
class LiveRegimeState:
    """Current regime assessment with probabilities."""

    primary: RegimeType = RegimeType.TRANSITION
    confidence: float = 0.5  # 0-1 confidence in primary regime
    probabilities: dict[str, float] = field(default_factory=dict)
    # Conditioning parameters
    sigma_multiplier: float = 1.0  # multiply baseline sigma by this
    mu_bias_high_f: float = 0.0  # add to baseline mu for HIGH market (°F)
    mu_bias_low_f: float = 0.0   # add to baseline mu for LOW market (°F)
    # True when no specific regime matched — signals the pattern doesn't fit the catalog
    unrecognized: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "primary": self.primary.value,
            "confidence": round(self.confidence, 3),
            "probabilities": {k: round(v, 3) for k, v in self.probabilities.items()},
            "sigma_multiplier": round(self.sigma_multiplier, 3),
            "mu_bias_high_f": round(self.mu_bias_high_f, 2),
            "mu_bias_low_f": round(self.mu_bias_low_f, 2),
            "unrecognized": self.unrecognized,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Regime conditioning parameters
# ---------------------------------------------------------------------------
REGIME_PARAMS: dict[RegimeType, dict] = {
    RegimeType.MARINE_STABLE: {
        "sigma_multiplier": 0.85,  # narrow uncertainty
        "mu_bias_high_f": 0.0,
        "mu_bias_low_f": 0.0,
        "min_confidence_for_sizing": 0.6,
    },
    RegimeType.INLAND_HEATING: {
        "sigma_multiplier": 1.0,
        "mu_bias_high_f": 0.0,
        "mu_bias_low_f": 0.0,
        "min_confidence_for_sizing": 0.5,
    },
    RegimeType.CLOUD_SUPPRESSION: {
        "sigma_multiplier": 0.80,  # tight constraint from cloud ceiling
        "mu_bias_high_f": -1.5,  # highs suppressed below model forecasts
        "mu_bias_low_f": +1.0,   # lows elevated (less radiational cooling under clouds)
        "min_confidence_for_sizing": 0.5,
    },
    RegimeType.CONVECTIVE_OUTFLOW: {
        "sigma_multiplier": 1.6,  # wide uncertainty — crashes possible
        "mu_bias_high_f": 0.0,
        "mu_bias_low_f": -2.0,   # lows can crash 5-10°F from outflow
        "min_confidence_for_sizing": 0.4,
    },
    RegimeType.FRONTAL_PASSAGE: {
        "sigma_multiplier": 1.4,  # wide during transition
        "mu_bias_high_f": -1.0,   # highs suppressed post-frontal
        "mu_bias_low_f": -1.5,    # lows drop with cold advection
        "min_confidence_for_sizing": 0.4,
    },
    RegimeType.TRANSITION: {
        "sigma_multiplier": 1.15,  # slightly wider than normal
        "mu_bias_high_f": 0.0,
        "mu_bias_low_f": 0.0,
        "min_confidence_for_sizing": 0.3,
    },
}


# ---------------------------------------------------------------------------
# Regime inference
# ---------------------------------------------------------------------------
def infer_live_regime(
    *,
    # Signal engine features
    cape: float | None = None,
    pw_mm: float | None = None,
    wind_dir_deg: float | None = None,
    cloud_fraction: float | None = None,
    dew_crash_active: bool = False,
    pressure_surge: bool = False,
    # BOCPD state
    bocpd_changepoint_prob: float = 0.0,
    # Time context
    hour_lst: float = 12.0,
    # LETKF analysis spread
    letkf_spread: float | None = None,
) -> LiveRegimeState:
    """Infer current weather regime from available signals.

    Uses a deterministic rule-based classifier with soft probabilities.
    Rules are ordered by specificity (most specific match wins).
    """
    probs: dict[str, float] = {r.value: 0.0 for r in RegimeType}

    # Rule 1: Convective outflow — CAPE + PW + afternoon hours
    if cape is not None and pw_mm is not None:
        if cape > 1500 and pw_mm > 35 and (hour_lst >= 17 or hour_lst <= 2):
            # Strong convective signal
            outflow_score = min(1.0, (cape / 3000) * (pw_mm / 45))
            probs[RegimeType.CONVECTIVE_OUTFLOW.value] += outflow_score * 0.7
        elif cape > 800 and pw_mm > 30:
            probs[RegimeType.CONVECTIVE_OUTFLOW.value] += 0.2

    # Rule 2: Frontal passage — continental wind + changepoint + dew crash
    continental_wind = False
    if wind_dir_deg is not None:
        continental_wind = 250 <= wind_dir_deg <= 360 or wind_dir_deg <= 30
    if continental_wind:
        probs[RegimeType.FRONTAL_PASSAGE.value] += 0.4
    if bocpd_changepoint_prob > 0.5:
        probs[RegimeType.FRONTAL_PASSAGE.value] += 0.3
    if dew_crash_active:
        probs[RegimeType.FRONTAL_PASSAGE.value] += 0.2
    if pressure_surge:
        probs[RegimeType.FRONTAL_PASSAGE.value] += 0.1

    # Rule 3: Cloud suppression — high cloud fraction
    if cloud_fraction is not None and cloud_fraction > 0.6:
        probs[RegimeType.CLOUD_SUPPRESSION.value] += 0.5 + 0.3 * (cloud_fraction - 0.6) / 0.4

    # Rule 4: Inland heating — clear sky + light winds + daytime
    if cloud_fraction is not None and cloud_fraction < 0.2 and 10 <= hour_lst <= 18:
        wind_light = wind_dir_deg is None  # no strong directional signal
        probs[RegimeType.INLAND_HEATING.value] += 0.4
        if cape is not None and cape < 500:
            probs[RegimeType.INLAND_HEATING.value] += 0.2

    # Rule 5: Marine stable — SE/E winds, moderate humidity, no extreme signals
    if wind_dir_deg is not None and 90 <= wind_dir_deg <= 180:
        probs[RegimeType.MARINE_STABLE.value] += 0.5
        if cape is not None and cape < 1000:
            probs[RegimeType.MARINE_STABLE.value] += 0.2

    # Transition gets a baseline probability that decreases as others increase
    max_non_transition = max(
        v for k, v in probs.items() if k != RegimeType.TRANSITION.value
    ) if probs else 0.0
    probs[RegimeType.TRANSITION.value] = max(0.1, 0.5 - max_non_transition * 0.5)

    # Normalize
    total = sum(probs.values())
    if total > 0:
        probs = {k: v / total for k, v in probs.items()}

    # Find primary
    primary_key = max(probs, key=probs.get)
    primary = RegimeType(primary_key)
    confidence = probs[primary_key]

    # Get conditioning parameters
    params = REGIME_PARAMS[primary]

    # Detect genuinely unrecognized patterns: TRANSITION wins AND
    # no specific regime accumulated meaningful probability
    unrecognized = (
        primary == RegimeType.TRANSITION
        and max_non_transition < 0.15  # no specific regime scored above noise
    )

    state = LiveRegimeState(
        primary=primary,
        confidence=confidence,
        probabilities=probs,
        sigma_multiplier=params["sigma_multiplier"],
        mu_bias_high_f=params.get("mu_bias_high_f", 0.0),
        mu_bias_low_f=params.get("mu_bias_low_f", 0.0),
    )

    if unrecognized:
        # Widen sigma further when we don't know what's happening
        state.unrecognized = True
        state.sigma_multiplier = max(state.sigma_multiplier, 1.3)
        state.notes.append("UNRECOGNIZED — no specific regime matched; widened sigma; flag for review")
    else:
        state.notes.append(f"regime={primary.value} ({confidence:.0%})")

    if cape is not None:
        state.notes.append(f"cape={cape:.0f}")
    if cloud_fraction is not None:
        state.notes.append(f"cloud={cloud_fraction:.2f}")

    return state
