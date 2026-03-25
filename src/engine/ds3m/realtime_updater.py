"""Real-time Bayesian updater for DS3M.

Sits between full DS3M inference cycles (every 60s) and provides
continuous distribution updates on each METAR observation (~20 min).

Instead of rerunning the full Mamba → PF → NSF pipeline, applies a
regime-conditioned scalar Kalman filter on the residuals + weather
event detectors that shift the skew-normal parameters.

Components:
  1. Regime-Conditioned KF: per-regime gain K, process noise Q,
     observation noise R — learned from settlement errors per regime.
     Different regimes track differently (sea breeze = tight, frontal = loose).
  2. Sea breeze detector: wind shift + dewpoint jump + temp stall
  3. Convective outflow detector: temp crash + gust spike
  4. Overnight warm advection monitor: 925hPa southerly + rising dewpoint
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Weather Event Detectors
# ──────────────────────────────────────────────────────────────────────

@dataclass
class METARSnapshot:
    """Single METAR observation for event detection."""
    timestamp: datetime
    temp_f: float
    dewpoint_f: float
    wind_dir: float        # degrees
    wind_speed_kt: float
    wind_gust_kt: float = 0.0
    cloud_cover_pct: float = 50.0
    pressure_mb: float = 1013.0


@dataclass
class WeatherEvent:
    """Detected weather event with impact on distribution."""
    event_type: str         # "sea_breeze", "convective_outflow", "overnight_warm"
    confidence: float       # 0-1
    mu_shift: float         # °F shift to apply to mu
    sigma_mult: float       # multiplier on sigma (>1 = widen, <1 = tighten)
    alpha_shift: float      # shift to skewness parameter
    description: str


class SeaBreezeDetector:
    """Detects sea breeze onset at KMIA from METAR time series.

    Criteria:
      - Wind shift from W/SW/NW → E/SE/S
      - Dewpoint jump > 2°F within 1 hour
      - Temperature plateau or drop after rising trend
      - Typically occurs 12:00-15:00 local in summer
    """

    def detect(self, history: list[METARSnapshot]) -> WeatherEvent | None:
        if len(history) < 4:
            return None

        latest = history[-1]
        # Look back ~1 hour
        one_hour_ago = [h for h in history if (latest.timestamp - h.timestamp).seconds <= 3600]
        if len(one_hour_ago) < 3:
            return None

        earliest = one_hour_ago[0]

        # Wind shift detection
        old_dir = earliest.wind_dir
        new_dir = latest.wind_dir
        was_offshore = (270 <= old_dir <= 360) or (0 <= old_dir <= 90) or (old_dir is None)
        is_onshore = 90 <= new_dir <= 210
        wind_shifted = was_offshore and is_onshore

        # Dewpoint jump
        dewpoint_jumped = (latest.dewpoint_f - earliest.dewpoint_f) > 2.0

        # Temperature stall
        temp_stalled = (latest.temp_f - earliest.temp_f) < 0.5

        if wind_shifted and dewpoint_jumped and temp_stalled:
            confidence = 0.85
            return WeatherEvent(
                event_type="sea_breeze",
                confidence=confidence,
                mu_shift=-1.5,    # cap the high ~1.5°F below current prediction
                sigma_mult=0.7,   # tighten — we're more certain now
                alpha_shift=-2.0, # shift left (sea breeze caps high)
                description=(
                    f"Sea breeze onset: wind {old_dir:.0f}°→{new_dir:.0f}°, "
                    f"dewpoint +{latest.dewpoint_f - earliest.dewpoint_f:.1f}°F, "
                    f"temp stalled at {latest.temp_f:.1f}°F"
                ),
            )

        # Partial detection (some but not all criteria)
        if wind_shifted and dewpoint_jumped:
            return WeatherEvent(
                event_type="sea_breeze",
                confidence=0.5,
                mu_shift=-0.8,
                sigma_mult=0.85,
                alpha_shift=-1.0,
                description="Possible sea breeze: wind shifted + dewpoint rising",
            )

        return None


class ConvectiveOutflowDetector:
    """Detects convective outflow near KMIA.

    Criteria:
      - Temperature drop > 3°F in 20 minutes
      - Wind gust > 25 kt
      - Often accompanied by pressure spike and cloud cover increase
    """

    def detect(self, history: list[METARSnapshot]) -> WeatherEvent | None:
        if len(history) < 2:
            return None

        latest = history[-1]
        # Look back ~20 minutes
        recent = [h for h in history if (latest.timestamp - h.timestamp).seconds <= 1200]
        if len(recent) < 2:
            return None

        earliest_recent = recent[0]
        temp_drop = earliest_recent.temp_f - latest.temp_f
        has_gust = latest.wind_gust_kt > 25

        if temp_drop > 3.0 and has_gust:
            return WeatherEvent(
                event_type="convective_outflow",
                confidence=0.9,
                mu_shift=-temp_drop * 0.7,  # high likely capped near current
                sigma_mult=0.6,              # very certain now
                alpha_shift=-3.0,            # strong left skew
                description=(
                    f"Convective outflow: temp dropped {temp_drop:.1f}°F, "
                    f"gust {latest.wind_gust_kt:.0f}kt"
                ),
            )
        elif temp_drop > 5.0:
            # Temp crash without gust data (gust may not be reported)
            return WeatherEvent(
                event_type="convective_outflow",
                confidence=0.7,
                mu_shift=-temp_drop * 0.5,
                sigma_mult=0.75,
                alpha_shift=-2.0,
                description=f"Possible outflow: temp dropped {temp_drop:.1f}°F rapidly",
            )

        return None


class OvernightWarmDetector:
    """Detects overnight warm advection that could set the daily high.

    Criteria:
      - 925 hPa southerly flow > 20 kt
      - Nighttime (8 PM - 6 AM local)
      - Rising dewpoint
      - DST active (CLI window extends to 12:59 AM EDT)
    """

    def detect(
        self,
        history: list[METARSnapshot],
        hour_local: float,
        is_dst: bool,
        wind_925_speed: float = 0,
        wind_925_dir: float = 0,
    ) -> WeatherEvent | None:
        if not is_dst:
            return None
        if not (hour_local >= 20 or hour_local <= 6):
            return None

        if 150 <= wind_925_dir <= 240 and wind_925_speed > 20:
            # Strong southerly LLJ overnight during DST
            confidence = min(0.9, 0.5 + (wind_925_speed - 20) * 0.02)

            # Check if temp is rising overnight (unusual)
            if len(history) >= 3:
                temp_trend = history[-1].temp_f - history[0].temp_f
                if temp_trend > 0:
                    confidence = min(0.95, confidence + 0.1)

            return WeatherEvent(
                event_type="overnight_warm",
                confidence=confidence,
                mu_shift=2.0,     # overnight high could beat daytime
                sigma_mult=1.3,   # uncertain — but biased high
                alpha_shift=2.0,  # right skew (upside surprise)
                description=(
                    f"Overnight warm advection: 925hPa "
                    f"{wind_925_dir:.0f}° @ {wind_925_speed:.0f}kt, "
                    f"DST window open"
                ),
            )
        return None


# ──────────────────────────────────────────────────────────────────────
# Regime-Conditioned Scalar Kalman Filter
# ──────────────────────────────────────────────────────────────────────

# Physical priors for each initial regime — these are starting points
# that get adapted from settlement errors over time.
_DEFAULT_REGIME_KF_PARAMS: dict[str, dict[str, float]] = {
    "continental": {"K": 0.25, "Q": 0.4, "R": 1.0},   # stable, moderate tracking
    "sea_breeze":  {"K": 0.40, "Q": 0.2, "R": 0.6},   # tight tracking after onset
    "frontal":     {"K": 0.15, "Q": 1.5, "R": 2.0},    # chaotic, trust model more
    "tropical":    {"K": 0.30, "Q": 0.3, "R": 0.8},    # moist airmass, moderate
    "nocturnal":   {"K": 0.35, "Q": 0.6, "R": 1.2},    # overnight can surprise
}
_FALLBACK_PARAMS = {"K": 0.30, "Q": 0.5, "R": 1.0}


@dataclass
class RegimeKFState:
    """Per-regime Kalman filter parameters + adaptation history."""
    K: float                          # Kalman gain
    Q: float                          # Process noise variance (°F²)
    R: float                          # Observation noise variance (°F²)
    P: float = 1.0                    # State error covariance
    innovations: deque = field(default_factory=lambda: deque(maxlen=336))  # 14d × 24h
    settlement_errors: deque = field(default_factory=lambda: deque(maxlen=60))  # ~2 months


class RegimeConditionedKF:
    """Scalar Kalman filter with per-regime gain, Q, and R.

    Each regime learns its own error dynamics from settlement outcomes:
      - Continental: stable airmass, moderate tracking
      - Sea breeze: after onset detection, tighten aggressively
      - Frontal: high process noise, trust model more than obs
      - Tropical/moist: steady tracking, maritime smoothing
      - Nocturnal warm: overnight surprises need faster response

    New regimes discovered by HDP get initialized from the closest
    existing regime (by atmospheric similarity) or from fallback params.

    The filter maintains a proper scalar KF with predict/update:
      Predict: P = P + Q
      Update:  S = P + R, K = P / S, x = x + K * innovation, P = (1 - K) * P
    Rather than fixed K, the gain emerges from learned Q/R ratio.
    """

    def __init__(self):
        self.regime_states: dict[str, RegimeKFState] = {}
        self._init_default_regimes()

    def _init_default_regimes(self):
        for name, params in _DEFAULT_REGIME_KF_PARAMS.items():
            self.regime_states[name] = RegimeKFState(
                K=params["K"], Q=params["Q"], R=params["R"],
            )

    def _get_regime_state(self, regime_name: str) -> RegimeKFState:
        """Get or create state for a regime (handles HDP-discovered regimes)."""
        if regime_name not in self.regime_states:
            log.info(f"Initializing KF params for new regime: {regime_name}")
            self.regime_states[regime_name] = RegimeKFState(
                K=_FALLBACK_PARAMS["K"],
                Q=_FALLBACK_PARAMS["Q"],
                R=_FALLBACK_PARAMS["R"],
            )
        return self.regime_states[regime_name]

    def update(
        self,
        predicted_temp: float,
        observed_temp: float,
        current_mu: float,
        current_sigma: float,
        current_alpha: float,
        regime_name: str,
        regime_probs: dict[str, float] | None = None,
    ) -> tuple[float, float, float]:
        """Apply regime-conditioned KF update given new METAR observation.

        If regime_probs is provided, blends updates across regimes weighted
        by posterior probability (soft conditioning). Otherwise uses the
        MAP regime (hard conditioning).

        Returns: (new_mu, new_sigma, new_alpha)
        """
        innovation = observed_temp - predicted_temp

        if regime_probs and len(regime_probs) > 1:
            # Soft conditioning: blend K across regimes by posterior weight
            blended_K = 0.0
            blended_sigma_mult = 0.0
            for rname, prob in regime_probs.items():
                if prob < 0.01:
                    continue
                rs = self._get_regime_state(rname)
                # Proper KF predict/update for this regime
                P_pred = rs.P + rs.Q
                S = P_pred + rs.R
                K_i = P_pred / S if S > 1e-8 else rs.K
                blended_K += prob * K_i
                # Sigma multiplier from innovation relative to regime's R
                norm_innov = abs(innovation) / math.sqrt(rs.R) if rs.R > 0 else abs(innovation)
                if norm_innov < 0.5:
                    blended_sigma_mult += prob * 0.95
                elif norm_innov < 1.0:
                    blended_sigma_mult += prob * 0.98
                else:
                    blended_sigma_mult += prob * (1.0 + 0.03 * min(norm_innov, 3.0))

            effective_K = blended_K
            sigma_mult = blended_sigma_mult if blended_sigma_mult > 0 else 1.0
        else:
            # Hard conditioning: use MAP regime only
            rs = self._get_regime_state(regime_name)
            P_pred = rs.P + rs.Q
            S = P_pred + rs.R
            effective_K = P_pred / S if S > 1e-8 else rs.K
            # Update covariance
            rs.P = (1.0 - effective_K) * P_pred

            norm_innov = abs(innovation) / math.sqrt(rs.R) if rs.R > 0 else abs(innovation)
            if norm_innov < 0.5:
                sigma_mult = 0.95
            elif norm_innov < 1.0:
                sigma_mult = 0.98
            else:
                sigma_mult = 1.0 + 0.03 * min(norm_innov, 3.0)

        # Clamp effective gain
        effective_K = max(0.05, min(0.7, effective_K))

        # Apply updates
        new_mu = current_mu + effective_K * innovation
        new_sigma = current_sigma * sigma_mult
        new_sigma = max(0.3, min(8.0, new_sigma))
        new_alpha = current_alpha  # modified by event detectors

        # Track innovation for the active regime
        rs_active = self._get_regime_state(regime_name)
        rs_active.innovations.append((innovation, current_sigma, effective_K))

        return new_mu, new_sigma, new_alpha

    def adapt_from_settlement(self, settlement_error: float, regime_name: str):
        """Post-settlement: adapt Q and R for this regime based on error.

        Uses the innovation-settlement error relationship to tune Q/R:
          - If settlement_error >> avg innovation: Q too low (under-tracking)
          - If settlement_error << avg innovation: R too low (over-tracking)
        """
        rs = self._get_regime_state(regime_name)
        rs.settlement_errors.append(settlement_error)

        if len(rs.innovations) < 10 or len(rs.settlement_errors) < 3:
            return

        avg_innov = sum(abs(i) for i, _, _ in rs.innovations) / len(rs.innovations)
        avg_settle = sum(abs(e) for e in rs.settlement_errors) / len(rs.settlement_errors)

        if avg_settle < 1e-6:
            return

        ratio = avg_settle / max(avg_innov, 1e-6)

        if ratio > 1.5:
            # Under-tracking: increase Q (trust obs more)
            rs.Q = min(3.0, rs.Q * 1.08)
            rs.R = max(0.2, rs.R * 0.95)
            log.info(f"Regime '{regime_name}' KF: under-tracking, Q→{rs.Q:.3f} R→{rs.R:.3f}")
        elif ratio < 0.5:
            # Over-tracking: increase R (trust model more)
            rs.Q = max(0.05, rs.Q * 0.95)
            rs.R = min(5.0, rs.R * 1.08)
            log.info(f"Regime '{regime_name}' KF: over-tracking, Q→{rs.Q:.3f} R→{rs.R:.3f}")

        # Reset covariance after adaptation
        rs.P = 1.0

    def to_dict(self) -> dict:
        """Serialize for persistence."""
        return {
            name: {"K": rs.K, "Q": rs.Q, "R": rs.R, "P": rs.P}
            for name, rs in self.regime_states.items()
        }

    def from_dict(self, data: dict):
        """Restore from persistence."""
        for name, params in data.items():
            self.regime_states[name] = RegimeKFState(
                K=params.get("K", _FALLBACK_PARAMS["K"]),
                Q=params.get("Q", _FALLBACK_PARAMS["Q"]),
                R=params.get("R", _FALLBACK_PARAMS["R"]),
                P=params.get("P", 1.0),
            )


# ──────────────────────────────────────────────────────────────────────
# Bracket Kalman Bank — parallel KFs for Kalshi temperature brackets
# ──────────────────────────────────────────────────────────────────────

# Default 10 brackets for Miami daily-high market.
# Edges are in °F: [(-inf, 60), [60, 65), [65, 70), ..., [95, 100), [100, +inf)]
DEFAULT_BRACKET_EDGES: list[tuple[float | None, float | None]] = [
    (None, 60.0),
    (60.0, 65.0),
    (65.0, 70.0),
    (70.0, 75.0),
    (75.0, 80.0),
    (80.0, 85.0),
    (85.0, 90.0),
    (90.0, 95.0),
    (95.0, 100.0),
    (100.0, None),
]

# Per-bracket defaults calibrated to Miami climatology.
# Tail brackets are more volatile and less certain; center brackets are stable.
# Q = process noise variance on the probability, R = observation noise variance.
_DEFAULT_BRACKET_QR: list[dict[str, float]] = [
    # <60: extreme cold tail — very rare, high uncertainty
    {"Q": 0.004, "R": 0.010},
    # 60-65: cold tail — rare but happens in winter fronts
    {"Q": 0.003, "R": 0.008},
    # 65-70: cool — occasional winter/spring
    {"Q": 0.0025, "R": 0.006},
    # 70-75: moderate — common winter high
    {"Q": 0.002, "R": 0.004},
    # 75-80: center-left — very common
    {"Q": 0.0015, "R": 0.003},
    # 80-85: climatological mode for Miami
    {"Q": 0.0012, "R": 0.0025},
    # 85-90: center-right — common summer
    {"Q": 0.0015, "R": 0.003},
    # 90-95: warm — frequent summer highs
    {"Q": 0.002, "R": 0.004},
    # 95-100: hot tail — heat events
    {"Q": 0.003, "R": 0.008},
    # 100+: extreme hot tail — very rare
    {"Q": 0.004, "R": 0.010},
]

# Innovation window length for health monitoring
_INNOVATION_WINDOW = 50


@dataclass
class _BracketKFState:
    """State for a single bracket's scalar Kalman filter."""
    x: float          # current probability estimate
    P: float          # error covariance
    Q: float          # process noise variance
    R: float          # observation noise variance
    innovations: deque = field(default_factory=lambda: deque(maxlen=_INNOVATION_WINDOW))


class BracketKalmanBank:
    """Parallel scalar Kalman filters for Kalshi temperature bracket probabilities.

    Each bracket maintains an independent KF tracking that bracket's probability.
    After all updates, probabilities are renormalized to sum to 1.0.

    Tail brackets use higher Q/R (more volatile, less certain observations),
    while center brackets near Miami's climatological mode use lower Q/R.

    The innovation monitor tracks residuals per bracket and flags persistent
    non-Gaussian behavior as an early regime-change signal — often firing
    before the HDP formally detects the shift.
    """

    def __init__(
        self,
        bracket_edges: list[tuple[float | None, float | None]] | None = None,
        bracket_qr: list[dict[str, float]] | None = None,
        initial_probs: list[float] | None = None,
    ):
        self.bracket_edges = bracket_edges or DEFAULT_BRACKET_EDGES
        n = len(self.bracket_edges)
        qr = bracket_qr or _DEFAULT_BRACKET_QR
        if len(qr) != n:
            raise ValueError(f"bracket_qr length {len(qr)} != bracket count {n}")

        # Initialize with uniform probs unless provided
        if initial_probs is None:
            initial_probs = [1.0 / n] * n
        if len(initial_probs) != n:
            raise ValueError(f"initial_probs length {len(initial_probs)} != bracket count {n}")

        self.filters: list[_BracketKFState] = [
            _BracketKFState(
                x=initial_probs[i],
                P=0.01,   # start with moderate uncertainty on prob
                Q=qr[i]["Q"],
                R=qr[i]["R"],
            )
            for i in range(n)
        ]

    # ── Core KF operations ────────────────────────────────────────────

    def predict(self) -> None:
        """Predict step: inflate covariance by process noise for each bracket."""
        for f in self.filters:
            f.P = f.P + f.Q

    def update(
        self,
        bracket_idx: int,
        observed_prob: float,
        regime_name: str | None = None,
        regime_probs: dict[str, float] | None = None,
    ) -> None:
        """Standard scalar KF update for a single bracket.

        Optionally modulated by regime: if regime_probs indicates a
        high-volatility regime (frontal, tropical), inflate R slightly
        to down-weight noisy observations.
        """
        f = self.filters[bracket_idx]

        # Optional regime modulation on R
        R_eff = f.R
        if regime_probs:
            # Frontal and tropical regimes inject extra observation uncertainty
            frontal_weight = regime_probs.get("frontal", 0.0)
            nocturnal_weight = regime_probs.get("nocturnal", 0.0)
            R_eff = f.R * (1.0 + 0.5 * frontal_weight + 0.3 * nocturnal_weight)
        elif regime_name in ("frontal",):
            R_eff = f.R * 1.5
        elif regime_name in ("nocturnal",):
            R_eff = f.R * 1.3

        # Standard scalar KF update
        innovation = observed_prob - f.x
        S = f.P + R_eff
        K = f.P / S if S > 1e-12 else 0.5
        K = max(0.01, min(0.95, K))  # clamp gain

        f.x = f.x + K * innovation
        f.x = max(0.0, min(1.0, f.x))  # probability bounds
        f.P = (1.0 - K) * f.P

        # Track innovation for health monitoring
        f.innovations.append((innovation, S))

    def update_all(
        self,
        observed_probs: list[float],
        regime_name: str | None = None,
        regime_probs: dict[str, float] | None = None,
    ) -> None:
        """Update all bracket KFs at once, then renormalize.

        Args:
            observed_probs: observed probability for each bracket (must sum ~1.0)
            regime_name: MAP regime name for optional modulation
            regime_probs: full posterior over regimes for soft modulation
        """
        n = len(self.filters)
        if len(observed_probs) != n:
            raise ValueError(f"observed_probs length {len(observed_probs)} != bracket count {n}")

        for i in range(n):
            self.update(i, observed_probs[i], regime_name=regime_name, regime_probs=regime_probs)

        self._renormalize()

    def _renormalize(self) -> None:
        """Renormalize bracket probabilities to sum to 1.0."""
        total = sum(f.x for f in self.filters)
        if total > 1e-12:
            for f in self.filters:
                f.x /= total
        else:
            # Degenerate case: reset to uniform
            n = len(self.filters)
            for f in self.filters:
                f.x = 1.0 / n

    def get_probabilities(self) -> list[float]:
        """Return current probability estimates for all brackets."""
        return [f.x for f in self.filters]

    # ── Settlement adaptation ─────────────────────────────────────────

    def adapt_from_settlement(self, bracket_idx: int, settled_outcome: bool) -> None:
        """Tune Q/R based on tracking error after a bracket settles.

        Args:
            bracket_idx: which bracket settled
            settled_outcome: True if the high landed in this bracket, False otherwise
        """
        f = self.filters[bracket_idx]
        settled_value = 1.0 if settled_outcome else 0.0
        tracking_error = abs(settled_value - f.x)

        if tracking_error > 0.3:
            # Large error: bracket was badly miscalibrated — increase Q
            f.Q = min(0.02, f.Q * 1.15)
            f.R = max(0.001, f.R * 0.92)
            log.info(
                f"Bracket {bracket_idx} settlement adapt: large error "
                f"({tracking_error:.3f}), Q→{f.Q:.5f} R→{f.R:.5f}"
            )
        elif tracking_error < 0.1:
            # Good tracking: tighten Q slightly, trust observations more
            f.Q = max(0.0005, f.Q * 0.95)
            f.R = max(0.001, f.R * 0.97)
        else:
            # Moderate error: nudge Q up slightly
            f.Q = min(0.02, f.Q * 1.03)

        # Reset covariance after settlement
        f.P = 0.01

    # ── Innovation monitor ────────────────────────────────────────────

    def check_innovation_health(self) -> dict[str, Any]:
        """Assess KF calibration health from innovation sequences.

        Returns a dict with:
          - mean_innovation: per-bracket mean innovation (should be ~0)
          - variance_ratio: actual innovation variance / expected (should be ~1.0)
          - regime_change_signal: True when innovations look persistently
            non-Gaussian (early warning of regime shift)
          - per_bracket: list of per-bracket detail dicts
        """
        per_bracket: list[dict[str, Any]] = []
        any_regime_signal = False

        for idx, f in enumerate(self.filters):
            innovations = list(f.innovations)
            n = len(innovations)
            if n < 5:
                per_bracket.append({
                    "bracket_idx": idx,
                    "n_innovations": n,
                    "mean_innovation": None,
                    "variance_ratio": None,
                    "regime_change_signal": False,
                })
                continue

            residuals = [inn for inn, _ in innovations]
            expected_vars = [S for _, S in innovations]

            mean_inn = sum(residuals) / n
            actual_var = sum((r - mean_inn) ** 2 for r in residuals) / n
            expected_var = sum(expected_vars) / n if expected_vars else 1.0
            var_ratio = actual_var / expected_var if expected_var > 1e-12 else float("inf")

            # Regime change heuristic: check for persistent sign bias
            # (non-zero mean) and excess variance (fat tails / non-Gaussian).
            # A well-tuned KF has zero-mean, unit-variance-ratio innovations.
            sign_bias = abs(mean_inn) > 0.03  # persistent directional bias
            var_excess = var_ratio > 2.0       # fat tails
            # Also check runs test: count sign changes
            sign_changes = sum(
                1 for i in range(1, n) if (residuals[i] >= 0) != (residuals[i - 1] >= 0)
            )
            expected_changes = (n - 1) / 2.0
            # Too few sign changes = persistent trend (regime shift)
            run_deficit = sign_changes < expected_changes * 0.5

            bracket_signal = (sign_bias and var_excess) or (sign_bias and run_deficit)
            if bracket_signal:
                any_regime_signal = True

            per_bracket.append({
                "bracket_idx": idx,
                "n_innovations": n,
                "mean_innovation": round(mean_inn, 6),
                "variance_ratio": round(var_ratio, 4),
                "regime_change_signal": bracket_signal,
            })

        return {
            "regime_change_signal": any_regime_signal,
            "per_bracket": per_bracket,
        }

    # ── Persistence ───────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Serialize for persistence."""
        return {
            "bracket_edges": [
                (lo, hi) for lo, hi in self.bracket_edges
            ],
            "filters": [
                {
                    "x": f.x,
                    "P": f.P,
                    "Q": f.Q,
                    "R": f.R,
                }
                for f in self.filters
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BracketKalmanBank":
        """Restore from persistence."""
        edges = [tuple(e) for e in data["bracket_edges"]]
        n = len(edges)
        instance = cls.__new__(cls)
        instance.bracket_edges = edges
        instance.filters = [
            _BracketKFState(
                x=fd["x"],
                P=fd["P"],
                Q=fd["Q"],
                R=fd["R"],
            )
            for fd in data["filters"]
        ]
        return instance


# ──────────────────────────────────────────────────────────────────────
# Real-Time Updater (combines everything)
# ──────────────────────────────────────────────────────────────────────

class RealTimeUpdater:
    """Orchestrates inter-cycle distribution updates.

    On each new METAR observation:
      1. Run weather event detectors
      2. Apply regime-conditioned KF on residuals (soft-blended by posterior)
      3. Apply event-driven adjustments to (mu, sigma, alpha)
      4. Return updated distribution parameters

    These updates are lightweight (~1ms) and run every 20 minutes
    between the full 60-second DS3M inference cycles.
    """

    def __init__(self):
        self.kf = RegimeConditionedKF()
        self.bracket_kf = BracketKalmanBank()
        self.sea_breeze = SeaBreezeDetector()
        self.outflow = ConvectiveOutflowDetector()
        self.overnight = OvernightWarmDetector()
        self.metar_history: deque[METARSnapshot] = deque(maxlen=24)  # ~8h at 20min
        self.active_events: list[WeatherEvent] = []

    def update(
        self,
        metar: METARSnapshot,
        ds3m_mu: float,
        ds3m_sigma: float,
        ds3m_alpha: float,
        predicted_hourly_temp: float,
        hour_local: float,
        is_dst: bool,
        regime_name: str = "continental",
        regime_probs: dict[str, float] | None = None,
        wind_925_speed: float = 0,
        wind_925_dir: float = 0,
        bracket_observed_probs: list[float] | None = None,
    ) -> tuple[float, float, float, list[WeatherEvent]]:
        """Process new METAR and return updated distribution.

        Args:
            regime_name: MAP regime from HDP (used for hard conditioning fallback)
            regime_probs: full posterior over regimes (enables soft KF blending)
            bracket_observed_probs: observed bracket probabilities (e.g. from
                Kalshi order book mid-prices). If provided, the BracketKalmanBank
                is predicted and updated; otherwise only predict is run.

        Returns: (new_mu, new_sigma, new_alpha, detected_events)
        """
        self.metar_history.append(metar)
        history = list(self.metar_history)

        # 1. Regime-conditioned KF update (soft-blended if posterior available)
        mu, sigma, alpha = self.kf.update(
            predicted_hourly_temp, metar.temp_f,
            ds3m_mu, ds3m_sigma, ds3m_alpha,
            regime_name=regime_name,
            regime_probs=regime_probs,
        )

        # 1b. Bracket Kalman Bank predict + update
        self.bracket_kf.predict()
        if bracket_observed_probs is not None:
            self.bracket_kf.update_all(
                bracket_observed_probs,
                regime_name=regime_name,
                regime_probs=regime_probs,
            )

        # 2. Event detection
        self.active_events = []

        event = self.sea_breeze.detect(history)
        if event:
            self.active_events.append(event)

        event = self.outflow.detect(history)
        if event:
            self.active_events.append(event)

        event = self.overnight.detect(
            history, hour_local, is_dst, wind_925_speed, wind_925_dir
        )
        if event:
            self.active_events.append(event)

        # 3. Apply event adjustments (weighted by confidence)
        for evt in self.active_events:
            mu += evt.mu_shift * evt.confidence
            sigma *= evt.sigma_mult ** evt.confidence
            alpha += evt.alpha_shift * evt.confidence
            log.info(f"Weather event: {evt.event_type} (conf={evt.confidence:.2f}): {evt.description}")

        # Safety clamps
        sigma = max(0.3, min(8.0, sigma))
        alpha = max(-10.0, min(10.0, alpha))

        return mu, sigma, alpha, self.active_events

    def adapt_from_settlement(
        self,
        settlement_error: float,
        regime_name: str,
        settled_bracket_idx: int | None = None,
    ):
        """Post-settlement adaptation for both KFs.

        Args:
            settlement_error: scalar error for the regime-conditioned KF
            regime_name: which regime was active
            settled_bracket_idx: if provided, adapts the bracket KF too.
                The bracket that the actual high landed in gets
                settled_outcome=True; all others get False.
        """
        self.kf.adapt_from_settlement(settlement_error, regime_name)

        if settled_bracket_idx is not None:
            for i in range(len(self.bracket_kf.filters)):
                self.bracket_kf.adapt_from_settlement(i, i == settled_bracket_idx)
