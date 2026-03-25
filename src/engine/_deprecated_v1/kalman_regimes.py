"""
Switching Kalman Filter for real-time weather regime detection.

Runs every engine cycle (every few minutes) and maintains probability
distributions over weather regimes. Each regime has its own linear-Gaussian
state-space model with learned transition dynamics.

State vector (5D):
    x = [temp_f, dew_f, pressure_hpa, wind_dir_sin, wind_dir_cos]

Wind direction is encoded as sin/cos to handle 360->0 wraparound.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("analysis_data/skf_config.json")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RegimeModel:
    """Per-regime linear-Gaussian state-space model."""
    name: str
    A: list[list[float]]           # 5x5 state transition matrix
    b: list[float]                 # 5x1 bias vector
    Q: list[list[float]]           # 5x5 process noise covariance
    R: list[list[float]]           # 5x5 measurement noise covariance
    self_transition_prob: float    # P(stay in this regime | currently in it)
    mu_shift_high: float           # learned mean adjustment for HIGH market
    mu_shift_low: float            # learned mean adjustment for LOW market
    sigma_scale_high: float        # learned sigma multiplier for HIGH
    sigma_scale_low: float         # learned sigma multiplier for LOW
    active_families: list[str]     # signal families that typically activate


@dataclass
class ObsVector:
    """Raw observation from weather station."""
    temp_f: float
    dew_f: float
    pressure_hpa: float
    wind_dir_deg: float
    timestamp_utc: str


@dataclass
class SKFState:
    """Output of a single SKF update cycle."""
    regime_probabilities: dict[str, float]
    most_likely_regime: str
    regime_confidence: float
    mu_shift_high: float        # probability-weighted average
    mu_shift_low: float
    sigma_scale_high: float
    sigma_scale_low: float
    active_families: dict[str, float]   # family name -> activation strength
    innovation_norm: float              # how surprised the filter is
    regime_switch_detected: bool


# ---------------------------------------------------------------------------
# Default regime models (used when no config file exists)
# ---------------------------------------------------------------------------

def _diag(vals: list[float]) -> list[list[float]]:
    """Build a diagonal matrix as nested lists."""
    n = len(vals)
    m = [[0.0] * n for _ in range(n)]
    for i, v in enumerate(vals):
        m[i][i] = v
    return m


def _identity_scaled(scale: float, n: int = 5) -> list[list[float]]:
    return _diag([scale] * n)


def _default_regime_models() -> list[dict]:
    """Return four conservative default regimes."""

    # 1. mixed_uncertain — near-identity, large noise
    mixed = dict(
        name="mixed_uncertain",
        A=_identity_scaled(0.98),
        b=[0.0, 0.0, 0.0, 0.0, 0.0],
        Q=_diag([2.0, 2.0, 0.5, 0.1, 0.1]),
        R=_diag([0.5, 0.5, 0.2, 0.05, 0.05]),
        self_transition_prob=0.85,
        mu_shift_high=0.0,
        mu_shift_low=0.0,
        sigma_scale_high=1.0,
        sigma_scale_low=1.0,
        active_families=[],
    )

    # 2. postfrontal_persistence — slight cooling, small noise, sticky
    pf_A = _identity_scaled(0.995)
    pf_A[0][0] = 0.995  # temp decays slightly
    postfrontal = dict(
        name="postfrontal_persistence",
        A=pf_A,
        b=[-0.05, -0.02, 0.0, 0.0, 0.0],   # slight cooling tendency
        Q=_diag([0.3, 0.3, 0.1, 0.02, 0.02]),
        R=_diag([0.4, 0.4, 0.15, 0.04, 0.04]),
        self_transition_prob=0.92,
        mu_shift_high=0.0,
        mu_shift_low=-1.0,
        sigma_scale_high=1.0,
        sigma_scale_low=0.95,
        active_families=["frontal", "pressure"],
    )

    # 3. boundary_break — coupled dynamics, volatile, transitions quickly
    bb_A = _identity_scaled(0.96)
    bb_A[0][2] = -0.02   # pressure->temp coupling
    bb_A[2][0] = 0.01    # temp->pressure coupling
    bb_A[1][0] = 0.03    # temp->dew coupling
    boundary = dict(
        name="boundary_break",
        A=bb_A,
        b=[0.0, 0.0, 0.0, 0.0, 0.0],
        Q=_diag([3.0, 2.5, 0.8, 0.15, 0.15]),
        R=_diag([0.6, 0.6, 0.25, 0.06, 0.06]),
        self_transition_prob=0.70,
        mu_shift_high=0.0,
        mu_shift_low=0.0,
        sigma_scale_high=1.0,
        sigma_scale_low=1.3,
        active_families=["seabreeze", "convective"],
    )

    # 4. heat_overrun — slight warming tendency
    hr_A = _identity_scaled(0.99)
    hr_A[0][0] = 0.998   # temp persists strongly
    heat = dict(
        name="heat_overrun",
        A=hr_A,
        b=[0.1, 0.02, 0.0, 0.0, 0.0],
        Q=_diag([0.5, 0.5, 0.15, 0.03, 0.03]),
        R=_diag([0.4, 0.4, 0.15, 0.04, 0.04]),
        self_transition_prob=0.88,
        mu_shift_high=0.5,
        mu_shift_low=0.0,
        sigma_scale_high=0.9,
        sigma_scale_low=1.0,
        active_families=["heat", "radiation"],
    )

    return [mixed, postfrontal, boundary, heat]


# ---------------------------------------------------------------------------
# Switching Kalman Filter
# ---------------------------------------------------------------------------

class SwitchingKalmanFilter:
    """
    Maintains parallel Kalman filters — one per regime — and updates
    regime probabilities based on per-regime innovation likelihoods.
    """

    N_STATES = 5

    def __init__(self, config_path: str | Path | None = None):
        """Load regime models from JSON config.  Falls back to defaults."""
        self.regimes: list[RegimeModel] = []
        self._load_config(config_path)

        # Per-regime filter state (initialised in reset())
        self._x: np.ndarray | None = None          # 5x1 current state
        self._P: dict[str, np.ndarray] = {}         # regime -> 5x5 covariance
        self._log_pi: dict[str, float] = {}         # log regime probs
        self._prev_regime: str | None = None

    # ---- config loading ----------------------------------------------------

    def _load_config(self, config_path: str | Path | None) -> None:
        path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        if path.exists():
            logger.info("Loading SKF config from %s", path)
            with open(path) as f:
                cfg = json.load(f)
            regime_dicts = cfg.get("regimes", cfg) if isinstance(cfg, dict) else cfg
            if isinstance(regime_dicts, dict):
                regime_dicts = list(regime_dicts.values())
            self.regimes = [self._dict_to_regime(d) for d in regime_dicts]
        else:
            logger.info("No SKF config at %s — using defaults", path)
            self.regimes = [self._dict_to_regime(d) for d in _default_regime_models()]

    @staticmethod
    def _dict_to_regime(d: dict) -> RegimeModel:
        return RegimeModel(
            name=d["name"],
            A=d["A"],
            b=d["b"],
            Q=d["Q"],
            R=d["R"],
            self_transition_prob=d["self_transition_prob"],
            mu_shift_high=d.get("mu_shift_high", 0.0),
            mu_shift_low=d.get("mu_shift_low", 0.0),
            sigma_scale_high=d.get("sigma_scale_high", 1.0),
            sigma_scale_low=d.get("sigma_scale_low", 1.0),
            active_families=d.get("active_families", []),
        )

    # ---- lifecycle ---------------------------------------------------------

    def reset(self, target_date: str) -> None:
        """Reset for a new climate day.  Uniform priors, identity covariance."""
        logger.info("SKF reset for %s with %d regimes", target_date, len(self.regimes))
        n = self.N_STATES
        self._x = None  # will be set on first observation
        self._P = {r.name: np.eye(n) * 5.0 for r in self.regimes}
        uniform = -math.log(len(self.regimes))
        self._log_pi = {r.name: uniform for r in self.regimes}
        self._prev_regime = None

    # ---- Change detector integration ----------------------------------------

    def notify_changepoint(self, cp_probability: float, channels: list[str] | None = None) -> None:
        """Incorporate a changepoint signal from the ChangeDetector.

        When the CUSUM/threshold detector fires, pushes the regime priors
        toward uniform — reducing confidence in the current regime and
        allowing the innovation likelihoods to rapidly re-select. The
        strength of the push is proportional to the changepoint probability.

        Called BEFORE update() each cycle.

        Parameters
        ----------
        cp_probability : float
            Changepoint probability (0-1) from the ChangeDetector.
        channels : list[str], optional
            Which obs channels triggered the detection (e.g., ["temp_f",
            "pressure_hpa", "wind_dir_sin"]). Logged for diagnostics.
        """
        if cp_probability < 0.05:
            return  # negligible, skip

        # Blend current priors toward uniform proportional to cp_probability
        n = len(self.regimes)
        uniform = -math.log(n)
        for name in self._log_pi:
            self._log_pi[name] = (
                (1.0 - cp_probability) * self._log_pi[name]
                + cp_probability * uniform
            )

        # Widen covariances — a changepoint means we're less certain
        # about the state trajectory
        inflate = 1.0 + 2.0 * cp_probability  # up to 3x at cp_prob=1.0
        for name in self._P:
            self._P[name] *= inflate

        ch_str = ", ".join(channels) if channels else "unknown"
        logger.info(
            "Changepoint signal (p=%.3f, channels=[%s]): priors pushed "
            "toward uniform, covariances inflated %.1fx",
            cp_probability, ch_str, inflate,
        )

    # ---- main update -------------------------------------------------------

    def update(self, obs: ObsVector, hours_elapsed: float) -> SKFState:
        """
        One predict-update cycle across all regime models.

        Returns an SKFState with regime probabilities, weighted mu/sigma
        adjustments, active families, and switch detection.
        """
        z = np.array(self._to_state_vector(obs))  # 5-vector observation

        # First observation — initialise state
        if self._x is None:
            self._x = z.copy()

        n = self.N_STATES
        H = np.eye(n)  # direct observation

        log_likelihoods: dict[str, float] = {}
        predicted_states: dict[str, np.ndarray] = {}
        predicted_covs: dict[str, np.ndarray] = {}
        innovations: dict[str, np.ndarray] = {}
        kalman_gains: dict[str, np.ndarray] = {}

        for regime in self.regimes:
            A = np.array(regime.A)
            b = np.array(regime.b)
            Q = np.array(regime.Q)
            R = np.array(regime.R)
            P = self._P[regime.name]

            # 1. Predict
            x_pred = A @ self._x + b
            P_pred = A @ P @ A.T + Q

            # 2. Innovation
            y = z - H @ x_pred

            # 3. Innovation covariance
            S = P_pred + R

            # 4. Log-likelihood
            try:
                S_inv = np.linalg.inv(S)
                sign, logdet = np.linalg.slogdet(S)
                if sign <= 0:
                    logdet = 30.0  # fallback for degenerate covariance
                ll = -0.5 * (y @ S_inv @ y + logdet + n * math.log(2 * math.pi))
            except np.linalg.LinAlgError:
                ll = -1e6  # degenerate — penalise heavily

            log_likelihoods[regime.name] = ll
            predicted_states[regime.name] = x_pred
            predicted_covs[regime.name] = P_pred
            innovations[regime.name] = y

            # 6. Kalman gain
            try:
                K = P_pred @ np.linalg.inv(S)
            except np.linalg.LinAlgError:
                K = np.zeros((n, n))
            kalman_gains[regime.name] = K

        # 5. Update regime log-probabilities
        new_log_pi: dict[str, float] = {}
        for regime in self.regimes:
            new_log_pi[regime.name] = (
                self._log_pi[regime.name]
                + log_likelihoods[regime.name]
                + math.log(max(regime.self_transition_prob, 1e-10))
            )

        # Normalise via log-sum-exp
        log_vals = list(new_log_pi.values())
        log_norm = self._log_sum_exp(log_vals)
        for name in new_log_pi:
            new_log_pi[name] -= log_norm
        self._log_pi = new_log_pi

        # Determine most likely regime
        best_regime = max(self._log_pi, key=self._log_pi.get)  # type: ignore[arg-type]
        regime_probs = {name: math.exp(lp) for name, lp in self._log_pi.items()}

        # 7. State update using most likely regime's Kalman gain
        K_best = kalman_gains[best_regime]
        x_pred_best = predicted_states[best_regime]
        y_best = innovations[best_regime]
        self._x = x_pred_best + K_best @ y_best

        # Update covariances for all regimes
        for regime in self.regimes:
            K = kalman_gains[regime.name]
            P_pred = predicted_covs[regime.name]
            self._P[regime.name] = (np.eye(n) - K @ H) @ P_pred

        # Compute probability-weighted outputs
        regime_map = {r.name: r for r in self.regimes}
        mu_shift_high = sum(
            regime_probs[r.name] * r.mu_shift_high for r in self.regimes
        )
        mu_shift_low = sum(
            regime_probs[r.name] * r.mu_shift_low for r in self.regimes
        )
        sigma_scale_high = sum(
            regime_probs[r.name] * r.sigma_scale_high for r in self.regimes
        )
        sigma_scale_low = sum(
            regime_probs[r.name] * r.sigma_scale_low for r in self.regimes
        )

        # Active families weighted by regime probability
        active_families: dict[str, float] = {}
        for regime in self.regimes:
            p = regime_probs[regime.name]
            for fam in regime.active_families:
                active_families[fam] = active_families.get(fam, 0.0) + p

        # Innovation norm (from best regime)
        innovation_norm = float(np.linalg.norm(y_best))

        # Regime switch detection
        regime_switch = (
            self._prev_regime is not None and best_regime != self._prev_regime
        )
        if regime_switch:
            logger.info(
                "SKF regime switch: %s -> %s (conf=%.3f)",
                self._prev_regime, best_regime, regime_probs[best_regime],
            )
        self._prev_regime = best_regime

        return SKFState(
            regime_probabilities=regime_probs,
            most_likely_regime=best_regime,
            regime_confidence=regime_probs[best_regime],
            mu_shift_high=mu_shift_high,
            mu_shift_low=mu_shift_low,
            sigma_scale_high=sigma_scale_high,
            sigma_scale_low=sigma_scale_low,
            active_families=active_families,
            innovation_norm=innovation_norm,
            regime_switch_detected=regime_switch,
        )

    # ---- trajectory prediction ---------------------------------------------

    def predict_trajectory(self, hours_ahead: int = 6) -> dict[str, list[float]]:
        """
        Predict future temp_f under each regime model, weighted by current
        probabilities.  Returns {regime_name: [temp_f at hour 1..hours_ahead]}.
        """
        if self._x is None:
            return {}

        trajectories: dict[str, list[float]] = {}
        for regime in self.regimes:
            A = np.array(regime.A)
            b = np.array(regime.b)
            x = self._x.copy()
            preds: list[float] = []
            for _ in range(hours_ahead):
                x = A @ x + b
                preds.append(float(x[0]))  # temp_f is index 0
            trajectories[regime.name] = preds

        return trajectories

    # ---- helpers -----------------------------------------------------------

    @staticmethod
    def _to_state_vector(obs: ObsVector) -> list[float]:
        """Convert ObsVector to 5D state [temp, dew, pressure, sin(wind), cos(wind)]."""
        rad = math.radians(obs.wind_dir_deg)
        return [
            obs.temp_f,
            obs.dew_f,
            obs.pressure_hpa,
            math.sin(rad),
            math.cos(rad),
        ]

    @staticmethod
    def _log_sum_exp(log_probs: list[float]) -> float:
        """Numerically stable log-sum-exp."""
        if not log_probs:
            return 0.0
        max_lp = max(log_probs)
        if max_lp == float("-inf"):
            return float("-inf")
        return max_lp + math.log(sum(math.exp(lp - max_lp) for lp in log_probs))


# ---------------------------------------------------------------------------
# Standalone smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    skf = SwitchingKalmanFilter()
    skf.reset("2026-03-19")

    # Simulate a few observations
    obs_sequence = [
        ObsVector(temp_f=78.0, dew_f=68.0, pressure_hpa=1015.0, wind_dir_deg=180.0, timestamp_utc="2026-03-19T12:00:00Z"),
        ObsVector(temp_f=79.5, dew_f=68.5, pressure_hpa=1014.8, wind_dir_deg=185.0, timestamp_utc="2026-03-19T12:05:00Z"),
        ObsVector(temp_f=80.0, dew_f=69.0, pressure_hpa=1014.5, wind_dir_deg=190.0, timestamp_utc="2026-03-19T12:10:00Z"),
        ObsVector(temp_f=76.0, dew_f=70.0, pressure_hpa=1016.0, wind_dir_deg=220.0, timestamp_utc="2026-03-19T12:15:00Z"),
    ]

    for i, obs in enumerate(obs_sequence):
        state = skf.update(obs, hours_elapsed=i * 5 / 60)
        print(f"\n--- Obs {i+1} ---")
        print(f"  Most likely: {state.most_likely_regime} ({state.regime_confidence:.3f})")
        print(f"  Probabilities: { {k: round(v, 4) for k, v in state.regime_probabilities.items()} }")
        print(f"  mu_shift: high={state.mu_shift_high:.3f}, low={state.mu_shift_low:.3f}")
        print(f"  sigma_scale: high={state.sigma_scale_high:.3f}, low={state.sigma_scale_low:.3f}")
        print(f"  Innovation norm: {state.innovation_norm:.3f}")
        print(f"  Switch detected: {state.regime_switch_detected}")

    # Trajectory prediction
    traj = skf.predict_trajectory(hours_ahead=4)
    print("\n--- Trajectory Prediction (4 hours) ---")
    for name, temps in traj.items():
        print(f"  {name}: {[round(t, 1) for t in temps]}")
