"""DS3M persistent state — full shadow system state with JSON serialization.

Stores particle filter state, regime dynamics, observation model, conformal
calibrator, and training history.  Follows the LETKFTuneState persistence
pattern: JSON with numpy arrays converted via .tolist() / np.array().
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from engine.ds3m.config import DS3MConfig

log = logging.getLogger(__name__)

UTC = timezone.utc


@dataclass
class DS3MState:
    """Full persistent state for the DS3M shadow system."""

    # Component states (stored as dicts, reconstructed on use)
    particle_state_dict: dict | None = None
    dynamics_dict: dict | None = None
    obs_model_dict: dict | None = None
    conformal_dict: dict | None = None

    # Config
    config: DS3MConfig = field(default_factory=DS3MConfig)

    # Tracking
    n_cycles: int = 0
    last_cycle_utc: str | None = None
    active_target_date: str | None = None

    # History for offline training (kept as lists of dicts for JSON)
    regime_posterior_history: list[list[float]] = field(default_factory=list)
    innovation_history: list[dict] = field(default_factory=list)

    # Regime discovery
    low_likelihood_streak: int = 0

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Convert state to JSON-safe dict."""
        return {
            "particle_state_dict": _numpy_safe(self.particle_state_dict),
            "dynamics_dict": _numpy_safe(self.dynamics_dict),
            "obs_model_dict": _numpy_safe(self.obs_model_dict),
            "conformal_dict": self.conformal_dict,
            "n_cycles": self.n_cycles,
            "last_cycle_utc": self.last_cycle_utc,
            "active_target_date": self.active_target_date,
            "regime_posterior_history": self.regime_posterior_history,
            "innovation_history": self.innovation_history,
            "low_likelihood_streak": self.low_likelihood_streak,
        }

    @classmethod
    def from_dict(cls, d: dict) -> DS3MState:
        """Reconstruct from dict loaded from JSON."""
        return cls(
            particle_state_dict=d.get("particle_state_dict"),
            dynamics_dict=d.get("dynamics_dict"),
            obs_model_dict=d.get("obs_model_dict"),
            conformal_dict=d.get("conformal_dict"),
            n_cycles=d.get("n_cycles", 0),
            last_cycle_utc=d.get("last_cycle_utc"),
            active_target_date=d.get("active_target_date"),
            regime_posterior_history=d.get("regime_posterior_history", []),
            innovation_history=d.get("innovation_history", []),
            low_likelihood_streak=d.get("low_likelihood_streak", 0),
        )

    def save(self, path: str | Path | None = None) -> None:
        """JSON serialization to disk."""
        p = Path(path or self.config.state_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2))
        log.debug("DS3MState saved to %s (%d cycles)", p, self.n_cycles)

    @classmethod
    def load(cls, path: str | Path | None = None) -> DS3MState:
        """Load from JSON.  Returns fresh state if file missing or corrupt."""
        default_path = DS3MConfig().state_path
        p = Path(path or default_path)
        if not p.exists():
            log.info("No DS3M state at %s — starting fresh", p)
            return cls()
        try:
            data = json.loads(p.read_text())
            state = cls.from_dict(data)
            log.info(
                "DS3M state loaded: %d cycles, target_date=%s",
                state.n_cycles,
                state.active_target_date,
            )
            return state
        except (json.JSONDecodeError, TypeError, KeyError) as exc:
            log.warning("Corrupt DS3M state at %s (%s), starting fresh", p, exc)
            return cls()

    # ------------------------------------------------------------------
    # Component accessors (lazy reconstruct from dicts)
    # ------------------------------------------------------------------

    def get_dynamics(self):
        """Reconstruct RegimeDynamics from stored dict, or build defaults."""
        from engine.ds3m.regime_dynamics import RegimeDynamics

        if self.dynamics_dict is not None:
            try:
                return RegimeDynamics.from_dict(self.dynamics_dict)
            except (KeyError, TypeError):
                log.warning("Failed to reconstruct RegimeDynamics, using defaults")
        return RegimeDynamics.from_catalog_defaults(
            k_regimes=self.config.k_regimes,
            self_prob=self.config.transition_self_prob,
        )

    def set_dynamics(self, dynamics) -> None:
        self.dynamics_dict = dynamics.to_dict()

    def get_obs_model(self):
        """Reconstruct ObservationModel from stored dict, or build defaults."""
        from engine.ds3m.observation_model import ObservationModel

        if self.obs_model_dict is not None:
            try:
                return ObservationModel.from_dict(self.obs_model_dict)
            except (KeyError, TypeError):
                log.warning("Failed to reconstruct ObservationModel, using defaults")
        return ObservationModel.from_defaults(self.config.obs_sigma_default_f)

    def set_obs_model(self, obs_model) -> None:
        self.obs_model_dict = obs_model.to_dict()

    def get_conformal(self):
        """Reconstruct ConformalCalibrator from stored dict, or build fresh."""
        from engine.ds3m.conformal_calibrator import ConformalCalibrator

        if self.conformal_dict is not None:
            try:
                return ConformalCalibrator.from_dict(self.conformal_dict)
            except (KeyError, TypeError):
                log.warning("Failed to reconstruct ConformalCalibrator, using defaults")
        return ConformalCalibrator(
            window=self.config.conformal_window,
            alpha=self.config.conformal_alpha,
        )

    def set_conformal(self, conformal) -> None:
        self.conformal_dict = conformal.to_dict()

    # ------------------------------------------------------------------
    # Day rollover
    # ------------------------------------------------------------------

    def reset_for_new_day(self, target_date: str) -> None:
        """Reset particle state when climate day rolls over.

        Keeps dynamics, obs model, conformal, and history intact — only
        clears particle cloud so it reinitializes from fresh obs.
        """
        prev = self.active_target_date
        self.active_target_date = target_date
        self.particle_state_dict = None
        self.low_likelihood_streak = 0
        log.info(
            "DS3M day rollover: %s -> %s (cleared particles, kept %d history entries)",
            prev,
            target_date,
            len(self.regime_posterior_history),
        )

    # ------------------------------------------------------------------
    # History management
    # ------------------------------------------------------------------

    def append_regime_posterior(self, posterior: list[float] | np.ndarray) -> None:
        """Record a regime posterior vector for offline training."""
        if isinstance(posterior, np.ndarray):
            posterior = posterior.tolist()
        self.regime_posterior_history.append(posterior)

    def append_innovation(self, innovation: dict) -> None:
        """Record an innovation for offline R training.

        Expected keys: source, innovation_f, timestamp_utc.
        """
        self.innovation_history.append(innovation)

    def trim_history(self, max_days: int = 60) -> None:
        """Keep history bounded.

        Assumes roughly 1 posterior per cycle (60s interval) for ~18h/day
        = ~1080 entries/day.  We keep max_days * 1100 entries.
        """
        max_entries = max_days * 1100
        if len(self.regime_posterior_history) > max_entries:
            excess = len(self.regime_posterior_history) - max_entries
            self.regime_posterior_history = self.regime_posterior_history[excess:]
            log.debug("Trimmed regime_posterior_history by %d entries", excess)

        max_innov = max_days * 1100
        if len(self.innovation_history) > max_innov:
            excess = len(self.innovation_history) - max_innov
            self.innovation_history = self.innovation_history[excess:]
            log.debug("Trimmed innovation_history by %d entries", excess)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _numpy_safe(d: dict | None) -> dict | None:
    """Recursively convert numpy arrays to lists in a dict for JSON."""
    if d is None:
        return None
    out = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, dict):
            out[k] = _numpy_safe(v)
        elif isinstance(v, (list, tuple)):
            out[k] = [
                x.tolist() if isinstance(x, np.ndarray) else x for x in v
            ]
        else:
            out[k] = v
    return out
