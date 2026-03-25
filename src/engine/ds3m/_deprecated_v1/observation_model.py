"""DS3M observation model — per-source Gaussian likelihood with adaptive R.

Each observation source (wethr, nws, iem, etc.) has its own error standard
deviation estimate that adapts online via the Sage-Husa EMA method, matching
the pattern used in letkf.py.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger(__name__)

# Default observation error std (deg F) per source.
_DEFAULT_SOURCE_R: dict[str, float] = {
    "wethr": 0.5,
    "nws": 0.4,
    "iem": 0.6,
}


@dataclass
class ObservationModel:
    """Per-source Gaussian observation likelihood with adaptive R."""

    source_r: dict[str, float] = field(default_factory=dict)
    sage_husa_b: float = 0.97
    r_min: float = 0.15
    r_max: float = 2.5

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def from_defaults(cls, obs_sigma: float = 0.5) -> ObservationModel:
        """Initialize with default R for common sources.

        Args:
            obs_sigma: fallback observation error std applied to any source
                       not in the built-in defaults.
        """
        source_r = dict(_DEFAULT_SOURCE_R)
        # Ensure at least the fallback is available
        source_r.setdefault("default", obs_sigma)
        log.info("ObservationModel initialized with %d sources", len(source_r))
        return cls(source_r=source_r)

    # ------------------------------------------------------------------
    # Likelihood
    # ------------------------------------------------------------------
    def log_likelihood(
        self,
        predicted_values: np.ndarray,
        observed_value: float,
        source_key: str,
    ) -> np.ndarray:
        """Gaussian log-likelihood per particle.

        Args:
            predicted_values: (N,) predicted state values from particles.
            observed_value: scalar observed value.
            source_key: identifies the observation source for R lookup.

        Returns:
            (N,) log-likelihood values.
        """
        r = self.source_r.get(source_key, self.source_r.get("default", 0.5))
        r = max(r, self.r_min)
        diff = predicted_values - observed_value
        # Gaussian log-likelihood: -0.5 * (diff/r)^2 - log(r) - 0.5*log(2*pi)
        return -0.5 * (diff / r) ** 2 - math.log(r) - 0.5 * math.log(2.0 * math.pi)

    # ------------------------------------------------------------------
    # Adaptive R (Sage-Husa)
    # ------------------------------------------------------------------
    def update_r(self, source_key: str, innovation: float) -> None:
        """Sage-Husa EMA update of observation error std for a source.

        Mirrors the pattern in letkf.py: EMA of innovation squared gives
        an estimate of R^2, then take sqrt and clamp.

        Args:
            source_key: observation source identifier.
            innovation: obs - predicted (scalar).
        """
        prev_r = self.source_r.get(
            source_key, self.source_r.get("default", 0.5)
        )
        b = self.sage_husa_b
        new_r_sq = (1.0 - b) * innovation ** 2 + b * prev_r ** 2
        new_r = math.sqrt(max(self.r_min ** 2, new_r_sq))
        new_r = max(self.r_min, min(self.r_max, new_r))
        self.source_r[source_key] = new_r

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "source_r": {k: round(v, 6) for k, v in self.source_r.items()},
            "sage_husa_b": self.sage_husa_b,
            "r_min": self.r_min,
            "r_max": self.r_max,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ObservationModel:
        return cls(
            source_r=dict(d["source_r"]),
            sage_husa_b=d.get("sage_husa_b", 0.97),
            r_min=d.get("r_min", 0.15),
            r_max=d.get("r_max", 2.5),
        )
