"""Layer 3: Adaptive conformal prediction for bracket probability calibration.

Maintains rolling nonconformity scores per market type (high / low) and
uses them to shrink raw DS3M probabilities toward 0.5 by a conformal
margin.  This guards against systematic over-confidence while the
particle filter is still learning regime parameters.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger(__name__)

_MIN_UPDATES = 5  # need at least this many settlements before adjusting


@dataclass
class ConformalCalibrator:
    """Rolling conformal calibration for DS3M bracket probabilities."""

    window: int = 30
    alpha: float = 0.1
    scores_high: list[float] = field(default_factory=list)
    scores_low: list[float] = field(default_factory=list)
    n_updates_high: int = 0
    n_updates_low: int = 0

    # ------------------------------------------------------------------
    # Calibrate a single raw probability
    # ------------------------------------------------------------------

    def calibrate(self, raw_prob: float, market_type: str) -> float:
        """Adjust *raw_prob* toward 0.5 by the conformal margin.

        When fewer than ``_MIN_UPDATES`` settlements have been observed
        for this market type, the raw probability is returned unchanged.
        """
        scores = self._scores_for(market_type)
        n_updates = self._n_updates_for(market_type)

        if n_updates < _MIN_UPDATES or len(scores) == 0:
            return raw_prob

        margin = self._conformal_margin(scores)

        if raw_prob > 0.5:
            return max(0.5, raw_prob - margin)
        if raw_prob < 0.5:
            return min(0.5, raw_prob + margin)
        return raw_prob

    # ------------------------------------------------------------------
    # Update with settlement outcomes
    # ------------------------------------------------------------------

    def update(
        self,
        market_type: str,
        predicted_probs: dict[str, float],
        actual_outcomes: dict[str, bool],
    ) -> dict:
        """Compute nonconformity scores and append to rolling window.

        Parameters
        ----------
        market_type      : "high" or "low".
        predicted_probs  : ticker -> DS3M P(YES) before calibration.
        actual_outcomes  : ticker -> True if bracket settled YES.

        Returns
        -------
        dict with coverage diagnostics for this update.
        """
        new_scores: list[float] = []
        for ticker, pred in predicted_probs.items():
            if ticker not in actual_outcomes:
                continue
            outcome = 1.0 if actual_outcomes[ticker] else 0.0
            score = abs(outcome - pred)
            new_scores.append(score)

        scores = self._scores_for(market_type)
        scores.extend(new_scores)

        # Cap rolling window
        while len(scores) > self.window:
            scores.pop(0)

        # Increment update counter
        if market_type == "high":
            self.n_updates_high += 1
        else:
            self.n_updates_low += 1

        n_updates = self._n_updates_for(market_type)
        margin = self._conformal_margin(scores) if len(scores) > 0 else 0.0

        return {
            "market_type": market_type,
            "n_new_scores": len(new_scores),
            "window_size": len(scores),
            "n_updates": n_updates,
            "conformal_margin": round(margin, 6),
            "coverage_rate": round(self.coverage_rate(market_type), 4),
        }

    # ------------------------------------------------------------------
    # Coverage diagnostic
    # ------------------------------------------------------------------

    def coverage_rate(self, market_type: str) -> float:
        """Fraction of nonconformity scores within the conformal interval.

        Returns 1.0 if no scores are available yet.
        """
        scores = self._scores_for(market_type)
        if len(scores) == 0:
            return 1.0
        margin = self._conformal_margin(scores)
        n_covered = sum(1 for s in scores if s <= margin)
        return n_covered / len(scores)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "window": self.window,
            "alpha": self.alpha,
            "scores_high": list(self.scores_high),
            "scores_low": list(self.scores_low),
            "n_updates_high": self.n_updates_high,
            "n_updates_low": self.n_updates_low,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ConformalCalibrator:
        return cls(
            window=data.get("window", 30),
            alpha=data.get("alpha", 0.1),
            scores_high=list(data.get("scores_high", [])),
            scores_low=list(data.get("scores_low", [])),
            n_updates_high=data.get("n_updates_high", 0),
            n_updates_low=data.get("n_updates_low", 0),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _scores_for(self, market_type: str) -> list[float]:
        if market_type == "high":
            return self.scores_high
        return self.scores_low

    def _n_updates_for(self, market_type: str) -> int:
        if market_type == "high":
            return self.n_updates_high
        return self.n_updates_low

    def _conformal_margin(self, scores: list[float]) -> float:
        """(1 - alpha) quantile of the nonconformity scores."""
        if len(scores) == 0:
            return 0.0
        q = 1.0 - self.alpha
        return float(np.quantile(scores, q))
