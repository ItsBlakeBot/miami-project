"""Platt scaling (logistic recalibration) for bracket probabilities.

Applies a learned logistic transform to raw bracket probabilities to
correct systematic miscalibration:

    p_cal = 1 / (1 + exp(-(a * logit(p_raw) + b)))

When a=1 and b=0, this is the identity (no change).  As settlement data
accumulates, a and b are fit via maximum-likelihood logistic regression
on a rolling buffer of (predicted, outcome) pairs.

Separate calibrators are maintained for HIGH and LOW markets.
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from scipy.optimize import minimize

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_MIN_SAMPLES = 10  # don't calibrate until we have this many samples
_BUFFER_SIZE = 500  # rolling window of recent (predicted, outcome) pairs
_CLIP_LO = 0.01
_CLIP_HI = 0.99
_LOGIT_CLIP = 1e-6  # avoid log(0) in logit


# ---------------------------------------------------------------------------
# Core math
# ---------------------------------------------------------------------------

def _logit(p: float) -> float:
    p = max(_LOGIT_CLIP, min(1.0 - _LOGIT_CLIP, p))
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


# ---------------------------------------------------------------------------
# PlattCalibrator
# ---------------------------------------------------------------------------

@dataclass
class PlattCalibrator:
    """Logistic recalibration parameters for one market type."""

    a: float = 1.0  # logistic slope  (identity default)
    b: float = 0.0  # logistic intercept
    n_samples: int = 0
    last_fit_utc: str | None = None

    def calibrate(self, p_raw: float) -> float:
        """Apply Platt scaling to a raw bracket probability.

        Returns p_raw unchanged if fewer than ``_MIN_SAMPLES`` observations
        have been collected.
        """
        if self.n_samples < _MIN_SAMPLES:
            return p_raw
        z = self.a * _logit(p_raw) + self.b
        p_cal = _sigmoid(z)
        return max(_CLIP_LO, min(_CLIP_HI, p_cal))

    def update(
        self,
        predicted_probs: list[float],
        outcomes: list[bool],
    ) -> None:
        """Refit a, b on the full history via maximum-likelihood.

        Uses L-BFGS-B on the negative log-likelihood:
            NLL = -sum(y_i * log(p_cal_i) + (1 - y_i) * log(1 - p_cal_i))
        """
        if len(predicted_probs) != len(outcomes):
            raise ValueError("predicted_probs and outcomes must have same length")
        n = len(predicted_probs)
        if n == 0:
            return

        self.n_samples = n

        # Pre-compute logits of raw probabilities
        logits = [_logit(p) for p in predicted_probs]
        ys = [1.0 if o else 0.0 for o in outcomes]

        def neg_log_likelihood(params: list[float]) -> float:
            a_, b_ = params
            nll = 0.0
            for li, yi in zip(logits, ys):
                z = a_ * li + b_
                p = _sigmoid(z)
                p = max(_LOGIT_CLIP, min(1.0 - _LOGIT_CLIP, p))
                nll -= yi * math.log(p) + (1.0 - yi) * math.log(1.0 - p)
            return nll

        result = minimize(
            neg_log_likelihood,
            x0=[self.a, self.b],
            method="L-BFGS-B",
            bounds=[(0.1, 10.0), (-5.0, 5.0)],
        )

        if result.success:
            self.a = float(result.x[0])
            self.b = float(result.x[1])
        else:
            log.warning("Platt calibrator fit did not converge: %s", result.message)

        self.last_fit_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Persistent state
# ---------------------------------------------------------------------------

@dataclass
class PlattCalibratorState:
    """Full state for save/load, with separate HIGH/LOW calibrators."""

    high: PlattCalibrator = field(default_factory=PlattCalibrator)
    low: PlattCalibrator = field(default_factory=PlattCalibrator)
    buffer: list[dict] = field(default_factory=list)
    # buffer entries: {predicted: float, outcome: bool, market_type: str, timestamp: str}

    # -- persistence --------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        blob = {
            "high": {"a": self.high.a, "b": self.high.b,
                     "n_samples": self.high.n_samples,
                     "last_fit_utc": self.high.last_fit_utc},
            "low":  {"a": self.low.a, "b": self.low.b,
                     "n_samples": self.low.n_samples,
                     "last_fit_utc": self.low.last_fit_utc},
            "buffer": self.buffer[-_BUFFER_SIZE:],
        }
        path.write_text(json.dumps(blob, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> PlattCalibratorState:
        path = Path(path)
        if not path.exists():
            return cls()
        try:
            blob = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("Could not load Platt state from %s: %s", path, exc)
            return cls()

        def _make_cal(d: dict) -> PlattCalibrator:
            return PlattCalibrator(
                a=d.get("a", 1.0),
                b=d.get("b", 0.0),
                n_samples=d.get("n_samples", 0),
                last_fit_utc=d.get("last_fit_utc"),
            )

        return cls(
            high=_make_cal(blob.get("high", {})),
            low=_make_cal(blob.get("low", {})),
            buffer=blob.get("buffer", [])[-_BUFFER_SIZE:],
        )


# ---------------------------------------------------------------------------
# Top-level integration helpers
# ---------------------------------------------------------------------------

def apply_platt_calibration(
    estimates: list[dict],
    state_path: str = "analysis_data/platt_state.json",
) -> list[dict]:
    """Load calibrator state and apply Platt scaling to each estimate.

    Each dict in *estimates* must contain at least:
        - ``probability`` (float): raw model probability
        - ``market_type`` (str): ``"high"`` or ``"low"``
        - ``ticker`` (str): bracket ticker

    Returns the same list with ``probability`` replaced by the calibrated
    value and a new ``raw_probability`` field preserving the original.
    """
    state = PlattCalibratorState.load(state_path)

    for est in estimates:
        p_raw = est["probability"]
        mtype = est["market_type"].lower()
        cal = state.high if mtype == "high" else state.low
        p_cal = cal.calibrate(p_raw)
        est["raw_probability"] = p_raw
        est["probability"] = round(p_cal, 4)

    return estimates


def update_platt_from_settlements(
    db_path: str,
    station: str,
    state_path: str = "analysis_data/platt_state.json",
) -> None:
    """Refit Platt calibrators from historical bracket estimates + settlements.

    Reads ``bracket_estimates`` and ``market_settlements`` from the DB,
    determines which brackets were YES (winning_side='yes') vs NO, and
    refits the HIGH and LOW calibrators independently.

    Should be called from the daily post-settlement workflow.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT
                be.ticker,
                be.market_type,
                be.probability  AS predicted,
                ms.winning_side
            FROM bracket_estimates be
            JOIN market_settlements ms
                ON be.ticker = ms.ticker
            WHERE be.station = ?
              AND ms.winning_side IS NOT NULL
              AND be.probability IS NOT NULL
            ORDER BY be.timestamp_utc
            """,
            (station,),
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        log.info("No settled bracket data found for station=%s — skipping Platt refit", station)
        return

    state = PlattCalibratorState.load(state_path)
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Rebuild rolling buffer from DB results (most recent _BUFFER_SIZE)
    buffer: list[dict] = []
    high_pred, high_out = [], []
    low_pred, low_out = [], []

    for row in rows:
        mtype = row["market_type"].lower()
        predicted = float(row["predicted"])
        outcome = row["winning_side"].lower() == "yes"

        buffer.append({
            "predicted": predicted,
            "outcome": outcome,
            "market_type": mtype,
            "timestamp": now_utc,
        })

        if mtype == "high":
            high_pred.append(predicted)
            high_out.append(outcome)
        else:
            low_pred.append(predicted)
            low_out.append(outcome)

    state.buffer = buffer[-_BUFFER_SIZE:]

    # Fit separate calibrators
    if high_pred:
        state.high.update(high_pred[-_BUFFER_SIZE:], high_out[-_BUFFER_SIZE:])
        log.info(
            "Platt HIGH refit: a=%.4f b=%.4f n=%d",
            state.high.a, state.high.b, state.high.n_samples,
        )
    if low_pred:
        state.low.update(low_pred[-_BUFFER_SIZE:], low_out[-_BUFFER_SIZE:])
        log.info(
            "Platt LOW refit: a=%.4f b=%.4f n=%d",
            state.low.a, state.low.b, state.low.n_samples,
        )

    state.save(state_path)
    log.info("Platt calibrator state saved to %s", state_path)
