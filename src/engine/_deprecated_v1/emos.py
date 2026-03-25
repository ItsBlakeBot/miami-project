"""Ensemble Model Output Statistics (EMOS) calibration layer.

Implements standard EMOS (Gneiting et al. 2005) for remaining-high and
remaining-low predictive distributions. Uses the closed-form Normal CRPS
for fast fitting via BFGS optimization.

Progression path (implement sequentially):
  1. Standard EMOS  (this module)
  2. SAR-SEMOS      (autoregressive error + seasonal Fourier terms)
  3. EMOS-GB        (gradient-boosted, automatic feature selection)

Key design decisions:
  - EMOS operates on remaining-target (not raw high/low)
  - Per-source forecast values are the predictors
  - Ensemble spread (if available) is a variance predictor
  - Fit via CRPS minimization using scipy BFGS
  - Rolling training window (default 40 climate days)

References:
  - Gneiting et al. (2005), MWR 133(5), "Calibrated Probabilistic
    Forecasting Using Ensemble Model Output Statistics"
  - Closed-form Normal CRPS: sigma * [z*(2*Phi(z)-1) + 2*phi(z) - 1/sqrt(pi)]
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from statistics import NormalDist

import numpy as np
from scipy.optimize import minimize

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_SQRT_PI = math.sqrt(math.pi)
_STD_NORM = NormalDist(0.0, 1.0)

# Minimum sigma to prevent numerical issues in CRPS
_SIGMA_FLOOR = 0.25  # °F


# ---------------------------------------------------------------------------
# Closed-form CRPS for Normal(mu, sigma)
# ---------------------------------------------------------------------------
def crps_normal(mu: float, sigma: float, obs: float) -> float:
    """Closed-form CRPS for N(mu, sigma) evaluated at observation.

    CRPS(N(mu,sigma), y) = sigma * [z*(2*Phi(z)-1) + 2*phi(z) - 1/sqrt(pi)]
    where z = (y - mu) / sigma.

    This is ~5000x faster than numerical integration.
    """
    sigma = max(sigma, _SIGMA_FLOOR)
    z = (obs - mu) / sigma
    phi_z = _STD_NORM.pdf(z)
    Phi_z = _STD_NORM.cdf(z)
    return sigma * (z * (2.0 * Phi_z - 1.0) + 2.0 * phi_z - 1.0 / _SQRT_PI)


def crps_normal_vectorized(
    mu: np.ndarray, sigma: np.ndarray, obs: np.ndarray
) -> np.ndarray:
    """Vectorized closed-form CRPS for arrays of Normal predictions."""
    sigma = np.maximum(sigma, _SIGMA_FLOOR)
    z = (obs - mu) / sigma
    from scipy.stats import norm

    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    return sigma * (z * (2.0 * Phi_z - 1.0) + 2.0 * phi_z - 1.0 / _SQRT_PI)


# ---------------------------------------------------------------------------
# Training sample dataclass
# ---------------------------------------------------------------------------
@dataclass
class EMOSTrainingSample:
    """One training row for EMOS fitting.

    EMOS operates on REMAINING MOVE, not absolute forecasts.
    This prevents double-correction: the forward curve already handles
    time-of-day effects, and EMOS corrects only the residual bias in
    the remaining-move prediction.

    For HIGH market:
      predicted_remaining = baseline_mu - running_high (how much more the model expects)
      actual_remaining    = CLI_high - running_high (how much more actually happened)
    For LOW market:
      predicted_remaining = running_low - baseline_mu (how much further down the model expects)
      actual_remaining    = running_low - CLI_low (how much further down it actually went)
    """

    target_date: str
    market_type: str  # "high" or "low"
    eval_utc: str
    hours_to_settlement: float

    # Predictor: individual source forecasts (source_key → forecast °F)
    source_forecasts: dict[str, float]

    # Predictor: predicted remaining move (from weighted baseline)
    predicted_remaining_f: float | None = None
    ensemble_spread: float | None = None

    # Predictor: consensus (weighted mean from baseline engine)
    consensus_f: float | None = None
    consensus_sigma: float | None = None

    # Target: actual remaining move (from CLI settlement)
    actual_remaining_f: float | None = None

    # Legacy compatibility
    observation_f: float | None = None


# ---------------------------------------------------------------------------
# EMOS coefficients
# ---------------------------------------------------------------------------
@dataclass
class EMOSCoefficients:
    """Fitted EMOS parameters for remaining-move calibration.

    Location model:  calibrated_remaining = a + b * predicted_remaining
    Scale model:     sigma^2 = c + d * ensemble_spread^2

    The calibrated remaining move is then added back to the running extreme:
      HIGH: calibrated_mu = running_high + calibrated_remaining
      LOW:  calibrated_mu = running_low - calibrated_remaining

    This ensures EMOS corrections scale naturally with time-of-day:
    late in the day, predicted_remaining is small → correction is small.
    """

    # Location (mean) parameters
    a: float = 0.0  # intercept (bias in remaining-move prediction)
    b: float = 1.0  # slope (systematic scaling error)

    # Scale (variance) parameters
    c: float = 1.0  # intercept (minimum variance)
    d: float = 0.5  # slope on ensemble spread squared

    # Metadata
    market_type: str = ""
    n_training_samples: int = 0
    mean_crps: float = float("inf")
    training_window_days: int = 40
    fit_utc: str = ""

    def predict_remaining(
        self,
        predicted_remaining: float,
        spread: float | None = None,
    ) -> tuple[float, float]:
        """Calibrate predicted remaining move → (calibrated_remaining, sigma).

        Args:
            predicted_remaining: Forward-curve's predicted remaining move (°F, ≥0)
            spread: Ensemble spread for variance prediction

        Returns:
            (calibrated_remaining, sigma) — remaining move in °F (≥0) and uncertainty.
        """
        remaining_cal = self.a + self.b * predicted_remaining
        remaining_cal = max(0.0, remaining_cal)  # remaining move can't be negative

        spread_val = spread if spread is not None else 1.0
        sigma_sq = self.c + self.d * spread_val**2
        sigma = max(math.sqrt(max(sigma_sq, 0.0)), _SIGMA_FLOOR)

        return remaining_cal, sigma

    # Legacy compatibility: predict() returns absolute mu for backward compat
    def predict(
        self,
        consensus_f: float,
        spread: float | None = None,
    ) -> tuple[float, float]:
        """Legacy: predict absolute mu/sigma. Use predict_remaining() for new code."""
        mu = self.a + self.b * consensus_f
        spread_val = spread if spread is not None else 1.0
        sigma_sq = self.c + self.d * spread_val**2
        sigma = max(math.sqrt(max(sigma_sq, 0.0)), _SIGMA_FLOOR)
        return mu, sigma

    def to_dict(self) -> dict:
        return {
            "a": round(self.a, 6),
            "b": round(self.b, 6),
            "c": round(self.c, 6),
            "d": round(self.d, 6),
            "market_type": self.market_type,
            "n_training_samples": self.n_training_samples,
            "mean_crps": round(self.mean_crps, 4),
            "training_window_days": self.training_window_days,
            "fit_utc": self.fit_utc,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EMOSCoefficients":
        return cls(
            a=d.get("a", 0.0),
            b=d.get("b", 1.0),
            c=d.get("c", 1.0),
            d=d.get("d", 0.5),
            market_type=d.get("market_type", ""),
            n_training_samples=d.get("n_training_samples", 0),
            mean_crps=d.get("mean_crps", float("inf")),
            training_window_days=d.get("training_window_days", 40),
            fit_utc=d.get("fit_utc", ""),
        )


# ---------------------------------------------------------------------------
# EMOS fitting
# ---------------------------------------------------------------------------
@dataclass
class EMOSFitConfig:
    """Configuration for EMOS fitting."""

    training_window_days: int = 40
    # Hard minimum: EMOS stays identity until this many remaining-move
    # training samples exist. With fewer, the coefficients are noise.
    # 25 per market type = ~25 settlement days with valid running extremes.
    min_training_samples: int = 25
    sigma_floor: float = _SIGMA_FLOOR
    # Bounds on parameters to prevent degenerate fits.
    # With weighted consensus (matching live inference), EMOS corrections should
    # be moderate. These bounds allow up to ~5°F shift at typical consensus values
    # while preventing sign flips or extreme overcorrection.
    a_bounds: tuple[float, float] = (-10.0, 10.0)
    b_bounds: tuple[float, float] = (0.5, 1.5)  # slope shouldn't flip sign or go extreme
    c_bounds: tuple[float, float] = (0.01, 25.0)  # positive variance floor
    d_bounds: tuple[float, float] = (0.0, 5.0)  # non-negative spread influence


def _emos_objective(
    params: np.ndarray,
    predicted_remaining: np.ndarray,
    spreads: np.ndarray,
    actual_remaining: np.ndarray,
) -> float:
    """Mean CRPS for EMOS remaining-move parameters.

    Calibrates: predicted_remaining → actual_remaining.
    Lower CRPS = better calibration.
    """
    a, b, c, d = params
    mu = np.maximum(0.0, a + b * predicted_remaining)  # remaining move ≥ 0
    sigma_sq = c + d * spreads**2
    sigma = np.sqrt(np.maximum(sigma_sq, _SIGMA_FLOOR**2))

    crps_vals = crps_normal_vectorized(mu, sigma, actual_remaining)
    return float(np.mean(crps_vals))


def fit_emos(
    samples: list[EMOSTrainingSample],
    market_type: str,
    config: EMOSFitConfig | None = None,
    fit_utc: str = "",
) -> EMOSCoefficients | None:
    """Fit EMOS coefficients from training samples via CRPS minimization.

    Args:
        samples: Training data with consensus forecasts and observations.
        market_type: "high" or "low".
        config: Fitting configuration.
        fit_utc: Timestamp of the fit (for metadata).

    Returns:
        EMOSCoefficients if fitting succeeds, None if insufficient data.
    """
    config = config or EMOSFitConfig()

    # Filter to relevant samples with remaining-move data
    valid = [
        s
        for s in samples
        if s.market_type == market_type
        and s.predicted_remaining_f is not None
        and s.actual_remaining_f is not None
    ]

    if len(valid) < config.min_training_samples:
        return None

    # Build arrays — remaining moves, not absolute values
    predicted_remaining = np.array([s.predicted_remaining_f for s in valid])
    actual_remaining = np.array([s.actual_remaining_f for s in valid])

    spreads = np.array(
        [
            s.ensemble_spread
            if s.ensemble_spread is not None
            else (s.consensus_sigma if s.consensus_sigma is not None else 1.0)
            for s in valid
        ]
    )

    # Initial guess: identity mapping (a=0, b=1) with moderate variance
    x0 = np.array([0.0, 1.0, 1.0, 0.5])

    bounds = [
        config.a_bounds,
        config.b_bounds,
        config.c_bounds,
        config.d_bounds,
    ]

    result = minimize(
        _emos_objective,
        x0,
        args=(predicted_remaining, spreads, actual_remaining),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 500, "ftol": 1e-8},
    )

    if not result.success and result.fun > 100.0:
        # Optimization failed badly — return None
        return None

    a, b, c, d = result.x
    mean_crps = float(result.fun)

    return EMOSCoefficients(
        a=float(a),
        b=float(b),
        c=float(c),
        d=float(d),
        market_type=market_type,
        n_training_samples=len(valid),
        mean_crps=mean_crps,
        training_window_days=config.training_window_days,
        fit_utc=fit_utc,
    )


# ---------------------------------------------------------------------------
# EMOS state manager (load / save / apply)
# ---------------------------------------------------------------------------
@dataclass
class EMOSState:
    """Persistent EMOS state for both market types."""

    high: EMOSCoefficients | None = None
    low: EMOSCoefficients | None = None

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "high": self.high.to_dict() if self.high else None,
            "low": self.low.to_dict() if self.low else None,
        }
        p.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "EMOSState":
        p = Path(path)
        if not p.exists():
            return cls()
        try:
            data = json.loads(p.read_text())
            return cls(
                high=(
                    EMOSCoefficients.from_dict(data["high"])
                    if data.get("high")
                    else None
                ),
                low=(
                    EMOSCoefficients.from_dict(data["low"])
                    if data.get("low")
                    else None
                ),
            )
        except (json.JSONDecodeError, KeyError):
            return cls()

    def calibrate(
        self,
        market_type: str,
        consensus_f: float,
        spread: float | None = None,
    ) -> tuple[float, float] | None:
        """Apply EMOS calibration to produce (mu, sigma).

        Returns None if no coefficients are available for this market type.
        """
        coeff = self.high if market_type == "high" else self.low
        if coeff is None:
            return None
        return coeff.predict(consensus_f, spread)


# ---------------------------------------------------------------------------
# Training data extraction from DB
# ---------------------------------------------------------------------------
def extract_training_samples(
    db,
    station: str,
    target_dates: list[str],
    *,
    market_type: str | None = None,
) -> list[EMOSTrainingSample]:
    """Extract EMOS training samples as REMAINING MOVES.

    For each settlement day, we compute:
      - predicted_remaining: weighted baseline mu - running extreme at the
        time models were freshest (~midday, when most models have updated)
      - actual_remaining: CLI settlement - running extreme at that same time

    This trains EMOS to correct forward-curve remaining-move predictions,
    which naturally scale with time-of-day (early = big moves, late = small).

    Uses settlement_source='cli' only (no DSM/preliminary data).
    """
    from datetime import datetime, timezone

    from .baseline_engine import BaselineEngine, BaselineEngineConfig
    from .source_registry import ForecastSourceSnapshot

    samples: list[EMOSTrainingSample] = []
    market_types = [market_type] if market_type else ["high", "low"]
    baseline_engine = BaselineEngine(BaselineEngineConfig(station=station))

    for tdate in target_dates:
        # Final CLI only
        settlement_rows = db.execute(
            """SELECT market_type, actual_value_f
               FROM event_settlements
               WHERE station = ? AND settlement_date = ?
                 AND actual_value_f IS NOT NULL
                 AND settlement_source = 'cli'""",
            (station, tdate),
        ).fetchall()
        if not settlement_rows:
            continue

        cli_vals: dict[str, float] = {}
        for sr in settlement_rows:
            cli_vals[sr["market_type"]] = sr["actual_value_f"]
        if not cli_vals:
            continue

        # Get running extremes at MIDDAY (~17:00Z = noon LST for KMIA).
        # This is the running high/low at the time most model forecasts have
        # been issued. Using the end-of-day extremes would give remaining=0
        # for every sample (since final extreme == CLI by definition).
        midday_utc = f"{tdate}T17:00:00Z"
        running = db.execute(
            """SELECT wethr_high_nws_f as run_high, wethr_low_nws_f as run_low
               FROM observations
               WHERE station = ? AND lst_date = ?
                 AND wethr_high_nws_f IS NOT NULL
                 AND timestamp_utc <= ?
               ORDER BY timestamp_utc DESC LIMIT 1""",
            (station, tdate, midday_utc),
        ).fetchone()

        run_high = running["run_high"] if running and running["run_high"] else None
        run_low = running["run_low"] if running and running["run_low"] else None

        # Build forecast snapshots (freshest per model/source)
        forecast_rows = db.execute(
            """SELECT source, model, forecast_high_f, forecast_low_f,
                      run_time, fetch_time_utc
               FROM model_forecasts
               WHERE station = ? AND forecast_date = ?
                 AND (forecast_high_f IS NOT NULL OR forecast_low_f IS NOT NULL)
               ORDER BY fetch_time_utc DESC""",
            (station, tdate),
        ).fetchall()
        if not forecast_rows:
            continue

        seen: dict[tuple[str, str], dict] = {}
        for row in forecast_rows:
            key = (row["model"], row["source"] or "unknown")
            if key not in seen:
                seen[key] = dict(row)

        snapshots: list[ForecastSourceSnapshot] = []
        for (model, source), row in seen.items():
            issued_str = row.get("run_time") or row.get("fetch_time_utc") or ""
            try:
                issued = datetime.fromisoformat(issued_str.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                issued = datetime.now(timezone.utc)
            snapshots.append(ForecastSourceSnapshot(
                source_name=f"{source}:{model}",
                family_name=source or "unknown",
                issued_at_utc=issued,
                target_date=tdate,
                forecast_high_f=row.get("forecast_high_f"),
                forecast_low_f=row.get("forecast_low_f"),
            ))
        if not snapshots:
            continue

        for mt in market_types:
            cli_val = cli_vals.get(mt)
            if cli_val is None:
                continue

            try:
                belief = baseline_engine.build_baseline(
                    snapshots, market_type=mt, target_date=tdate,
                )
            except Exception:
                continue
            if belief.distribution.mu is None:
                continue

            weighted_mu = belief.distribution.mu
            weighted_sigma = belief.distribution.sigma or 1.0

            # Compute remaining moves
            if mt == "high":
                if run_high is None:
                    continue
                predicted_remaining = max(0.0, weighted_mu - run_high)
                actual_remaining = max(0.0, cli_val - run_high)
            else:  # low
                if run_low is None:
                    continue
                predicted_remaining = max(0.0, run_low - weighted_mu)
                actual_remaining = max(0.0, run_low - cli_val)

            source_fcsts = {}
            for snap in snapshots:
                val = snap.forecast_high_f if mt == "high" else snap.forecast_low_f
                if val is not None:
                    source_fcsts[snap.source_name] = float(val)
            vals = list(source_fcsts.values())
            spread = float(np.std(vals)) if len(vals) > 1 else weighted_sigma

            samples.append(
                EMOSTrainingSample(
                    target_date=tdate,
                    market_type=mt,
                    eval_utc="",
                    hours_to_settlement=0.0,
                    source_forecasts=source_fcsts,
                    predicted_remaining_f=predicted_remaining,
                    ensemble_spread=spread,
                    consensus_f=weighted_mu,
                    consensus_sigma=weighted_sigma,
                    actual_remaining_f=actual_remaining,
                    observation_f=float(cli_val),
                )
            )

    return samples


# ---------------------------------------------------------------------------
# Convenience: fit from DB
# ---------------------------------------------------------------------------
def fit_emos_from_db(
    db,
    station: str,
    lookback_days: int = 40,
    min_samples: int = 25,
    reference_date: str | None = None,
) -> EMOSState:
    """Fit EMOS for both market types from the database.

    Args:
        db: SQLite connection with row_factory=sqlite3.Row
        station: Station identifier
        lookback_days: Number of climate days to look back for training
        min_samples: Minimum training samples required
        reference_date: Date to compute lookback from (default: today)

    Returns:
        EMOSState with fitted coefficients (or None for insufficient data).
    """
    from datetime import date, timedelta

    if reference_date is None:
        ref = date.today()
    else:
        ref = date.fromisoformat(reference_date)

    # Generate target dates for training window
    target_dates = [
        (ref - timedelta(days=i)).isoformat() for i in range(1, lookback_days + 1)
    ]

    config = EMOSFitConfig(
        training_window_days=lookback_days,
        min_training_samples=min_samples,
    )

    samples = extract_training_samples(db, station, target_dates)

    now_utc = ref.isoformat() + "T00:00:00Z"

    high_coeff = fit_emos(samples, "high", config, fit_utc=now_utc)
    low_coeff = fit_emos(samples, "low", config, fit_utc=now_utc)

    return EMOSState(high=high_coeff, low=low_coeff)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """Fit EMOS coefficients from the database and save to JSON."""
    import argparse
    import sqlite3

    parser = argparse.ArgumentParser(description="Fit EMOS calibration coefficients")
    parser.add_argument("--db", required=True, help="Path to miami_collector.db")
    parser.add_argument("--station", default="KMIA", help="Station identifier")
    parser.add_argument("--lookback-days", type=int, default=40, help="Training window in climate days")
    parser.add_argument("--min-samples", type=int, default=25, help="Minimum training samples")
    parser.add_argument("--reference-date", default=None, help="Reference date (YYYY-MM-DD)")
    parser.add_argument("--out", default="analysis_data/emos_state.json", help="Output path for coefficients")
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row

    print(f"Fitting EMOS for {args.station} with {args.lookback_days}-day lookback...")
    state = fit_emos_from_db(
        conn,
        station=args.station,
        lookback_days=args.lookback_days,
        min_samples=args.min_samples,
        reference_date=args.reference_date,
    )
    conn.close()

    state.save(args.out)

    for mt in ("high", "low"):
        coeff = state.high if mt == "high" else state.low
        if coeff is not None:
            print(f"  {mt}: a={coeff.a:.4f}, b={coeff.b:.4f}, c={coeff.c:.4f}, d={coeff.d:.4f}")
            print(f"        samples={coeff.n_training_samples}, mean_crps={coeff.mean_crps:.4f}")
        else:
            print(f"  {mt}: insufficient data")

    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
