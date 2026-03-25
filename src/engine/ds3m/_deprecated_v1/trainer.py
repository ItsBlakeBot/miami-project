"""DS3M offline training — daily parameter updates from settlement data.

Called after CLI settlement data arrives to update:
  1. Regime transition matrix (MLE from soft posteriors)
  2. Per-regime drift/sigma (from settlement errors)
  3. Per-source observation R (batch Sage-Husa from innovations)
"""

from __future__ import annotations

import logging
import math
import sqlite3
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Transition matrix training
# ---------------------------------------------------------------------------

def train_transition_matrix(
    regime_posteriors: list[list[float]],
    k_regimes: int,
    sticky_kappa: float = 50.0,
    dirichlet_alpha: float = 1.0,
) -> np.ndarray:
    """MLE update of K x K transition matrix from soft regime counts.

    For each consecutive pair of posteriors, compute soft transition counts:
      C[i,j] += p(regime=i at t) * p(regime=j at t+1)

    Add Dirichlet prior + sticky kappa to diagonal.  Normalize rows.

    Returns
    -------
    np.ndarray of shape (K, K), row-stochastic.
    """
    if len(regime_posteriors) < 2:
        log.warning("Need >= 2 posteriors for transition training, got %d", len(regime_posteriors))
        # Return uniform with sticky diagonal
        T = np.full((k_regimes, k_regimes), dirichlet_alpha)
        np.fill_diagonal(T, dirichlet_alpha + sticky_kappa)
        row_sums = T.sum(axis=1, keepdims=True)
        return T / np.maximum(row_sums, 1e-12)

    # Initialize counts with Dirichlet prior + sticky diagonal
    counts = np.full((k_regimes, k_regimes), dirichlet_alpha)
    np.fill_diagonal(counts, dirichlet_alpha + sticky_kappa)

    # Accumulate soft expected transition counts
    for t in range(1, len(regime_posteriors)):
        prev = np.array(regime_posteriors[t - 1])
        curr = np.array(regime_posteriors[t])
        # Outer product: joint probability under mean-field assumption
        counts += np.outer(prev, curr)

    # Normalize rows
    row_sums = counts.sum(axis=1, keepdims=True)
    T = counts / np.maximum(row_sums, 1e-12)

    log.info(
        "Transition matrix trained from %d posteriors, diag=[%.3f..%.3f]",
        len(regime_posteriors),
        T.diagonal().min(),
        T.diagonal().max(),
    )
    return T


# ---------------------------------------------------------------------------
# 2. Dynamics training from settlements
# ---------------------------------------------------------------------------

def train_dynamics_from_settlements(
    db: sqlite3.Connection,
    dynamics_dict: dict,
    regime_posteriors: list[list[float]],
    station: str = "KMIA",
    lookback_days: int = 30,
) -> dict:
    """Update per-regime drift/sigma from settlement errors.

    For each settled day in the lookback window:
      1. Get actual high/low from event_settlements
      2. Get the DS3M estimate that was logged for that day (from ds3m_estimates)
      3. Compute error = actual - model_estimate
      4. Update drift via regime-weighted EMA
      5. Update sigma via regime-weighted variance

    Returns updated dynamics dict.
    """
    from engine.ds3m.regime_dynamics import RegimeDynamics

    dynamics = RegimeDynamics.from_dict(dynamics_dict)

    # Fetch settled days with both high and low actuals
    rows = db.execute(
        """SELECT es_h.settlement_date,
                  es_h.actual_value_f AS actual_high,
                  es_l.actual_value_f AS actual_low
           FROM event_settlements es_h
           JOIN event_settlements es_l
             ON es_h.settlement_date = es_l.settlement_date
            AND es_h.station = es_l.station
           WHERE es_h.station = ?
             AND es_h.market_type = 'high'
             AND es_l.market_type = 'low'
             AND es_h.actual_value_f IS NOT NULL
             AND es_l.actual_value_f IS NOT NULL
           ORDER BY es_h.settlement_date DESC
           LIMIT ?""",
        (station, lookback_days),
    ).fetchall()

    if not rows:
        log.info("No settlements found for dynamics training")
        return dynamics.to_dict()

    # Fetch corresponding DS3M estimates for each settled day
    ema_alpha = 0.1  # EMA smoothing factor
    k = dynamics.k_regimes

    drift_high_accum = np.zeros(k)
    drift_low_accum = np.zeros(k)
    var_high_accum = np.zeros(k)
    var_low_accum = np.zeros(k)
    weight_accum = np.zeros(k)

    for row in rows:
        settle_date = row["settlement_date"]
        actual_high = row["actual_high"]
        actual_low = row["actual_low"]

        # Look up the last DS3M estimate for this settlement date
        est_row = db.execute(
            """SELECT probability, market_type, ticker
               FROM ds3m_estimates
               WHERE station = ? AND target_date = ?
               ORDER BY timestamp_utc DESC""",
            (station, settle_date),
        ).fetchall()

        if not est_row:
            continue

        # Look up the regime posterior that was active for this day
        # Use a simple index: most recent posteriors correspond to most recent days
        # This is approximate — production would use exact timestamps
        idx = min(len(regime_posteriors) - 1, rows.index(row))
        if idx < 0 or idx >= len(regime_posteriors):
            continue

        posterior = np.array(regime_posteriors[idx])
        if len(posterior) != k:
            continue

        # Compute errors as actual - drift (how much the remaining move overshot)
        error_high = actual_high - dynamics.drift_high
        error_low = actual_low - dynamics.drift_low

        # Regime-weighted accumulation
        drift_high_accum += posterior * error_high
        drift_low_accum += posterior * error_low
        var_high_accum += posterior * error_high ** 2
        var_low_accum += posterior * error_low ** 2
        weight_accum += posterior

    # Update drift and sigma via EMA blend with current values
    safe_weights = np.maximum(weight_accum, 1e-8)
    new_drift_high = drift_high_accum / safe_weights
    new_drift_low = drift_low_accum / safe_weights
    new_var_high = var_high_accum / safe_weights - new_drift_high ** 2
    new_var_low = var_low_accum / safe_weights - new_drift_low ** 2

    # Clamp variance floors
    new_sigma_high = np.sqrt(np.maximum(new_var_high, 0.25))
    new_sigma_low = np.sqrt(np.maximum(new_var_low, 0.25))

    # EMA blend with existing
    dynamics.drift_high = (1.0 - ema_alpha) * dynamics.drift_high + ema_alpha * new_drift_high
    dynamics.drift_low = (1.0 - ema_alpha) * dynamics.drift_low + ema_alpha * new_drift_low
    dynamics.sigma_high = (1.0 - ema_alpha) * dynamics.sigma_high + ema_alpha * new_sigma_high
    dynamics.sigma_low = (1.0 - ema_alpha) * dynamics.sigma_low + ema_alpha * new_sigma_low

    log.info(
        "Dynamics trained from %d settlements: drift_high=[%.3f..%.3f], sigma_high=[%.3f..%.3f]",
        len(rows),
        dynamics.drift_high.min(),
        dynamics.drift_high.max(),
        dynamics.sigma_high.min(),
        dynamics.sigma_high.max(),
    )
    return dynamics.to_dict()


# ---------------------------------------------------------------------------
# 3. Observation R training
# ---------------------------------------------------------------------------

def train_observation_r(
    innovation_history: list[dict],
    obs_model_dict: dict,
    sage_husa_b: float = 0.97,
) -> dict:
    """Batch update of per-source R from accumulated innovations.

    Each entry in innovation_history has:
      - source: str
      - innovation_f: float (obs - predicted)
      - timestamp_utc: str

    Applies Sage-Husa EMA in chronological order to update R per source.

    Returns updated obs_model dict.
    """
    from engine.ds3m.observation_model import ObservationModel

    obs_model = ObservationModel.from_dict(obs_model_dict)

    if not innovation_history:
        log.info("No innovations for R training")
        return obs_model.to_dict()

    # Sort by timestamp for chronological processing
    sorted_innov = sorted(innovation_history, key=lambda x: x.get("timestamp_utc", ""))

    n_updates = 0
    for entry in sorted_innov:
        source = entry.get("source", "default")
        innov_f = entry.get("innovation_f")
        if innov_f is None:
            continue
        obs_model.update_r(source, innov_f)
        n_updates += 1

    log.info(
        "Observation R trained from %d innovations across %d sources: %s",
        n_updates,
        len(obs_model.source_r),
        {k: round(v, 4) for k, v in obs_model.source_r.items()},
    )
    return obs_model.to_dict()


# ---------------------------------------------------------------------------
# 4. Daily training entry point
# ---------------------------------------------------------------------------

def run_daily_training(
    db_path: str | Path,
    ds3m_state: "DS3MState",
    station: str = "KMIA",
) -> dict:
    """Post-settlement training entry point.

    Steps:
      1. Check if enough settlements exist
      2. Train transition matrix from regime posterior history
      3. Train dynamics from settlement errors
      4. Train observation R from innovation history
      5. Update ds3m_state in place
      6. Return training metrics

    Parameters
    ----------
    db_path : path to the sqlite3 database.
    ds3m_state : DS3MState to update in place.
    station : ICAO station code.

    Returns
    -------
    dict with training metrics and status.
    """
    from engine.ds3m.state import DS3MState  # noqa: F811 — deferred to avoid circular

    metrics: dict = {"status": "skipped", "station": station}
    cfg = ds3m_state.config

    db = sqlite3.connect(str(db_path))
    db.row_factory = sqlite3.Row
    try:
        # 1. Check settlement count
        row = db.execute(
            """SELECT COUNT(DISTINCT settlement_date) AS n
               FROM event_settlements
               WHERE station = ? AND actual_value_f IS NOT NULL""",
            (station,),
        ).fetchone()
        n_settlements = row["n"] if row else 0
        metrics["n_settlements"] = n_settlements

        if n_settlements < cfg.min_settlements_for_training:
            log.info(
                "DS3M training skipped: only %d settlements (need %d)",
                n_settlements,
                cfg.min_settlements_for_training,
            )
            return metrics

        # 2. Train transition matrix
        if len(ds3m_state.regime_posterior_history) >= 2:
            new_T = train_transition_matrix(
                ds3m_state.regime_posterior_history,
                k_regimes=cfg.k_regimes,
                sticky_kappa=cfg.sticky_kappa,
                dirichlet_alpha=cfg.dirichlet_alpha,
            )
            dynamics = ds3m_state.get_dynamics()
            dynamics.transition_matrix = new_T
            ds3m_state.set_dynamics(dynamics)
            metrics["transition_diag_range"] = [
                round(float(new_T.diagonal().min()), 4),
                round(float(new_T.diagonal().max()), 4),
            ]
        else:
            metrics["transition_skipped"] = "insufficient posteriors"

        # 3. Train dynamics from settlement errors
        if ds3m_state.dynamics_dict is not None and len(ds3m_state.regime_posterior_history) > 0:
            updated_dyn = train_dynamics_from_settlements(
                db,
                ds3m_state.dynamics_dict,
                ds3m_state.regime_posterior_history,
                station=station,
                lookback_days=cfg.training_lookback_days,
            )
            ds3m_state.dynamics_dict = updated_dyn
            metrics["dynamics_updated"] = True
        else:
            metrics["dynamics_skipped"] = "no dynamics or posteriors"

        # 4. Train observation R from innovation history
        if ds3m_state.obs_model_dict is not None and len(ds3m_state.innovation_history) > 0:
            updated_obs = train_observation_r(
                ds3m_state.innovation_history,
                ds3m_state.obs_model_dict,
                sage_husa_b=cfg.sage_husa_b,
            )
            ds3m_state.obs_model_dict = updated_obs
            metrics["obs_r_updated"] = True

            # Clear processed innovations
            ds3m_state.innovation_history = []
        else:
            metrics["obs_r_skipped"] = "no obs model or innovations"

        # 5. Trim history to prevent unbounded growth
        ds3m_state.trim_history(max_days=60)

        metrics["status"] = "completed"
        log.info("DS3M daily training completed: %s", metrics)
        return metrics

    finally:
        db.close()
