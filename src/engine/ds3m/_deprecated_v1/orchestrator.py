"""DS3M shadow orchestrator — inference cycle + async loop.

Runs alongside the production pipeline in shadow mode, writing to ds3m_*
tables for comparison.  Does NOT affect production inference or trading.

Runnable standalone:
    python -m engine.ds3m.orchestrator --db miami_collector.db --date 2026-03-23

Or importable:
    from engine.ds3m.orchestrator import run_ds3m_shadow_cycle
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from engine.ds3m.config import DS3MConfig
from engine.ds3m.state import DS3MState

log = logging.getLogger(__name__)

UTC = timezone.utc


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _read_latest_obs(db: sqlite3.Connection, station: str, now_utc: datetime) -> dict | None:
    """Read latest observation from DB.

    Returns dict with temp_f, running_high_f, running_low_f, source,
    or None if no recent observation is available.
    """
    as_of = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    row = db.execute(
        """SELECT temperature_f, dew_point_f, wind_speed_mph,
                  timestamp_utc
           FROM observations WHERE station = ? AND temperature_f IS NOT NULL
             AND timestamp_utc <= ?
           ORDER BY timestamp_utc DESC LIMIT 1""",
        (station, as_of),
    ).fetchone()
    if not row:
        return None

    # Determine staleness — skip if obs is > 20 minutes old
    obs_ts = row["timestamp_utc"]
    try:
        from engine.climate_clock import parse_utc_timestamp
        obs_dt = parse_utc_timestamp(obs_ts)
        age_min = (now_utc - obs_dt).total_seconds() / 60.0
        if age_min > 20.0:
            log.debug("Latest obs is %.1f min old — stale", age_min)
            return None
    except Exception:
        pass

    return {
        "temp_f": row["temperature_f"],
        "dew_point_f": row["dew_point_f"],
        "wind_speed_mph": row["wind_speed_mph"],
        "timestamp_utc": obs_ts,
        "source": "observations",
    }


def _read_running_extremes(
    db: sqlite3.Connection, station: str, target_date: str, now_utc: datetime,
) -> tuple[float | None, float | None]:
    """Read running high/low for the climate day from observations.

    Prefers wethr NWS envelope (whole-degree F) over raw MAX/MIN.
    """
    from engine.climate_clock import climate_day_bounds_utc

    start_utc, end_utc = climate_day_bounds_utc(target_date)
    effective_end = min(end_utc, now_utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    start_str = start_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Try NWS envelope first
    row = db.execute(
        """SELECT MAX(wethr_high_nws_f), MIN(wethr_low_nws_f)
           FROM observations WHERE station = ?
             AND timestamp_utc >= ? AND timestamp_utc < ?
             AND wethr_high_nws_f IS NOT NULL""",
        (station, start_str, effective_end),
    ).fetchone()
    if row and row[0] is not None:
        return (row[0], row[1])

    # Fallback to raw temperature_f
    row = db.execute(
        """SELECT MAX(temperature_f), MIN(temperature_f)
           FROM observations WHERE station = ? AND temperature_f IS NOT NULL
             AND timestamp_utc >= ? AND timestamp_utc < ?""",
        (station, start_str, effective_end),
    ).fetchone()
    return (row[0], row[1]) if row else (None, None)


def _read_production_estimates(
    db: sqlite3.Connection, station: str, target_date: str,
) -> dict[str, float]:
    """Read latest production bracket_estimates for comparison.

    Returns {ticker: probability}.
    """
    rows = db.execute(
        """SELECT ticker, probability
           FROM bracket_estimates
           WHERE station = ? AND target_date = ?
           ORDER BY timestamp_utc DESC""",
        (station, target_date),
    ).fetchall()

    # Take the most recent estimate per ticker
    prod: dict[str, float] = {}
    for row in rows:
        ticker = row["ticker"]
        if ticker not in prod:
            prod[ticker] = row["probability"]
    return prod


def _read_active_brackets(db: sqlite3.Connection, target_date: str) -> list:
    """Read active brackets for the target date.

    Returns list of dicts with ticker, market_type, floor_strike, cap_strike,
    settlement_floor, settlement_ceil.
    """
    rows = db.execute(
        """SELECT ticker, market_type, floor_strike, cap_strike,
                  settlement_floor, settlement_ceil
           FROM active_brackets
           WHERE target_date = ?
           ORDER BY market_type, floor_strike""",
        (target_date,),
    ).fetchall()
    return [dict(r) for r in rows]


def _read_market_prices(db: sqlite3.Connection, target_date: str) -> dict[str, float]:
    """Read latest market prices from market_snapshots.

    Returns {ticker: best_yes_ask_cents}.
    """
    rows = db.execute(
        """SELECT ticker, best_yes_ask_cents
           FROM market_snapshots
           WHERE forecast_date = ?
             AND best_yes_ask_cents IS NOT NULL
           ORDER BY id DESC""",
        (target_date,),
    ).fetchall()

    prices: dict[str, float] = {}
    for row in rows:
        ticker = row["ticker"]
        if ticker not in prices:
            prices[ticker] = row["best_yes_ask_cents"]
    return prices


def _ensure_ds3m_tables(db: sqlite3.Connection) -> None:
    """Create ds3m_estimates and ds3m_comparison tables if needed."""
    db.executescript("""
        CREATE TABLE IF NOT EXISTS ds3m_estimates (
            station TEXT NOT NULL,
            target_date TEXT NOT NULL,
            market_type TEXT NOT NULL,
            ticker TEXT NOT NULL,
            probability REAL,
            raw_probability REAL,
            regime_posterior TEXT,
            ess REAL,
            timestamp_utc TEXT NOT NULL,
            PRIMARY KEY (station, target_date, ticker)
        );
        CREATE TABLE IF NOT EXISTS ds3m_comparison (
            station TEXT NOT NULL,
            target_date TEXT NOT NULL,
            ticker TEXT NOT NULL,
            ds3m_probability REAL,
            production_probability REAL,
            market_price_cents REAL,
            ds3m_edge REAL,
            production_edge REAL,
            timestamp_utc TEXT NOT NULL,
            PRIMARY KEY (station, target_date, ticker)
        );
    """)


def _write_ds3m_estimates(
    db: sqlite3.Connection,
    station: str,
    target_date: str,
    estimates: list[dict],
    timestamp_utc: str,
) -> int:
    """Write DS3M estimates to ds3m_estimates table."""
    count = 0
    for est in estimates:
        try:
            db.execute(
                """INSERT OR REPLACE INTO ds3m_estimates
                   (station, target_date, market_type, ticker, probability,
                    raw_probability, regime_posterior, ess, timestamp_utc)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    station,
                    target_date,
                    est.get("market_type", ""),
                    est["ticker"],
                    est.get("probability"),
                    est.get("raw_probability"),
                    json.dumps(est.get("regime_posterior", [])),
                    est.get("ess"),
                    timestamp_utc,
                ),
            )
            count += 1
        except Exception as exc:
            log.warning("Failed to write DS3M estimate for %s: %s", est.get("ticker"), exc)
    db.commit()
    return count


def _write_ds3m_comparison(
    db: sqlite3.Connection,
    station: str,
    target_date: str,
    comparisons: list[dict],
    timestamp_utc: str,
) -> int:
    """Write production vs DS3M comparison to ds3m_comparison table."""
    count = 0
    for comp in comparisons:
        try:
            db.execute(
                """INSERT OR REPLACE INTO ds3m_comparison
                   (station, target_date, ticker, ds3m_probability,
                    production_probability, market_price_cents,
                    ds3m_edge, production_edge, timestamp_utc)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    station,
                    target_date,
                    comp["ticker"],
                    comp.get("ds3m_probability"),
                    comp.get("production_probability"),
                    comp.get("market_price_cents"),
                    comp.get("ds3m_edge"),
                    comp.get("production_edge"),
                    timestamp_utc,
                ),
            )
            count += 1
        except Exception as exc:
            log.warning("Failed to write DS3M comparison for %s: %s", comp.get("ticker"), exc)
    db.commit()
    return count


# ---------------------------------------------------------------------------
# Bootstrap / initialization
# ---------------------------------------------------------------------------

def _initialize_from_production(ds3m_state: DS3MState, db_path: str | Path, station: str = "KMIA") -> None:
    """Bootstrap DS3M state from production system components.

    1. Load EMOS state for dynamics initialization
    2. Load LETKF R estimates for observation model
    3. Use regime catalog for transition matrix prior
    4. Sample initial particles from baseline distribution
    """
    from engine.ds3m.regime_dynamics import RegimeDynamics
    from engine.ds3m.observation_model import ObservationModel

    cfg = ds3m_state.config

    # 1. Initialize dynamics from regime catalog
    dynamics = RegimeDynamics.from_catalog_defaults(
        k_regimes=cfg.k_regimes,
        self_prob=cfg.transition_self_prob,
    )

    # Try to refine from EMOS state if available
    try:
        from engine.emos import EMOSState
        emos = EMOSState.load("analysis_data/emos_state.json")
        if emos.high and hasattr(emos.high, "sigma"):
            # Use EMOS sigma as a guide for dynamics sigma
            for k in range(min(cfg.k_regimes, len(dynamics.sigma_high))):
                dynamics.sigma_high[k] = max(dynamics.sigma_high[k], emos.high.sigma * 0.5)
        if emos.low and hasattr(emos.low, "sigma"):
            for k in range(min(cfg.k_regimes, len(dynamics.sigma_low))):
                dynamics.sigma_low[k] = max(dynamics.sigma_low[k], emos.low.sigma * 0.5)
        log.info("DS3M dynamics refined from EMOS state")
    except Exception:
        log.debug("EMOS state not available for DS3M bootstrap", exc_info=True)

    ds3m_state.set_dynamics(dynamics)

    # 2. Initialize observation model — try LETKF R estimates
    obs_model = ObservationModel.from_defaults(cfg.obs_sigma_default_f)
    try:
        from engine.letkf import LETKFTuneState
        tune = LETKFTuneState.load()
        if tune.initial_r_estimates:
            # Use LETKF R estimates as starting point for DS3M obs model
            for src, r_val in tune.initial_r_estimates.items():
                obs_model.source_r[src] = max(cfg.r_min, min(cfg.r_max, r_val))
            log.info("DS3M obs model initialized from LETKF R: %s", obs_model.source_r)
    except Exception:
        log.debug("LETKF tune state not available for DS3M bootstrap", exc_info=True)

    ds3m_state.set_obs_model(obs_model)

    # 3. Conformal calibrator starts fresh (no history yet)
    from engine.ds3m.conformal_calibrator import ConformalCalibrator
    conformal = ConformalCalibrator(
        window=cfg.conformal_window,
        alpha=cfg.conformal_alpha,
    )
    ds3m_state.set_conformal(conformal)

    log.info("DS3M state bootstrapped from production components")


def _initialize_particles(
    ds3m_state: DS3MState,
    obs_temp_f: float,
    running_high_f: float | None,
    running_low_f: float | None,
) -> dict:
    """Sample initial particle cloud from current observation.

    Returns particle_state_dict with keys:
      particles_high, particles_low, regime_ids, weights
    """
    cfg = ds3m_state.config
    rng = np.random.default_rng()
    n = cfg.n_particles
    k = cfg.k_regimes

    # Initialize high particles: current obs + noise (remaining upside)
    base_high = obs_temp_f if running_high_f is None else running_high_f
    particles_high = base_high + rng.normal(0.0, cfg.default_sigma_f, n)

    # Initialize low particles: current obs + noise (remaining downside)
    base_low = obs_temp_f if running_low_f is None else running_low_f
    particles_low = base_low + rng.normal(0.0, cfg.default_sigma_f, n)

    # Uniform regime assignment
    regime_ids = rng.integers(0, k, size=n)

    # Equal weights
    weights = np.full(n, 1.0 / n)

    particle_state = {
        "particles_high": particles_high.tolist(),
        "particles_low": particles_low.tolist(),
        "regime_ids": regime_ids.tolist(),
        "weights": weights.tolist(),
    }
    ds3m_state.particle_state_dict = particle_state

    log.info(
        "Particles initialized: n=%d, base_high=%.1f, base_low=%.1f",
        n, base_high, base_low,
    )
    return particle_state


# ---------------------------------------------------------------------------
# Main shadow cycle
# ---------------------------------------------------------------------------

def run_ds3m_shadow_cycle(
    db_path: str | Path,
    ds3m_state: DS3MState,
    running_high_f: float | None = None,
    running_low_f: float | None = None,
    market_prices: dict[str, float] | None = None,
    station: str = "KMIA",
    target_date: str | None = None,
    eval_time_utc: datetime | None = None,
) -> dict:
    """One DS3M shadow inference cycle.

    Steps:
      1. Read latest observation from DB
      2. Compute remaining moves: obs_remaining_high = obs_temp - running_high
      3. If no particle state, initialize from current obs + EMOS
      4. Run DIMMPF step (predict -> update -> resample)
      5. Extract bracket probabilities via weighted empirical CDF
      6. Apply conformal calibration
      7. Read production estimates for comparison
      8. Compute edges vs market
      9. Write to ds3m_estimates and ds3m_comparison tables
     10. Track regime discovery (low-likelihood detection)
     11. Increment cycle counter, save state periodically

    Returns summary dict.
    """
    from engine.bracket_pricer import Bracket
    from engine.climate_clock import climate_date_for_utc
    from engine.ds3m.density_estimator import ParticleDensityEstimator
    from engine.edge_detector import kalshi_fee_cents

    cfg = ds3m_state.config
    now_utc = eval_time_utc or datetime.now(UTC)
    timestamp_utc = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Determine target date using climate day logic (UTC-5 fixed, 05:00Z boundary)
    if target_date is None:
        target_date = climate_date_for_utc(now_utc)

    # Handle day rollover
    if ds3m_state.active_target_date != target_date:
        ds3m_state.reset_for_new_day(target_date)

    db = sqlite3.connect(str(db_path))
    db.row_factory = sqlite3.Row
    _ensure_ds3m_tables(db)

    try:
        # 1. Read latest observation
        obs = _read_latest_obs(db, station, now_utc)
        if obs is None:
            log.debug("DS3M cycle skipped — no recent observation")
            return {"status": "skipped", "reason": "no_obs", "target_date": target_date}

        obs_temp_f = obs["temp_f"]

        # 2. Read running extremes if not provided
        if running_high_f is None or running_low_f is None:
            db_high, db_low = _read_running_extremes(db, station, target_date, now_utc)
            if running_high_f is None:
                running_high_f = db_high
            if running_low_f is None:
                running_low_f = db_low

        # 3. Initialize if needed
        if ds3m_state.particle_state_dict is None:
            if ds3m_state.dynamics_dict is None:
                _initialize_from_production(ds3m_state, db_path, station)
            _initialize_particles(ds3m_state, obs_temp_f, running_high_f, running_low_f)

        # Load particle state
        ps = ds3m_state.particle_state_dict
        particles_high = np.array(ps["particles_high"])
        particles_low = np.array(ps["particles_low"])
        regime_ids = np.array(ps["regime_ids"], dtype=np.intp)
        weights = np.array(ps["weights"])

        # Reconstruct components
        dynamics = ds3m_state.get_dynamics()
        obs_model = ds3m_state.get_obs_model()
        conformal = ds3m_state.get_conformal()

        rng = np.random.default_rng()
        n = len(weights)

        # 4. DIMMPF step: predict -> update -> resample
        # Predict: regime transition + state propagation
        new_regime_ids = dynamics.predict_regime(regime_ids, rng)
        new_high, new_low = dynamics.predict_state(
            particles_high, particles_low, new_regime_ids, rng,
        )

        # Update: compute observation likelihood
        # Use current temp as observation against predicted remaining move
        log_lik_high = obs_model.log_likelihood(new_high, obs_temp_f, obs.get("source", "default"))
        log_lik_low = obs_model.log_likelihood(new_low, obs_temp_f, obs.get("source", "default"))
        log_lik = log_lik_high + log_lik_low

        # Update weights in log space for numerical stability
        log_weights = np.log(np.maximum(weights, 1e-300)) + log_lik
        max_lw = np.max(log_weights)
        log_weights -= max_lw  # shift for stability
        new_weights = np.exp(log_weights)
        weight_sum = new_weights.sum()
        if weight_sum > 0:
            new_weights /= weight_sum
        else:
            new_weights = np.full(n, 1.0 / n)
            log.warning("DS3M weight collapse — reset to uniform")

        # Compute ESS
        ess = 1.0 / np.sum(new_weights ** 2) if np.sum(new_weights ** 2) > 0 else 0.0

        # Resample if ESS too low
        if ess < cfg.ess_threshold_fraction * n:
            indices = rng.choice(n, size=n, p=new_weights)
            new_high = new_high[indices]
            new_low = new_low[indices]
            new_regime_ids = new_regime_ids[indices]
            new_weights = np.full(n, 1.0 / n)
            ess = float(n)

        # Compute regime posterior (marginal over particles)
        regime_posterior = np.zeros(dynamics.k_regimes)
        for k_idx in range(dynamics.k_regimes):
            regime_posterior[k_idx] = np.sum(new_weights[new_regime_ids == k_idx])
        regime_sum = regime_posterior.sum()
        if regime_sum > 0:
            regime_posterior /= regime_sum

        # Store updated particles
        ds3m_state.particle_state_dict = {
            "particles_high": new_high.tolist(),
            "particles_low": new_low.tolist(),
            "regime_ids": new_regime_ids.tolist(),
            "weights": new_weights.tolist(),
        }

        # Track regime posterior for offline training
        ds3m_state.append_regime_posterior(regime_posterior)

        # Track innovation for adaptive R
        predicted_mean = float(np.average(new_high, weights=new_weights))
        innovation = obs_temp_f - predicted_mean
        ds3m_state.append_innovation({
            "source": obs.get("source", "default"),
            "innovation_f": innovation,
            "timestamp_utc": timestamp_utc,
        })

        # Adaptive R update (online)
        obs_model.update_r(obs.get("source", "default"), innovation)
        ds3m_state.set_obs_model(obs_model)

        # 5. Extract bracket probabilities via weighted empirical CDF
        brackets_raw = _read_active_brackets(db, target_date)
        if not brackets_raw:
            log.debug("No active brackets for %s", target_date)
            ds3m_state.n_cycles += 1
            ds3m_state.last_cycle_utc = timestamp_utc
            return {
                "status": "ok",
                "target_date": target_date,
                "n_brackets": 0,
                "ess": round(ess, 1),
                "regime_posterior": regime_posterior.tolist(),
            }

        # Build Bracket objects
        brackets: list[Bracket] = []
        for br in brackets_raw:
            brackets.append(Bracket(
                ticker=br["ticker"],
                market_type=br["market_type"],
                floor_f=br["settlement_floor"],
                ceiling_f=br["settlement_ceil"],
                directional="range",
            ))

        # Convert remaining-move particles to absolute temperatures.
        # Brackets are in absolute °F, so: abs_high = running_high + remaining_high
        # and abs_low = running_low - remaining_low.
        abs_high = new_high + (running_high_f or 0.0)
        abs_low = (running_low_f or 0.0) - new_low

        density_est = ParticleDensityEstimator(bandwidth_f=cfg.kde_bandwidth_f)
        raw_probs = density_est.all_bracket_probabilities(
            abs_high, abs_low, new_weights, brackets,
            running_high_f=running_high_f,
            running_low_f=running_low_f,
        )

        # 6. Apply conformal calibration
        calibrated_probs: dict[str, float] = {}
        for b in brackets:
            raw_p = raw_probs.get(b.ticker)
            if raw_p is None:
                continue
            calibrated_probs[b.ticker] = conformal.calibrate(raw_p, b.market_type)

        ds3m_state.set_conformal(conformal)

        # 7. Read production estimates for comparison
        prod_estimates = _read_production_estimates(db, station, target_date)

        # 8. Read market prices
        if market_prices is None:
            market_prices = _read_market_prices(db, target_date)

        # Build estimate and comparison records
        estimates: list[dict] = []
        comparisons: list[dict] = []

        for b in brackets:
            ticker = b.ticker
            ds3m_prob = calibrated_probs.get(ticker)
            raw_prob = raw_probs.get(ticker)
            if ds3m_prob is None:
                continue

            est = {
                "ticker": ticker,
                "market_type": b.market_type,
                "probability": round(ds3m_prob, 6),
                "raw_probability": round(raw_prob, 6) if raw_prob is not None else None,
                "regime_posterior": regime_posterior.tolist(),
                "ess": round(ess, 1),
            }
            estimates.append(est)

            # Comparison with production + market
            mkt_price = market_prices.get(ticker)
            prod_prob = prod_estimates.get(ticker)

            ds3m_edge = None
            if mkt_price is not None:
                fee = kalshi_fee_cents(mkt_price)
                ds3m_edge = ds3m_prob * 100.0 - mkt_price - fee

            prod_edge = None
            if prod_prob is not None and mkt_price is not None:
                fee = kalshi_fee_cents(mkt_price)
                prod_edge = prod_prob * 100.0 - mkt_price - fee

            comparisons.append({
                "ticker": ticker,
                "ds3m_probability": round(ds3m_prob, 6),
                "production_probability": round(prod_prob, 6) if prod_prob is not None else None,
                "market_price_cents": mkt_price,
                "ds3m_edge": round(ds3m_edge, 2) if ds3m_edge is not None else None,
                "production_edge": round(prod_edge, 2) if prod_edge is not None else None,
            })

        # 9. Write to DB
        n_est = _write_ds3m_estimates(db, station, target_date, estimates, timestamp_utc)
        n_comp = _write_ds3m_comparison(db, station, target_date, comparisons, timestamp_utc)

        # 9b. Run DS3M paper trader on fresh estimates
        try:
            from engine.ds3m.paper_trader import run_ds3m_paper_trading
            pt_result = run_ds3m_paper_trading(db_path, station=station, target_date=target_date)
            notes.append(f"ds3m_paper_trader: {pt_result.get('entries', 0)} entries, {pt_result.get('exits', 0)} exits")
        except Exception:
            log.debug("DS3M paper trader skipped", exc_info=True)

        # 10. Track regime discovery (low-likelihood detection)
        avg_log_lik = float(np.mean(log_lik))
        if avg_log_lik < cfg.likelihood_gap_threshold:
            ds3m_state.low_likelihood_streak += 1
        else:
            ds3m_state.low_likelihood_streak = 0

        regime_gap_detected = ds3m_state.low_likelihood_streak >= cfg.gap_consecutive_cycles

        # 11. Increment cycle counter
        ds3m_state.n_cycles += 1
        ds3m_state.last_cycle_utc = timestamp_utc

        # Compute summary edge stats
        edges = [c["ds3m_edge"] for c in comparisons if c["ds3m_edge"] is not None]
        avg_edge = round(np.mean(edges), 2) if edges else None

        result = {
            "status": "ok",
            "target_date": target_date,
            "timestamp_utc": timestamp_utc,
            "n_brackets": len(estimates),
            "n_written": n_est,
            "n_comparisons": n_comp,
            "ess": round(ess, 1),
            "regime_posterior": regime_posterior.tolist(),
            "avg_edge": avg_edge,
            "avg_log_lik": round(avg_log_lik, 3),
            "regime_gap_detected": regime_gap_detected,
            "low_likelihood_streak": ds3m_state.low_likelihood_streak,
            "n_cycles": ds3m_state.n_cycles,
        }

        log.info(
            "DS3M cycle %d: %s | %d brackets | ESS=%.0f | avg_edge=%s | regime_gap=%s",
            ds3m_state.n_cycles,
            target_date,
            len(estimates),
            ess,
            avg_edge,
            regime_gap_detected,
        )
        return result

    finally:
        db.close()


# ---------------------------------------------------------------------------
# Async shadow loop
# ---------------------------------------------------------------------------

async def ds3m_shadow_loop(cfg, store, live=None) -> None:
    """Async shadow loop running every 60 seconds.

    Pattern matches inference_loop from runner.py:
      - Load or initialize DS3MState
      - On climate day rollover, reset particles + trigger daily training
      - Every 60s: read data, run cycle, log results
      - Save state every 10 cycles
      - On shutdown: save state
    """
    from engine.ds3m.trainer import run_daily_training

    ds3m_cfg = DS3MConfig.load()
    interval = ds3m_cfg.shadow_cycle_secs
    ds3m_state = DS3MState.load()
    ds3m_state.config = ds3m_cfg

    db_path = store._path
    prev_target_date: str | None = ds3m_state.active_target_date

    # Wait for initial data to accumulate
    await asyncio.sleep(30)

    log.info("DS3M shadow loop started (interval=%ds)", interval)

    try:
        while True:
            try:
                now_utc = datetime.now(UTC)

                # Pass live market prices if available
                market_prices_override = live.market_prices if live else None

                result = run_ds3m_shadow_cycle(
                    db_path,
                    ds3m_state,
                    market_prices=market_prices_override,
                    eval_time_utc=now_utc,
                )

                status = result.get("status", "unknown")
                current_date = result.get("target_date")

                # Day rollover: trigger daily training
                if prev_target_date is not None and current_date != prev_target_date:
                    log.info(
                        "DS3M day rollover %s -> %s — running daily training",
                        prev_target_date, current_date,
                    )
                    try:
                        train_metrics = run_daily_training(
                            db_path, ds3m_state, station="KMIA",
                        )
                        log.info("DS3M daily training: %s", train_metrics)
                    except Exception:
                        log.warning("DS3M daily training failed", exc_info=True)

                prev_target_date = current_date

                if status == "ok":
                    n_est = result.get("n_brackets", 0)
                    ess = result.get("ess", 0)
                    avg_edge = result.get("avg_edge")
                    log.info(
                        "DS3M shadow: %s | %d estimates | ESS=%.0f | edge=%s",
                        current_date, n_est, ess, avg_edge,
                    )

                # Save state periodically
                if ds3m_state.n_cycles % 12 == 0 and ds3m_state.n_cycles > 0:  # every ~60s at 5s cadence
                    ds3m_state.save()

            except Exception:
                log.warning("DS3M shadow cycle error", exc_info=True)

            await asyncio.sleep(interval)

    finally:
        # Save on shutdown
        ds3m_state.save()
        log.info("DS3M shadow loop stopped, state saved (%d cycles)", ds3m_state.n_cycles)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="DS3M shadow inference cycle")
    parser.add_argument("--db", required=True, help="Path to sqlite3 database")
    parser.add_argument("--date", default=None, help="Target date (YYYY-MM-DD)")
    parser.add_argument("--station", default="KMIA", help="ICAO station code")
    args = parser.parse_args()

    state = DS3MState.load()
    result = run_ds3m_shadow_cycle(
        args.db, state, station=args.station, target_date=args.date,
    )
    state.save()
    print(json.dumps(result, indent=2, default=str))
