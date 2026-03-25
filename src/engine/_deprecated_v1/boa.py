"""Bernstein Online Aggregation (BOA) with sleeping experts for source weighting.

Implements online learning for combining probabilistic forecasts from
multiple NWP sources that update at different frequencies. Uses the
sleeping experts framework (Freund et al. 1997, Kleinberg et al. 2010)
to ensure fair comparison: models are only evaluated in rounds when
they have issued a new forecast.

Key properties:
  - Sleeping expert regret: O(sqrt(T_k * log(K))) where T_k = rounds expert k
    was awake. This is information-theoretically optimal.
  - Models that refresh more often (HRRR hourly) get more update opportunities
    but NO per-round advantage over less frequent models (GFS 6-hourly).
  - CRPS loss for calibration-aware weighting.
  - Forget rate for non-stationarity (seasonal adaptation).
  - Second-order Bernstein correction for fast convergence.

Design:
  - Each source produces a point forecast (mu_source) and uncertainty (sigma_source)
  - After settlement, only "awake" sources (those with a NEW forecast since last
    evaluation) get their weights updated
  - Sources without new forecasts keep their current weights unchanged
  - T1.4 trust multipliers serve as initial prior weights
  - BOA takes over ongoing learning after initialization

References:
  - Wintenberger (2017), Machine Learning — original BOA theory
  - Berrisch & Ziel (2021), J. Econometrics — CRPS Learning application
  - Kleinberg, Niculescu-Mizil, Sharma (2010) — sleeping expert regret bounds
  - Pfitzner et al. (2025), arXiv:2506.15216 — sleeping experts for temperature
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from scipy.stats import norm as _norm_dist

from .emos import crps_normal

# ---------------------------------------------------------------------------
# Threshold-weighted CRPS (Gneiting & Ranjan 2011)
# ---------------------------------------------------------------------------
_TWCRPS_GRID_POINTS = 200
_TWCRPS_SIGMA_RANGE = 5.0  # integrate over mu ± 5*sigma
_TWCRPS_BUMP_H = 1.0       # bandwidth for Gaussian bumps (°F)


def compute_twcrps_normal(
    mu: float,
    sigma: float,
    obs: float,
    bracket_boundaries: list[float],
    *,
    n_points: int = _TWCRPS_GRID_POINTS,
    h: float = _TWCRPS_BUMP_H,
) -> float:
    """Threshold-weighted CRPS for N(mu, sigma) via numerical quadrature.

    twCRPS(F, y; w) = integral of w(t) * (F(t) - 1{y <= t})^2 dt

    The weight function w(t) is a sum of Gaussian bumps centered on bracket
    boundaries, so the score emphasizes accuracy near the boundaries where
    trading decisions are made.

    Args:
        mu: Forecast mean (°F).
        sigma: Forecast std dev (°F).
        obs: Observed value (°F).
        bracket_boundaries: Bracket boundary temperatures (°F).
        n_points: Number of quadrature grid points.
        h: Bandwidth for Gaussian bumps around boundaries (°F).

    Returns:
        twCRPS value (non-negative; lower is better).
    """
    sigma = max(sigma, 0.1)
    # Integration grid: cover the forecast distribution and all boundaries
    lo = min(mu - _TWCRPS_SIGMA_RANGE * sigma, min(bracket_boundaries) - 4 * h)
    hi = max(mu + _TWCRPS_SIGMA_RANGE * sigma, max(bracket_boundaries) + 4 * h)
    t = np.linspace(lo, hi, n_points)
    dt = t[1] - t[0]

    # Weight function: sum of Gaussian bumps on boundaries
    w = np.zeros(n_points)
    for b in bracket_boundaries:
        w += np.exp(-0.5 * ((t - b) / h) ** 2)

    # CDF of N(mu, sigma) at grid points
    F_t = _norm_dist.cdf(t, loc=mu, scale=sigma)

    # Indicator: 1{obs <= t}
    ind = (obs <= t).astype(float)

    # twCRPS = integral w(t) * (F(t) - 1{y<=t})^2 dt  (trapezoid rule)
    integrand = w * (F_t - ind) ** 2
    return float(np.trapz(integrand, dx=dt))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_DEFAULT_LEARNING_RATE = 0.5
_MIN_WEIGHT = 1e-6  # prevent any source from reaching zero weight


@dataclass
class BOAConfig:
    """Configuration for BOA online learning."""

    learning_rate: float = _DEFAULT_LEARNING_RATE
    min_weight: float = _MIN_WEIGHT
    # Forget rate: fraction of cumulative loss to discard each step
    # (0.0 = full memory, 0.05 = forget 5% per step for non-stationarity)
    forget_rate: float = 0.02
    # Use second-order correction (Bernstein-style)
    use_second_order: bool = True


# ---------------------------------------------------------------------------
# Per-source state
# ---------------------------------------------------------------------------
@dataclass
class SourceState:
    """Tracking state for a single forecast source (sleeping expert)."""

    source_key: str
    weight: float = 1.0
    cumulative_loss: float = 0.0
    cumulative_squared_loss: float = 0.0
    n_updates: int = 0       # rounds this expert was awake and evaluated
    n_rounds_total: int = 0  # total rounds (including sleeping)
    last_crps: float | None = None
    last_run_time: str | None = None  # ISO timestamp of last model run ingested

    def to_dict(self) -> dict:
        return {
            "source_key": self.source_key,
            "weight": round(self.weight, 8),
            "cumulative_loss": round(self.cumulative_loss, 6),
            "cumulative_squared_loss": round(self.cumulative_squared_loss, 6),
            "n_updates": self.n_updates,
            "n_rounds_total": self.n_rounds_total,
            "last_crps": round(self.last_crps, 6) if self.last_crps is not None else None,
            "last_run_time": self.last_run_time,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SourceState":
        return cls(
            source_key=d["source_key"],
            weight=d.get("weight", 1.0),
            cumulative_loss=d.get("cumulative_loss", 0.0),
            cumulative_squared_loss=d.get("cumulative_squared_loss", 0.0),
            n_updates=d.get("n_updates", 0),
            n_rounds_total=d.get("n_rounds_total", 0),
            last_crps=d.get("last_crps"),
            last_run_time=d.get("last_run_time"),
        )


# ---------------------------------------------------------------------------
# BOA state (one per market type)
# ---------------------------------------------------------------------------
@dataclass
class BOAState:
    """Full BOA state for one market type with sleeping experts."""

    market_type: str
    sources: dict[str, SourceState] = field(default_factory=dict)
    config: BOAConfig = field(default_factory=BOAConfig)
    total_updates: int = 0

    def get_weights(self) -> dict[str, float]:
        """Return normalized weight dictionary."""
        if not self.sources:
            return {}
        raw = {k: max(v.weight, self.config.min_weight) for k, v in self.sources.items()}
        total = sum(raw.values())
        if total <= 0:
            n = len(raw)
            return {k: 1.0 / n for k in raw}
        return {k: v / total for k, v in raw.items()}

    def update(
        self,
        source_forecasts: dict[str, tuple[float, float]],
        observation: float,
        awake_sources: set[str] | None = None,
        bracket_boundaries: list[float] | None = None,
    ) -> dict[str, float]:
        """Update weights after observing settlement (sleeping expert protocol).

        Args:
            source_forecasts: {source_key: (mu, sigma)} — each source's
                predictive distribution at the time of forecast.
            observation: The actual settlement value (°F).
            awake_sources: Set of source keys that issued a NEW forecast since
                the last evaluation. If None, ALL sources in source_forecasts
                are treated as awake (backward-compatible).
            bracket_boundaries: Active bracket boundary temperatures (°F).
                When provided, uses threshold-weighted CRPS (twCRPS) instead
                of standard CRPS, emphasizing accuracy near bracket edges
                where trading decisions are made (Gneiting & Ranjan 2011).
                When None, uses standard closed-form CRPS (backward-compatible).

        Returns:
            Updated normalized weights.

        Sleeping expert protocol:
            - Only "awake" sources get their weights updated (CRPS computed,
              cumulative loss adjusted, weight recalculated).
            - "Sleeping" sources keep their current weights unchanged.
            - This ensures models that refresh less often (GFS 6h) are not
              penalized for rounds when they had no new information.
            - Regret bound: O(sqrt(T_k * log(K))) per expert k, where T_k
              is the number of rounds k was awake (Kleinberg et al. 2010).
        """
        if awake_sources is None:
            awake_sources = set(source_forecasts.keys())

        eta = self.config.learning_rate

        # Ensure all sources have state
        for key in source_forecasts:
            if key not in self.sources:
                self.sources[key] = SourceState(source_key=key)

        # Increment total round count for all known sources
        for key in self.sources:
            self.sources[key].n_rounds_total += 1

        # Compute loss ONLY for awake sources
        # Use twCRPS when bracket boundaries are provided, otherwise standard CRPS
        use_twcrps = bracket_boundaries is not None and len(bracket_boundaries) > 0
        losses: dict[str, float] = {}
        for key in awake_sources:
            if key not in source_forecasts:
                continue
            mu, sigma = source_forecasts[key]
            if use_twcrps:
                loss = compute_twcrps_normal(mu, sigma, observation, bracket_boundaries)
            else:
                loss = crps_normal(mu, sigma, observation)
            losses[key] = loss
            self.sources[key].last_crps = loss

        # Apply forget rate to ALL sources' cumulative losses
        forget = self.config.forget_rate
        if forget > 0:
            for key in self.sources:
                self.sources[key].cumulative_loss *= (1.0 - forget)
                self.sources[key].cumulative_squared_loss *= (1.0 - forget)

        # Update cumulative losses and weights ONLY for awake sources
        for key, loss in losses.items():
            state = self.sources[key]
            state.cumulative_loss += loss
            state.cumulative_squared_loss += loss * loss
            state.n_updates += 1

        # Recompute weights for ALL sources (including sleeping ones)
        # Sleeping sources keep their existing cumulative_loss unchanged,
        # so their weights only change due to the forget rate decay.
        for key, state in self.sources.items():
            exponent = -eta * state.cumulative_loss
            if self.config.use_second_order:
                # Bernstein correction: subtract squared term (Wintenberger 2017)
                # Penalizes experts with high-variance losses (stabilizes convergence)
                exponent -= eta * eta * state.cumulative_squared_loss
            state.weight = exponent  # store log-weight temporarily

        # Normalize via log-sum-exp for numerical stability
        if self.sources:
            log_weights = {k: v.weight for k, v in self.sources.items()}
            max_lw = max(log_weights.values()) if log_weights else 0.0
            for key in log_weights:
                self.sources[key].weight = math.exp(log_weights[key] - max_lw)

        self.total_updates += 1
        return self.get_weights()

    def to_dict(self) -> dict:
        return {
            "market_type": self.market_type,
            "total_updates": self.total_updates,
            "config": {
                "learning_rate": self.config.learning_rate,
                "forget_rate": self.config.forget_rate,
                "use_second_order": self.config.use_second_order,
            },
            "sources": {k: v.to_dict() for k, v in self.sources.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BOAState":
        cfg_d = d.get("config", {})
        config = BOAConfig(
            learning_rate=cfg_d.get("learning_rate", _DEFAULT_LEARNING_RATE),
            forget_rate=cfg_d.get("forget_rate", 0.02),
            use_second_order=cfg_d.get("use_second_order", True),
        )
        sources = {
            k: SourceState.from_dict(v)
            for k, v in d.get("sources", {}).items()
        }
        return cls(
            market_type=d.get("market_type", ""),
            sources=sources,
            config=config,
            total_updates=d.get("total_updates", 0),
        )


# ---------------------------------------------------------------------------
# Full BOA manager (both market types)
# ---------------------------------------------------------------------------
@dataclass
class BOAManager:
    """Manages BOA state for both high and low markets."""

    high: BOAState = field(default_factory=lambda: BOAState(market_type="high"))
    low: BOAState = field(default_factory=lambda: BOAState(market_type="low"))

    def get_weights(self, market_type: str) -> dict[str, float]:
        """Get current normalized weights for a market type."""
        state = self.high if market_type == "high" else self.low
        return state.get_weights()

    def update(
        self,
        market_type: str,
        source_forecasts: dict[str, tuple[float, float]],
        observation: float,
        awake_sources: set[str] | None = None,
        bracket_boundaries: list[float] | None = None,
    ) -> dict[str, float]:
        """Update weights after settlement (sleeping expert protocol).

        Args:
            market_type: "high" or "low"
            source_forecasts: {source_key: (mu, sigma)}
            observation: Settlement value (°F)
            awake_sources: Sources with new forecasts. None = all awake.
            bracket_boundaries: Bracket boundary temps (°F) for twCRPS.
                None = use standard CRPS.
        """
        state = self.high if market_type == "high" else self.low
        return state.update(source_forecasts, observation, awake_sources, bracket_boundaries)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "high": self.high.to_dict(),
            "low": self.low.to_dict(),
        }
        p.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "BOAManager":
        p = Path(path)
        if not p.exists():
            return cls()
        try:
            data = json.loads(p.read_text())
            return cls(
                high=BOAState.from_dict(data["high"]) if data.get("high") else BOAState(market_type="high"),
                low=BOAState.from_dict(data["low"]) if data.get("low") else BOAState(market_type="low"),
            )
        except (json.JSONDecodeError, KeyError):
            return cls()

    def summary(self) -> dict:
        """Produce a summary of current BOA state for logging."""
        result = {}
        for mt, state in [("high", self.high), ("low", self.low)]:
            weights = state.get_weights()
            top_sources = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:5]
            result[mt] = {
                "n_sources": len(state.sources),
                "total_updates": state.total_updates,
                "top_sources": {k: round(v, 4) for k, v in top_sources},
            }
        return result


# ---------------------------------------------------------------------------
# Batch initialization from historical data
# ---------------------------------------------------------------------------
def initialize_boa_from_history(
    db,
    station: str,
    lookback_days: int = 40,
    reference_date: str | None = None,
    config: BOAConfig | None = None,
) -> BOAManager:
    """Initialize BOA weights by replaying historical settlements.

    Uses sleeping expert protocol: for each settlement day, determines
    which models had a NEW forecast (different run_time from previous day)
    and only updates weights for those models.

    Args:
        db: SQLite connection with row_factory=sqlite3.Row
        station: Station identifier
        lookback_days: Number of days to replay
        reference_date: Reference date (default: today)
        config: BOA configuration

    Returns:
        BOAManager with weights initialized from historical replay.
    """
    from datetime import date, datetime, timedelta

    config = config or BOAConfig()

    if reference_date is None:
        ref = date.today()
    else:
        ref = date.fromisoformat(reference_date)

    # Generate dates in chronological order (oldest first)
    target_dates = [
        (ref - timedelta(days=i)).isoformat()
        for i in range(lookback_days, 0, -1)
    ]

    manager = BOAManager(
        high=BOAState(market_type="high", config=config),
        low=BOAState(market_type="low", config=config),
    )

    # Track each model's last known run_time to detect new runs
    last_run_times: dict[str, str] = {}

    for tdate in target_dates:
        # Get settlement
        settlements = db.execute(
            """SELECT market_type, actual_value_f
               FROM event_settlements
               WHERE station = ? AND settlement_date = ?
                 AND actual_value_f IS NOT NULL""",
            (station, tdate),
        ).fetchall()

        if not settlements:
            continue

        settle_map: dict[str, float] = {}
        for row in settlements:
            settle_map[row["market_type"]] = row["actual_value_f"]

        # Get model forecasts with run_time for sleep/wake detection
        forecast_rows = db.execute(
            """SELECT model, source, forecast_high_f, forecast_low_f,
                      run_time, fetch_time_utc
               FROM model_forecasts
               WHERE station = ? AND forecast_date = ?
                 AND (forecast_high_f IS NOT NULL OR forecast_low_f IS NOT NULL)
               ORDER BY id DESC""",
            (station, tdate),
        ).fetchall()

        if not forecast_rows:
            continue

        # Deduplicate: latest per (model, source)
        seen: dict[tuple[str, str], dict] = {}
        for row in forecast_rows:
            key = (row["model"], row["source"] or "unknown")
            if key not in seen:
                seen[key] = row

        for mt in ("high", "low"):
            obs = settle_map.get(mt)
            if obs is None:
                continue

            source_fcsts: dict[str, tuple[float, float]] = {}
            awake: set[str] = set()
            all_vals: list[float] = []

            for (model, source), row in seen.items():
                val = row["forecast_high_f"] if mt == "high" else row["forecast_low_f"]
                if val is not None:
                    all_vals.append(float(val))
                    source_key = f"{source}:{model}"
                    source_fcsts[source_key] = (float(val), 1.5)

                    # Determine if this source is "awake" (has a new run)
                    run_time_str = row["run_time"] or row["fetch_time_utc"] or ""
                    state_key = f"{mt}:{source_key}"
                    prev_run = last_run_times.get(state_key)
                    if prev_run is None or run_time_str != prev_run:
                        awake.add(source_key)
                    last_run_times[state_key] = run_time_str

            if not source_fcsts:
                continue

            # Update sigma from ensemble spread
            if len(all_vals) >= 2:
                spread = float(np.std(all_vals))
                source_fcsts = {
                    k: (mu, max(0.5, spread)) for k, (mu, _) in source_fcsts.items()
                }

            manager.update(mt, source_fcsts, obs, awake_sources=awake)

    return manager


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """Initialize BOA from historical data and save state."""
    import argparse
    import sqlite3

    parser = argparse.ArgumentParser(description="Initialize BOA source weights (sleeping experts)")
    parser.add_argument("--db", required=True, help="Path to miami_collector.db")
    parser.add_argument("--station", default="KMIA", help="Station identifier")
    parser.add_argument("--lookback-days", type=int, default=40)
    parser.add_argument("--reference-date", default=None)
    parser.add_argument("--learning-rate", type=float, default=0.5)
    parser.add_argument("--forget-rate", type=float, default=0.02)
    parser.add_argument("--out", default="analysis_data/boa_state.json")
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row

    config = BOAConfig(
        learning_rate=args.learning_rate,
        forget_rate=args.forget_rate,
    )

    print(f"Initializing BOA (sleeping experts) for {args.station} from {args.lookback_days}-day history...")
    manager = initialize_boa_from_history(
        conn, args.station,
        lookback_days=args.lookback_days,
        reference_date=args.reference_date,
        config=config,
    )
    conn.close()

    manager.save(args.out)

    summary = manager.summary()
    for mt in ("high", "low"):
        s = summary[mt]
        print(f"  {mt}: {s['n_sources']} sources, {s['total_updates']} updates")
        for src, w in s["top_sources"].items():
            print(f"    {src}: {w:.4f}")

    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
