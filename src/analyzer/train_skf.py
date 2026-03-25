"""
Training pipeline for the Switching Kalman Filter.

Reads regime_labels and historical observations from the database,
fits per-regime state-space parameters (A, b, Q, R), and learns
mu_shift / sigma_scale corrections from settlement errors.

Outputs a JSON config consumed by SwitchingKalmanFilter at runtime.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class SKFTrainer:
    """Fit SKF parameters from regime-labeled historical data."""

    N_STATES = 5

    def __init__(self, db_path: str, station: str = "KMIA", utc_offset: int = -5):
        self.db_path = db_path
        self.station = station
        self.utc_offset = utc_offset

    # ------------------------------------------------------------------
    # Main training entry point
    # ------------------------------------------------------------------

    def train(self, min_days_per_regime: int = 2) -> dict:
        """
        1. Query regime_labels for all labeled days.
        2. Extract unique regimes from phase_summary entries.
        3. Group days by dominant regime (most hours).
        4. For each regime with enough labeled days, fit parameters.
        5. Regimes with < min_days get mixed_uncertain defaults.
        6. Return config dict (also suitable for JSON serialization).
        """
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row

        # Ensure hourly_obs view exists (aggregates 1-min observations)
        from collector.store.schema import VIEWS
        for view_ddl in VIEWS:
            con.execute(view_ddl)

        # ---- 1. Load regime labels ----
        rows = con.execute(
            "SELECT target_date, phase_summary, signal_families_active "
            "FROM regime_labels WHERE station = ? ORDER BY target_date",
            (self.station,),
        ).fetchall()

        if not rows:
            logger.warning("No regime_labels found for %s — using all defaults", self.station)
            con.close()
            return {"regimes": self._all_defaults()}

        # ---- 2-3. Parse phase_summary, find dominant regime per day ----
        day_regime: dict[str, str] = {}            # date -> dominant regime
        regime_days: dict[str, list[str]] = defaultdict(list)  # regime -> [dates]
        day_families: dict[str, list[str]] = {}    # date -> active families

        for row in rows:
            target_date = row["target_date"]
            phase_summary = row["phase_summary"]
            families_raw = row["signal_families_active"]

            # Parse families — may be list of strings or list of dicts with family_name
            fams: list[str] = []
            if families_raw:
                try:
                    parsed_fams = json.loads(families_raw) if families_raw.startswith("[") else [
                        f.strip() for f in families_raw.split(",") if f.strip()
                    ]
                    for f in parsed_fams:
                        if isinstance(f, str):
                            fams.append(f)
                        elif isinstance(f, dict) and "family_name" in f:
                            fams.append(f["family_name"])
                except (json.JSONDecodeError, TypeError):
                    pass
            day_families[target_date] = fams

            # Parse phase_summary to get regime durations
            # Expected format: JSON list of {regime, start_hour, end_hour, ...}
            regimes_hours: Counter[str] = Counter()
            if phase_summary:
                try:
                    phases = json.loads(phase_summary) if isinstance(phase_summary, str) else phase_summary
                    if isinstance(phases, list):
                        for phase in phases:
                            regime_name = phase.get("regime", phase.get("name", "unknown"))
                            start_h = phase.get("start_hour", 0)
                            end_h = phase.get("end_hour", start_h + 1)
                            regimes_hours[regime_name] += max(end_h - start_h, 1)
                    elif isinstance(phases, str):
                        # Simple string label
                        regimes_hours[phases] += 24
                except (json.JSONDecodeError, TypeError):
                    regimes_hours[str(phase_summary)] += 24

            if regimes_hours:
                dominant = regimes_hours.most_common(1)[0][0]
            else:
                dominant = "mixed_uncertain"

            day_regime[target_date] = dominant
            regime_days[dominant].append(target_date)

        logger.info(
            "Regime distribution: %s",
            {k: len(v) for k, v in regime_days.items()},
        )

        # ---- 4. Fit parameters per regime ----
        fitted_regimes: list[dict] = []

        for regime_name, dates in regime_days.items():
            if len(dates) < min_days_per_regime:
                logger.info(
                    "Regime '%s' has %d days (< %d) — using defaults",
                    regime_name, len(dates), min_days_per_regime,
                )
                fitted_regimes.append(self._default_regime(regime_name))
                continue

            logger.info("Fitting regime '%s' from %d days", regime_name, len(dates))

            # a. Query hourly obs
            obs_seqs = self._query_obs_sequences(con, dates)

            # b. Build state vector sequences
            state_sequences: list[list[list[float]]] = []
            for date in sorted(obs_seqs.keys()):
                obs_list = obs_seqs[date]
                if len(obs_list) < 3:
                    continue
                seq = [self._obs_to_state(o) for o in obs_list]
                state_sequences.append(seq)

            if not state_sequences or sum(len(s) for s in state_sequences) < 4:
                logger.warning("Insufficient obs for regime '%s' — using defaults", regime_name)
                fitted_regimes.append(self._default_regime(regime_name))
                continue

            # c-d. Fit A, b, Q
            A, b_vec, Q = self._fit_transition_matrices(state_sequences)

            # e. Measurement noise R from within-hour variance
            R = self._estimate_measurement_noise(con, dates)

            # f. Self-transition probability from phase durations
            self_trans = self._estimate_self_transition(con, dates, regime_name)

            # g. mu_shift and sigma_scale from settlement errors
            mu_high, mu_low, sig_high, sig_low = self._estimate_market_adjustments(
                con, dates
            )

            # h. Active families
            all_fams: Counter[str] = Counter()
            for d in dates:
                for f in day_families.get(d, []):
                    all_fams[f] += 1
            # Keep families that appear in > 30% of regime days
            threshold = max(1, int(0.3 * len(dates)))
            active_fams = [f for f, c in all_fams.most_common() if c >= threshold]

            fitted_regimes.append(dict(
                name=regime_name,
                A=A.tolist(),
                b=b_vec.tolist(),
                Q=Q.tolist(),
                R=R.tolist(),
                self_transition_prob=float(self_trans),
                mu_shift_high=float(mu_high),
                mu_shift_low=float(mu_low),
                sigma_scale_high=float(sig_high),
                sigma_scale_low=float(sig_low),
                active_families=active_fams,
            ))

        con.close()

        # Ensure mixed_uncertain exists as a fallback
        names = {r["name"] for r in fitted_regimes}
        if "mixed_uncertain" not in names:
            fitted_regimes.insert(0, self._default_regime("mixed_uncertain"))

        config = {"regimes": fitted_regimes}
        return config

    # ------------------------------------------------------------------
    # Observation queries
    # ------------------------------------------------------------------

    def _query_obs_sequences(
        self, con: sqlite3.Connection, dates: list[str]
    ) -> dict[str, list[dict]]:
        """Get hourly obs for given dates.  Returns {date: [obs_dicts]}."""
        result: dict[str, list[dict]] = defaultdict(list)

        placeholders = ",".join("?" for _ in dates)
        rows = con.execute(
            f"""
            SELECT target_date, hour_utc, temp_f, dew_f, pressure_hpa, wind_dir
            FROM hourly_obs
            WHERE station = ? AND target_date IN ({placeholders})
            ORDER BY target_date, hour_utc
            """,
            [self.station] + dates,
        ).fetchall()

        for row in rows:
            d = dict(row)
            # Skip rows with missing critical fields
            if any(d.get(k) is None for k in ("temp_f", "dew_f", "pressure_hpa", "wind_dir")):
                continue
            result[d["target_date"]].append(d)

        return dict(result)

    def _obs_to_state(self, obs: dict) -> list[float]:
        """Convert an obs dict to a 5D state vector."""
        wind_rad = math.radians(float(obs["wind_dir"]))
        return [
            float(obs["temp_f"]),
            float(obs["dew_f"]),
            float(obs["pressure_hpa"]),
            math.sin(wind_rad),
            math.cos(wind_rad),
        ]

    # ------------------------------------------------------------------
    # Parameter fitting
    # ------------------------------------------------------------------

    def _fit_transition_matrices(
        self, sequences: list[list[list[float]]]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fit A, b, Q from state transition sequences via OLS.

        Collects all (x_t, x_{t+1}) pairs, stacks into X (Nx5) and Y (Nx5),
        and solves Y = X @ A.T + ones @ b.T via least squares.
        """
        xs: list[list[float]] = []
        ys: list[list[float]] = []
        for seq in sequences:
            for i in range(len(seq) - 1):
                xs.append(seq[i])
                ys.append(seq[i + 1])

        X = np.array(xs)  # N x 5
        Y = np.array(ys)  # N x 5
        N = X.shape[0]

        # Augment X with column of ones for bias: [X, 1] @ [A.T; b.T] = Y
        X_aug = np.column_stack([X, np.ones(N)])  # N x 6

        # Solve for each output dimension
        # result shape: 6 x 5
        params, _, _, _ = np.linalg.lstsq(X_aug, Y, rcond=None)

        A = params[:5, :].T   # 5 x 5
        b_vec = params[5, :]   # 5

        # Residuals and process noise covariance
        Y_pred = X_aug @ params  # N x 5
        residuals = Y - Y_pred   # N x 5
        Q = np.cov(residuals, rowvar=False)

        # Ensure Q is positive semi-definite (add small diagonal if needed)
        min_eig = np.min(np.linalg.eigvalsh(Q))
        if min_eig < 1e-8:
            Q += np.eye(self.N_STATES) * (1e-8 - min_eig)

        return A, b_vec, Q

    def _estimate_measurement_noise(
        self, con: sqlite3.Connection, dates: list[str]
    ) -> np.ndarray:
        """
        Estimate R from within-hour observation variance.

        Uses the variance of obs within the same hour across different
        days as a proxy for measurement noise.
        """
        placeholders = ",".join("?" for _ in dates)
        rows = con.execute(
            f"""
            SELECT hour_utc, temp_f, dew_f, pressure_hpa, wind_dir
            FROM hourly_obs
            WHERE station = ? AND target_date IN ({placeholders})
            ORDER BY hour_utc
            """,
            [self.station] + dates,
        ).fetchall()

        if len(rows) < 5:
            return np.diag([0.5, 0.5, 0.2, 0.05, 0.05])

        # Group by hour, compute variance per state dimension
        hour_states: dict[int, list[list[float]]] = defaultdict(list)
        for row in rows:
            d = dict(row)
            if any(d.get(k) is None for k in ("temp_f", "dew_f", "pressure_hpa", "wind_dir")):
                continue
            hour_states[d["hour_utc"]].append(self._obs_to_state(d))

        variances = []
        for hour, states in hour_states.items():
            if len(states) >= 2:
                arr = np.array(states)
                variances.append(np.var(arr, axis=0))

        if variances:
            mean_var = np.mean(variances, axis=0)
            # Floor at small positive values
            mean_var = np.maximum(mean_var, [0.1, 0.1, 0.05, 0.01, 0.01])
            return np.diag(mean_var)
        else:
            return np.diag([0.5, 0.5, 0.2, 0.05, 0.05])

    def _estimate_self_transition(
        self, con: sqlite3.Connection, dates: list[str], regime_name: str
    ) -> float:
        """Estimate self-transition probability from phase durations."""
        total_hours = 0
        regime_hours = 0

        for date in dates:
            row = con.execute(
                "SELECT phase_summary FROM regime_labels "
                "WHERE station = ? AND target_date = ?",
                (self.station, date),
            ).fetchone()
            if not row or not row["phase_summary"]:
                continue

            try:
                phases = json.loads(row["phase_summary"])
                if isinstance(phases, list):
                    for phase in phases:
                        rn = phase.get("regime", phase.get("name", ""))
                        start_h = phase.get("start_hour", 0)
                        end_h = phase.get("end_hour", start_h + 1)
                        duration = max(end_h - start_h, 1)
                        total_hours += duration
                        if rn == regime_name:
                            regime_hours += duration
            except (json.JSONDecodeError, TypeError):
                pass

        if total_hours > 0 and regime_hours > 0:
            # Approximate: fraction of time in this regime suggests persistence
            frac = regime_hours / total_hours
            # Map to self-transition: higher fraction -> stickier
            return min(0.98, max(0.5, 0.6 + 0.35 * frac))
        else:
            return 0.85  # default

    def _estimate_market_adjustments(
        self, con: sqlite3.Connection, dates: list[str]
    ) -> tuple[float, float, float, float]:
        """
        Learn mu_shift and sigma_scale from settlement errors.

        mu_shift_high = mean(cli_high - model_consensus_high) for these dates
        mu_shift_low  = mean(cli_low - model_consensus_low) for these dates
        sigma_scale   = std(errors) / baseline_std
        """
        placeholders = ",".join("?" for _ in dates)

        # Try to get settlement data
        try:
            rows = con.execute(
                f"""
                SELECT target_date,
                       cli_high, cli_low,
                       model_consensus_high, model_consensus_low
                FROM daily_summary
                WHERE station = ? AND target_date IN ({placeholders})
                  AND cli_high IS NOT NULL
                  AND model_consensus_high IS NOT NULL
                """,
                [self.station] + dates,
            ).fetchall()
        except sqlite3.OperationalError:
            # Table or columns may not exist yet
            return 0.0, 0.0, 1.0, 1.0

        if len(rows) < 2:
            return 0.0, 0.0, 1.0, 1.0

        errors_high = []
        errors_low = []
        for row in rows:
            d = dict(row)
            eh = d["cli_high"] - d["model_consensus_high"]
            el = d["cli_low"] - d["model_consensus_low"]
            errors_high.append(eh)
            errors_low.append(el)

        mu_high = float(np.mean(errors_high))
        mu_low = float(np.mean(errors_low))

        # sigma_scale: std of errors relative to a baseline of 1.5F
        baseline_std = 1.5
        std_high = float(np.std(errors_high)) if len(errors_high) > 1 else baseline_std
        std_low = float(np.std(errors_low)) if len(errors_low) > 1 else baseline_std

        sig_high = max(0.5, min(2.0, std_high / baseline_std))
        sig_low = max(0.5, min(2.0, std_low / baseline_std))

        return mu_high, mu_low, sig_high, sig_low

    # ------------------------------------------------------------------
    # Defaults
    # ------------------------------------------------------------------

    def _default_regime(self, name: str) -> dict:
        """Return a conservative default regime model with wider noise."""
        return dict(
            name=name,
            A=np.eye(self.N_STATES, dtype=float).tolist(),
            b=[0.0] * self.N_STATES,
            Q=np.diag([3.0, 3.0, 0.8, 0.15, 0.15]).tolist(),
            R=np.diag([0.5, 0.5, 0.2, 0.05, 0.05]).tolist(),
            self_transition_prob=0.85,
            mu_shift_high=0.0,
            mu_shift_low=0.0,
            sigma_scale_high=1.0,
            sigma_scale_low=1.0,
            active_families=[],
        )

    def _all_defaults(self) -> list[dict]:
        """Return the full set of default regimes when no data exists."""
        from src.engine.kalman_regimes import _default_regime_models
        return _default_regime_models()

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save_config(self, config: dict, output_path: str) -> None:
        """Write config to JSON, converting any numpy arrays to lists."""
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        def _convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            if isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            raise TypeError(f"Cannot serialise {type(obj)}")

        with open(output, "w") as f:
            json.dump(config, f, indent=2, default=_convert)

        logger.info("SKF config saved to %s (%d regimes)", output, len(config.get("regimes", [])))


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

def main() -> None:
    """CLI entrypoint: miami-train-skf"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Train SKF from regime-labeled data")
    parser.add_argument("--db", required=True, help="Path to SQLite database")
    parser.add_argument("--output", default="analysis_data/skf_trained_candidate.json",
                        help="Output path for SKF config JSON (defaults to candidate, not live config)")
    parser.add_argument("--min-days", type=int, default=2,
                        help="Minimum labeled days per regime to fit (default: 2)")
    args = parser.parse_args()

    trainer = SKFTrainer(args.db)
    config = trainer.train(min_days_per_regime=args.min_days)
    trainer.save_config(config, args.output)

    # Print summary
    for regime in config.get("regimes", []):
        name = regime["name"]
        st = regime["self_transition_prob"]
        mh = regime["mu_shift_high"]
        ml = regime["mu_shift_low"]
        print(f"  {name}: self_trans={st:.3f}  mu_shift=({mh:+.2f}, {ml:+.2f})")


if __name__ == "__main__":
    main()
