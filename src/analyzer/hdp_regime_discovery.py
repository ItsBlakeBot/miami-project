"""HDP-Sticky HMM for shadow-mode regime discovery.

Runs nightly as a batch job on the previous day's surface observations.
Discovers regime structure automatically (number of regimes is inferred,
not fixed). Writes results to regime_labels_hdp_test table for comparison
against AI-labeled regimes.

This module NEVER touches the live pipeline. It exists purely to evaluate
whether HDP-Sticky can eventually replace AI-based regime labeling.

Uses pure Python + numpy — no pyhsmm dependency.
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class HDPRegimeResult:
    """Output from one day's HDP-Sticky analysis."""
    station: str
    target_date: str
    n_regimes_discovered: int
    regime_sequence: list[int]           # regime index per obs
    regime_timestamps: list[str]         # UTC timestamp per obs
    regime_params: list[dict]            # per-regime {mean, cov} of emission
    transition_matrix: list[list[float]] # K x K estimated transitions
    phase_summary: list[dict]            # [{start_lst, end_lst, regime_id, description}]


class StickyHDPHMM:
    """Sticky HDP-HMM with Gibbs sampling inference.

    Implements the sticky HDP-HMM from Fox et al. (2011) using the
    direct assignment sampler. The "sticky" parameter (kappa) encourages
    self-transitions, preventing rapid oscillation between states.

    Parameters
    ----------
    alpha : float
        Concentration parameter for the top-level DP. Controls how
        readily new regimes are created. Higher = more regimes.
    gamma : float
        Concentration parameter for the base DP. Controls the overall
        regime diversity.
    kappa : float
        Sticky parameter. Added to the self-transition count.
        Higher = regimes persist longer. 50-200 is typical for 5-min obs.
    max_regimes : int
        Upper bound on regimes for computational tractability.
    n_iter : int
        Number of Gibbs sampling iterations.
    """

    def __init__(
        self,
        alpha: float = 5.0,
        gamma: float = 2.0,
        kappa: float = 100.0,
        max_regimes: int = 15,
        n_iter: int = 200,
        dim: int = 4,
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.kappa = kappa
        self.max_regimes = max_regimes
        self.n_iter = n_iter
        self.dim = dim

        # NIW prior for emission distributions
        self._mu0 = np.zeros(dim)
        self._kappa0 = 0.5
        self._nu0 = float(dim + 2)
        self._psi0 = np.eye(dim) * 2.0

    def fit(self, obs: np.ndarray) -> tuple[np.ndarray, list[dict], np.ndarray]:
        """Run Gibbs sampling on observations.

        Parameters
        ----------
        obs : np.ndarray of shape (T, dim)
            Observation matrix. Each row is [temp_f, dew_f, pressure_hpa, wind_speed_mph].

        Returns
        -------
        z : np.ndarray of shape (T,)
            Final regime assignment per timestep.
        params : list[dict]
            Per-regime emission parameters {mean: ndarray, cov: ndarray}.
        trans : np.ndarray of shape (K, K)
            Estimated transition matrix.
        """
        T, D = obs.shape
        assert D == self.dim

        # Initialize: assign all obs to regime 0
        K = 1
        z = np.zeros(T, dtype=int)

        # Run Gibbs iterations
        for iteration in range(self.n_iter):
            # Step 1: Resample regime assignments
            for t in range(T):
                # Remove current assignment
                old_k = z[t]

                # Count transitions in/out of each regime (excluding t)
                n_k = np.zeros(K + 1)  # +1 for potential new regime
                for k in range(K):
                    n_k[k] = np.sum(z == k)
                n_k[old_k] -= 1  # exclude current

                # Compute log-probability for each existing regime + new
                log_probs = np.full(K + 1, -np.inf)

                for k in range(K):
                    if n_k[k] == 0 and k != old_k:
                        continue
                    # Prior: CRP with sticky
                    if t > 0 and z[t - 1] == k:
                        prior = n_k[k] + self.kappa
                    else:
                        prior = n_k[k]
                    if prior <= 0:
                        continue
                    log_probs[k] = np.log(prior) + self._emission_log_likelihood(
                        obs[t], obs[z == k] if np.sum(z == k) > 0 else None, k
                    )

                # New regime probability
                log_probs[K] = np.log(self.alpha) + self._emission_log_likelihood(
                    obs[t], None, -1
                )

                # Normalize and sample
                log_probs -= _log_sum_exp_arr(log_probs[log_probs > -np.inf])
                probs = np.exp(log_probs)
                probs = np.maximum(probs, 0)
                probs /= probs.sum() + 1e-30

                new_k = np.random.choice(K + 1, p=probs)
                z[t] = new_k

                # If new regime was chosen, expand K
                if new_k == K and K < self.max_regimes:
                    K += 1

            # Step 2: Remove empty regimes and relabel
            unique = np.unique(z)
            if len(unique) < K:
                mapping = {old: new for new, old in enumerate(sorted(unique))}
                z = np.array([mapping[zi] for zi in z])
                K = len(unique)

            if (iteration + 1) % 50 == 0:
                log.debug("HDP-Sticky iter %d: K=%d regimes", iteration + 1, K)

        # Compute final emission parameters
        params = []
        for k in range(K):
            mask = z == k
            if np.sum(mask) < 2:
                params.append({"mean": self._mu0.tolist(), "cov": self._psi0.tolist()})
                continue
            obs_k = obs[mask]
            mean = obs_k.mean(axis=0)
            cov = np.cov(obs_k, rowvar=False)
            if cov.ndim == 0:
                cov = np.eye(self.dim) * float(cov)
            params.append({"mean": mean.tolist(), "cov": cov.tolist()})

        # Compute transition matrix
        trans = np.zeros((K, K))
        for t in range(1, T):
            trans[z[t - 1], z[t]] += 1
        # Normalize rows
        row_sums = trans.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        trans /= row_sums

        return z, params, trans

    def _emission_log_likelihood(
        self, x: np.ndarray, cluster_obs: np.ndarray | None, k: int
    ) -> float:
        """Predictive log-likelihood under NIW posterior for regime k."""
        d = self.dim

        if cluster_obs is None or len(cluster_obs) == 0:
            # Prior predictive (no data in cluster)
            mu_n = self._mu0
            kappa_n = self._kappa0
            nu_n = self._nu0
            psi_n = self._psi0
        else:
            n = len(cluster_obs)
            x_bar = cluster_obs.mean(axis=0)
            kappa_n = self._kappa0 + n
            nu_n = self._nu0 + n
            mu_n = (self._kappa0 * self._mu0 + n * x_bar) / kappa_n
            S = np.zeros((d, d))
            for xi in cluster_obs:
                diff = xi - x_bar
                S += np.outer(diff, diff)
            diff0 = x_bar - self._mu0
            psi_n = self._psi0 + S + (self._kappa0 * n / kappa_n) * np.outer(
                diff0, diff0
            )

        # Student-t predictive
        df = nu_n - d + 1
        if df <= 0:
            return -50.0
        scale = psi_n * (kappa_n + 1) / (kappa_n * df)

        return _multivariate_t_logpdf(x, mu_n, scale, df, d)


def _multivariate_t_logpdf(x, mu, sigma, df, d):
    """Log-pdf of multivariate Student-t."""
    try:
        L = np.linalg.cholesky(sigma)
    except np.linalg.LinAlgError:
        L = np.linalg.cholesky(sigma + np.eye(d) * 1e-6)

    diff = x - mu
    solve = np.linalg.solve(L, diff)
    maha = float(np.dot(solve, solve))
    log_det = 2.0 * np.sum(np.log(np.diag(L)))

    log_num = math.lgamma((df + d) / 2.0)
    log_den = (
        math.lgamma(df / 2.0)
        + (d / 2.0) * math.log(df * math.pi)
        + 0.5 * log_det
        + ((df + d) / 2.0) * math.log(1.0 + maha / df)
    )
    return log_num - log_den


def _log_sum_exp_arr(arr):
    """Numerically stable log-sum-exp for an array."""
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return 0.0
    m = np.max(arr)
    return m + np.log(np.sum(np.exp(arr - m)))


class HDPShadowRunner:
    """Nightly batch runner for HDP-Sticky regime discovery.

    Queries the previous day's surface obs from the collector DB,
    runs the Sticky HDP-HMM, and stores results in regime_labels_hdp_test.

    Parameters
    ----------
    db_path : str or Path
        Path to the collector SQLite database.
    station : str
        Station identifier (default: KMIA).
    utc_offset : int
        UTC offset for LST conversion (default: -5 for EST).
    """

    def __init__(
        self, db_path: str | Path, station: str = "KMIA", utc_offset: int = -5
    ):
        self.db_path = Path(db_path)
        self.station = station
        self.utc_offset = utc_offset

    def run(self, target_date: str) -> HDPRegimeResult | None:
        """Run HDP-Sticky on a single day's data.

        Parameters
        ----------
        target_date : str
            Date in YYYY-MM-DD format. Must be a completed climate day.

        Returns
        -------
        HDPRegimeResult or None if insufficient data.
        """
        # Climate day bounds: midnight-midnight LST = 05:00Z to 05:00Z+1
        utc_start = f"{target_date}T{-self.utc_offset:02d}:00:00Z"

        # Compute next day for end bound
        dt = datetime.strptime(target_date, "%Y-%m-%d")
        next_day = (dt + timedelta(days=1)).strftime("%Y-%m-%d")
        utc_end = f"{next_day}T{-self.utc_offset:02d}:00:00Z"

        conn = sqlite3.connect(str(self.db_path), timeout=10)
        try:
            # Pull surface obs for the climate day
            rows = conn.execute(
                """SELECT timestamp_utc, temperature_f, dew_point_f,
                          pressure_hpa, wind_speed_mph
                   FROM observations
                   WHERE station = ?
                     AND timestamp_utc >= ? AND timestamp_utc < ?
                     AND temperature_f IS NOT NULL
                     AND dew_point_f IS NOT NULL
                     AND pressure_hpa IS NOT NULL
                   ORDER BY timestamp_utc""",
                (self.station, utc_start, utc_end),
            ).fetchall()

            if len(rows) < 20:
                log.warning(
                    "Only %d obs for %s — skipping HDP", len(rows), target_date
                )
                return None

            timestamps = [r[0] for r in rows]
            obs = np.array([[r[1], r[2], r[3], r[4] or 0.0] for r in rows])

            # Standardize for better numerical behavior
            obs_mean = obs.mean(axis=0)
            obs_std = obs.std(axis=0)
            obs_std[obs_std < 1e-6] = 1.0
            obs_normed = (obs - obs_mean) / obs_std

            # Run HDP-Sticky
            model = StickyHDPHMM(
                alpha=5.0,
                gamma=2.0,
                kappa=100.0,  # strong stickiness for 5-min obs
                max_regimes=10,
                n_iter=200,
                dim=4,
            )

            z, params, trans = model.fit(obs_normed)

            # Convert params back to original scale
            orig_params = []
            for p in params:
                mean = np.array(p["mean"]) * obs_std + obs_mean
                cov = np.array(p["cov"]) * np.outer(obs_std, obs_std)
                orig_params.append(
                    {
                        "mean": mean.tolist(),
                        "cov": cov.tolist(),
                    }
                )

            # Build phase summary from regime sequence
            phases = self._build_phases(z, timestamps)

            K = len(orig_params)
            log.info(
                "HDP discovered %d regimes for %s from %d obs",
                K,
                target_date,
                len(rows),
            )

            return HDPRegimeResult(
                station=self.station,
                target_date=target_date,
                n_regimes_discovered=K,
                regime_sequence=z.tolist(),
                regime_timestamps=timestamps,
                regime_params=orig_params,
                transition_matrix=trans.tolist(),
                phase_summary=phases,
            )
        finally:
            conn.close()

    def store(self, result: HDPRegimeResult) -> None:
        """Store HDP results in the regime_labels_hdp_test table."""
        conn = sqlite3.connect(str(self.db_path), timeout=10)
        try:
            conn.execute(
                """INSERT OR REPLACE INTO regime_labels_hdp_test
                   (station, target_date, n_regimes, regime_sequence,
                    regime_params, transition_matrix, phase_summary)
                   VALUES (?,?,?,?,?,?,?)""",
                (
                    result.station,
                    result.target_date,
                    result.n_regimes_discovered,
                    json.dumps(result.regime_sequence),
                    json.dumps(result.regime_params),
                    json.dumps(result.transition_matrix),
                    json.dumps(result.phase_summary),
                ),
            )
            conn.commit()
            log.info(
                "Stored HDP results for %s: %d regimes",
                result.target_date,
                result.n_regimes_discovered,
            )
        finally:
            conn.close()

    def _build_phases(self, z: np.ndarray, timestamps: list[str]) -> list[dict]:
        """Convert regime sequence into phase summary."""
        if len(z) == 0:
            return []

        phases = []
        current_regime = z[0]
        phase_start = 0

        for t in range(1, len(z)):
            if z[t] != current_regime:
                phases.append(
                    {
                        "start_time_utc": timestamps[phase_start],
                        "end_time_utc": timestamps[t - 1],
                        "start_hour_lst": self._utc_to_lst_hour(timestamps[phase_start]),
                        "end_hour_lst": self._utc_to_lst_hour(timestamps[t - 1]),
                        "regime_id": int(current_regime),
                        "n_obs": t - phase_start,
                    }
                )
                current_regime = z[t]
                phase_start = t

        # Final phase
        phases.append(
            {
                "start_time_utc": timestamps[phase_start],
                "end_time_utc": timestamps[-1],
                "start_hour_lst": self._utc_to_lst_hour(timestamps[phase_start]),
                "end_hour_lst": self._utc_to_lst_hour(timestamps[-1]),
                "regime_id": int(current_regime),
                "n_obs": len(z) - phase_start,
            }
        )

        return phases

    def _utc_to_lst_hour(self, ts: str) -> int:
        """Extract LST hour from a UTC timestamp string."""
        try:
            if "T" in ts:
                hour_utc = int(ts.split("T")[1][:2])
            else:
                hour_utc = int(ts.split(" ")[1][:2])
            return (hour_utc + self.utc_offset) % 24
        except (IndexError, ValueError):
            return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entrypoint for nightly HDP shadow run."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run HDP-Sticky regime discovery on previous day's data"
    )
    parser.add_argument("--db", required=True, help="Path to collector DB")
    parser.add_argument(
        "--date",
        help="Target date (YYYY-MM-DD). Defaults to yesterday.",
    )
    parser.add_argument("--station", default="KMIA")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
    )

    if args.date:
        target_date = args.date
    else:
        target_date = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")

    runner = HDPShadowRunner(args.db, station=args.station)
    result = runner.run(target_date)

    if result:
        runner.store(result)
        print(f"HDP discovered {result.n_regimes_discovered} regimes for {target_date}")
        for phase in result.phase_summary:
            print(
                f"  {phase['start_hour_lst']:02d}:00-{phase['end_hour_lst']:02d}:00 LST: "
                f"regime {phase['regime_id']} ({phase['n_obs']} obs)"
            )
    else:
        print(f"Insufficient data for {target_date}")


if __name__ == "__main__":
    main()
