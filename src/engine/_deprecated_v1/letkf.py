"""Regional Cluster LETKF (Local Ensemble Transform Kalman Filter).

Lightweight surface-only LETKF for spatial state assimilation across trading
station clusters. Propagates information from stations that observe weather
features first to stations that haven't seen them yet.

Architecture:
  - Background ensemble: GEFS 31-member or per-model forecasts as pseudo-ensemble
  - State vector: surface temperature at each station in the cluster
  - Observations: ASOS, FAWN, NDBC buoy temperatures within cluster
  - Localization: Gaspari-Cohn function with configurable radius (50-200km)
  - Inflation: Relaxation-to-Prior Spread (RTPS) with alpha ~0.7-0.9
  - Fallback: for sparse clusters (<3 obs within radius), skip LETKF update

Key reference: Hunt et al. (2007), Physica D — original LETKF algorithm

Design decisions:
  - Surface-only (no vertical levels) — trades atmospheric completeness for
    computational simplicity and robustness
  - Per-cluster instances — SE Florida, NE Corridor, Texas, etc. are independent
  - Obs-triggered updates via LiveState event system
  - Self-tuning: innovation-based adaptive R estimation (Sage-Husa method)
  - Output: analyzed temperature at each trading station in the cluster,
    plus analysis uncertainty (ensemble spread)

T3.1 — Regional cluster LETKF design
T3.2 — Observation operator + error model
T3.3 — LETKF core implementation
T3.4 — Self-tuning IMM-KF fallback for sparse regions
"""

from __future__ import annotations

import json
import logging
import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cluster definitions
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StationDef:
    """A station within a cluster."""

    code: str  # ICAO code (e.g., "KMIA")
    name: str
    lat: float
    lon: float
    is_trading: bool = False  # True if this is a Kalshi trading station
    obs_sigma_f: float = 0.5  # observation error std (°F)


@dataclass(frozen=True)
class ClusterDef:
    """A geographic cluster of stations for LETKF."""

    name: str
    stations: tuple[StationDef, ...]
    localization_radius_km: float = 100.0
    ensemble_size: int = 31
    rtps_alpha: float = 0.8  # RTPS inflation parameter

    @property
    def n_stations(self) -> int:
        return len(self.stations)

    @property
    def trading_stations(self) -> list[StationDef]:
        return [s for s in self.stations if s.is_trading]


# SE Florida cluster definition
SE_FLORIDA_CLUSTER = ClusterDef(
    name="se_florida",
    stations=(
        StationDef("KMIA", "Miami International", 25.7959, -80.2870, is_trading=True),
        StationDef("KFLL", "Fort Lauderdale", 26.0726, -80.1527, is_trading=True),
        StationDef("KOPF", "Miami/Opa Locka", 25.907, -80.278),
        StationDef("KTMB", "Kendall-Tamiami", 25.648, -80.433),
        StationDef("KHWO", "Hollywood/N. Perry", 26.001, -80.240),
        StationDef("KHST", "Homestead AFB", 25.489, -80.384),
        StationDef("KFXE", "Fort Lauderdale Exec", 26.197, -80.171),
        StationDef("KPMP", "Pompano Beach", 26.247, -80.111),
        StationDef("KBCT", "Boca Raton", 26.379, -80.108),
        StationDef("KPBI", "West Palm Beach", 26.683, -80.096),
        # FAWN stations (higher obs error due to siting variability)
        StationDef("FAWN440", "FAWN Homestead", 25.51, -80.50, obs_sigma_f=1.0),
        StationDef("FAWN420", "FAWN Ft. Lauderdale", 26.09, -80.24, obs_sigma_f=1.0),
        StationDef("FAWN425", "FAWN Wellington", 26.68, -80.30, obs_sigma_f=1.0),
        StationDef("FAWN410", "FAWN Belle Glade", 26.66, -80.63, obs_sigma_f=1.0),
    ),
    localization_radius_km=150.0,
    ensemble_size=31,
    rtps_alpha=0.8,
)


# ---------------------------------------------------------------------------
# Gaspari-Cohn localization function
# ---------------------------------------------------------------------------
def gaspari_cohn(distance_km: float, cutoff_km: float) -> float:
    """Gaspari-Cohn 5th-order correlation function.

    Returns a taper value in [0, 1]:
      - 1.0 at distance 0
      - Smoothly decreases to 0 at distance = cutoff_km
      - 0 beyond cutoff_km

    This is the standard localization function for EnKF/LETKF.
    """
    if cutoff_km <= 0:
        return 1.0 if distance_km == 0 else 0.0

    r = distance_km / (cutoff_km / 2.0)  # normalize to half-width

    if r >= 2.0:
        return 0.0
    elif r >= 1.0:
        return (
            (1.0 / 12.0) * r**5
            - 0.5 * r**4
            + (5.0 / 8.0) * r**3
            + (5.0 / 3.0) * r**2
            - 5.0 * r
            + 4.0
            - (2.0 / 3.0) / r
        )
    else:
        return (
            -(1.0 / 4.0) * r**5
            + 0.5 * r**4
            + (5.0 / 8.0) * r**3
            - (5.0 / 3.0) * r**2
            + 1.0
        )


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two (lat, lon) points."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ---------------------------------------------------------------------------
# Observation vector
# ---------------------------------------------------------------------------
@dataclass
class SurfaceObs:
    """A single surface observation for LETKF assimilation."""

    station_code: str
    lat: float
    lon: float
    temp_f: float
    obs_sigma_f: float = 0.5  # observation error std
    timestamp_utc: str = ""


# ---------------------------------------------------------------------------
# LETKF core
# ---------------------------------------------------------------------------
@dataclass
class LETKFState:
    """State of the LETKF for one cluster."""

    cluster: ClusterDef
    # Ensemble matrix: (n_stations, ensemble_size) — temperature in °F
    ensemble: np.ndarray | None = None
    # Analysis mean: (n_stations,)
    analysis_mean: np.ndarray | None = None
    # Analysis spread: (n_stations,)
    analysis_spread: np.ndarray | None = None
    # Adaptive R estimation (Sage-Husa)
    r_estimate: dict[str, float] = field(default_factory=dict)
    n_updates: int = 0
    # Tunable parameters (set from InferenceConfig)
    _sage_husa_b: float = 0.97
    _max_inflation: float = 2.0

    def initialize_from_forecasts(
        self,
        forecast_means: dict[str, float],
        forecast_spread: float = 2.0,
    ) -> None:
        """Initialize ensemble from forecast means + perturbations.

        Args:
            forecast_means: {station_code: forecast_temp_f}
            forecast_spread: Initial ensemble spread (°F)
        """
        n = self.cluster.n_stations
        k = self.cluster.ensemble_size
        rng = np.random.default_rng(42)

        self.ensemble = np.zeros((n, k))
        for i, station in enumerate(self.cluster.stations):
            mean = forecast_means.get(station.code, 75.0)  # default if missing
            self.ensemble[i, :] = mean + rng.normal(0, forecast_spread, size=k)

        self.analysis_mean = np.mean(self.ensemble, axis=1)
        self.analysis_spread = np.std(self.ensemble, axis=1)

    def update(self, observations: list[SurfaceObs]) -> dict[str, tuple[float, float]]:
        """Run one LETKF analysis cycle.

        Args:
            observations: List of surface temperature observations.

        Returns:
            {station_code: (analyzed_temp_f, analysis_spread_f)} for trading stations.
        """
        if self.ensemble is None:
            return {}

        if len(observations) < 2:
            # Too few obs — skip update, return current state
            return self._trading_station_output()

        n = self.cluster.n_stations
        k = self.cluster.ensemble_size
        stations = self.cluster.stations

        # Ensemble mean and perturbations
        x_mean = np.mean(self.ensemble, axis=1)  # (n,)
        x_pert = self.ensemble - x_mean[:, np.newaxis]  # (n, k)

        # Build observation vector and operator
        p = len(observations)
        y_obs = np.zeros(p)
        R_diag = np.zeros(p)
        obs_lats = np.zeros(p)
        obs_lons = np.zeros(p)

        for j, obs in enumerate(observations):
            y_obs[j] = obs.temp_f
            # Use adaptive R if available, otherwise obs_sigma
            r_est = self.r_estimate.get(obs.station_code, obs.obs_sigma_f)
            R_diag[j] = max(0.1, r_est) ** 2
            obs_lats[j] = obs.lat
            obs_lons[j] = obs.lon

        # LETKF: process each grid point (station) independently
        x_analysis = np.zeros_like(self.ensemble)

        for i in range(n):
            station = stations[i]

            # Find observations within localization radius
            local_obs_idx = []
            local_weights = []
            for j, obs in enumerate(observations):
                dist = haversine_km(station.lat, station.lon, obs.lat, obs.lon)
                gc = gaspari_cohn(dist, self.cluster.localization_radius_km)
                if gc > 0.001:  # only include if localization weight is nonzero
                    local_obs_idx.append(j)
                    local_weights.append(gc)

            if not local_obs_idx:
                # No local observations — keep prior
                x_analysis[i, :] = self.ensemble[i, :]
                continue

            # Local observation matrices
            p_local = len(local_obs_idx)
            y_local = y_obs[local_obs_idx]
            R_local = np.diag(R_diag[local_obs_idx] / np.array(local_weights))

            # Observation operator H: maps state to obs space
            # For surface temperature, H selects the nearest station's value
            # In our simplified model, each obs maps to one station
            # H_local: (p_local, k) — ensemble obs perturbations
            H_x_pert = np.zeros((p_local, k))
            H_x_mean = np.zeros(p_local)
            for jj, obs_idx in enumerate(local_obs_idx):
                obs = observations[obs_idx]
                # Find which state element this obs corresponds to
                # (nearest station in the cluster)
                nearest_idx = i  # default: observation influences this station
                min_dist = float("inf")
                for ii, st in enumerate(stations):
                    d = haversine_km(st.lat, st.lon, obs.lat, obs.lon)
                    if d < min_dist:
                        min_dist = d
                        nearest_idx = ii
                H_x_pert[jj, :] = x_pert[nearest_idx, :]
                H_x_mean[jj] = x_mean[nearest_idx]

            # Innovation
            d_local = y_local - H_x_mean

            # LETKF transform (Hunt et al. 2007)
            # Pa_tilde = [(k-1)*I + (H*Xb')^T * R^{-1} * (H*Xb')]^{-1}
            R_inv = np.diag(1.0 / np.diag(R_local))
            C = H_x_pert.T @ R_inv  # (k, p_local)
            Pa_tilde_inv = (k - 1) * np.eye(k) + C @ H_x_pert  # (k, k)

            try:
                Pa_tilde = np.linalg.inv(Pa_tilde_inv)
            except np.linalg.LinAlgError:
                # Singular matrix — skip this station
                x_analysis[i, :] = self.ensemble[i, :]
                continue

            # Analysis weights
            wa_mean = Pa_tilde @ C @ d_local  # (k,) — mean weight
            # Square root of Pa_tilde for perturbation weights
            eigvals, eigvecs = np.linalg.eigh(Pa_tilde)
            eigvals = np.maximum(eigvals, 1e-10)  # ensure positive
            Wa_pert = eigvecs @ np.diag(np.sqrt((k - 1) * eigvals)) @ eigvecs.T

            # Update ensemble at this station
            x_analysis[i, :] = (
                x_mean[i]
                + x_pert[i, :] @ (Wa_pert + wa_mean[:, np.newaxis])
            ).flatten()[:k]

        # RTPS inflation: relax posterior spread toward prior spread
        alpha = self.cluster.rtps_alpha
        prior_spread = np.std(self.ensemble, axis=1)
        posterior_spread = np.std(x_analysis, axis=1)

        for i in range(n):
            if posterior_spread[i] > 0 and prior_spread[i] > 0:
                inflation = 1.0 + alpha * (prior_spread[i] - posterior_spread[i]) / posterior_spread[i]
                inflation = max(1.0, min(inflation, self._max_inflation))  # cap inflation
                x_analysis[i, :] = np.mean(x_analysis[i, :]) + inflation * (
                    x_analysis[i, :] - np.mean(x_analysis[i, :])
                )

        # Store updated ensemble
        self.ensemble = x_analysis
        self.analysis_mean = np.mean(x_analysis, axis=1)
        self.analysis_spread = np.std(x_analysis, axis=1)
        self.n_updates += 1

        # Sage-Husa adaptive R estimation — use PRIOR mean (x_mean), not posterior
        # Using posterior would underestimate innovation variance → R shrinks → over-trusts obs
        for j, obs in enumerate(observations):
            nearest_idx = 0
            min_dist = float("inf")
            for ii, st in enumerate(stations):
                d = haversine_km(st.lat, st.lon, obs.lat, obs.lon)
                if d < min_dist:
                    min_dist = d
                    nearest_idx = ii
            innovation = obs.temp_f - x_mean[nearest_idx]  # prior mean, NOT analysis_mean
            # EMA of innovation squared → estimate of R
            b = self._sage_husa_b  # forgetting factor (tunable, default 0.97)
            prev_r = self.r_estimate.get(obs.station_code, obs.obs_sigma_f)
            new_r_sq = (1 - b) * innovation**2 + b * prev_r**2
            self.r_estimate[obs.station_code] = math.sqrt(max(0.1, new_r_sq))

        return self._trading_station_output()

    def get_diagnostics(self) -> dict:
        """Return diagnostic state for human review / autotuning."""
        spread_dict = {}
        if self.analysis_spread is not None:
            for i, s in enumerate(self.cluster.stations):
                spread_dict[s.code] = round(float(self.analysis_spread[i]), 3)
        return {
            "n_updates": self.n_updates,
            "r_estimates": dict(self.r_estimate),
            "analysis_spread": spread_dict,
            "cluster": self.cluster.name,
        }

    def autotune_from_settlement(
        self,
        station_code: str,
        observed_value_f: float,
    ) -> dict[str, float]:
        """Post-settlement autotuning: compare LETKF analysis to observed outcome.

        Adjusts RTPS inflation and Sage-Husa forgetting factor based on
        innovation statistics. Returns adjustment deltas for logging.
        """
        if self.analysis_mean is None or self.analysis_spread is None:
            return {}

        station_idx = None
        for i, s in enumerate(self.cluster.stations):
            if s.code == station_code:
                station_idx = i
                break
        if station_idx is None:
            return {}

        analysis_temp = float(self.analysis_mean[station_idx])
        analysis_spread = float(self.analysis_spread[station_idx])
        innovation = observed_value_f - analysis_temp
        adjustments: dict[str, float] = {"innovation_f": round(innovation, 2)}

        if abs(innovation) > 2.0 * max(analysis_spread, 0.1):
            # Analysis was too confident — increase RTPS inflation
            adjustments["rtps_alpha_delta"] = 0.02
            adjustments["recommendation"] = "increase_inflation"

        r_est = self.r_estimate.get(station_code, 0.5)
        if abs(innovation) > 2.0 * r_est:
            # R too small — Sage-Husa needs faster adaptation
            adjustments["sage_husa_b_delta"] = -0.01
            adjustments["recommendation"] = "decrease_forgetting"

        return adjustments

    def _trading_station_output(self) -> dict[str, tuple[float, float]]:
        """Extract analyzed temp + spread for trading stations."""
        if self.analysis_mean is None or self.analysis_spread is None:
            return {}
        result = {}
        for i, station in enumerate(self.cluster.stations):
            if station.is_trading:
                result[station.code] = (
                    round(float(self.analysis_mean[i]), 2),
                    round(float(self.analysis_spread[i]), 3),
                )
        return result


# ---------------------------------------------------------------------------
# LETKF autotuning — post-settlement innovation tracking + adaptive R/alpha
# ---------------------------------------------------------------------------

# Maximum days of innovation history to keep per station
_INNOVATION_WINDOW = 7

# Thresholds for autotuning decisions
_BIAS_THRESHOLD_F = 0.3  # mean innovation bias triggering R adjustment
_SPREAD_RATIO_THRESHOLD = 1.5  # innovation std / spread ratio triggering alpha bump
_R_DRIFT_THRESHOLD = 0.5  # flag stations where R drifted > this from initial


@dataclass
class LETKFTuneState:
    """Persistent autotuning state for the LETKF.

    Tracks per-station innovation statistics over a rolling window
    and adjusts R scaling / RTPS alpha accordingly.
    """

    # {station_code: deque of (obs - analysis) for highs}
    innovation_history_high: dict[str, deque] = field(default_factory=dict)
    # {station_code: deque of (obs - analysis) for lows}
    innovation_history_low: dict[str, deque] = field(default_factory=dict)
    # {station_code: initial R estimate at first update}
    initial_r_estimates: dict[str, float] = field(default_factory=dict)
    # Number of settlement days processed
    n_settlements: int = 0

    def to_dict(self) -> dict:
        """Serialize for JSON persistence."""
        return {
            "innovation_history_high": {
                k: list(v) for k, v in self.innovation_history_high.items()
            },
            "innovation_history_low": {
                k: list(v) for k, v in self.innovation_history_low.items()
            },
            "initial_r_estimates": self.initial_r_estimates,
            "n_settlements": self.n_settlements,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LETKFTuneState":
        state = cls()
        for k, v in d.get("innovation_history_high", {}).items():
            state.innovation_history_high[k] = deque(v, maxlen=_INNOVATION_WINDOW)
        for k, v in d.get("innovation_history_low", {}).items():
            state.innovation_history_low[k] = deque(v, maxlen=_INNOVATION_WINDOW)
        state.initial_r_estimates = d.get("initial_r_estimates", {})
        state.n_settlements = d.get("n_settlements", 0)
        return state

    @classmethod
    def load(cls, path: str | Path = "analysis_data/letkf_tune_state.json") -> "LETKFTuneState":
        p = Path(path)
        if p.exists():
            try:
                return cls.from_dict(json.loads(p.read_text()))
            except (json.JSONDecodeError, TypeError, KeyError):
                log.warning("Corrupt LETKF tune state at %s, starting fresh", p)
        return cls()

    def save(self, path: str | Path = "analysis_data/letkf_tune_state.json") -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2))


def autotune_letkf(
    letkf_state: LETKFState,
    observed_high_f: float | None,
    observed_low_f: float | None,
    tune_state: LETKFTuneState | None = None,
    tune_state_path: str | Path = "analysis_data/letkf_tune_state.json",
) -> dict:
    """Post-settlement LETKF autotuning.

    Compares LETKF-analyzed temperatures to observed settlement highs/lows,
    tracks innovation statistics in a rolling window, and adjusts R scaling
    and RTPS inflation alpha when systematic patterns are detected.

    Args:
        letkf_state: The current LETKFState (will be mutated if tuning needed).
        observed_high_f: Observed settlement high temperature (°F).
        observed_low_f: Observed settlement low temperature (°F).
        tune_state: Existing tune state (loaded if None).
        tune_state_path: Path for persisting tune state.

    Returns:
        Summary dict with tuning actions taken.
    """
    if tune_state is None:
        tune_state = LETKFTuneState.load(tune_state_path)

    if letkf_state.analysis_mean is None:
        return {"status": "skipped", "reason": "no_analysis"}

    actions: list[str] = []
    stations = letkf_state.cluster.stations

    # Snapshot initial R estimates on first settlement
    if not tune_state.initial_r_estimates:
        for st in stations:
            if st.code in letkf_state.r_estimate:
                tune_state.initial_r_estimates[st.code] = letkf_state.r_estimate[st.code]
            else:
                tune_state.initial_r_estimates[st.code] = st.obs_sigma_f

    # Record innovations for each trading station
    for i, st in enumerate(stations):
        if not st.is_trading:
            continue

        analysis_temp = float(letkf_state.analysis_mean[i])

        if observed_high_f is not None:
            innovation_high = observed_high_f - analysis_temp
            if st.code not in tune_state.innovation_history_high:
                tune_state.innovation_history_high[st.code] = deque(maxlen=_INNOVATION_WINDOW)
            tune_state.innovation_history_high[st.code].append(innovation_high)

        if observed_low_f is not None:
            innovation_low = observed_low_f - analysis_temp
            if st.code not in tune_state.innovation_history_low:
                tune_state.innovation_history_low[st.code] = deque(maxlen=_INNOVATION_WINDOW)
            tune_state.innovation_history_low[st.code].append(innovation_low)

    tune_state.n_settlements += 1

    # Need at least 3 settlements before making tuning decisions
    if tune_state.n_settlements < 3:
        tune_state.save(tune_state_path)
        return {"status": "accumulating", "n_settlements": tune_state.n_settlements, "actions": []}

    # Analyze innovation statistics and adjust R / alpha per trading station
    r_adjustments: dict[str, dict] = {}

    for st in stations:
        if not st.is_trading:
            continue

        for mt, history_dict in [
            ("high", tune_state.innovation_history_high),
            ("low", tune_state.innovation_history_low),
        ]:
            innovations = history_dict.get(st.code)
            if not innovations or len(innovations) < 3:
                continue

            inn_arr = np.array(list(innovations))
            inn_mean = float(np.mean(inn_arr))
            inn_std = float(np.std(inn_arr))

            # Check for consistent bias -> adjust R scaling
            # Positive bias = analysis is too low (undertrusting obs)
            # Negative bias = analysis is too high (overtrusting obs)
            current_r = letkf_state.r_estimate.get(st.code, st.obs_sigma_f)

            if abs(inn_mean) > _BIAS_THRESHOLD_F:
                # Bias detected: if analysis undertrusts obs (positive innovation),
                # decrease R to trust observations more, and vice versa.
                if inn_mean > 0:
                    # Analysis too low -> decrease R (trust obs more)
                    new_r = current_r * 0.9
                    direction = "decrease"
                else:
                    # Analysis too high -> increase R (trust obs less)
                    new_r = current_r * 1.1
                    direction = "increase"

                new_r = max(0.15, min(2.5, new_r))  # safety clamp
                letkf_state.r_estimate[st.code] = new_r
                action = (
                    f"{st.code} {mt}: R {direction} {current_r:.3f}->{new_r:.3f} "
                    f"(bias={inn_mean:+.2f}°F)"
                )
                actions.append(action)
                r_adjustments[f"{st.code}_{mt}"] = {
                    "old_r": round(current_r, 4),
                    "new_r": round(new_r, 4),
                    "bias": round(inn_mean, 3),
                    "direction": direction,
                }

            # Check if innovations are too large -> increase RTPS alpha
            analysis_spread = float(letkf_state.analysis_spread[
                next(j for j, s in enumerate(stations) if s.code == st.code)
            ])
            if analysis_spread > 0 and inn_std / analysis_spread > _SPREAD_RATIO_THRESHOLD:
                old_alpha = letkf_state.cluster.rtps_alpha
                # Can't mutate frozen ClusterDef, but we can note it for config adjustment
                suggested_alpha = min(0.95, old_alpha + 0.02)
                action = (
                    f"{st.code} {mt}: suggest RTPS alpha {old_alpha:.2f}->{suggested_alpha:.2f} "
                    f"(inn_std/spread={inn_std / analysis_spread:.2f})"
                )
                actions.append(action)

    # Persist
    tune_state.save(tune_state_path)

    for a in actions:
        log.info("LETKF autotune: %s", a)

    return {
        "status": "tuned" if actions else "stable",
        "n_settlements": tune_state.n_settlements,
        "actions": actions,
        "r_adjustments": r_adjustments if r_adjustments else None,
    }


def letkf_diagnostics(
    letkf_state: LETKFState,
    tune_state: LETKFTuneState | None = None,
    tune_state_path: str | Path = "analysis_data/letkf_tune_state.json",
) -> dict:
    """Generate LETKF diagnostic summary for human review.

    Returns a dict with:
      - per-station analysis spread
      - innovation statistics (mean, std over rolling window)
      - adaptive R estimates vs initial R
      - stations flagged for R drift
    """
    if tune_state is None:
        tune_state = LETKFTuneState.load(tune_state_path)

    stations_diag: dict[str, dict] = {}

    for i, st in enumerate(letkf_state.cluster.stations):
        if not st.is_trading:
            continue

        spread = (
            round(float(letkf_state.analysis_spread[i]), 3)
            if letkf_state.analysis_spread is not None
            else None
        )
        current_r = letkf_state.r_estimate.get(st.code, st.obs_sigma_f)
        initial_r = tune_state.initial_r_estimates.get(st.code, st.obs_sigma_f)
        r_drift = abs(current_r - initial_r)

        diag: dict = {
            "analysis_spread": spread,
            "current_r": round(current_r, 4),
            "initial_r": round(initial_r, 4),
            "r_drift": round(r_drift, 4),
            "r_drift_flagged": r_drift > _R_DRIFT_THRESHOLD,
        }

        # Innovation stats for highs
        inn_high = tune_state.innovation_history_high.get(st.code)
        if inn_high and len(inn_high) >= 2:
            arr = np.array(list(inn_high))
            diag["innovation_high_mean"] = round(float(np.mean(arr)), 3)
            diag["innovation_high_std"] = round(float(np.std(arr)), 3)
            diag["innovation_high_n"] = len(inn_high)

        # Innovation stats for lows
        inn_low = tune_state.innovation_history_low.get(st.code)
        if inn_low and len(inn_low) >= 2:
            arr = np.array(list(inn_low))
            diag["innovation_low_mean"] = round(float(np.mean(arr)), 3)
            diag["innovation_low_std"] = round(float(np.std(arr)), 3)
            diag["innovation_low_n"] = len(inn_low)

        stations_diag[st.code] = diag

    return {
        "n_updates": letkf_state.n_updates,
        "n_settlements": tune_state.n_settlements,
        "stations": stations_diag,
    }
