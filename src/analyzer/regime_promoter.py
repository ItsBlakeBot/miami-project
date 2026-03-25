"""Regime discovery -> human review -> catalog promotion pipeline.

Connects the HDP-Sticky HMM shadow lane (hdp_regime_discovery.py) to the
live regime catalog (regime_catalog.py) via a human-in-the-loop review step.

Flow:
  1. HDP-HMM discovers latent states from surface obs.
  2. analyze_regime_alignment() compares discovered states against the 6-regime catalog.
  3. generate_review_report() writes a JSON report for human inspection.
  4. Human reviews, approves/dismisses candidates.
  5. promote_regime() writes approved regimes to promoted_regimes.json.
  6. The live catalog can load promoted regimes at startup (additive only).

Uses only stdlib + dataclasses.  No numpy dependency — all math is plain Python.
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ANALYSIS_DIR = Path("analysis_data")
REVIEW_REPORT_PATH = ANALYSIS_DIR / "regime_review_report.json"
PROMOTED_REGIMES_PATH = ANALYSIS_DIR / "promoted_regimes.json"

# ---------------------------------------------------------------------------
# Catalog regime atmospheric signatures (reference for overlap computation)
# ---------------------------------------------------------------------------
# Each signature is a dict of {feature: (mean, std)} describing the typical
# atmospheric state when the regime is active.  These are approximate
# climatological values for KMIA derived from the catalog definitions.
_CATALOG_SIGNATURES: dict[str, dict[str, tuple[float, float]]] = {
    "marine_stable": {
        "temperature_f": (82.0, 4.0),
        "dew_point_f": (73.0, 3.0),
        "pressure_hpa": (1015.0, 2.0),
        "wind_speed_mph": (10.0, 4.0),
    },
    "inland_heating": {
        "temperature_f": (88.0, 5.0),
        "dew_point_f": (70.0, 4.0),
        "pressure_hpa": (1014.0, 2.5),
        "wind_speed_mph": (5.0, 3.0),
    },
    "cloud_suppression": {
        "temperature_f": (80.0, 3.0),
        "dew_point_f": (72.0, 3.0),
        "pressure_hpa": (1016.0, 2.0),
        "wind_speed_mph": (8.0, 4.0),
    },
    "convective_outflow": {
        "temperature_f": (78.0, 6.0),
        "dew_point_f": (72.0, 4.0),
        "pressure_hpa": (1013.0, 3.0),
        "wind_speed_mph": (15.0, 8.0),
    },
    "frontal_passage": {
        "temperature_f": (68.0, 8.0),
        "dew_point_f": (55.0, 10.0),
        "pressure_hpa": (1020.0, 4.0),
        "wind_speed_mph": (12.0, 6.0),
    },
    "transition": {
        "temperature_f": (82.0, 6.0),
        "dew_point_f": (70.0, 5.0),
        "pressure_hpa": (1015.0, 3.0),
        "wind_speed_mph": (8.0, 5.0),
    },
}

# Feature order matching HDP-HMM emission: [temp_f, dew_f, pressure_hpa, wind_speed_mph]
_FEATURE_KEYS = ["temperature_f", "dew_point_f", "pressure_hpa", "wind_speed_mph"]


# ---------------------------------------------------------------------------
# RegimeCandidate dataclass
# ---------------------------------------------------------------------------
@dataclass
class RegimeCandidate:
    """A candidate regime discovered by HDP-HMM, pending human review."""

    hmm_state_id: int
    proposed_name: str  # auto-generated from atmospheric signature
    atmospheric_signature: dict  # mean values for each feature
    forecast_error_stats: dict  # mu_bias_high, mu_bias_low, sigma_scale, sample_count
    example_dates: list[str]  # dates when this regime was dominant
    overlap_with_catalog: dict[str, float]  # {existing_regime: overlap_fraction}
    confidence: float  # how distinct from existing regimes (0-1)
    recommendation: str  # "promote", "hold", "dismiss"
    recommendation_reason: str
    status: str = "pending_review"  # pending_review | promoted | dismissed | held
    letkf_diagnostics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Atmospheric signature helpers
# ---------------------------------------------------------------------------
def _extract_atmospheric_signature(regime_params: dict) -> dict:
    """Extract a readable atmospheric signature from HDP-HMM emission params.

    Parameters
    ----------
    regime_params : dict
        Must contain 'mean' (list of 4 floats) in the order
        [temp_f, dew_f, pressure_hpa, wind_speed_mph].

    Returns
    -------
    dict with named features and their mean values.
    """
    mean = regime_params.get("mean", [0.0] * 4)
    sig: dict = {}
    for i, key in enumerate(_FEATURE_KEYS):
        sig[key] = round(mean[i], 2) if i < len(mean) else 0.0
    return sig


def _compute_overlap(hmm_sig: dict, catalog_sig: dict[str, tuple[float, float]]) -> float:
    """Compute Gaussian overlap between an HMM state signature and a catalog regime.

    Uses a product-of-univariate-Gaussian approximation: for each feature,
    the overlap is exp(-0.5 * (mu1 - mu2)^2 / (s1^2 + s2^2)).  The overall
    overlap is the geometric mean across features.

    Returns a value in [0, 1] where 1 means identical signatures.
    """
    log_overlaps: list[float] = []
    for key in _FEATURE_KEYS:
        hmm_val = hmm_sig.get(key, 0.0)
        cat_mean, cat_std = catalog_sig.get(key, (0.0, 10.0))
        # Assume HMM state std is comparable to catalog std
        combined_var = cat_std ** 2 + cat_std ** 2  # symmetric assumption
        if combined_var < 1e-6:
            combined_var = 1e-6
        diff_sq = (hmm_val - cat_mean) ** 2
        log_overlaps.append(-0.5 * diff_sq / combined_var)
    if not log_overlaps:
        return 0.0
    # Geometric mean of per-feature overlaps
    return math.exp(sum(log_overlaps) / len(log_overlaps))


def _auto_name(sig: dict) -> str:
    """Generate a proposed name from an atmospheric signature."""
    temp = sig.get("temperature_f", 80.0)
    dew = sig.get("dew_point_f", 70.0)
    wind = sig.get("wind_speed_mph", 8.0)
    pressure = sig.get("pressure_hpa", 1015.0)

    parts: list[str] = []

    # Temperature character
    if temp > 90:
        parts.append("hot")
    elif temp < 70:
        parts.append("cool")
    elif temp > 83:
        parts.append("warm")
    else:
        parts.append("mild")

    # Moisture character
    spread = temp - dew
    if spread < 5:
        parts.append("humid")
    elif spread > 15:
        parts.append("dry")

    # Wind character
    if wind > 15:
        parts.append("windy")
    elif wind < 5:
        parts.append("calm")

    # Pressure character
    if pressure > 1020:
        parts.append("highpressure")
    elif pressure < 1010:
        parts.append("lowpressure")

    return "_".join(parts) if parts else "unknown_regime"


# ---------------------------------------------------------------------------
# 2. analyze_regime_alignment()
# ---------------------------------------------------------------------------
def analyze_regime_alignment(
    hdp_results: list[dict],
    catalog_signatures: dict[str, dict[str, tuple[float, float]]] | None = None,
) -> tuple[list[RegimeCandidate], list[str]]:
    """Compare HDP-HMM discovered states against the live regime catalog.

    Parameters
    ----------
    hdp_results : list[dict]
        Each dict represents one day's HDP output with keys:
          target_date, regime_params (list of {mean, cov}),
          regime_sequence (list of int), n_regimes_discovered.
    catalog_signatures : dict, optional
        Override catalog signatures for testing.  Defaults to _CATALOG_SIGNATURES.

    Returns
    -------
    candidates : list[RegimeCandidate]
        One per unique HMM state found across all provided days.
    potentially_obsolete : list[str]
        Catalog regime names with < 0.2 total overlap against any HMM state.
    """
    if catalog_signatures is None:
        catalog_signatures = _CATALOG_SIGNATURES

    # Aggregate HMM states across days
    # Key: state_id -> {params, dates, obs_count}
    state_agg: dict[int, dict] = {}

    for day in hdp_results:
        target_date = day.get("target_date", "unknown")
        params_list = day.get("regime_params", [])
        sequence = day.get("regime_sequence", [])

        for state_id, params in enumerate(params_list):
            if state_id not in state_agg:
                state_agg[state_id] = {
                    "params": params,
                    "dates": [],
                    "obs_count": 0,
                }
            state_agg[state_id]["dates"].append(target_date)
            state_agg[state_id]["obs_count"] += sequence.count(state_id)

    # Build candidates
    candidates: list[RegimeCandidate] = []

    for state_id, agg in state_agg.items():
        sig = _extract_atmospheric_signature(agg["params"])

        # Compute overlap with each catalog regime
        overlaps: dict[str, float] = {}
        for regime_name, cat_sig in catalog_signatures.items():
            overlaps[regime_name] = round(_compute_overlap(sig, cat_sig), 4)

        max_overlap = max(overlaps.values()) if overlaps else 0.0
        is_novel = max_overlap < 0.3
        n_days = len(set(agg["dates"]))

        # Confidence: how distinct from existing regimes
        # Lower max overlap = more distinct = higher confidence
        confidence = round(max(0.0, min(1.0, 1.0 - max_overlap)), 4)

        # Recommendation logic
        if not is_novel:
            recommendation = "dismiss"
            reason = (
                f"High overlap ({max_overlap:.2f}) with existing regime "
                f"'{max(overlaps, key=overlaps.get)}'"
            )
        elif n_days < 5:
            recommendation = "hold"
            reason = (
                f"Novel (max overlap {max_overlap:.2f}) but only {n_days} example "
                f"day(s); need >= 5 for statistical confidence"
            )
        else:
            recommendation = "promote"
            reason = (
                f"Novel (max overlap {max_overlap:.2f}), {n_days} example days, "
                f"confidence {confidence:.2f}"
            )

        # Placeholder forecast error stats — filled in by calibration data if available
        forecast_error_stats = {
            "mu_bias_high": 0.0,
            "mu_bias_low": 0.0,
            "sigma_scale": 1.0,
            "sample_count": agg["obs_count"],
        }

        candidate = RegimeCandidate(
            hmm_state_id=state_id,
            proposed_name=_auto_name(sig),
            atmospheric_signature=sig,
            forecast_error_stats=forecast_error_stats,
            example_dates=sorted(set(agg["dates"])),
            overlap_with_catalog=overlaps,
            confidence=confidence,
            recommendation=recommendation,
            recommendation_reason=reason,
        )
        candidates.append(candidate)

    # Detect potentially obsolete catalog regimes
    # A catalog regime is "potentially obsolete" if no HMM state overlaps
    # with it above 0.2
    potentially_obsolete: list[str] = []
    for regime_name in catalog_signatures:
        best_overlap = 0.0
        for c in candidates:
            overlap_val = c.overlap_with_catalog.get(regime_name, 0.0)
            if overlap_val > best_overlap:
                best_overlap = overlap_val
        if best_overlap < 0.2:
            potentially_obsolete.append(regime_name)

    return candidates, potentially_obsolete


# ---------------------------------------------------------------------------
# 3. generate_review_report()
# ---------------------------------------------------------------------------
def generate_review_report(
    candidates: list[RegimeCandidate],
    potentially_obsolete: list[str],
    output_path: Path | None = None,
) -> Path:
    """Write a human-readable JSON report for regime review.

    Parameters
    ----------
    candidates : list[RegimeCandidate]
        Output from analyze_regime_alignment().
    potentially_obsolete : list[str]
        Catalog regimes that may no longer be relevant.
    output_path : Path, optional
        Where to write the report.  Defaults to REVIEW_REPORT_PATH.

    Returns
    -------
    Path to the written report.
    """
    if output_path is None:
        output_path = REVIEW_REPORT_PATH

    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "summary": {
            "total_candidates": len(candidates),
            "promote_recommended": sum(1 for c in candidates if c.recommendation == "promote"),
            "hold_recommended": sum(1 for c in candidates if c.recommendation == "hold"),
            "dismiss_recommended": sum(1 for c in candidates if c.recommendation == "dismiss"),
            "potentially_obsolete_regimes": potentially_obsolete,
        },
        "candidates": [],
    }

    for c in candidates:
        # Recommended catalog parameters for promotion
        sig = c.atmospheric_signature
        temp = sig.get("temperature_f", 80.0)
        wind = sig.get("wind_speed_mph", 8.0)

        # Conservative defaults for new regimes
        rec_sigma = 1.15  # slightly wide — new regime, limited data
        if wind > 15:
            rec_sigma = 1.4
        elif temp > 90 or temp < 65:
            rec_sigma = 1.2

        candidate_entry = {
            "hmm_state_id": c.hmm_state_id,
            "proposed_name": c.proposed_name,
            "atmospheric_signature": c.atmospheric_signature,
            "example_dates": c.example_dates,
            "forecast_error_stats": c.forecast_error_stats,
            "overlap_with_catalog": c.overlap_with_catalog,
            "confidence": c.confidence,
            "recommendation": c.recommendation,
            "recommendation_reason": c.recommendation_reason,
            "status": c.status,
            "recommended_catalog_params": {
                "sigma_multiplier": round(rec_sigma, 3),
                "mu_bias_high_f": round(c.forecast_error_stats.get("mu_bias_high", 0.0), 2),
                "mu_bias_low_f": round(c.forecast_error_stats.get("mu_bias_low", 0.0), 2),
                "min_confidence_for_sizing": 0.6,  # conservative for new regimes
            },
            "letkf_diagnostics": c.letkf_diagnostics,
        }
        report["candidates"].append(candidate_entry)

    output_path.write_text(json.dumps(report, indent=2) + "\n")
    log.info("Regime review report written to %s (%d candidates)", output_path, len(candidates))
    return output_path


# ---------------------------------------------------------------------------
# 4. promote_regime()
# ---------------------------------------------------------------------------
def promote_regime(
    candidate: RegimeCandidate,
    promoted_path: Path | None = None,
) -> dict:
    """Promote an approved RegimeCandidate to the promoted regimes file.

    Appends the regime to promoted_regimes.json.  The live regime catalog
    can load these at startup to extend itself beyond the hardcoded 6 regimes.

    New regimes enter with conservative sizing (min_confidence_for_sizing = 0.6).

    Parameters
    ----------
    candidate : RegimeCandidate
        Must have recommendation == "promote" (enforced here).
    promoted_path : Path, optional
        Defaults to PROMOTED_REGIMES_PATH.

    Returns
    -------
    dict describing the promoted regime entry.
    """
    if promoted_path is None:
        promoted_path = PROMOTED_REGIMES_PATH

    promoted_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing promoted regimes
    existing: list[dict] = []
    if promoted_path.exists():
        try:
            existing = json.loads(promoted_path.read_text())
        except (json.JSONDecodeError, ValueError):
            log.warning("Could not parse %s — starting fresh", promoted_path)
            existing = []

    # Check for duplicate
    for entry in existing:
        if entry.get("proposed_name") == candidate.proposed_name:
            log.warning(
                "Regime '%s' already promoted — updating in place",
                candidate.proposed_name,
            )
            existing = [e for e in existing if e.get("proposed_name") != candidate.proposed_name]
            break

    # Build the promoted entry with conservative parameters
    sig = candidate.atmospheric_signature
    wind = sig.get("wind_speed_mph", 8.0)
    temp = sig.get("temperature_f", 80.0)

    sigma_mult = 1.15
    if wind > 15:
        sigma_mult = 1.4
    elif temp > 90 or temp < 65:
        sigma_mult = 1.2

    promoted_entry = {
        "name": candidate.proposed_name,
        "hmm_state_id": candidate.hmm_state_id,
        "atmospheric_signature": candidate.atmospheric_signature,
        "sigma_multiplier": round(sigma_mult, 3),
        "mu_bias_high_f": round(candidate.forecast_error_stats.get("mu_bias_high", 0.0), 2),
        "mu_bias_low_f": round(candidate.forecast_error_stats.get("mu_bias_low", 0.0), 2),
        "min_confidence_for_sizing": 0.6,  # conservative for new regimes
        "promoted_at": datetime.utcnow().isoformat() + "Z",
        "example_dates": candidate.example_dates,
        "confidence_at_promotion": candidate.confidence,
        "source": "hdp_hmm_discovery",
    }

    existing.append(promoted_entry)
    promoted_path.write_text(json.dumps(existing, indent=2) + "\n")

    log.info("Promoted regime '%s' to %s", candidate.proposed_name, promoted_path)
    return promoted_entry


# ---------------------------------------------------------------------------
# 5. incorporate_letkf_diagnostics()
# ---------------------------------------------------------------------------
def incorporate_letkf_diagnostics(
    candidates: list[RegimeCandidate],
    letkf_stats: dict,
) -> list[RegimeCandidate]:
    """Enrich regime candidates with LETKF innovation/spread diagnostics.

    Parameters
    ----------
    candidates : list[RegimeCandidate]
        Candidates from analyze_regime_alignment().
    letkf_stats : dict
        LETKF diagnostic output, expected keys:
          - per_date : dict[str, dict]  mapping date -> {innovation_mean, innovation_std,
            spread_mean, spread_std, spatial_disagreement}

    Returns
    -------
    The same list of candidates, mutated with letkf_diagnostics populated.
    """
    per_date = letkf_stats.get("per_date", {})
    if not per_date:
        log.debug("No LETKF per-date stats available; skipping enrichment")
        return candidates

    for candidate in candidates:
        date_innovations: list[float] = []
        date_spreads: list[float] = []
        high_disagreement_dates: list[str] = []

        for d in candidate.example_dates:
            if d not in per_date:
                continue
            day_stats = per_date[d]
            inn = day_stats.get("innovation_mean")
            sprd = day_stats.get("spread_mean")
            disagree = day_stats.get("spatial_disagreement", 0.0)

            if inn is not None:
                date_innovations.append(inn)
            if sprd is not None:
                date_spreads.append(sprd)
            if disagree > 0.5:
                high_disagreement_dates.append(d)

        diag: dict = {}
        if date_innovations:
            diag["mean_innovation"] = round(
                sum(date_innovations) / len(date_innovations), 4
            )
        if date_spreads:
            diag["mean_spread"] = round(
                sum(date_spreads) / len(date_spreads), 4
            )
        if high_disagreement_dates:
            diag["high_spatial_disagreement_dates"] = high_disagreement_dates
            diag["spatial_disagreement_flag"] = True
            candidate.recommendation_reason += (
                f"; LETKF spatial disagreement high on {len(high_disagreement_dates)} date(s)"
            )

        candidate.letkf_diagnostics = diag

    return candidates


# ---------------------------------------------------------------------------
# 6. Integration hook: run_regime_review_cycle()
# ---------------------------------------------------------------------------
def _load_hdp_results(db_path: Path, station: str, lookback_days: int = 30) -> list[dict]:
    """Load recent HDP-HMM results from the regime_labels_hdp_test table."""
    conn = sqlite3.connect(str(db_path), timeout=10)
    results: list[dict] = []
    try:
        rows = conn.execute(
            """SELECT target_date, n_regimes, regime_sequence,
                      regime_params, transition_matrix, phase_summary
               FROM regime_labels_hdp_test
               WHERE station = ?
               ORDER BY target_date DESC
               LIMIT ?""",
            (station, lookback_days),
        ).fetchall()

        for row in rows:
            try:
                results.append({
                    "target_date": row[0],
                    "n_regimes_discovered": row[1],
                    "regime_sequence": json.loads(row[2]) if row[2] else [],
                    "regime_params": json.loads(row[3]) if row[3] else [],
                    "transition_matrix": json.loads(row[4]) if row[4] else [],
                    "phase_summary": json.loads(row[5]) if row[5] else [],
                })
            except (json.JSONDecodeError, TypeError) as exc:
                log.warning("Skipping corrupt HDP row for %s: %s", row[0], exc)
    except sqlite3.OperationalError as exc:
        log.warning("Could not load HDP results (table may not exist): %s", exc)
    finally:
        conn.close()

    return results


def _load_letkf_diagnostics(db_path: Path, station: str) -> dict:
    """Load LETKF diagnostics from analysis_data if available."""
    letkf_path = ANALYSIS_DIR / "letkf_diagnostics.json"
    if not letkf_path.exists():
        return {}
    try:
        data = json.loads(letkf_path.read_text())
        # Filter to station if present
        if isinstance(data, dict) and "per_date" in data:
            return data
        return {}
    except (json.JSONDecodeError, ValueError) as exc:
        log.warning("Could not parse LETKF diagnostics: %s", exc)
        return {}


def run_regime_review_cycle(
    db_path: str | Path,
    station: str = "KMIA",
    lookback_days: int = 30,
    output_path: Path | None = None,
) -> str:
    """Run the full regime review cycle.

    1. Loads latest HDP-HMM results from the DB.
    2. Runs alignment analysis against the live catalog.
    3. Loads LETKF diagnostics if available.
    4. Generates the human-review report.
    5. Returns the path to the report.

    Callable from the daily post-settlement workflow.

    Parameters
    ----------
    db_path : str or Path
        Path to the collector SQLite database.
    station : str
        Station identifier (default: KMIA).
    lookback_days : int
        How many days of HDP results to consider.
    output_path : Path, optional
        Override report output path.

    Returns
    -------
    str : path to the generated review report.
    """
    db_path = Path(db_path)

    # Step 1: Load HDP-HMM results
    hdp_results = _load_hdp_results(db_path, station, lookback_days)
    if not hdp_results:
        log.info("No HDP-HMM results found for %s — skipping regime review", station)
        return ""

    log.info("Loaded %d days of HDP-HMM results for regime review", len(hdp_results))

    # Step 2: Run alignment analysis
    candidates, potentially_obsolete = analyze_regime_alignment(hdp_results)

    if potentially_obsolete:
        log.info(
            "Potentially obsolete catalog regimes: %s",
            ", ".join(potentially_obsolete),
        )

    # Step 3: Load LETKF diagnostics if available
    letkf_stats = _load_letkf_diagnostics(db_path, station)
    if letkf_stats:
        candidates = incorporate_letkf_diagnostics(candidates, letkf_stats)
        log.info("Enriched candidates with LETKF diagnostics")

    # Step 4: Generate the review report
    report_path = generate_review_report(candidates, potentially_obsolete, output_path)

    log.info(
        "Regime review cycle complete: %d candidates (%d promote, %d hold, %d dismiss)",
        len(candidates),
        sum(1 for c in candidates if c.recommendation == "promote"),
        sum(1 for c in candidates if c.recommendation == "hold"),
        sum(1 for c in candidates if c.recommendation == "dismiss"),
    )

    return str(report_path)
