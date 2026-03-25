"""T7.3: Concept drift detection for weather trading.

Monitors for distributional shifts in features and forecast errors that
indicate the model needs recalibration. Detects:
  - Seasonal transitions (winter → spring → summer calibration drift)
  - NWS model upgrades (HRRR/GFS version changes)
  - Data quality changes (sensor degradation, new obs sources)
  - Systematic forecast bias drift

Uses Population Stability Index (PSI) to compare recent feature distributions
against a reference period. PSI > 0.1 = moderate drift, PSI > 0.25 = severe.
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class DriftAlert:
    """A single drift detection alert."""

    feature: str
    psi: float
    severity: str  # "none", "moderate", "severe"
    reference_mean: float
    recent_mean: float
    shift: float  # recent_mean - reference_mean

    def to_dict(self) -> dict:
        return {
            "feature": self.feature,
            "psi": round(self.psi, 4),
            "severity": self.severity,
            "reference_mean": round(self.reference_mean, 3),
            "recent_mean": round(self.recent_mean, 3),
            "shift": round(self.shift, 3),
        }


@dataclass
class DriftReport:
    """Concept drift detection report."""

    report_date: str
    station: str
    alerts: list[DriftAlert] = field(default_factory=list)
    recalibration_recommended: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "report_date": self.report_date,
            "station": self.station,
            "n_alerts": len(self.alerts),
            "recalibration_recommended": self.recalibration_recommended,
            "alerts": [a.to_dict() for a in self.alerts],
            "notes": self.notes,
        }


def population_stability_index(
    reference: np.ndarray,
    recent: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Population Stability Index between two distributions.

    PSI = Σ (p_recent - p_reference) * ln(p_recent / p_reference)

    Interpretation:
      PSI < 0.1: no significant drift
      0.1 ≤ PSI < 0.25: moderate drift — monitor closely
      PSI ≥ 0.25: severe drift — recalibrate

    Uses equal-frequency binning from the reference distribution.
    """
    if len(reference) < 10 or len(recent) < 10:
        return 0.0

    # Create bins from reference distribution (equal-frequency)
    quantiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(reference, quantiles)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    # Compute bin proportions
    ref_counts = np.histogram(reference, bins=bin_edges)[0].astype(float)
    rec_counts = np.histogram(recent, bins=bin_edges)[0].astype(float)

    # Normalize to proportions (with floor to avoid division by zero)
    epsilon = 1e-4
    ref_props = np.maximum(ref_counts / ref_counts.sum(), epsilon)
    rec_props = np.maximum(rec_counts / rec_counts.sum(), epsilon)

    # PSI
    psi = np.sum((rec_props - ref_props) * np.log(rec_props / ref_props))
    return max(0.0, float(psi))


def detect_concept_drift(
    db: sqlite3.Connection,
    station: str = "KMIA",
    reference_days: int = 60,
    recent_days: int = 7,
    reference_date: str | None = None,
) -> DriftReport:
    """Detect concept drift by comparing recent vs reference feature distributions.

    Compares the distribution of forecast errors and atmospheric features over
    a recent window against a longer reference window.

    Args:
        db: SQLite connection with row_factory=sqlite3.Row
        station: Station identifier
        reference_days: Length of reference window (days)
        recent_days: Length of recent window (days)
        reference_date: End date for recent window (default: today)
    """
    if reference_date is None:
        ref = date.today()
    else:
        ref = date.fromisoformat(reference_date)

    recent_start = (ref - timedelta(days=recent_days)).isoformat()
    reference_start = (ref - timedelta(days=reference_days)).isoformat()
    recent_end = ref.isoformat()

    report = DriftReport(report_date=recent_end, station=station)

    # Feature 1: Forecast error distribution (high)
    _check_forecast_error_drift(db, station, reference_start, recent_start, recent_end, "high", report)
    _check_forecast_error_drift(db, station, reference_start, recent_start, recent_end, "low", report)

    # Feature 2: CAPE distribution
    _check_atmospheric_drift(db, station, reference_start, recent_start, recent_end, "cape", report)

    # Feature 3: Precipitable water distribution
    _check_atmospheric_drift(db, station, reference_start, recent_start, recent_end, "precipitable_water_mm", report)

    # Feature 4: Temperature distribution
    _check_temperature_drift(db, station, reference_start, recent_start, recent_end, report)

    # Determine recalibration recommendation
    severe_count = sum(1 for a in report.alerts if a.severity == "severe")
    moderate_count = sum(1 for a in report.alerts if a.severity == "moderate")

    if severe_count >= 1:
        report.recalibration_recommended = True
        report.notes.append(f"{severe_count} severe drift alerts — EMOS recalibration recommended")
    elif moderate_count >= 2:
        report.recalibration_recommended = True
        report.notes.append(f"{moderate_count} moderate drift alerts — consider recalibration")

    return report


def _check_forecast_error_drift(
    db: sqlite3.Connection,
    station: str,
    ref_start: str,
    rec_start: str,
    rec_end: str,
    market_type: str,
    report: DriftReport,
) -> None:
    """Check drift in forecast error distribution."""
    field_name = "forecast_high_f" if market_type == "high" else "forecast_low_f"
    settle_type = market_type

    rows = db.execute(
        f"""SELECT mf.{field_name} - es.actual_value_f AS error
            FROM model_forecasts mf
            JOIN event_settlements es
              ON es.station = mf.station
              AND es.settlement_date = mf.forecast_date
              AND es.market_type = ?
            WHERE mf.station = ?
              AND mf.forecast_date BETWEEN ? AND ?
              AND mf.{field_name} IS NOT NULL
              AND es.actual_value_f IS NOT NULL""",
        (settle_type, station, ref_start, rec_end),
    ).fetchall()

    if len(rows) < 20:
        return

    errors = np.array([r["error"] for r in rows])

    # Split into reference and recent
    ref_rows = db.execute(
        f"""SELECT mf.{field_name} - es.actual_value_f AS error
            FROM model_forecasts mf
            JOIN event_settlements es
              ON es.station = mf.station
              AND es.settlement_date = mf.forecast_date
              AND es.market_type = ?
            WHERE mf.station = ?
              AND mf.forecast_date BETWEEN ? AND ?
              AND mf.{field_name} IS NOT NULL
              AND es.actual_value_f IS NOT NULL""",
        (settle_type, station, ref_start, rec_start),
    ).fetchall()

    rec_rows = db.execute(
        f"""SELECT mf.{field_name} - es.actual_value_f AS error
            FROM model_forecasts mf
            JOIN event_settlements es
              ON es.station = mf.station
              AND es.settlement_date = mf.forecast_date
              AND es.market_type = ?
            WHERE mf.station = ?
              AND mf.forecast_date BETWEEN ? AND ?
              AND mf.{field_name} IS NOT NULL
              AND es.actual_value_f IS NOT NULL""",
        (settle_type, station, rec_start, rec_end),
    ).fetchall()

    if len(ref_rows) < 10 or len(rec_rows) < 5:
        return

    ref_errors = np.array([r["error"] for r in ref_rows])
    rec_errors = np.array([r["error"] for r in rec_rows])

    psi = population_stability_index(ref_errors, rec_errors)
    severity = "severe" if psi >= 0.25 else ("moderate" if psi >= 0.1 else "none")

    if severity != "none":
        report.alerts.append(DriftAlert(
            feature=f"forecast_error_{market_type}",
            psi=psi,
            severity=severity,
            reference_mean=float(np.mean(ref_errors)),
            recent_mean=float(np.mean(rec_errors)),
            shift=float(np.mean(rec_errors) - np.mean(ref_errors)),
        ))


def _check_atmospheric_drift(
    db: sqlite3.Connection,
    station: str,
    ref_start: str,
    rec_start: str,
    rec_end: str,
    field: str,
    report: DriftReport,
) -> None:
    """Check drift in atmospheric feature distribution."""
    ref_rows = db.execute(
        f"""SELECT {field} FROM atmospheric_data
            WHERE station = ? AND valid_time_utc BETWEEN ? AND ?
              AND {field} IS NOT NULL""",
        (station, ref_start + "T00:00:00Z", rec_start + "T00:00:00Z"),
    ).fetchall()

    rec_rows = db.execute(
        f"""SELECT {field} FROM atmospheric_data
            WHERE station = ? AND valid_time_utc BETWEEN ? AND ?
              AND {field} IS NOT NULL""",
        (station, rec_start + "T00:00:00Z", rec_end + "T23:59:59Z"),
    ).fetchall()

    if len(ref_rows) < 10 or len(rec_rows) < 5:
        return

    ref_vals = np.array([r[field] for r in ref_rows])
    rec_vals = np.array([r[field] for r in rec_rows])

    psi = population_stability_index(ref_vals, rec_vals)
    severity = "severe" if psi >= 0.25 else ("moderate" if psi >= 0.1 else "none")

    if severity != "none":
        report.alerts.append(DriftAlert(
            feature=field,
            psi=psi,
            severity=severity,
            reference_mean=float(np.mean(ref_vals)),
            recent_mean=float(np.mean(rec_vals)),
            shift=float(np.mean(rec_vals) - np.mean(ref_vals)),
        ))


def _check_temperature_drift(
    db: sqlite3.Connection,
    station: str,
    ref_start: str,
    rec_start: str,
    rec_end: str,
    report: DriftReport,
) -> None:
    """Check drift in observed temperature distribution."""
    ref_rows = db.execute(
        """SELECT temperature_f FROM observations
           WHERE station = ? AND timestamp_utc BETWEEN ? AND ?
             AND temperature_f IS NOT NULL""",
        (station, ref_start + "T00:00:00Z", rec_start + "T00:00:00Z"),
    ).fetchall()

    rec_rows = db.execute(
        """SELECT temperature_f FROM observations
           WHERE station = ? AND timestamp_utc BETWEEN ? AND ?
             AND temperature_f IS NOT NULL""",
        (station, rec_start + "T00:00:00Z", rec_end + "T23:59:59Z"),
    ).fetchall()

    if len(ref_rows) < 50 or len(rec_rows) < 20:
        return

    ref_temps = np.array([r["temperature_f"] for r in ref_rows])
    rec_temps = np.array([r["temperature_f"] for r in rec_rows])

    psi = population_stability_index(ref_temps, rec_temps)
    severity = "severe" if psi >= 0.25 else ("moderate" if psi >= 0.1 else "none")

    if severity != "none":
        report.alerts.append(DriftAlert(
            feature="observed_temperature",
            psi=psi,
            severity=severity,
            reference_mean=float(np.mean(ref_temps)),
            recent_mean=float(np.mean(rec_temps)),
            shift=float(np.mean(rec_temps) - np.mean(ref_temps)),
        ))


def main() -> None:
    """Run drift detection and print report."""
    import argparse

    parser = argparse.ArgumentParser(description="Detect concept drift")
    parser.add_argument("--db", required=True)
    parser.add_argument("--station", default="KMIA")
    parser.add_argument("--reference-days", type=int, default=60)
    parser.add_argument("--recent-days", type=int, default=7)
    parser.add_argument("--out", default="analysis_data/drift_report.json")
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row

    report = detect_concept_drift(conn, args.station, args.reference_days, args.recent_days)
    conn.close()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report.to_dict(), indent=2))

    print(f"Drift report for {report.station} ({report.report_date}):")
    print(f"  Recalibration recommended: {report.recalibration_recommended}")
    print(f"  Alerts: {len(report.alerts)}")
    for a in report.alerts:
        print(f"    [{a.severity}] {a.feature}: PSI={a.psi:.3f}, shift={a.shift:+.2f}")
    for n in report.notes:
        print(f"  Note: {n}")


if __name__ == "__main__":
    main()
