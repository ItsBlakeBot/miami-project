"""Forward curve builder — extracts multi-model hourly forecasts from the DB.

Standalone replacement for the forward curve logic that was previously embedded
in signals.signals.SignalExtractor._build_forward_curve. Queries model_forecasts,
atmospheric_data, and pressure_levels to produce a 12-hour lookahead curve.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone


# Models whose hourly point forecasts feed the forward curve.
FORWARD_MODELS = {
    "NBM": "nbm_temp_f",
    "GFS-Global": "gfs_temp_f",
    "ECMWF-IFS": "ecmwf_temp_f",
    "GFS-HRRR": "hrrr_temp_f",
    "NAM": "nam_temp_f",
}


def build_forward_curve(
    db_path: str,
    station: str,
    utc_offset: int,
    target_date: str,
    *,
    now_utc: datetime | None = None,
) -> list[dict]:
    """Build a 12-hour forward-looking forecast curve from current time.

    Returns a list of dicts matching the forward_curves table columns.
    """
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)

    conn = sqlite3.connect(db_path, timeout=10)
    try:
        rows: list[dict] = []
        current_hour = now_utc.replace(minute=0, second=0, microsecond=0)

        for offset in range(1, 13):
            fwd_utc = current_hour + timedelta(hours=offset)
            fwd_str = fwd_utc.strftime("%Y-%m-%d %H:%M:%S")
            fwd_str_iso = fwd_utc.strftime("%Y-%m-%dT%H:%M")
            fwd_next = (fwd_utc + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")
            fwd_next_iso = (fwd_utc + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M")
            utc_hour = fwd_utc.hour

            # --- Multi-model temps ---
            model_temps: dict[str, float] = {}
            for model_name in FORWARD_MODELS:
                row = conn.execute(
                    """SELECT COALESCE(
                           raw_temperature_f,
                           json_extract(source_record_json, '$.temperature_f')
                       ) as temp_f
                       FROM model_forecasts
                       WHERE station=? AND model=?
                         AND (valid_time >= ? AND valid_time < ?
                              OR valid_time >= ? AND valid_time < ?)
                         AND (raw_temperature_f IS NOT NULL
                              OR json_extract(source_record_json, '$.temperature_f') IS NOT NULL)
                       ORDER BY fetch_time_utc DESC LIMIT 1""",
                    (station, model_name,
                     fwd_str, fwd_next,
                     fwd_str_iso, fwd_next_iso),
                ).fetchone()
                if row and row[0] is not None:
                    model_temps[model_name] = round(float(row[0]), 1)

            # Compute spread across available models
            temps = list(model_temps.values())
            model_min = round(min(temps), 1) if temps else None
            model_max = round(max(temps), 1) if temps else None
            model_spread = round(model_max - model_min, 1) if model_min is not None and model_max is not None else None

            # --- Atmospheric forecast ---
            atmos = conn.execute(
                """SELECT cape, precipitable_water_mm, precipitation_probability,
                          COALESCE(precipitation_mm, 0), shortwave_radiation
                   FROM atmospheric_data
                   WHERE station=?
                     AND (valid_time_utc >= ? AND valid_time_utc < ?
                          OR valid_time_utc >= ? AND valid_time_utc < ?)
                   ORDER BY id DESC LIMIT 1""",
                (station, fwd_str, fwd_next, fwd_str_iso, fwd_next_iso),
            ).fetchone()

            # --- Pressure levels ---
            plev = conn.execute(
                """SELECT temp_850_c, temp_925_c, wind_dir_850, wind_speed_850
                   FROM pressure_levels
                   WHERE station=?
                     AND (valid_time_utc >= ? AND valid_time_utc < ?
                          OR valid_time_utc >= ? AND valid_time_utc < ?)
                   ORDER BY id DESC LIMIT 1""",
                (station, fwd_str, fwd_next, fwd_str_iso, fwd_next_iso),
            ).fetchone()

            rows.append(dict(
                valid_hour_utc=f"{utc_hour:02d}Z",
                hours_ahead=offset,
                nbm_temp_f=model_temps.get("NBM"),
                gfs_temp_f=model_temps.get("GFS-Global"),
                ecmwf_temp_f=model_temps.get("ECMWF-IFS"),
                hrrr_temp_f=model_temps.get("GFS-HRRR"),
                nam_temp_f=model_temps.get("NAM"),
                model_min_f=model_min,
                model_max_f=model_max,
                model_spread_f=model_spread,
                cape=round(atmos[0], 0) if atmos and atmos[0] is not None else None,
                pw_mm=round(atmos[1], 1) if atmos and atmos[1] is not None else None,
                precip_prob=round(atmos[2], 0) if atmos and atmos[2] is not None else None,
                precip_mm=round(atmos[3], 1) if atmos and atmos[3] is not None else None,
                solar_wm2=round(atmos[4], 0) if atmos and atmos[4] is not None else None,
                temp_850_c=round(plev[0], 1) if plev and plev[0] is not None else None,
                temp_925_c=round(plev[1], 1) if plev and plev[1] is not None else None,
                wind_dir_850=round(plev[2], 0) if plev and plev[2] is not None else None,
                wind_speed_850=round(plev[3], 0) if plev and plev[3] is not None else None,
            ))

        return rows
    finally:
        conn.close()
