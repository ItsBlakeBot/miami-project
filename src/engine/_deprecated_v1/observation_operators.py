"""T3.2: Observation operators for LETKF spatial assimilation.

Defines how each data source (ASOS, FAWN, buoy, METAR) maps to the
LETKF state vector (surface temperature at cluster stations).

Each operator specifies:
  - How to convert raw observations to the state space
  - Observation error characteristics (R matrix diagonal)
  - Quality control flags
  - Source-specific biases and corrections

The observation operator H maps state → observation space:
  y_predicted = H(x_state) + noise(R)

For surface temperature, H is simple: nearest-neighbor interpolation
from the cluster grid to the observation location. The complexity is
in getting R (observation error) right for each source type.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from engine.letkf import SurfaceObs, haversine_km


class ObsSourceType(Enum):
    """Observation source classification."""

    ASOS = "asos"            # Automated Surface Observing System (airport stations)
    FAWN = "fawn"            # Florida Automated Weather Network (agricultural)
    NDBC_BUOY = "ndbc_buoy"  # National Data Buoy Center (sea surface temp)
    METAR = "metar"          # Manual/automated aviation weather reports
    GOES_SAT = "goes_sat"    # GOES-19 satellite-derived products
    HRRR_MODEL = "hrrr"      # HRRR model analysis (not a true observation)
    WETHR = "wethr"          # Wethr SSE real-time observations


# ---------------------------------------------------------------------------
# Observation error models (R diagonal entries)
# ---------------------------------------------------------------------------
# These are the expected standard deviation of observation errors in °F.
# Smaller = more trusted. Calibrated from literature + Miami-specific experience.

OBS_ERROR_SIGMA_F: dict[ObsSourceType, float] = {
    ObsSourceType.ASOS: 0.5,       # Gold standard: calibrated, sheltered, maintained
    ObsSourceType.WETHR: 0.5,      # Wethr SSE uses NWS-verified ASOS data
    ObsSourceType.METAR: 0.7,      # METAR may have rounding, delay, or manual entry
    ObsSourceType.FAWN: 1.0,       # Agricultural siting: near crops, varying exposure
    ObsSourceType.NDBC_BUOY: 0.3,  # SST sensors are accurate, but measure water not air
    ObsSourceType.GOES_SAT: 2.0,   # Satellite-derived temp has large retrieval error
    ObsSourceType.HRRR_MODEL: 1.5, # Model analysis, not direct measurement
}


# ---------------------------------------------------------------------------
# Observation operator: source → SurfaceObs
# ---------------------------------------------------------------------------
@dataclass
class ObsOperatorConfig:
    """Configuration for observation operator."""

    # Maximum age (minutes) for an observation to be considered valid
    max_age_minutes: float = 30.0
    # Maximum distance (km) for an observation to influence a station
    max_distance_km: float = 200.0
    # Minimum observations required for LETKF update
    min_obs_for_update: int = 2
    # Apply buoy SST → air temp correction?
    buoy_sst_to_air_correction_f: float = 2.0  # SST is typically ~2°F warmer than air at night


def classify_obs_source(station_code: str) -> ObsSourceType:
    """Classify an observation's source type from its station code.

    Convention:
      - FAWN stations: code starts with "FAWN"
      - NDBC buoys: code starts with numbers (e.g., "41113")
      - GOES-19: code starts with "GOES"
      - HRRR: code starts with "HRRR"
      - Everything else: ASOS/METAR
    """
    code = station_code.upper().strip()
    if not code:
        return ObsSourceType.ASOS  # default for empty codes
    if code.startswith("FAWN"):
        return ObsSourceType.FAWN
    if code[0].isdigit():
        return ObsSourceType.NDBC_BUOY
    if code.startswith("GOES"):
        return ObsSourceType.GOES_SAT
    if code.startswith("HRRR"):
        return ObsSourceType.HRRR_MODEL
    # Default: ASOS
    return ObsSourceType.ASOS


def get_obs_sigma(station_code: str) -> float:
    """Get observation error std dev for a station based on its source type."""
    source_type = classify_obs_source(station_code)
    return OBS_ERROR_SIGMA_F[source_type]


def build_letkf_obs_from_db(
    db,
    station: str,
    timestamp_utc: str,
    config: ObsOperatorConfig | None = None,
) -> list[SurfaceObs]:
    """Build LETKF observation vector from all available DB sources.

    Queries:
      1. nearby_observations (IEM ASOS stations)
      2. fawn_observations (FAWN mesonet)
      3. observations (primary station — Wethr SSE)
      4. sst_observations (NDBC buoys, with SST→air correction)

    Returns SurfaceObs list suitable for LETKFState.update().
    """
    config = config or ObsOperatorConfig()
    from datetime import timedelta

    cutoff = timestamp_utc  # current time
    # Compute lookback
    # Parse timestamp
    from datetime import datetime, timezone
    try:
        now = datetime.fromisoformat(cutoff.replace("Z", "+00:00"))
    except ValueError:
        return []
    lb = (now - timedelta(minutes=config.max_age_minutes)).strftime("%Y-%m-%dT%H:%M:%SZ")

    obs_list: list[SurfaceObs] = []
    seen: set[str] = set()

    # 1. Nearby ASOS stations (IEM)
    try:
        rows = db.execute(
            """SELECT stid, latitude, longitude, air_temp_f
               FROM nearby_observations
               WHERE timestamp_utc BETWEEN ? AND ?
                 AND air_temp_f IS NOT NULL
               ORDER BY timestamp_utc DESC""",
            (lb, cutoff),
        ).fetchall()

        for row in rows:
            stid = row["stid"]
            if stid in seen:
                continue
            seen.add(stid)
            sigma = get_obs_sigma(stid)
            obs_list.append(SurfaceObs(
                station_code=stid,
                lat=row["latitude"],
                lon=row["longitude"],
                temp_f=row["air_temp_f"],
                obs_sigma_f=sigma,
                timestamp_utc=cutoff,
            ))
    except Exception:
        pass

    # 2. FAWN stations
    try:
        rows = db.execute(
            """SELECT station_id, station_name, air_temp_f
               FROM fawn_observations
               WHERE timestamp_utc BETWEEN ? AND ?
                 AND air_temp_f IS NOT NULL
               ORDER BY timestamp_utc DESC""",
            (lb, cutoff),
        ).fetchall()

        # FAWN stations need lat/lon — use our known stations
        from engine.letkf import SE_FLORIDA_CLUSTER
        fawn_coords = {
            s.code: (s.lat, s.lon)
            for s in SE_FLORIDA_CLUSTER.stations
            if s.code.startswith("FAWN")
        }

        for row in rows:
            sid = f"FAWN{row['station_id']}"
            if sid in seen:
                continue
            seen.add(sid)
            coords = fawn_coords.get(sid)
            if coords is None:
                continue
            obs_list.append(SurfaceObs(
                station_code=sid,
                lat=coords[0],
                lon=coords[1],
                temp_f=row["air_temp_f"],
                obs_sigma_f=get_obs_sigma(sid),
                timestamp_utc=cutoff,
            ))
    except Exception:
        pass

    # 3. Primary station observations (Wethr SSE)
    try:
        row = db.execute(
            """SELECT temperature_f
               FROM observations
               WHERE station = ? AND timestamp_utc BETWEEN ? AND ?
                 AND temperature_f IS NOT NULL
               ORDER BY timestamp_utc DESC LIMIT 1""",
            (station, lb, cutoff),
        ).fetchone()

        if row and station not in seen:
            seen.add(station)
            # Get station coordinates from config or cluster
            from engine.letkf import SE_FLORIDA_CLUSTER
            for s in SE_FLORIDA_CLUSTER.stations:
                if s.code == station:
                    obs_list.append(SurfaceObs(
                        station_code=station,
                        lat=s.lat,
                        lon=s.lon,
                        temp_f=row["temperature_f"],
                        obs_sigma_f=get_obs_sigma(station),
                        timestamp_utc=cutoff,
                    ))
                    break
    except Exception:
        pass

    # 4. NDBC buoy SST (with correction to approximate air temp)
    try:
        rows = db.execute(
            """SELECT station_id, latitude, longitude, sst_c
               FROM sst_observations
               WHERE timestamp_utc BETWEEN ? AND ?
                 AND sst_c IS NOT NULL
               ORDER BY timestamp_utc DESC""",
            (lb, cutoff),
        ).fetchall()

        for row in rows:
            sid = row["station_id"]
            if sid in seen:
                continue
            seen.add(sid)
            # Convert SST (Celsius) to air temp (Fahrenheit) with correction
            sst_f = row["sst_c"] * 9.0 / 5.0 + 32.0
            # SST is typically warmer than air over water; apply correction
            air_approx_f = sst_f - config.buoy_sst_to_air_correction_f
            obs_list.append(SurfaceObs(
                station_code=sid,
                lat=row["latitude"],
                lon=row["longitude"],
                temp_f=air_approx_f,
                obs_sigma_f=get_obs_sigma(sid),
                timestamp_utc=cutoff,
            ))
    except Exception:
        pass

    return obs_list
