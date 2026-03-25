"""Cloud cover estimation from METAR sky codes and FAWN solar radiation.

Provides cloud fraction (0.0 = clear, 1.0 = overcast) from two sources:
  1. METAR sky cover codes (SKC/CLR/FEW/SCT/BKN/OVC) — from nearby_observations
  2. FAWN solar radiation clearness index — ratio of measured to expected clear-sky

Cloud cover constrains daytime high temperature: more clouds = less solar
heating = lower highs. This is one of the highest-value signals for bracket
pricing during heating hours.

Usage:
    cloud = estimate_cloud_fraction(db, station, now_utc, lat, lon)
    # Returns CloudEstimate with fraction, source, and confidence
"""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# METAR sky cover code mapping (standard NWS)
# ---------------------------------------------------------------------------
SKY_COVER_FRACTION: dict[str, float] = {
    "SKC": 0.0,     # Sky clear (manual report)
    "CLR": 0.0,     # Clear (automated, no clouds below 12,000ft)
    "FEW": 0.125,   # Few (1/8 to 2/8 coverage)
    "SCT": 0.375,   # Scattered (3/8 to 4/8)
    "BKN": 0.75,    # Broken (5/8 to 7/8)
    "OVC": 1.0,     # Overcast (8/8)
    "VV": 1.0,      # Vertical visibility (obscured sky, treat as overcast)
    "OVX": 1.0,     # Obscured sky
}


def metar_sky_to_fraction(code: str | None) -> float | None:
    """Convert METAR sky cover code to numeric fraction [0, 1]."""
    if code is None:
        return None
    return SKY_COVER_FRACTION.get(code.upper().strip())


# ---------------------------------------------------------------------------
# Solar clearness index (FAWN)
# ---------------------------------------------------------------------------
def expected_clear_sky_radiation(
    hour_utc: float,
    day_of_year: int,
    lat_deg: float = 25.7959,
) -> float:
    """Approximate clear-sky solar radiation at surface (W/m²).

    Simple model: max_solar * cos(zenith_angle)
    where zenith depends on latitude, day of year, and hour.
    """
    # Solar declination (approximate)
    decl = 23.45 * math.sin(math.radians(360.0 / 365.0 * (day_of_year - 81)))
    decl_rad = math.radians(decl)
    lat_rad = math.radians(lat_deg)

    # Hour angle (solar noon = 0)
    # Approximate solar noon at 17:00 UTC for lon=-80 (12:00 LST)
    solar_noon_utc = 17.0  # rough for KMIA longitude
    hour_angle = 15.0 * (hour_utc - solar_noon_utc)  # degrees
    hour_angle_rad = math.radians(hour_angle)

    # Cosine of zenith angle
    cos_zenith = (
        math.sin(lat_rad) * math.sin(decl_rad)
        + math.cos(lat_rad) * math.cos(decl_rad) * math.cos(hour_angle_rad)
    )

    if cos_zenith <= 0:
        return 0.0  # sun below horizon

    # Top-of-atmosphere solar constant * atmospheric transmittance * cos(zenith)
    # 1361 W/m² * ~0.75 clear-sky transmittance
    return 1361.0 * 0.75 * cos_zenith


def fawn_clearness_index(
    measured_solar_wm2: float,
    hour_utc: float,
    day_of_year: int,
    lat_deg: float = 25.7959,
) -> float | None:
    """Compute clearness index: measured / expected clear-sky radiation.

    Returns value in [0, 1] where:
      ~1.0 = clear sky
      ~0.2 = heavy overcast
      None = sun below horizon or invalid
    """
    expected = expected_clear_sky_radiation(hour_utc, day_of_year, lat_deg)
    if expected < 50.0:  # sun too low for meaningful comparison
        return None
    ratio = max(0.0, min(1.2, measured_solar_wm2 / expected))
    return round(ratio, 3)


def clearness_to_cloud_fraction(clearness: float | None) -> float | None:
    """Convert clearness index to approximate cloud fraction.

    Mapping: clear sky (clearness ~1.0) → cloud_fraction ~0.0
             overcast (clearness ~0.2) → cloud_fraction ~1.0
    """
    if clearness is None:
        return None
    # Linear mapping: cloud = 1 - clearness, clamped to [0, 1]
    return round(max(0.0, min(1.0, 1.0 - clearness)), 3)


# ---------------------------------------------------------------------------
# Combined cloud estimate
# ---------------------------------------------------------------------------
@dataclass
class CloudEstimate:
    """Combined cloud cover estimate."""

    fraction: float  # 0.0 = clear, 1.0 = overcast
    source: str  # "metar", "fawn_solar", "blended"
    confidence: float  # 0.0 to 1.0
    n_metar_stations: int = 0
    fawn_clearness: float | None = None
    metar_mean: float | None = None


def estimate_cloud_fraction(
    db: sqlite3.Connection,
    station: str,
    now_utc: datetime,
    lat: float = 25.7959,
    lon: float = -80.287,
    lookback_minutes: int = 60,
) -> CloudEstimate | None:
    """Estimate current cloud fraction from METAR obs + FAWN solar.

    Priority:
      1. METAR sky cover codes from nearby stations (spatial average)
      2. FAWN solar radiation clearness index
      3. Blend if both available

    Args:
        db: SQLite connection with row_factory=sqlite3.Row
        station: Trading station (e.g., "KMIA")
        now_utc: Current UTC time
        lat, lon: Station coordinates
        lookback_minutes: How far back to look for observations

    Returns:
        CloudEstimate or None if no data available.
    """
    cutoff = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    lookback_ts = (
        now_utc.replace(second=0, microsecond=0)
    )
    # Approximate lookback
    from datetime import timedelta
    lb = (now_utc - timedelta(minutes=lookback_minutes)).strftime("%Y-%m-%dT%H:%M:%SZ")

    # 1. METAR sky cover from nearby_observations
    metar_rows = db.execute(
        """SELECT sky_cover_code, distance_mi
           FROM nearby_observations
           WHERE timestamp_utc BETWEEN ? AND ?
             AND sky_cover_code IS NOT NULL
           ORDER BY timestamp_utc DESC
           LIMIT 30""",
        (lb, cutoff),
    ).fetchall()

    metar_fractions: list[tuple[float, float]] = []  # (fraction, distance_mi)
    for row in metar_rows:
        frac = metar_sky_to_fraction(row["sky_cover_code"])
        if frac is not None:
            dist = max(0.5, float(row["distance_mi"] or 1.0))  # floor at 0.5mi
            metar_fractions.append((frac, dist))

    metar_mean: float | None = None
    if metar_fractions:
        # Inverse-distance weighting: closer stations have more influence
        weights = [1.0 / d for _, d in metar_fractions]
        total_w = sum(weights)
        if total_w > 0:
            metar_mean = sum(f * w for (f, _), w in zip(metar_fractions, weights)) / total_w
        else:
            metar_mean = sum(f for f, _ in metar_fractions) / len(metar_fractions)

    # 2. FAWN solar radiation clearness index
    hour_utc = now_utc.hour + now_utc.minute / 60.0
    doy = now_utc.timetuple().tm_yday

    fawn_rows = db.execute(
        """SELECT solar_radiation_wm2
           FROM fawn_observations
           WHERE timestamp_utc BETWEEN ? AND ?
             AND solar_radiation_wm2 IS NOT NULL
             AND solar_radiation_wm2 > 0
           ORDER BY timestamp_utc DESC
           LIMIT 5""",
        (lb, cutoff),
    ).fetchall()

    fawn_clearness: float | None = None
    fawn_cloud: float | None = None
    if fawn_rows:
        avg_solar = sum(r["solar_radiation_wm2"] for r in fawn_rows) / len(fawn_rows)
        fawn_clearness = fawn_clearness_index(avg_solar, hour_utc, doy, lat)
        fawn_cloud = clearness_to_cloud_fraction(fawn_clearness)

    # 3. Combine
    if metar_mean is not None and fawn_cloud is not None:
        # Blend: weight METAR slightly higher (more spatial coverage)
        fraction = 0.6 * metar_mean + 0.4 * fawn_cloud
        return CloudEstimate(
            fraction=round(fraction, 3),
            source="blended",
            confidence=0.8,
            n_metar_stations=len(metar_fractions),
            fawn_clearness=fawn_clearness,
            metar_mean=round(metar_mean, 3),
        )
    elif metar_mean is not None:
        return CloudEstimate(
            fraction=round(metar_mean, 3),
            source="metar",
            confidence=0.6,
            n_metar_stations=len(metar_fractions),
            metar_mean=round(metar_mean, 3),
        )
    elif fawn_cloud is not None:
        return CloudEstimate(
            fraction=round(fawn_cloud, 3),
            source="fawn_solar",
            confidence=0.5,
            fawn_clearness=fawn_clearness,
        )

    return None
