#!/usr/bin/env python3
"""Audit weather data completeness for regime classification and inference."""

import sqlite3
from datetime import datetime, timedelta, timezone

db = sqlite3.connect("miami_collector.db")
db.row_factory = sqlite3.Row
now = datetime.now(timezone.utc)
cutoff_1h = (now - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
cutoff_6h = (now - timedelta(hours=6)).strftime("%Y-%m-%dT%H:%M:%SZ")

print("=== WEATHER DATA COMPLETENESS AUDIT ===")
print(f"Time: {now.strftime('%Y-%m-%d %H:%M UTC')}")
print()

# 1. Primary obs
print("--- 1. PRIMARY OBS ---")
row = db.execute(
    """SELECT temperature_f, dew_point_f, pressure_hpa, wind_heading_deg,
              wind_speed_mph, wind_gust_mph, timestamp_utc
       FROM observations WHERE station = 'KMIA'
       ORDER BY timestamp_utc DESC LIMIT 1"""
).fetchone()
if row:
    for k in row.keys():
        v = row[k]
        print(f"  {k:25s} = {v}  [{'OK' if v is not None else 'MISSING'}]")

# 2. Atmospheric data by model
print("\n--- 2. ATMOSPHERIC DATA ---")
rows = db.execute(
    """SELECT model, COUNT(*) as n, MAX(fetch_time_utc) as latest,
              AVG(cape) as avg_cape, AVG(precipitable_water_mm) as avg_pw,
              AVG(shortwave_radiation) as avg_swr
       FROM atmospheric_data WHERE station = 'KMIA'
       GROUP BY model ORDER BY latest DESC"""
).fetchall()
for r in rows:
    cape_s = f"{r['avg_cape']:.0f}" if r["avg_cape"] else "NULL"
    pw_s = f"{r['avg_pw']:.1f}" if r["avg_pw"] else "NULL"
    swr_s = f"{r['avg_swr']:.0f}" if r["avg_swr"] else "NULL"
    print(f"  {r['model']:20s} n={r['n']:5d}  CAPE={cape_s:>6s}  PW={pw_s:>6s}  SWR={swr_s:>6s}  latest={r['latest']}")

# 3. Cloud cover
print("\n--- 3. CLOUD COVER ---")
metar_count = db.execute(
    "SELECT COUNT(*) FROM nearby_observations WHERE sky_cover_code IS NOT NULL AND timestamp_utc > ?",
    (cutoff_1h,),
).fetchone()[0]
print(f"  METAR sky_cover (1h): {metar_count}")
fawn_solar = db.execute(
    """SELECT station_id, solar_radiation_wm2, timestamp_utc
       FROM fawn_observations WHERE solar_radiation_wm2 IS NOT NULL
       ORDER BY timestamp_utc DESC LIMIT 3"""
).fetchall()
for r in fawn_solar:
    print(f"  FAWN {r['station_id']} solar={r['solar_radiation_wm2']:.1f} W/m2  ts={r['timestamp_utc']}")

# 4. Nearby stations
print("\n--- 4. NEARBY STATIONS (6h) ---")
nearby = db.execute(
    """SELECT stid as station, network, COUNT(*) as n, MAX(timestamp_utc) as latest,
              AVG(distance_mi) as d
       FROM nearby_observations WHERE timestamp_utc > ?
       GROUP BY stid ORDER BY d ASC LIMIT 20""",
    (cutoff_6h,),
).fetchall()
for r in nearby:
    d = r["d"] or 0
    print(f"  {r['station']:10s} {(r['network'] or '?'):10s} n={r['n']:4d}  dist={d:.0f}mi  latest={r['latest']}")

# 5. FAWN
print("\n--- 5. FAWN STATIONS ---")
fawn = db.execute(
    """SELECT station_id, station_name, COUNT(*) as n, MAX(timestamp_utc) as latest
       FROM fawn_observations WHERE timestamp_utc > ?
       GROUP BY station_id ORDER BY n DESC""",
    (cutoff_6h,),
).fetchall()
for r in fawn:
    print(f"  {r['station_id']:6s} {(r['station_name'] or '?'):20s} n={r['n']:5d}  latest={r['latest']}")

# 6. Upper-air
print("\n--- 6. UPPER-AIR ---")
plev = db.execute(
    """SELECT temp_925_c, temp_850_c, temp_700_c, temp_500_c,
              wind_speed_850, wind_dir_850, rh_850, valid_time_utc
       FROM pressure_levels WHERE station = 'KMIA'
       ORDER BY valid_time_utc DESC LIMIT 1"""
).fetchone()
if plev:
    print(f"  925hPa: {plev['temp_925_c']}C  850hPa: {plev['temp_850_c']}C  700hPa: {plev['temp_700_c']}C  500hPa: {plev['temp_500_c']}C")
    print(f"  850mb wind: {plev['wind_speed_850']} m/s @ {plev['wind_dir_850']}deg  RH: {plev['rh_850']}%")
    print(f"  valid: {plev['valid_time_utc']}")

# 7. Model forecasts
print("\n--- 7. MODEL SOURCES (6h) ---")
fcst = db.execute(
    """SELECT source, model, COUNT(*) as n, MAX(fetch_time_utc) as latest
       FROM model_forecasts WHERE fetch_time_utc > ?
       GROUP BY source, model ORDER BY latest DESC LIMIT 25""",
    (cutoff_6h,),
).fetchall()
for r in fcst:
    print(f"  {(r['source'] or '?'):12s} {r['model']:22s} n={r['n']:4d}  latest={r['latest']}")

# 8. Regime requirements
print("\n=== REGIME REQUIREMENTS vs AVAILABILITY ===")
has_cape = any(r["avg_cape"] for r in rows if r["avg_cape"])
has_pw = any(r["avg_pw"] for r in rows if r["avg_pw"])
has_wind = row and row["wind_heading_deg"] is not None
has_cloud = metar_count > 0 or len(fawn_solar) > 0
has_dew = row and row["dew_point_f"] is not None
has_pressure = row and row["pressure_hpa"] is not None

reqs = {
    "MARINE_STABLE":       {"wind": has_wind, "CAPE": has_cape, "dew": has_dew},
    "INLAND_HEATING":      {"cloud": has_cloud, "CAPE": has_cape, "solar": len(fawn_solar) > 0},
    "CLOUD_SUPPRESSION":   {"cloud": has_cloud, "METAR_sky": metar_count > 0, "solar": len(fawn_solar) > 0},
    "CONVECTIVE_OUTFLOW":  {"CAPE": has_cape, "PW": has_pw, "radar": False},
    "FRONTAL_PASSAGE":     {"wind": has_wind, "BOCPD": True, "dew": has_dew, "pressure": has_pressure},
    "TRANSITION":          {"default": True},
}
for regime, checks in reqs.items():
    missing = [k for k, v in checks.items() if not v]
    print(f"  {regime:25s} {'COMPLETE' if not missing else 'MISSING: ' + ', '.join(missing)}")

# 9. Table sizes
print("\n--- TABLE SIZES ---")
for t in db.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall():
    n = db.execute(f"SELECT COUNT(*) FROM [{t['name']}]").fetchone()[0]
    if n > 0:
        print(f"  {t['name']:30s} {n:>8d}")

db.close()
