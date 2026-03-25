#!/usr/bin/env python3
"""End-to-end data flow test: trace weather data through the full pipeline.

Simulates following a piece of weather data from raw observation through
regime classification, probability estimation, and trade decision.
"""

import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from engine.orchestrator import InferenceConfig, run_inference_cycle

DB_PATH = "miami_collector.db"
STATION = "KMIA"

db = sqlite3.connect(DB_PATH)
db.row_factory = sqlite3.Row

print("=" * 70)
print("END-TO-END DATA FLOW TEST")
print("=" * 70)

# 1. Current raw obs
print("\n--- STEP 1: Raw observation data ---")
obs = db.execute(
    """SELECT temperature_f, dew_point_f, pressure_hpa, wind_heading_deg,
              wind_speed_mph, timestamp_utc
       FROM observations WHERE station = ? ORDER BY timestamp_utc DESC LIMIT 1""",
    (STATION,),
).fetchone()
if obs:
    print(f"  Temp: {obs['temperature_f']}°F")
    print(f"  Dew:  {obs['dew_point_f']}°F (may be NULL — filled from nearby)")
    print(f"  Pres: {obs['pressure_hpa']} hPa (may be NULL — filled from nearby)")
    print(f"  Wind: {obs['wind_heading_deg']}° at {obs['wind_speed_mph']} mph")
    print(f"  Time: {obs['timestamp_utc']}")

# 2. Nearby station backup (for dew/pressure)
print("\n--- STEP 2: Nearby station backup for missing fields ---")
nearby = db.execute(
    """SELECT stid, dew_point_f, pressure_slp_hpa, air_temp_f, distance_mi, timestamp_utc
       FROM nearby_observations
       WHERE dew_point_f IS NOT NULL AND timestamp_utc > datetime('now', '-30 minutes')
       ORDER BY distance_mi ASC LIMIT 3"""
).fetchall()
for r in nearby:
    print(f"  {r['stid']:8s} {r['distance_mi']:.0f}mi  dew={r['dew_point_f']}°F  pres={r['pressure_slp_hpa']}hPa  temp={r['air_temp_f']}°F  ts={r['timestamp_utc']}")

# 3. Atmospheric data (HRRR + GOES + Open-Meteo)
print("\n--- STEP 3: Atmospheric data (CAPE, PW, cloud) ---")
for model in ["HRRR", "GOES-19-SAT", "best_match"]:
    row = db.execute(
        """SELECT cape, precipitable_water_mm, shortwave_radiation, fetch_time_utc
           FROM atmospheric_data
           WHERE station = ? AND model = ? ORDER BY fetch_time_utc DESC LIMIT 1""",
        (STATION, model),
    ).fetchone()
    if row:
        print(f"  {model:20s} CAPE={row['cape']}  PW={row['precipitable_water_mm']}  SWR={row['shortwave_radiation']}  ts={row['fetch_time_utc']}")

# 4. FAWN station data
print("\n--- STEP 4: FAWN spatial network ---")
fawn = db.execute(
    """SELECT station_id, station_name, air_temp_f, solar_radiation_wm2, timestamp_utc
       FROM fawn_observations
       ORDER BY timestamp_utc DESC LIMIT 3"""
).fetchall()
for r in fawn:
    print(f"  FAWN {r['station_id']} ({r['station_name']}): {r['air_temp_f']}°F  solar={r['solar_radiation_wm2']} W/m2  ts={r['timestamp_utc']}")

# 5. Model forecasts (top 5 freshest)
print("\n--- STEP 5: Model forecast sources (top 5 freshest) ---")
fcst = db.execute(
    """SELECT source, model, forecast_high_f, forecast_low_f, run_time, fetch_time_utc
       FROM model_forecasts
       WHERE station = ? AND forecast_date = date('now')
       ORDER BY fetch_time_utc DESC LIMIT 5""",
    (STATION,),
).fetchall()
for r in fcst:
    print(f"  {r['source']:12s} {r['model']:20s} H={r['forecast_high_f']}  L={r['forecast_low_f']}  run={r['run_time']}  fetched={r['fetch_time_utc']}")

# 6. Run actual inference cycle
print("\n--- STEP 6: Running inference cycle ---")
db.close()

result = run_inference_cycle(DB_PATH, station=STATION)
print(f"  Target date: {result.target_date}")
print(f"  Timestamp:   {result.timestamp_utc}")
print(f"  # Estimates: {len(result.estimates)}")

# Parse notes
regime_notes = [n for n in result.notes if "regime:" in n or "regime=" in n]
emos_notes = [n for n in result.notes if "emos" in n]
boa_notes = [n for n in result.notes if "boa" in n]
letkf_notes = [n for n in result.notes if "letkf" in n]
skf_notes = [n for n in result.notes if "skf_dormant" in n]

print(f"\n  Regime: {regime_notes}")
print(f"  EMOS:   {emos_notes}")
print(f"  BOA:    {boa_notes}")
print(f"  LETKF:  {letkf_notes}")
print(f"  SKF:    {skf_notes[:1]}{'...' if len(skf_notes) > 1 else ''}")

# 7. High/low beliefs
print("\n--- STEP 7: Adjusted beliefs ---")
if result.high_belief:
    hb = result.high_belief
    print(f"  HIGH: mu={hb.distribution.mu:.1f}°F  sigma={hb.distribution.sigma:.2f}°F")
if result.low_belief:
    lb = result.low_belief
    print(f"  LOW:  mu={lb.distribution.mu:.1f}°F  sigma={lb.distribution.sigma:.2f}°F")

# 8. Bracket estimates with edge
print("\n--- STEP 8: Bracket estimates (with edge) ---")
for est in sorted(result.estimates, key=lambda e: (e.market_type, e.floor_f or 0)):
    edge_str = f"edge={est.edge:+.1f}¢" if est.edge is not None else "edge=N/A"
    mkt_str = f"mkt={est.market_price}¢" if est.market_price is not None else "mkt=N/A"
    print(f"  {est.ticker:30s} P={est.model_probability:.1%}  {mkt_str:12s}  {edge_str}")

# 9. Trade opportunities (positive edge)
print("\n--- STEP 9: Trade opportunities (positive edge after fees) ---")
opportunities = [e for e in result.estimates if e.edge is not None and e.edge > 0]
for opp in sorted(opportunities, key=lambda e: e.edge, reverse=True):
    side = "YES" if opp.edge > 0 else "NO"
    print(f"  {side} {opp.ticker:30s} edge={opp.edge:+.1f}¢  P_model={opp.model_probability:.1%}  P_mkt={opp.market_price}¢")

no_opps = [e for e in result.estimates if e.no_edge is not None and e.no_edge > 0]
for opp in sorted(no_opps, key=lambda e: e.no_edge, reverse=True):
    print(f"  NO  {opp.ticker:30s} edge={opp.no_edge:+.1f}¢  P_no={opp.no_probability:.1%}  mkt_no={opp.no_market_price}¢")

if not opportunities and not no_opps:
    print("  (no positive-edge opportunities found)")

# 10. All inference notes
print("\n--- STEP 10: Full inference notes ---")
for note in result.notes:
    print(f"  {note}")

print("\n" + "=" * 70)
print("END-TO-END TEST COMPLETE")
print("=" * 70)
