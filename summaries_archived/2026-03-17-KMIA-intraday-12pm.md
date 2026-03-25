# KMIA Mar 17, 2026 — Intraday Signal Analysis (12:00 PM ET / 17:00Z)

## Regime
**Post-frontal day.** Front passed KMIA at ~03:00Z (10 PM local, Mar 16). Now in continental air mass with NW flow. This is the coldest setup in our 3-day dataset.

## Current Conditions (11:10 AM local)
- **Temp: 64.4°F** — dropping. Was 66.2 all morning, now falling through 64.
- **Dew point: 59.0°F** — in freefall. Was 69.8 at midnight, 66.2 at dawn, 59.0 now. Still dropping.
- **Wind: NNW 340-350° at 7-13 mph** — continental intrusion confirmed. Sustained since 12:30 PM local.
- **Pressure: 1017.6-1017.9 hPa** — rising steadily (was 1014.2 at midnight). Post-frontal high building.
- **CAPE: ~0 J/kg** — dead atmosphere. No convection risk.
- **PW: 37.1 mm** — moderate but drying.
- **SST: 76.6°F** — maritime buffer present but NW wind means air is continental, not maritime.
- **Sky: partly cloudy** — clearing trend.

## Running Extremes
- **Running high: 71.6°F** at 12:50 AM (05:50Z) — this was leftover pre-frontal warmth.
- **Running low: 64.4°F** at 11:05 AM (16:05Z) — still dropping as of this writing.

## DSM Updates
- DSM received at 07:22Z: high=71°F, low=68°F (valid through 07:00Z only — predates NW wind arrival at KMIA surface).

## Nearby Stations (all confirming cold regime)
| Station | Temp | Dew | Wind Dir | Wind Spd |
|---|---|---|---|---|
| KOPF (8mi) | 65.0°F | 58.0°F | 350° | 13.8 mph |
| KTMB (14mi) | 64.0°F | 60.0°F | 350° | 13.8 mph |
| KHWO (14mi) | 68.0°F | 58.0°F | 360° | 13.8 mph |
| KFLL (21mi) | 66.0°F | 60.0°F | 360° | 9.2 mph |
| KHST (22mi) | 67.3°F | 59.4°F | 320° | 15.0 mph |
| KFXE (24mi) | 65.0°F | 60.0°F | 330° | 6.9 mph |
| FAWN Homestead | 63.7°F | 60.6°F | — | 12.6 mph |

All stations NW/N wind. All dew points 58-60°F. Regional cold regime confirmed.
FAWN Homestead coldest at 63.7°F — inland cooling already exceeding coastal stations.

## FAWN Detail
- Temp trajectory: 68.0 (midnight) → 64.5 (dawn) → 66.4 (9AM) → 62.2 (10:45AM) — dropping hard
- Dew point: 66.8 → 63.8 → 65.9 → 60.6 — crashing in sync
- Rain: 1.13 inches total today (all pre-dawn, 05:45-08:00Z). Now dry.
- Gust: 21.6 mph at 11:00 AM — gusty NW flow

## Model Consensus

### HIGH
| Model | Forecast |
|---|---|
| NAM | 79.0°F ← likely stale run, ignore |
| NAM4KM | 76.3°F ← likely stale |
| HRRR | 75.5°F ← stale |
| ARPEGE | 75.0°F |
| ECMWF-IFS | ~71°F (ensemble μ=71.1, σ=1.2) |
| GFS-Global | 71.0°F |
| GFS-HRRR | 71.0°F |
| GFS-MOS | 72.0°F |
| GEM-Global | 69.6°F |
| RAP | 71.8°F |

**Reality check**: Running high already 71.6°F. Current temp 64.4°F and falling with NW wind. No model shows warming above 72 in fresh runs. The high is locked at 71-72°F. NAM/HRRR high forecasts of 75-79°F are from pre-frontal runs and are garbage.

### LOW (tonight)
| Model | Forecast |
|---|---|
| RAP | 48.8°F |
| HRRR | 51.6°F |
| GFS-Global | 53.9°F |
| ECMWF-IFS ens median | ~58°F (σ=1.7, range 54-62) |
| GEM-Global | 53.0°F |
| NAM4KM | 55.8°F |
| ARPEGE | 54.7°F |
| GEM-GDPS | 55.0°F (wethr) / 57.4°F (openmeteo) |

**Consensus**: 54-58°F range. Cold models (RAP, HRRR, GFS) cluster at 49-54. Warm models (ECMWF ens) at 56-60. Bias-corrected consensus = 56.3°F ± 4.4°F.

## Market Pricing vs Signal Assessment

### HIGH Market
| Bracket | Market Mid | Signal Assessment |
|---|---|---|
| T73 (under 73) | 94c | **Correctly priced.** High is locked at 71.6. No mechanism to exceed 73. ~97% probability. |
| B73.5 (73-73.5) | 4c | Correctly priced. |
| B75.5+ | 1c | Dead. |

**No edge on HIGH.** Market has it right. Running high is set and won't be exceeded.

### LOW Market
| Bracket | Market Mid | Signal Assessment | Our Est. Prob |
|---|---|---|---|
| T57 (under 57) | 19c | **UNDERPRICED.** NW wind locked, dew crashing, CAPE=0, clear sky tonight = max radiative cooling. RAP/HRRR/GFS all below 54. | 45-50% |
| B57.5 (57-57.5) | 22c | Fairly priced. | 15-20% |
| B59.5 (57.5-59.5) | 26c | **OVERPRICED.** Market's most-likely bracket but our signals point colder. | 15-18% |
| B61.5 (59.5-61.5) | 19c | Overpriced given dew trajectory. | 10-12% |
| B63.5 (61.5-63.5) | 4c | Correctly priced. | 3-5% |
| T64 (above 64) | 4c | Correctly priced — running low already 64.4 and dropping. | 2-3% |

### Recommended Trade
**YES on T57 (under 57°F) at 22c ask.**

Signal basis (not AI reasoning — deterministic checks):
1. ✅ Wind NW 340-350° sustained → CONTINENTAL_INTRUSION = True
2. ✅ Dew point 59°F and falling → tonight's dew likely 53-56°F → floor = 55-58°F
3. ✅ CAPE = 0 → no convection, pure radiative cooling
4. ✅ Pressure 1017.9 rising → post-frontal high = clear skies tonight
5. ✅ RAP (low bias: -3.1°F over 3 days, so actual ~52°F) + HRRR (51.6) + GFS (53.9) all under 54
6. ✅ FAWN already at 62.2°F inland and dropping — inland will crater tonight
7. ⚠️ Risk: SST 76.6°F maritime buffer if wind calms or backs SE overnight. Models don't show this.

**Expected value**: 45% × $1.00 - 55% × $0.22 = $0.45 - $0.12 = +$0.33 per dollar risked. Favorable.

## Signals Active
- **OBS_RUNNING_COLD**: -3.1°F below NBM hourly forecast
- **CONTINENTAL_INTRUSION**: NW wind sustained for 4+ hours
- **DEW_POINT_CRASH**: 69.8 → 59.0°F over 11 hours (10.8°F total drop)
- **PRESSURE_RISING**: +3.7 hPa since midnight (post-frontal high)
- **FAWN_LEAD**: Homestead 1.7°F colder than KMIA, tracking ahead
- **ENSEMBLE_DIVERGE**: ECMWF 71.1 vs GFS 78.2 on HIGH (GFS stale — see data quality note)

## Data Quality Issue Detected
**GFS Ensemble HIGH consensus showing 78.2°F is STALE.** The model_scorer is averaging all historical GFS ensemble runs (240 rows across 8 runs spanning 38 hours) instead of using only the latest run's 30 members. This caused the signal engine to flag B75.5 as "underpriced" with 15% model probability — a false signal. The actual latest GFS ensemble members forecast 71°F for the high, consistent with all other fresh models. Bug logged for fix in model_scorer.py `_get_latest_forecasts()`.

## Comparison to Mar 15-16 at Same Hour
| Signal | Mar 15 @ noon | Mar 16 @ noon | Mar 17 @ noon |
|---|---|---|---|
| Regime | Tropical moisture | Tropical moisture | Post-frontal continental |
| Wind | SE 113° | SSE 158° | NNW 340° |
| Dew point | 71.6°F | 73.4°F | 59.0°F |
| CAPE | 3560 J/kg | 3600 J/kg | 0 J/kg |
| Pressure | 1015.2 hPa | 1013.9 hPa | 1017.9 hPa |
| Running high | 81.0°F | 82.4°F | 71.6°F |
| Running low | 75.2°F | 75.2°F | 64.4°F |
| Volatility risk | Extreme (outflow) | Extreme (frontal) | Low (stable) |
| Best bet type | NO dead brackets | NO dead brackets | YES on cold low |

Key insight: Mar 15-16 were "fade the extremes" days (bet NO on brackets that were already dead). Mar 17 is a "direction bet" day (bet YES on a specific cold outcome). Different regime = different strategy.
