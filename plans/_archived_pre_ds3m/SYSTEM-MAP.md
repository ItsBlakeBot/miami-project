# Miami Project — Complete System Map

**Station:** KMIA (Miami International Airport)
**Purpose:** Trade Kalshi temperature bracket markets using real-time weather data, model consensus, and regime-aware probability estimation.
**Last updated:** 2026-03-19

---

## How to Read This Document

This is the canonical reference for the entire system. Every Python module is listed. Every data path is traced. If you're an AI picking up this project, start here. If you're a human debugging a pipeline break, ctrl-F the module name.

---

## System Overview

```
╔═════════════════════════════════════════════════════════════════════════╗
║                        COLLECTOR LAYER                                  ║
║  14 async data sources → SQLite (WAL mode, ~385MB)                     ║
║  Sources: Wethr SSE/REST, Open-Meteo, NWS, Kalshi WS/REST,           ║
║           FAWN, NDBC buoys, IEM nearby stations                        ║
╚══════════════════════════════╤══════════════════════════════════════════╝
                               │ reads from DB every ~5 min
╔══════════════════════════════▼══════════════════════════════════════════╗
║                     ENGINE LAYER (intraday, every cycle)                ║
║                                                                         ║
║  signal_engine ──→ baseline_engine ──→ short_range + station_transfer  ║
║       │                                        │                        ║
║       ▼                                        ▼                        ║
║  ┌─────────────────┐    ┌──────────────────────────────────┐           ║
║  │ CHANGE DETECTOR │    │ REGIME CLASSIFIER                │           ║
║  │ (CUSUM+threshold│───→│ (deterministic rules + SKF)      │           ║
║  │  industrial)    │    │                                  │           ║
║  └─────────────────┘    │  SKF: 5D Kalman, per-regime      │           ║
║   Layer 1: SPECI-style  │  models, forward trajectory,     │           ║
║     temp/dew/pres/wind   │  Bayesian probability update     │           ║
║     gust/sky thresholds  │                                  │           ║
║   Layer 2: Per-channel   │  notify_changepoint() ←──────── │           ║
║     CUSUM on detrended   │  reweights priors on change     │           ║
║     Fourier residuals    └──────────────┬───────────────────┘           ║
║                                         │                               ║
║                                         ▼                               ║
║                          signal_families (anti-double-counted)          ║
║                                         │                               ║
║                                         ▼                               ║
║                          residual_estimator (mu/sigma adjustments)      ║
║                           ├── family-gated adjustments                  ║
║                           └── SKF-gated adjustments (trust-ramped)     ║
║                                         │                               ║
║                                         ▼                               ║
║                    bracket_pricer → edge_detector → policy              ║
╚══════════════════════════════╤══════════════════════════════════════════╝
                               │ probability estimates
╔══════════════════════════════▼══════════════════════════════════════════╗
║                        TRADER (separate repo)                           ║
║  Reads bracket probabilities → executes on Kalshi                      ║
║  City-agnostic — will serve multiple city bots from one account        ║
╚═════════════════════════════════════════════════════════════════════════╝


═══════════════════════ POST-SETTLEMENT (daily) ═══════════════════════

  ~04:45 LST   CLI settlement arrives (collector ingests)
       │
       ▼
  09:00 LST ──┬── post_settlement_calibrator ──→ cusum_config.json
              │    (CUSUM ARL + SKF EMA)          skf_config.json
              │                                    calibration_log.jsonl
              │
              └── hdp_regime_discovery ──────────→ regime_labels_hdp_test
                  (shadow mode, silent)             (compare vs AI labels)
       │
       ▼
  09:30 LST ──── daily_data_builder ──→ prompt_builder ──→ save_prompt
                 (OHLC + all sources)   (+ calibration      │
                  ~55-74K tokens)        context)            │
                                                             ▼
                                                    [OpenClaw → Claude]
                                                             │
                                                             ▼
                                                    ai_review_parser
                                                     ├─→ regime_labels table
                                                     ├─→ reviews/ai/{date}.md
                                                     └─→ threshold_history.jsonl
                                                             │
                                                             ▼
                                                    train_skf (batch OLS refit)
                                                             │
                                                             ▼
                                                    skf_config.json
                                                    (loaded by SKF at next startup)


══════════════════ SELF-LEARNING FEEDBACK LOOPS ══════════════════════

  ┌──────────────────────────────────────────────────────────────┐
  │  STATISTICAL (automatic, daily at 09:00)                     │
  │                                                              │
  │  Settlement error → EMA updates:                             │
  │    CUSUM h: ↓ on misses, ↑ on false alarms (±0.3/day)      │
  │    SKF mu_shift: toward observed error (±0.3°F/day)         │
  │    SKF sigma_scale: widen/tighten vs CI (±0.05/day)         │
  │                                                              │
  │  Conservative: α=0.15, clamped, single outliers don't wreck │
  └──────────────────────────────┬───────────────────────────────┘
                                 │ values visible in prompt
  ┌──────────────────────────────▼───────────────────────────────┐
  │  AI REVIEW (daily at 09:30, sanity check)                    │
  │                                                              │
  │  Sees auto-tuned CUSUM + SKF parameters                     │
  │  Sees calibration drift history (last 3 days)               │
  │  Can override: "mu_shift_low drifting too negative,         │
  │                  March 16 was an outlier — reset to -1.2"   │
  │  AI recommendations take precedence on conflict             │
  └──────────────────────────────┬───────────────────────────────┘
                                 │ labels feed batch retraining
  ┌──────────────────────────────▼───────────────────────────────┐
  │  SKF BATCH REFIT (daily after AI review)                     │
  │                                                              │
  │  All regime_labels → OLS fit per regime → skf_config.json   │
  │  Overrides incremental EMA with full batch fit               │
  │  More principled: considers all labeled days simultaneously  │
  └──────────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────────────┐
  │  HDP SHADOW (daily at 09:00, silent comparison)              │
  │                                                              │
  │  Discovers regime count from obs alone (no AI needed)        │
  │  Writes to regime_labels_hdp_test                           │
  │  After 30+ days: compare vs AI labels                       │
  │  Goal: eventual API independence                            │
  └──────────────────────────────────────────────────────────────┘
```

---

## Invariants (Never Violated)

1. **LST only** — climate day is midnight-midnight EST (UTC-5 fixed year-round, no DST). UTC window: 05:00Z to 05:00Z.
2. **CLI is truth** — NWS Climate Report is the only settlement source for Kalshi markets. Obs extremes are used for path/timing analysis, not settlement.
3. **HIGH and LOW are independent markets** — scored, modeled, and traded separately at every layer.
4. **No double-counting** — baseline handles model consensus; residuals handle what models miss. Signals never re-inject information the models already captured.
5. **Freshness-aware** — 1 vote per model, weighted by accuracy + recency decay.
6. **Reviews are annotations** — they inform training data and threshold tuning, never used as live features.

---

## Collector Layer — Data Ingestion

### Entry Point
- **`collector/entrypoints.py`** — CLI entry (`miami-collector`). Wraps `runner.py` with maintenance hooks (scoring, pruning).

### Orchestrator
- **`collector/runner.py`** — Main async orchestrator. Runs 14 concurrent collection loops. Each source has its own polling interval and error handling.

### Configuration
- **`collector/config.py`** — Loads from TOML + `.env` overlay. API keys, station config, polling intervals.
- **`collector/types.py`** — Shared data types: `ModelForecast`, `Observation`, `MarketSnapshot`, etc.

### Data Sources (14 async loops)

| Module | Source | Data | Frequency |
|--------|--------|------|-----------|
| `sources/wethr_stream.py` | Wethr SSE | 1-min obs, new_high/low events, CLI settlement, DSM | Real-time stream |
| `sources/wethr.py` | Wethr REST | NWP forecasts (HRRR, NAM, GFS, RAP, etc.) | Every ~15 min |
| `sources/openmeteo.py` | Open-Meteo | 11 deterministic + 82 ensemble models, pressure levels, atmospheric | Every ~15 min |
| `sources/nws.py` | NWS | ASOS observations, CLI settlement reports | Every ~5 min |
| `sources/kalshi_ws.py` | Kalshi WS | Orderbook deltas, trades, fills | Real-time stream |
| `sources/kalshi_rest.py` | Kalshi REST | Market discovery, settlement checks, bracket definitions | Every ~5 min |
| `sources/fawn.py` | FAWN | Homestead ag station (air temp, dew, wind, solar, soil, rain) | Every ~15 min |
| `sources/ndbc.py` | NDBC | Offshore buoy data (SST, air temp, wind, pressure) | Every ~30 min |
| `sources/iem.py` | IEM | Nearby ASOS stations (60+ stations, temp delta vs KMIA) | Every ~5 min |

**Retired:** `sources/synoptic.py` — permanently down, replaced by IEM.

### Storage
- **`store/schema.py`** — Schema v11. 20+ tables. Key tables: `observations`, `model_forecasts`, `model_consensus`, `forward_curves`, `event_settlements`, `atmospheric_data`, `pressure_levels`, `fawn_observations`, `nearby_observations`, `sst_observations`, `market_snapshots`, `regime_labels`.
- **`store/db.py`** — SQLite wrapper with WAL mode, connection pooling, dedupe helpers, migration logic. Methods for every table.

### Maintenance
- **`collector/prune.py`** — Database retention and pruning (`miami-prune-db`). Removes stale ensemble rows, old market snapshots.
- **`collector/forward_curve.py`** — Extract multi-model hourly forecasts from DB for forward curve snapshots.

---

## Engine Layer — Inference Pipeline

Data flows top-to-bottom through these modules every cycle. Each module reads from the one above and passes structured output to the one below.

### Stage 1: Raw Signal Extraction
```
collector DB
    ↓
signal_engine.py → SignalState (7 signals + active_flags + model_tracking)
```

- **`engine/signal_engine.py`** — The entry point for each inference cycle. Queries the collector DB and extracts a `SignalState` with 7 raw signals:
  1. **Model consensus** — consensus_f, consensus_sigma, n_models
  2. **Obs divergence** — obs_current_f, obs_trend_2hr, obs_vs_consensus, projected_extreme_f
  3. **CAPE + PW outflow** — cape_current, pw_mm, outflow_risk, cape_trend_1hr
  4. **Wind direction** — wind_dir_deg, continental_intrusion, wind_shift_detected
  5. **Dew point floor** — dew_point_f, evening_dew_mean_f, estimated_low_floor_f, dew_crash_active
  6. **Nearby station lead** — fawn_temp_f, fawn_crash_detected, nearby_divergence_f
  7. **Pressure** — pressure_hpa, pressure_3hr_trend, pressure_surge
  - Plus: running_high_f, running_low_f, forward_max/min, model_tracking profiles, active_flags list

### Stage 2: Baseline Belief
```
SignalState + source_registry
    ↓
baseline_engine.py → BaselineBelief (quantile distribution + model_trust)
```

- **`engine/source_registry.py`** — Provider-agnostic interface for forecast sources. `ForecastSourceSnapshot` dataclass. Groups by family (openmeteo, wethr, nws). Handles deduplication to latest per source.
- **`engine/baseline_engine.py`** — Model-first probabilistic belief. For each source snapshot: apply slow bias correction (from bias_memory), compute freshness weight (exp decay over 6hr), compute tracking weight (exp decay from MAE^2). Combine quantiles across weighted sources. Output: `BaselineBelief` with 7-quantile distribution + `model_trust` score (0-1).
- **`engine/bias_memory.py`** — Slow structural bias correction. JSON-backed store at `analysis_data/bias_memory.json`. EMA with alpha=0.15. Keyed by `station|family|market_type|checkpoint_bucket`.

### Stage 3: Feature Extraction
```
BaselineBelief + latest obs
    ↓
short_range_updater.py → ShortRangeFeatures
station_transfer.py → StationTransferFeatures
```

- **`engine/short_range_updater.py`** — Extracts compact short-range features from local obs: `temp_trend_1h/3h`, `dew_trend_1h`, `pressure_trend_3h`, `wind_turn_3h`, plus derived scores: `solar_support_score`, `persistence_score`, `advection_score`, `no_daytime_recovery_score`.
- **`engine/station_transfer.py`** — Translates nearby station observations into KMIA-relevant hints: `inland_minus_kmia_temp_f`, `coastal_minus_kmia_temp_f`, `inland_cooling_lead_score`, `coastal_lag_score`, `kmia_transfer_hint_high/low_f`, `microclimate_confidence`.

### Stage 4: Change Detection + Regime Classification
```
Raw obs (every 5 min)
    ↓
changepoint_detector.py → ChangeDetectorState (which channels fired, probability)
    ↓ (feeds changepoint signal into SKF)
regime_classifier.py → RegimeState (primary_regime + SKF probabilities)
    ↑
kalman_regimes.py (SwitchingKalmanFilter — runs parallel, feeds into RegimeState)
```

- **`engine/changepoint_detector.py`** — Two-layer change detection from industrial process monitoring:
  - **Layer 1: SPECI-style fixed thresholds** — fires instantly on known event signatures. Checks: temp drop >3°F/15min, dew point jump >4°F/30min, pressure surge >1.0 hPa/30min, wind shift >60°/15min, gust-sustained spread >15mph, sky cover change >60%/30min. Zero warm-up, zero historical data needed.
  - **Layer 2: Per-channel CUSUM on detrended residuals** — runs independent CUSUM (Page's test) on each of 8 obs channels after removing diurnal expected values via 3-harmonic Fourier fit. Fires when cumulative deviation from expected exceeds threshold. O(1) per update, 2 floats of state per channel.
  - Output: `ChangeDetectorState` with `fired`, `changepoint_probability` (0-1, scales with number of channels firing), `channels_fired` (which variables drove the change — gives the SKF context for regime selection), `layer` (1=threshold, 2=CUSUM).
  - Includes `fit_diurnal(db_path, lookback_days)` to calibrate the expected value model from historical obs.

- **`engine/regime_classifier.py`** — Deterministic weather-type classification. 4 regimes with additive scoring:
  - `boundary_break` (cliff_down): dew crash + wind rotation + pressure rebound + advection
  - `postfrontal_persistence` (cold_grind): north flow + pressure rising + no recovery + dead CAPE
  - `heat_overrun` (high_locked_early / late_heat_run): obs hot + solar + high locked + coastal lag
  - `mixed_uncertain` (normal_diurnal): default when no regime scores >= 0.55
  - `RegimeState` also carries SKF-derived fields: `skf_probabilities`, `skf_mu_shift_high/low`, `skf_sigma_scale_high/low`, `skf_active_families`

- **`engine/kalman_regimes.py`** — Switching Kalman Filter for real-time regime detection. 5D state vector: `[temp_f, dew_f, pressure_hpa, wind_dir_sin, wind_dir_cos]`. Maintains per-regime linear-Gaussian models (A, b, Q, R). Each cycle: predict forward under each regime model, compare to actual obs, update regime probabilities via Bayes rule. Outputs: probability distribution over regimes, weighted mu/sigma adjustments, innovation norm (surprise metric).
  - `notify_changepoint(probability, channels)` — called by ChangeDetector when a change fires. Pushes regime priors toward uniform and inflates state covariances proportional to changepoint probability, allowing rapid regime re-selection.
  - Loads trained parameters from `analysis_data/skf_config.json`
  - Falls back to conservative defaults if no config exists

### Stage 5: Signal Families
```
RegimeState + ShortRangeFeatures + StationTransferFeatures
    ↓
signal_families.py → SignalFamilyState (5 families + clamps)
```

- **`engine/signal_families.py`** — Collapses raw features into 5 anti-double-counted signal families:
  1. `heat_overrun` (0-1): obs hot + solar + regime match + high lock
  2. `boundary_break` (0-1): dew crash + wind rotation + pressure rebound + regime match
  3. `postfrontal_persistence` (0-1): no recovery + advection + persistence + regime match + obs cold
  4. `microclimate_confirmation` (0-1): inland lead + coastal lag + microclimate confidence
  5. `extreme_lock` (0-1): high_locked_early + warm_tail_removed + low_endpoint_locked
  - Plus `clamps` list: boolean triggers that constrain the distribution

### Stage 6: Residual Adjustment
```
BaselineBelief + RegimeState + SignalFamilyState + StationTransferFeatures
    ↓
residual_estimator.py → AdjustedBelief (shifted/scaled distribution)
```

- **`engine/residual_estimator.py`** — Applies regime-gated and signal-family-weighted adjustments to the baseline distribution:
  - **Gating:** Each family's adjustment is scaled by `(1.0 - model_trust)` — when models are highly trusted, residuals are suppressed.
  - **HIGH-side:** heat_overrun shifts mu up, microclimate hint, high_locked_early clamps upper bound
  - **LOW-side:** boundary_break and postfrontal_persistence shift mu down (dominant gets 1.4x, secondary 0.5x), warm_tail_removed clamps upper, low_endpoint_locked compresses sigma
  - **SKF adjustments** (after family adjustments): `skf_trust = min(1.0, n_training_days / 10)` gates how aggressively SKF corrections apply. `delta_mu += skf_mu_shift * skf_trust`, `sigma_mult *= (1 + (skf_sigma_scale - 1) * skf_trust)`
  - Output: `AdjustedBelief` with transformed `PredictiveDistribution`, `ResidualAdjustment` (delta_mu, sigma_multiplier, clamps, notes)

### Stage 7: Pricing and Execution
```
AdjustedBelief
    ↓
bracket_pricer.py → bracket probabilities
    ↓
edge_detector.py → fee-aware edge stats (YES + NO)
    ↓
bracket_estimates (DB) + regime_confidence
    ↓
trading/execution policy (shared) → side + size decisions
```

- **`engine/bracket_pricer.py`** — Converts predictive distribution into bracket probabilities via Normal CDF integration over each bracket's range (floor, cap).
- **`engine/edge_detector.py`** — Fee-aware edge calculations (`kalshi_fee_cents`) for YES/NO sides.
- **`engine/orchestrator.py`** — Writes inference outputs to `bracket_estimates` (`probability`, `mu`, `sigma`, `active_signals`, `regime_confidence`).
- **Shared trading decision policy** — Implemented in `weather-trader/src/execution/trader_policy.py` and consumed by Miami `paper_trader.py` for side/size selection. Keeps execution logic reusable across future cities/live modes.
- **`engine/quantile_combiner.py`** — Quantile-based distribution utilities. `PredictiveDistribution` with 7-quantile grid. Transformations: `shift_distribution`, `scale_sigma`, `clamp_upper`, `clamp_lower`.

### Shared Engine Utilities
- **`engine/climate_clock.py`** — LST (UTC-5 fixed) climate-day utilities. Midnight-to-midnight boundaries. All time math goes through here.

---

## Analyzer Layer — Post-Settlement Learning

Runs once daily after CLI settlement arrives (~11Z / 6am EST). Builds training data, labels regimes, and retrains the SKF.

### Daily AI Review Pipeline

```
CLI settlement arrives
    ↓
daily_data_builder.py  →  DailyDataPackage (all sources, OHLC aggregated)
    ↓
prompt_builder.py  →  Complete LLM prompt (~55-74K tokens)
    ↓
[OpenClaw cron → Claude]  →  Raw AI output (7 YAML blocks + narrative)
    ↓
ai_review_parser.py  →  ParsedReview → regime_labels table + reviews/ai/ + threshold_history.jsonl
    ↓
train_skf.py  →  Retrained skf_trained_candidate.json (promotion-gated)
```

**Module details:**

- **`analyzer/daily_data_builder.py`** — Pulls ALL collected data for a completed climate day. Renders via OHLC hourly aggregation with RAPID_CHANGE injection for sub-hour transitions. Detection thresholds: 3°F temp, 3°F dew, 45° wind, 0.5 hPa pressure. Includes: surface obs, model forecasts + errors, forward curves, atmospheric data, pressure levels, FAWN, nearby stations, SST buoys, market snapshots, model consensus, previous 3 days' regime labels.

- **`analyzer/prompt_builder.py`** — Assembles the prompt template with dynamic data. Substitutes `{{REGIME_DEFINITIONS}}`, `{{SIGNAL_FAMILY_DEFINITIONS}}`, `{{DAILY_DATA}}`, `{{PREVIOUS_DAYS_CONTEXT}}`, `{{TARGET_DATE}}`. Includes `save_prompt()` for writing to disk for OpenClaw.

- **`analyzer/prompts/daily_review.md`** — The LLM instruction template. Defines: role/constraints, calibrated confidence guardrails, current regime/family definitions, classification/labeling/phase instructions, required 7-section YAML output format, narrative review structure.

- **`analyzer/ai_review_parser.py`** — Parses 7 YAML sections + narrative markdown from AI output. Lenient extraction (each section independent). Stores: `regime_labels` table (upsert), `reviews/ai/` (narrative), `analysis_data/threshold_history.jsonl` (recommendations). Includes `backfill_from_human_reviews()` for seeding from existing manual reviews.

- **`analyzer/train_skf.py`** — Reads `regime_labels`, groups days by dominant regime, fits per-regime state-space parameters (A, b, Q, R) via OLS on hourly obs sequences. Learns mu_shift/sigma_scale from settlement errors. Saves to `analysis_data/skf_trained_candidate.json` (not auto-live). Promotion to `skf_config.json` is gated by statistical audit flow. CLI: `miami-train-skf`.

### HDP-Sticky Shadow Mode (Regime Discovery)

- **`analyzer/hdp_regime_discovery.py`** — Shadow-mode HDP-Sticky HMM for automatic regime discovery. Runs nightly as a batch job on the previous day's obs (pure Python + numpy, no pyhsmm). Discovers the number of regimes from data using the Chinese Restaurant Process. Writes results to `regime_labels_hdp_test` table. **Never touches the live pipeline.** Purpose: evaluate whether HDP-Sticky can eventually replace AI-based regime labeling. CLI: `miami-hdp-shadow`.

### Other Analyzer Tools

- **`analyzer/model_scorer.py`** — Standalone model scoring. 15-day rolling window with time-decay weighting. MAE, bias, RMSE per model per market_type. Used by collector scoring hooks.
- **`analyzer/obs_analyzer.py`** — Post-hoc verification: obs vs forecasts, cloud impact, FAWN scoring, nearby station scoring. (Candidate for retirement — functionality moving to daily_data_builder.)
- **`analyzer/build_training_snapshots.py`** — Build checkpoint-level training rows from collector DB. Raw-data side of hybrid workflow. CLI: `miami-build-training-snapshots`.
- **`analyzer/backtest_residual_bot.py`** — Replay training snapshots through baseline→residual path. Computes naive vs adjusted MAE. CLI: `miami-backtest-residual-bot`.
- **`analyzer/review_labels.py`** — Extract structured labels from human-written daily reviews. CLI: `miami-extract-review-labels`.
- **`analyzer/signal_evaluator.py`** — Score each signal's predictive accuracy against CLI settlements. Hit rate, false positive rate, effect sizes.
- **`analyzer/provisional_qc.py`** — Quality-control helpers for provisional local observations.
- **`analyzer/station_cluster.py`** — Station-cluster summarization for local transfer features.
- **`analyzer/types.py`** — Shared data types for analyzer layer.

---

## Database Schema (v12)

### Core Data Tables

| Table | Purpose | Key Fields |
|-------|---------|------------|
| `observations` | 1-min KMIA surface obs | temp_f, dew_f, wind, pressure, sky, timestamp_utc, lst_date |
| `model_forecasts` | All model forecasts | model, source, high_f, low_f, run_time, valid_time |
| `model_consensus` | Bias-adjusted weighted consensus | model, bias, forecast_f, mae, final_weight, consensus_f |
| `forward_curves` | 12hr lookahead curves every 30min | snapshot_time, per-model temps, cape, pw, solar, pressure levels |
| `event_settlements` | CLI high/low truth | market_type, actual_value_f, settlement_source, raw_text |
| `atmospheric_data` | CAPE, PW, solar, soil, precip | valid_time_utc, cape, pw, radiation fields, soil, precip |
| `pressure_levels` | 925/850/700/500 hPa | temp, wind, geopotential, RH per level |
| `fawn_observations` | Homestead ag station | air_temp_f, dew, wind, solar, rain, soil_temp |
| `nearby_observations` | 60+ ASOS stations | stid, temp_f, temp_delta_vs_kmia, wind, pressure, dew |
| `sst_observations` | Offshore buoys | water_temp, air_temp, wind, pressure |
| `market_snapshots` | Kalshi bracket prices | ticker, bid/ask/last cents, volume, spread |

### Learning/Regime Tables

| Table | Purpose | Key Fields |
|-------|---------|------------|
| `regime_labels` | AI/human daily regime classification | regimes_active (JSON array), phase_summary, signal_labels, threshold_recommendations |
| `signal_events` | Per-cycle signal snapshots | regime, mu, sigma, signal_adjustments (JSON), atmospheric context |
| `signal_scores` | Post-settlement signal grading | signal_grades (JSON), forward_curve_error, signal_net_impact |
| `signal_calibration` | Learned signal parameters | mean_shift, std_shift, hit_rate, sample_count per regime×signal |
| `regime_labels_hdp_test` | HDP-Sticky shadow regime discovery | n_regimes, regime_sequence, regime_params, transition_matrix, phase_summary |

### Operational Tables

| Table | Purpose |
|-------|---------|
| `market_settlements` | Kalshi bracket settlement results |
| `active_brackets` | Currently tradeable brackets |
| `bracket_estimates` | Bot's probability estimates per bracket |
| `collection_runs` | Collector run logs (status, records, errors) |
| `diurnal_scores` | Per-hour model accuracy |
| `model_scores` | Per-model summary scores |
| `cloud_obs` | Cloud cover observations |
| `sse_events` | Raw Wethr SSE events |

---

## File Tree (src/)

```
src/
├── analyzer/
│   ├── __init__.py
│   ├── ai_review_parser.py      ← Parse AI review output, store to DB
│   ├── backtest_residual_bot.py  ← Replay snapshots through modular path
│   ├── build_training_snapshots.py ← Build training rows from collector DB
│   ├── daily_data_builder.py     ← Assemble full day's data for AI prompt
│   ├── model_scorer.py           ← Model MAE/bias scoring
│   ├── obs_analyzer.py           ← Obs vs forecast verification (legacy candidate)
│   ├── prompt_builder.py         ← Assemble LLM prompt from template + data
│   ├── prompts/
│   │   └── daily_review.md       ← LLM instruction template
│   ├── provisional_qc.py         ← QC helpers for provisional obs
│   ├── review_labels.py          ← Extract labels from human reviews
│   ├── signal_evaluator.py       ← Score signal predictive accuracy
│   ├── station_cluster.py        ← Station cluster summarization
│   ├── hdp_regime_discovery.py    ← HDP-Sticky shadow mode (nightly batch)
│   ├── train_skf.py              ← Train SKF from regime-labeled data
│   └── types.py                  ← Shared analyzer types
│
├── collector/
│   ├── __init__.py
│   ├── config.py                 ← TOML + .env configuration
│   ├── entrypoints.py            ← CLI entry point (miami-collector)
│   ├── forward_curve.py          ← Forward curve extraction
│   ├── prune.py                  ← DB pruning (miami-prune-db)
│   ├── runner.py                 ← Async orchestrator (14 loops)
│   ├── signals.py                ← Legacy wrapper (candidate for removal)
│   ├── types.py                  ← Shared collector types
│   ├── sources/
│   │   ├── fawn.py               ← FAWN agricultural station
│   │   ├── iem.py                ← IEM nearby stations
│   │   ├── kalshi_auth.py        ← Kalshi authentication
│   │   ├── kalshi_rest.py        ← Kalshi REST API
│   │   ├── kalshi_ws.py          ← Kalshi WebSocket
│   │   ├── ndbc.py               ← NDBC buoy data
│   │   ├── nws.py                ← NWS ASOS + CLI
│   │   ├── openmeteo.py          ← Open-Meteo models + pressure levels
│   │   ├── synoptic.py           ← [DEAD — replaced by IEM]
│   │   ├── wethr.py              ← Wethr REST forecasts
│   │   └── wethr_stream.py       ← Wethr SSE real-time obs
│   └── store/
│       ├── __init__.py
│       ├── db.py                 ← SQLite wrapper + migrations
│       └── schema.py             ← Schema v12 DDL
│
├── engine/
│   ├── __init__.py
│   ├── baseline_engine.py        ← Model-first bias-corrected belief
│   ├── bias_memory.py            ← Slow structural bias (EMA, JSON-backed)
│   ├── bracket_pricer.py         ← Distribution → bracket probabilities
│   ├── climate_clock.py          ← LST/UTC-5 time utilities
│   ├── edge_detector.py          ← Model vs market edge detection
│   ├── estimator.py              ← [OLD monolithic estimator — being replaced]
│   ├── changepoint_detector.py    ← CUSUM + threshold change detection (industrial)
│   ├── kalman_regimes.py         ← Switching Kalman Filter (5D, per-regime models)
│   ├── policy.py                 ← Trade policy / decision layer
│   ├── quantile_combiner.py      ← PredictiveDistribution + transformations
│   ├── regime_classifier.py      ← Deterministic regime classification + SKF fields
│   ├── residual_estimator.py     ← Regime-gated mu/sigma adjustments
│   ├── short_range_updater.py    ← Short-range trend features from obs
│   ├── signal_engine.py          ← 7-signal extraction from DB
│   ├── signal_families.py        ← Anti-double-counted signal family grouping
│   ├── source_registry.py        ← Provider-agnostic forecast source interface
│   └── station_transfer.py       ← Nearby station → KMIA transfer hints
│
├── settlement/
│   └── climate_clock.py          ← [Alias — see engine/climate_clock.py]
│
├── signals/
│   ├── daily_review.py           ← [LEGACY — marked for retirement]
│   ├── diurnal_scorer.py         ← [LEGACY — marked for retirement]
│   ├── signal_evaluator.py       ← [Reference copy — canonical is analyzer/]
│   ├── signal_scorer.py          ← [LEGACY — marked for retirement]
│   └── signals.py                ← [LEGACY — 1000+ lines, being absorbed]
│
└── trading/
    ├── __init__.py
    └── paper_trader.py           ← Paper execution adapter (consumes shared TraderPolicy)
```

**Note on architecture:** core inference remains in `engine/`. Trade decision/sizing policy is shared from
`/Users/blakebot/blakebot/weather-trader/src/execution/trader_policy.py` so future cities/live execution can reuse a single policy surface.

---

## CLI Entrypoints

| Command | Module | Purpose |
|---------|--------|---------|
| `miami-collector` | `collector.entrypoints:cli_entry` | Run the data collector |
| `miami-prune-db` | `collector.prune:main` | Prune old DB records |
| `miami-extract-review-labels` | `analyzer.review_labels:main` | Extract labels from human reviews |
| `miami-build-training-snapshots` | `analyzer.build_training_snapshots:main` | Build training rows |
| `miami-backtest-residual-bot` | `analyzer.backtest_residual_bot:main` | Replay backtests |
| `miami-train-skf` | `analyzer.train_skf:main` | Train SKF from regime labels |
| `miami-hdp-shadow` | `analyzer.hdp_regime_discovery:main` | Nightly HDP-Sticky shadow run |

---

## Key Data Files

| Path | Purpose |
|------|---------|
| `miami_collector.db` | Main SQLite database (~385MB) |
| `analysis_data/bias_memory.json` | Slow bias corrections per source/market |
| `analysis_data/skf_config.json` | Live SKF regime model parameters |
| `analysis_data/skf_trained_candidate.json` | Newly retrained SKF candidate (promotion-gated) |
| `analysis_data/threshold_history.jsonl` | AI threshold recommendations log |
| `reviews/*.md` | Human-written daily reviews |
| `reviews/ai/*.md` | AI-generated daily reviews |

---

## Plan Documents

| File | Purpose | Status |
|------|---------|--------|
| `plans/regime-classification-ai-labeling.md` | Regime system + AI labeling + SKF build | COMPLETE — all 8 steps done |
| `plans/residual-signal-bot-v1.md` | Core signal bot architecture | CURRENT — canonical design doc |
| `plans/trading-bot-v1.md` | Rules-based trading philosophy | CURRENT — execution framework |
| `plans/repo-blueprint.md` | Cleanup/refactor map | CURRENT — drives organization |
| `plans/hybrid-workflow.md` | Reviews + labels + bot workflow | CURRENT — operational framework |
| `plans/HANDOFF-2026-03-19-BMO.md` | Status handoff from today | CURRENT — starting-state snapshot |
| `plans/SYSTEM-MAP.md` | This document | CURRENT — canonical reference |
| `BOT_ARCHITECTURE.md` | Full system architecture + trading strategy | CURRENT — detailed target design |

---

## What's Working Now

- Collector: 14 async data sources running, DB populated
- Engine: Full modular pipeline (signal_engine → baseline → features → change detection → regime → families → residual → pricing)
- Inference runtime state persists CUSUM/SKF across cycles with day-rollover resets and config hot-reload.
- Paper execution uses shared TraderPolicy from weather-trader (confidence + EV + market-aware sizing) and supports multi-position entries.
- Change detection: CUSUM + SPECI-style thresholds — verified on March 16 frontal passage (fires at 22:00 LST with temp + dew crash, escalates to P(cp)=1.0 by 23:00 with 4 channels)
- SKF: Running with 4 default regime models, integrated with change detector via notify_changepoint()
- Analyzer: AI review system built (data builder, prompt template, parser, SKF trainer)
- HDP-Sticky: Shadow mode module built, writes to regime_labels_hdp_test, ready for nightly cron at 0300 LST
- Regime labels: 4 human reviews backfilled
- Tests: 35 passing, all modules import cleanly

## What's Next

1. **Wire daily AI cron** (OpenClaw, post-CLI settlement ~11Z)
2. **Wire HDP shadow cron** (0300 LST nightly, `miami-hdp-shadow --db miami_collector.db`)
3. **Integrate change detector + SKF live** into the engine loop
4. **Accumulate 10+ AI-labeled days** to train SKF from real data
5. **Compare HDP-Sticky vs AI labels** after 30+ days — evaluate if HDP can replace AI
6. **Tune CUSUM thresholds** using PELT (ruptures) offline changepoint analysis on historical data
7. **Clean up legacy modules** (signals/, trading/ aliases, old estimator.py)
8. **Promote shared TraderPolicy to live adapter** (weather-trader execution path) for Kalshi execution
9. **Evaluate DS3M** at 3+ months when sufficient training data exists
