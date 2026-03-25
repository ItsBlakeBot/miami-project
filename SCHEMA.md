# DS3M System Schema Map

**Last updated:** 2026-03-25
**Purpose:** Annotated data flow diagram for humans and AI agents to understand the full system.

---

## Data Flow: Collection → Inference → Trading

```
═══════════════════════════════════════════════════════════════════════
                        EXTERNAL DATA SOURCES
═══════════════════════════════════════════════════════════════════════

  Wethr SSE ──────────┐  1-min METAR obs (temp, dew, wind, sky, pressure)
  Wethr REST ─────────┤  NWP forecasts (HRRR, RAP, NBM, GFS, ECMWF)
  Open-Meteo API ─────┤  11 deterministic + 2 ensemble models, pressure levels
  Herbie (GRIB) ──────┤  GFS/ECMWF/NBM native grids, HRRR CAPE/PW
  NWS API ────────────┤  ASOS obs (backup), CLI settlement reports
  GOES-19 ────────────┤  Cloud fraction, derived stability index
  FAWN ───────────────┤  Soil temp/moisture, solar radiation (FL Mesonet)
  NDBC ───────────────┤  Buoy SST (VAKF1 Virginia Key, 6-min)
  IEM ────────────────┤  Nearby ASOS network obs (30+ SE FL stations)
  Kalshi REST/WS ─────┘  Market prices, orderbook, settlements

                      │
                      ▼
═══════════════════════════════════════════════════════════════════════
                    COLLECTOR LAYER (src/collector/)
              12 async loops → SQLite WAL (miami_collector.db)
═══════════════════════════════════════════════════════════════════════

  runner.py orchestrates all collection loops. Each source writes to
  its own table(s). Data is append-only with UNIQUE constraints to
  prevent duplicates.

  ┌─────────────────────────────────────────────────────────────────┐
  │                    SQLITE DATABASE (36 tables)                   │
  │                                                                  │
  │  LIVE DATA (written every cycle)                                │
  │  ├── observations          METAR obs (temp, wind, sky, pressure)│
  │  ├── model_forecasts       NWP high/low per model per run       │
  │  ├── atmospheric_data      CAPE, PW, radiation, soil temp       │
  │  ├── pressure_levels       T/wind at 925/850/700/500 hPa       │
  │  ├── forward_curves        Hourly model traces (diurnal profile)│
  │  ├── nearby_observations   30+ SE FL ASOS stations              │
  │  ├── fawn_observations     FL ag mesonet (soil, solar)          │
  │  ├── sst_observations      NDBC buoy (water temp, air, wind)    │
  │  ├── market_snapshots      Kalshi orderbook (bid/ask/vol/depth) │
  │  └── active_brackets       Current tradeable brackets           │
  │                                                                  │
  │  SETTLEMENT & SCORING                                           │
  │  ├── event_settlements     CLI actual high/low per day          │
  │  ├── market_settlements    Kalshi bracket outcomes (win/lose)   │
  │  ├── model_scores          Per-model MAE/bias/RMSE (rolling)    │
  │  ├── model_consensus       Weighted ensemble state              │
  │  └── signal_calibration    Signal family hit rates              │
  │                                                                  │
  │  DS3M-SPECIFIC                                                  │
  │  ├── ds3m_estimates        Bracket probs per inference cycle    │
  │  ├── ds3m_comparison       DS3M vs production (CRPS, edge)     │
  │  ├── ds3m_paper_trades     Paper trade lifecycle                │
  │  ├── ds3m_paper_trade_settlements  Trade P&L                   │
  │  ├── ds3m_training_log     Training metrics per session         │
  │  └── regime_labels         Discovered regime assignments        │
  │                                                                  │
  │  ENRICHED BACKFILL (historical training data)                   │
  │  ├── enriched_asos         269K hourly obs (10 stations, 3yr)  │
  │  ├── enriched_atmosphere   28K hrs CAPE/CIN/cloud (Open-Meteo) │
  │  ├── enriched_sst          255K records (VAKF1 buoy)           │
  │  └── cli_daily_backfill    11.4K daily max/min (05Z-05Z UTC)   │
  │                                                                  │
  │  LEGACY (still populated, read by old engine)                   │
  │  ├── bracket_estimates     Old production bracket probs         │
  │  ├── signal_events         Signal engine snapshots              │
  │  ├── paper_trades          Old paper trader lifecycle           │
  │  └── paper_trade_settlements  Old trade P&L                    │
  └─────────────────────────────────────────────────────────────────┘

                      │
                      ▼
═══════════════════════════════════════════════════════════════════════
              DS3M v2 INFERENCE ENGINE (src/engine/ds3m/)
                    5-second cycles, MPS/CUDA accelerated
═══════════════════════════════════════════════════════════════════════

  orchestrator_v2.py runs the main loop. Each cycle:

  ┌─────────────────────────────────────────────────────────────────┐
  │                                                                  │
  │  STEP 1: FEATURE EXTRACTION                                     │
  │  feature_vector.py                                               │
  │  ┌──────────────────────────────────┐                            │
  │  │ Reads: observations,             │                            │
  │  │        model_forecasts,          │  Outputs: (T, 33) tensor  │
  │  │        atmospheric_data          │  T = buffer of last 48h   │
  │  │        (latest rows per table)   │  at 5-min resolution      │
  │  └──────────────────────────────────┘                            │
  │                     │                                            │
  │                     ▼                                            │
  │  STEP 2: MAMBA TEMPORAL ENCODING                                │
  │  mamba_encoder.py — 8-layer Mamba2, d_model=256, d_state=32    │
  │  ┌──────────────────────────────────┐                            │
  │  │ Input:  (1, T, 33) features      │                            │
  │  │ Output: (1, T, 256) hidden seq   │                            │
  │  │         h_t = last hidden state  │  256-dim representation   │
  │  │                                   │  of weather state + trend │
  │  │ Captures: diurnal cycles,        │                            │
  │  │   synoptic transitions,          │                            │
  │  │   regime-dependent dynamics      │                            │
  │  └──────────────────────────────────┘                            │
  │                     │                                            │
  │                     ▼                                            │
  │  STEP 3: PARTICLE FILTER + REGIME DISCOVERY                     │
  │  diff_particle_filter.py + hdp_regime.py                        │
  │  ┌──────────────────────────────────┐                            │
  │  │ 500 particles, each carrying:    │                            │
  │  │   - remaining_high/low (°F)      │                            │
  │  │   - regime_id (K regimes)        │                            │
  │  │   - 32-dim latent state          │                            │
  │  │   - log_weight                   │                            │
  │  │                                   │                            │
  │  │ HDP discovers regimes:           │  Outputs:                  │
  │  │   - Learnable concentration α    │  - Particle cloud          │
  │  │   - Merge redundant regimes      │  - Regime posterior (K,)   │
  │  │   - Split bimodal regimes        │  - ESS (filter health)     │
  │  │   - Per-regime emission noise    │  - Latent state (32,)      │
  │  │                                   │                            │
  │  │ Known regimes:                   │                            │
  │  │   0=Continental, 1=Sea breeze,   │                            │
  │  │   2=Frontal, 3=Tropical,         │                            │
  │  │   4=Nocturnal, 5+=HDP-discovered │                            │
  │  └──────────────────────────────────┘                            │
  │                     │                                            │
  │                     ▼                                            │
  │  STEP 4: DENSITY ESTIMATION                                     │
  │  neural_spline_flow.py — 10 coupling layers, 24 spline bins    │
  │  ┌──────────────────────────────────┐                            │
  │  │ Conditioning: h_t (256) +        │                            │
  │  │   particle stats (19) +          │                            │
  │  │   regime posterior (5)           │                            │
  │  │   = 280-dim context              │                            │
  │  │                                   │  Outputs:                  │
  │  │ Learns continuous CDF over       │  - Full temperature CDF    │
  │  │ temperature space via invertible │  - log p(x) at any point  │
  │  │ spline transforms               │  - Bracket probabilities   │
  │  └──────────────────────────────────┘                            │
  │                     │                                            │
  │                     ▼                                            │
  │  STEP 5: REAL-TIME UPDATE + CALIBRATION                         │
  │  realtime_updater.py + conformal_calibrator.py                  │
  │  ┌──────────────────────────────────┐                            │
  │  │ Regime-Conditioned Scalar KF:    │                            │
  │  │   Per-regime Q/R priors          │                            │
  │  │   Soft-blended by posterior      │                            │
  │  │                                   │                            │
  │  │ Bracket KF Bank (10 parallel):   │                            │
  │  │   Per-bracket probability KF     │  Outputs:                  │
  │  │   Tail brackets = wider Q/R      │  - Filtered probabilities  │
  │  │   Innovation monitor:            │  - Regime change signal    │
  │  │     → regime_change_signal       │  - Calibrated probs        │
  │  │                                   │                            │
  │  │ Conformal calibration:           │                            │
  │  │   Post-hoc coverage guarantee    │                            │
  │  └──────────────────────────────────┘                            │
  │                     │                                            │
  │                     ▼                                            │
  │  STEP 6: BRACKET PRICING                                        │
  │  bracket_pricer_v2.py                                            │
  │  ┌──────────────────────────────────┐                            │
  │  │ NSF CDF → bracket probabilities  │                            │
  │  │ Compare model P vs market price  │  Outputs:                  │
  │  │ Compute edge per bracket         │  - 10 bracket probs        │
  │  │ Identify tradeable opportunities │  - Edge vs market (¢)      │
  │  └──────────────────────────────────┘  - Recommended trades      │
  │                     │                                            │
  │                     ▼                                            │
  │  STEP 7: PAPER TRADING                                          │
  │  paper_trader_v2.py                                              │
  │  ┌──────────────────────────────────┐                            │
  │  │ Regime-conditioned position caps │                            │
  │  │ Fractional Kelly sizing          │                            │
  │  │ Maker/taker routing:             │                            │
  │  │   edge > 6¢ → taker (immediate) │  Writes:                   │
  │  │   edge 3-6¢ → maker (limit)     │  - ds3m_paper_trades       │
  │  │ Max 8 open, 2 per bracket,      │  - ds3m_paper_trade_       │
  │  │ 15 daily limit, 10-min cooldown │    settlements              │
  │  └──────────────────────────────────┘                            │
  │                                                                  │
  └─────────────────────────────────────────────────────────────────┘

                      │
                      ▼
═══════════════════════════════════════════════════════════════════════
                    LIVE DASHBOARD (src/dashboard/)
                    FastAPI + SSE → http://localhost:8050
═══════════════════════════════════════════════════════════════════════

  ┌─────────────────────────────────────────────────────────────────┐
  │  Real-time panels:                                               │
  │  ├── Current temp + projection curve                             │
  │  ├── Active regime (color-coded) + confidence                    │
  │  ├── Bracket table (model P, market price, edge, liquidity)     │
  │  ├── Open + settled trades with P&L                              │
  │  ├── ESS gauge (particle filter health)                          │
  │  ├── Regime posterior timeline                                   │
  │  └── Innovation monitor alerts                                   │
  └─────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════
                    TRAINING PIPELINE
═══════════════════════════════════════════════════════════════════════

  ┌─────────────────────────────────────────────────────────────────┐
  │  HISTORICAL DATA (enriched backfill tables)                      │
  │  ├── enriched_asos ─────── 269K hourly obs, 10 stations         │
  │  ├── enriched_atmosphere ─ 28K hours CAPE/CIN/cloud             │
  │  ├── enriched_sst ──────── 255K buoy records                    │
  │  └── cli_daily_backfill ── 11.4K daily targets (05Z-05Z UTC)   │
  │                                                                  │
  │  AUGMENTATION (augmentation.py)                                  │
  │  ├── Temporal jitter (±1-3 steps, p=0.5)                        │
  │  ├── Feature noise (calibrated per-feature σ, p=0.7)           │
  │  ├── Station dropout (zero one group, p=0.15)                   │
  │  └── Regime splicing (swap halves, p=0.1)                       │
  │  Effective sample multiplier: ~6x → 50-80K independent samples │
  │                                                                  │
  │  PHASE 1: Mamba pre-training                                    │
  │  ├── Loss: 0.3×MSE_next + 0.3×MSE_remaining + 0.4×CRPS_max   │
  │  ├── AdamW (lr=1e-3, wd=1e-3), cosine LR, patience=10        │
  │  └── Output: analysis_data/mamba_pretrained.pt                  │
  │                                                                  │
  │  PHASE 2: NSF fine-tuning on bracket settlements                │
  │  ├── Loss: CRPS + bracket Brier score                           │
  │  └── Output: analysis_data/nsf_trained.pt                       │
  │                                                                  │
  │  NIGHTLY: Post-settlement incremental update                    │
  │  └── 120-day rolling window retrain                             │
  └─────────────────────────────────────────────────────────────────┘
```

---

## Key Table Relationships

```
event_settlements.actual_value_f  ←──  Ground truth (CLI high/low)
        │
        ├──→ model_scores (MAE/bias per model, computed daily)
        ├──→ ds3m_comparison (DS3M vs production CRPS)
        ├──→ market_settlements (which bracket won)
        └──→ paper_trade_settlements (P&L on paper trades)

model_forecasts ──→ feature_vector [0-10] ──→ Mamba
observations    ──→ feature_vector [11-16] ──→ Mamba
atmospheric_data ─→ feature_vector [17-22] ──→ Mamba
(timestamps)    ──→ feature_vector [23-28] ──→ Mamba ──→ h_t (256)
                                                              │
                                                              ▼
market_snapshots ─────────────────────────────→ bracket_pricer (edge calc)
                                                              │
                                                              ▼
                                                    ds3m_paper_trades
```

---

## Time Conventions

- **All timestamps in UTC** (Z time) throughout the system
- **CLI climate day**: 05:00 UTC to 04:59 UTC (= midnight-to-midnight EST)
- **Settlement**: NWS CLI report, typically published 05:30-06:30 UTC
- **IEM station codes**: 3-letter (MIA, FLL, OPF) — mapped to 4-letter ICAO (KMIA, KFLL, KOPF)
- **Market hours**: Kalshi brackets trade until settlement
