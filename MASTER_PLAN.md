# DS3M Weather Trading System — Master Plan

**Station:** KMIA (Miami International Airport)
**Market:** Kalshi temperature bracket contracts (HIGH/LOW)
**Last updated:** 2026-03-25
**Status:** DS3M v2 training in progress

---

## System Overview

DS3M (Deep Switching State Space Model) is a unified particle-filter-based inference system that replaces the old multi-component production pipeline with a single coherent generative model. It learns weather regimes, temporal dynamics, and bracket probabilities autonomously from data.

```
COLLECTOR (unchanged)              DS3M v2 ENGINE                    TRADING
├─ Wethr SSE (1-min obs)          ├─ Mamba Encoder (8L, d=256)     ├─ Paper Trader v2
├─ Wethr REST (NWP forecasts)     ├─ Differentiable PF (500p)      │  ├─ Regime-conditioned sizing
├─ Open-Meteo (11 det models)     ├─ HDP Regime Discovery           │  ├─ Fractional Kelly
├─ Herbie (GFS/ECMWF/NBM)        │  ├─ Learnable concentration     │  ├─ Maker/taker routing
├─ NWS API (ASOS + CLI)          │  ├─ Merge-split moves           │  └─ Per-regime position caps
├─ HRRR (CAPE, PW, BL)           │  └─ Per-regime emission noise   ├─ Bracket KF Bank (10 KFs)
├─ GOES-19 (cloud, DSI)          ├─ Neural Spline Flow (10T, 24B)  │  └─ Innovation monitoring
├─ FAWN (soil, solar)            ├─ Conformal Calibration           └─ [Future] Live Execution
├─ NDBC (buoy SST)               ├─ Regime-Conditioned KF
├─ IEM (nearby ASOS)             ├─ Real-time Updater              DASHBOARD (FastAPI+SSE)
├─ Kalshi REST/WS (markets)      └─ Bracket Pricer v2              └─ localhost:8050
└─ Inference loop (5-sec)
        │
        ▼
   SQLite WAL (36 tables)
```

---

## Model Architecture (4.4M parameters)

| Component | Params | Config | Role |
|-----------|--------|--------|------|
| Mamba Encoder | 3.6M | d_model=256, d_state=32, 8 layers | Temporal feature extraction from 33-dim weather vectors |
| DPF | 55K | 500 particles, d_latent=32, 5+ regimes | Differentiable particle filter with HDP regime discovery |
| NSF | 784K | 10 transforms, 24 bins, hidden=128 | Continuous density estimation for bracket pricing |

### 33-Feature Vector (input to Mamba)

| Indices | Group | Features |
|---------|-------|----------|
| 0-10 | NWP | HRRR, RAP, NBM (50th/10th/90th), GFS, ECMWF, GEFS mean/spread, model spread |
| 11-16 | Observations | temp, dewpoint, wind dir/speed, cloud cover, running max |
| 17-22 | Atmospheric | CAPE, CIN, T925, T850, wind 925 speed/dir |
| 23-28 | Temporal | hour sin/cos (UTC), DOY sin/cos, lead time to 05Z, DST flag |
| 29-32 | Derived | HRRR 3-day bias, NBM 3-day bias, dewpoint Δ1h, wind dir Δ1h |

### Training Data

| Source | Records | Coverage |
|--------|---------|----------|
| IEM ASOS (10 SE FL stations) | 269K hourly obs | 2023-01 to 2026-03 |
| Open-Meteo (CAPE/CIN/cloud) | 28K hours | 2023-01 to 2026-03 |
| NDBC Virginia Key SST | 255K records | 2023-01 to 2026-03 |
| CLI daily max/min targets | 11.4K station-days | 05Z-05Z UTC windowing |

### Training Pipeline

**Phase 1 — Mamba Pre-training** (50 epochs)
- Multi-task: 0.3×MSE_next_temp + 0.3×MSE_remaining_move + 0.4×CRPS_daily_max
- Augmentation: temporal jitter, feature noise, station dropout, regime splicing
- AdamW (lr=1e-3, weight_decay=1e-3), cosine annealing, early stopping (patience=10)

**Phase 2 — NSF Fine-tuning** (100 epochs)
- CRPS + bracket Brier score on settlement outcomes
- 120-day rolling window, retrained nightly post-settlement

### Training Performance: Pre-computed Feature Tensors

**CRITICAL FOR FUTURE DS3M TRAINING:** All 33 features are pre-computed into
`(N, 33)` tensors at dataset init time (once, ~2 seconds). The `__getitem__`
method then does a simple tensor slice + augmentation (~0.04ms per sample).

This is **~10,000x faster** than computing features on-the-fly (which requires
per-timestep Python dict lookups for atmospheric data, datetime parsing for
temporal features, and numpy↔torch conversions). The on-the-fly approach made
each epoch take 30+ minutes; pre-computed tensors bring it to ~2-3 minutes.

**Always use this pattern for any DS3M variant or new station training:**
1. Load all data from DB into numpy arrays (once)
2. Pre-compute full feature tensors per station (vectorized numpy/torch)
3. Store pre-computed `(T, 33)` tensor slices in each sample dict
4. `__getitem__` = `.clone()` + augmentation only

---

## DS3M Module Map

```
src/engine/ds3m/
├── mamba_encoder.py        — Mamba2 SSM backbone (8L, d=256)
├── diff_particle_filter.py — Differentiable PF + HDP regime discovery
├── neural_spline_flow.py   — Conditional NSF density estimation
├── hdp_regime.py           — HDP-Sticky HMM regime manager
├── skew_normal.py          — Skew-normal distribution + CRPS
├── feature_vector.py       — 33-feature extraction from DB
├── realtime_updater.py     — Regime-conditioned KF + Bracket KF bank
├── bracket_pricer_v2.py    — CDF → bracket probability conversion
├── conformal_calibrator.py — Post-hoc probability calibration
├── paper_trader_v2.py      — Regime-conditioned paper trading
├── orchestrator_v2.py      — Main loop: feature→Mamba→DPF→NSF→trade
├── augmentation.py         — Training data augmentation pipeline
├── training_pipeline.py    — Dataset classes + DS3MTrainer
├── train.py                — CLI training runner
├── config.py               — Unified hyperparameter config
├── cli_backfill.py         — IEM ASOS daily max/min backfill
├── enriched_backfill.py    — Multi-source historical data ingest
└── _deprecated_v1/         — Old DS3M v1 files (archived)
```

---

## What's Deprecated

The old production engine (`src/engine/_deprecated_v1/`) contained 30+ separate modules:
- baseline_engine, emos, boa, bracket_pricer, regime_catalog, letkf, sigma_climatology, etc.

These are **fully replaced** by DS3M v2. The collector layer (`src/collector/`) is unchanged — DS3M reads from the same SQLite tables.

The old DS3M v1 (`src/engine/ds3m/_deprecated_v1/`) had 10 files with non-differentiable particle filter, manual regime dynamics, and no Mamba backbone.

Old plan files are archived in `plans/_archived_pre_ds3m/`.

---

## Remaining Work

### Immediate (post-training)
- [ ] Verify trained model weights produce reasonable bracket probabilities
- [ ] Run smoke test: full Mamba→DPF→NSF→bracket pipeline
- [ ] Wire paper_trader_v2 into orchestrator for live paper trading
- [ ] Launch dashboard on port 8050

### Short-term
- [ ] Nightly retraining pipeline (post-settlement auto-update)
- [ ] Conformal calibration from accumulated settlements
- [ ] HDP merge-split monitoring and regime naming
- [ ] Innovation monitor → regime change early warning

### Medium-term
- [ ] Trading center bot (see TRADING_BOT_DESIGN.md)
- [ ] Multi-station expansion (KFLL, KPBI, etc.)
- [ ] Live execution on Kalshi (graduate from paper trading)
- [ ] Cloud GPU deployment for faster training

### Long-term: GraphMamba Spatial Expansion

Upgrade the temporal-only Mamba to a **GraphMamba** architecture that adds explicit
spatial reasoning across the 10 SE Florida stations.

**Architecture:**
```
Station graph (10 nodes, distance-weighted edges):

  KPBI ── KBCT ── KFLL ── KMIA
    │                │       │
  KPMP ── KFXE ── KOPF ── KTMB
                             │
                           KHST ── KHWO
```

Each node runs its own temporal Mamba encoder (shared weights), then graph
attention layers aggregate spatial context between stations. Edges can be:
- **Static**: geographic distance (fixed adjacency matrix)
- **Dynamic**: wind-direction-dependent (upwind stations get higher attention)

**What it captures that current Mamba cannot:**
- Sea breeze propagation timing (coastal → inland, 20-40 min lag)
- Frontal passage sequencing (KPBI hours before KMIA)
- Urban heat island gradients (KMIA vs rural KHST)
- Spatial correlation structure for multi-station bracket trading

**When to build:** After KMIA-only DS3M is validated and profitable. GraphMamba
becomes high-value when we expand to trade KFLL/KPBI brackets (3x revenue surface)
and need spatial context for each station's forecast.

**Estimated scope:** ~500K-1M additional params, new `graph_mamba.py` module,
timestamp alignment across stations, adjacency matrix construction.
