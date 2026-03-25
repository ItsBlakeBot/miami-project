# Weather Trading Bot — Master Plan

**Last updated:** 2026-03-22
**Station:** KMIA (Miami International Airport) — scaling to all ~20 Kalshi trading stations
**Canonical document.** This file is the single source of truth for architecture, status, and execution.
**Supersedes:** weather-trading-architecture-roadmap.md, weather-trading-ticketized-build-order.md, weather-trading-implementation-checklist.md

---

## 1. System Overview

Two repositories, one mission: trade Kalshi temperature bracket markets using real-time weather data, model consensus, and regime-aware probability estimation.

| Repo | Owns | Location |
|------|------|----------|
| `miami-project` | Data collection, inference, calibration, regime analysis, bracket probability generation | `/Users/blakebot/blakebot/miami-project` |
| `weather-trader` | Portfolio sizing, risk policy, execution (paper/live), trade selection from orchestrator outputs | `/Users/blakebot/blakebot/weather-trader` |

**Runtime:** `launchd` plist `com.blakebot.miami-collector` — always-on, auto-restart on crash.

---

## 2. Architecture Tracks

### Option A — Live / Production
The production path: spatial assimilation, calibrated probabilistic forecasting, interpretable regimes, adaptive trading.

### Option C — Shadow / R&D
The challenger path: DS3M, sticky HDP-HMM, mixture-of-experts, conformal prediction. No trade authority until promoted.

**Core rule:** Option A is live. Option C is shadow. C earns promotion only through measurable superiority over 30-60 climate days.

---

## 3. Data Ingestion Layer (17 Async Loops)

### Current Sources

| # | Source | Adapter | Cadence | Data Type | Status |
|---|--------|---------|---------|-----------|--------|
| 1 | Wethr SSE | `sources/wethr_stream.py` | 1-min streaming | Primary obs (temp, dew, wind, pressure) + running envelope | ✅ Live |
| 2 | Wethr REST | `sources/wethr.py` | 5-min poll | 19+ model forecasts (NBM, NAM, HRRR, MOS variants, etc.) | ✅ Live |
| 3 | Open-Meteo | `sources/openmeteo.py` | 5-min poll | 11 deterministic models + GFS 31-member + ECMWF 51-member ensembles + atmospheric | ✅ Live |
| 4 | NWS | `sources/nws.py` | 2.5-min poll | ASOS obs + CLI settlements | ✅ Live |
| 5 | Kalshi WS | `sources/kalshi_ws.py` | Streaming | Orderbook deltas, bracket snapshots | ✅ Live |
| 6 | Kalshi REST | `sources/kalshi_rest.py` | 10-min poll | Active bracket/ticker discovery | ✅ Live |
| 7 | IEM | `sources/iem.py` | 5-min poll | Nearby ASOS stations within 200km | ✅ Live (expanded from 50mi) |
| 8 | FAWN | `sources/fawn.py` | 15-min poll | 10 Florida stations within 200km (expanded from 1) | ✅ Live |
| 9 | NDBC | `sources/ndbc.py` | 30-min poll | SST buoy observations | ✅ Live |
| 10 | NWS Gridpoint | `sources/nws.py` | 30-min poll | Mixing height, gridpoint metadata | ✅ Live |
| 11 | Pressure Levels | `sources/openmeteo.py` | Hourly | 925/850/700/500 hPa temp, wind | ✅ Live |
| 12 | Atmospheric | `sources/openmeteo.py` | 5-min poll | CAPE, PW, precip | ✅ Live |
| 13 | HRRR | `sources/hrrr.py` | Hourly | CAPE, PW, cloud, radiation, upper-air temps, 850mb wind, HPBL | ✅ Live |
| 14 | GOES-19 | `sources/goes.py` | 10-min | Satellite cloud mask, CAPE, lifted index, stability indices | ✅ Live |
| 15 | Synoptic | (stub) | — | Disabled (went paid-only) | ❌ Stub |
| 16 | Model Scoring | `analyzer/model_scorer.py` | Daily | Per-model MAE scoring | ✅ Live |
| 17 | Inference | orchestrator.py | Obs-triggered / 5-min fallback | Full inference pipeline + paper trader | ✅ Live |

### Sources NOT Yet Ingested (Prioritized)

| Priority | Source | Value | Effort | Status |
|----------|--------|-------|--------|--------|
| P1 | RTMA/URMA analyzed fields | 2.5km hourly analyzed temp/cloud/wind/pressure | Medium | NOT_STARTED |
| P1 | MRMS composite reflectivity | Outflow detection (15-30 min early warning) | Medium | NOT_STARTED |
| P2 | Radiosonde 72202 (Miami) | 00Z/12Z upper-air profiles via Siphon | Low | NOT_STARTED |
| P2 | GOES-19 GLM lightning | Convective activity indicator | Low | NOT_STARTED |
| P3 | NYSM (NY State Mesonet) | 127 stations for NE Corridor scaling | Low | NOT_STARTED |
| P3 | TexMesonet | Texas stations for KAUS/KSAT/KDFW scaling | Low | NOT_STARTED |
| Skip | SynopticPy | Requires .edu email for free tier | — | Blocked |
| Skip | ASOS 1-min via IEM | 18-36h delay, not real-time | — | Not useful live |

---

## 4. Inference Pipeline (Orchestrator)

The inference pipeline runs every time new data arrives (obs-triggered via LiveState) with a 5-minute fallback. All steps run for both **high** and **low** markets.

```
Step 1:  SignalEngine.extract()
           → running high/low, obs, atmospheric signals

Step 1b: LETKF spatial assimilation (when cluster state available)
           → analyzed temperature at trading stations from nearby obs network

Step 2:  Build remaining-move forecast snapshots (forward valid_time only)

Step 3:  BaselineEngine.build_baseline()
           → BOA-weighted source combination (T1.5 overrides T1.4 when sufficient data)
           → per-source weights from freshness + tracking error + family multipliers

Step 3b: EMOS remaining-move calibration (when sufficient data)
           → calibrates the predicted remaining move, not the absolute forecast
           → trained on: predicted_remaining → actual_remaining (from CLI settlement)
           → naturally scales with time-of-day (late day = small remaining = small correction)
           → currently IDENTITY (no correction) until 25+ settlement days accumulate
           → daily recalibrator auto-refits as data grows

Step 4:  ChangeDetector (BOCPD primary, CUSUM backup)
           → changepoint probability, run length, persistence confidence

Step 4a: Cloud cover estimation (METAR sky codes + FAWN solar clearness index)

Step 4b: Live regime catalog inference
           → 6 regimes: marine_stable, inland_heating, cloud_suppression,
             convective_outflow, frontal_passage, transition
           → sigma_multiplier + mu_bias conditioning

Step 5:  Remaining-move prior (sigma_climatology.py)
           → blends forward curve remaining move with IEM 6yr empirical remaining potential
           → sigma compressed by p_lock (probability extreme is already locked)
           → changepoint-aware: BOCPD probability widens sigma cap during regime breaks
           → prevents late-day sigma inflation (e.g., sigma=1.6 at 5PM for locked high)

Step 5b: Regime conditioning + running extreme anchoring
           → sigma scaled by regime, mu biased by regime (market-type-specific)
           → mu floored/ceilinged at running high/low
           → SKF: DORMANT (corrections disabled, still logs for research)

Step 6:  Bracket pricing (truncated normal CDF)
           → P(bracket) for each active bracket
           → physical constraints: zero out impossible brackets

Step 7:  Edge detection (fee-aware)
           → Kalshi taker fee: ceil(0.07 × P × (1-P) × 100)
           → YES edge, NO edge per bracket

Step 8:  Write estimates to bracket_estimates table

Step 9:  Bracket arbitrage scan (diagnostic)
           → flag sum-of-prices violations

Step 10: Paper trader cycle (entries, exits, settlement resolution)
```

### Tunable Parameters (analysis_data/inference_config.json)

All thresholds are configurable and hot-reloaded without restart:

| Parameter | Default | Controls |
|-----------|---------|----------|
| `boa_min_updates` | 5 | Settlement updates before BOA overrides source trust |
| `regime_confidence_gate` | 0.3 | Min confidence to apply regime sigma/mu conditioning |
| `regime_mu_bias_min_f` | 0.1 | Min °F bias to apply regime mu correction |
| `skf_corrections_enabled` | false | SKF dormant (true to reactivate) |
| `letkf_min_obs` | 2 | Min observations for LETKF update |
| `letkf_sage_husa_b` | 0.97 | Forgetting factor for adaptive R estimation |
| `letkf_max_inflation` | 2.0 | Max RTPS inflation factor |

---

## 5. Daily Recalibration Pipeline (auto_recalibrator.py)

Runs post-settlement (~09:00 LST) each day:

1. **AnEn archiving** — pair today's forecasts with CLI observations for long-term analog ensemble
2. **Drift detection** — PSI on feature distributions; flags concept drift
3. **EMOS refit** — remaining-move calibration, 40-day window (20-day if drift detected), min 25 samples per market type before EMOS engages (stays identity until then)
4. **BOA weight update** — settlement CRPS → sleeping expert protocol
5. **P&L regime detection** — BOCPD on daily P&L → position scaling recommendations
6. **Regime catalog review** — flag unrecognized patterns for HDP-HMM discovery
7. **Health monitoring** — rolling Brier, CRPS, calibration slope, Sharpe, alerts

## 5b. AI Review Cron Job (OpenClaw / GPT-5.4 high-thinking)

**Purpose:** Daily analytical review after recalibration. The AI cron is **advisory only**: it evaluates what happened, why, and what to improve.

**Critical constraint (must):**
- The AI cron is **read-only** and **cannot implement changes by itself**.
- It may propose code/config/threshold changes with rationale, but any production changes require explicit human (or separately approved agent) action.

**Setup:** Configure an OpenClaw scheduled task that fires ~10:00 LST daily (after the auto_recalibrator finishes at ~09:00).
Hard-pinning GPT-5.4 with high-thinking mode is acceptable for this cron.

### Inputs to review each day
- Latest inference notes + regime states (core classifier + regime catalog conditioning)
- EMOS state (`analysis_data/emos_state.json`) and whether run is fitted vs cold-start identity
- BOA state/weights (`analysis_data/boa_state.json`) by side (high vs low)
- Source trust refresh/state (`analysis_data/source_trust_refresh.json`, `analysis_data/source_trust_state.json`)
- Settlement outcomes vs projections
- Paper trader entries/exits/marks/P&L
- Weather feature traces (dew/wind/pressure/cloud/CAPE/PW)
- **HDP Sticky log**: `regime_labels_hdp_test` (SQLite table in `miami_collector.db`)

### What the AI cron should do
1. **Regime fit assessment**
   - Assess whether EMOS + regime catalog appeared to fit the day’s data for existing regimes.
   - Evaluate whether core regime classifications were plausible.

2. **Novel regime judgment**
   - If behavior is outside known regimes, state whether “novel/unspecified” was appropriate.
   - Propose candidate new regime(s) with concise rationale and discriminating features.

3. **Regime naming suggestions (all regime layers)**
   - Propose naming/labeling refinements for:
     - core regime classifier labels,
     - regime catalog labels,
     - EMOS regime-facing summaries (high/low behavior labels),
     - HDP Sticky discoveries in `regime_labels_hdp_test`.
   - Include mapping proposals with confidence and rationale.

4. **Projection + calibration quality**
   - Review mu/sigma behavior (over-compression/over-expansion flags).
   - Report EMOS mode (fitted vs cold-start) and sample depth implications.
   - Report BOA side-specific family weighting behavior and plausibility.

5. **Trade quality summary**
   - Compare EV-ranked opportunities vs realized outcomes.
   - Call out positives, negatives, misses, and false positives.

6. **Actionable recommendations (advisory only)**
   - Safe, bounded next-day suggestions with explicit why.
   - No direct mutation of production config/code.

### Required output format
Produce two outputs:

1) **Human-readable daily summary** (markdown):
- What happened today
- Why the model behaved that way
- Positives
- Negatives
- What we learned
- What to try next (advisory)

2) **Structured JSON payload**:
```json
{
  "date": "YYYY-MM-DD",
  "regime_fit": {
    "status": "ok|mismatch|uncertain",
    "core_called": "mixed_uncertain",
    "catalog_conditioning": "marine_stable",
    "notes": "..."
  },
  "novel_regime_assessment": {
    "novel_flag": false,
    "reasoning": "...",
    "candidate_new_regimes": [
      {"name": "...", "confidence": 0.0, "rationale": "..."}
    ]
  },
  "regime_naming_suggestions": {
    "core_regime_labels": [
      {"current": "mixed_uncertain", "suggested": "...", "confidence": 0.0, "rationale": "..."}
    ],
    "regime_catalog_labels": [
      {"current": "marine_stable", "suggested": "...", "confidence": 0.0, "rationale": "..."}
    ],
    "emos_behavior_labels": {
      "high": [{"suggested": "...", "confidence": 0.0, "rationale": "..."}],
      "low": [{"suggested": "...", "confidence": 0.0, "rationale": "..."}]
    },
    "hdp_sticky": {
      "log_source": "miami_collector.db:regime_labels_hdp_test",
      "mappings": [
        {
          "hdp_regime_id": 0,
          "suggested_name": "...",
          "confidence": 0.0,
          "rationale": "..."
        }
      ]
    }
  },
  "projection_quality": {
    "high": {"mu_ok": true, "sigma_ok": true, "notes": "..."},
    "low": {"mu_ok": true, "sigma_ok": false, "notes": "..."}
  },
  "calibration_status": {
    "emos_mode": "fitted|cold_start_identity",
    "emos_samples": 0,
    "boa_summary": "...",
    "source_trust_stability": "stable|drifting|insufficient_data"
  },
  "trade_quality": {
    "summary": "good|fair|poor",
    "positives": [],
    "negatives": [],
    "notes": "..."
  },
  "next_day_suggestions": [
    {"change": "...", "reasoning": "...", "risk": "low|medium|high"}
  ],
  "advisory_only": true
}
```

**Approval flow:** AI suggestions are advisory only. No automatic implementation.

---

## 6. Module Inventory

### Engine Layer (src/engine/)

| Module | Purpose | Ticket | Status |
|--------|---------|--------|--------|
| `orchestrator.py` | End-to-end inference cycle | — | ✅ Live |
| `signal_engine.py` | 7 deterministic signals (CAPE, wind, dew, etc.) | — | ✅ Live |
| `baseline_engine.py` | Skill-weighted source combination | T1.1 | ✅ Live |
| `dynamic_weights.py` | Freshness + tracking + family multiplier weighting | T1.1 | ✅ Live |
| `emos.py` | EMOS calibration (closed-form CRPS, BFGS fitting) | T1.2 | ✅ Live |
| `boa.py` | BOA sleeping expert source weighting | T1.5 | ✅ Live (wired into orchestrator) |
| `source_trust.py` | Static source trust multipliers (T1.4 interim prior) | T1.4 | ✅ Live (BOA overrides when ready) |
| `letkf.py` | Regional cluster LETKF (SE Florida: 14 stations) | T3.1-3.3 | ✅ Live |
| `observation_operators.py` | Obs source classification + error models for LETKF | T3.2 | ✅ Live |
| `bocpd.py` | Bayesian Online Changepoint Detection | T2.1 | ✅ Live |
| `changepoint_detector.py` | 3-layer detection (SPECI → CUSUM → BOCPD) | T2.1-2.2 | ✅ Live |
| `kalman_regimes.py` | Switching Kalman Filter (5D state, DORMANT) | — | ✅ Dormant (logs for research) |
| `regime_catalog.py` | 6-regime live classifier + sigma/mu conditioning | T4.1-4.2 | ✅ Live |
| `cloud_cover.py` | Cloud fraction from METAR + FAWN solar | TA8.1 | ✅ Live |
| `bracket_pricer.py` | Truncated normal CDF → bracket probabilities | — | ✅ Live |
| `bracket_arbitrage.py` | Cross-bracket sum-of-prices monitor | T6.4 | ✅ Live (diagnostic) |
| `edge_detector.py` | Fee-aware edge computation | — | ✅ Live |
| `sigma_climatology.py` | IEM ASOS 6yr empirical σ_remaining | — | ✅ Live |
| `residual_estimator.py` | Post-regime mu/sigma corrections | — | ✅ Live |
| `short_range_updater.py` | Recent obs adjustment (0-12h) | — | ✅ Live |
| `station_transfer.py` | Station-pair drift modeling | — | ✅ Live |
| `signal_families.py` | Signal family state aggregation | — | ✅ Live |
| `regime_classifier.py` | Deterministic + SKF overlay | — | ✅ Live |
| `bias_memory.py` | Slow per-source bias learning | — | ✅ Live |
| `quantile_combiner.py` | Distribution algebra (combine, shift, scale) | — | ✅ Live |
| `source_registry.py` | Forecast source snapshot management | — | ✅ Live |
| `replay_context.py` | Backtest-safe timestamp parsing | — | ✅ Live |
| `policy.py` | Trading policy interface | — | ✅ Live |

### Collector Layer (src/collector/)

| Module | Purpose | Status |
|--------|---------|--------|
| `runner.py` | 17 async collection loops + inference | ✅ Live |
| `sources/wethr.py` | Wethr REST forecast polling | ✅ Live |
| `sources/wethr_stream.py` | Wethr SSE streaming obs | ✅ Live |
| `sources/openmeteo.py` | Open-Meteo forecasts + ensembles + atmospheric | ✅ Live |
| `sources/nws.py` | NWS ASOS obs + CLI + gridpoint | ✅ Live |
| `sources/kalshi_ws.py` | Kalshi WebSocket market data | ✅ Live |
| `sources/kalshi_rest.py` | Kalshi REST ticker discovery | ✅ Live |
| `sources/iem.py` | IEM nearby ASOS stations (200km radius) | ✅ Live |
| `sources/fawn.py` | FAWN Florida mesonet (10 stations) | ✅ Live |
| `sources/ndbc.py` | NDBC SST buoy observations | ✅ Live |
| `sources/hrrr.py` | HRRR via Herbie + AWS S3 (hourly) | ✅ Live |
| `sources/goes.py` | GOES-19 satellite products (10-min) | ✅ Live |
| `sources/synoptic.py` | Synoptic (disabled — paid only) | ❌ Stub |

### Analyzer Layer (src/analyzer/)

| Module | Purpose | Status |
|--------|---------|--------|
| `auto_recalibrator.py` | Daily recalibration pipeline orchestrator | ✅ Live |
| `health_monitor.py` | Rolling quality monitors + alerts | ✅ Live |
| `pnl_regime_detector.py` | BOCPD on P&L for trading regime detection | ✅ Live |
| `drift_detector.py` | PSI-based concept drift detection | ✅ Live |
| `signal_deprecation.py` | Signal contribution tracking + auto-downweight | ✅ Live |
| `anen_archiver.py` | Analog Ensemble data archiving (long-term) | ✅ Live |
| `model_scorer.py` | Per-model MAE scoring | ✅ Live |
| `source_trust_backfill.py` | Historical source performance extraction | ✅ Live |
| `source_trust_refresh.py` | Rolling source trust refresh | ✅ Live |
| `canonical_replay_bundle.py` | Canonical evaluation harness | ✅ Live |
| `changepoint_replay_compare.py` | BOCPD vs CUSUM comparison | ✅ Live |
| `archive_audit.py` | Archive completeness audit | ✅ Live |

### Trading Layer (src/trading/)

| Module | Purpose | Status |
|--------|---------|--------|
| `paper_trader.py` | Paper trade entry/exit/settlement simulation | ✅ Live |

---

## 7. Multi-Station Scaling Strategy

### Regional Clusters

| Cluster | Trading Stations | Supporting Obs | LETKF/IMM-KF |
|---------|-----------------|----------------|---------------|
| **SE Florida** | KMIA, KFLL | KOPF, KHWO, KTMB, KHST, KFXE, KPMP, KBCT, KPBI, FAWN (10 stations), NDBC buoys | LETKF (14 stations) |
| **NE Corridor** | KNYC, KPHL, KBOS, KDCA | KLGA, KJFK, KEWR, NYSM (127 stations) | LETKF |
| **Texas Triangle** | KAUS, KSAT, KHOU, KDFW | KDAL, IEM nearby ASOS, TexMesonet | LETKF |
| **Upper Midwest** | KMDW, KMSP | KORD + nearby ASOS | LETKF |
| **Mountain/Desert** | KDEN, KPHX, KLAS | Sparse — fall back to IMM-KF | IMM-KF |
| **West Coast** | KLAX, KSFO, KSEA | Sparse coastal — IMM-KF with marine obs | IMM-KF |

### Cold-Start Pipeline for New Stations
1. **Week 1-2:** Semi-local EMOS trained on climatologically similar stations
2. **Week 3-6:** Blend regional + local data via BOA weights
3. **Month 2+:** Local model dominates, regional provides regularization

---

## 8. Ticket Status (as of 2026-03-22)

### ✅ DONE

| Ticket | Description |
|--------|-------------|
| T0.1 | Archive audit + report artifacts |
| T0.2 | Canonical replay bundle (authoritative comparison path) |
| T0.3 | Metric package (CRPS, Brier, calibration, EV, worst-day) |
| T0.4 | Source-trust backfill extractor + artifacts |
| T1.1 | Dynamic weighting engine + baseline integration |
| T1.2 | EMOS remaining-move calibration (CRPS, BFGS; identity until 25+ days) |
| T2.1 | BOCPD + run-length outputs + comparison bundle |
| T2.2 | CUSUM downgraded to backup/diagnostic |
| T3.1 | Regional cluster LETKF design (SE Florida) |
| T3.2 | Observation operators + error models |
| T3.3 | LETKF core implementation (Gaspari-Cohn, RTPS, Sage-Husa) |
| T4.1 | Regime catalog (6 regimes) |
| T4.2 | Regime conditioning wired into bracket pricing |
| TA8.1 | Cloud cover (METAR + FAWN solar) |
| TA8.2 | HRRR adapter via Herbie (CAPE, PW, cloud, radiation, upper-air) |
| TA8.3 | GOES-19 satellite adapter (cloud, CAPE, stability) |
| T6.4 | Bracket arbitrage monitoring |
| T7.1 | Health monitor (Brier, CRPS, calibration, alerts) |
| T7.2 | P&L regime detector (BOCPD on P&L) |
| T7.3 | Concept drift detector (PSI) |
| T7.4 | Signal deprecation tracking |
| — | Remaining-move prior wired in (sigma climatology + changepoint-aware sigma cap) |
| — | DSM settlement ingestion disabled (only final CLI accepted) |
| — | Premature CLI gate: settlement_date must have ended (now > D+1 05:00Z) |
| — | FAWN expanded (1 → 10 stations within 200km) |
| — | IEM expanded (50mi → 200km radius) |
| — | SKF set to dormant with research logging |
| — | BOA wired into live inference (was disconnected) |
| — | All thresholds tunable via inference_config.json |
| — | Config hot-reload (no restart needed) |

### 🔄 IN PROGRESS

| Ticket | Description | Remaining |
|--------|-------------|-----------|
| T1.4 | Source trust multipliers | Validate stability across windows (9/15 days covered) |
| T1.5 | BOA online source weighting | Surface weights in canonical replay; prove CRPS gain |

### ⏳ NOT STARTED (Prioritized)

| Priority | Ticket | Description |
|----------|--------|-------------|
| P1 | T1.6 | Semi-local EMOS + station embeddings (multi-station cold-start) |
| P1 | T6.1 | Day-ahead / intraday mode separation |
| P1 | T6.2 | Adaptive Kelly + edge auto-tuning |
| P1 | T6.3 | Cross-station portfolio risk |
| P1 | TA8.4 | MRMS radar for outflow detection |
| P1 | TA8.5 | RTMA/URMA ingestion (analyzed cloud + temp + wind) |
| P2 | T1.3 | Optional BMA for multimodal cases |
| P2 | T3.4 | Self-tuning IMM-KF for sparse regions |
| P2 | T5.1 | Sticky HDP-HMM offline regime discovery |
| P2 | TA8.6 | Radiosonde 72202 (Miami 00Z/12Z soundings) |
| P2 | TA8.7 | GOES-19 GLM lightning |
| P2 | — | Open-Meteo cost optimization (migrate CAPE/PW/cloud to HRRR) |
| P3 | TC1 | Sticky HDP-HMM offline |
| P3 | TC2 | MoE / dynamic gating shadow |
| P3 | TC4 | DS3M shadow challenger |
| P3 | TC5 | DMA shadow challenger |
| P3 | TC6 | Conformal prediction overlay |
| P3 | TA8.8 | HRRR + obs archiving for Analog Ensemble (12-18 month investment) |

---

## 9. Option C — Shadow / R&D Track

| Component | Implementation | Data Needs | Priority |
|-----------|---------------|------------|----------|
| **Conformal Prediction** | Calibration wrapper on EMOS | Current data sufficient | TC6 — deploy as check |
| **DMA** | ~100 lines Python, 2 forgetting factors | Current data sufficient | TC5 — shadow challenger |
| **Sticky HDP-HMM** | `pyhsmm` library | 6+ months of data | TC1 — regime discovery |
| **MoE gating** | Dense 2-4 expert gating | 100-200 days | TC2 — deferred |
| **DS3M** | PyTorch, variational inference | 1+ year ideally | TC4 — long-term |
| **Recurrent Sticky HDP-HMM** | Custom Gibbs sampler | 2+ years | — — premature |

---

## 10. Evaluation & Promotion

### Canonical Metrics
- Remaining-high/low MAE, RMSE, CRPS
- Bracket Brier score and log loss
- Calibration reliability curves
- Sharpness/spread diagnostics
- EV after fees
- Realized paper-trade quality
- Turnover, drawdown, concentration

### Promotion Gates
Components advance through: **replay → paper → live shadow → partial live → full live**

### Kill Conditions
- Calibration collapse
- Severe overtrading
- Unstable day-to-day outputs
- Dependence on features not reliably available live
- Unexplained late-day decision quality degradation

---

## 11. Non-Negotiable Rules

1. **Option A is live. Option C is shadow.** No collapse.
2. **Option C does not place trades** without explicit promotion.
3. **One canonical replay/evaluation harness** for all comparisons.
4. **Orchestrator / trader / executor remain separate.**
5. **AI review is advisory, not self-authorizing.**
6. **Model scoring is separate for highs vs lows** (memory: feedback_separate_high_low_scoring.md).
7. **Wethr envelope is NWS-verified** — never apply ASOS rounding floor to running extremes (memory: feedback_wethr_envelope.md).
8. **Wethr SSE must be a standalone relay service** for multi-city scaling (memory: project_wethr_relay_architecture.md).
9. **Only FINAL CLI settlements** go into `event_settlements`. DSM/preliminary data updates LiveState running extremes only. CLI ingestion is gated by `now > climate_day_end (05:00Z D+1)`.
10. **EMOS calibrates remaining move**, not absolute forecasts. Forward curves handle time-of-day effects; EMOS corrects only the residual bias in the forward curve's remaining-move prediction.

---

## 12. Key Artifacts

| Artifact | Path | Purpose |
|----------|------|---------|
| EMOS coefficients | `analysis_data/emos_state.json` | Fitted EMOS a/b/c/d for high + low |
| BOA weights | `analysis_data/boa_state.json` | Per-source sleeping expert weights |
| Source trust state | `analysis_data/source_trust_state.json` | Static family multipliers (T1.4) |
| Inference config | `analysis_data/inference_config.json` | All tunable thresholds |
| Sigma climatology | `analysis_data/sigma_clim_KMIA_m*.json` | IEM ASOS 6yr empirical σ_remaining |
| Drift report | `analysis_data/drift_report.json` | PSI feature distribution drift |
| P&L regime | `analysis_data/pnl_regime.json` | Trading performance regime state |
| Replay bundle | `analysis_data/canonical_replay_bundle.json` | Canonical evaluation results |
| BOA audit | `analysis_data/boa_family_weight_audit.md` | Evidence: which sources BOA favors |

---

## 13. Debugging Quickstart

Run from `miami-project/` root.

```bash
# Full test suite
./.venv/bin/python -m pytest tests/ -x --tb=short

# EMOS fit
PYTHONPATH=src ./.venv/bin/python -m engine.emos --db miami_collector.db --station KMIA

# BOA initialization from history
PYTHONPATH=src ./.venv/bin/python -m engine.boa --db miami_collector.db --station KMIA

# Canonical replay
PYTHONPATH=src ./.venv/bin/python -m analyzer.canonical_replay_bundle \
  --db miami_collector.db --station KMIA --include-changepoint-compare \
  --out analysis_data/canonical_replay_bundle.json \
  --md-out analysis_data/canonical_replay_bundle.md

# Source trust refresh
PYTHONPATH=src ./.venv/bin/python -m analyzer.source_trust_refresh \
  --db miami_collector.db --station KMIA --lookback-days 45

# Paper trader smoke test
PYTHONPATH=src ./.venv/bin/python -m trading.paper_trader \
  --db miami_collector.db --station KMIA --max-quote-age-minutes 20

# Archive audit
PYTHONPATH=src ./.venv/bin/python -m analyzer.archive_audit \
  --db miami_collector.db --station KMIA
```

---

## 14. Test Suite

**162 tests, all passing** (0.68s as of 2026-03-22).

Key test files for new modules:
- `test_emos.py` — CRPS closed-form, fitting, serialization, bounds
- `test_boa.py` — Sleeping expert protocol, weight normalization, CRPS updates
- `test_letkf.py` — Gaspari-Cohn, haversine, initialization, multi-obs update
- `test_cloud_cover.py` — Sky codes, radiation model, clearness index
- `test_regime_catalog.py` — Regime classification, conditioning parameters
