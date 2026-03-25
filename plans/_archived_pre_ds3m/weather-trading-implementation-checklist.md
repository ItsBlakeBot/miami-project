# Weather Trading Implementation Checklist

Updated: 2026-03-22
Reference: `plans/weather-trading-architecture-roadmap.md`

## Progress snapshot (as of 2026-03-22)

Completed modules/tests so far:
- `miami-project/src/analyzer/archive_audit.py`
- `miami-project/src/engine/replay_context.py`
- `miami-project/src/engine/dynamic_weights.py`
- `miami-project/src/engine/bocpd.py`
- `miami-project/src/engine/changepoint_detector.py`
- `miami-project/src/engine/source_trust.py`
- `miami-project/src/analyzer/source_trust_backfill.py`
- `miami-project/src/analyzer/source_trust_refresh.py`
- `miami-project/src/analyzer/changepoint_replay_compare.py`
- `miami-project/src/analyzer/canonical_replay_bundle.py`
- `miami-project/tests/test_archive_audit.py`
- `miami-project/tests/test_replay_context.py`
- `miami-project/tests/test_orchestrator_replay_guards.py`
- `miami-project/tests/test_dynamic_weights.py`
- `miami-project/tests/test_baseline_engine_dynamic_weighting.py`
- `miami-project/tests/test_bocpd.py`
- `miami-project/tests/test_changepoint_bocpd.py`
- `miami-project/tests/test_source_trust_backfill.py`
- `miami-project/tests/test_source_trust_refresh.py`
- `miami-project/tests/test_source_trust.py`
- `miami-project/tests/test_changepoint_compare.py`
- `miami-project/tests/test_canonical_replay_bundle.py`
- `miami-project/tests/test_replay_remaining_backtest_metrics.py`

Generated outputs:
- `miami-project/analysis_data/archive_audit_report.{md,json}`
- `miami-project/analysis_data/source_trust_backfill.{md,json}`
- `miami-project/analysis_data/source_trust_refresh.json`
- `miami-project/analysis_data/source_trust_state.json`
- `miami-project/analysis_data/changepoint_compare.{md,json}`
- `miami-project/analysis_data/canonical_replay_bundle.{md,json}`

## How to use this file

This is the execution checklist for the weather trading roadmap.
Use it when implementing work so future agents can quickly see:
- what track a task belongs to,
- what has already been completed,
- what must happen before promotion,
- what must remain shadow-only.

For debugging commands and log/report locations, use:
- `plans/weather-trading-architecture-roadmap.md` → sections:
  - **Logging, audit, and debugging locations**
  - **Debugging quickstart (copy/paste)**

---

## Global guardrails

- [ ] **Preserve Option A / Option C separation**
- [ ] **Do not give Option C live execution authority**
- [ ] **Preserve orchestrator vs trader vs executor separation**
- [ ] **Use the canonical replay/evaluation harness for all comparisons**
- [ ] **Use forward-only historical data in replay**
- [ ] **Keep AI review advisory, not self-authorizing**

---

# Shared Foundation

## A0 / C0 shared substrate

### Archive + schema
- [x] Confirm one canonical archive for forecasts, obs, nearby stations, FAWN/mesonet, buoys, cloud proxies, quotes, settlements
- [ ] Confirm schema can reconstruct a historical eval timestamp without leakage
- [x] Confirm quote timestamps and freshness metadata are always stored
- [ ] Confirm remaining-high / remaining-low targets are derivable historically

### Replay harness
- [x] One canonical replay entry point exists (`analyzer/canonical_replay_bundle.py`)
- [x] Historical replay supports arbitrary eval timestamps
- [x] Replay obeys data availability and freshness constraints
- [ ] Replay can reproduce paper-style decision flow

### Evaluation metrics
- [x] Remaining-high MAE / RMSE / CRPS
- [x] Remaining-low MAE / RMSE / CRPS
- [x] Bracket Brier score
- [x] Bracket log loss
- [x] Calibration curves / reliability tables
- [x] Sharpness / spread diagnostics
- [x] EV after fees
- [x] Paper-trade quality metrics
- [ ] Turnover / drawdown / concentration diagnostics

### Reporting
- [x] One canonical report format exists
- [x] Per-regime breakdowns exist
- [x] Per-time-to-settlement breakdowns exist
- [x] Worst-day failure analysis exists

### Source trust + sparse-data policy
- [x] Interim source conflict fallback order documented
- [x] Interim sparse-window guardrail documented
- [x] Implement source/family trust multipliers from realized skill artifact
- [x] Add guardrails for trust auto-tuning (minimum sample sizes, bounded daily multiplier change)
- [x] Ensure single-source full-authority is blocked in sparse windows

### Backfill
- [x] Minimum useful backfill scope documented in roadmap
- [x] Build backfill extraction job
- [x] Emit per-source/per-family MAE/RMSE/bias + reliability by time-to-settlement
- [x] Wire backfill outputs into dynamic weighting priors

---

# Option A — Live / Production Checklist

## A1 — Calibration Engine

### A1a — Dynamic aggregation
- [x] Implement dynamic source weighting
- [x] Condition weights on freshness
- [x] Condition weights on time-to-settlement
- [x] Condition weights on recent rolling skill
- [x] Condition weights on regime probabilities / context

### A1b — EMOS calibration
- [ ] Add standard EMOS calibration for remaining-high distribution
- [ ] Add standard EMOS calibration for remaining-low distribution
- [ ] EMOS fit via CRPS minimization (scipy)
- [ ] Rolling 30-40 day training window
- [ ] Replay against current baseline
- [ ] Verify CRPS/log-score improvement
- [ ] Verify calibration is not degraded
- [ ] SAR-SEMOS upgrade: autoregressive error + seasonal Fourier terms
- [ ] EMOS-GB upgrade: gradient-boosted for mixture distributions

### A1c — BMA branch
- [ ] Add optional BMA branch for multimodal disagreement cases
- [ ] Define "multimodal enough" trigger condition

### A1d — BOA online source weighting
- [ ] Implement BOA core (per-source weight vector, CRPS-based update)
- [ ] Seed initial weights from T1.4 trust multipliers
- [ ] Emit source weight history for auditing
- [ ] BOA weights appear in canonical replay bundle
- [ ] BOA combination beats static/rolling weighting in CRPS

### A1e — Semi-local EMOS + station embeddings
- [ ] Implement semi-local EMOS (Lerch & Baran 2017)
- [ ] Define climatological similarity metric (climate zone, latitude, coast distance, elevation)
- [ ] Cold-start pipeline: regional → blended → local
- [ ] New station achieves competitive calibration within 2 weeks
- [ ] Optional: neural EMOS with station embeddings (Rasp & Lerch 2018)

## A2 — BOCPD online break layer
- [x] Add BOCPD implementation
- [x] Output break probability
- [x] Output run length / time-since-last-break
- [x] Integrate BOCPD into inference outputs
- [ ] Keep CUSUM as diagnostic / backup only during transition
- [x] Compare BOCPD vs current CUSUM behavior on replay
- [ ] Verify fewer junk flips / better timing on real regime changes

## A3 — Regional cluster LETKF + self-tuning IMM-KF

### A3a — Cluster design
- [ ] Define geographic clusters and membership
- [ ] Define state vector per cluster (surface temp, dew, pressure per station)
- [ ] Define trading targets vs supporting obs per cluster
- [ ] Define localization radius (50-100km)
- [ ] Define fallback rule (<3 supporting stations → IMM-KF)
- [ ] Define background ensemble source (GEFS 31 or HRRR time-lagged)

### A3b — Observation operators
- [ ] Build observation operators by source family (ASOS, FAWN, NDBC, METAR)
- [ ] Define per-source observation error assumptions (R matrix)
- [ ] Handle multi-cadence observation ingestion
- [ ] Connect to BOA/trust weights

### A3c — LETKF implementation
- [ ] Implement LETKF core (Hunt et al. 2007)
- [ ] Add localization (Gaspari-Cohn function)
- [ ] Add covariance inflation (RTPS method)
- [ ] Ensemble size: 20-50 members
- [ ] Replay on historical eval windows
- [ ] Validate skill vs feature-only baselines
- [ ] Validate numerical stability (no covariance explosions)
- [ ] Demonstrate cross-station information propagation

### A3d — Self-tuning IMM-KF
- [ ] Implement innovation-based adaptive estimation (Sage-Husa method)
- [ ] Auto-tune R (measurement noise) from innovation covariance
- [ ] Auto-tune Q (process noise)
- [ ] Replay shows equal or better skill than hand-tuned parameters
- [ ] No divergence in extended replay windows

## A4 — Small live regime model
- [ ] Define interpretable live regime catalog
- [ ] Add regime inference using BOCPD + state summaries + residual features
- [ ] Output regime probabilities online
- [ ] Condition forecast aggregation on regime state
- [ ] Condition calibration on regime when appropriate
- [ ] Evaluate regime-conditioned skill vs unconditional skill
- [ ] Evaluate regime transition stability

## A5 — Sticky HDP-HMM offline regime review loop
- [ ] Build offline sticky HDP-HMM experiment pipeline (pyhsmm)
- [ ] Learn candidate regime counts from historical data
- [ ] Learn transition structure from historical data
- [ ] Summarize candidate regimes for AI review
- [ ] Summarize candidate regimes for human review
- [ ] Explicitly approve / reject live regime catalog updates
- [ ] Document accepted changes to live regime catalog

## A6 — Trading architecture upgrades

### A6a — Day-ahead / intraday separation
- [ ] Day-ahead mode: trigger on fresh model cycles, conservative Kelly, entry-fee-only positions
- [ ] Intraday mode: obs-triggered, increasing Kelly, higher edge threshold for round-trips
- [ ] Separate edge thresholds per mode
- [ ] Paper trading tracks entries by mode
- [ ] Fee accounting reflects mode-specific costs

### A6b — Adaptive Kelly + edge auto-tuning
- [ ] Implement adaptive Kelly fraction (start 0.15x)
- [ ] Baker-McHale shrinkage correction
- [ ] Drawdown constraint: f ≤ 2 × S² × D_max
- [ ] Multiple parallel Kelly fractions with multiplicative weights
- [ ] Edge threshold auto-adjustment (widen on losses, tighten on wins)
- [ ] Floor at fee_cost × 2, cap at 25%
- [ ] Parameter history logged for auditing

### A6c — Cross-station portfolio risk
- [ ] Maximum portfolio heat: 15-25% of bankroll
- [ ] Per-station limit: 5-10% per station-day
- [ ] Correlation-adjusted exposure for same-cluster stations
- [ ] Daily loss circuit breaker (3% of bankroll)
- [ ] Drawdown scaling: 50% at 10%, 75% at 20%, stop at 30%
- [ ] Correlation matrix from forecast error correlations (not raw temp)

### A6d — Cross-bracket arbitrage detection
- [ ] Monitor bracket probability sum violations
- [ ] Log opportunities even if not traded

## A7 — Automated health monitoring

### A7a — Rolling quality monitors
- [ ] Rolling 30-day Brier score per station, per source
- [ ] Alert at +0.02 from baseline
- [ ] Per-signal Brier decomposition (calibration + resolution)
- [ ] Per-source CRPS tracking
- [ ] Weekly summary artifact

### A7b — Trading performance regime detection
- [ ] BOCPD on daily P&L series
- [ ] Auto-scale rules on regime shift detection
- [ ] Rolling 30-day Sharpe tracking
- [ ] Regime shift events logged

### A7c — Concept drift detection
- [ ] PSI on feature distributions
- [ ] NWS model upgrade monitoring
- [ ] Weekly auto-recalibration trigger
- [ ] Seasonal drift handling

### A7d — Signal deprecation
- [ ] Per-signal contribution tracking (winning vs losing trades)
- [ ] Auto-downweight after 2+ weeks below noise floor
- [ ] Weekly audit log

## A8 — New data source ingestion

### A8a — Cloud coverage
- [ ] Parse METAR cloud obs from NWS feed (SKC/FEW/SCT/BKN/OVC → fraction)
- [ ] Use FAWN solar radiation as cloud proxy (already in DB)
- [ ] Add Open-Meteo cloud cover fields (if keeping) OR HRRR TCDC via Herbie
- [ ] GOES-19 Clear Sky Mask from AWS S3 `noaa-goes19` (highest quality, most effort)
- [ ] RTMA analyzed cloud cover via Herbie `noaa-rtma-pds` (blended analysis, practical)
- [ ] Cloud cover integrated into high-temperature forecasting

### A8b — Open-Meteo cost optimization
- [ ] Drop unused fields (lifted_index, boundary_layer_height, soil_moisture, direct/diffuse radiation)
- [ ] Build HRRR adapter via Herbie + AWS S3
- [ ] Replace Open-Meteo atmospheric (CAPE, PW) with HRRR
- [ ] Evaluate: keep Open-Meteo for ensembles only vs. full replacement
- [ ] Cost reduction quantified

### A8c — Expanded surface observation network
- [ ] Expand IEM queries to more network types
- [ ] Add NYSM (NY State Mesonet, 127 stations) for NE Corridor
- [ ] Add NJWXNET for KPHL/KNYC support
- [ ] Add TexMesonet for Texas stations
- [ ] Consider MADIS registration for broadest coverage
- [ ] Achieve 3-5 supporting stations per trading station

### A8d — MRMS radar for outflow detection
- [ ] Ingest MRMS composite reflectivity from AWS S3 (noaa-mrms-pds)
- [ ] 50km radius around trading stations
- [ ] Outflow early warning integrated into signal engine

### A8e — HRRR + obs archiving for Analog Ensemble
- [ ] Archive HRRR forecasts paired with station observations
- [ ] Systematic storage for all trading stations
- [ ] After 12-18 months: implement NCAR AnEn method

## A9 — Rollout
- [ ] Replay promotion complete
- [ ] Paper promotion complete
- [ ] Live shadow promotion complete
- [ ] Partial live promotion complete
- [ ] Full live promotion complete

### Promotion checks
- [ ] Remaining-target skill improved
- [ ] Bracket calibration improved or unchanged
- [ ] EV after fees improved
- [ ] Turnover remains acceptable
- [ ] Drawdown remains acceptable
- [ ] Operational stability verified

---

# Option C — Shadow / R&D Checklist

## C1 — Advanced offline regime discovery
- [ ] Run sticky HDP-HMM shadow experiments (pyhsmm)
- [ ] Compare discovered states to live regime catalog
- [ ] Evaluate transition asymmetry / persistence
- [ ] Test recurrent sticky HDP-HMM only if justified
- [ ] Record proposals without changing live behavior

## C2 — Mixture-of-experts / dynamic gating shadow
- [ ] Build shadow gating layer conditioned on state / regime / time-to-settlement
- [ ] Compare shadow gating vs Option A BOA weighting
- [ ] Test calibration behavior under shadow gating
- [ ] Test overfit sensitivity across windows/regimes

## C3 — DS3M shadow challenger
- [ ] Confirm archival dataset is sufficient for DS3M training
- [ ] Define DS3M training targets
- [ ] Build DS3M shadow training pipeline
- [ ] Build DS3M shadow inference pipeline
- [ ] Score DS3M on canonical replay windows
- [ ] Compare DS3M against Option A on proper scoring rules

## C4 — DMA shadow challenger
- [ ] Implement Dynamic Model Averaging (Raftery et al. 2010)
- [ ] Compare DMA vs BOA on replay
- [ ] If DMA can't beat BOA, DS3M needs to do much better

## C5 — Conformal prediction overlay
- [ ] Implement Adaptive Conformal Inference (ACI) for bracket coverage
- [ ] Use as calibration diagnostic on EMOS outputs
- [ ] Compare coverage guarantees vs parametric EMOS intervals

## C6 — Shadow evaluation
- [ ] Full-sample evaluation exists
- [ ] Rolling-window evaluation exists
- [ ] Per-regime evaluation exists
- [ ] Per-time-to-settlement evaluation exists
- [ ] Worst-day analysis exists
- [ ] Operational stability notes exist

## C7 — Promotion gate for shadow challengers
- [ ] Challenger shows better proper scoring rules
- [ ] Challenger preserves or improves calibration
- [ ] Challenger improves EV after fees
- [ ] Challenger does not materially worsen turnover / drawdown
- [ ] Challenger survives extended shadow window (target: 30-60 climate days)
- [ ] Explicit approval granted before live integration

---

# Cross-Track Evaluation Checklist

## Head-to-head requirements
- [ ] Same replay window
- [ ] Same data availability constraints
- [ ] Same freshness constraints
- [ ] Same downstream trade policy assumptions

## Required comparison tables
- [ ] Option A vs current baseline
- [ ] Option C vs Option A
- [ ] Option C vs current baseline
- [ ] Per-regime table
- [ ] Per-time-to-settlement table
- [ ] Best days / worst days table

## Kill conditions
- [ ] Calibration collapse flagged
- [ ] Severe overtrading flagged
- [ ] Hidden data leakage flagged
- [ ] Feature availability mismatch flagged
- [ ] Excess instability flagged

---

# Completion marker

This checklist is only complete when:
- Option A is live and stable,
- Option C is running as a disciplined shadow challenger,
- promotion between them is evidence-based rather than vibe-based.
