# Weather Trading Ticketized Build Order

Updated: 2026-03-22
Reference docs:
- `plans/weather-trading-architecture-roadmap.md`
- `plans/weather-trading-implementation-checklist.md`

## Purpose

This file converts the architecture roadmap into concrete build tickets.
Future agents should use this file when deciding what to implement next.

## Status legend
- `NOT_STARTED`
- `IN_PROGRESS`
- `DONE`
- `BLOCKED`

## Current implementation status (as of 2026-03-22)
- `T0.1` → **DONE** (archive audit module + reports)
- `T0.2` → **DONE** (canonical replay bundle is the authoritative comparison/report path)
- `T0.3` → **DONE** (canonical bundle includes remaining-target MAE/RMSE/CRPS, TTL/per-regime cuts, trade-quality cuts, EV rollups, worst-day diagnostics)
- `T0.4` → **DONE** (extractor + metrics artifact + weighting prior integration; 9224 records)
- `T1.1` → **DONE** (dynamic weighting module + baseline integration + tests)
- `T1.4` → **IN_PROGRESS** (guardrails + rolling refresh implemented; validation still sample-limited at 9/15 days)
- `T1.2` → **DONE** (EMOS core module + orchestrator integration + CLI; fitted on 8 settlement days)
- `T1.5` → **IN_PROGRESS** (BOA core + state tracking implemented; replay/canonical integration and promotion gates still pending)
- `TA8.1` → **DONE** (cloud cover from METAR sky codes + FAWN solar clearness index; wired into orchestrator)
- `TA8.2` → **DONE** (HRRR adapter via Herbie; CAPE/PW/cloud/radiation/upper-air; collection loop in runner.py)
- FAWN expanded: 1 → 10 stations within 200km (config.toml + multi-station client)
- IEM expanded: 50mi → 125mi radius (~200km)
- `T2.1` → **DONE** (BOCPD + run-length outputs wired; replay comparison bundle generated)

## Pause marker (resume next session — updated 2026-03-22)

### Completed this session (major infra lift; status reconciled)
Core Sprint 0-3 foundations plus major Sprint 6-8 modules landed.
P1 calibration/weighting tickets `T1.4` and `T1.5` remain in-progress pending final validation/integration gates.

### What's running live
- Runner has 17 async collection tasks (incl. HRRR + GOES-19 satellite)
- EMOS coefficients fitted and loaded by orchestrator
- BOA state initialized via sleeping expert protocol (offline path currently authoritative)
- LETKF initialized for SE Florida cluster (14 stations)
- Cloud cover (METAR + FAWN solar) wired into orchestrator changepoint detector
- Daily recalibration pipeline: drift → EMOS refit → BOA update → regime review → P&L → health

### New modules created this session
- `src/engine/emos.py` — EMOS calibration
- `src/engine/boa.py` — BOA with sleeping experts
- `src/engine/letkf.py` — Regional cluster LETKF
- `src/engine/cloud_cover.py` — Cloud from METAR + FAWN solar
- `src/engine/regime_catalog.py` — Live 6-regime catalog
- `src/engine/observation_operators.py` — Formal obs operators for LETKF
- `src/engine/bracket_arbitrage.py` — Cross-bracket arb monitor
- `src/collector/sources/hrrr.py` — HRRR via Herbie + AWS S3
- `src/collector/sources/goes.py` — GOES-19 satellite (live CAPE + cloud)
- `src/analyzer/health_monitor.py` — Rolling quality monitors
- `src/analyzer/pnl_regime_detector.py` — BOCPD on P&L
- `src/analyzer/drift_detector.py` — PSI-based concept drift
- `src/analyzer/signal_deprecation.py` — Signal contribution tracking
- `src/analyzer/anen_archiver.py` — Analog Ensemble data archiving
- `src/analyzer/auto_recalibrator.py` — Full daily recalibration pipeline

### Test suite: 162/162 passing

### What the next agent should do
1. **Finish T1.4 validation gate** — run rolling refresh stability checks across windows/anchors; explicitly confirm or defer DONE based on sample-depth criteria
2. **Finish T1.5 integration gate** — wire BOA weights/history into canonical replay outputs and verify replay CRPS improvement vs static/rolling weighting
3. **Then T4.2** — wire regime_catalog.py into orchestrator (condition sigma/mu on regime state)
4. **Then TA8.3/TA8.4** — surface network expansion + MRMS adapter
5. Keep Option C in shadow until Option A promotion gates pass

### Constraints for next agent
- Preserve **Option A live / Option C shadow** separation
- Do not modify the daily recalibration pipeline without testing end-to-end
- Do not create a second comparison harness (use canonical_replay_bundle)
- All regime catalog changes must go through the proposal → AI review → human approval flow
- Regime unrecognized threshold is data-driven (10th percentile of historical confidence), NOT hardcoded

Primary artifacts currently in repo:
- `/Users/blakebot/blakebot/miami-project/src/analyzer/archive_audit.py`
- `/Users/blakebot/blakebot/miami-project/src/engine/replay_context.py`
- `/Users/blakebot/blakebot/miami-project/src/engine/dynamic_weights.py`
- `/Users/blakebot/blakebot/miami-project/src/engine/bocpd.py`
- `/Users/blakebot/blakebot/miami-project/src/engine/changepoint_detector.py`
- `/Users/blakebot/blakebot/miami-project/src/analyzer/source_trust_backfill.py`
- `/Users/blakebot/blakebot/miami-project/src/analyzer/source_trust_refresh.py`
- `/Users/blakebot/blakebot/miami-project/src/analyzer/changepoint_replay_compare.py`
- `/Users/blakebot/blakebot/miami-project/src/engine/source_trust.py`
- `/Users/blakebot/blakebot/miami-project/src/analyzer/canonical_replay_bundle.py`
- `/Users/blakebot/blakebot/miami-project/analysis_data/archive_audit_report.{md,json}`
- `/Users/blakebot/blakebot/miami-project/analysis_data/source_trust_backfill.{json,md}`
- `/Users/blakebot/blakebot/miami-project/analysis_data/source_trust_refresh.json`
- `/Users/blakebot/blakebot/miami-project/analysis_data/source_trust_state.json`
- `/Users/blakebot/blakebot/miami-project/analysis_data/changepoint_compare.{json,md}`
- `/Users/blakebot/blakebot/miami-project/analysis_data/canonical_replay_bundle.{json,md}`
- `/Users/blakebot/blakebot/miami-project/analysis_data/boa_family_weight_audit.md`

---

# Priority Model

- **P0** = foundational / blocking
- **P1** = high-value production work
- **P2** = important but not blocking
- **P3** = shadow / advanced research / later promotion work

---

# Sprint 0 — Foundation and Alignment

## T0.1 — Canonical archive audit  **[DONE]**
**Priority:** P0 | **Track:** Shared | **Owner:** `miami-project`

### Goal
Verify that the current archive can reconstruct historical inference with no leakage.

### Done when
- archive gap report exists; missing fields listed; blockers for replay known

---

## T0.2 — Canonical replay harness hardening  **[DONE]**
**Priority:** P0 | **Track:** Shared | **Owner:** `miami-project`

### Goal
Create one authoritative replay path used by all contenders.

### Done when
- one replay entry point used for model comparison; no parallel hidden replay path

---

## T0.3 — Canonical metric/report package  **[DONE]**
**Priority:** P0 | **Track:** Shared | **Owner:** `miami-project`

### Goal
Standardize comparison outputs (CRPS, Brier, reliability, EV, worst-day diagnostics).

### Done when
- every serious experiment can emit the same report bundle

---

## T0.4 — Source-trust backfill bootstrap  **[DONE]**
**Priority:** P0 | **Track:** Shared | **Owner:** `miami-project`

### Goal
Backfill minimum historical source performance to seed source trust convergence.

### Done when
- reusable backfill artifact exists and seeds source-trust priors

---

# Sprint 1 — Calibration Engine (Highest ROI)

## T1.1 — Dynamic weighting engine  **[DONE]**
**Priority:** P1 | **Track:** Option A | **Owner:** `miami-project`

### Goal
Replace static weighting with context-aware dynamic source weights.

---

## T1.2 — EMOS calibration layer  **[DONE]**
**Priority:** P1 | **Track:** Option A | **Owner:** `miami-project`

### Goal
Calibrate remaining-high / remaining-low predictive distributions using EMOS.

### Tasks
- implement standard EMOS (Gneiting et al. 2005) for remaining-high
  - Normal distribution: mean = a + b₁f₁ + b₂f₂ + ..., variance = c + d·S²
  - Fit via CRPS minimization (scipy.optimize.minimize or gradient descent)
  - Training window: rolling 30-40 climate days minimum
- implement standard EMOS for remaining-low
- add fit/eval path in canonical replay bundle
- expose calibrated distribution outputs downstream (mu_emos, sigma_emos)
- document: EMOS operates on remaining-target (not raw high/low), using source consensus as predictors

### Progression path (implement sequentially after standard EMOS proves out)
1. **Standard EMOS** → validate CRPS improvement
2. **SAR-SEMOS** (Jobst et al. 2024) → add autoregressive error + seasonal Fourier terms
3. **EMOS-GB** → gradient-boosted EMOS for mixture distributions + automatic feature selection

### Depends on
- T1.1

### Done when
- EMOS distributions are replayed and scored
- calibration report beats or matches current system

---

## T1.3 — Optional BMA branch  **[NOT_STARTED]**
**Priority:** P2 | **Track:** Option A | **Owner:** `miami-project`

### Goal
Support multimodal disagreement cases without forcing BMA as default.

### Depends on
- T1.2

---

## T1.4 — Adaptive source trust multipliers  **[IN_PROGRESS]**
**Priority:** P1 | **Track:** Option A | **Owner:** `miami-project`

### Goal
Auto-tune source/family trust from realized skill with conservative sparse-data guardrails.

### Progress note (2026-03-22)
- Backfill-prior loader, BaselineEngine integration, persistent state, max-step guardrail, sparse-window caps, and rolling refresh runner are implemented.
- Real-data refresh: 45-day lookback → 9 covered days (`sufficient_days=false`).
- Remaining gate to mark DONE: validate multiplier stability across multiple anchors/windows and define explicit minimum evidence threshold.

---

## T1.5 — BOA online source weighting  **[IN_PROGRESS]**
**Priority:** P1 | **Track:** Option A | **Owner:** `miami-project`

### Goal
Replace static/rolling source trust with optimal CRPS-based online learning via Bernstein Online Aggregation.

### Tasks
- implement BOA core: maintain per-source weight vector, update after each settlement via CRPS loss
- BOA is exponentially concave with logarithmic regret → provably converges to best fixed combination
- use T1.4 trust multipliers as initial prior weights; BOA takes over ongoing learning
- emit source weight history for auditing
- add BOA weights to canonical replay bundle output

### Implementation notes
- Reference: Berrisch et al. (2023) — CRPS-based online learning for probabilistic forecast combination
- Each source produces Normal(mu_source, sigma_source); BOA learns optimal combination weights
- Update rule: w_t+1(k) = w_t(k) · exp(-η · CRPS_t(k)) / Σ_j w_t(j) · exp(-η · CRPS_t(j))
- Learning rate η can be adaptive (decreasing schedule or fixed small value)
- Logarithmic regret bound means this is provably near-optimal

### Current status clarification (2026-03-22)
- BOA core and persisted state (`analysis_data/boa_state.json`) exist and show side-specific behavior.
- Low-side currently favors openmeteo overall; high-side currently favors wethr MOS experts.
- Remaining gate to mark DONE: surface BOA weights/history in canonical replay outputs and show replay CRPS gain vs static/rolling baseline.

### Depends on
- T1.2 (EMOS produces the calibrated distributions that BOA combines)

### Done when
- BOA weights running in replay
- BOA combination beats static/rolling weighting in CRPS on replay
- weight history auditable

---

## T1.6 — Semi-local EMOS with station embeddings  **[NOT_STARTED]**
**Priority:** P1 | **Track:** Option A | **Owner:** `miami-project`

### Goal
Enable fast multi-station scaling by training EMOS across climatologically similar stations.

### Tasks
- implement semi-local EMOS (Lerch & Baran 2017): augment training data for target station with data from similar stations
- define "climatologically similar" metric: same climate zone, similar latitude, similar distance-to-coast, similar elevation
- cold-start pipeline:
  - Week 1-2: use regional EMOS trained on similar stations
  - Week 3-6: blend regional + local data via BOA weights
  - Month 2+: local model dominates, regional provides regularization
- optional: neural EMOS with station identity embeddings (Rasp & Lerch 2018) — station embedding approach where new station initializes from nearest similar station

### Depends on
- T1.2, T1.5

### Done when
- new station achieves competitive calibration within 2 weeks of data collection
- semi-local EMOS CRPS is not worse than fully-local EMOS on established stations

---

# Sprint 2 — Changepoint Detection

## T2.1 — BOCPD prototype  **[DONE]**
**Priority:** P1 | **Track:** Option A | **Owner:** `miami-project`

### Goal
Upgrade online break detection from raw CUSUM to probabilistic changepoint detection.

---

## T2.2 — CUSUM downgrade to backup/diagnostic  **[DONE]**
**Priority:** P2 | **Track:** Option A | **Owner:** `miami-project`

### Goal
Keep CUSUM as diagnostic and fallback, not the main online break engine.

### Depends on
- T2.1

---

## T2.3 — SKF evaluation for downgrade to backup  **[DONE]**
**Priority:** P2 | **Track:** Option A | **Owner:** `miami-project`

### Goal
Evaluate the Switching Kalman Filter for downgrade to backup/diagnostic role, similar to CUSUM.
The SKF will be superseded by the regional LETKF (T3.3) for dense clusters and by the
self-tuning IMM-KF (T3.4) for sparse regions.

### Rationale
The current SKF (5D state: temp, dew, pressure, wind_sin, wind_cos) operates per-station
only. It cannot propagate information across stations (e.g., sea breeze front seen at KFLL
before KMIA). The regional LETKF addresses this limitation. However, the SKF should not be
removed until the LETKF proves itself in replay — it currently provides value for regime
detection and should remain as a fallback.

### Tasks
- evaluate LETKF vs SKF on replay once T3.3 is complete
- if LETKF outperforms: demote SKF to diagnostic/backup layer
- if LETKF fails: keep SKF as primary per-station filter
- document transition criteria and fallback procedure
- keep SKF logging for comparison (same approach as CUSUM → BOCPD transition)

### Depends on
- T3.3 (LETKF implementation)
- T3.4 (self-tuning IMM-KF)

### Done when
- replay comparison report exists (LETKF vs SKF)
- transition decision is explicit and documented

---

# Sprint 3 — State Assimilation (Regional Cluster LETKF + Self-Tuning IMM-KF)

## T3.1 — Regional cluster LETKF design  **[DONE]**
**Priority:** P1 | **Track:** Option A | **Owner:** `miami-project`

### Goal
Design the regional cluster architecture for spatial state assimilation across trading stations.

### Tasks
- define geographic clusters and membership:
  - SE Florida: KMIA, KFLL + FAWN + NDBC buoys + IEM nearby
  - NE Corridor: KNYC, KPHL, KBOS, KDCA + IEM nearby + NYSM
  - Texas Triangle: KAUS, KSAT, KHOU, KDFW + IEM nearby
  - Upper Midwest: KMDW, KMSP + IEM nearby
  - Mountain/Desert: KDEN, KPHX, KLAS (sparse → IMM-KF fallback)
  - West Coast: KLAX, KSFO, KSEA (sparse → IMM-KF fallback)
- define state vector per cluster: surface temp, dew, pressure at each member station
- define which stations are trading targets vs. supporting obs
- define localization radius: 50-100km (Gaspari-Cohn function)
- define fallback rule: <3 supporting stations → use IMM-KF
- define background ensemble: GEFS 31-member or HRRR time-lagged ensemble
- document observation operators by source family:
  - ASOS: direct measurement, σ_obs ≈ 0.5°F
  - FAWN: direct measurement, σ_obs ≈ 1.0°F (siting quality varies)
  - NDBC buoy SST: direct, σ_obs ≈ 0.3°F (but measures SST, not 2m air temp)
  - METAR cloud obs: categorical → quantized fraction
- handle different observation cadences: 1-min ASOS, 15-min FAWN, hourly buoy

### Done when
- cluster specification document/module exists
- observation sources mapped to state dimensions per cluster
- fallback criteria documented

---

## T3.2 — Observation operator + error model layer  **[DONE]**
**Priority:** P1 | **Track:** Option A | **Owner:** `miami-project`

### Goal
Formalize how each source updates the state.

### Tasks
- build observation operators by source family
- define per-source observation error assumptions (R matrix diagonal entries)
- encode source reliability hooks (connect to BOA weights)
- handle multi-cadence observation ingestion

### Depends on
- T3.1

### Done when
- each source family has a formal measurement mapping
- error model tested against innovation statistics

---

## T3.3 — Localized LETKF implementation  **[DONE]**
**Priority:** P1 | **Track:** Option A | **Owner:** `miami-project`

### Goal
Implement lightweight surface-only LETKF for dense regional clusters.

### Tasks
- implement LETKF core (Hunt et al. 2007):
  - ensemble size: 20-50 members (use GEFS 31 as background)
  - localization: Gaspari-Cohn function with configurable radius (50-100km default)
  - covariance inflation: relaxation-to-prior-spread (RTPS) method
  - update frequency: obs-triggered via LiveState event
- add stability guards: eigenvalue clipping, ensemble spread floor
- replay on historical eval windows
- compare LETKF state estimates vs. current feature-only approach

### Depends on
- T3.2

### Done when
- LETKF state updates run in replay
- stability + skill report exists
- demonstrated information propagation across stations within cluster

---

## T3.4 — Self-tuning IMM-KF for sparse regions  **[NOT_STARTED]**
**Priority:** P1 | **Track:** Option A | **Owner:** `miami-project`

### Goal
Upgrade existing Switching Kalman Filter with online adaptive noise estimation for stations without dense nearby networks.

### Tasks
- implement innovation-based adaptive estimation:
  - compute innovation: ν = z_obs - H·x_predicted
  - sample covariance of recent innovations ≈ H·P·H' + R
  - if actual innovation covariance > expected → increase R (or Q)
  - Sage-Husa method: exponential moving average of innovation outer products
- auto-tune R (measurement noise) and Q (process noise) continuously
- no manual noise parameter tuning needed
- computational cost is minimal for 5D state vector

### Depends on
- T3.1 (uses fallback criteria)

### Done when
- IMM-KF self-tunes R/Q from innovation statistics
- replay shows equal or better skill than hand-tuned parameters
- no divergence in extended replay windows

---

# Sprint 4 — Live Regime Model

## T4.1 — Live regime catalog design  **[DONE]**
**Priority:** P1 | **Track:** Option A | **Owner:** `miami-project`

### Goal
Define a compact interpretable live regime set.

### Tasks
- draft initial regime catalog (marine stable, inland heating, cloud suppression, outflow/convective crash, radiational cooling, transition/mixed)
- define features used to separate regimes
- document intended meaning of each regime

### Depends on
- T2.1, T3.3

---

## T4.2 — Live regime inference module  **[DONE]**
**Priority:** P1 | **Track:** Option A | **Owner:** `miami-project`

### Goal
Infer regime probabilities from BOCPD + LETKF/IMM-KF state summaries + residual features.

### Depends on
- T4.1

---

# Sprint 5 — Offline Regime Discovery

## T5.1 — Sticky HDP-HMM offline discovery pipeline  **[NOT_STARTED]**
**Priority:** P2 | **Track:** Option A support / Option C shared | **Owner:** `miami-project`

### Goal
Discover regime candidates offline using sticky HDP-HMM (Fox et al.).

### Implementation notes
- Python: `pyhsmm` package (Matthew Johnson, MIT license)
- Sticky parameter controls regime persistence (prevents rapid switching)
- Auto-discovers number of regimes (no need to pre-specify K)
- Training: hourly obs + atmospheric features over 60+ climate days

### Depends on
- T0.2, T0.3

---

## T5.2 — AI + human regime review workflow  **[NOT_STARTED]**
**Priority:** P2 | **Track:** Option A support | **Owner:** workspace docs

### Goal
Prevent silent regime-catalog drift. Explicit acceptance workflow for regime proposals.

### Implementation note
Add an OpenClaw scheduled task (cron job) that:
1. Runs HDP-HMM regime discovery weekly on the latest data
2. Compares discovered regimes to the live catalog (`regime_catalog.py`)
3. Generates a naming proposal for any new regimes found
4. Sends the proposal to the AI review pipeline for human acceptance
5. Only accepted regimes get added to the live catalog

The OpenClaw cron ensures this runs automatically without manual intervention.
Human approval is still required before any catalog change takes effect.

### OpenClaw integration
The daily recalibration pipeline (`auto_recalibrator.py`) writes regime proposals to
`analysis_data/regime_proposals/{date}.json` when an unrecognized pattern is detected
(no regime confidence > 40%). The OpenClaw scheduled task should:
1. Check for new proposal files in `analysis_data/regime_proposals/`
2. Read the atmospheric context (CAPE, PW, wind, cloud) from the proposal
3. Name the proposed regime using meteorological reasoning
4. Compare against the existing catalog in `src/engine/regime_catalog.py`
5. If it's genuinely new: propose a catalog addition with name, description, and conditioning parameters
6. If it matches an existing regime with low confidence: propose threshold adjustments
7. Write the AI review to `reviews/ai/regime_proposals/{date}.md`
8. Flag for human acceptance before any code changes

**This is a critical feedback loop: bot detects unknown → AI names it → human approves → bot learns.**

---

# Sprint 6 — Trading Architecture Upgrades

## T6.1 — Day-ahead / intraday mode separation  **[DONE]**
**Priority:** P1 | **Track:** Option A | **Owner:** `miami-project` + `weather-trader`

### Goal
Separate trading into two distinct modes with different objectives, sizing, and fee logic.

### Tasks
- **Day-ahead mode** (D-1 evening / D-0 early morning):
  - trigger: fresh overnight model cycles (GFS 00z, HRRR 00z/06z, ECMWF 00z)
  - method: full EMOS-calibrated ensemble → bracket probabilities
  - sizing: conservative Kelly (0.10-0.15x, high uncertainty)
  - fee advantage: held to settlement = entry fee only (no exit fee)
  - edge source: speed of model ingestion (markets take 30-120 min to reprice after new model run)
- **Intraday mode** (D-0 through settlement):
  - trigger: every new observation (obs-triggered via LiveState)
  - method: LETKF/IMM-KF → updated truncated normal CDF → bracket pricing
  - sizing: increasing Kelly as confidence grows through day (up to 0.20-0.25x)
  - fee consideration: round-trips pay entry + exit fees → need higher edge threshold
  - key behaviors: trade into positions when market is slow, unwind when edge erodes

### Depends on
- T1.2 (EMOS needed for day-ahead calibrated distributions)

### Done when
- two-mode infrastructure exists with separate edge thresholds
- paper trading tracks day-ahead vs intraday entries separately
- fee accounting reflects mode-specific costs

---

## T6.2 — Adaptive Kelly + edge auto-tuning  **[DONE]**
**Priority:** P1 | **Track:** Option A | **Owner:** `weather-trader`

### Goal
Replace fixed Kelly fraction and edge threshold with self-adjusting parameters.

### Tasks
- **Adaptive Kelly fraction**:
  - start at 0.15x (matching existing Kalshi weather bot baseline)
  - apply Baker-McHale shrinkage correction: shrink proportional to estimation error (≈ 1 - 1/N where N = calibration sample size)
  - constrain by drawdown: f ≤ 2 × S² × D_max (S = rolling 30-day Sharpe, D_max = max acceptable drawdown)
  - run multiple fractional Kelly strategies in parallel (0.10, 0.15, 0.20, 0.25) and weight by realized performance using multiplicative weights update
- **Edge threshold auto-adjustment**:
  - minimum = 2 × round_trip_fees + estimation_uncertainty_buffer
  - for maker orders at 50¢: minimum ~2¢ edge. For taker: minimum ~5¢
  - auto-widen by 1pp when rolling 50-trade PnL turns negative
  - auto-tighten by 0.5pp when rolling PnL exceeds 2× threshold
  - floor at fee_cost × 2, cap at 25%

### Depends on
- T6.1

### Done when
- Kelly fraction auto-adjusts from realized performance
- edge threshold auto-adjusts from rolling PnL
- paper trading logs parameter history for auditing

---

## T6.3 — Cross-station portfolio risk  **[DONE]**
**Priority:** P1 | **Track:** Option A | **Owner:** `weather-trader`

### Goal
Manage multi-station concurrent risk with correlation awareness.

### Tasks
- maximum portfolio heat: 15-25% of bankroll at risk at any time
- per-station limit: 5-10% of bankroll per station-day
- correlation adjustment: for correlated stations (same cluster, ρ > 0.7), add exposure as concentrated
- daily loss circuit breaker: 3% of bankroll → reduce all position sizes by 50%
- drawdown scaling:
  - 10% drawdown from peak: reduce all sizes by 50%
  - 20% drawdown: reduce by 75%, enter diagnostic mode
  - 30% drawdown: stop trading, full system review
- recovery: increase sizes gradually after 20 consecutive positive-EV trades, not after lucky wins

### Depends on
- multi-station infrastructure

### Done when
- portfolio risk limits enforced in paper and live
- correlation matrix estimated from historical forecast error correlations (not raw temp correlations)
- drawdown triggers tested in simulation

---

## T6.4 — Cross-bracket arbitrage detection  **[DONE]**
**Priority:** P2 | **Track:** Option A | **Owner:** `weather-trader`

### Goal
Monitor bracket strips for probability sum violations.

### Tasks
- monitor: sum of bracket best-asks. If < $1.00 minus total fees → pure arbitrage flag
- monitor: sum of bracket best-bids. If > $1.00 minus total fees → sell all flag
- log opportunities even if not traded (thin, fast, mostly informational)

### Done when
- arbitrage monitor running and logging in paper trading

---

# Sprint 7 — Automated Health Monitoring + Self-Healing

## T7.1 — Rolling forecast quality monitors  **[DONE]**
**Priority:** P1 | **Track:** Option A | **Owner:** `miami-project`

### Goal
Detect model degradation before it costs money.

### Tasks
- rolling 30-day Brier score per station, per source
- alert threshold: +0.02 from baseline
- per-signal Brier score decomposition (calibration + resolution separately)
- per-source CRPS tracking alongside BOA reweighting
- weekly summary report artifact

### Done when
- monitors running daily after settlement
- alert system exists (even if just log/artifact-based initially)

---

## T7.2 — Trading performance regime detection  **[DONE]**
**Priority:** P1 | **Track:** Option A | **Owner:** `weather-trader`

### Goal
Detect trading regime shifts using the system's own BOCPD on its own P&L.

### Tasks
- apply BOCPD to daily P&L series
- when P&L regime shift detected: reduce sizes, widen edge thresholds, enter diagnostic mode
- track rolling 30-day Sharpe ratio
- log regime shift events to parameter_history

### Depends on
- T2.1 (BOCPD implementation exists)

### Done when
- P&L regime detection running on paper P&L
- auto-scaling rules triggered and logged

---

## T7.3 — Concept drift detection  **[DONE]**
**Priority:** P2 | **Track:** Option A | **Owner:** `miami-project`

### Goal
Detect seasonal drift and NWS model upgrades before calibration degrades.

### Tasks
- Population Stability Index (PSI) on feature distributions vs. recent training data
- monitor for NWS model upgrades (HGEFS deployment, GFS updates)
- trigger weekly auto-recalibration of EMOS parameters using most recent 30-60 days
- seasonal drift handling: winter calibration ≠ summer

### Done when
- PSI computed weekly
- recalibration triggers documented and automated

---

## T7.4 — Signal deprecation  **[DONE]**
**Priority:** P2 | **Track:** Option A | **Owner:** `miami-project`

### Goal
Auto-downweight signals that stop contributing to edge.

### Tasks
- track per-signal contribution to winning vs losing trades
- if a signal's resolution component drops below noise floor for 2+ weeks → auto-downweight
- BOA handles forecast source deprecation; this covers non-forecast signals (CAPE, wind, dew point, etc.)
- weekly audit log

### Done when
- signal contribution tracking running
- auto-downweight logic implemented and logged

---

# Sprint 8 — New Data Source Ingestion

## TA8.1 — Cloud coverage ingestion  **[DONE]**
**Priority:** P1 | **Track:** Option A | **Owner:** `miami-project`

### Goal
Add cloud coverage data to constrain daytime high temperature forecasts.

### Tasks (in priority order)
1. **Parse METAR cloud obs from existing NWS obs feed** (lowest effort, highest value):
   - Extract SKC/FEW/SCT/BKN/OVC from raw METAR text
   - Quantize: CLR=0, FEW=0.125, SCT=0.375, BKN=0.75, OVC=1.0
   - Store in observations table or new cloud_observations table
   - Already have the raw data — just need to parse it
2. **Add Open-Meteo cloud cover fields** (if keeping Open-Meteo):
   - Add `cloud_cover`, `cloud_cover_low`, `cloud_cover_mid`, `cloud_cover_high` to atmospheric request
   - Alternatively: use HRRR TCDC field via Herbie if dropping Open-Meteo
3. **Use FAWN solar radiation as cloud proxy** (already in DB):
   - Low solar_radiation_wm2 relative to expected clear-sky → cloudy
   - Clear-sky model: simple function of solar zenith angle + day of year
   - This is the cheapest "cloud observation" we can build — real measurement, already archived
4. **GOES-19 Clear Sky Mask** (highest quality, most effort):
   - Product: ABI-L2-ACMF/ACMC (CONUS, 2km, 5-min)
   - GOES-19 replaced GOES-16 as GOES-East on April 7, 2025
   - AWS S3: `noaa-goes19` bucket (free, no auth)
   - Python: `goes2go` package
   - Highest spatial/temporal resolution but requires NetCDF parsing
5. **RTMA analyzed cloud cover** (blended, very practical):
   - 2.5km hourly, blends satellite + surface obs + model data
   - AWS S3: `noaa-rtma-pds` (free), Python via Herbie

### Done when
- at least METAR cloud obs + FAWN solar proxy available in inference pipeline
- cloud cover used in high-temperature forecasting (reduce sigma or shift mu on cloudy days)

---

## TA8.2 — Open-Meteo cost optimization  **[DONE]**
**Priority:** P1 | **Track:** Option A | **Owner:** `miami-project`

### Goal
Reduce or eliminate Open-Meteo API costs by replacing with free alternatives where possible.

### Current Open-Meteo audit (2026-03-22)

**What Open-Meteo provides (5 categories)**:
1. **11 deterministic model forecasts** (GFS, HRRR, ECMWF, ICON, GEM, JMA, UKMO, MetNo, KNMI, GraphCast, AIFS)
2. **2 ensemble forecast sets** (GFS 31-member, ECMWF 51-member) — **hardest to replace**
3. **Pressure level data** (925/850/700/500 hPa from GFS) — only 850/925 consumed
4. **Atmospheric parameters** (CAPE, PW, radiation, soil temp, precip, boundary layer)
5. **Soil/moisture data** — not consumed in trading logic

**What Wethr already provides** (no additional cost):
- NBM, NAM, NAM-4KM, ARPEGE, RAP, LAV-MOS, GFS-MOS, NAM-MOS, NBS-MOS
- NBM is NWS's own calibrated blend — high value

**Phased replacement plan**:

| Phase | Action | Savings | Effort |
|-------|--------|---------|--------|
| **Immediate** | Stop fetching unused fields (lifted_index, boundary_layer_height, soil_moisture, direct/diffuse radiation) | ~10-15% API calls | Low |
| **Phase 1** | Add HRRR CAPE/PW ingestion via Herbie + AWS S3. Deprecate Open-Meteo atmospheric endpoint. | ~30% API calls | Medium |
| **Phase 2** | Add direct GFS/HRRR forecast ingestion via Herbie. Keep Open-Meteo for non-US models only. | ~50% API calls | High |
| **Phase 3** | Evaluate: keep Open-Meteo for ensembles only (hardest to replace: 82 GRIB files per cycle) OR drop entirely | ~80-100% | Very High |

### Tasks
1. Identify and remove unused atmospheric variable fetches
2. Build HRRR ingestion adapter using `herbie-data` package + AWS S3
3. Parse CAPE (`CAPE:surface`), PW (`PWAT:entire atmosphere`), cloud cover (`TCDC:entire atmosphere`)
4. Route critical atmospheric fields from HRRR instead of Open-Meteo
5. Evaluate remaining Open-Meteo value vs. cost

### Done when
- unused variables dropped from API requests
- HRRR adapter providing CAPE/PW/cloud cover
- cost reduction quantified

---

## TA8.3 — Expanded surface observation network  **[NOT_STARTED]**
**Priority:** P2 | **Track:** Option A | **Owner:** `miami-project`

### Goal
Expand nearby station coverage for all trading stations beyond current IEM ASOS.

### Tasks
- **IEM expansion**: IEM already aggregates ASOS, AWOS, RWIS, COOP, school nets. Expand queries to include more network types per trading station.
- **State mesonet direct APIs** (Synoptic/SynopticData went paid-only; avoid):
  - NYSM (NY State Mesonet): 127 stations, free REST API → critical for KNYC, KPHL
  - NJWXNET: 65+ stations → for KPHL, KNYC support
  - TexMesonet: 90+ stations → for KAUS, KSAT, KDFW support
- **MADIS registration**: NOAA's multi-network aggregator (30,000+ stations, free for research, ~1h latency)
- **NOTE**: Do NOT use SynopticPy/SynopticData — requires paid API key now

### Done when
- at least 3-5 nearby supporting stations per trading station
- state mesonet APIs integrated for NE Corridor and Texas clusters

---

## TA8.4 — MRMS radar for outflow detection  **[NOT_STARTED]**
**Priority:** P2 | **Track:** Option A | **Owner:** `miami-project`

### Goal
Add radar-based convective outflow early warning.

### Tasks
- ingest MRMS composite reflectivity within 50km of trading stations
- AWS S3: `noaa-mrms-pds` bucket (free, 1km, 2-min updates)
- detect approaching convection → flag in signal engine
- 15-30 minute early warning on outflow events (the biggest tail risk for Florida lows)

### Done when
- MRMS adapter ingesting and storing reflectivity
- outflow signal engine updated to consume radar proximity data

---

## TA8.5 — HRRR + obs archiving for Analog Ensemble  **[DONE]**
**Priority:** P2 | **Track:** Option A | **Owner:** `miami-project`

### Goal
Begin systematic archiving of HRRR forecasts paired with station observations for future Analog Ensemble (AnEn).

### Tasks
- archive HRRR forecast + paired ASOS observation at each trading station
- store: (station, forecast_date, model_init_time, forecast_hour, forecast_temp, observed_temp, observed_high, observed_low, settlement)
- after 12-18 months: implement NCAR Analog Ensemble (Delle Monache et al.)
  - find N most similar historical forecasts to current forecast
  - use corresponding historical observations as ensemble members
  - intrinsically calibrated, no parametric assumptions

### Done when
- archiving pipeline running for all trading stations
- data accumulating in systematic archive table

---

# Sprint 9 — Rollout

## T9.1 — Option A replay gate  **[NOT_STARTED]**
**Priority:** P1 | **Track:** Option A | **Owner:** `miami-project`

### Goal
Verify Option A stack beats current baseline before rollout.

### Depends on
- T1.x, T2.x, T3.x, T4.x as applicable

### Done when
- replay report exists; promotion decision explicit

---

## T9.2 — Paper rollout for Option A  **[NOT_STARTED]**
**Priority:** P1 | **Track:** Option A | **Owner:** `weather-trader` + `miami-project`

### Depends on
- T9.1

---

## T9.3 — Live shadow / partial-live rollout  **[NOT_STARTED]**
**Priority:** P2 | **Track:** Option A | **Owner:** `weather-trader`

### Depends on
- T9.2

---

# Option C — Shadow Build Order

## TC1 — Sticky HDP shadow experiments  **[NOT_STARTED]**
**Priority:** P3 | **Track:** Option C | **Owner:** `miami-project`

### Goal
Explore richer offline regime discovery.

### Implementation
- Python: `pyhsmm` package (Matthew Johnson, MIT license)
- Sticky HDP-HMM auto-discovers regime count

---

## TC2 — Recurrent sticky HDP experiment  **[NOT_STARTED]**
**Priority:** P3 | **Track:** Option C

### Goal
Test whether nonstationary regime persistence adds value. Only proceed if TC1 shows promise.

---

## TC3 — Mixture-of-experts gating shadow  **[NOT_STARTED]**
**Priority:** P3 | **Track:** Option C | **Owner:** `miami-project`

### Goal
Test richer learned source gating. Shadow MoE conditioned on state / regime / time-to-settlement.

---

## TC4 — DS3M shadow challenger  **[NOT_STARTED]**
**Priority:** P3 | **Track:** Option C | **Owner:** shadow research

### Goal
Train Deep Switching State Space Model as shadow challenger.
Combines deep learning with state-space switching (discrete latent regime changes).

---

## TC5 — DMA shadow challenger  **[NOT_STARTED]**
**Priority:** P3 | **Track:** Option C | **Owner:** shadow research

### Goal
Implement Dynamic Model Averaging (Raftery et al. 2010) as simpler shadow challenger to BOA.
Time-varying model weights + time-varying parameters.
Good benchmark: if DMA can't beat BOA, DS3M needs to do much better.

---

## TC6 — Conformal prediction overlay  **[NOT_STARTED]**
**Priority:** P3 | **Track:** Option C | **Owner:** miami-project

### Goal
Distribution-free coverage guarantees for bracket probabilities.
Adaptive Conformal Inference (ACI) provides time-series-appropriate prediction intervals.
Low implementation cost, high diagnostic value.

---

## TC7 — Shadow challenger evaluation bundle  **[NOT_STARTED]**
**Priority:** P3 | **Track:** Option C

### Goal
Standard package for evaluating all shadow contenders against Option A.

---

# Evaluation / Promotion Tickets

## TE1 — Head-to-head benchmark runner  **[NOT_STARTED]**
**Priority:** P1 | **Track:** Shared | **Owner:** `miami-project`

### Goal
Benchmark baseline, Option A, and Option C challengers in one run.

---

## TE2 — Promotion dashboard/report  **[NOT_STARTED]**
**Priority:** P2 | **Track:** Shared

### Goal
Make promotion decisions explicit and reviewable.

---

## TE3 — Kill-switch / rollback criteria doc  **[NOT_STARTED]**
**Priority:** P2 | **Track:** Shared

### Goal
Ensure production can revert quickly if a promoted candidate misbehaves.

---

# Suggested Work Sequence for Next Sessions

## Immediate priority order
1. **T1.4** — Validate and close source trust (IN_PROGRESS)
2. **T1.2** — EMOS calibration layer (highest ROI next ticket)
3. **T1.5** — BOA online source weighting (can parallel T1.2)
4. **TA8.1** — Cloud coverage ingestion (low-hanging fruit, high value)
5. **TA8.2** — Open-Meteo cost optimization (HRRR adapter replaces expensive API calls)
6. **T2.2** — CUSUM downgrade
7. **T3.1** — Regional cluster LETKF design
8. **T3.2-T3.4** — Observation operators + LETKF + self-tuning IMM-KF
9. **T4.1-T4.2** — Live regime model
10. **T6.1-T6.3** — Trading architecture (day-ahead/intraday, adaptive Kelly, portfolio risk)
11. **T7.1-T7.4** — Health monitoring
12. **T5.1-T5.2** — Offline HDP regime discovery
13. **T9.x** — Rollout

## Shadow can begin in parallel after foundation
- TC1 Sticky HDP shadow experiments
- TC3 MoE gating shadow
- TC4-TC5 DS3M / DMA shadow lanes
- TC6 Conformal prediction
- TC7 shadow challenger evaluation

---

# Final reminder for future agents

If you are unsure what to do next:
- do not improvise a new architecture,
- do not merge Option A and Option C informally,
- do not promote shadow logic because it sounds smarter,
- start from the highest-priority unfinished ticket in this file.
