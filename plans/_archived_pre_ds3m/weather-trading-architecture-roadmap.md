# Weather Trading Architecture Roadmap

Updated: 2026-03-22
Status: Canonical plan for future agents

## Purpose

This document is the reference plan for the next-generation weather trading architecture.
It exists to prevent drift, reduce re-litigation, and keep future agents aligned on:

- what the **live / production path** is,
- what the **shadow / R&D path** is,
- how those paths stay separate,
- what gets evaluated, how it gets promoted, and what must **not** silently change.

This document should be preferred over chat-history reconstruction when deciding what to build next.

---

## Executive Decision Snapshot

Two architecture paths are being pursued:

### Option A — **Live / Production Track**
The production architecture should move toward:

- **Regional cluster LETKF** for continuous spatial state assimilation (dense regions) + **self-tuning IMM-KF** (sparse regions)
- **BOCPD** for online break / changepoint probability
- **Small live regime model** (IMM / HMM style) for interpretable regime probabilities
- **BOA (Bernstein Online Aggregation)** for optimal CRPS-based source weighting
- **EMOS → SAR-SEMOS → EMOS-GB** as the probabilistic calibration progression
- **Semi-local EMOS with station embeddings** for multi-station cold-start scaling
- **Sticky HDP-HMM offline** for regime discovery and catalog refinement
- **Day-ahead / intraday mode separation** with distinct sizing and fee logic
- **Adaptive Kelly + cross-station portfolio risk** in the trading layer
- **Automated health monitoring** (Brier/CRPS monitors, P&L BOCPD, concept drift detection)
- **Current trading / portfolio / execution layer downstream**

### Option C — **Shadow / R&D Track**
The ambitious research lane should run in parallel as a challenger:

- richer multi-model / multi-source assimilation
- sticky / recurrent sticky HDP regime discovery
- mixture-of-experts or dynamic gating
- **DS3M** (Deep Switching State Space Model) as a shadow challenger
- **DMA (Dynamic Model Averaging)** as a simpler shadow challenger
- **Conformal prediction** for distribution-free bracket coverage
- no execution authority until it earns promotion

### Core policy
- **Option A is the live plan.**
- **Option C is the shadow plan.**
- **DS3M stays open and should accumulate data continuously.**
- Option C may influence Option A only through explicit evaluation and promotion gates.

---

## Multi-Station Scaling Strategy

### Target: All Wethr Trading Stations
The system is designed to scale from KMIA to all ~20 Kalshi temperature trading stations.
This is not a future aspiration — the architecture must support it from the start.

### Regional Cluster Architecture
Stations are grouped into geographic clusters that share spatial observation networks:

| Cluster | Trading Stations | Supporting Obs Network |
|---------|-----------------|----------------------|
| **SE Florida** | KMIA, KFLL | KOPF, KHWO, KTMB, FAWN (42 stations), NDBC buoys |
| **NE Corridor** | KNYC, KPHL, KBOS, KDCA | KLGA, KJFK, KEWR, NYSM (127 stations) |
| **Texas Triangle** | KAUS, KSAT, KHOU, KDFW | KDAL, IEM nearby ASOS |
| **Upper Midwest** | KMDW, KMSP | KORD + nearby ASOS |
| **Mountain/Desert** | KDEN, KPHX, KLAS | Sparse — IMM-KF fallback |
| **West Coast** | KLAX, KSFO, KSEA | Sparse coastal — IMM-KF with marine obs |

### Why Clusters Matter
- Within a cluster, a **regional LETKF** propagates information from stations that observe a weather feature first to stations that haven't seen it yet (e.g., sea breeze frontal passage in SE Florida)
- **Semi-local EMOS** allows new stations to cold-start by borrowing calibration from climatologically similar stations in the same or adjacent clusters
- **Cross-station correlation-aware sizing** prevents concentrated bets on correlated outcomes (KMIA + KFLL high-temp positions are ~70% correlated)
- For sparse clusters (<3 supporting stations within localization radius), fall back to per-station **self-tuning IMM-KF**

---

## Model Ingestion Inventory (as of 2026-03-22)

### Current sources and models

**From Wethr REST API** (5-min polling):
- NBM, NAM, NAM-4KM, ARPEGE, RAP
- LAV-MOS, GFS-MOS, NAM-MOS, NBS-MOS
- Plus any additional models Wethr adds (model names are API-driven)

**From Open-Meteo** (5-min polling):
- Deterministic: GFS-Global, GFS-HRRR (hourly cycles, 1.5h lag), GFS-GraphCast, ECMWF-IFS, ECMWF-AIFS, ICON-Global, GEM-Global, JMA-GSM, UKMO-Global, MetNo-Nordic, KNMI-Harmonie
- Ensemble: GFS-Ensemble (31 members), ECMWF-IFS-Ensemble (51 members)

**From NWS** (2.5-min polling):
- NWS forecast versions

**Total**: ~19+ distinct deterministic models + 2 ensemble sets

### HRRR Status
HRRR is ingested via Open-Meteo as "GFS-HRRR" with **hourly initialization** and **1.5-hour availability lag**. Wethr may also provide HRRR under its own naming. Both are stored separately (keyed by source + model) — no cross-source deduplication conflict.

### NBM Status
NBM is ingested via **Wethr** REST API. NBM is one of the most valuable models — it is NWS's own calibrated post-processed blend. It should be treated as a high-trust source.

### Models NOT currently ingested (candidates for addition)
- **RTMA/URMA** (2.5km hourly surface analysis — NWS's analysis of "what happened")
- **MOS station-specific guidance** (already partially via Wethr: GFS-MOS, NAM-MOS, NBS-MOS, LAV-MOS)
- **HRRR direct from AWS S3** (if Open-Meteo's 1.5h lag is too slow; direct HRRR available ~1h after init via Herbie)

---

## New Data Sources to Add

### Priority 1: Cloud Coverage (Critical for High Forecasts)

**METAR/ASOS cloud observations** (already partially available):
- SKC/FEW/SCT/BKN/OVC in METAR reports → parse from NWS obs feed
- Quantize: CLR=0, FEW=0.125, SCT=0.375, BKN=0.75, OVC=1.0
- Already ingested stations report this — just need to extract and use it

**GOES-19 Clear Sky Mask (ACM)**:
- NOTE: GOES-19 replaced GOES-16 as GOES-East on April 7, 2025. Use `noaa-goes19`.
- Product: ABI-L2-ACMF/ACMC (Clear Sky Mask, CONUS)
- Resolution: 2km, every 5 minutes
- 4-level mask: confidently clear / probably clear / probably cloudy / confidently cloudy
- Access: AWS S3 bucket `noaa-goes19` (us-east-1, free, no auth)
- Format: NetCDF4
- Python: `goes2go` package (`pip install goes2go`) or direct S3 via `s3fs`
- This is the highest-value new data source for constraining daytime high temperature

**RTMA Analyzed Cloud Cover**:
- RTMA (Real-Time Mesoscale Analysis) includes analyzed cloud cover + ceiling
- 2.5km resolution, hourly, blends satellite + surface obs + model data
- AWS S3: `noaa-rtma-pds` bucket (free)
- Python: Herbie (`Herbie("2026-03-22", model="rtma", product="anl")`)
- NWS "analysis of record" — represents what actually happened, not a forecast
- Also includes analyzed temperature, dewpoint, wind, visibility, pressure

**HRRR cloud fields**:
- HRRR outputs total cloud cover (TCDC) at surface level
- Available via AWS S3 or Herbie: `Herbie("2026-03-22", model="hrrr", fxx=1).xarray("TCDC")`
- Hourly resolution, 3km grid

**Open-Meteo cloud cover**:
- Open-Meteo API provides `cloud_cover`, `cloud_cover_low`, `cloud_cover_mid`, `cloud_cover_high`
- Already available through the existing API — just need to add the fields to the request

**FAWN solar radiation** (already ingested):
- FAWN Homestead provides `solar_radiation_wm2`
- Solar radiation is a direct proxy for cloud cover — low solar = cloudy
- Already in the database; can be used immediately as a cloud proxy

### Priority 2: Expanded Surface Observation Network

**Iowa Environmental Mesonet (IEM)** (already used for nearby stations):
- IEM aggregates: ASOS, AWOS, RWIS, state mesonets, COOP, school nets
- Free API, no auth required
- Already ingested for nearby ASOS stations — expand to include more networks

**MesoWest / Synoptic Data**:
- MesoWest is sunsetting December 31, 2026 (director retiring, legacy software EOL)
- SynopticData (successor) has a free tier: 5K requests/month, 5M service units. Open Access Program (unlimited) requires .edu email.
- SynopticPy Python package still works with free account token
- However, IEM is the better primary choice — free, no account, no rate limits, nationwide ASOS coverage
- Direct state mesonet APIs supplement IEM for non-ASOS networks (FAWN already done; add NYSM, TexMesonet)

**State Mesonet Direct APIs** (free, per-state):
| Network | Stations | Coverage | API |
|---------|----------|----------|-----|
| NYSM (NY State Mesonet) | 127 | New York | Free web access; real-time feed $200 one-time setup; commercial license required for trading |
| Oklahoma Mesonet | 120+ | Oklahoma (not a trading state, but nearby) | Free |
| TexMesonet (West TX) | 90+ | West Texas (for KAUS/KSAT/KDFW nearby) | Free |
| NJWXNET | 65+ | New Jersey (for KPHL/KNYC) | Free |

**MADIS (Meteorological Assimilation Data Ingest System)**:
- NOAA's multi-network aggregator: 30,000+ stations
- Includes ASOS, AWOS, maritime, mesonet, profiler networks
- Access: MADIS data portal (requires registration, free for research)
- Most comprehensive single source but has data latency (~1 hour)

### Priority 3: Radar for Outflow Detection

**MRMS (Multi-Radar Multi-Sensor)**:
- NOAA composite reflectivity product
- 1km resolution, 2-minute updates
- AWS S3: `noaa-mrms-pds` bucket (free)
- Critical for Florida: convective outflow causes 5-10°F temperature crashes
- Use within 50km radius of trading stations for outflow early warning

### Priority 4: Long-Term Data Archiving

**HRRR + obs archiving for Analog Ensemble (AnEn)**:
- Start archiving HRRR forecasts + paired ASOS observations at every trading station
- After 12-18 months: build Analog Ensemble (NCAR method)
- AnEn finds N most similar historical forecasts → uses corresponding observations as ensemble members
- Intrinsically calibrated, no parametric assumptions
- Begin archiving NOW — this is a long-term investment

---

## Source Trust Policy (interim) + Auto-Tuning Direction

### Interim live default (use now)

When sources conflict materially, use this priority as a **fallback prior**:

1. Settlement station observations (authoritative for settlement context)
2. Nearby official stations (same source family / quality controls)
3. FAWN / mesonet
4. Buoy
5. Cloud / overcast proxy

Exception: in clear marine-onshore setups, buoy influence can be temporarily
upweighted relative to inland non-official feeds.

### Sparse-window default (use now)

- Require at least 2 fresh independent sources (including >=1 official) for
  normal new entries.
- If this condition is not met: hold/trim only, reduce sizing, and avoid
  assigning single-source full authority.

### Long-run policy: BOA Online Source Weighting

The interim hand-tuned multipliers will be replaced by **Bernstein Online Aggregation (BOA)** with CRPS loss:

- Each forecast source produces a predictive distribution
- After each settlement, compute CRPS loss per source
- BOA automatically reweights with logarithmic regret bounds (provably converges to best fixed combination)
- No manual tuning, no rolling windows to configure
- Interim trust multipliers serve as the initial prior until BOA has sufficient data
- Guardrails remain: minimum sample counts, bounded day-over-day changes, sparse-window caps

### Minimum useful backfill spec (executed 2026-03-21)

**Executed**: `source_trust_backfill.py` produced 9224 records over 8 covered days (openmeteo MAE≈3.13°F, wethr≈5.79°F). Sufficient days = false (target 30).

---

## Operational path map (authoritative)

### Repositories
- `miami-project`: `/Users/blakebot/blakebot/miami-project`
- `weather-trader`: `/Users/blakebot/blakebot/weather-trader`

### Canonical planning docs
- Architecture roadmap (this file):
  - `/Users/blakebot/blakebot/miami-project/plans/weather-trading-architecture-roadmap.md`
- Execution checklist:
  - `/Users/blakebot/blakebot/miami-project/plans/weather-trading-implementation-checklist.md`
- Ticketized build order:
  - `/Users/blakebot/blakebot/miami-project/plans/weather-trading-ticketized-build-order.md`

### Current core implementation modules (as of 2026-03-22)
- Archive audit module:
  - `miami-project/src/analyzer/archive_audit.py`
- Replay/live time guardrails:
  - `miami-project/src/engine/replay_context.py`
- Dynamic weighting module:
  - `miami-project/src/engine/dynamic_weights.py`
- Baseline integration point:
  - `miami-project/src/engine/baseline_engine.py`
- BOCPD + changepoint integration:
  - `miami-project/src/engine/bocpd.py`
  - `miami-project/src/engine/changepoint_detector.py`
  - `miami-project/src/engine/orchestrator.py`
- Source-trust backfill extractor + rolling refresh + prior loader:
  - `miami-project/src/analyzer/source_trust_backfill.py`
  - `miami-project/src/analyzer/source_trust_refresh.py`
  - `miami-project/src/engine/source_trust.py`
- Changepoint replay comparison bundle:
  - `miami-project/src/analyzer/changepoint_replay_compare.py`
- Canonical replay bundle entrypoint:
  - `miami-project/src/analyzer/canonical_replay_bundle.py`

### Test modules currently used for these phases
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

---

## Logging, audit, and debugging locations

### Archive/replay readiness output
- Markdown report: `miami-project/analysis_data/archive_audit_report.md`
- JSON report: `miami-project/analysis_data/archive_audit_report.json`

### Source-trust backfill output
- Markdown report: `miami-project/analysis_data/source_trust_backfill.md`
- JSON artifact: `miami-project/analysis_data/source_trust_backfill.json`
- Rolling refresh/state: `miami-project/analysis_data/source_trust_refresh.json`, `source_trust_state.json`

### Changepoint BOCPD-vs-CUSUM comparison output
- Markdown report: `miami-project/analysis_data/changepoint_compare.md`
- JSON artifact: `miami-project/analysis_data/changepoint_compare.json`

### Canonical replay bundle output
- Markdown report: `miami-project/analysis_data/canonical_replay_bundle.md`
- JSON artifact: `miami-project/analysis_data/canonical_replay_bundle.json`

### Paper-trading diagnostics in DB
- Trades: table `paper_trades`
- Mark history: table `paper_trade_marks`
- Settlement outcomes: table `paper_trade_settlements`

### Inference/trading runtime notes
- Orchestrator notes flow through `InferenceCycleResult.notes`
- Paper trader cycle returns include: `entries`, `exits`, `adaptive` summary metadata

---

## Debugging quickstart (copy/paste)

Run from `miami-project` repo root unless noted.

### 1) Archive audit
```bash
PYTHONPATH=src ./.venv/bin/python -m analyzer.archive_audit \
  --db miami_collector.db --station KMIA \
  --out-md analysis_data/archive_audit_report.md \
  --out-json analysis_data/archive_audit_report.json
```

### 2) Focused test run for roadmap modules
```bash
./.venv/bin/python -m pytest -q \
  tests/test_archive_audit.py \
  tests/test_replay_context.py \
  tests/test_orchestrator_replay_guards.py \
  tests/test_dynamic_weights.py \
  tests/test_baseline_engine_dynamic_weighting.py \
  tests/test_source_trust_backfill.py \
  tests/test_source_trust_refresh.py \
  tests/test_source_trust.py \
  tests/test_bocpd.py \
  tests/test_changepoint_bocpd.py \
  tests/test_changepoint_compare.py \
  tests/test_canonical_replay_bundle.py \
  tests/test_replay_remaining_backtest_metrics.py
```

### 3) Source-trust backfill artifact refresh
```bash
PYTHONPATH=src ./.venv/bin/python -m analyzer.source_trust_backfill \
  --db miami_collector.db --station KMIA --lookback-days 90 --min-days 30 \
  --out analysis_data/source_trust_backfill.json \
  --md-out analysis_data/source_trust_backfill.md
```

### 4) Source-trust rolling refresh/state update
```bash
PYTHONPATH=src ./.venv/bin/python -m analyzer.source_trust_refresh \
  --db miami_collector.db --station KMIA --lookback-days 45 --min-days 15 \
  --summary-out analysis_data/source_trust_backfill.json \
  --summary-md-out analysis_data/source_trust_backfill.md \
  --state-path analysis_data/source_trust_state.json \
  --out analysis_data/source_trust_refresh.json
```

### 5) Changepoint BOCPD-vs-CUSUM comparison refresh
```bash
PYTHONPATH=src ./.venv/bin/python -m analyzer.changepoint_replay_compare \
  --db miami_collector.db --station KMIA \
  --out analysis_data/changepoint_compare.json \
  --md-out analysis_data/changepoint_compare.md
```

### 6) Canonical replay bundle refresh
```bash
PYTHONPATH=src ./.venv/bin/python -m analyzer.canonical_replay_bundle \
  --db miami_collector.db --station KMIA --include-changepoint-compare \
  --out analysis_data/canonical_replay_bundle.json \
  --md-out analysis_data/canonical_replay_bundle.md
```

### 7) Paper trader smoke
```bash
PYTHONPATH=src ./.venv/bin/python -m trading.paper_trader \
  --db miami_collector.db --station KMIA --max-quote-age-minutes 20
```

---

## Non-Negotiable Separation Rules

### 1. Do not collapse Option A and Option C into one vague stack
- Option A is for reliable, interpretable live behavior.
- Option C is for expressive, higher-risk shadow research.

### 2. Option C does not place trades
- Shadow models may score, rank, explain, and compare.
- Shadow models may not directly drive live or paper execution without explicit promotion.

### 3. One canonical replay/evaluation harness
- All comparisons must run on the same archived data and replay logic.

### 4. Orchestrator vs trader responsibilities remain separate
- **Orchestrator / inference layer** owns state estimation, forecast distributions, regime probabilities, and bracket probabilities.
- **Trader / portfolio layer** owns EV-after-fees, sizing, concentration, portfolio constraints, and execution decisions.
- **Executor (paper/live)** logs and simulates/places orders.

### 5. AI review is advisory, not self-authorizing
- AI may propose regime merges/splits or architecture changes.
- Production changes require explicit acceptance.

---

## Why Option A Is the Live Choice

Option A is the best blend of spatial awareness, calibration, regime awareness, interpretability, live-trading safety, and manageable implementation risk.

The system already ingests spatially informative inputs (nearby stations, FAWN, buoys, cloud proxies, atmospheric data). With multi-station scaling to all Wethr trading stations, a **regional cluster LETKF** becomes the natural state assimilation approach — propagating information across stations in dense regions while falling back to IMM-KF for sparse regions.

The key production advantages:
- **BOA source weighting** provably converges to optimal source combination (logarithmic regret)
- **EMOS calibration** is the gold standard for NWP post-processing (Gneiting et al. 2005)
- **Semi-local EMOS** enables fast cold-start at new stations by borrowing calibration from similar stations
- **Regional LETKF** exploits spatial covariance that independent per-station filters waste
- **Day-ahead/intraday separation** optimizes for different fee structures and edge sources

---

## Why Option C Stays Open

Option C has the highest upside if enough data accumulates and the shadow lane proves robust:
- DS3M may eventually outperform structured methods
- Richer regime discovery may expose blind spots
- Nonlinear learned gating may outperform hand-built weighting
- Conformal prediction may provide better coverage guarantees
- DMA may outperform BOA in non-stationary environments

However, Option C has higher risks: overfitting, harder debugging, weaker interpretability, harder promotion standards, greater operational complexity.

---

# Shared Foundation (Required Before Either Path Can Truly Win)

## Shared Foundation Goal
Build a single archival, replay, and evaluation substrate used by both Option A and Option C.

## Shared Foundation Requirements

### Canonical archive of:
- model forecasts by source / run / valid time
- current and historical observations
- nearby stations / FAWN / mesonet / buoys
- cloud coverage observations and proxies
- market quotes and snapshot timestamps
- remaining-move targets and settlement outcomes
- replay-time inference outputs
- paper/live trade logs and outcomes

### Canonical replay harness
Must support: historical eval timestamps, forward-only data availability, same data latency constraints, same quote freshness rules.

### Canonical evaluation metrics
At minimum:
- remaining-high MAE / RMSE / CRPS
- remaining-low MAE / RMSE / CRPS
- bracket probability Brier score
- bracket log loss
- calibration curves / reliability
- sharpness / spread behavior
- EV after fees
- realized paper-trading quality
- turnover / drawdown / concentration

### Anti-leakage rules
- no future observations in replay state estimation
- no future settlements in model fitting for a replay timestamp
- no quote usage that violates live availability constraints

---

# Option A — Live / Production Roadmap

## Option A Summary

Option A should become the production architecture in phased form:

1. stronger dynamic aggregation via BOA + EMOS calibration
2. better online break detection via BOCPD
3. regional cluster LETKF state assimilation + self-tuning IMM-KF fallback
4. compact interpretable live regime model
5. offline sticky HDP review loop
6. day-ahead / intraday trading mode separation
7. adaptive Kelly + cross-station portfolio risk management
8. automated health monitoring and self-healing
9. new data source ingestion (cloud cover, expanded mesonet, radar)
10. staged rollout through replay → paper → live shadow → partial live → full live

---

## Phase A0 — Foundation Lockdown

### Goal
Create the shared archive, replay harness, and metric framework.

### Deliverables
- canonical archived data schema
- one replay entry point for historical inference cycles
- one evaluation pipeline used by every contender
- one leaderboard/report format

### Status: MOSTLY DONE (T0.1-T0.4)

---

## Phase A1 — Calibration Engine (Highest ROI)

### Goal
Replace static blending with optimal source weighting and robust probabilistic calibration.

### Build

**A1a — BOA Online Source Weighting**:
- Implement Bernstein Online Aggregation with CRPS loss
- Each source produces a predictive distribution (Normal with source-specific mu, sigma)
- After each settlement, compute CRPS loss per source → BOA reweights
- Logarithmic regret bound guarantees convergence to optimal combination
- Current T1.4 trust multipliers become the initial prior; BOA takes over ongoing learning
- Reference: Berrisch et al. (2023)

**A1b — EMOS Calibration Layer** (progression):
1. **Standard EMOS** (Gneiting et al. 2005): Normal distribution, mean = a + b1*f1 + b2*f2 + ..., variance = c + d*S^2. Fit via CRPS minimization. Minimum ~30-40 days training data.
2. **SAR-SEMOS** (Jobst et al. 2024): Adds autoregressive error structure + seasonal Fourier terms. Significantly improves CRPS at all lead times.
3. **EMOS-GB** (gradient-boosted EMOS): Handles mixture distributions, automatic feature selection, can exploit sigma climatology. Target for Month 3+.

**A1c — Semi-Local EMOS with Station Embeddings** (for multi-station scaling):
- New stations cold-start using EMOS trained on climatologically similar stations (Lerch & Baran 2017)
- Neural network EMOS with station identity embeddings (Rasp & Lerch 2018) learns shared patterns while maintaining station-specific adjustments
- Cold-start pipeline: Week 1-2 use regional EMOS → Week 3-6 blend regional + local via BOA → Month 2+ local dominates

**A1d — Optional BMA branch** for genuinely multimodal disagreement cases

### Success criteria
- improved CRPS and log score vs current combiner
- equal or better sharpness subject to calibration
- improved reliability curves in replay
- new station cold-start achieves competitive calibration within 2 weeks

---

## Phase A2 — BOCPD Online Break Layer

### Goal
Replace raw threshold-heavy changepoint dependence with probabilistic online break detection.

### Build
- BOCPD as the main online break detector
- Outputs: changepoint probability, run length, persistence confidence
- Keep CUSUM only as diagnostic / backup

### Status: MOSTLY DONE (T2.1)

---

## Phase A3 — Regional Cluster LETKF + Self-Tuning IMM-KF

### Goal
Treat the local atmospheric environment as a joint spatial state and assimilate all sources together.

### Build

**A3a — Regional Cluster Design**:
- Define geographic clusters (SE Florida, NE Corridor, Texas, Midwest, Mountain, West Coast)
- Define which stations are trading targets vs. supporting obs within each cluster
- Define state vector per cluster: surface temp, dew, pressure at each member station
- Define localization radius: 50-100km (literature standard for surface temperature)
- Define fallback rule: if cluster has <3 supporting stations within localization radius → use IMM-KF

**A3b — Observation Operator + Error Model Layer**:
- Formal measurement mapping per source family
- Source-specific observation error assumptions (ASOS: σ ≈ 0.5°F, FAWN: σ ≈ 1.0°F, buoy SST: σ ≈ 0.3°F)
- Handle different cadences: 1-min ASOS, 15-min FAWN, hourly buoy

**A3c — Lightweight Surface-Only LETKF Implementation**:
- Background ensemble: NWP ensemble members (GEFS 31 or HRRR time-lagged)
- Ensemble size: 20-50 members (literature standard for surface networks)
- Localization: Gaspari-Cohn function with configurable radius
- Covariance inflation: relaxation-to-prior-spread (RTPS) method
- Update frequency: every time new obs arrive (obs-triggered via LiveState)

**A3d — Self-Tuning IMM-KF for Sparse Regions**:
- Upgrade existing SKF with innovation-based adaptive estimation
- Auto-tune R (measurement noise) from innovation sequence covariance
- Auto-tune Q (process noise) via Sage-Husa or Mehra's method
- No manual noise parameter tuning needed

### Success criteria
- state estimates improve remaining-move forecast skill
- replay beats feature-only or ad hoc update baselines
- no unstable covariance explosions or divergence
- LETKF clusters demonstrate information propagation across stations

---

## Phase A4 — Small Live Regime Model

### Goal
Add a compact interpretable regime layer on top of the continuous state.

### Build
A small regime catalog (marine stable, inland heating, cloud suppression, outflow/convective crash, radiational cooling, transition/mixed).

Use BOCPD outputs + EnKF/IMM-KF state summaries + residual behavior as inputs.

### Success criteria
- regime-conditioned forecasts beat unconditional forecasts
- regime probabilities align with meteorological intuition
- regime transitions are stable enough for live use

---

## Phase A5 — Sticky HDP-HMM Offline Regime Review Loop

### Goal
Use richer offline regime discovery to improve the live catalog without handing it live authority.

### Build
- Run sticky HDP-HMM offline on historical days
- Learn candidate regime counts, transition structure, redundant/missing states
- Feed results into AI review + human review
- Explicitly approve or reject live catalog updates

---

## Phase A6 — Trading Architecture Upgrades

### Goal
Optimize the trading layer for fee efficiency, multi-station risk, and self-adjustment.

### Build

**A6a — Day-Ahead / Intraday Mode Separation**:
- **Day-ahead mode** (D-1 evening / D-0 early morning):
  - Trigger: fresh overnight model cycles (GFS 00z, HRRR 00z/06z)
  - Method: full EMOS-calibrated ensemble → bracket probabilities
  - Sizing: conservative Kelly (high uncertainty)
  - Fee advantage: positions held to settlement pay only entry fees
  - Edge source: speed of model ingestion (markets take 30-120 min to reprice)
- **Intraday mode** (D-0 through settlement):
  - Trigger: every new observation (obs-triggered inference)
  - Method: LETKF/IMM-KF assimilation → updated truncated normal CDF
  - Sizing: increasing Kelly as confidence grows
  - Fee consideration: round-trips pay double fees → need higher edge threshold

**A6b — Adaptive Kelly + Edge Auto-Tuning**:
- Start at 0.15x Kelly (matching existing Kalshi weather bot baseline)
- Apply Baker-McHale shrinkage correction based on calibration sample size
- Constrain by drawdown: f ≤ 2 × S² × D_max (S = rolling Sharpe)
- Edge threshold auto-adjustment:
  - Minimum = 2 × round_trip_fees + estimation_uncertainty_buffer
  - Maker orders at 50¢: minimum ~2¢ edge. Taker orders: minimum ~5¢ edge
  - Auto-widen by 1pp when rolling 50-trade PnL turns negative
  - Auto-tighten by 0.5pp when rolling PnL exceeds 2× threshold

**A6c — Cross-Station Portfolio Risk**:
- Maximum portfolio heat: 15-25% of bankroll at risk at any time
- Per-station limit: 5-10% of bankroll per station-day
- Correlation adjustment: for correlated stations (same cluster), add exposure as concentrated
- Daily loss circuit breaker: 3% of bankroll
- Drawdown scaling: half-size at 10% drawdown, quarter at 20%, stop at 30%

**A6d — Cross-Bracket Arbitrage Detection**:
- Monitor bracket strips for probability sum violations (sum of bracket prices ≠ 1.00)
- When sum of best-asks < $1.00 minus total fees: flag pure arbitrage
- Low priority — opportunities are thin and fast

### Success criteria
- improved EV after fees from mode separation
- reduced drawdown from risk controls
- adaptive Kelly shows better risk-adjusted returns than fixed fraction

---

## Phase A7 — Automated Health Monitoring + Self-Healing

### Goal
Detect model degradation, trading regime shifts, and concept drift before they cost money.

### Build

**A7a — Rolling Forecast Quality Monitors**:
- Rolling 30-day Brier score per station, per source
- Alert at +0.02 from baseline
- Per-signal Brier score decomposition (calibration + resolution separately)
- Per-source CRPS tracking (BOA handles reweighting, but we need to detect systemic issues)

**A7b — Trading Performance Regime Detection**:
- BOCPD on daily P&L series (apply your own changepoint detector to your own performance)
- When P&L regime shift detected: reduce position sizes, widen edge thresholds, enter diagnostic mode
- Track rolling 30-day Sharpe ratio

**A7c — Concept Drift Detection**:
- Population Stability Index (PSI) on feature distributions vs. training data
- Detect seasonal drift (winter calibration ≠ summer)
- Trigger weekly auto-recalibration of EMOS parameters using most recent 30-60 days
- Monitor NWS model upgrades (HGEFS deployment, GFS updates) — each changes error characteristics

**A7d — Signal Deprecation**:
- If a signal's resolution component drops below noise floor for 2+ weeks, auto-downweight
- BOA handles this for forecast sources; extend to non-forecast signals
- Weekly audit log of signal contributions to winning vs losing trades

### Success criteria
- degradation detected before material P&L impact
- seasonal transitions handled smoothly
- no surprise model performance collapses

---

## Phase A8 — New Data Source Ingestion

### Goal
Add cloud coverage, expanded mesonet, and radar data to improve inference quality.

### Build

**A8a — Cloud Coverage Ingestion**:
- Parse METAR cloud observations from existing NWS obs feed
- Add Open-Meteo cloud cover fields to forecast request
- Add GOES-19 Clear Sky Mask (ACM) ingestion from AWS S3 (`noaa-goes19` bucket)
- Add RTMA analyzed cloud cover via Herbie (`noaa-rtma-pds` bucket)
- Use FAWN solar radiation as immediate cloud proxy (already in DB)

**A8b — Expanded Surface Observation Network**:
- Expand IEM nearby station queries to include more networks per trading station
- Add direct state mesonet APIs: NYSM (NY), NJWXNET (NJ), TexMesonet (TX)
- Consider MADIS registration for broadest multi-network coverage

**A8c — MRMS Radar for Outflow Detection**:
- Ingest MRMS composite reflectivity within 50km of trading stations
- Detect approaching convection → shift temperature distribution + widen uncertainty
- 15-30 minute early warning window for outflow events

**A8d — HRRR + Obs Archiving for Future Analog Ensemble**:
- Begin systematic archiving of HRRR forecasts paired with station observations
- After 12-18 months: implement NCAR Analog Ensemble (AnEn) method
- AnEn finds N most similar historical forecasts → uses corresponding obs as ensemble members
- Intrinsically calibrated, no parametric assumptions

---

## Phase A9 — Rollout and Promotion to Live

### Required rollout order
1. replay
2. paper trading
3. live shadow
4. partial live authority
5. full live authority

### Promotion gates
Option A components should only graduate when they show:
- better remaining-target skill
- better or equal bracket calibration
- better EV after fees
- acceptable turnover and drawdown
- operational stability

### Failure rule
If calibration degrades materially or state estimation becomes unstable, promotion stops and the system falls back to the previous stable layer.

---

# Option C — Shadow / R&D Roadmap

## Option C Summary

Option C is the high-upside research lane exploring more expressive modeling:
- richer regime discovery
- richer source gating
- more expressive learned nonlinear state-space models
- DS3M as a long-run contender
- DMA as a simpler shadow challenger
- Conformal prediction for distribution-free coverage

---

## Phase C0 — Shared Archive and Replay Compatibility

Use the same canonical data substrate as Option A.

---

## Phase C1 — Advanced Offline Regime Discovery

### Build
- Sticky HDP-HMM (Fox et al.): auto-discovers number of persistent regimes
- Python: `pyhsmm` package (Matthew Johnson, MIT license)
- Potentially recurrent sticky HDP-HMM later
- Compare discovered regimes to live regime catalog

### Questions to answer
- Are the live regimes missing meaningful states?
- Are some live regimes redundant?
- Are transition probabilities asymmetric or context-dependent?

---

## Phase C2 — Mixture-of-Experts / Dynamic Gating Shadow

### Build
Shadow gating layer conditioned on state / regime / time-to-settlement / recent source behavior.
Possible forms: probabilistic gating, neural gating, mixture-of-experts head.

### Overfitting guard
With ~100-200 days of data, use regularization heavily. Cross-validate on time-forward splits only.

---

## Phase C3 — DS3M Shadow Challenger

### Build
Deep Switching State Space Model as a serious future challenger.
- Combines deep learning (neural network emissions) with state-space switching (discrete latent regimes)
- Related to but distinct from S4/Mamba (those are continuous SSMs; DS3M has discrete regime switches)
- Training: needs substantial sequential data (target 6+ months of hourly obs + forecasts)
- No trade authority until it earns promotion

---

## Phase C4 — DMA Shadow Challenger

### Build
Dynamic Model Averaging (Raftery et al. 2010) as a simpler challenger to BOA:
- Time-varying model weights + time-varying parameters
- May outperform BOA in non-stationary environments (regime shifts, seasonal transitions)
- Lower implementation complexity than DS3M
- Good benchmark: if DMA can't beat BOA, DS3M needs to do much better

---

## Phase C5 — Conformal Prediction Overlay

### Build
Distribution-free coverage guarantees for bracket probabilities:
- Adaptive Conformal Inference (ACI) provides time-series-appropriate prediction intervals
- Can be used as a calibration check on EMOS outputs
- If EMOS says P(bracket) = 0.70 but conformal coverage says the interval should be wider, flag it
- Low implementation cost, high diagnostic value

---

## Phase C6 — Shadow Challenger Evaluation + Promotion

### Required comparison dimensions
Same as Option A: CRPS, log score, Brier, calibration, EV after fees, stability across regimes.

### Promotion rules
No Option C component gets promoted because it is novel. Only if measurably better, stable, calibratable, operationally maintainable, and understandable enough to control risk. A challenger should survive **30-60 climate days** of shadow evaluation before live authority.

---

# Evaluation Phase — Option A vs Option C

## Evaluation design

### Shared replay windows
Evaluate by: full sample, rolling recent, regime buckets, early/mid/late day, high vs low markets, calm vs break-heavy days.

### Compare three levels
1. **Forecast quality** (CRPS, MAE)
2. **Probability quality** (Brier, calibration, log score)
3. **Trade quality** (EV after fees, Sharpe, drawdown)

A contender that improves one but damages the other two should not be promoted casually.

## Kill conditions
- calibration collapse
- severe overtrading
- unstable outputs day-to-day
- dependence on features not reliably available live
- unexplained degradation in late-day decision quality

---

# AI Review Policy

## Where AI review is useful
- summarizing sticky HDP proposals
- suggesting regime merges/splits
- surfacing recurring failure clusters
- drafting diagnostics and comparisons

## Where AI review should not have automatic authority
- changing live regime catalog without acceptance
- changing promotion thresholds silently
- swapping live architecture lanes without evaluation

---

# Repo / Component Ownership

## `miami-project`
Owns: data collection, archived state, inference/orchestration, replay harness, regime analysis, LETKF/BOCPD/EMOS/HDP work, bracket probability generation.

## `weather-trader`
Owns: portfolio middleman, sizing and risk policy, execution logic, live vs paper execution, downstream trade selection from orchestrator outputs.

## Shadow artifacts
Shadow research outputs may live in analyzer/research modules in `miami-project` but should not bypass the canonical replay/evaluation framework.

---

# Anti-Drift Checklist for Future Agents

Before making a major architecture change, ask:
- Is this work for **Option A live** or **Option C shadow**?
- Does it preserve the orchestrator / trader / executor separation?
- Does it use the canonical replay and archive?
- Does it change calibration, regime behavior, or state estimation?
- Is there a clear promotion path or is this just experimental?
- If experimental, is it safely contained in the shadow lane?

If the answer is unclear, stop and clarify before continuing.

---

# Final Operating Principle

The roadmap is deliberately two-track:
- **Option A** should become the best trustworthy live system.
- **Option C** should become the best ambitious challenger.

The goal is not to pick one forever. The goal is to let the structured system earn real money while the research system tries to beat it honestly. When Option C is truly better, it can be promoted. Until then, Option A gets the production seat.
