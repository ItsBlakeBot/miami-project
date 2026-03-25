# Weather Trading — Opus Handoff (2026-03-22)

Owner handoff from: BMO
Target: Opus
Status date: 2026-03-22

---

## Canonical Plan Files (use these only)

- `/Users/blakebot/blakebot/miami-project/plans/weather-trading-architecture-roadmap.md`
- `/Users/blakebot/blakebot/miami-project/plans/weather-trading-ticketized-build-order.md`
- `/Users/blakebot/blakebot/miami-project/plans/weather-trading-implementation-checklist.md`

Do **not** use older workspace plan copies as source-of-truth.

---

## What’s Completed (confirmed)

### Foundation / Shared
- `T0.1` DONE — archive audit + report artifacts
- `T0.2` DONE — canonical replay/report path consolidated
- `T0.3` DONE — canonical bundle includes:
  - remaining-target MAE/RMSE/CRPS
  - TTL cuts
  - regime cuts
  - trade-quality cuts
  - EV rollups
  - worst-day diagnostics
- `T0.4` DONE — source-trust backfill extractor + artifacts
- `T1.1` DONE — dynamic weighting + baseline integration + tests
- `T2.1` DONE — BOCPD + run-length outputs + comparison bundle

### Expanded modules landed (per current repo)
- `src/engine/emos.py`
- `src/engine/boa.py`
- `src/engine/letkf.py`
- `src/engine/cloud_cover.py`
- `src/engine/regime_catalog.py`
- `src/engine/observation_operators.py`
- `src/engine/bracket_arbitrage.py`
- `src/collector/sources/hrrr.py`
- `src/collector/sources/goes.py`
- `src/analyzer/health_monitor.py`
- `src/analyzer/pnl_regime_detector.py`
- `src/analyzer/drift_detector.py`
- `src/analyzer/signal_deprecation.py`
- `src/analyzer/anen_archiver.py`
- `src/analyzer/auto_recalibrator.py`

### Data-ingestion expansion (confirmed)
- Runner now starts **17** async loops
- FAWN stations expanded to 10 station IDs
- IEM radius expanded to 125 miles (~200 km)

---

## Reconciliations / Fixes completed by BMO

1. **Status over-claim reconciliation** in ticketized doc:
   - `T1.4` moved to `IN_PROGRESS`
   - `T1.5` moved to `IN_PROGRESS`
   - Added explicit remaining promotion gates for each

2. **Runner log hygiene fix**:
   - `src/collector/runner.py` log text corrected from 15 to 17 loops

3. **BOA evidence audit added**:
   - `analysis_data/boa_family_weight_audit.md`
   - Documents side-specific BOA behavior:
     - LOW side currently favors openmeteo overall
     - HIGH side currently favors wethr MOS experts
   - Also distinguishes BOA from source-trust multipliers

4. **Source-trust cadence plumbing present**:
   - `src/analyzer/source_trust_refresh.py`
   - `analysis_data/source_trust_refresh.json`
   - `analysis_data/source_trust_state.json`

---

## Current Active Gates (where to continue)

### `T1.4` — Adaptive source trust multipliers (**IN_PROGRESS**)
What is done:
- Backfill-prior loader + baseline integration
- Persistent multiplier state
- Max-step drift cap
- Sparse-window authority caps
- Rolling refresh runner

What remains:
- Validate stability across multiple lookback windows/anchor dates
- Define explicit evidence threshold to promote from sample-limited to done
- Current real-data status is still sample-thin (9 covered days in recent run)

### `T1.5` — BOA online source weighting (**IN_PROGRESS**)
What is done:
- BOA core exists with sleeping expert protocol
- Persisted BOA state exists (`analysis_data/boa_state.json`)

What remains:
- Surface BOA weights/history in canonical replay outputs
- Demonstrate replay CRPS gain vs static/rolling weighting baseline
- Pass promotion gate with auditable evidence

---

## Key Evidence Artifacts

- `/Users/blakebot/blakebot/miami-project/analysis_data/source_trust_backfill.json`
- `/Users/blakebot/blakebot/miami-project/analysis_data/source_trust_refresh.json`
- `/Users/blakebot/blakebot/miami-project/analysis_data/source_trust_state.json`
- `/Users/blakebot/blakebot/miami-project/analysis_data/boa_state.json`
- `/Users/blakebot/blakebot/miami-project/analysis_data/boa_family_weight_audit.md`
- `/Users/blakebot/blakebot/miami-project/analysis_data/changepoint_compare.json`
- `/Users/blakebot/blakebot/miami-project/analysis_data/canonical_replay_bundle.json`

---

## Suggested Next Execution Order for Opus

1. **Finish `T1.4` gate** (stability/evidence threshold)  
2. **Finish `T1.5` gate** (canonical replay BOA output + CRPS delta proof)  
3. Then proceed to next priority ticket from ticketized plan (currently `T4.2` in that file)

---

## Guardrails

- Keep **Option A live / Option C shadow** separation
- Do not create a second replay comparison harness
- Use canonical bundle path for reporting
- If a claim is sample-limited, label it explicitly instead of promoting early

---

## One-line resume prompt (optional)

"Resume from `T1.4`/`T1.5` IN_PROGRESS gates in `plans/weather-trading-ticketized-build-order.md`; validate T1.4 stability thresholds, then integrate BOA history/weights into canonical replay and prove CRPS gain before promoting T1.5 to DONE."
