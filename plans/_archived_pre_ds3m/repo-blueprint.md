# Miami Repo Blueprint

## Purpose

This is the canonical cleanup / refactor blueprint for the Miami project.

It exists to answer four questions clearly:

1. what stays as the foundation
2. what gets merged or retired
3. what still needs to be built
4. how the plan docs map to the actual repo

This blueprint is intentionally more operational than the architecture docs.
The other plans describe the *why* and the *target logic*.
This file describes the *repo shape* we want Codex/BMO to enforce.

---

## Current repo reality

The repo already has a real modular scaffold, but it is not yet fully cleaned.

### Already present and worth keeping

- `src/settlement/climate_clock.py`
- `src/analysis/review_labels.py`
- `src/analysis/build_training_snapshots.py`
- `src/engine/source_registry.py`
- `src/engine/bias_memory.py`
- `src/engine/quantile_combiner.py`
- `src/engine/short_range_updater.py`
- `src/engine/station_transfer.py`
- `src/engine/regime_classifier.py`
- `src/engine/signal_families.py`
- `src/engine/residual_estimator.py`
- `src/trading/bracket_pricer.py`
- `src/trading/edge_detector.py`
- `src/trading/policy.py`
- `src/collector/prune.py`

### Still incomplete / mismatched

- `src/engine/baseline_engine.py` was missing from the original scaffold and must exist
- pruning existed but was not wired through the package entrypoint
- legacy modules still remain beside the new modular stack
- no Mermaid repo architecture artifact existed
- execution and backtest layers are still only partially represented

---

## Canonical target structure (updated 2026-03-19)

```text
src/
  analyzer/
    build_training_snapshots.py     # snapshot builder for SKF training
    daily_data_builder.py           # assembles daily review prompt data
    daily_post_settlement.py        # nightly pipeline entry
    hdp_regime_discovery.py         # shadow HDP-SHMM regime finder
    model_scorer.py                 # MAE scoring + consensus
    obs_analyzer.py                 # cloud impact + FAWN verification
    post_settlement_calibrator.py   # CUSUM h/k + SKF mu/sigma EMA
    prompt_builder.py               # AI review prompt generator
    review_labels.py                # extract regime labels from reviews
    station_cluster.py              # inland/coastal cluster logic
    train_skf.py                    # fit SKF regime transitions

  collector/
    runner.py                       # 15-loop async orchestrator + LiveState
    entrypoints.py                  # CLI entry
    prune.py                        # DB pruning
    forward_curve.py                # 12h forecast curve builder
    config.py                       # TOML config loader
    types.py                        # Observation, MarketSnapshot, etc.
    store/
      db.py                         # SQLite store with dedup caches
      schema.py                     # table definitions
    sources/
      wethr.py                      # Wethr REST (forecasts)
      wethr_stream.py               # Wethr SSE (1-min obs)
      kalshi_rest.py                # Kalshi REST (discovery + settlements)
      kalshi_ws.py                  # Kalshi WS (orderbook deltas)
      nws.py                        # NWS (ASOS obs + CLI truth)
      openmeteo.py                  # Open-Meteo (det + ensemble + atmo)
      iem.py                        # IEM nearby stations
      ndbc.py                       # NDBC SST buoys
      fawn.py                       # FAWN Homestead

  engine/
    orchestrator.py                 # central inference pipeline coordinator
    signal_engine.py                # raw state: running high/low, trends
    baseline_engine.py              # forecast consensus → PredictiveDistribution
    short_range_updater.py          # obs vs baseline features
    station_transfer.py             # inland/coastal cluster deltas
    changepoint_detector.py         # SPECI thresholds + per-channel CUSUM
    kalman_regimes.py               # 5D Switching Kalman Filter
    regime_classifier.py            # rule-based + SKF overlay
    signal_families.py              # warm/cool tail signal activation
    residual_estimator.py           # mu shift + sigma scale
    bracket_pricer.py               # CDF integration over settlement bounds
    edge_detector.py                # Kalshi fee-aware edge: ceil(0.07×P×(1-P)×100)
    quantile_combiner.py            # PredictiveDistribution
    source_registry.py              # model → family mapping
    bias_memory.py                  # historical bias tracking

  trading/
    paper_trader.py                 # paper trading with fee-aware edge gates

plans/
  repo-blueprint.md
  bot-architecture.mmd
  PLAN-2026-03-19-PM.md

reviews/
  ai/                               # AI-generated daily summaries
  prompts/                          # prompts used to generate reviews

cron/
  daily-post-settlement.md          # nightly pipeline instructions
```

---

## Keep / merge / retire / build

## Keep as canonical foundations

### collector backbone
Keep:
- `collector/runner.py`
- `collector/store/*`
- `collector/sources/*`
- `collector/config.py`
- `collector/types.py`
- `collector/model_scorer.py`

Why:
- these are the live ingestion spine
- they hold the actual collector and DB value
- they should be trimmed, not rewritten casually

### settlement and analysis backbone
Keep:
- `settlement/climate_clock.py`
- `analysis/review_labels.py`
- `analysis/build_training_snapshots.py`

Why:
- these encode the LST/KMIA-truth framing
- they are directly aligned with the current review/training workflow

### new modular engine pieces
Keep:
- `engine/source_registry.py`
- `engine/bias_memory.py`
- `engine/quantile_combiner.py`
- `engine/short_range_updater.py`
- `engine/station_transfer.py`
- `engine/regime_classifier.py`
- `engine/signal_families.py`
- `engine/residual_estimator.py`

Why:
- this is the right decomposition for the baseline → regime → families → residual architecture

---

## Merge / refactor

### `engine/estimator.py`
Status:
- still contains older direct-estimation logic

Action:
- keep temporarily as reference and compatibility layer
- gradually shrink its responsibility
- move canonical baseline logic to `baseline_engine.py`
- move canonical residual logic to `residual_estimator.py`
- retire older direct additive heuristics once imports and tests are moved

### `engine/signal_engine.py`
Status:
- still important, but should become the raw state builder rather than a place for final belief logic

Action:
- keep as the raw feature/state assembly layer
- ensure it feeds `baseline_engine`, `short_range_updater`, `regime_classifier`, and `signal_families`

### `signals/signal_evaluator.py`
Status:
- useful as a historical scoring/reference tool, but aligned to older signal ontology

Action:
- keep for now as an evaluation utility
- treat as reference/bridge code, not canonical live-bot logic
- eventually either update to family/regime language or move to `analysis/`

---

## Retire after verification

These should not be treated as first-class canonical modules long-term.
They should be retired only after import/usage verification.

### clear duplication
- `collector/signals.py`
- `signals/signals.py`

Rule:
- pick one surviving home for any logic still needed
- do not keep two giant drift-prone copies

### likely legacy / older workflow
- `signals/daily_review.py`
- `collector/obs_analyzer.py`
- `collector/diurnal_scorer.py`
- `engine/signal_scorer.py`

Rule:
- quarantine first if uncertain
- delete only after confirming no runtime path still depends on them

---

## Build next

### required next modules
- more complete `baseline_engine.py` integration into live flow

### now added as scaffold
- `analysis/backtest_residual_bot.py`
  - initial replay harness for snapshot rows through the modular residual path
  - meant for MAE comparison and architecture validation, not final production scoring

### optional later split
If `trading/` grows materially, consider:
- `trading/order_router.py`
- `trading/position_manager.py`

But do not create these just for aesthetics before they have real responsibilities.

---

## Wiring rules

### 1. Entry point must reflect maintenance hooks
Canonical script:
- `miami-collector -> collector.entrypoints:cli_entry`

Reason:
- pruning exists and should actually run from the supported package entrypoint

### 2. Plans must match the repo, not fantasy state
Do not claim:
- no duplicates
- no legacy
- fully cleaned architecture

unless the files actually reflect that state.

### 3. Reviews remain annotations, not live weather features
Canonical location:
- `reviews/YYYY-MM-DD.md`

Use them for:
- regime labels
- path labels
- lessons
- retrieval/training annotations

Do not treat prose review text as direct live model input.

### 4. LST and KMIA are hard invariants
All architecture, backtests, snapshots, and review tooling must preserve:
- LST climate day
- KMIA CLI settlement truth

---

## Migration order

### Phase 1 — blueprint and wiring
1. create repo blueprint doc
2. create Mermaid architecture file
3. add `baseline_engine.py`
4. wire `miami-collector` entrypoint to `collector.entrypoints:cli_entry`

### Phase 2 — compatibility cleanup
1. map imports/usages of duplicated and legacy modules
2. redirect remaining references to canonical modules
3. quarantine or delete duplicate signal files only after verification

### Phase 3 — live integration
1. connect `signal_engine.py` to `baseline_engine.py`
2. ensure baseline → short-range → regime → families → residual → pricing is the canonical path
3. add smoke tests for importability and basic object flow

### Phase 4 — backtest and retirement
1. add `analysis/backtest_residual_bot.py`
2. test March 16 / 17 anchor cases through the modular path
3. retire old direct-estimator logic once parity is good enough

---

## Definition of “clean enough”

The repo should be considered structurally clean when all of the following are true:

1. one canonical home exists for signal-family logic
2. one canonical entrypoint exists for collector startup
3. baseline, regime, families, residual, and trading layers are separated
4. plan docs describe the actual repo shape
5. Mermaid architecture file exists and matches the code layout
6. legacy modules are either explicitly marked reference-only or removed

That is the target this blueprint is meant to drive.
