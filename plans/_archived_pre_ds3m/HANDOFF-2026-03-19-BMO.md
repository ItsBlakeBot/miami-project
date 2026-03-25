# Handoff — 2026-03-19 — BMO

## What was done in this pass

### New files added
- `src/engine/baseline_engine.py`
- `src/analysis/backtest_residual_bot.py`
- `plans/repo-blueprint.md`
- `plans/bot-architecture.mmd`
- `tests/test_baseline_engine.py`
- `tests/test_backtest_residual_bot.py`

### Existing files updated
- `pyproject.toml`
  - `miami-collector` now points to `collector.entrypoints:cli_entry`
  - added `miami-prune-db`
  - added `miami-backtest-residual-bot`
- `plans/residual-signal-bot-v1.md`
- `plans/hybrid-workflow.md`
- `plans/trading-bot-v1.md`
- `src/engine/residual_estimator.py`
  - now imports shared `BaselineBelief` from `baseline_engine.py`
- `src/collector/signals.py`
  - replaced duplicate giant implementation with compatibility wrapper to `signals.signals`
- `src/signals/daily_review.py`
- `src/collector/diurnal_scorer.py`
- `src/engine/signal_scorer.py`
  - all marked legacy/reference-only in module docstrings

---

## Current architectural status

The repo now has a real documented modular path:

`collector DB -> signal_engine -> baseline_engine -> short_range_updater / station_transfer -> regime_classifier -> signal_families -> residual_estimator -> trading/*`

But this is still **partial convergence**, not complete cleanup.

### Important truth
- the blueprint exists now
- the Mermaid architecture exists now
- the baseline engine scaffold exists now
- the backtest scaffold exists now
- duplicate / legacy surfaces are **not fully removed yet**
- live collector flow still uses older pieces in places

---

## Safe conclusions from the audit so far

### Canonical foundations worth keeping
- `collector/runner.py`
- `collector/store/*`
- `collector/sources/*`
- `collector/model_scorer.py`
- `settlement/climate_clock.py`
- `analysis/review_labels.py`
- `analysis/build_training_snapshots.py`
- `engine/source_registry.py`
- `engine/bias_memory.py`
- `engine/quantile_combiner.py`
- `engine/short_range_updater.py`
- `engine/station_transfer.py`
- `engine/regime_classifier.py`
- `engine/signal_families.py`
- `engine/residual_estimator.py`
- `engine/baseline_engine.py`
- `trading/bracket_pricer.py`
- `trading/edge_detector.py`
- `trading/policy.py`

### Legacy / duplication still present
- `signals/signals.py` is still the real large legacy signal implementation
- `collector/signals.py` is now only a wrapper
- `signals/daily_review.py` appears orphaned / legacy
- `collector/diurnal_scorer.py` appears orphaned / legacy
- `engine/signal_scorer.py` appears orphaned / legacy
- `collector/obs_analyzer.py` is still referenced by `collector/runner.py`
- `engine/estimator.py` still exists as older direct-estimation logic

---

## Things verified

### Imports / compile
Smoke compile succeeded for:
- baseline engine
- residual estimator
- backtest scaffold
- legacy-wrapper modules

### Tests
Ran in project venv:
- full suite passed: `35 passed`

### Backtest scaffold
Command now available:
```bash
miami-backtest-residual-bot \
  --input analysis_data/training_snapshots.ndjson \
  --output analysis_data/backtest_residual_results.ndjson
```

Result on current snapshot file at time of handoff:
- naive MAE: `3.171°F`
- adjusted MAE: `3.109°F`
- improvement: `0.062°F`

Interpretation:
- scaffold is executable
- modular path is live enough to measure
- performance gain is modest and should not be oversold

---

## Recommended next moves

### 1. Tighten canonical ownership between old and new signal layers
Main area to resolve next:
- `signals.signals`
- `engine.signal_engine`
- `collector.obs_analyzer`
- `engine.estimator`

Goal:
- decide what remains canonical for live state-building
- migrate useful pieces deliberately
- shrink older modules without breaking collector runtime

### 2. Keep `engine.estimator.py` as reference for now, but keep de-powering it
- do **not** delete blindly yet
- it still contains old direct estimation logic and useful ideas
- the canonical path should keep moving toward `baseline_engine.py` + `residual_estimator.py`

### 3. Avoid blind deletion
Before deleting anything, verify:
- imports
- runtime paths
- scripts/cron/service usage

Especially be careful with:
- `signals/signals.py`
- `collector/obs_analyzer.py`
- anything touched by `collector/runner.py`

### 4. Keep plans honest
`repo-blueprint.md` is now the canonical cleanup/refactor map.
If the repo changes materially, update:
- `plans/repo-blueprint.md`
- `plans/bot-architecture.mmd`
- whichever architecture plan is now lagging reality

---

## Hard invariants to preserve
- LST climate-day logic only
- KMIA CLI settlement truth only
- reviews are annotations / labels, not free-form live features
- do not claim “fully cleaned” until duplicate / legacy surfaces are actually retired

---

## If picking up immediately
Start by reading:
1. `plans/repo-blueprint.md`
2. `plans/bot-architecture.mmd`
3. `plans/hybrid-workflow.md`
4. `src/engine/baseline_engine.py`
5. `src/analysis/backtest_residual_bot.py`
6. `src/collector/runner.py`
7. `src/signals/signals.py`
8. `src/engine/signal_engine.py`

That should get you back into the state quickly.
