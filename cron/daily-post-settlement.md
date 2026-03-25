# Daily Post-Settlement Recalibration + Advisory Review — KMIA

Run this every day after CLI settlement arrives (~04:45 LST / ~09:45 UTC). Target run time: **09:00 LST (14:00 UTC)**.

## Purpose

This workflow should reflect the **current live bot stack**:
- daily recalibration via `auto_recalibrator`
- EMOS + BOA + source trust + regime catalog monitoring
- HDP shadow regime discovery for naming/discovery support
- AI review that is **advisory only** (read-only recommendations)

> Legacy CUSUM/SKF-only framing is not the focus of this cron.

All commands run from `/Users/blakebot/blakebot/miami-project` with `PYTHONPATH=src`.

## Step 1: Determine target date (`<DATE>`)

Use climate-day logic in fixed UTC-5 (LST, no DST):
- normally use **yesterday**
- if current UTC hour < 05:00, use **two calendar days ago**

## Step 2: Run the current recalibration pipeline

```bash
cd /Users/blakebot/blakebot/miami-project
PYTHONPATH=src .venv/bin/python -m analyzer.auto_recalibrator --db miami_collector.db --station KMIA --reference-date <DATE>
```

This should update/report:
- AnEn archive status
- drift detection output
- EMOS refit state (`analysis_data/emos_state.json`)
- BOA update state (`analysis_data/boa_state.json`)
- P&L regime detection
- health report (`analysis_data/health_report.json`)
- regime review/proposals (`analysis_data/regime_proposals/<DATE>.json` when applicable)

## Step 3: Run HDP shadow discovery/logging

```bash
cd /Users/blakebot/blakebot/miami-project
PYTHONPATH=src .venv/bin/python -m analyzer.hdp_regime_discovery --db miami_collector.db --date <DATE>
```

Confirm writes to SQLite table:
- `regime_labels_hdp_test`

## Step 4: Build the advisory AI review (read-only)

Create two artifacts:
1. `reviews/ai/<DATE>.md` (human narrative)
2. `reviews/ai/<DATE>.json` (structured payload)

The review must evaluate:
- regime fit for core + catalog + EMOS behavior
- novelty/unspecified regime judgment
- naming suggestions for core/catalog/EMOS/HDP regimes
- projection/calibration quality (mu/sigma behavior)
- BOA/source-trust plausibility and drift/health context
- paper trading quality (entries/exits/marks/P&L)

**Critical constraint:** advisory-only. Do not directly mutate production code/config in this review step.

## Step 5: Optional parse/store (best effort)

```bash
cd /Users/blakebot/blakebot/miami-project
PYTHONPATH=src .venv/bin/python -m analyzer.ai_review_parser parse --input reviews/ai/<DATE>.md --db miami_collector.db
```

If parser format mismatch occurs, report it and continue (do not fail the entire cron solely for parse mismatch).

## Step 6: Verify inference still runs

```bash
cd /Users/blakebot/blakebot/miami-project
PYTHONPATH=src .venv/bin/python -m engine.orchestrator --db miami_collector.db
```

## Final summary requirements

Report must include:
- target date used
- recalibrator outputs (drift, EMOS mode, BOA update, P&L regime, health alerts)
- HDP status (`regime_labels_hdp_test` write confirmation)
- regime-fit + novel-regime assessment
- naming suggestions (core/catalog/EMOS/HDP)
- advisory recommendations (read-only)
- exact files written
- any failures with exact command + stderr
