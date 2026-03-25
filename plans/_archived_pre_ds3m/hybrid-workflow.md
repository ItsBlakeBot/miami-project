# Hybrid Workflow — Raw Data + Reviews + LLM Labels + Production Bot

## Why this exists

The Miami project should not choose between:

1. **raw structured weather/market data**, and
2. **human-readable daily reviews**.

The best approach is hybrid:

- **raw data** trains and drives the bot
- **reviews** define regimes, signal families, and case-study lessons
- **LLMs** help extract labels and compare cases
- the **live bot** remains deterministic / probabilistic / auditable

This file is the operational workflow for that setup.

Companion planning artifacts:
- `plans/repo-blueprint.md` — canonical repo cleanup/refactor map
- `plans/bot-architecture.mmd` — Mermaid view of the intended architecture

Use this document for workflow and artifact flow.
Use `repo-blueprint.md` for file ownership and cleanup decisions.

---

## Division of labor

## 1. Collector DB = source of truth

Primary source for:
- observations
- model forecasts
- market snapshots
- nearby stations
- FAWN
- settlements

This is where the bot should learn from.

### DB answers
- what happened, exactly?
- when did it happen?
- what did the models say then?
- what did the market price then?
- what did KMIA CLI settle at?

---

## 2. Daily reviews = human/LLM regime annotations

Canonical directory:
- `reviews/`

Daily reviews are useful for:
- primary regime labels
- path class labels
- signal family naming
- clamp/lock states
- traps/opportunities worth remembering
- qualitative failure review

Daily reviews are **not** the live feature source for the production bot.

### Reviews answer
- what kind of day was this?
- which signal family mattered?
- which mechanism actually drove the settle?
- what lesson should be encoded for future cases?

---

## 3. Extracted review labels = bridge layer

Generated artifact:
- `analysis_data/review_labels.ndjson`

Built by:
- `miami-extract-review-labels reviews --output analysis_data/review_labels.ndjson`

This converts narrative reviews into structured fields like:
- `primary_regime`
- `background_regime`
- `path_class`
- `signal_labels`
- `clamp_labels`
- `best_high_expression`
- `best_low_expression`

### Review labels answer
- what should the raw-data rows be tagged with?

---

## 4. Training snapshots = structured learning rows

Generated artifact:
- `analysis_data/training_snapshots.ndjson`

Built by:
- `miami-build-training-snapshots`

Each row represents a fixed LST checkpoint within a climate day and includes:
- latest obs state
- running high/low
- naive model consensus at that checkpoint
- settlement truth
- review-derived regime/path labels
- signal/clamp labels from the review side

### Training snapshots answer
- what did the bot know at 1 AM / 4 AM / 10 AM / 7 PM?
- what residual remained vs the model baseline?
- what regime was that snapshot part of?

---

## 5. Production bot = structured inference only

The live production bot should consume:
- collector DB / live feeds
- calibrated source-family baselines
- regime classifier
- signal families
- station-transfer features
- short-range updater features

The live bot should **not** consume free-form summaries as direct features.

### Production bot answers
- what is the current predictive distribution?
- which brackets are mispriced?
- should we signal or trade?

---

## 6. LLMs = analysis helpers, not the weather engine

Use LLMs for:
- extracting labels from reviews
- clustering similar days
- proposing new regime distinctions
- summarizing failure modes
- assisting architecture refinement

Do **not** use LLMs as the only production inference layer.

---

## Data flow

```text
collector DB
  ├─> canonical daily review (markdown)
  ├─> review label extraction (structured JSON)
  ├─> training snapshot builder (structured JSON)
  └─> production inference features

review labels + training snapshots
  └─> backtests / calibration / residual learning

production inference
  └─> bracket pricing / edge detection / execution
```

---

## Why this is better than summaries alone

Summaries are useful, but they are:
- lossy
- hindsight-heavy
- hard to backtest directly
- bad at preserving exact timing/state transitions

Raw checkpoint data preserves:
- exact timing
- live model state
- live market state
- live local observations
- residual target values

So the right workflow is:
- **reviews teach ontology**
- **snapshots teach the model**

---

## Existing scaffold added to the project

### Settlement utilities
- `src/settlement/climate_clock.py`

### Review label extraction
- `src/analysis/review_labels.py`
- CLI: `miami-extract-review-labels`

### Training snapshot builder
- `src/analysis/build_training_snapshots.py`
- CLI: `miami-build-training-snapshots`

### Backtest scaffold
- `src/analysis/backtest_residual_bot.py`
- CLI: `miami-backtest-residual-bot`

### Output artifacts
- `analysis_data/review_labels.ndjson`
- `analysis_data/training_snapshots.ndjson`
- `analysis_data/backtest_residual_results.ndjson`

---

## Minimal operating procedure

### Step 1 — keep writing canonical daily reviews
Use `reviews/YYYY-MM-DD.md` only.

### Step 2 — regenerate structured labels
```bash
miami-extract-review-labels reviews --output analysis_data/review_labels.ndjson
```

### Step 3 — regenerate checkpoint snapshots
```bash
miami-build-training-snapshots
```

### Step 4 — inspect backtests / calibration
Use snapshots as the main input to any learned or rule-based residual model.

Current scaffold command:
```bash
miami-backtest-residual-bot \
  --input analysis_data/training_snapshots.ndjson \
  --output analysis_data/backtest_residual_results.ndjson
```

This is an architecture-validation scaffold, not the final production scorer.

### Step 5 — refine reviews and labels
If a new failure mode appears, update the review template or label extraction logic.

---

## Design principles

1. **LST only** for settlement and checkpoints
2. **KMIA CLI is truth**, not generalized Miami heat
3. **Reviews are annotations, not raw features**
4. **Live bot remains modular and auditable**
5. **Every artifact should be regenerable from source data**
6. **If a component can’t be debugged independently, it is too coupled**

---

## Next recommended build steps

After this scaffold, the next real modules to implement are:

1. `engine/source_registry.py`
2. `engine/bias_memory.py`
3. `engine/quantile_combiner.py`
4. `data/provisional_qc.py`
5. `data/station_cluster.py`
6. `engine/short_range_updater.py`
7. `engine/station_transfer.py`
8. `engine/regime_classifier.py`
9. `engine/signal_families.py`
10. `engine/residual_estimator.py`

That sequence keeps the hybrid workflow modular instead of burying everything inside one giant estimator.
