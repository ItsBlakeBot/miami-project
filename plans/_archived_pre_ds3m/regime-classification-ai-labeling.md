# Regime Classification & AI Labeling System

## Context

The miami-project uses a bias-corrected model mean with hardcoded signal adjustments. The gap: no awareness that different weather regimes produce systematically different model errors. A post-frontal day needs different corrections than a sea-breeze day. This system builds:

1. **Post-settlement AI labeling** — data gathering, prompt construction, output parsing, storage (LLM call handled externally via OpenClaw cron)
2. **Switching Kalman Filter** — real-time regime detection trained from AI-labeled data
3. **Regime-aware adjustments** — SKF probabilities feed mu/sigma corrections into the pricing pipeline

Both regimes AND signal families are fully flexible — the AI can create, merge, or split them freely. A day can have any number of regimes across its phases.

---

## Implementation Status

All 8 build steps are **COMPLETE**. 35 tests passing. All modules import cleanly. End-to-end prompt generation verified for March 16 (~55K tokens) and March 17 (~74K tokens). SKF running with 4 default regime models. 4 human reviews backfilled into `regime_labels`.

---

## Architecture

```
Post-Settlement (1x/day)         Intraday (every cycle)
┌──────────────────────┐         ┌──────────────────────┐
│ daily_data_builder   │         │ signal_engine        │
│   ↓                  │         │   ↓                  │
│ prompt_builder       │         │ SwitchingKalmanFilter│
│   ↓                  │         │   ↓                  │
│ [OpenClaw → Claude]  │         │ regime_classifier    │
│   ↓                  │         │   ↓                  │
│ ai_review_parser     │         │ residual_estimator   │
│   ↓                  │         │   ↓                  │
│ regime_labels table  │────────→│ bracket_pricer       │
│   ↓                  │         └──────────────────────┘
│ train_skf            │
│   ↓                  │
│ skf_config.json      │────────→ (loaded by SKF at startup)
└──────────────────────┘
```

**Data flow:** Post-settlement AI reviews label each day → labels train the SKF → SKF runs intraday producing regime probabilities → probabilities gate mu/sigma adjustments in the residual estimator → adjustments feed the bracket pricer.

---

## Step 1: Schema — `regime_labels` table [DONE]

**Files modified:** `src/collector/store/schema.py` (version bumped to 11), `src/collector/store/db.py`

```sql
CREATE TABLE IF NOT EXISTS regime_labels (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    station TEXT NOT NULL,
    target_date TEXT NOT NULL,
    regimes_active TEXT NOT NULL,       -- JSON array of regime strings (unlimited)
    path_class TEXT,
    confidence_tags TEXT,              -- JSON array of strings
    phase_summary TEXT,                -- JSON: [{start_hour_lst, end_hour_lst, description, regime}]
    model_performance TEXT,            -- JSON: {model_key: {high_error, low_error, notes}}
    signal_labels TEXT,                -- JSON: [{time_lst, signal, label, family, description}]
    signal_families_active TEXT,       -- JSON: [{family, strength, members, description}]
    threshold_recommendations TEXT,    -- JSON: [{parameter, current, recommended, rationale, confidence}]
    review_path TEXT,
    review_source TEXT DEFAULT 'ai',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(station, target_date)
)
```

`regimes_active` is a JSON array — a day can have any number of regime labels. The `phase_summary` carries which regime applies to which time window, so the per-phase regime assignments are the real source of truth.

**DB methods added:** `upsert_regime_label()`, `get_regime_labels()`, `get_recent_regime_labels()`

**Current contents:** 4 rows backfilled from human reviews (2026-03-14 through 2026-03-17).

---

## Step 2: Data Builder — `src/analyzer/daily_data_builder.py` [DONE]

Pulls ALL collected data for a completed climate day into a `DailyDataPackage` dataclass. No data is deemed useless — everything the collector gathers gets included.

**Data rendering: OHLC hourly with RAPID_CHANGE injection.**

Rather than dumping raw obs (which produced 9.5MB / ~2.4M tokens — impossible to fit in any prompt), the data builder aggregates surface obs into OHLC-style hourly rows (open/high/low/close for temp, dew, pressure; wind direction range; max gust). When a rapid change crosses thresholds within an hour, an extra `RAPID_CHANGE` row is injected showing the exact sub-hour transition window.

**Transition detection thresholds:** 3°F temp, 3°F dew, 45° wind, or 0.5 hPa pressure. Intentionally sensitive — the AI filters noise, we don't want to miss precursor signals.

**Example (March 16 frontal passage):**
```
| 20:00 | 80.6 | 80.6 | 78.8 | 78.8 | 69.8 | 69.8 | 200→270 | 5.3 |     | 1013.2    | 50 |              |
| 21:00 | 78.8 | 78.8 | 73.4 | 73.4 | 69.8 | 69.8 | 260→310 | 9.3 | 28  | 1013→1014 | 75 |              |
| 21:15→21:55 | 78.8 | 78.8 | 73.4 | 73.4 | 69.8 |   | 240→310 |  | 28  | 1013.5    |    | RAPID_CHANGE |
| 22:00 | 73.4 | 73.4 | 69.8 | 69.8 | 64.4 | 64.4 | 310→0   | 6.1 | 24  | 1014.2    | 75 |              |
```

The 9°F crash in 95 minutes is fully preserved. The OHLC format shows directionality (open→close) that min/max alone would lose.

**Aggregation applied to large tables:**
- **Surface obs:** OHLC hourly + RAPID_CHANGE injection (159 raw → ~30 rows)
- **Nearby stations:** OHLC per station per hour (3390 raw → ~500 rows with temp_high/low/delta_mean)
- **Forward curves:** Sampled to 96 rows max (300 raw → 96)
- **Atmospheric data:** Sampled to 96 rows max (3066 raw → 96)
- **Pressure levels:** Sampled to 96 rows max (1104 raw → 96)
- **Market snapshots:** 8 key price points per ticker (100K raw → ~300)
- **FAWN, SST, model forecasts, consensus:** Kept at full resolution (already small)

**Resulting prompt sizes:** March 16 = ~55K tokens. March 17 = ~74K tokens. Both fit comfortably in any modern LLM context window.

**Contents gathered (all sources):**
- Surface obs (OHLC + transitions)
- Model forecasts vs CLI settlement (every model, per-model error, HIGH/LOW scored independently, run_time included)
- Forward curve evolution (sampled snapshots showing how projections evolved)
- Atmospheric data (CAPE, PW, solar, BL height, soil temp/moisture, precip)
- Pressure levels (925/850/700/500 hPa temps, winds, geopotential, RH)
- FAWN Homestead obs (15-min native: air temp, dew, wind, solar, rain, soil temp)
- Nearby stations (per-station hourly OHLC with temp delta vs KMIA)
- SST buoy data (water temp, air temp, wind, pressure per buoy)
- Market price progression (sampled per ticker)
- Model consensus (bias-adjusted weights and consensus evolution)
- Previous 3 days' regime labels

---

## Step 3: Prompt Template — `src/analyzer/prompts/daily_review.md` [DONE]

Standalone instruction document for the LLM. Uses `{{PLACEHOLDER}}` markers for dynamic content insertion.

**Key sections:**

1. **Role & constraints** — climate day midnight-midnight LST (UTC-5 fixed), CLI is only truth, HIGH/LOW scored separately

2. **Calibrated confidence, not paralysis:**
   - CAN recommend changes from a single day — but be honest about confidence
   - Has access to previous 3 days' reviews and recommendations
   - First-time pattern: `low` confidence. Seen 2+ times: `medium`. Confirmed 3+ times: `high`.
   - Downstream system uses confidence level to scale how aggressively it applies changes
   - Prefer small incremental adjustments (10-20%) at low confidence, larger at high
   - Create new regime/family labels rather than forcing into existing categories
   - No data is useless

3. **Current regime definitions** — dynamically inserted from config dict mirroring `regime_classifier.py`

4. **Current signal families** — dynamically inserted, explicitly states AI is NOT limited to existing families

5. **Regime classification** — NOT limited to any fixed number per day, each phase gets its own, free to create new types in snake_case, consider ALL data sources

6. **Signal family instructions** — can define new families, split existing ones, merge co-occurring ones

7. **Phase breakdown** — no limit on number of phases

8. **Signal labeling** — per-signal events with time, name, label, family, description

9. **Model performance** — per-model error vs CLI, HIGH and LOW separately, explain WHY

10. **Threshold adjustment recommendations** — for both regime classifier AND signal family weights

11. **Previous days' context** — dynamically inserted

12. **Today's data** — dynamically inserted full DailyDataPackage

13. **Required output format** — 7 YAML blocks:
    - Structured summary (`regimes_active` list, `path_class`, `confidence_tags`)
    - Phase breakdown (unlimited phases, each with regime)
    - Signal labels (with family assignment)
    - Signal families active (with strength, member signals, descriptions — including new families)
    - Model performance
    - Threshold recommendations
    - Narrative review (full prose markdown: executive summary, ground truth, regime assessment, phase breakdown, signal timeline, signal-to-adjustment notes, model performance, spatial confirmation, ocean influence, upper air, market timing, bottom line)

---

## Step 4: Prompt Builder — `src/analyzer/prompt_builder.py` [DONE]

**Function:** `build_prompt(data_text, target_date, recent_context) -> str`

- Reads template from `prompts/daily_review.md`
- Substitutes `{{REGIME_DEFINITIONS}}` with current thresholds from config dict
- Substitutes `{{SIGNAL_FAMILY_DEFINITIONS}}` with current family definitions
- Substitutes `{{DAILY_DATA}}` with rendered DailyDataPackage
- Substitutes `{{PREVIOUS_DAYS_CONTEXT}}` with recent review summaries
- Substitutes `{{TARGET_DATE}}` with the target date
- Falls back to inline minimal template if file missing

**Helper:** `build_recent_context(recent_labels) -> str` — formats regime_labels rows into context text

**Helper:** `save_prompt(prompt, target_date, output_dir) -> Path` — writes to disk for OpenClaw

---

## Step 5: Output Parser — `src/analyzer/ai_review_parser.py` [DONE]

Parses LLM output (7 YAML blocks + narrative markdown) into `ParsedReview` dataclass.

**Storage:**
- Upserts into `regime_labels` table (JSON-serializes list/dict fields)
- Writes narrative markdown to `reviews/ai/{target_date}.md`
- Appends threshold recommendations to `analysis_data/threshold_history.jsonl`

**Backfill:** `backfill_from_human_reviews(reviews_dir, db_path)` — parses existing human reviews, extracts regime/signal info, inserts into regime_labels with `review_source='human'`. 4 reviews backfilled successfully.

**Robustness:** Lenient YAML extraction — finds ```yaml fences, tries `yaml.safe_load()`, falls back to regex per block independently. A failure in one section doesn't break others.

**CLI:**
```bash
python -m analyzer.ai_review_parser parse --input raw_output.txt --db miami_collector.db
python -m analyzer.ai_review_parser backfill --reviews-dir reviews/ --db miami_collector.db
```

---

## Step 6: Switching Kalman Filter — `src/engine/kalman_regimes.py` [DONE]

**State vector** (5D):
```
x = [temp_f, dew_f, pressure_hpa, wind_dir_sin, wind_dir_cos]
```
Wind direction encoded as sin/cos to handle 360°→0° wraparound.

**Per-regime model** (`RegimeModel`):
- `name`, `A` (5x5 transition), `b` (5x1 bias), `Q` (process noise), `R` (measurement noise)
- `self_transition_prob` — P(stay in this regime)
- `mu_shift_high/low`, `sigma_scale_high/low` — learned per-market adjustments
- `active_families: list[str]` — signal families that typically activate

**Main class** `SwitchingKalmanFilter`:
- `__init__(config_path)` — loads from `analysis_data/skf_config.json` if exists, else uses defaults
- `reset(target_date)` — uniform priors for new climate day
- `update(obs, hours_elapsed) -> SKFState` — full predict/update cycle across all regime models with Bayesian probability update via log-sum-exp
- `predict_trajectory(hours_ahead)` — probability-weighted forward projection

**Output** (`SKFState`):
- `regime_probabilities` — dict of regime name → probability
- `most_likely_regime`, `regime_confidence`
- `mu_shift_high/low`, `sigma_scale_high/low` — probability-weighted averages across regimes
- `active_families: dict[str, float]` — family activation weighted by regime probabilities
- `innovation_norm` — how surprised the filter is (tripwire for novel situations)
- `regime_switch_detected` — true if most_likely regime changed since last update

**Default regime models** (used when no config exists):
1. `mixed_uncertain` — wide noise, near-identity transition, self_transition=0.85
2. `postfrontal_persistence` — slight cooling tendency, small noise, self_transition=0.92, mu_shift_low=-1.0
3. `boundary_break` — large noise (volatile), self_transition=0.7, sigma_scale_low=1.3
4. `heat_overrun` — slight warming tendency, self_transition=0.88, mu_shift_high=+0.5

**Verified:** Smoke test with synthetic obs correctly identifies postfrontal_persistence with 0.999 confidence on appropriate inputs.

---

## Step 7: SKF Training — `src/analyzer/train_skf.py` [DONE]

**Class:** `SKFTrainer(db_path, station, utc_offset)`

**Process:**
1. Query `regime_labels` for all labeled days
2. Extract unique regimes from `phase_summary` entries + `regimes_active`
3. Group days by dominant regime (most hours)
4. For regimes with >= 2 labeled days:
   - Query hourly obs for those days → compute 5D state vectors
   - Fit A, b via ordinary least squares: x_{t+1} ≈ A @ x_t + b
   - Compute Q from residuals (process noise covariance)
   - Learn self_transition_prob from phase durations
   - Learn mu_shift and sigma_scale from settlement errors conditioned on regime
   - Learn active_families from signal_families_active field
5. Regimes with < 2 days: use `mixed_uncertain` parameters with widened noise
6. Save to `analysis_data/skf_config.json`

**Entrypoint:** `miami-train-skf` in pyproject.toml. Idempotent — reads all labels, refits from scratch each run.

---

## Step 8: Integration [DONE]

**`src/engine/regime_classifier.py`** — `RegimeState` extended with:
```python
skf_probabilities: dict[str, float] = field(default_factory=dict)
skf_mu_shift_high: float = 0.0
skf_mu_shift_low: float = 0.0
skf_sigma_scale_high: float = 1.0
skf_sigma_scale_low: float = 1.0
skf_active_families: dict[str, float] = field(default_factory=dict)
```

**`src/engine/residual_estimator.py`** — SKF adjustment path added after existing family-based adjustments:
```python
skf_trust = min(1.0, n_training_days / 10)  # ramps 0→1 as data grows
delta_mu += regime.skf_mu_shift * skf_trust
sigma_mult *= (1.0 + (regime.skf_sigma_scale - 1.0) * skf_trust)
```

The `_n_training_days` key in `skf_probabilities` tracks training volume. With 4 human reviews, `skf_trust` = 0.4 — conservative. At 10+ labeled days, full trust.

**`pyproject.toml`** — added `numpy>=2.0`, `pyyaml>=6.0`, `miami-train-skf` entrypoint.

---

## Files Created

| File | Layer | Purpose |
|------|-------|---------|
| `src/analyzer/daily_data_builder.py` | analyzer | Gather full day's data, OHLC aggregation |
| `src/analyzer/prompt_builder.py` | analyzer | Assemble LLM prompt from template + data + calibration context |
| `src/analyzer/prompts/daily_review.md` | analyzer | LLM instruction template |
| `src/analyzer/ai_review_parser.py` | analyzer | Parse AI output, store to DB + files |
| `src/analyzer/train_skf.py` | analyzer | Train SKF from labeled data |
| `src/analyzer/hdp_regime_discovery.py` | analyzer | HDP-Sticky shadow mode (nightly batch, regime discovery) |
| `src/analyzer/post_settlement_calibrator.py` | analyzer | Auto-tune CUSUM + SKF parameters via EMA after settlement |
| `src/engine/kalman_regimes.py` | engine | Switching Kalman Filter (5D, per-regime, forward prediction) |
| `src/engine/changepoint_detector.py` | engine | CUSUM + SPECI-style threshold change detection |

## Files Modified

| File | Change |
|------|--------|
| `src/collector/store/schema.py` | Added regime_labels + regime_labels_hdp_test tables, bumped to v12 |
| `src/collector/store/db.py` | Added migration + regime_labels methods |
| `src/engine/regime_classifier.py` | Extended RegimeState with 6 SKF fields |
| `src/engine/residual_estimator.py` | Added SKF adjustment pathway (trust-gated) |
| `pyproject.toml` | Added numpy + pyyaml deps, 4 new entrypoints |

---

## Verification Results

| Test | Status |
|------|--------|
| `pytest` — 35 existing tests | PASS |
| All modules import cleanly | PASS |
| `daily_data_builder.py` March 16 — OHLC with frontal RAPID_CHANGE | PASS (55K tokens) |
| `daily_data_builder.py` March 17 — complete data all sources | PASS (74K tokens) |
| `prompt_builder.py` end-to-end prompt assembly | PASS |
| `ai_review_parser.py` backfill from 4 human reviews | PASS (4 rows in regime_labels) |
| SKF smoke test with synthetic obs | PASS (postfrontal_persistence 0.999 confidence) |
| CUSUM + SKF on March 16 frontal passage | PASS (fires at 22:00 LST, escalates to P=1.0 by 23:00) |

---

## Daily Operational Schedule

All times in LST (UTC-5 fixed, no DST).

```
~04:45  CLI settlement arrives (collector ingests automatically)
        The climate day is now complete. All data is in the DB.

 09:00  miami-calibrate --db miami_collector.db --date <yesterday>
        Statistical self-tuning:
        - CUSUM: adjust h per channel (ARL feedback vs AI labels)
        - SKF: adjust mu_shift/sigma_scale per regime (EMA vs settlement error)
        Writes: analysis_data/cusum_config.json, analysis_data/skf_config.json
        Appends: analysis_data/calibration_log.jsonl

 09:00  miami-hdp-shadow --db miami_collector.db --date <yesterday>
        HDP-Sticky shadow discovery (runs in parallel with calibrate):
        - Discovers regime count from obs data alone
        - Writes to regime_labels_hdp_test table
        - Never touches live pipeline

 09:30  Build AI review prompt:
        - daily_data_builder assembles full day's data (OHLC + all sources)
        - prompt_builder inserts regime/family definitions + calibration context
        - save_prompt writes to disk for OpenClaw

 09:30  OpenClaw feeds prompt to Claude → captures raw output

 09:30  ai_review_parser stores:
        - Structured YAML → regime_labels table
        - Narrative → reviews/ai/{date}.md
        - Threshold recommendations → analysis_data/threshold_history.jsonl

 09:30  miami-train-skf --db miami_collector.db
        Retrain SKF from all accumulated regime labels
        Writes: analysis_data/skf_config.json (overrides calibrator's incremental updates
        with full batch refit — this is intentional, the batch fit is more principled)
```

**Live intraday (every ~5 min):**
```
ChangeDetector.update(obs) → fires on SPECI thresholds or CUSUM
    ↓ if fired
SKF.notify_changepoint(probability, channels) → reweights regime priors
    ↓
SKF.update(obs) → regime probabilities + mu/sigma adjustments
    ↓
regime_classifier + signal_families + residual_estimator → bracket pricing
```

---

## Self-Learning Feedback Loops

### Statistical (automatic, daily)

```
Settlement error
    ↓
post_settlement_calibrator.py
    ├── CUSUM h: EMA toward fewer false alarms / fewer misses (α=0.15, ±0.3/day)
    ├── SKF mu_shift: EMA toward observed settlement error per regime (α=0.15, ±0.3°F/day)
    └── SKF sigma_scale: widen if error outside CI, tighten if inside (±0.05/day)
```

Clamped to prevent runaway tuning. Conservative enough that a single outlier day doesn't wreck the parameters.

### AI review (daily, sanity check)

```
AI sees:
    ├── Current auto-tuned CUSUM thresholds (h and k per channel)
    ├── Current auto-tuned SKF parameters (mu_shift, sigma_scale per regime)
    ├── Recent calibration history (last 3 days of updates)
    └── Full day's data
AI produces:
    ├── Regime labels (phases, transitions, regime types)
    ├── Signal family evaluation
    ├── Model performance assessment
    └── Threshold recommendations (can override auto-tuned values)
```

AI recommendations take precedence over statistical auto-tuning when they conflict. The statistical loop handles the steady-state drift. The AI handles the "this doesn't make meteorological sense" override.

### SKF batch retraining (daily, after AI review)

```
All regime_labels (AI + human)
    ↓
train_skf.py (OLS fit on grouped regime data)
    ↓
skf_config.json (replaces incremental EMA updates with batch-fitted parameters)
```

The batch refit is more principled than the incremental EMA updates because it considers all labeled days simultaneously. The EMA updates from the calibrator are effectively overridden each day by the batch refit — the calibrator's value is in the *intermediate* period before the next batch run, and in providing the AI with visibility into the statistical drift direction.

### HDP shadow (daily, silent comparison)

```
Previous day's obs
    ↓
HDP-Sticky Gibbs sampler (discovers regime count automatically)
    ↓
regime_labels_hdp_test table (never touches live pipeline)
```

After 30+ days, compare HDP discoveries vs AI labels. If HDP consistently finds the same structure, it can replace the AI labeling for regime discovery — enabling a fully local, API-free system.

---

## Next Steps

### Immediate: Wire Up the Daily Cron Jobs

1. **Set up two cron entries:**
   - 09:00 LST: `miami-calibrate` + `miami-hdp-shadow` (can run in parallel)
   - 09:30 LST: prompt build → OpenClaw → parse → `miami-train-skf` (sequential chain)

2. **Wire change detector + SKF into the live engine loop:**
   ```python
   detector = ChangeDetector()
   detector.fit_diurnal(db_path, station)
   detector.reset()

   skf = SwitchingKalmanFilter("analysis_data/skf_config.json")
   skf.reset(target_date)

   # Each cycle:
   cd_state = detector.update(obs_dict, hour_lst, minutes)
   if cd_state.fired:
       skf.notify_changepoint(cd_state.changepoint_probability, cd_state.channels_fired)
   skf_state = skf.update(obs_vector, hours_elapsed)
   ```

### Short-Term: Accumulate Training Data (Days 1-14)

3. **Let the system run for 10-14 days.** During this period:
   - SKF runs with default parameters (conservative)
   - `skf_trust` ramps from 0.4 → 1.0 as reviews accumulate
   - CUSUM thresholds self-tune toward fewer false alarms
   - HDP shadow accumulates comparison data silently
   - Monitor `analysis_data/calibration_log.jsonl` for drift trends

### Medium-Term: Evaluate and Tune (Days 14-30)

4. **Compare HDP vs AI labels.** Do they discover the same regimes?
5. **Evaluate CUSUM tuning.** Are the auto-tuned thresholds producing the right false alarm rate (~1/day/channel)?
6. **Evaluate SKF performance.** Are regime-aware adjustments improving bracket pricing accuracy vs flat-bias baseline?
7. **Tune `skf_trust` ramp.** Faster if performing well, slower if noisy.

### Long-Term: Autonomy (Day 30+)

8. **If HDP matches AI labels consistently**, consider replacing daily AI call with HDP + local LLM for narrative review only.
9. **Evaluate DS3M** when 3+ months of labeled data exist — can learn nonlinear regime dynamics that SKF's linear assumption misses.
10. **Cross-city generalization.** Architecture is station-agnostic. Only data builder queries and CUSUM scale parameters are station-specific.
