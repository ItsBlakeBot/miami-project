# Handoff for Claude Code Opus — Miami Project

Use this as the working brief for the next implementation pass.

Repo path:
- `/Users/blakebot/blakebot/miami-project`

Environment notes:
- Run commands from repo root with `PYTHONPATH=src`
- Project venv is at `.venv/`
- This repo is **not git-tracked** right now, so inspect/edit files directly and summarize changed files explicitly
- Keep architecture honest; do not claim systems are live/cleaned unless they actually are

---

## First read these files

1. `plans/HANDOFF-2026-03-19-BMO.md`
2. `plans/repo-blueprint.md`
3. `plans/bot-architecture.mmd`
4. `plans/hybrid-workflow.md`
5. `plans/trading-bot-v1.md`
6. `cron/daily-post-settlement.md`
7. `src/analyzer/post_settlement_calibrator.py`
8. `src/analyzer/prompt_builder.py`
9. `src/analyzer/prompts/daily_review.md`
10. `src/analyzer/ai_review_parser.py`
11. `src/analyzer/train_skf.py`
12. `src/analyzer/hdp_regime_discovery.py`
13. `src/engine/orchestrator.py`
14. `src/collector/store/schema.py`
15. `src/trading/` (all current modules)

---

## Ground truth / architecture invariants

Do not break these:

- **Climate day = midnight–midnight LST (UTC-5 fixed year-round)**
- **KMIA CLI settlement is the only truth for market outcome**
- **HIGH and LOW are separate markets**
- Reviews are annotations / labels, not direct free-form live features
- The repo already has a modular path forming around:
  - `signal_engine -> baseline_engine -> short_range_updater / station_transfer -> regime_classifier -> signal_families -> residual_estimator -> trading/*`
- Avoid blind deletion of older modules until imports/runtime paths are verified

---

## Strategic direction from Blake

### 1. Statistical correction comes first; AI audits it
We do **not** want a flow where:
- AI makes/validates an adjustment
- then a later crude statistical step clobbers it

Desired behavior:
1. deterministic stats step generates a **proposal**
2. AI audits that proposal
3. deterministic promotion step decides:
   - approve
   - approve_with_dampening
   - hold
   - reject

The AI should explicitly comment on whether the statistical correction was:
- correct
- overreactionary
- underwhelming
- mixed

### 2. HDP is a long-horizon shadow learner
HDP is **not expected to be great now**.
It is expected to improve over **weeks to months**.

Interpretation:
- It is okay if it starts too sensitive or too weak
- It should run daily and accumulate comparison data
- Over time it may replace or materially alter the current SKF/CUSUM framing
- After ~6 months, Blake expects a deeper model direction too (described informally as D3SM / deep learning progression)

### 3. Paper trader should always be hunting
The paper trading bot is not a once-a-day evaluator.
It should continuously:
- inspect latest model output / bracket estimates
- inspect current observations / regime breaks vs model baseline
- inspect current Kalshi market conditions via active WS snapshots
- look for entries with real displayed liquidity
- manage held paper positions dynamically
- exit early when new model runs / observations / price changes break the thesis

Philosophy:
- profit does **not** require winning every trade
- profit does **not** require holding everything to settlement
- entries, EV, and exits matter

### 4. Retention direction
- The SQL DB can keep growing for now
- Long-term plan is to remove days older than **6 months**
- Do **not** over-optimize pruning now unless needed for the work you are doing

---

## Current repo reality

Already added in prior pass:
- `src/engine/baseline_engine.py`
- `src/analysis/backtest_residual_bot.py`
- `plans/repo-blueprint.md`
- `plans/bot-architecture.mmd`
- tests for baseline engine + backtest scaffold

Also already adjusted:
- `collector.signals` now wraps canonical `signals.signals`
- some obvious orphan modules marked legacy/reference-only
- `miami-collector` now points to `collector.entrypoints:cli_entry`
- backtest scaffold already runs

Do not re-discover this from scratch.

---

## Important current problem areas

There are partial / WIP edits already started around:
- `src/analyzer/post_settlement_calibrator.py`
- `src/analyzer/prompt_builder.py`
- `src/analyzer/prompts/daily_review.md`
- `src/analyzer/ai_review_parser.py`

Do **not** assume they are finished or coherent.
Inspect and complete them carefully.

Another major architectural problem:
- `train_skf.py` currently writes `analysis_data/skf_config.json`
- that mixes structural SKF learning with correction ownership
- this risks late training clobbering the promoted live correction config

That needs to be made safer in this pass.

---

# Implementation tasks

## Task A — Daily post-settlement driver

Create a deterministic driver module/CLI, ideally:
- `src/analyzer/daily_post_settlement.py`

Preferred command shape:
- `miami-daily-post-settlement prepare --date YYYY-MM-DD`
- `miami-daily-post-settlement finalize --date YYYY-MM-DD --review reviews/ai/YYYY-MM-DD.md`
- `miami-daily-post-settlement run --date YYYY-MM-DD --review reviews/ai/YYYY-MM-DD.md`

Use existing modules where possible:
- `analyzer.post_settlement_calibrator`
- `analyzer.daily_data_builder`
- `analyzer.prompt_builder`
- `analyzer.ai_review_parser`
- `analyzer.hdp_regime_discovery`
- `analyzer.train_skf`
- `engine.orchestrator`

### Desired flow

#### Prepare phase
1. determine target date (if omitted)
2. run deterministic statistical calibration to generate **proposed** config artifacts
3. build prompt with:
   - daily data package
   - recent context
   - calibration proposal context
4. save prompt to `reviews/prompts/review_prompt_<date>.md`

#### Finalize phase
1. parse/store AI review
2. apply/promotion step from parsed statistical audit
3. run HDP shadow **after** parse/store (important)
4. run SKF training into a **candidate/trained artifact**, not directly into the live correction-owned config
5. run verification / inference smoke test
6. write a concise summary

#### Run phase
Can combine prepare + finalize if a review file is supplied.

Update `pyproject.toml` if you add a CLI.
Update `cron/daily-post-settlement.md` to reflect the actual implemented flow.

---

## Task B — Safe stats proposal -> AI audit -> promotion flow

### Goal
Statistical post-settlement corrections should produce **proposed** configs first.
AI should audit them.
A deterministic promotion step should then decide what becomes live.

### Requirements

#### 1. Separate live vs proposed artifacts
At minimum create/maintain artifacts like:
- `analysis_data/cusum_config.json` (live)
- `analysis_data/cusum_config.proposed.json` (proposal)
- `analysis_data/skf_config.json` (live correction-aware runtime artifact)
- `analysis_data/skf_config.proposed.json` (proposal)
- `analysis_data/skf_trained_candidate.json` (candidate from training)

If you use slightly different naming, keep it clear and coherent.

#### 2. AI prompt must clearly act as audit
The prompt should tell the AI:
- stats already ran
- the shown changes are proposals, not truth
- audit whether they are right-sized
- return a `statistical_audit` block

Desired block shape:
```yaml
statistical_audit:
  overall_verdict: approve | approve_with_dampening | hold | reject
  dampening_factor: 0.0-1.0
  cusum_verdict: correct | overreactionary | underwhelming | mixed | hold
  skf_verdict: correct | overreactionary | underwhelming | mixed | hold
  reasoning: "..."
  notes:
    - "..."
```

Backward compatibility matters:
- old reviews without this block should still parse

#### 3. Parser/storage
- extend parser to read the statistical audit block if present
- keep old 6-block reviews working
- store audit history in:
  - `analysis_data/statistical_audit_history.jsonl`

#### 4. Promotion/apply step
Implement a deterministic promotion step driven by the parsed audit.

Support:
- `approve` -> apply full proposal
- `approve_with_dampening` -> blend live toward proposal using factor
- `hold` -> keep live config unchanged
- `reject` -> keep live config unchanged and record rejection

Be explicit in logs/history about what was promoted and why.

#### 5. Do not let late training clobber live correction ownership
This is critical.
If full clean separation is too invasive for one pass, at minimum:
- make `train_skf.py` write a **candidate/trained artifact** separate from the live runtime config
- keep live config ownership coherent in the finalize/promote flow
- do not leave a setup where training runs last and overwrites the promoted live config blindly

If you can cleanly separate **structural SKF parameters** from **error-correction parameters**, do that.
If not, do the safest intermediate version and document the remaining caveat honestly.

---

## Task C — HDP shadow automation

### Current status
- Code exists
- It was not actually running nightly until manually tested
- Manual tests showed it works, but is still noisy/twitchy on some days

Manual observed outputs already seen:
- `2026-03-16` -> 6 regimes (fragmented / twitchy)
- `2026-03-17` -> 4 regimes (more plausible)
- `2026-03-18` -> 5 regimes (plausible-ish but still noisy)

### Desired behavior
- run automatically every day in the post-settlement flow
- run **after AI review parse/store**
- stay shadow-only
- write to `regime_labels_hdp_test`
- do not influence the primary live label path yet

Optional nice-to-have if practical:
- add light smoothing / minimum dwell / micro-phase merge helper
- but do **not** turn this into a giant HDP research branch in this pass

Important framing:
- HDP is supposed to improve over months
- it is okay if it is imperfect now
- the point is to start accumulating daily comparison data

---

## Task D — Paper trading infra + bot

### Goal
Create the minimum viable continuous paper-trading system.
This is for trading-policy learning, not core weather truth.

### Existing useful data already in DB
- `market_snapshots`
- `active_brackets`
- `bracket_estimates`
- `market_settlements`

### What is missing
We do not yet have actual paper order / paper position / paper pnl lifecycle tables.

### Add schema support
At minimum add:
- `paper_trades`
- `paper_trade_marks` (preferred)
- `paper_trade_settlements`

Add useful indexes too.

### Implement a paper trading module/CLI
Suggested:
- `src/trading/paper_trader.py`
- plus script entry in `pyproject.toml`

Support at least:
- single cycle mode
- watch mode with interval seconds

### Entry logic requirements
The bot should always be hunting using:
- latest bracket estimates
- current market snapshots
- current active brackets
- latest model interpretation already flowing into estimates

Only log a paper entry if:
- displayed liquidity exists on the side/price being used
- do **not** pretend we filled at a price with no resting liquidity

Keep the entry policy pragmatic:
- use edge / EV thresholds
- configurable constants are fine
- log enough info for future diagnostics

Suggested logged fields include:
- timestamp
- target_date
- ticker
- market_type
- side
- entry_price
- displayed size/liquidity
- model probability
- implied probability
- edge
- EV
- reason tags / active signals
- regime/path metadata if available
- strategy version / config snapshot if practical
- status

### Exit logic requirements
The bot should evaluate held positions for deterioration due to:
- new model runs / updated bracket estimates
- observation-driven regime breaks vs prior thesis
- live market movement

It should be willing to exit early.
Do **not** assume all trades should be held to settlement.

This is core guidance from Blake:
- you do not need to win every trade
- you do not need to hold every trade to settlement
- entries, EV, and exits matter

### Settlement logic
- resolve open paper trades using `market_settlements` when available
- write realized outcome / pnl to settlement tables

### Keep it pragmatic
It does not need to be the final perfect trading system.
It does need to be structurally useful and data-generating.

---

## Task E — Tests / verification

Add tests where practical, especially for:
- statistical audit parsing
- promotion/apply behavior
- any pure policy logic in the paper trader

Run relevant tests and the full suite if reasonable.
Include exact commands and results in your summary.

---

## Deliverables expected in your final summary

Please summarize:
1. files added/changed
2. commands run
3. test results
4. any remaining caveats
5. whether docs now match the implemented flow

Also mention explicitly if you had to choose an intermediate safety step instead of a full architectural separation.

---

## Suggested immediate execution order

1. inspect and stabilize the WIP stats/prompt/parser files
2. implement promotion/apply flow
3. implement `daily_post_settlement.py`
4. wire HDP shadow into finalize/run
5. make `train_skf.py` write a candidate artifact instead of clobbering live config
6. add paper trading schema + bot
7. update docs
8. run tests

---

## One-line north star

Build a **safe daily post-settlement pipeline** and a **useful continuous paper trader** without pretending the experimental parts are already mature.
