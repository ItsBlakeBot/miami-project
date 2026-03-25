# Residual Signal Bot v1 — Design for Weather Trading Without Double Counting

## Purpose

Build a Python weather trading bot that:

1. uses **fresh model guidance as the baseline belief**
2. uses **live observations/signals only as residual corrections**
3. avoids **double counting** signals that the models already priced in
4. remains **auditable**, **deterministic**, and easy to backtest
5. uses **LST (UTC-5 fixed)** for all settlement windows, timestamps, and climate-day logic

This document is the implementation plan for the next version of the Miami bot logic.

Implementation blueprint companions:
- `plans/repo-blueprint.md` — canonical keep/merge/retire/build map for the repo
- `plans/bot-architecture.mmd` — Mermaid architecture view of the intended live path

Important scope rule:
- this file defines the **target architecture and weather logic**
- `repo-blueprint.md` defines the **repo cleanup/refactor plan**
- do not claim the repo is fully cleaned until the blueprint work is actually complete

---

## Core Philosophy

The bot should **not** ask:

> “What is the final high/low from signals alone?”

It should ask:

> “Given what the freshest usable models already believe, what are live observations telling us that the models are missing?”

That means the system is built as:

- **models = baseline**
- **signals = residual correction**
- **regime = context / gating**
- **market = execution layer**

This is the cleanest way to avoid double counting.

---

## Settlement Clock Rule

Everything in this bot must use **LST**, not DST wall-clock time.

### Canonical climate-day definition
- **Climate day:** `00:00–23:59 LST`
- **UTC window for KMIA:** `05:00Z` to `04:59:59Z` next day
- All:
  - review timestamps
  - training snapshots
  - regime labels
  - model evaluation checkpoints
  - running high/low logic
  - market timing logic
  must be computed on this basis.

This is required because Kalshi settles against NWS climate data using **local standard time**.

---

## Problem Statement: Where Double Counting Happens

There are two different double-counting failures we need to prevent.

### 1) Double counting **against the models**
Example:
- fresh models already capture post-frontal NNW flow
- then the bot applies a full extra NNW cooling shift on top
- result: the same meteorological fact gets counted twice

### 2) Double counting **among the signals themselves**
Example on a boundary day:
- dew crash
- wind rotation
- pressure rebound

These are not three independent facts. They are usually one underlying event:
- **boundary passage / regime break**

If the bot adds three separate large adjustments, conviction gets inflated artificially.

---

## Design Summary

The system should have three weather layers and one trading layer:

```text
fresh model baseline
    ↓
regime classifier
    ↓
signal-family residual adjuster
    ↓
final high/low distributions
    ↓
market comparison + execution
```

### Final formulas

At a high level:

```text
final_mu = model_mu + captured_gate * captured_residual + blind_residual
final_sigma = model_sigma * volatility_multiplier + sigma_additive
```

Where:

- `model_mu` / `model_sigma` come from fresh models only
- `captured_residual` comes from signal families that models often already see
- `blind_residual` comes from local/microclimate signals models systematically miss
- `captured_gate = 1 - model_trust`
- `model_trust` is high when models are fresh and tracking live obs well

So:
- if models are fresh and right → signals get less power
- if models are stale or missing live conditions → signals get more power

---

## Separate the Problems: HIGH and LOW

Do not build one generic temperature estimator.

Build separate pipelines for:

- **HIGH residual estimation**
- **LOW residual estimation**

Reason:
- highs are driven by heating, cloud, mixing, timing of peak
- lows are driven by dew point floor, advection, boundary timing, radiative cooling, local microclimate

The bot should store and train these separately.

---

## Architecture

## Module Layout

Recommended new/updated modules:

```text
src/
  engine/
    baseline_engine.py         # fresh-model baseline belief
    regime_classifier.py       # regime/path labels from live state
    signal_families.py         # collapse raw signals into anti-double-counted families
    residual_estimator.py      # convert family scores into delta-mu / delta-sigma
    trade_policy.py            # compare belief vs market and decide actions

  collector/
    model_scorer.py            # keep, but strengthen freshness filtering
    runner.py                  # unchanged data collection backbone

  signals/
    signal_evaluator.py        # keep, but extend to score families + residual edge

  analysis/
    build_training_snapshots.py
    backtest_residual_bot.py
```

Existing modules that still matter:
- `collector/model_scorer.py`
- `engine/signal_engine.py`
- `engine/estimator.py`
- `signals/signal_evaluator.py`

But v1 should shift from a pure “consensus + additive signals” estimator to a **baseline + residual family** estimator.

---

## Layer 1 — Model Baseline Engine

## Goal
Produce the best **fresh-model baseline belief** for final high and final low.

Outputs:
- `mu_high_model`
- `sigma_high_model`
- `mu_low_model`
- `sigma_low_model`
- `model_trust_high`
- `model_trust_low`

## Rules

### 1. Use freshest usable model run only
For each canonical model:
- take the latest valid run only
- ignore older runs once a newer run exists
- do not let fast-refresh models dominate by frequency alone

### 2. Canonicalize and dedupe sources
If the same model appears via multiple providers:
- pick one canonical representation
- do not count both

### 3. Treat ensembles carefully
Do **not** let large ensembles swamp the baseline by raw member count.

Preferred v1 behavior:
- deterministic models: 1 vote each
- ensemble families: use only for spread / uncertainty, or as one family-level contribution
- if using members directly, collapse to summary stats first

### 4. Weight by skill and freshness
Use existing scoring history plus run age.

Suggested baseline weight:

```python
skill_weight = 1 / max(mae, mae_floor) ** weight_power
freshness_weight = math.exp(-run_age_hours / freshness_scale)
tracking_weight = math.exp(-(tracking_error ** 2) / tracking_scale)
final_weight = skill_weight * freshness_weight * tracking_weight
```

Good defaults:
- `mae_floor = 0.75`
- `freshness_scale = 6 hours`
- `tracking_scale = 4.0`

### 5. Forward-curve tracking matters
For each model, compare what it implied for already-observed hours vs actual obs.

This yields:
- `recent_error`
- `mean_error`
- `tracking_error`
- `residual scatter`

Use that to compute `model_trust`.

## Model trust

Suggested v1:

```python
freshness = exp(-run_age_hours / 6)
track_quality = exp(-(best_tracking_error ** 2) / 4)
spread_penalty = exp(-max(ensemble_sigma - sigma_ref, 0) / sigma_scale)
model_trust = clip(freshness * track_quality * spread_penalty, 0, 1)
```

Interpretation:
- **high trust**: fresh models, tracking obs, limited spread
- **low trust**: stale models, missing obs path, noisy spread

---

## Layer 2 — Raw Signal Engine

The existing `signal_engine.py` is still useful as the raw feature producer.

It should continue to output live values like:
- current obs
- dew point
- wind direction/speed
- pressure and trend
- FAWN state
- nearby station divergence
- running high/low
- obs vs consensus
- trajectory
- forward model max/min

But raw signals should **not** directly drive trades.

They should feed the next layer.

---

## Layer 3 — Regime Classifier

## Goal
Turn noisy weather state into a small number of regime/path labels.

This gives context for how strongly the bot should trust each family of signals.

### Proposed regime labels

Primary regime labels:
- `heat_overrun`
- `boundary_break`
- `postfrontal_persistence`
- `outflow_volatility`
- `radiative_floor`
- `sea_breeze_warm_hold`
- `frontal_transition`
- `mixed_uncertain`

Path labels:
- `cliff_down`
- `cold_grind`
- `high_locked_early`
- `late_heat_run`
- `flat_then_break`
- `normal_diurnal`

## First two anchor cases

### March 16, 2026
- high archetype: `heat_overrun`
- low archetype: `boundary_break`
- path labels: `late_heat_run` (high side), `flat_then_break` / `cliff_down` (low side)

### March 17, 2026
- high archetype: `high_locked_early`
- low archetype: `postfrontal_persistence`
- path label: `cold_grind`

These should be used as early unit-test examples.

---

## Layer 4 — Signal Families (Anti-Double-Counting Layer)

This is the most important new layer.

Instead of adding every raw signal independently, collapse them into a few **families**.

One mechanism → one family score.

## Proposed families

### 1. Heat Overrun Family
Used mostly for highs.

Raw ingredients:
- obs running hotter than baseline
- regional nearby warmth confirmation
- repeated high plateau / no immediate collapse
- lack of suppressive cloud or convective limiter

Purpose:
- detect days where the model baseline is too cool on the high side

### 2. Boundary Break Family
Used mostly for lows.

Raw ingredients:
- dew crash
- wind rotation into new air mass
- pressure trough then recovery
- rapid late temperature fall

Purpose:
- detect late path breaks that the market and/or stale models may still underprice

### 3. Post-Frontal Persistence Family
Used mostly for lows.

Raw ingredients:
- sustained NNW/N flow
- rising pressure
- falling dew point
- zero/near-zero CAPE
- no daytime recovery

Purpose:
- detect cold grind / persistence days where the edge is gradual, not dramatic

### 4. Microclimate Confirmation Family
Used mainly for lows, sometimes highs.

Raw ingredients:
- FAWN lead
- nearby inland station crash
- station divergence cluster
- local dew-floor evidence

Purpose:
- capture local timing / floor information that broad models often miss

### 5. Extreme Lock Family
Used as constraints, not ordinary shifts.

Raw ingredients:
- high already locked early
- running low already below warm-tail brackets
- obs plateau strong enough to cap upside
- tail outcomes now physically implausible

Purpose:
- clamp or remove parts of the distribution, rather than applying soft additive shifts

---

## Family Scoring

Each family should output a normalized score:

```python
family_score in [0.0, 1.0]
```

Not every input needs equal weight.

### Example: boundary break score

```python
boundary_break_score = clip(
    0.40 * dew_crash_score +
    0.30 * wind_rotation_score +
    0.30 * pressure_rebound_score,
    0.0,
    1.0,
)
```

### Example: post-frontal persistence score

```python
postfrontal_persistence_score = clip(
    0.30 * continental_flow_score +
    0.25 * pressure_rise_score +
    0.20 * dew_drop_score +
    0.15 * no_daytime_recovery_score +
    0.10 * zero_cape_score,
    0.0,
    1.0,
)
```

The exact weights should initially be hand-tuned from cases and then calibrated later.

---

## Model-Captured vs Model-Blind Families

This split is how we prevent double counting against the baseline.

## Model-captured families
These are often already visible in fresh NWP:
- heat overrun (partially)
- post-frontal persistence (often yes)
- broad obs divergence
- wind regime
- pressure trend
- no daytime recovery

These should be **gated by model trust**.

## Model-blind families
These are often local / underresolved / timing-sensitive:
- microclimate confirmation
- FAWN lead
- nearby crash cluster
- station-specific dew-floor behavior
- very local boundary timing

These should be allowed to affect the estimate more directly.

---

## Residual Estimator

## Goal
Convert:
- model baseline
- regime labels
- family scores
- model trust

into:
- `delta_mu_high`
- `delta_sigma_high`
- `delta_mu_low`
- `delta_sigma_low`

## High-level formulas

### Mean adjustment

```python
captured_gate = 1.0 - model_trust

captured_residual = (
    w_heat * heat_overrun_score +
    w_break * boundary_break_score_for_high_context +
    ...
)

blind_residual = (
    w_micro * microclimate_confirmation_score +
    w_lock  * lock_adjustment
)

final_mu = model_mu + captured_gate * captured_residual + blind_residual
```

### Sigma adjustment

```python
final_sigma = (
    model_sigma * volatility_multiplier
    + sigma_additive
)
```

Where volatility multiplier might rise when:
- outflow risk is high
- boundary break signals are mixed/partial
- nearby station divergence is large
- ensemble spread is wide

And might compress when:
- high is locked early
- warm tail is clearly removed
- observed plateau strongly constrains one side

---

## Clamps vs Soft Shifts

Not every signal should produce a smooth adjustment.

### Soft shifts
Use these when confidence is partial:
- heat overrun score 0.4
- post-frontal persistence score 0.5
- microclimate score 0.3

### Hard clamps
Use these when the state becomes physically constrained:
- `high_locked_early`
- `warm_tail_removed`
- running low already below a bracket floor
- high already reached with no credible reheating path

Examples:

```python
if high_locked_early:
    final_mu_high = min(final_mu_high, running_high + 0.5)
    final_sigma_high = min(final_sigma_high, 0.8)

if warm_tail_removed:
    final_mu_low = min(final_mu_low, running_low + 1.0)
```

These are not ordinary penalties. They are distribution constraints.

---

## Suggested v1 Python Data Structures

## Baseline belief

```python
@dataclass
class BaselineBelief:
    market_type: str
    target_date: str
    mu: float
    sigma: float
    model_trust: float
    n_models: int
    freshest_run_utc: str | None
    model_weights: dict[str, float]
    tracking_errors: dict[str, float]
```

## Regime state

```python
@dataclass
class RegimeState:
    primary_regime: str
    path_class: str
    confidence: float
    tags: list[str]
```

## Signal family state

```python
@dataclass
class SignalFamilyState:
    heat_overrun: float = 0.0
    boundary_break: float = 0.0
    postfrontal_persistence: float = 0.0
    microclimate_confirmation: float = 0.0
    extreme_lock: float = 0.0
    notes: list[str] = field(default_factory=list)
```

## Residual output

```python
@dataclass
class ResidualAdjustment:
    delta_mu: float
    delta_sigma_mult: float
    delta_sigma_add: float
    clamps: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
```

---

## Training Data Design

Do not train directly on raw final temperature.

Train on **residuals relative to the baseline**.

## Snapshot generation
For each settled climate day, generate snapshots at fixed LST checkpoints, for example:
- 1 AM
- 4 AM
- 7 AM
- 10 AM
- 1 PM
- 4 PM
- 7 PM
- 10 PM

At each checkpoint store:
- target date
- eval time LST
- baseline high/low mu + sigma
- model trust
- regime label
- family scores
- running high/low
- live obs features
- market prices (optional)
- final CLI high/low

## Targets

```python
high_residual = cli_high - baseline_high_mu
low_residual  = cli_low  - baseline_low_mu
```

This is the thing we actually want to learn.

Not “what was the final temp?”
But “what did the live state add beyond the model baseline?”

---

## How March 16 and March 17 Should Be Used

These two cases are not enough to fit robust production weights.

But they are enough to define:
- mechanism families
- clamp logic
- regime labels
- early unit tests

## March 16, 2026

### High lesson
- model baseline too cool
- observed/regional heating outran consensus
- **heat_overrun** family should raise the high residual

### Low lesson
- the day stayed warm until late
- then a boundary stack formed
- dew crash + wind rotation + pressure rebound = one mechanism
- **boundary_break** family should shift the low colder late, not all day

## March 17, 2026

### High lesson
- high was basically locked in the first hour
- later daytime warmth risk should be clamped down
- **extreme_lock** / `high_locked_early` should compress high-side sigma

### Low lesson
- no need for a dramatic cliff
- persistent NNW flow + rising pressure + drying dew + no daytime recovery was enough
- **postfrontal_persistence** family should push the low colder steadily through the day

These should become explicit test fixtures.

---

## Initial v1 Rules Before Any ML

Before fitting any learned model, implement these hand-built rules.

### Rule 1 — Fresh models win by default
If models are fresh and tracking live obs well, do not let captured signals dominate.

### Rule 2 — One mechanism, one family
Never add dew crash + wind shift + pressure rebound independently if they represent one boundary event.

### Rule 3 — Local signals get special status
FAWN, nearby crashes, and local dew-floor evidence may affect the distribution even when the broad models look fine.

### Rule 4 — Running extremes create clamps
If the high is already functionally locked or the warm tail is dead, treat that as a hard constraint.

### Rule 5 — Highs and lows are separate
Never reuse one adjustment formula for both.

### Rule 6 — Persistence days are not cliff days
The bot must be able to move colder without waiting for an explosive trigger.

### Rule 7 — All clocks are LST
No exceptions.

---

## Suggested Implementation Order

## Phase A — Refactor baseline
1. tighten freshness filtering in `collector/model_scorer.py`
2. expose a clean `BaselineBelief`
3. ensure only freshest usable runs contribute
4. verify no duplicate model families are being counted twice

## Phase B — Add regime + families
1. create `engine/regime_classifier.py`
2. create `engine/signal_families.py`
3. map raw `SignalState` into:
   - regime labels
   - family scores
   - clamp conditions

## Phase C — Residual estimator
1. create `engine/residual_estimator.py`
2. replace direct additive signal stacking with:
   - captured vs blind residual logic
   - model-trust gating
   - clamp handling

## Phase D — Snapshot builder and backtest
1. create `analysis/build_training_snapshots.py`
2. create `analysis/backtest_residual_bot.py`
3. backtest March 16 and 17 first as sanity checks
4. then expand to all settled days

## Phase E — Execution integration
1. feed final bracket probabilities into `trade_policy.py`
2. compare with market mids / best bid-ask
3. add edge thresholds, order rules, and risk controls

---

## Recommended Backtest Questions

For each settled day and checkpoint, answer:

1. Did the baseline alone beat the current estimator?
2. Did family-based residuals improve calibration vs raw additive signals?
3. Did captured-gating reduce over-adjustment on fresh-model days?
4. Did microclimate signals help on local timing cases?
5. Did clamp logic reduce silly late-day tails?
6. Did the system correctly distinguish:
   - heat overrun days
   - boundary break days
   - post-frontal persistence days?

---

## Where LLMs Should and Should Not Be Used

## Good uses of LLMs
- label regimes from completed daily reviews
- summarize failure cases
- propose new signal families or regime distinctions
- help cluster similar historical days
- help write/post-process human-readable daily reviews

## Bad uses of LLMs
- deciding live trades directly
- hallucinating temperatures from prose alone
- replacing deterministic live feature extraction
- producing unauditable probability outputs in production

The production bot should stay deterministic/statistical.

---

## Concrete v1 Pseudocode

```python
def evaluate_cycle(now_utc):
    state_high = signal_engine.build_state(now_utc, market_type="high")
    state_low  = signal_engine.build_state(now_utc, market_type="low")

    baseline_high = baseline_engine.build_baseline(state_high)
    baseline_low  = baseline_engine.build_baseline(state_low)

    regime_high = regime_classifier.classify(state_high, baseline_high)
    regime_low  = regime_classifier.classify(state_low, baseline_low)

    fam_high = signal_families.compute(state_high, baseline_high, regime_high)
    fam_low  = signal_families.compute(state_low, baseline_low, regime_low)

    adj_high = residual_estimator.adjust_high(
        baseline=baseline_high,
        regime=regime_high,
        families=fam_high,
    )
    adj_low = residual_estimator.adjust_low(
        baseline=baseline_low,
        regime=regime_low,
        families=fam_low,
    )

    belief_high = apply_adjustment(baseline_high, adj_high)
    belief_low  = apply_adjustment(baseline_low, adj_low)

    high_brackets = bracket_engine.price("high", belief_high)
    low_brackets  = bracket_engine.price("low", belief_low)

    edges = market_compare(high_brackets, low_brackets)
    orders = trade_policy.select(edges, regime_high, regime_low)
    return orders
```

---

## Key v1 Principle to Remember

If a signal is already visible in the freshest, well-tracking models, it should have **reduced incremental power**.

If a signal is local, underresolved, or timing-sensitive in a way the models are likely missing, it can have **independent power**.

That is the whole game.

---

## Case-Derived Implementation Rules from the New Reviews

This section translates the rebuilt March 16 / March 17 daily reviews directly into implementable bot logic.

The point is not just to preserve narrative lessons — it is to encode the exact anti-double-counting behavior the bot should use in production.

---

## March 16 / March 17: What the Bot Should Learn

### March 16, 2026 — `heat_overrun` + `boundary_break`

This day contains **two separate tradeable mechanisms** that must not be mixed together:

#### High-side mechanism
- the day ran **hotter than cool consensus**
- obs were already hot early
- nearby stations confirmed regional heating
- the afternoon high plateau proved the overrun was real, not a one-tick anomaly

Bot lesson:
- this is a **high residual** case, not a low-side case
- the bot should shift the **high** upward through a `heat_overrun` family
- it should **not** reuse the same logic to push the low colder until the later boundary evidence appears

#### Low-side mechanism
- the day stayed warm well into the evening
- then the state changed quickly via:
  - dew easing lower
  - pressure trough then recovery
  - wind rotation away from the marine regime
  - late surface break
- warm-middle low brackets looked safe until they suddenly were not

Bot lesson:
- do **not** cool the low aggressively at noon just because the day ends cold
- the low-side adjustment should behave as a **state machine**:
  1. `late_low_risk_flag`
  2. `boundary_watch`
  3. `low_bias_down_small`
  4. `low_bias_down_medium`
  5. `low_bias_down_large`
  6. `warm_low_brackets_dead`
  7. `low_endpoint_locked`

This sequence should produce a **time-evolving low residual**, not one giant all-day penalty.

### March 17, 2026 — `high_locked_early` + `postfrontal_persistence`

This day teaches the opposite lesson.

#### High-side mechanism
- the high was effectively printed in the opening hour
- after the regime turned colder, there was no meaningful daytime recovery path

Bot lesson:
- this is not a “heat overrun” day
- the high side should move into a **constraint state**, not continue to trade like an open-ended heating problem
- `high_locked_early` should act like a **clamp/compression rule**, not like a small additive adjustment

#### Low-side mechanism
- no dramatic crash was needed
- the edge came from **persistent continental cooling**:
  - NNW/N flow
  - rising pressure
  - dew collapse into the 50s
  - zero CAPE
  - no daytime recovery
- the best trade was not the prettiest exact bracket; it was the colder directional tail (`T57`)

Bot lesson:
- this is a **cold persistence** case
- the low should drift colder via a `postfrontal_persistence` family all day
- the bot should not wait for a 3–5°F cliff signal to act

---

## Non-Additivity Rules (Critical)

These rules are mandatory. They are the heart of the anti-double-counting design.

### Rule A — One physical mechanism = one family adjustment
The following raw features are often the same event and must not all be added independently:

#### Boundary-break cluster
- dew crash
- wind rotation
- pressure rebound after trough
- rapid surface temp fall

These must be collapsed into **one** `boundary_break` family score.

#### Post-frontal persistence cluster
- NNW/N flow
- rising pressure
- falling dew point
- no daytime recovery
- zero CAPE / dead atmosphere

These must be collapsed into **one** `postfrontal_persistence` family score.

#### Heat-overrun cluster
- obs running hotter than baseline
- nearby stations hotter than baseline
- repeated high plateau
- lack of suppressive cooling signal

These must be collapsed into **one** `heat_overrun` family score.

### Rule B — Do not let one mechanism hit both high and low the same way
Example:
- March 16 high overrun is a **high residual** story
- March 16 boundary break is a **low residual** story

The bot must keep them separate.

A high-side heat family should not automatically imply a low-side warm hold.
A late low-side boundary break should not retroactively distort the high-side estimate.

### Rule C — Running extremes create constraints, not soft nudges
Examples:
- `high_locked_early`
- `warm_tail_removed`
- `low_endpoint_locked`

These are not ordinary signal-family shifts. They should create:
- sigma compression
- truncation / clamp behavior
- bracket invalidation logic

### Rule D — Market information is not weather information
The bot may use market structure for execution or secondary diagnostics, but bracket prices must **not** directly become weather features in the residual model.

Allowed use:
- identify where edge exists
- identify whether a known weather state is already priced

Not allowed use:
- using market repricing itself as if it were meteorological evidence

---

## Raw Signal → Family Map

This is the first-pass mapping implied by the new reviews.

| Raw signal / label | Family | High or Low | Model-captured or Model-blind | Notes |
|---|---|---|---|---|
| `high_bias_up_small` | `heat_overrun` | High | Mostly captured | Use more when models are stale or tracking poorly |
| `high_bias_up_medium` | `heat_overrun` | High | Mostly captured | Midday confirmation state |
| `high_bias_locked_in` | `extreme_lock` | High | Captured + clamp | Move from shift mode to lock mode |
| `late_low_risk_flag` | `boundary_break` | Low | Mostly captured | Early warning only; low amplitude |
| `boundary_watch` | `boundary_break` | Low | Mostly captured | Wait for confirmation before large shift |
| `low_bias_down_small` | `boundary_break` | Low | Mostly captured | Transitional state |
| `low_bias_down_medium` | `boundary_break` | Low | Mixed | Surface confirmation begins |
| `low_bias_down_large` | `boundary_break` | Low | Mixed | Full break underway |
| `warm_low_brackets_dead` | `extreme_lock` | Low | Clamp | Remove warm-tail probability |
| `low_endpoint_locked` | `extreme_lock` | Low | Clamp | Compress sigma late |
| `high_locked_early` | `extreme_lock` | High | Clamp | No more casual warm rebound |
| `postfrontal_continental_confirmed` | `postfrontal_persistence` | Low | Mostly captured | Gate by model trust |
| `no_daytime_recovery` | `postfrontal_persistence` | Low/High | Mostly captured | High cap + low drift colder |
| `under57_live` | execution/diagnostic | Low | Not a weather family | Trade expression, not weather evidence |
| FAWN lead/crash | `microclimate_confirmation` | Mostly Low | Blind | Can override some model trust |
| nearby crash cluster | `microclimate_confirmation` | Mostly Low | Blind | More about sigma + timing |
| dew-floor evidence | `microclimate_confirmation` | Low | Blind | Especially useful near settlement |

Key implementation rule:
- labels like `under57_live` are **downstream diagnostics / trade interpretations**, not core meteorological drivers.
- the bot should learn from the weather state that produced `under57_live`, not from the trade label itself.

---

## Family Priority and Precedence

When multiple families are active, do not simply sum them.

Use this order of operations:

1. **Extreme lock / clamps**
2. **Microclimate confirmation**
3. **Boundary break** or **postfrontal persistence**
4. **Heat overrun**
5. **Minor residual nudges**

### Why this order?

#### 1. Extreme lock comes first
If the high is already functionally done, or the warm tail is physically dead, that should constrain the entire distribution before smaller residuals are considered.

#### 2. Microclimate confirmation comes before broad residual nudges
FAWN / nearby-station timing can reveal local reality that the big models underresolve.

#### 3. Boundary break and post-frontal persistence are mutually dominant low-side mechanisms
They should usually not both fire strongly at once.

Interpretation:
- `boundary_break` = abrupt regime shift
- `postfrontal_persistence` = steady cold regime continuation

If both scores are high, prefer the one matching the observed path:
- rapid cliff → `boundary_break`
- slow grind → `postfrontal_persistence`

#### 4. Heat overrun mainly belongs to the high side
Do not let a strong high-side heat family spill over into fake confidence on the low.

---

## Proposed State Machines

## A. March-16-style low-side state machine (`boundary_break`)

```text
neutral_warm_evening
  → late_low_risk_flag
  → boundary_watch
  → boundary_confirmation
  → warm_tail_removed
  → low_endpoint_locked
```

### Suggested transitions

#### `neutral_warm_evening` → `late_low_risk_flag`
Triggered when:
- afternoon/evening dew is easing lower vs prior warm regime
- but temp path still broadly warm

Effect:
- small negative low residual
- slight sigma increase
- no bracket-killing yet

#### `late_low_risk_flag` → `boundary_watch`
Triggered when:
- pressure trough is identified and rebound begins
- winds rotate meaningfully away from marine/southerly regime

Effect:
- medium caution
- slightly colder low distribution
- keep tail open, do not clamp yet

#### `boundary_watch` → `boundary_confirmation`
Triggered when:
- gusty W/WNW/NW turn or equivalent surface signature appears
- rapid temp drop and/or dew crash begins

Effect:
- larger downward low shift
- increase confidence that middle warm brackets are wrong

#### `boundary_confirmation` → `warm_tail_removed`
Triggered when:
- running low is already below warm bracket zone
- or surface path makes recovery implausible

Effect:
- hard clamp to remove warm low outcomes
- sigma may compress on warm side while remaining broad enough on the cold side

#### `warm_tail_removed` → `low_endpoint_locked`
Triggered when:
- obs low nearly matches final settle zone near midnight LST
- additional downside exists but major warm outcomes are gone

Effect:
- compress sigma further
- convert from discovery mode to endgame mode

## B. March-17-style low-side state machine (`postfrontal_persistence`)

```text
postfrontal_confirmed
  → no_daytime_recovery
  → warm_tail_removed
  → cold_grind_endgame
  → low_endpoint_locked
```

### Suggested transitions

#### `postfrontal_confirmed`
Triggered when:
- sustained NNW/N flow
- pressure rising
- dew falling
- no obvious warm-sector recovery mechanism

Effect:
- moderate colder low residual
- no need for large sigma spike

#### `postfrontal_confirmed` → `no_daytime_recovery`
Triggered when:
- midday temps remain suppressed despite sun
- the day cannot rebound into a warmer distribution

Effect:
- stronger colder shift
- high-side rebound paths also get capped

#### `no_daytime_recovery` → `warm_tail_removed`
Triggered when:
- running low is already below warm low brackets
- those brackets are no longer physically plausible

Effect:
- remove warm tail
- keep colder directional tail alive

#### `warm_tail_removed` → `cold_grind_endgame`
Triggered when:
- evening temps continue to fall steadily without a crash

Effect:
- continue walking low colder
- no need for volatility premium if the path is smooth and stable

#### `cold_grind_endgame` → `low_endpoint_locked`
Triggered when:
- upper 50s are reached late with no bounce

Effect:
- tighten final sigma
- move from residual adjustment to settlement-lock mode

---

## How `model_trust` Should Gate Each Family

Not all families should be gated equally.

### Gating matrix

| Family | Gate by model trust? | Why |
|---|---|---|
| `heat_overrun` | Yes, strongly | Fresh models often partially capture hot regime |
| `boundary_break` | Yes at first, less after confirmation | Models may see setup broadly but miss local timing |
| `postfrontal_persistence` | Yes, moderately | Fresh models often capture the cold regime but may understate persistence |
| `microclimate_confirmation` | No or weakly | Local and timing-sensitive; often model-blind |
| `extreme_lock` | No | A locked extreme is an observed constraint, not a model opinion |

### Recommended v1 formulas

```python
captured_gate = 1.0 - model_trust
boundary_gate = max(0.25, 1.0 - 0.75 * model_trust)
persistence_gate = max(0.30, 1.0 - 0.70 * model_trust)
heat_gate = 1.0 - model_trust
micro_gate = 1.0
lock_gate = 1.0
```

Interpretation:
- even high-trust models do not fully suppress boundary timing and persistence information
- but they do suppress broad “obs hotter/colder than consensus” signals a lot

---

## Sigma Rules from the Two Cases

These cases imply that sigma handling matters as much as mean shifts.

### March 16 behavior
- high side: once the afternoon plateau repeated, high sigma should **compress**
- low side: during boundary-watch / pre-confirmation, low sigma should **expand** because timing uncertainty is high
- after warm-tail removal and endpoint lock, sigma should **compress again**

### March 17 behavior
- high side: once `high_locked_early` is active, high sigma should compress hard
- low side: this is not an outflow-style day, so sigma does **not** need a giant expansion
- instead, low mu should walk colder while sigma stays moderate and then compresses late

### Practical rule

```python
if regime == 'boundary_break' and not extreme_lock:
    sigma_low *= 1.15 to 1.40
elif regime == 'postfrontal_persistence':
    sigma_low *= 0.95 to 1.10

if high_locked_early or high_bias_locked_in:
    sigma_high *= 0.60 to 0.85

if warm_tail_removed or low_endpoint_locked:
    compress warm side / overall sigma depending on implementation
```

---

## Suggested Residual Equations by Side

## High residual

```python
delta_mu_high = (
    heat_gate * w_heat * heat_overrun_score
    + micro_gate * w_micro_high * micro_heat_confirmation_score
)
```

Then apply clamp logic:

```python
if high_locked_early:
    mu_high = min(mu_high + delta_mu_high, running_high + high_lock_buffer)
    sigma_high *= high_lock_sigma_mult
elif high_bias_locked_in:
    sigma_high *= high_plateau_sigma_mult
```

## Low residual

```python
delta_mu_low = (
    boundary_gate * w_boundary * boundary_break_score
    + persistence_gate * w_persist * postfrontal_persistence_score
    + micro_gate * w_micro_low * microclimate_confirmation_score
)
```

Then clamp logic:

```python
if warm_tail_removed:
    remove_or_severely_discount_warm_tail()

if low_endpoint_locked:
    sigma_low *= low_lock_sigma_mult
```

Implementation note:
- `boundary_break_score` and `postfrontal_persistence_score` should generally compete for dominance rather than stack linearly at full strength.

Suggested v1 blend:

```python
dominant_low_score = max(boundary_break_score, postfrontal_persistence_score)
secondary_low_score = min(boundary_break_score, postfrontal_persistence_score)

delta_mu_low = (
    dominant_gate * w_dom * dominant_low_score
    + 0.35 * secondary_gate * w_sec * secondary_low_score
    + micro_gate * w_micro_low * microclimate_confirmation_score
)
```

This prevents the bot from counting both low-side regimes as if they were fully independent.

---

## Training / Snapshot Schema Additions

To support later learning, each checkpoint row should store not just raw features, but also **family and lock states**.

### Required snapshot fields

```python
{
  'target_date': ...,
  'eval_time_lst': ...,
  'market_type': ...,
  'baseline_mu': ...,
  'baseline_sigma': ...,
  'model_trust': ...,
  'primary_regime': ...,
  'path_class': ...,
  'heat_overrun_score': ...,
  'boundary_break_score': ...,
  'postfrontal_persistence_score': ...,
  'microclimate_confirmation_score': ...,
  'extreme_lock_score': ...,
  'high_locked_early': ...,
  'warm_tail_removed': ...,
  'low_endpoint_locked': ...,
  'running_high': ...,
  'running_low': ...,
  'cli_high': ...,
  'cli_low': ...,
  'high_residual_target': cli_high - baseline_high_mu,
  'low_residual_target': cli_low - baseline_low_mu,
}
```

This is important because later LLM- or model-assisted analysis will work much better if the data already contains the **abstractions** we actually care about.

---

## First Unit Tests the Bot Must Pass

These should be encoded as deterministic backtest assertions.

### March 16 — High side
- by midday LST, high estimate should have shifted upward vs cool baseline
- high sigma should compress after repeated 87.8°F plateau
- `B86.5` should become clearly favorable by noon-ish

### March 16 — Low side
- low estimate should **not** overcool too early
- low sigma should widen into the boundary-watch stage
- after boundary confirmation, warm middle brackets should be aggressively discounted
- by late evening, `B74.5` should be strongly disfavored and colder tail favored

### March 17 — High side
- high should be treated as mostly solved in the first hour
- later warm rebound should not be assigned much probability

### March 17 — Low side
- low should drift colder throughout the day even without a dramatic crash
- exact middle-cold brackets should not be preferred over the colder directional tail once persistence is confirmed
- late-evening sigma should not explode; it should tighten as the endgame approaches

---

## What Not to Train On Directly

The new reviews make this clearer.

Do **not** train directly on:
- market labels like `under57_live`
- bracket outcomes as if they were meteorological inputs
- a single isolated dew crash row without regime context
- stale model runs mixed with fresh ones
- bad sensor prints (example: spurious FAWN dew row on March 17)

Train on:
- fresh baseline residuals
- regime/path state
- family scores
- lock/clamp state
- final CLI settlement

---

## Recommended Next Coding Step

Before building a live trader, implement these three pieces in order:

1. **`signal_families.py`**
   - convert raw `SignalState` → family scores + lock states
2. **`regime_classifier.py`**
   - produce `primary_regime` + `path_class`
3. **`build_training_snapshots.py`**
   - write out residual-learning rows using LST checkpoints

Only after those exist should we rewrite `estimator.py` into a full residual estimator.

---

## Research-Derived Design Additions

These additions come from the external literature Blake shared and are worth baking into the architecture now.

---

## 1. Add a Slow Bias-Correction Layer *Before* the Residual Signal Layer

### Why
The NOAA/MDL bias-correction note is a good reminder that there are really **two kinds of forecast error**:

1. **slow-moving structural bias**
   - model changes
   - site-specific warm/cold tendency
   - projection-specific drift
   - seasonal/air-mass biases

2. **fast event residuals**
   - boundary timing
   - local dew-floor behavior
   - microclimate lead/lag
   - late regime breaks

These should not be handled by the same mechanism.

### Design implication
Add a distinct baseline-bias component:

```text
raw model forecast
  → slow bias correction
  → calibrated baseline distribution
  → live residual signal adjustment
```

### Recommended implementation
For each:
- station
- model family
- target variable (`high`, `low`)
- forecast projection / eval checkpoint

carry a decaying-average bias estimate:

```python
delta_next = (1 - alpha) * delta_prev + alpha * recent_error
```

Where `recent_error = forecast - CLI_truth` once verified.

### Important warning from the paper
This kind of correction **cannot respond quickly to same-day event failures**, because the error only becomes known after verification.

So:
- use decay-average bias correction only for **slow baseline correction**
- do **not** rely on it for event timing / late-break logic
- do **not** let it replace the live residual layer

### Practical v1 rule
Use a slow bias layer to shift each model before model combination, but keep the shift small and stable.

Examples:
- if a model has recently been too warm on KMIA lows by ~1.2°F, cool that model’s low baseline slightly
- if a model has recently been too cool on highs by ~0.8°F, warm that model’s high baseline slightly

But never let this layer explain March-16-style boundary timing or March-17-style persistence by itself.

---

## 2. Add a LAMP-Style Very-Short-Range Updater

### Why
The LAMP material strongly supports the idea that **very-short-range local guidance should be built from a combination of**:

- the latest observations
- simple advective / persistence signals
- preexisting MOS/model guidance

That is almost exactly what we need for late-day high/low trading.

### Design implication
Add a **short-range updater** that operates on top of the calibrated baseline and just before the residual estimator.

Recommended scope:
- strongest impact in the final ~0–9 hours of the climate day
- especially useful for:
  - late low timing
  - early locked highs
  - local divergence from the broader grid-scale state

### Inputs for the short-range updater
- latest KMIA obs
- 1h / 2h / 3h temperature trend
- dew trend
- pressure trend
- wind rotation and persistence
- nearby-station trend cluster
- FAWN state
- baseline distribution from the model layer
- simple persistence / advection projections

### Role in the architecture
This updater should not be a separate trader. It should feed the residual layer with short-range features.

Think of it as:
- **baseline bias correction** = slow memory
- **LAMP-style updater** = very short-range local state
- **signal families** = decision abstraction

### Key takeaway
The bot should not choose between “models” and “recent observations.”
It should explicitly combine both, with recent obs getting more influence as settlement approaches.

---

## 3. Combine Models Probabilistically, Not Just by Mean

### Why
The probabilistic ML paper is very aligned with what we want:
- calibrate each model first
- produce a probabilistic forecast for each
- combine those forecasts probabilistically
- then derive a full predictive distribution

That is better than naively averaging means.

### Design implication
The baseline engine should eventually combine **quantiles/distributions**, not just scalar means.

### Recommended baseline-combination approach
For each canonical model family:
1. apply slow bias correction
2. estimate a calibrated predictive distribution for final high/low
3. convert that distribution to quantiles (e.g. p05, p10, p25, p50, p75, p90, p95)
4. combine across model families via weighted quantile averaging
5. reconstruct a full distribution from the combined quantiles

This is cleaner than:
- averaging means
- averaging sigmas
- or throwing all members into one bucket without structure

### Why this helps us specifically
For bracket trading, we care about the full distribution, not just the midpoint.

Examples:
- removing a warm tail
- compressing sigma after `high_locked_early`
- keeping a colder directional tail alive on 3/17

These are naturally distribution-level operations.

---

## 4. Separate Grid-Scale Bias from Point-Scale / Microclimate Variability

### Why
The ecPoint paper’s biggest useful idea for us is this:

> forecast error is not just bias — it is also unresolved point-scale variability whose structure depends on weather type.

For rainfall the paper frames this as:
- grid-scale bias
- sub-grid variability

For our temperature problem, the analogous split is:

- **grid-scale / model-family bias**
- **point-site / microclimate variability and timing**

### Design implication
Do not let one sigma term do all the work.

Split uncertainty into at least two pieces:

```python
sigma_total^2 = sigma_model^2 + sigma_point^2
```

Where:
- `sigma_model` = model disagreement / calibrated forecast uncertainty
- `sigma_point` = local unresolved variability conditioned on regime/path class

### Examples

#### March 16 low
- `sigma_model` might say “cooling likely”
- but `sigma_point` should expand during `boundary_watch`, because local timing is unresolved

#### March 17 low
- `sigma_model` and `sigma_point` should both be moderate, not explosive
- then compress late as the cold-grind endgame becomes obvious

### Key lesson
The residual layer should adjust **both**:
- mean (`mu`)
- point-scale uncertainty (`sigma_point`)

---

## 5. Use Weather-Type / Regime Trees for Calibration

### Why
The ecPoint paper is a strong argument for **physically chosen weather-type classes**, not just generic black-box regression.

Their method uses governing variables to define weather types, then calibrates bias and spread conditional on those types.

That fits our use case extremely well.

### Design implication
The `regime_classifier.py` should not just emit labels for humans.
It should directly control:
- which family scores are allowed to dominate
- which calibration tables are used
- how sigma is adjusted
- how microclimate evidence is weighted

### Candidate governing variables for Miami temperature trading
These should probably appear in the regime tree / weather-type logic:

- local solar time (LST)
- hours remaining to settlement
- current temp anomaly vs baseline
- dew anomaly / dew trend
- pressure tendency
- wind direction family
- wind speed / gustiness
- CAPE / convective signal
- cloud cover / solar suppression
- nearby-station divergence
- FAWN-KMIA spread
- recent model tracking error

### Practical framing
The regime tree should answer questions like:
- Is this a heat-overrun day or not?
- Is this a boundary-break setup or a cold-persistence setup?
- Is local variability high or low?
- Should we trust point-site evidence more than broad models right now?

---

## 6. Consider Non-Local Calibration, Not Just KMIA-Only Training

### Why
The ecPoint framework gets leverage by calibrating on broader data, not requiring a huge local sample for every specific regime.

That is attractive here because:
- Miami-specific settled examples will be limited for a while
- some regimes are rare
- local station-only training will be noisy at first

### Design implication
We should keep two calibration levels:

#### Level A — local KMIA calibration
- station-specific bias
- station-specific microclimate behavior
- local low-floor behavior

#### Level B — non-local / pooled calibration
- regime-conditioned sigma rules
- generic heat-overrun behavior
- generic boundary-break timing behavior
- generic post-frontal persistence behavior

### Practical recommendation
Use pooled/regime-level calibration for early v1 learning, then gradually localize as the settled sample grows.

---

## 7. Calibrate and Verify on Separate Periods

### Why
Both the post-processing literature and common forecasting practice point to the same issue:
- calibration and verification should be separated
- otherwise apparent gains can be fake or overstated

### Design implication
When backtesting:
- use one block of historical settled days for calibration
- use later/out-of-sample days for verification
- never tune family thresholds on the exact days used to claim success

### Practical v1 recommendation
For now:
- use March 16 and 17 as **design / unit-test anchor cases**, not performance proof
- once more settled days are available, move to rolling-origin evaluation

---

## 8. Add a Per-Model Error-Profile Layer, Not Just Per-Model MAE

### Why
The probabilistic framework paper emphasizes learning a model’s **error profile**, not just one average skill number.

That is important.

A model can be:
- good on calm days
- bad on boundary days
- too warm on lows under NNW flow
- fine on highs but bad on late low tails

### Design implication
Instead of storing only:
- one MAE per model

store conditional error profiles like:
- MAE by regime
- signed bias by regime
- MAE by hours-to-settlement bucket
- MAE by daytime vs overnight checkpoint
- performance under stale vs fresh cycles

This should feed both:
- model weights
- model trust

---

## 9. Add an “Observation Value Curve” as Settlement Approaches

### Why
The LAMP logic and our own March 16/17 reviews both imply the same thing:
- observations should matter more as settlement gets closer

### Design implication
The bot should explicitly increase the weight of local observations with decreasing time-to-settlement.

### Suggested v1 behavior
For each checkpoint compute:

```python
obs_value = f(hours_to_settlement, regime, model_trust)
```

Where:
- obs value rises as settlement nears
- obs value rises when model trust is low
- obs value rises when microclimate confirmation is strong
- obs value rises more slowly on stable nonlocal regimes than on timing-sensitive boundary days

This prevents a common failure mode:
- underreacting late because the model baseline still dominates too much

---

## 10. Keep the Architecture Auditable

### Why
All four sources point, implicitly or explicitly, toward structured post-processing — not black-box vibes.

That matches what we want.

### Design implication
Every cycle should log:
- raw baseline before correction
- slow bias correction applied
- calibrated quantiles
- regime/path classification
- family scores
- gating values
- clamp states
- final quantiles / bracket probabilities
- reasons for major changes

This makes it possible to answer:
- why did the bot shift colder?
- was it model bias correction, persistence regime, or microclimate evidence?
- did it overreact because multiple raw signals were counted as separate evidence?

---

## Architecture Revision from the Literature

The architecture should now be thought of as:

```text
raw model families
  → slow bias correction (decaying average / error profile)
  → calibrated per-model distributions
  → probabilistic baseline combination (quantiles)
  → LAMP-style short-range updater features
  → regime / weather-type classification
  → signal-family residual adjustment
  → clamps / tail removal
  → final predictive distribution
  → market comparison / trade policy
```

That is better than the earlier simpler view because it cleanly separates:
- structural bias
- broad model uncertainty
- local short-range state
- event-regime residuals
- execution

---

## Most Important New Takeaways

If I compress all four sources into the most useful design changes, it is this:

1. **Add a slow bias-correction layer before the residual layer**
2. **Use a LAMP-style short-range updater driven by recent observations**
3. **Combine models as calibrated distributions / quantiles, not just means**
4. **Condition both bias and sigma on regime / weather type**
5. **Separate grid-scale/model error from point-site/microclimate variability**
6. **Use pooled regime calibration early, then localize later**
7. **Make observation value rise as settlement approaches**

---

## Edge Triage from Brainstorming (Keep / Modify / Drop)

This section translates the brainstormed “Miami Weather Bot Edges” into architecture decisions.

The standard for inclusion is strict:
- does this improve **KMIA CLI settlement prediction**?
- can it be implemented **modularly**?
- can it be **debugged independently**?
- does it add new information, or does it just sound sophisticated?

If an idea does not survive those tests, it should not become part of the core bot.

---

## Keep as Core Architecture

These ideas are genuinely useful and should be part of the Miami bot design.

### 1. Settlement Source Precision — **KEEP, highest priority**

This is the best edge and should anchor the whole system.

#### Correct framing
The bot is not trying to predict “Miami weather” in a generic sense.
It is trying to predict:
- the **official KMIA CLI daily high**
- the **official KMIA CLI daily low**
- over the **LST climate day**
- as settled by Kalshi

#### Architecture consequence
Add an explicit **settlement layer** with its own module(s):

```text
settlement/
  climate_clock.py
  contract_mapping.py
  cli_truth.py
  official_station.py
```

#### Responsibilities

##### `climate_clock.py`
- define the climate day in **LST only**
- convert UTC timestamps to LST settlement buckets
- compute hours-to-settlement
- expose helpers like:
  - `climate_day_bounds(target_date)`
  - `to_lst(ts_utc)`
  - `hours_remaining(ts_utc, target_date)`

##### `contract_mapping.py`
- map market tickers to:
  - target date
  - market type (`high`, `low`)
  - bracket floor/ceiling
  - under/over directional semantics

##### `cli_truth.py`
- parse/store the official CLI once available
- expose authoritative settlement high/low and event times
- keep truth distinct from observational path data

##### `official_station.py`
- define station-specific constants for KMIA:
  - official ICAO/CLI source
  - climate-day rules
  - any known reporting quirks

#### Debugging rule
Every final forecast object must carry:
- `target_date`
- `market_type`
- `lst_window_start`
- `lst_window_end`
- `official_station='KMIA'`

No module should infer settlement semantics on its own.

---

### 2. Custom Multi-Source Dynamic Weighting — **KEEP**

This is a real edge and fits the architecture cleanly.

#### Correct framing
The edge is not “make the fanciest blend.”
The edge is:
- dynamic weighting by **regime**
- dynamic weighting by **time to settlement**
- dynamic weighting by **live tracking skill**
- dynamic weighting by **source freshness**

#### Architecture consequence
Implement a dedicated baseline module stack:

```text
engine/
  source_registry.py
  bias_memory.py
  baseline_engine.py
  quantile_combiner.py
```

#### Responsibilities

##### `source_registry.py`
Each source must implement a common interface.

```python
class ForecastSource(Protocol):
    name: str
    family: str
    def snapshot(self, target_date: str, eval_time_utc: datetime) -> ForecastSourceSnapshot: ...
```

Each snapshot should contain:
- `source_name`
- `family_name`
- `issued_at`
- `forecast_high`
- `forecast_low`
- optional quantiles
- tracking metadata
- freshness metadata

##### `bias_memory.py`
Maintain per-source slow bias memory.

Suggested key dimensions:
- station
- source family
- target variable (`high`/`low`)
- season / regime bucket (optional later)
- eval checkpoint bucket (e.g. morning / midday / evening / final-hours)

##### `baseline_engine.py`
- apply slow bias corrections
- compute source freshness
- compute tracking error
- compute source weight
- emit calibrated baseline distributions for high/low

##### `quantile_combiner.py`
- combine per-source predictive quantiles rather than just means
- expose:
  - `combined_quantiles`
  - derived `mu`
  - derived `sigma`
  - source contribution breakdown

#### Debugging rule
Source weights must be logged explicitly for every cycle.
If the bot is wrong, we must be able to answer:
- which sources dominated?
- were they stale?
- were they all the same family?

---

### 3. Latency + Mesonet / Hyperlocal Observation Ingest — **KEEP**

This is a real information edge, but only if quality-controlled and translated correctly.

#### Correct framing
The edge is not “raw extra sensors = free money.”
The edge is:
- faster access to **relevant local state changes**
- especially near settlement
- before slower participants or broad model baselines fully digest them

#### Architecture consequence
Build a separate local-observation ingestion and QC layer:

```text
data/
  kmia_obs_adapter.py
  nearby_obs_adapter.py
  fawn_adapter.py
  weatherstem_adapter.py   # optional plugin
  provisional_qc.py
  station_cluster.py
```

#### Responsibilities

##### `provisional_qc.py`
Must detect and flag:
- impossible dew points
- backward timestamps
- outlier jumps
- sensor dropouts
- obviously stale feeds

##### `station_cluster.py`
Build local station groups such as:
- coastal airport group
- inland/western group
- southern/Homestead group
- immediate-nearby airport group

The goal is to derive cluster features like:
- leading cold cluster
- warm hold cluster
- coastal lag vs inland crash
- median cluster temp trend

#### Debugging rule
Never let raw provisional data directly change the final forecast.
All provisional feeds must first produce:
- QC flags
- feature transforms
- confidence score

---

### 4. Automated Forecast-vs-Market Edge Detection — **KEEP**

This is part of the execution stack, not the weather stack.

#### Correct framing
The weather model should output calibrated probabilities.
A separate market layer should decide whether those probabilities imply an actionable trade.

#### Architecture consequence
Create a strict separation:

```text
engine/
  bracket_pricer.py
  edge_detector.py
execution/
  trade_policy.py
  order_router.py
```

#### `edge_detector.py` responsibilities
Given:
- final predictive distribution
- live market book
- fees/slippage assumptions

compute:
- fair probability per bracket
- fair value per bracket
- tradable edge after fees/slippage
- directional vs exact-bracket ranking

#### Debugging rule
The market layer must not feed back into the weather belief.
Market prices are for execution/diagnostics only.

---

### 5. Probabilistic Tail Trading — **KEEP, but only as a distribution feature**

This is useful, but only if tails are calibrated.

#### Correct framing
The edge is not “bet tails because people are dumb.”
The edge is:
- compute better tail probabilities than the market
- especially when regime/path implies asymmetric tail risk

#### Architecture consequence
Tail logic belongs in:
- `quantile_combiner.py`
- `distribution_ops.py`
- `bracket_pricer.py`

Add explicit support for:
- warm-tail removal
- cold-tail extension
- one-sided sigma compression
- lock states near settlement

#### Debugging rule
Tail bets must be auditable via:
- which family widened or narrowed the tail?
- was the tail supported by regime/path evidence?
- did the tail survive out-of-sample verification?

---

## Keep, but Modify Heavily

These ideas are useful only after reframing.

### 6. Multi-AI Ensemble Blending — **MODIFY heavily**

#### Why the original framing is weak
“Run GraphCast + Pangu + FuXi + AIFS and blend them” sounds powerful, but by itself it is not a true architecture principle.

Problems:
- these models are highly correlated at the synoptic scale
- they are not settlement-specific by default
- treating each as a fully independent vote inflates confidence
- raw AI output is not automatically better at KMIA daily high/low settlement than calibrated traditional guidance

#### What to keep
Treat AI models as an **optional upstream model family**, not as four separate magical edges.

#### Architecture consequence
Implement AI models behind a single plugin boundary:

```text
sources/
  ai_family/
    graphcast_adapter.py
    pangu_adapter.py
    fuxi_adapter.py
    aifs_adapter.py
    ai_family_blender.py
```

#### Rule
For v1, collapse them into one family, e.g. `ai_global`.

That family may internally blend multiple AI models, but the baseline combiner should initially see them as:
- one family
- one set of quantiles
- one contribution weight

Only split them later if backtests prove they add independent signal.

#### Debugging rule
If AI models are wrong together, the bot should lose one family vote, not four.

---

### 7. Hyperlocal Urban Heat Island Correction — **MODIFY into a KMIA transfer-function module**

#### Why the original framing is dangerous
A blanket statement like “Miami neighborhoods run 6°F hotter than KMIA, so add +6°F” is exactly the wrong target.

The contract settles on **KMIA CLI**, not on general Miami heat.

#### What to keep
Use hyperlocal data only to predict what **KMIA** will do.

So the correct concept is not:
- `urban_heat_bias`

It is:
- `official_station_transfer`

#### Architecture consequence
Add a dedicated module:

```text
engine/
  station_transfer.py
```

#### Responsibilities of `station_transfer.py`
Transform local non-KMIA observations into KMIA-relevant features, such as:
- inland-vs-coastal spread
- FAWN-KMIA spread
- western-station crash lead
- sea-breeze penetration proxy
- nocturnal decoupling risk
- humidity retention differences

Critically, this module should output:
- **features or residual hints**
- not a direct additive “+6°F correction”

#### Example outputs
```python
@dataclass
class StationTransferFeatures:
    inland_minus_kmia_temp: float
    inland_cooling_lead_score: float
    coastal_lag_score: float
    microclimate_confidence: float
    kmia_transfer_hint_high: float
    kmia_transfer_hint_low: float
```

The `kmia_transfer_hint_*` values should be weak priors, not direct truth.

#### Debugging rule
If the bot changes its view because of hyperlocal data, logs must say:
- which station cluster drove the change?
- how does that historically map to KMIA?
- was it a high-side or low-side effect?

---

### 8. Non-Forecast Execution Edges — **MODIFY into optional execution plugins**

#### What to keep
- faster execution
- order automation
- optional cross-market comparison
- optional latency optimization

#### What to avoid
Do not fuse these with weather modeling.

#### Architecture consequence
Keep execution-only ideas behind a separate module boundary:

```text
execution/
  order_router.py
  position_manager.py
  cross_market_compare.py   # optional
  latency_metrics.py
```

#### Rule
Execution plugins should consume:
- fair values from the weather stack
- live books from exchanges

They should not alter weather probabilities.

---

## Drop or Explicitly Reject as Core Architecture

These are not strong enough — or are too misleading — to become design principles.

### A. “AI crushes traditional MOS, so AI blending is the main alpha” — **DROP as a planning assumption**
We may use AI models, but we should not assume raw AI superiority at settlement-station high/low prediction without verification.

### B. “Simple >10% discrepancy = auto-signal” — **DROP as the execution rule**
Edge thresholds must be bid/ask-, fee-, slippage-, and confidence-aware.
A flat 10% rule is too crude.

### C. “Urban heat island = direct additive +6°F bias layer” — **DROP completely**
That targets the wrong variable. Predict KMIA, not neighborhood heat.

### D. “Tail betting is inherently profitable” — **DROP as a blanket belief**
Tail trading is only useful if the tails are calibrated and regime-conditioned.

---

## Recommended Modular Build for Codex / GPT-5.4

The design should be implemented as composable modules with narrow responsibilities.

### Core weather stack

```text
settlement/
  climate_clock.py
  contract_mapping.py
  cli_truth.py
  official_station.py

sources/
  gfs_adapter.py
  hrrr_adapter.py
  nbm_adapter.py
  ecmwf_adapter.py
  ai_family/
    graphcast_adapter.py
    pangu_adapter.py
    fuxi_adapter.py
    aifs_adapter.py
    ai_family_blender.py

data/
  kmia_obs_adapter.py
  nearby_obs_adapter.py
  fawn_adapter.py
  weatherstem_adapter.py
  provisional_qc.py
  station_cluster.py

engine/
  source_registry.py
  bias_memory.py
  baseline_engine.py
  quantile_combiner.py
  short_range_updater.py
  station_transfer.py
  regime_classifier.py
  signal_families.py
  residual_estimator.py
  distribution_ops.py
  bracket_pricer.py
  edge_detector.py

execution/
  trade_policy.py
  order_router.py
  position_manager.py
  cross_market_compare.py   # optional
  latency_metrics.py        # optional

analysis/
  build_training_snapshots.py
  backtest_baseline.py
  backtest_residual_bot.py
  backtest_execution.py
```

### Architectural rule
Each module must answer **one question only**:
- settlement semantics?
- source ingestion?
- baseline calibration?
- local obs interpretation?
- regime classification?
- residual adjustment?
- pricing?
- execution?

If one module is doing multiple of these, it is too broad.

---

## Recommended Default Decisions for v1

These are defaults that Codex should implement unless Blake explicitly overrides them.

### Default 1 — AI models are optional and collapsed into one family
- implement adapters if easy
- but baseline sees one `ai_global` family
- if unavailable, bot still works without them

### Default 2 — WeatherSTEM / mesonet data are feature sources, not truth sources
- use them for transfer features and microclimate confirmation
- never let them directly replace KMIA or CLI

### Default 3 — Nearby/FAWN can always affect sigma, but affect mu only under conditions
Nearby/FAWN/local clusters may affect:
- `sigma` always, through uncertainty / confirmation
- `mu` only when:
  - `hours_to_settlement <= 6`
  - or `microclimate_confirmation_score >= threshold`
  - or the regime is explicitly timing-sensitive

### Default 4 — Directional tails beat exact middle brackets when the regime is still evolving
In edge ranking:
- if distribution is still moving and sigma is not compressed, prefer directional tail expressions
- exact brackets should only dominate when the distribution is already concentrated

### Default 5 — Observation value rises toward settlement
Suggested monotone rule:

```python
obs_weight = clip(1 - hours_to_settlement / 18, 0.15, 1.0)
```

Then modulate by:
- model trust
- regime/path
- local microclimate confidence

### Default 6 — Boundary-break and post-frontal-persistence are mutually dominant low-side families
Do not let both apply at full strength simultaneously.
Use dominant/secondary blending as already specified elsewhere in this doc.

### Default 7 — Market logic stays downstream
No market price may be used as a weather feature in the residual model.

---

## Implementation Notes for Each Surviving Edge

### Settlement precision implementation notes
- unit-test climate-day conversion with DST dates
- unit-test March 16 and March 17 against their correct LST windows
- require all reviews/snapshots/backtests to store LST timestamps explicitly

### Dynamic weighting implementation notes
- store weights by source family, not source instance only
- separate high and low source weights
- log pre- and post-bias-corrected source forecasts

### Local ingest implementation notes
- create a QC confidence score per feed
- support missing feeds gracefully
- never crash the cycle because WeatherSTEM or FAWN is unavailable

### AI-family implementation notes
- if adapters are missing or unavailable, return `source_unavailable`
- do not block the baseline combiner
- support a config flag: `enable_ai_family=false`

### Tail-pricing implementation notes
- expose p05/p10/p25/p50/p75/p90/p95
- compute per-bracket fair values from the final distribution
- support asymmetric/clamped distributions

### Execution implementation notes
- compute fair value vs best bid/ask, not midpoint only
- include fee/slippage model before signaling edge
- keep cross-market arb optional and entirely outside weather belief

---

## Minimal End-to-End Cycle (Recommended)

This is the cleanest cycle for a functioning but debuggable v1 bot.

```text
1. Build settlement context (LST, target date, contract mapping)
2. Pull canonical model-family snapshots
3. Apply slow bias memory to each source family
4. Build per-source predictive quantiles
5. Combine into baseline quantiles/distribution
6. Pull KMIA + nearby + FAWN + optional WeatherSTEM data
7. QC all local feeds and derive station-cluster features
8. Build LAMP-style short-range updater features
9. Run station-transfer module (local state → KMIA hints)
10. Classify regime/path
11. Compute signal-family scores
12. Apply residual adjustments + clamps
13. Produce final predictive distribution
14. Price brackets from the final distribution
15. Compare to live market book
16. Apply execution policy / optional routing
17. Log all intermediate objects for debugging
```

If the code path does more than this in v1, it is probably trying to be too clever too early.

---

## Recommended First Coding Order After This Doc

1. `settlement/climate_clock.py`
2. `engine/source_registry.py`
3. `engine/bias_memory.py`
4. `engine/quantile_combiner.py`
5. `data/provisional_qc.py`
6. `data/station_cluster.py`
7. `engine/short_range_updater.py`
8. `engine/station_transfer.py`
9. `engine/regime_classifier.py`
10. `engine/signal_families.py`
11. `engine/residual_estimator.py`
12. `engine/bracket_pricer.py`
13. `engine/edge_detector.py`
14. `execution/trade_policy.py`

This order preserves modular debugging and avoids burying the real weather logic inside the execution layer.

---

## Open Questions / Decisions for Blake

These do not block the design doc, but they should be decided before coding the trading version.

1. **Preferred checkpoint times for training snapshots**
   - fixed clock times only?
   - or also “event-driven” snapshots when a family score crosses threshold?

2. **How aggressive should clamp logic be?**
   - conservative clamps (only when certainty is extreme)
   - or more aggressive late-day tail removal?

3. **Do you want ensemble members in the baseline at all, or only deterministic models + ensemble spread?**

4. **Should FAWN / nearby station signals be allowed to override high model trust, or only modify sigma?**

5. **Do we want regime-specific weights hand-coded first, or a global residual model with regime as an input feature?**

6. **How do you want trade policy to behave on “pretty exact bracket vs directional tail” situations?**
   - default toward directional tail?
   - or only when confidence exceeds a threshold?

---

## Bottom Line

The correct architecture is:

- **fresh models create the baseline**
- **signals are grouped into families**
- **families predict residuals, not raw final temps**
- **model trust gates model-captured signals**
- **microclimate signals retain independent power**
- **clamps handle locked/extreme states**
- **highs and lows are modeled separately**
- **everything runs on LST**

That should give us a bot that learns from March 16 / March 17 correctly without naively counting the same weather fact three times.
