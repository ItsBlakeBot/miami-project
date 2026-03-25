# Miami Weather Trading Bot — Architecture Plan

## System Overview

```
Phase 1 (NOW)          Phase 2              Phase 3           Phase 4            Phase 5
DATA COLLECTION  --->  ENSEMBLE ENGINE --->  PRICING     --->  EXECUTION    --->  SETTLEMENT
                                             ENGINE            ENGINE             & LEARNING

[Wethr SSE]            [Model Weights]       [Bracket Probs]   [Kelly Sizing]     [CLI Ingest]
[Wethr REST]     --->  [Weighted Mean]  ---> [vs Market    --> [Taker/Maker] ---> [Model Scoring]
[Open-Meteo]           [EMOS Calibrate]      [Implied Prob]    [Safety Gates]     [Weight Update]
[NWS]                  [Obs Condition]       [Edge Calc]       [Order Lifecycle]  [PnL Tracking]
[Kalshi WS/REST]       [Distribution Fit]                      [Kalshi WS Orders]
       |
       v
  [SQLite DB]
```

## Phase 1: Data Collection (Current)

Collecting raw data for KMIA (Miami International Airport):

| Source | Data | Transport | Frequency |
|--------|------|-----------|-----------|
| Wethr SSE | 1-min observations, new_high/low, DSM, CLI events | SSE stream | Real-time |
| Wethr REST | NWP model forecasts (daily + hourly), NWS forecast versions | REST | 5 min |
| Open-Meteo | 11 deterministic models, 82 ensemble members, pressure levels | REST | 5 min / hourly |
| NWS | Latest ASOS observation, CLI reports | REST | 2.5 min / 30 min |
| Kalshi WS | Orderbook deltas, ticker, trades | WebSocket | Real-time |
| Kalshi REST | Market discovery, settlement checks | REST | 10 min |

**Database**: SQLite with WAL mode, 13 tables, indexed for time-series queries.

**Scoring Pipeline** (daily after CLI):
1. Daily model MAE/bias/RMSE — HIGH and LOW scored independently
2. Diurnal scoring — every model valid_time vs nearest 1-min obs
3. Cloud impact tracking — obs vs forecast sky cover paired with temp error

## Phase 2: Ensemble & Distribution Engine (Planned)

### Model Weight Calculation
- **Freshness-aware**: Use each model's latest run only (1 vote per model)
- Weight by accuracy score (power-law: `weight = 1 / MAE^power`)
- HIGH and LOW weights computed independently
- Optional mild freshness decay for models >12h stale
- Prevents fast-updating models (GFS hourly) from dominating slow-but-accurate ones (ECMWF 12h)

### Weighted Ensemble
- Weighted mean of all models' forecasts
- Weighted standard deviation for spread estimate
- Separate ensemble for HIGH and LOW

### EMOS Calibration
- Ensemble Model Output Statistics
- Calibrate ensemble spread to match observed reliability
- Fit linear regression: `calibrated = a * raw_mean + b`
- Calibrate variance: `calibrated_var = c * raw_var + d`

### Observation Conditioning
- As the day progresses, narrow the distribution
- Running high becomes a hard floor (temp already reached)
- Warming stall detection: if temp plateaus during peak heating, cap the distribution
- Dew point floor for lows: in Miami, dew point constrains minimum temperature
- Update distribution mean/variance based on obs-model divergence

### Distribution Fitting
- Fit Student-t distribution to ensemble + obs-conditioned parameters
- Heavier tails than Normal to capture rare events
- Compute CDF for each bracket boundary

## Phase 3: Pricing Engine (Planned)

### Bracket Probability Calculation
- For each bracket [floor, cap]: `P(bracket) = CDF(cap) - CDF(floor)`
- Handle open-ended brackets (below-all, above-all)
- Probabilities must sum to 1.0 across all brackets

### Market Comparison
- Extract implied probabilities from Kalshi orderbook mid-prices
- `implied_prob = mid_price / 100`
- Compute edge: `our_prob - implied_prob` per bracket
- Track edge over time for decay analysis

### Edge Filtering
- Minimum edge threshold to trade (e.g., 5 cents)
- Morning edge window: largest divergence before 11am EST
- Edge erodes through the day as market incorporates obs data
- Never trade penny contracts (<5c) on the short side

## Phase 4: Execution Engine (Planned)

### Position Sizing
- Kelly criterion: `f = edge / (odds - 1)` adjusted for uncertainty
- Cold-start fraction: reduce size during first 2 weeks of scoring
- Per-bracket exposure limits
- Total portfolio exposure cap

### Order Types
- **Taker mode**: Cross the spread when edge exceeds threshold + spread cost
- **Maker mode**: Post limit orders at model fair value, capture spread
- Hybrid: taker for high-confidence, maker for marginal edges

### Safety Gates
- Observation staleness check: don't trade if last obs >10 min old
- Blackout window around DSM time (~11:00Z)
- Penny contract protection: never sell at <5c (tail risk is catastrophic)
- Maximum position size per bracket
- Cool-down period after large losses

### Order Lifecycle (via Kalshi WebSocket)
- Place order -> monitor fill via WS `fill` channel
- Cancel stale orders (unfilled after N minutes)
- Track partial fills
- `user_orders` channel for order status updates

## Phase 5: Settlement & Learning (Planned)

### CLI Settlement
- Parse CLI report issued ~5:30am EST (covers previous climate day)
- Extract official MAX and MIN temperature
- **Critical**: Use the "FOR" date (climate day), not the issue date

### Model Scoring
- Update MAE, bias, RMSE per model (15-day rolling window)
- HIGH and LOW scored independently
- Diurnal scoring at every model valid_time
- Recalculate ensemble weights

### Performance Tracking
- Per-trade PnL
- Brier score for probability calibration
- Market vs model accuracy comparison
- Cumulative returns with drawdown tracking

## Trading Strategy

### Kalshi Fee Structure

| | Formula | Max Fee (at 50¢) |
|---|---|---|
| **Taker** | `0.07 × C × P × (1-P)` | 1.75¢ |
| **Maker** | `0.0175 × C × P × (1-P)` | ~0.44¢ |

- Maker fee is 1/4 of taker — **maker-first strategy is critical**
- **Fee rebate tiers**: first $750 full fees → $750-$2K at 60% rebate → >$2K at 80% rebate
- At 80% rebate: effective taker max ~0.35¢, maker max ~0.088¢
- No settlement fees. Weather contracts use standard rates (no S&P/Nasdaq discount)
- Fee formula is symmetric for YES/NO at same price level

### EV Calculation

```
Single bracket:  EV = p_est - p_market
Per-contract:    profit_if_win = (1 - q),  loss_if_lose = q
                 EV = p × (1 - q) - (1 - p) × q = p - q
Edge ratio:      edge% = (p - q) / q
```

Full bracket distribution: compute `EV_i = p_i - q_i` for all brackets. Bracket prices
sum to ~1.0 + vig; you need aggregate edge exceeding the vig just to break even.

### Kelly Criterion for Binary Contracts

```
f* = (p - q) / (1 - q)
```

Where `p` = estimated probability, `q` = market price. This is the fraction of bankroll
to spend buying contracts at price `q`.

**Example**: Model says P(84-86F) = 0.35, market at 0.28:
- `f* = (0.35 - 0.28) / (1 - 0.28) = 0.097` → spend 9.7% of bankroll
- At quarter Kelly: 2.4% of bankroll

**Use quarter Kelly (0.25x)**. Full Kelly is optimal only with perfect probability
estimates. We have estimation error from: model bias, ensemble-to-probability conversion,
limited calibration data, and regime dependence. Quarter Kelly gives ~50% of full Kelly's
growth rate with dramatically smaller drawdowns.

### Bankroll Management

| Constraint | Limit | Rationale |
|---|---|---|
| Total capital at risk | 15-25% | Survival margin through drawdowns |
| Per bracket | 5% max | Hard cap regardless of Kelly output |
| Per event (all brackets) | 8-10% | Same-event brackets are correlated |
| Per day (all stations) | 20% | Cross-station weather correlation |
| Drawdown: halve size | -15% from peak | Protect capital during cold streaks |
| Drawdown: pause trading | -25% from peak | Re-evaluate calibration before continuing |

### Maker vs Taker Strategy

| Scenario | Strategy |
|---|---|
| **Default** | Post maker orders at fair value. Let fills come to you. |
| **Strong signal, time-sensitive** | Cross spread as taker (warming stall, CLI imminent) |
| **Wide spread (>5¢)** | Post inside the spread for better fill price |
| **Thin liquidity** | Limit orders only, reduce size |
| **< 5¢ contracts** | Skip entirely |

Morning window (6am-11am local) has the most edge — models updated overnight, market
hasn't fully repriced. By afternoon, obs are converging and edge shrinks.

### Penny Contract Rules

Skip contracts priced below 5¢:
- Tail calibration is the weakest part of any weather model
- Thin liquidity → wide spreads → bad fills
- Fee impact is proportionally huge (1.75¢ on a 5¢ contract = 35%)
- The extreme scenarios these pay off on are exactly where model confidence is lowest

### Correlated Position Handling

Same-event brackets are **mutually exclusive** — only one wins per settlement.
- Cov(bracket_i, bracket_j) = `-p_i × p_j` (negative correlation)
- Do NOT size each bracket independently — treat as a single portfolio
- **Heuristic**: Pick 1-2 best brackets per event by edge-to-price ratio
- Total event exposure capped at 8-10% regardless of how many brackets show edge
- Cross-station: reduce combined sizing 30-50% for stations within 200mi under same synoptic pattern

---

## Prediction Uncertainty Framework

The core problem: **how uncertain should we really be?** Raw model output and even
EMOS-calibrated distributions can be overconfident in ways that don't show up until
you've lost money. This framework adds layers of uncertainty accounting on top of
the base EMOS calibration.

### Layer 1: EMOS Calibration (Base)

```
Corrected Mean     = a + b × ensemble_mean
Corrected Variance = c + d × ensemble_variance
```

Coefficients trained on 30-60 day rolling window of forecast-observation pairs.
Separate fits for HIGH and LOW. This corrects systematic bias and underdispersion
(ensembles are typically overconfident — spread is too narrow).

Bracket probabilities: fit calibrated Normal (or Student-t for heavier tails),
integrate over each bracket range.

### Layer 2: Calibration Quality Score

EMOS tells you the *corrected* distribution, but not how much to trust it.
Track calibration quality continuously:

**Brier Score Decomposition**:
```
BS = Reliability - Resolution + Uncertainty
```
- **Reliability**: How close are your stated probabilities to observed frequencies?
  (Low = well-calibrated)
- **Resolution**: Can you distinguish high-prob days from low-prob days?
  (High = informative forecasts)
- **Uncertainty**: Base rate entropy (not controllable)

Compute Brier reliability for each probability bin (0-10%, 10-20%, ..., 90-100%).
If your "70% confident" brackets only hit 55% of the time, you are systematically
overconfident in that range. Apply a **calibration adjustment factor**:

```
adjusted_prob = observed_hit_rate_for_bin(raw_prob)
```

This is essentially a second-pass recalibration on top of EMOS, using your own
trading history.

**Prediction Interval Coverage**:
Track what percentage of observations fall within your 80% prediction intervals.
- Target: 80% coverage
- If only 65% of obs land inside → intervals are too narrow → widen variance by
  `0.80 / 0.65 = 1.23x`
- If 90% land inside → intervals are too wide → tighten (but be conservative —
  overconfidence is costlier than underconfidence)

### Layer 3: Regime-Dependent Uncertainty

Not all weather days are equally predictable. Classify into regimes and apply
regime-specific uncertainty multipliers:

| Regime | Predictability | Variance Multiplier | Detection |
|---|---|---|---|
| **Stable high pressure** | High | 0.8x | Light winds, CLR/FEW sky, low CAPE |
| **Land/sea breeze battle** | Medium | 1.2x | Onshore flow, SCT-BKN near coast |
| **Frontal passage** | Low | 1.5x | Ensemble bimodality, high spread |
| **Tropical moisture** | Low | 1.5x | High PW (>50mm), CAPE >1000 |
| **Post-frontal dry** | High | 0.8x | NW flow, low dew points, CLR sky |
| **Radiation fog/low stratus** | Low (for lows) | 1.3x | Light wind, high RH at sunset |

Detection uses data we already collect: CAPE, PW, pressure levels, sky cover, wind.
Regime classification feeds into the variance term of the EMOS distribution:
```
final_variance = emos_variance × regime_multiplier
```

### Layer 4: Sample Size Penalty

EMOS coefficients trained on limited data are themselves uncertain. Apply a
penalty that widens the distribution when calibration data is thin:

```
sample_penalty = 1 + (k / sqrt(n_calibration_days))
```

Where `k` is a tunable parameter (~2-3) and `n_calibration_days` is the number
of days in the training window with valid forecast-observation pairs.

- At 60 days of data: penalty = `1 + 2/7.7 = 1.26x` variance
- At 120 days: penalty = `1 + 2/10.95 = 1.18x`
- At 365 days: penalty = `1 + 2/19.1 = 1.10x`

This prevents overconfidence during cold-start and after regime changes where
historical data may not represent current conditions.

### Layer 5: Ensemble Agreement Score

Beyond raw spread, measure *qualitative* ensemble agreement:

**Bimodality detection**: If the ensemble distribution has two peaks (e.g., half
the models say 82F, half say 88F due to a sea breeze timing split), the standard
deviation understates uncertainty. Fit a Gaussian mixture model; if two-component
BIC < one-component BIC, flag bimodal and **use the wider component** or
**skip the trade entirely** (market is a coin flip between two regimes).

**Model cluster analysis**: Group models by forecast value. If you have 3 tight
clusters rather than a smooth spread, uncertainty is regime-dependent, not Gaussian.
Widen variance or use the mixture model directly for bracket probabilities.

**Outlier model flag**: If one model is >3σ from the ensemble mean, either it
sees something others don't (potentially valuable) or it's broken. Track whether
the outlier model has been the *right* outlier historically — if not, downweight
and widen slightly.

### Layer 6: Temporal Confidence Decay

Your forecast edge decays as settlement approaches and the market incorporates
the same observations you see:

| Time to Settlement | Confidence Multiplier | Kelly Multiplier |
|---|---|---|
| T-24h to T-18h | 1.0x | 1.0x (full quarter-Kelly) |
| T-18h to T-12h | 0.9x | 0.85x |
| T-12h to T-6h | 0.75x | 0.70x |
| T-6h to T-3h | 0.6x | 0.50x |
| T-3h to settlement | 0.4x | 0.30x |

The confidence multiplier widens the distribution (less certain); the Kelly
multiplier reduces position size (less edge). These compound: wider distribution
= lower peak bracket probability = smaller Kelly fraction anyway.

**Exception**: Intraday signals that the market hasn't priced can *increase*
local confidence:
- Running high already exceeded a bracket floor → that bracket locks to ~100%
- Warming stall confirmed by 3+ consecutive flat readings → cap distribution
- Sea breeze arrival detected 2h early → shift distribution down

### Layer 7: Meta-Uncertainty (Uncertainty About Our Uncertainty)

The deepest layer: how well does our *entire pipeline* perform?

**Rolling Brier Skill Score (BSS)**:
```
BSS = 1 - (BS_model / BS_climatology)
```

Compare our bracket probabilities against a naive climatology baseline. If BSS
is low (<0.05), our model adds almost no value over "just use the historical
average." In this case:
- Reduce all Kelly fractions by 50%
- Widen all distributions by 1.5x
- Focus on data collection improvement rather than trading

**Profit-Weighted Calibration**: Track not just whether predictions are calibrated,
but whether the *trades we actually make* are calibrated. It's possible to be
well-calibrated overall but systematically wrong on the specific brackets where
we have the most capital deployed (adverse selection). Compute Brier score
weighted by position size — this is the number that actually matters for PnL.

**Adaptive Kelly Fraction**: Instead of fixed quarter-Kelly, let the Kelly
fraction float based on recent calibration quality:

```
kelly_mult = base_mult × calibration_quality
```

Where `base_mult = 0.25` and `calibration_quality` ranges from 0.5 (poor recent
calibration → use 1/8 Kelly) to 1.5 (excellent calibration → use 3/8 Kelly).
Calibration quality derived from rolling 14-day Brier reliability score.

### Putting It All Together

For each bracket trade decision:

```
1. EMOS calibration      → base distribution (mean, variance)
2. Calibration quality   → adjust probabilities by historical reliability
3. Regime detection      → multiply variance by regime factor
4. Sample size penalty   → widen variance for limited training data
5. Ensemble agreement    → check for bimodality, widen if needed
6. Temporal decay        → widen variance based on time to settlement
7. Meta-uncertainty      → adjust Kelly fraction based on pipeline BSS

Final: bracket_prob → Kelly sizing → bankroll limits → order decision
```

Each layer only *widens* uncertainty or *reduces* position size. The system is
structurally conservative: every source of doubt makes us trade smaller, never
larger. The only way to trade bigger is to *earn* it through demonstrated
calibration quality over time.

---

## Market Trading Principles

### Adverse Selection Awareness

When you post a maker order and get filled, ask: **why did someone cross the
spread to trade against me?** Possible reasons:

1. **Informed flow**: They have better/newer information (e.g., they see a cloud
   bank on radar you haven't priced in). This is the dangerous case.
2. **Rebalancing flow**: Someone closing a position or hedging. Uninformed — this
   is the fill you want.
3. **Retail/recreational flow**: Kalshi has significant retail participation on
   weather. Often uninformed and price-insensitive.

**Detecting adverse selection**:
- Track fill rate vs subsequent market movement. If the market consistently moves
  against you within 5 minutes of a fill, you're being adversely selected.
- Compute "realized spread": your fill price minus the midpoint 5 min after fill.
  Negative realized spread = adverse selection is eating your edge.
- If realized spread is consistently negative, you need to either widen your
  quotes (post further from mid) or reduce maker activity during those periods.

**Time-of-day pattern**: Adverse selection is highest when new model runs drop
(00Z, 06Z, 12Z, 18Z GFS cycles, plus whenever ECMWF updates). Sophisticated
participants reprice immediately; if your quotes are stale, you get picked off.

**Mitigation**:
- Cancel and rebook all resting orders within 2 minutes of a model update
- During the first 15 minutes after a major model cycle, reduce maker size by 50%
  or switch to taker-only (you want to be the one repricing, not getting picked off)
- Track which brackets get filled most often — if penny brackets fill suspiciously
  fast, someone knows something about the tail

### Position Exit & Repricing Rules

Entry is half the trade. Managing the position after entry is the other half.

**When to exit before settlement:**

| Condition | Action |
|---|---|
| Edge evaporates (model update moves our prob toward market) | Exit at market if position is small; post maker exit if large |
| Edge reverses (our prob now < market price) | Exit immediately as taker — don't hope |
| Running high/low locks a bracket | If we own YES and it's locked, hold to settlement (free money). If we own NO, exit immediately |
| Drawdown limit hit on this event | Close all positions for this event |
| Regime change detected (unexpected precip, frontal timing shift) | Re-evaluate all positions; close any where new regime invalidates the thesis |

**When to add to a position:**
- Model update *increases* our edge AND new information supports the thesis
- Never add just because the price moved against you (that's averaging down on hope)
- Adding must pass the same Kelly/bankroll checks as a new entry

**Repricing resting orders:**
- After every model update: cancel all resting orders, recompute fair values, rebook
- After every significant observation (>1°F deviation from forecast curve): reprice
- Stale order timeout: cancel any unfilled order older than 30 minutes
- If partially filled, treat the filled portion as a position and the unfilled
  remainder as a fresh order decision

**Never do:**
- Hold a losing position hoping it comes back without a model-based reason
- Average down into a losing position without new information
- Let a resting order sit overnight without refreshing it against latest model data

### Liquidity Timing Patterns

Kalshi weather market liquidity follows predictable daily patterns:

| Time (EST) | Liquidity | Spreads | Notes |
|---|---|---|---|
| 12am - 6am | Very thin | Wide (5-15¢) | Overnight; few participants. Opportunity for patient makers |
| 6am - 8am | Building | Narrowing | Models updating; early movers repricing |
| 8am - 11am | **Peak** | **Tightest (2-5¢)** | Morning edge window. Best fills, most volume |
| 11am - 2pm | Good | Moderate | Obs arriving, distribution narrowing |
| 2pm - 5pm | Declining | Widening | High temp usually locked by now; less uncertainty |
| 5pm - 8pm | Low | Wide | Low temp not yet settled; some activity |
| 8pm - 12am | Very thin | Very wide | Low temp overnight; thin market |

**Implications:**
- Concentrate taker activity in 8am-11am window when spreads are tightest
- Use overnight hours for patient maker orders at favorable prices
- Don't try to exit large positions in thin markets — you'll move the price against yourself
- LOW temp brackets have a second liquidity window around sunset (6-8pm) when cooling begins

**Settlement-day dynamics:**
- HIGH brackets: liquidity spikes around 2-4pm when the high is nearly locked
- LOW brackets: liquidity spikes around 5-7am next morning as the CLI approaches
- After CLI publication (~5:30am EST): market goes to 0/100 fast. No point trading.

### No-Trade Conditions

Sometimes the best trade is no trade. Sit out entirely when:

**Data quality issues:**
- SSE stream down >10 minutes (stale observations)
- NWS API returning errors (no independent obs verification)
- Model data >6 hours stale (Open-Meteo outage)
- Kalshi WS disconnected (can't see orderbook)

**Market structure issues:**
- All brackets have spreads >10¢ (cost of entry/exit is too high)
- Total bracket prices sum to >1.08 (excessive vig — need 8%+ edge just to break even)
- Unusual volume spike in a bracket with no weather justification (possible insider/info asymmetry)
- Market maker appears to have pulled all quotes (liquidity vacuum)

**Weather regime issues:**
- Ensemble is bimodal with no clear favorite (coin flip — not a trade)
- Tropical system within 48h of station (models are notoriously bad at TC temp impacts)
- Record-breaking event in progress (outside calibration range — our uncertainty framework has no data here)
- Frontal passage timing uncertainty >6 hours (pre/post-frontal temp can differ 15°F+)

**Bot state issues:**
- Drawdown >15% from peak (halve size) or >25% (full stop)
- Cold start: first 14 days of operation, use 1/8 Kelly max
- EMOS coefficients just retrained with <30 days of data in new window
- Calibration quality score in bottom quartile of recent history

**The discipline to not trade** is as important as the algorithm that decides when
to trade. Every dollar not lost on a bad trade is a dollar available for a good one.

### Multi-Day Portfolio Management

At any point, we may have exposure across multiple settlement dates:
- T+0 HIGH (settling today)
- T+0 LOW (settling tomorrow morning)
- T+1 HIGH (settling tomorrow)
- T+1 LOW (settling day-after-tomorrow morning)
- T+2 and beyond (if Kalshi lists them)

**Correlation across days:**
- Consecutive-day temps are correlated (~0.6-0.8 autocorrelation for daily highs)
- A warm bias today likely persists tomorrow
- Same-day HIGH and LOW are moderately correlated (~0.4-0.5)
- This means: if you're long warm brackets for T+0 HIGH and T+1 HIGH, your actual
  risk is higher than the sum of individual position risks

**Portfolio rules:**
- Track **net directional exposure**: sum of (position × bracket_midpoint) across
  all events. If strongly net-warm across multiple days, you have concentrated risk.
- Maximum 30% of bankroll across all active settlement dates combined
- When adding T+1 positions, account for existing T+0 exposure — reduce T+1 sizing
  if both positions are directionally aligned
- Diversify across HIGH and LOW when possible (lower correlation than HIGH-HIGH or LOW-LOW)

**Rolling positions forward:**
- If you have a T+0 position that's near breakeven and the same edge exists in T+1,
  don't reflexively roll. The T+1 bracket might have different liquidity/pricing.
- Each day's position should be independently justified, not a carryover from yesterday.

### Edge Attribution & Performance Analysis

Track *where* your PnL comes from to improve over time:

**Attribution dimensions:**

| Dimension | Question It Answers |
|---|---|
| **By market type** | Are we better at HIGH or LOW? |
| **By bracket position** | Do we make money on center brackets (60-70¢) or edges (10-20¢)? |
| **By time of entry** | Morning entries vs afternoon entries — which produce better PnL? |
| **By signal source** | Which signals actually predict edge? Warming stall? Cloud bust? Model divergence? |
| **By order type** | Maker fills vs taker entries — which have better realized edge? |
| **By regime** | Stable high pressure days vs frontal days — where do we have skill? |
| **By model cycle** | Trades placed after 00Z GFS vs 12Z ECMWF — which model updates drive the best trades? |

**Key metrics to track per trade:**

```
- Entry time, price, side (YES/NO), quantity
- Signal(s) that triggered the trade
- Model probability at entry
- Market price at entry
- Model probability at settlement
- Settlement outcome (win/lose)
- Realized PnL (after fees)
- Realized edge = (settlement_value - entry_price) - fee
- Was this a maker or taker fill?
- Time between entry and settlement
```

**What to do with attribution data:**
- If HIGH trades consistently outperform LOW: allocate more capital to HIGH
- If maker fills have negative realized spread: widen quotes or reduce maker activity
- If morning trades outperform afternoon: reduce position limits after 12pm
- If a specific signal (e.g., warming stall) has high win rate: increase Kelly mult
  when that signal fires
- If a regime (e.g., tropical moisture) has poor performance: add it to no-trade list

**Monthly review cadence:**
- Compute cumulative PnL by each attribution dimension
- Identify top 3 sources of edge and top 3 sources of loss
- Adjust strategy parameters (Kelly mult, no-trade conditions, maker/taker split)
- Update EMOS training window if seasonal regime has shifted

---

## Key Design Principles

1. **HIGH and LOW are always scored independently** — never averaged or combined
2. **Freshness-aware ensembling** — 1 vote per model, weighted by accuracy
3. **Observation conditioning narrows the distribution** as the day progresses
4. **Penny contract protection** — never sell at <5c (KPHL lost thousands)
5. **Morning edge window** — largest model-market divergence before 11am
6. **CLI "FOR" date** — settlement uses the climate day, not the issue date
7. **Modularity** — each component is independently testable and replaceable
8. **Structural conservatism** — every uncertainty layer can only widen distributions or reduce size
9. **Earn your confidence** — Kelly fraction floats up only with demonstrated calibration quality
10. **Maker-first** — 4x cheaper fees, only cross the spread for time-sensitive signals

## Miami Station Metadata

| Field | Value |
|-------|-------|
| Station | KMIA |
| Latitude | 25.7959 |
| Longitude | -80.2870 |
| NWS Office | MFL |
| NWS Grid | 75,52 |
| Timezone | Etc/GMT+5 (EST) |
| UTC Offset | -5 hours |
| DSM Time | 11:00Z |
| Elevation | 2.4m |
| Regime | Gulf Subtropical |
| Kalshi High Series | KXHIGHMIA |
| Kalshi Low Series | KXLOWTMIA |
| CLI Location | MIA |

## Signal Catalog

| Signal | Description | Source |
|--------|-------------|--------|
| Station-specific warming curve | Desert stations warm 14-16F after 9am, coastal 0-2F | Observations |
| Warming stall detector | Temp plateaus during peak heating = model bust likely | 1-min obs |
| Running high as hard floor | Once observed high exceeds strike, bracket is locked | SSE new_high |
| Dew point floor for lows | In Miami, dew point constrains minimum temperature | Observations |
| Cloud-temp correlation | Cloud busts may predict temp busts | Cloud obs table |
| Diurnal curve tracking | Per-model accuracy at specific hours of day | Diurnal scores |
| Obs-conditioned distribution shift | Narrow ensemble as obs deviate from forecast | Observations + forecasts |
| Market YES bias | Market overprices YES by ~6.8 cents on average | Market snapshots |
