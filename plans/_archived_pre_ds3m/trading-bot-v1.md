# Trading Bot v1 — Rules-Based Weather Market Trader

## Philosophy
No AI in the loop. Pure Python, pure math, runs 24/7. The edge is **data speed** — we have 5-minute obs, 15-minute FAWN, and real-time nearby stations. The market prices off 6-12 hour model cycles. We fill the gap between model runs with deterministic signals.

Planning companions:
- `plans/repo-blueprint.md` — canonical cleanup/refactor blueprint
- `plans/bot-architecture.mmd` — Mermaid architecture for the modular path

This file describes the trading-oriented rules path.
It should stay consistent with the repo blueprint and the residual-signal architecture.

Self-adjusting through feedback loops, not neural nets. Every parameter tunes itself from settlement history. No hardcoded magic numbers that rot.

---

## Architecture

```
runner.py (already running — 14 collection loops)
    ↓ writes to SQLite
    ↓
signal_engine.py (NEW — reads DB, outputs SignalState every 5 min)
    ↓
estimator.py (NEW — converts signals into probability distribution)
    ↓
trader.py (NEW — compares estimated probs to market, executes)
    ↓
ledger.py (NEW — tracks P/L, adjusts parameters from outcomes)
```

### Data Flow
```
                    ┌─────────────────────────────────┐
                    │         miami_collector.db       │
                    │  (obs, forecasts, markets, etc)  │
                    └──────┬──────────────────────────┘
                           │ read every 5 min
                    ┌──────▼──────────────────────────┐
                    │        signal_engine.py          │
                    │  Extracts 7 core signals from DB │
                    │  Outputs: SignalState dataclass   │
                    └──────┬──────────────────────────┘
                           │
                    ┌──────▼──────────────────────────┐
                    │         estimator.py             │
                    │  Builds P(high), P(low) distro   │
                    │  Normal(μ=consensus+adj, σ=ens)  │
                    │  Computes P(bracket) for each    │
                    └──────┬──────────────────────────┘
                           │
                    ┌──────▼──────────────────────────┐
                    │          trader.py               │
                    │  Compares P(bracket) to mkt price│
                    │  Edge > threshold → place order  │
                    │  Kelly fraction for sizing       │
                    └──────┬──────────────────────────┘
                           │
                    ┌──────▼──────────────────────────┐
                    │          ledger.py               │
                    │  Tracks every trade + outcome    │
                    │  Feeds back into parameter tuning│
                    └─────────────────────────────────┘
```

---

## Module 1: signal_engine.py

Runs every 5 minutes. Reads DB. Outputs a `SignalState` — a flat struct of numbers. No interpretation, no decisions.

### SignalState Dataclass
```python
@dataclass
class SignalState:
    timestamp_utc: str
    target_date: str
    market_type: str           # "high" or "low"
    hours_remaining: float     # hours until climate day end (05:00Z)

    # Signal 1: Model consensus (bias-corrected)
    consensus_f: float         # skill-weighted, bias-corrected mean
    consensus_sigma: float     # ensemble spread (std dev)
    raw_consensus_f: float     # unajusted mean (for comparison)
    bias_correction_f: float   # how much we shifted (consensus - raw)
    n_models: int

    # Signal 2: Obs divergence
    obs_current_f: float | None
    obs_trend_2hr: float | None    # °F change over last 2 hours
    obs_vs_consensus: float | None # obs_current - consensus
    projected_extreme_f: float | None  # linear extrapolation

    # Signal 3: CAPE + PW outflow risk
    cape_current: float | None
    pw_mm: float | None
    outflow_risk: bool         # cape > threshold AND pw > threshold
    cape_trend_1hr: float | None  # J/kg/hr change

    # Signal 4: Wind direction
    wind_dir_deg: float | None
    continental_intrusion: bool  # wind in [250, 360] range
    wind_shift_detected: bool    # direction changed >90° in 2 hours

    # Signal 5: Dew point floor
    dew_point_f: float | None
    evening_dew_mean_f: float | None
    estimated_low_floor_f: float | None  # dew + buffer
    dew_crash_active: bool       # >3°F drop in 30 min

    # Signal 6: Nearby station lead
    fawn_temp_f: float | None
    fawn_crash_detected: bool    # >3°F drop in 15 min at FAWN
    fawn_lead_minutes: float | None
    nearby_divergence_f: float | None  # max |delta| vs KMIA
    nearby_crash_count: int      # how many stations crashing

    # Signal 7: Pressure
    pressure_hpa: float | None
    pressure_3hr_trend: float | None  # hPa change
    pressure_surge: bool         # >1.0 hPa rise in 1 hour
    pressure_trough: bool        # at or near day minimum

    # Composite
    active_flags: list[str]      # list of triggered signal names
```

### How Each Signal is Computed

**Signal 1 — Bias-Corrected Consensus:**
Already exists in `adjusted_forecasts` table. The scoring pipeline computes per-model MAE and bias over a 15-day rolling window, then builds a skill-weighted consensus with time-decay.

Self-adjusting: Every CLI settlement updates model scores → bias corrections shift automatically. A model that was cold-biased last week but corrects itself gets reweighted within days.

```python
# Already in scoring pipeline:
weight = 1 / max(mae, MAE_FLOOR) ** WEIGHT_POWER
time_weight = 0.5 ** (run_age_hours / RUN_AGE_HALFLIFE_HOURS)
# consensus = Σ(forecast * weight * time_weight) / Σ(weight * time_weight)
```

**Signal 2 — Obs Divergence + Trajectory:**
```python
obs_current = latest obs temperature_f from observations table
obs_2hr_ago = obs from 2 hours prior
obs_trend_2hr = (obs_current - obs_2hr_ago) / 2  # °F/hr

# For HIGH markets only, during heating hours (10Z-21Z):
if market_type == "high" and in_heating_window:
    hours_of_sun_left = solar_decline_hour - current_hour
    projected_max = obs_current + obs_trend_2hr * hours_of_sun_left * decay
    # decay = 0.7 — heating rate slows as you approach peak
```

Self-adjusting: `decay` factor calibrated from historical obs trajectories vs actual highs. After each settlement, we log (projected_max, actual_cli) and refit decay.

**Signal 3 — CAPE + PW Threshold:**
```python
cape = latest from atmospheric_data (gfs_seamless)
pw = latest precipitable_water_mm

outflow_risk = (cape > cape_threshold) and (pw > pw_threshold)
```

Self-adjusting: `cape_threshold` and `pw_threshold` start at 2500 / 40mm (from Mar 15-16 data). After each day, we log whether outflow actually occurred (detected by temp crash >5°F in 1 hour). Over time:
- If outflow_risk=True and crash happened → threshold confirmed
- If outflow_risk=True and no crash → threshold too sensitive, raise it
- If outflow_risk=False and crash happened → threshold too high, lower it
- Uses simple logistic regression on (cape, pw) → did_crash? to find optimal boundary

**Signal 4 — Wind Direction:**
```python
wind_dir = latest obs wind_heading_deg (circular mean of last 30 min)
continental = 250 <= wind_dir <= 360  # W/NW/N
# Also check for rapid shift:
wind_2hr_ago = circular mean from 2 hours prior
angular_delta = circular_difference(wind_dir, wind_2hr_ago)
wind_shift = abs(angular_delta) > 90
```

Self-adjusting: The 250-360° continental range is physics-based (SE Florida geography) and doesn't need tuning. But we track: when continental_intrusion=True, how much did the low drop vs dew point? This refines the `dew_buffer` in Signal 5.

**Signal 5 — Dew Point Floor:**
```python
# Evening dew point mean (18:00-00:00 local = 23:00-05:00Z)
evening_dew = mean of dew_point_f from obs in evening window

# Estimated low floor
dew_buffer = rolling_mean(cli_low - evening_dew, last_N_days)
estimated_floor = evening_dew + dew_buffer
```

Self-adjusting: `dew_buffer` is literally just the rolling average of (CLI low - evening dew mean) over the last N settlement days. Mar 15: 70 - 68.3 = 1.7. Mar 16: 69 - 67.6 = 1.4. Current buffer ≈ 1.5°F. As more days accumulate, this converges to the local climate relationship. Separate buffers for outflow days vs calm days.

**Signal 6 — Nearby Station Lead:**
```python
# FAWN Homestead (closest inland agricultural station)
fawn_now = latest fawn_observations air_temp_f
fawn_15min_ago = fawn 15 min prior
fawn_crash = (fawn_15min_ago - fawn_now) > 3.0

# Nearby ASOS stations
for station in closest_5_asos:
    delta = station_temp - kmia_temp
    if abs(delta) > divergence_threshold:
        flag it
```

Self-adjusting: `fawn_lead_minutes` is computed retrospectively — how many minutes before KMIA's crash did FAWN crash? This builds a distribution over time. Median lead time becomes the expected warning window.

**Signal 7 — Pressure:**
```python
pressure_now = latest obs pressure_hpa
pressure_3hr = obs pressure from 3 hours ago
pressure_3hr_trend = pressure_now - pressure_3hr
pressure_surge = (1hr delta > surge_threshold)
pressure_trough = pressure_now <= min(all_pressures_today) + 0.5
```

Self-adjusting: `surge_threshold` starts at 1.0 hPa (from Mar 15 data: 3.1 hPa surge confirmed outflow). After each day, we log (surge_magnitude, did_crash?) and use the same logistic approach as CAPE/PW.

---

## Module 2: estimator.py

Takes a `SignalState`, outputs a probability distribution over brackets.

### Core Logic
```python
def estimate_distribution(state: SignalState) -> BracketDistribution:
    # Start with bias-corrected model consensus
    mu = state.consensus_f
    sigma = max(state.consensus_sigma, SIGMA_FLOOR)

    # Apply signal adjustments (additive shifts to mu, multiplicative to sigma)
    adjustments = []

    # Obs divergence: if obs running hot/cold, shift mu
    if state.obs_vs_consensus is not None and state.hours_remaining > 2:
        obs_weight = obs_divergence_weight(state.hours_remaining)
        mu += state.obs_vs_consensus * obs_weight
        adjustments.append(f"obs_shift={state.obs_vs_consensus * obs_weight:+.1f}")

    # Trajectory extrapolation: use projected extreme
    if state.projected_extreme_f is not None:
        traj_weight = trajectory_weight(state.hours_remaining)
        mu = mu * (1 - traj_weight) + state.projected_extreme_f * traj_weight
        adjustments.append(f"traj_target={state.projected_extreme_f:.1f}")

    # Outflow risk: widen sigma for LOW markets
    if state.outflow_risk and state.market_type == "low":
        sigma *= outflow_sigma_multiplier  # e.g., 1.5x wider
        adjustments.append("outflow_risk_widen")

    # Continental intrusion: shift LOW mu down
    if state.continental_intrusion and state.market_type == "low":
        mu -= continental_shift_f  # e.g., -3°F
        adjustments.append(f"continental_shift=-{continental_shift_f}")

    # Dew point floor: hard lower bound for LOW
    if state.market_type == "low" and state.estimated_low_floor_f:
        # Truncate distribution below the floor
        floor = state.estimated_low_floor_f
        adjustments.append(f"floor={floor:.1f}")

    # Nearby crash: if FAWN crashing, shift LOW down immediately
    if state.fawn_crash_detected and state.market_type == "low":
        mu -= fawn_crash_shift_f
        adjustments.append("fawn_crash")

    # Convert Normal(mu, sigma) to per-bracket probabilities
    brackets = get_active_brackets(state.target_date, state.market_type)
    probs = {}
    for b in brackets:
        p = normal_cdf(b.cap, mu, sigma) - normal_cdf(b.floor, mu, sigma)
        probs[b.ticker] = p

    return BracketDistribution(mu=mu, sigma=sigma, probs=probs, adjustments=adjustments)
```

### Self-Adjusting Parameters

Every parameter in estimator.py is tuned from settlement outcomes:

| Parameter | What it controls | How it self-adjusts |
|---|---|---|
| `obs_divergence_weight` | How much obs shift the mean | Regression: obs_divergence at time T → final settlement error. Minimize MAE. |
| `trajectory_weight` | Blend between consensus and extrapolation | Same: track projected_extreme accuracy vs consensus accuracy per hours_remaining |
| `outflow_sigma_multiplier` | How much outflow risk widens uncertainty | Track: when outflow_risk=True, what was the actual spread of outcomes? Calibrate σ so that 68% of outcomes fall within ±1σ |
| `continental_shift_f` | How far NW wind drops the low | Rolling mean of (pre-wind-shift estimate - CLI low) on continental days |
| `fawn_crash_shift_f` | How much FAWN crash predicts KMIA drop | Rolling mean of (FAWN crash magnitude → KMIA final crash magnitude) |
| `SIGMA_FLOOR` | Minimum uncertainty | Set so that the worst forecast miss in the last 15 days is within 2σ |

**Calibration Loop** (runs daily after CLI settlement):
```python
def calibrate(ledger: Ledger, lookback_days: int = 15):
    """Refit all estimator parameters from recent settlements."""
    for date in recent_settlement_dates(lookback_days):
        cli_high = get_settlement(date, "high")
        cli_low = get_settlement(date, "low")

        # For every SignalState snapshot we logged that day:
        for snapshot in get_signal_snapshots(date):
            # Compare our estimate at that time to the actual CLI
            actual = cli_high if snapshot.market_type == "high" else cli_low
            predicted_mu = snapshot.estimated_mu
            error = actual - predicted_mu

            # Log: (hours_remaining, obs_divergence, error)
            # This feeds into the parameter regressions
            calibration_log.append({
                "hours_remaining": snapshot.hours_remaining,
                "obs_divergence": snapshot.obs_vs_consensus,
                "cape": snapshot.cape_current,
                "outflow_risk": snapshot.outflow_risk,
                "continental": snapshot.continental_intrusion,
                "predicted_mu": predicted_mu,
                "actual": actual,
                "error": error,
            })

    # Refit parameters (simple linear/logistic regressions)
    refit_obs_weight(calibration_log)
    refit_trajectory_weight(calibration_log)
    refit_outflow_multiplier(calibration_log)
    refit_continental_shift(calibration_log)
    refit_sigma_floor(calibration_log)
```

---

## Module 3: trader.py

Compares our probability estimates to market prices. Places orders when edge exceeds threshold.

### Core Logic
```python
def evaluate_trades(dist: BracketDistribution, market: list[BracketPrice]) -> list[TradeSignal]:
    trades = []
    for bracket in market:
        our_prob = dist.probs.get(bracket.ticker, 0)
        market_yes = bracket.ask / 100  # convert cents to dollars
        market_no = (100 - bracket.bid) / 100

        # YES edge: our probability minus what we'd pay
        yes_edge = our_prob - market_yes
        # NO edge: (1 - our probability) minus what we'd pay for NO
        no_edge = (1 - our_prob) - market_no

        if yes_edge > EDGE_THRESHOLD:
            size = kelly_size(our_prob, market_yes, bankroll, KELLY_FRACTION)
            trades.append(TradeSignal("YES", bracket.ticker, market_yes, our_prob, yes_edge, size))

        elif no_edge > EDGE_THRESHOLD:
            size = kelly_size(1 - our_prob, market_no, bankroll, KELLY_FRACTION)
            trades.append(TradeSignal("NO", bracket.ticker, market_no, 1 - our_prob, no_edge, size))

    return trades
```

### Kelly Sizing
```python
def kelly_size(prob: float, price: float, bankroll: float, fraction: float = 0.25) -> int:
    """Conservative fractional Kelly criterion."""
    # Kelly: f* = (p * b - q) / b, where b = (1/price - 1), q = 1-p
    b = (1 / price) - 1
    q = 1 - prob
    f_star = (prob * b - q) / b
    f_star = max(f_star, 0)

    # Fraction Kelly (0.25 = quarter Kelly, much safer)
    position_dollars = bankroll * f_star * fraction

    # Convert to contracts (each contract = price in cents)
    contracts = int(position_dollars / price)

    # Hard caps
    contracts = min(contracts, MAX_CONTRACTS_PER_TRADE)
    contracts = max(contracts, 0)
    return contracts
```

### Self-Adjusting Parameters

| Parameter | Start | Self-adjusts how |
|---|---|---|
| `EDGE_THRESHOLD` | 0.15 (15%) | If win rate < 55% over last 30 trades, raise threshold by 0.02. If win rate > 70%, lower by 0.01. Floor at 0.08, cap at 0.25. |
| `KELLY_FRACTION` | 0.25 (quarter Kelly) | If max drawdown in last 30 days exceeds 15% of bankroll, reduce to 0.15. If drawdown < 5%, allow increase to 0.30. |
| `MAX_CONTRACTS_PER_TRADE` | 50 | Scales with bankroll. = bankroll / 100, capped at 200. |

---

## Module 4: ledger.py

Tracks every signal snapshot, every trade, every outcome. This is the memory.

### Tables (new, in miami_collector.db)

```sql
-- Every 5-minute signal snapshot
CREATE TABLE signal_snapshots (
    id INTEGER PRIMARY KEY,
    timestamp_utc TEXT NOT NULL,
    target_date TEXT NOT NULL,
    market_type TEXT NOT NULL,
    hours_remaining REAL,
    consensus_f REAL,
    consensus_sigma REAL,
    obs_current_f REAL,
    obs_trend_2hr REAL,
    projected_extreme_f REAL,
    estimated_mu REAL,          -- final estimate after adjustments
    estimated_sigma REAL,
    cape REAL,
    pw_mm REAL,
    outflow_risk INTEGER,       -- boolean
    continental INTEGER,
    dew_point_f REAL,
    estimated_floor_f REAL,
    fawn_crash INTEGER,
    pressure_trend REAL,
    active_flags TEXT,          -- JSON list
    created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

-- Every trade placed
CREATE TABLE trades (
    id INTEGER PRIMARY KEY,
    timestamp_utc TEXT NOT NULL,
    target_date TEXT NOT NULL,
    market_type TEXT NOT NULL,
    ticker TEXT NOT NULL,
    side TEXT NOT NULL,          -- "YES" or "NO"
    price_cents INTEGER,
    contracts INTEGER,
    our_probability REAL,
    edge REAL,
    signal_snapshot_id INTEGER REFERENCES signal_snapshots(id),
    kalshi_order_id TEXT,
    status TEXT DEFAULT 'pending',  -- pending, filled, cancelled
    fill_price_cents INTEGER,
    created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

-- Settlement outcomes for calibration
CREATE TABLE trade_outcomes (
    id INTEGER PRIMARY KEY,
    trade_id INTEGER REFERENCES trades(id),
    cli_value_f REAL,
    winning_side TEXT,          -- "YES" or "NO"
    pnl_cents INTEGER,         -- per contract
    total_pnl_cents INTEGER,   -- pnl * contracts
    created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

-- Parameter history (for debugging/auditing tuning)
CREATE TABLE parameter_history (
    id INTEGER PRIMARY KEY,
    parameter_name TEXT NOT NULL,
    old_value REAL,
    new_value REAL,
    reason TEXT,
    calibration_date TEXT,
    created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
```

### Self-Adjusting Feedback Loops

```
Daily (after CLI settlement):
  1. Score all trades from yesterday → trade_outcomes
  2. Run calibration on signal_snapshots + trade_outcomes
  3. Update estimator parameters
  4. Log parameter changes to parameter_history

Weekly:
  5. Compute rolling Sharpe, win rate, max drawdown
  6. Adjust EDGE_THRESHOLD and KELLY_FRACTION
  7. Flag any signal that hasn't contributed to a winning trade in 2 weeks
```

---

## Rollout Phases

### Phase 1: Shadow Mode (Week 1-2)
- signal_engine.py runs every 5 min, writes to signal_snapshots
- estimator.py computes probabilities, logs them
- **No trades placed**
- Compare our estimates to actual settlements
- Tune initial parameters

### Phase 2: Paper Trading (Week 3-4)
- trader.py runs but logs "would-have-traded" to trades table with status='paper'
- After each settlement, compute paper P/L
- Verify edge is positive over 20+ paper trades
- Calibration loop running daily

### Phase 3: Live Small (Week 5+)
- trader.py places real orders via Kalshi API
- MAX_CONTRACTS = 10, KELLY_FRACTION = 0.15
- Human reviews every trade for first week
- Automatic after proven stable

### Phase 4: Scale
- Increase limits based on Sharpe and drawdown metrics
- Add more stations (KFLL, KTPA, etc.)
- Multi-market (high + low simultaneously)

---

## What Makes This Self-Adjusting (Not ML, But Adaptive)

The system uses **online learning** principles without neural nets:

1. **Rolling bias correction**: Model bias updates daily with each CLI settlement. A model that was 3°F cold last week but 1°F cold this week sees its correction shrink automatically.

2. **Parameter regression**: `obs_divergence_weight`, `continental_shift`, etc. are refit daily from a 15-day rolling window of (signal_state, actual_outcome) pairs. This is linear regression, not deep learning.

3. **Threshold optimization**: CAPE/PW/pressure thresholds are tuned via logistic regression on (atmospheric_state → did_crash_happen?). As we accumulate more crash/no-crash days, the decision boundary sharpens.

4. **Confidence calibration**: Every day we check: when we said P(bracket)=0.60, did it win ~60% of the time? If we're overconfident, σ increases. Underconfident, σ decreases. This is Platt scaling.

5. **Position sizing feedback**: Kelly fraction and edge threshold auto-adjust from drawdown and win-rate metrics. Bad streaks → smaller bets. Good streaks → slightly larger (with hard caps).

6. **Signal deprecation**: If a signal hasn't contributed to edge in 2+ weeks (tracked by logging which signals were active on winning vs losing trades), it gets flagged for review. Dead signals get downweighted automatically.

The key difference from ML: every parameter has a **physical interpretation** and adjusts on a **known axis**. The bias correction adjusts °F. The CAPE threshold adjusts J/kg. Nothing is a black box. When something goes wrong, you can read the parameter_history table and trace exactly what changed and why.

---

## Files to Create

```
miami-project/
├── src/
│   ├── trading/
│   │   ├── __init__.py
│   │   ├── signal_engine.py    # SignalState extraction (5-min loop)
│   │   ├── estimator.py        # Probability distribution builder
│   │   ├── trader.py           # Edge detection + order placement
│   │   ├── ledger.py           # P/L tracking + calibration
│   │   ├── calibrator.py       # Daily parameter tuning
│   │   └── config.py           # All tunable parameters with defaults
│   └── collector/              # (existing — no changes needed)
├── plans/
│   └── trading-bot-v1.md       # This document
└── summaries/                  # (existing — daily analysis)
```

## Dependencies on Existing Code
- **signals.py**: SignalExtractor already computes most of what signal_engine.py needs. We either call into it or extract the SQL queries.
- **runner.py**: Already collecting all data we need. No changes.
- **schema.py**: Add 4 new tables (signal_snapshots, trades, trade_outcomes, parameter_history).
- **model_scorer.py**: Already computing bias-corrected consensus. signal_engine.py reads from adjusted_forecasts.
- **Kalshi API**: Need authenticated REST client for order placement. Existing WS connection handles market data.

## Open Questions
1. **Kalshi API auth**: Do we have API keys for order placement, or only market data?
2. **Bankroll**: Starting capital determines position sizing. Kelly fraction meaningless without this.
3. **Latency tolerance**: Is 5-minute signal refresh fast enough, or do we need sub-minute for volatile brackets?
4. **Multi-bracket positions**: Should we ever hold YES on two adjacent brackets (hedging), or always single-bracket?
