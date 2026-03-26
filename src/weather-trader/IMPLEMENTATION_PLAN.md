# Weather Trader — TSSM Implementation Plan
## Trading State Space Model — Multi-City Execution Engine

**Status**: Design complete, implementation beginning
**Purpose**: Accept weather signals from city DS3M bots, decide trades, execute on Kalshi
**Architecture**: Market Mamba-3 + MoE Policy + Discrete Action Head + Formulaic Risk Manager

---

## Design Principle

```
"The trader has ZERO weather knowledge. Cities are the weather brains.
 The trader is purely: signals + market state → edge → execute."
```

The Weather Trader receives **frozen DS3MSignal packets** from each city's weather bot.
It NEVER modifies or retrains the weather models. It learns:
- How the MARKET behaves (price patterns, spread dynamics, fill rates)
- WHEN to act on weather signals (timing, urgency, regime-aware execution)
- HOW MUCH to bet (sizing, portfolio correlation, drawdown management)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     CITY WEATHER BOTS                           │
│                                                                  │
│  KMIA DS3M ─┐                                                   │
│  KFLL DS3M ─┤──→ DS3MSignal packets (frozen, every 5 sec)      │
│  KTPA DS3M ─┤     • bracket_probs (6 floats per city)           │
│  KJAX DS3M ─┘     • regime_posterior, mamba_embedding            │
│  (future)         • filtered_temp, running_max/min, intervals    │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│                KALSHI MARKET DATA COLLECTOR                      │
│                                                                  │
│  WebSocket feed:                                                 │
│    • Order book deltas (bid/ask at each ¢ level)                │
│    • Trade executions (price, size, taker_side)                 │
│    • Market lifecycle events                                     │
│                                                                  │
│  Derived features (computed every 5 sec):                       │
│    • Mid-price per bracket                                       │
│    • Spread per bracket (ask - bid)                             │
│    • Volume rate (trades/min)                                    │
│    • Order flow imbalance (buy vol - sell vol) / total vol      │
│    • Price velocity (5-min price change)                        │
│    • Book depth asymmetry (bid depth - ask depth)               │
│    • Implied probability = mid_price / 100                      │
│                                                                  │
│  Stored in: trader.db → market_features table                   │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│              MARKET STATE ENCODER (Mamba-3)                      │
│                                                                  │
│  Input sequence (per bracket, per city):                        │
│    [market_features || ds3m_signal_features]                    │
│    Lookback: 120 steps (10 min at 5-sec, or 2h at 1-min)       │
│                                                                  │
│  Market feature vector (per bracket, ~20 dims):                 │
│    mid_price, spread, volume_rate, flow_imbalance,              │
│    price_velocity_5m, price_velocity_30m, price_velocity_1h,    │
│    book_depth_bid, book_depth_ask, book_asymmetry,              │
│    time_to_settlement_hours, hour_sin, hour_cos,                │
│    dow_sin, dow_cos, is_pre_market, is_post_nwp_release,       │
│    bracket_position (0=lowest...5=highest),                     │
│    bracket_width_f, bracket_midpoint_f                          │
│                                                                  │
│  DS3M signal features (per city, ~25 dims):                     │
│    bracket_prob (for this bracket), regime_posterior (5-6),      │
│    filtered_temp, uncertainty, running_max, running_min,         │
│    predicted_max, predicted_min, hours_to_settlement,            │
│    model_confidence, mamba_embedding_summary (PCA to 8 dims)    │
│                                                                  │
│  Architecture:                                                   │
│    Mamba-3, d_model=192, d_state=16 (MIMO), 4 layers           │
│    Complex-valued states (captures intraday oscillations)        │
│    FiLM conditioning from dominant regime latent                 │
│                                                                  │
│  Output: 192-dim market state embedding per bracket             │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│                  CROSS-BRACKET ATTENTION                         │
│                                                                  │
│  All 6 bracket embeddings attend to each other                  │
│  Learns: "if 85-90 is overpriced, 90-95 is likely underpriced" │
│  2-layer transformer with 4 heads                               │
│  Adds cross-bracket context to each embedding                   │
│                                                                  │
│  For multi-city: cross-city attention too                       │
│    "Miami 90-95 correlates with FLL 88-93"                      │
│                                                                  │
│  Output: context-enriched 192-dim embedding per bracket         │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│                  MoE POLICY LAYER                                │
│                                                                  │
│  Router: MLP(regime_latent + market_state → gate logits)        │
│  Top-2 routing with load balancing loss                         │
│                                                                  │
│  Expert 1: WEATHER EDGE EXPERT                                  │
│    Specializes in: DS3M prob vs market price discrepancies      │
│    Strongest when: regime is stable, DS3M confidence high       │
│    Typical action: buy underpriced brackets near settlement     │
│                                                                  │
│  Expert 2: TIMING EXPERT                                        │
│    Specializes in: intraday patterns, NWP model release windows │
│    Learns: "market is 4¢ slow to react after 18Z GFS release"  │
│    Typical action: front-run predictable market adjustments     │
│                                                                  │
│  Expert 3: ORDER FLOW EXPERT                                    │
│    Specializes in: maker/taker dynamics, spread exploitation    │
│    Learns: fill probability at each price level, optimal offset │
│    Typical action: post limit orders at spread midpoint         │
│                                                                  │
│  Expert 4: RISK/PORTFOLIO EXPERT                                │
│    Specializes in: position management, exit timing             │
│    Learns: when to cut losses, cross-bracket hedging            │
│    Typical action: exit losing positions before settlement      │
│                                                                  │
│  Expert 5: RESERVED (sentiment / news)                          │
│  Expert 6: RESERVED (cross-city arbitrage)                      │
│                                                                  │
│  Each expert: 2-layer MLP, 256 hidden, ReLU, → action logits   │
│  MoE output: weighted sum of top-2 expert outputs               │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│              DISCRETE ACTION HEAD                                │
│                                                                  │
│  Dueling DQN architecture:                                      │
│    State value:  V(s) = MLP(moe_output) → scalar               │
│    Advantage:    A(s,a) = MLP(moe_output) → |actions| logits    │
│    Q-value:      Q(s,a) = V(s) + A(s,a) - mean(A)             │
│                                                                  │
│  Action space per bracket (discrete):                           │
│    Action type: {BUY_YES, BUY_NO, HOLD, EXIT}  (4 choices)     │
│    Size bucket: {1, 2, 3, 5, 8, 13, 21} contracts (7 choices)  │
│    Route: {MAKER, TAKER}  (2 choices)                           │
│    Limit offset: {0, 1, 2, 3, 5}¢ from mid  (5 choices, maker) │
│                                                                  │
│  Factored action head (not full cross-product):                 │
│    action_type_logits = MLP_type(embedding) → 4                 │
│    size_logits = MLP_size(embedding) → 7                        │
│    route_logits = MLP_route(embedding) → 2                      │
│    offset_logits = MLP_offset(embedding) → 5                    │
│                                                                  │
│  During training: ε-greedy or Boltzmann exploration             │
│  During inference: argmax with temperature annealing            │
│                                                                  │
│  Alternative: Discrete SAC with categorical policy              │
│    (evaluate both, pick winner on backtest Sharpe)              │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│           HARD RISK MANAGER (formulaic, NOT learned)            │
│                                                                  │
│  *** THIS LAYER CANNOT BE OVERRIDDEN BY THE POLICY ***          │
│                                                                  │
│  Position limits:                                                │
│    • Max contracts per bracket: 25                               │
│    • Max contracts per city: 50                                  │
│    • Max total portfolio exposure: $500                          │
│    • Max open positions: 12                                      │
│                                                                  │
│  Loss limits:                                                    │
│    • Max daily loss: $100 → STOP trading for day                │
│    • Max weekly loss: $250 → STOP + alert                       │
│    • Max drawdown from peak: 30% → STOP + require manual reset  │
│    • Per-trade max loss: $50                                     │
│                                                                  │
│  Correlation checks:                                             │
│    • Adjacent brackets (85-90 + 90-95) count as correlated      │
│    • Same-direction bets across correlated brackets capped       │
│    • Cross-city same-regime bets recognized as correlated        │
│                                                                  │
│  Timing rules:                                                   │
│    • Min 60 sec between trades on same bracket                  │
│    • No new positions in last 30 min before settlement           │
│    • Max 30 trades per day per city                              │
│                                                                  │
│  Override rules:                                                 │
│    • Kelly ceiling: never bet more than 5% of bankroll           │
│    • Regime uncertainty: halve size if no regime > 50% posterior │
│    • Model disagreement: veto if DS3M and market differ by >40% │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│              EXECUTION ENGINE                                    │
│                                                                  │
│  Modes: shadow (log only), paper (simulated fills), live        │
│                                                                  │
│  Maker execution:                                                │
│    • Post limit order at mid - offset                           │
│    • Monitor fill via WebSocket                                  │
│    • Cancel + repost if unfilled after 2 min                    │
│    • Escalate to taker if edge is decaying                      │
│                                                                  │
│  Taker execution:                                                │
│    • Market order at best ask/bid                                │
│    • Immediate fill guaranteed                                   │
│    • Higher fees but captures time-sensitive edge                │
│                                                                  │
│  Paper trading simulation:                                       │
│    • Simulates fills using historical spread + volume data       │
│    • Maker orders: fill probability model from Kalshi data      │
│    • Taker orders: instant fill at ask + fee                    │
│    • Realistic slippage on larger orders                         │
│                                                                  │
│  Settlement:                                                     │
│    • Watches for NWS CLI release (typically next morning)       │
│    • Computes P&L per trade, per bracket, per city              │
│    • Updates ledger, performance metrics, regime attribution    │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│              LEDGER + PERFORMANCE TRACKING                       │
│                                                                  │
│  Per-trade: entry_time, exit_time, side, size, price, fees,     │
│             settlement_outcome, pnl, regime_at_entry,           │
│             expert_weights_at_entry, ds3m_confidence             │
│                                                                  │
│  Aggregated metrics:                                             │
│    • Daily/weekly/monthly P&L by city                           │
│    • Sharpe ratio (rolling 30-day)                              │
│    • Win rate by regime, by expert, by bracket position         │
│    • Maker vs taker performance comparison                      │
│    • Edge decay analysis (P&L vs hours-to-settlement)           │
│    • Expert utilization and performance attribution             │
│                                                                  │
│  Feeds back to:                                                  │
│    • Adaptive Kelly parameters                                   │
│    • Expert pruning/weighting                                    │
│    • Risk limit adjustments                                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Implementation Phases

### Phase 1: Kalshi Data Foundation
**Status**: IN PROGRESS (data pull running)
**Files**: `data/kalshi_puller.py`, `data/market_feature_builder.py`

1. **Pull all historical Kalshi weather trades** (2021-present)
   - Miami (KMIA) — primary
   - Fort Lauderdale, Tampa, Orlando, Jacksonville, Key West
   - All SE US cities available
   - Tick-by-tick trades + 1-min candles + market metadata

2. **Build market feature dataset**
   - For each 5-min window of each bracket's life:
     - Compute: mid_price, spread, volume_rate, flow_imbalance
     - Compute: price_velocity (5m, 30m, 1h), book depth (from candle proxy)
   - Join with DS3M backtest outputs (bracket_probs, regime, etc.)
   - Create training-ready replay buffer

3. **Build fill probability model**
   - From historical trades: P(fill | limit_price, spread, volume, time_to_settlement)
   - Simple logistic regression or XGBoost
   - Used for paper trading simulation accuracy

### Phase 2: Market Mamba-3 Encoder
**Files**: `model/market_mamba3.py`, `model/cross_bracket_attention.py`

1. **Market Mamba-3 implementation**
   - d_model=192, d_state=16 (MIMO), 4 layers
   - Input: market features (20d) + DS3M signal features (25d) = 45d per step
   - Sequence length: 120 steps (2h lookback at 1-min)
   - FiLM conditioning layer from regime posterior
   - Output: 192-dim embedding per bracket

2. **Cross-bracket attention**
   - 6 bracket embeddings → 2-layer transformer → context-enriched embeddings
   - Multi-city: stack all cities' brackets, add city position encoding
   - Learns cross-bracket and cross-city correlations

3. **Pure-PyTorch fallback for Mac inference**
   - Same approach as weather Mamba-3 fallback
   - Market Mamba is smaller (4 layers) → faster inference

### Phase 3: MoE Policy + Action Head
**Files**: `model/moe_policy.py`, `model/action_head.py`

1. **MoE layer implementation**
   - 4 experts initially (weather edge, timing, order flow, risk/portfolio)
   - 2 reserved slots
   - Top-2 routing with load balancing auxiliary loss
   - Each expert: 2-layer MLP, 256 hidden, ReLU

2. **Discrete action head**
   - Factored: separate heads for action_type, size, route, limit_offset
   - Dueling DQN: V(s) + A(s,a) - mean(A)
   - Also implement Discrete SAC variant for comparison

3. **Reward function**
   ```python
   reward = (
       pnl_after_fees                           # Raw profit
       - lambda_drawdown * drawdown_penalty      # Penalize drawdowns
       - lambda_cost * transaction_cost           # Penalize excessive trading
       + lambda_sharpe * rolling_sharpe_bonus     # Reward consistency
   )
   ```

### Phase 4: Training Pipeline
**Files**: `training/replay_buffer.py`, `training/offline_rl.py`, `training/trainer.py`

1. **Replay buffer construction**
   - From historical Kalshi data + DS3M backtest outputs
   - State: (market_features, ds3m_signals, portfolio_state, time_features)
   - Action: (what was traded historically — or optimal action from hindsight)
   - Reward: (settlement P&L - fees)
   - Next_state: (market features at next timestep)

2. **Offline RL training (Conservative Q-Learning)**
   - CQL penalizes Q-values for out-of-distribution actions
   - Prevents overestimation on actions the historical data didn't take
   - Trains on years of replay data without needing a simulator

3. **Behavioral cloning warmup**
   - First 5 epochs: imitate the existing rule-based trader_policy.py
   - Gives the MoE + action head a reasonable starting policy
   - Then switch to CQL for refinement

4. **Backtest validation**
   - Walk-forward: train on months 1-6, test on month 7, retrain on 1-7, test on 8, ...
   - Metrics: Sharpe, max drawdown, win rate, P&L per contract
   - Compare against: rule-based baseline, random baseline, market-only baseline

### Phase 5: Live Integration
**Files**: `integration/signal_receiver.py`, `integration/kalshi_ws.py`, `runner.py`

1. **Signal receiver**
   - Listens for DS3MSignal packets from each city bot
   - Via SQLite shared tables, Unix sockets, or ZeroMQ
   - Buffers into market state encoder's input sequence

2. **Kalshi WebSocket collector**
   - Real-time order book deltas + trade feed
   - Computes market features every 5 seconds
   - Stores to trader.db for future training

3. **Inference loop**
   ```
   Every 5 seconds:
     1. Receive DS3M signals from all connected cities
     2. Update market feature buffer from Kalshi WS
     3. Forward pass: Market Mamba → Cross-bracket Attn → MoE → Action Head
     4. Filter through Risk Manager
     5. Execute (shadow/paper/live)
     6. Log to ledger
   ```

4. **Warm start**
   - Load pre-trained weights
   - First 2 weeks: shadow mode (log decisions, don't execute)
   - Next 2 weeks: paper mode (simulated execution with fill model)
   - Then: live mode with minimum sizing
   - Gradually increase sizing as performance validates

### Phase 6: RSSM World Model (Future)
**Files**: `model/world_model.py`, `training/dreamer.py`

1. **DreamerV3-style RSSM**
   - Reuses Market Mamba-3 as the dynamics model
   - Adds: representation model, observation model, reward model
   - "Imagination": simulate 4-8h of future market dynamics
   - Policy optimization on imagined rollouts

2. **When to build**:
   - After 3-6 months of live trading data
   - When the offline RL policy has been validated
   - The Market Mamba weights transfer directly into the RSSM

---

## File Structure

```
src/weather-trader/
├── IMPLEMENTATION_PLAN.md       # This file
├── __init__.py
├── config.py                    # TraderConfig, CityConfig, etc.
├── runner.py                    # Main async loop
│
├── data/                        # Data collection + processing
│   ├── kalshi_puller.py         # Historical trade data from Kalshi API
│   ├── market_feature_builder.py # Compute market features from raw data
│   ├── replay_buffer.py         # Training replay buffer construction
│   └── backtest_aligner.py      # Align DS3M backtest with Kalshi prices
│
├── model/                       # Neural network components
│   ├── market_mamba3.py         # Market State Encoder (Mamba-3)
│   ├── market_mamba3_cpu.py     # Pure-PyTorch Mac fallback
│   ├── cross_bracket_attn.py   # Cross-bracket + cross-city attention
│   ├── moe_policy.py           # MoE policy layer (4-6 experts)
│   ├── action_head.py          # Dueling DQN / Discrete SAC heads
│   └── film_conditioning.py    # FiLM regime conditioning
│
├── training/                    # Training infrastructure
│   ├── offline_rl.py           # Conservative Q-Learning trainer
│   ├── behavioral_cloning.py   # Warmup from rule-based policy
│   ├── backtest.py             # Walk-forward backtesting
│   └── train.py                # CLI training runner
│
├── execution/                   # Order execution (existing, enhanced)
│   ├── __init__.py
│   ├── bracket_publisher.py    # Bracket discovery
│   ├── kalshi_client.py        # REST + WS Kalshi client
│   ├── kalshi_ws_collector.py  # NEW: Real-time market data collector
│   ├── portfolio.py            # Position tracking
│   ├── risk.py                 # Hard risk manager (NOT learned)
│   ├── trader.py               # Trade evaluation + execution
│   └── trader_policy.py        # Legacy rule-based policy (baseline)
│
├── ledger/                      # Trade recording + performance
│   ├── __init__.py
│   ├── db.py                   # SQLite schema
│   ├── ledger.py               # Trade logging
│   └── performance.py          # NEW: Sharpe, regime attribution, expert analysis
│
└── integration/                 # Multi-city signal integration
    ├── signal_receiver.py      # Receives DS3MSignal from city bots
    ├── signal_protocol.py      # DS3MSignal dataclass + serialization
    └── city_registry.py        # Tracks connected cities + health
```

---

## Multi-City Scaling Plan

```
Phase A (Now):     KMIA only → single city, 6 brackets
Phase B (Month 2): KFLL, KTPA → 3 cities, 18 brackets
Phase C (Month 4): KJAX, KORL, KEYW → 6 cities, 36 brackets
Phase D (Month 6): NYC, CHI, ATL → 9 cities, 54 brackets
                   + cross-city attention + portfolio optimization

Each new city requires:
  1. Deploy a DS3M weather bot (collector + model + orchestrator)
  2. Train city-specific GraphMamba-3 (can warm-start from KMIA weights)
  3. Register with Weather Trader signal receiver
  4. Weather Trader auto-discovers new bracket markets
  5. MoE policy generalizes to new city (shared experts)
  6. City-specific fine-tuning after 1 week of data
```

---

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Discrete vs continuous actions | **Discrete** | Kalshi has discrete price grid (1¢), discrete contracts. Scales to 54 brackets. |
| DQN vs SAC | **Both, pick winner** | Implement Dueling DQN and Discrete SAC, backtest both, deploy winner |
| MoE vs FiLM for regime | **MoE** | With 4+ years Kalshi data, enough per-regime. MoE experts specialize deeply. |
| Online vs offline RL | **Offline first** | CQL on historical data. Online fine-tune after paper trading validates. |
| Risk: learned vs formulaic | **Formulaic (hard)** | NEVER let the model override max loss. This is a business rule, not a prediction. |
| Multi-city architecture | **Shared model, city embeddings** | One Market Mamba serves all cities. City-specific behavior via learned embeddings. |
| Signal transport | **SQLite shared tables** | Simple, reliable, same-machine. Upgrade to ZeroMQ for multi-machine. |

---

## Dependencies

```toml
[project.optional-dependencies]
trader = [
    "torch>=2.4",
    "torchdiffeq>=0.2.4",
    "websockets>=12.0",
    "aiohttp>=3.10",
]
trader-cuda = [
    "mamba-ssm>=2.3.1",
]
```

---

## Success Metrics

| Metric | Target (Paper) | Target (Live) |
|--------|----------------|---------------|
| Sharpe ratio (30-day rolling) | > 1.5 | > 1.0 |
| Win rate | > 55% | > 52% |
| Max drawdown | < 15% | < 10% |
| Avg edge per trade (¢) | > 3¢ | > 2¢ |
| Daily trades (KMIA only) | 5-15 | 3-10 |
| Maker fill rate | > 40% | > 35% |
| Expert utilization balance | No expert < 10% | No expert < 10% |
