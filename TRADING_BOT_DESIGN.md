# Trading Center Bot — Design Document

**Status:** Brainstorm / Pre-implementation
**Date:** 2026-03-25
**Context:** DS3M provides bracket probabilities. This document designs the intelligence layer that converts those probabilities into optimal trade execution.

---

## The Problem

DS3M tells us P(bracket) with calibrated confidence. But that's only half the battle. The trading center needs to decide:
- **When** to enter (timing within the day)
- **When** to exit (before settlement if edge evaporates)
- **How much** to size (beyond simple Kelly)
- **Where** to route (maker vs taker, which bracket)
- **Whether** to trade at all (regime-aware abstention)

Each of these has its own signal landscape, and the current paper_trader_v2 handles them with static rules. We can do much better.

---

## Architecture: Three-Layer Trading Intelligence

```
┌───────────────────────────────────────────────────────────────────┐
│  LAYER 1: MARKET MICROSTRUCTURE MODEL                             │
│                                                                    │
│  Inputs: market_snapshots (bid/ask/vol/depth/spread over time)    │
│                                                                    │
│  Learns:                                                           │
│  ├── Intraday price trajectory patterns                            │
│  ├── Liquidity cycles (when spreads are tightest)                 │
│  ├── Volume-weighted price momentum                                │
│  ├── Orderbook imbalance → short-term direction                   │
│  └── Market maker behavior (who's providing liquidity, when)      │
│                                                                    │
│  Outputs:                                                          │
│  ├── optimal_entry_window: time range with best fill probability  │
│  ├── spread_forecast: expected bid-ask spread next N minutes      │
│  ├── liquidity_score: 0-1 tradability rating                      │
│  └── market_momentum: directional pressure (market moving toward  │
│      or away from our model price)                                 │
└───────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────────────┐
│  LAYER 2: ENTRY/EXIT SIGNAL ENGINE                                │
│                                                                    │
│  Combines DS3M output + market microstructure + temporal signals  │
│                                                                    │
│  ENTRY SIGNALS:                                                    │
│  ├── Morning model consensus shift (06-09 UTC)                    │
│  │   First NWP runs of the day often move brackets 3-5¢.         │
│  │   If DS3M edge increases after 12Z models → strong entry.     │
│  │                                                                 │
│  ├── Early morning price bias                                      │
│  │   Kalshi brackets systematically underprice tails at 6am EST   │
│  │   because retail traders haven't woken up yet. Thin liquidity  │
│  │   = wider spreads but bigger mispricings.                       │
│  │   → Enter tail brackets early if DS3M shows edge > 8¢          │
│  │                                                                 │
│  ├── Sea breeze timing signal                                      │
│  │   DS3M regime posterior shifts to sea_breeze → market hasn't   │
│  │   repriced yet. Typically 18-20 UTC (1-3pm EST).               │
│  │   → Fade high brackets when sea breeze regime activates        │
│  │                                                                 │
│  ├── NWP convergence signal                                        │
│  │   When model_spread (feature[10]) drops below 2°F, all models │
│  │   agree → high conviction entry on center brackets              │
│  │                                                                 │
│  ├── Innovation monitor signal                                     │
│  │   Bracket KF bank detects regime change before HDP.            │
│  │   → Pause new entries until regime stabilizes                   │
│  │   → OR: aggressive entry if new regime favors our position     │
│  │                                                                 │
│  └── Sentiment/flow signal                                         │
│      Orderbook imbalance sustained >60% on one side for >10 min  │
│      = retail consensus building. If DS3M disagrees, fade it.     │
│                                                                    │
│  EXIT SIGNALS:                                                     │
│  ├── Edge decay: model P converged to market P → no more edge    │
│  ├── Regime flip: HDP posterior shifts → thesis invalidated       │
│  ├── Stop-loss: mark-to-market loss > 15¢ per contract           │
│  ├── Time decay: <2 hours to settlement, edge < 4¢ → close       │
│  └── Liquidity exit: spread widens >6¢ → exit at market          │
│                                                                    │
│  Outputs:                                                          │
│  ├── entry_signals: [{bracket, side, edge, confidence, reason}]   │
│  ├── exit_signals: [{trade_id, reason, urgency}]                  │
│  └── abstain_signal: bool (don't trade this cycle)                │
└───────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────────────┐
│  LAYER 3: PORTFOLIO OPTIMIZER                                     │
│                                                                    │
│  Given entry/exit signals, optimizes across the full bracket      │
│  surface to maximize risk-adjusted returns.                        │
│                                                                    │
│  SIZING:                                                           │
│  ├── Fractional Kelly (f* = edge / odds, capped at 0.25 Kelly)   │
│  ├── Regime-conditioned caps:                                      │
│  │   Continental: up to 5% bankroll per bracket                   │
│  │   Sea breeze:  up to 4% (higher transition uncertainty)        │
│  │   Frontal:     up to 3% (tail risk from cold fronts)           │
│  │   Nocturnal:   up to 6% (stable, high conviction)              │
│  ├── Correlation-aware: adjacent brackets are correlated →        │
│  │   don't double-count. If long 80-85 and 85-90, treat as       │
│  │   one position for risk purposes.                               │
│  └── Drawdown throttle: reduce sizing by 50% after 3 consecutive │
│      losing days. Resume full size after 2 winners.               │
│                                                                    │
│  ROUTING:                                                          │
│  ├── Maker (limit order) if:                                       │
│  │   edge 3-6¢, spread > 3¢, liquidity_score > 0.5              │
│  │   → Place limit at model price, wait for fill                  │
│  ├── Taker (market order) if:                                      │
│  │   edge > 6¢, OR regime_change_signal, OR time < 3hr to settle │
│  │   → Cross the spread, accept fill cost                         │
│  └── Cancel stale makers after 30 min unfilled                    │
│                                                                    │
│  PORTFOLIO CONSTRAINTS:                                            │
│  ├── Max 8 open positions across all brackets                     │
│  ├── Max 2 positions per bracket (1 each side max)                │
│  ├── Max 15 round trips per day                                   │
│  ├── No new entries within 10 min of a previous fill              │
│  ├── Max 30% of bankroll at risk at any time                      │
│  └── Per-regime daily loss limit: stop after -50¢ in any regime  │
└───────────────────────────────────────────────────────────────────┘
```

---

## DS3M for the Trader: Market State Space Model

The DS3M framework (Mamba → Particle Filter → NSF) can be applied to the **market itself**, not just the weather:

### Market-DS3M Concept

```
Market Feature Vector (20 features):
├── [0-3]   Price dynamics: mid_price, spread, bid_depth, ask_depth
├── [4-7]   Flow: volume_5min, trade_imbalance, large_order_flag, cancel_rate
├── [8-11]  Cross-bracket: adjacent_bracket_prices, bracket_correlation
├── [12-15] Weather context: DS3M_edge, regime_posterior_entropy, ESS, lead_time
├── [16-19] Temporal: hour_sin/cos, minutes_to_settlement, is_pre_cli_window

Market Regimes (learned by HDP):
├── Quiet:      thin book, wide spreads, low volume (overnight/early AM)
├── Active:     tight spreads, high volume, informed trading (post-NWP)
├── Repricing:  rapid directional move, books thinning (regime change)
├── Pinned:     market converged on settlement value, no edge left
└── Panicked:   extreme volume spike, retail pile-in (rare events)

Outputs:
├── P(fill | limit_price, side) — fill probability for maker orders
├── P(adverse_selection) — probability our order gets picked off
├── optimal_entry_time — when to place the order for best execution
└── market_regime → informs position sizing and order type
```

This would be a **separate, smaller model** (Mamba d_model=64, 4 layers, ~200K params) trained on market_snapshots data. It runs alongside the weather DS3M and feeds signals to Layer 2.

---

## Exploitable Market Patterns (Observed)

### 1. Early Morning Tail Mispricing
- **When:** 06:00-11:00 UTC (1-6 AM EST)
- **What:** Tail brackets (<65°F, >95°F) trade at wider spreads and lower prices than fair value
- **Why:** Low retail activity, market makers widen spreads, no one is actively pricing tails
- **Edge:** DS3M can identify tail scenarios (frontal passages, record heat) earlier than the market reprices
- **Strategy:** Place maker orders on tails at model price during quiet hours, get filled when market catches up

### 2. Post-Model-Run Repricing Lag
- **When:** 13:00-15:00 UTC (after 12Z model runs)
- **What:** Fresh HRRR/GFS shift the DS3M forecast 1-3°F, but Kalshi prices lag by 15-30 minutes
- **Why:** Retail traders don't monitor NWP output; market makers may not have sub-hourly repricing
- **Edge:** DS3M ingests new model data within 5 seconds of availability via Herbie
- **Strategy:** Aggressive taker entries when new model data significantly shifts our edge

### 3. Sea Breeze Regime Fade
- **When:** 18:00-21:00 UTC (1-4 PM EST)
- **What:** Sea breeze onset caps temperature rise. Market still prices as if continental heating continues
- **Why:** Sea breeze is a local phenomenon most market participants don't model explicitly
- **Edge:** DS3M's regime detector identifies sea breeze onset from wind direction shift + dewpoint spike
- **Strategy:** Short high brackets (85-90, 90-95) when regime flips to sea_breeze

### 4. Settlement Convergence Arbitrage
- **When:** 02:00-05:00 UTC (night before settlement)
- **What:** Brackets with >90% certainty still trade at 88-92¢ instead of 95-98¢
- **Why:** Settlement risk premium + thin overnight liquidity
- **Edge:** By midnight EST, running high/low is nearly final. DS3M's remaining_move estimate shrinks to ±0.5°F
- **Strategy:** Buy near-certain brackets at small discounts, collect the settlement premium

### 5. Cold Front Underreaction
- **When:** Day before frontal passage
- **What:** Market slowly reprices low brackets but underestimates magnitude
- **Why:** Fronts are spatially large events but their local temperature impact is hard to price from NWP alone
- **Edge:** DS3M's frontal regime has learned from 3 years of SE Florida frontal passages
- **Strategy:** Buy low brackets aggressively when DS3M assigns >40% posterior to frontal regime

---

## Implementation Priority

1. **Orderbook analytics** — compute spread, depth, imbalance, volume from market_snapshots (data already collected)
2. **Entry timing rules** — implement the 5 entry signals above with backtestable logic
3. **Exit intelligence** — edge decay + regime flip + stop-loss with proper mark-to-market
4. **Market-DS3M** — train small Mamba model on market_snapshots data (requires ~30 days of tick data)
5. **Portfolio optimizer** — correlation-aware sizing, drawdown throttle, per-regime loss limits

---

## Data Requirements

All data for Layers 1-3 is **already being collected**:

| Need | Table | Status |
|------|-------|--------|
| Orderbook snapshots | market_snapshots | Collecting (10-min intervals) |
| Bracket prices | market_snapshots.last_price_cents | Collecting |
| Bid/ask depth | market_snapshots.*_depth | Collecting |
| DS3M probabilities | ds3m_estimates | Writing per inference cycle |
| Regime posterior | ds3m_estimates.regime_posterior | Writing per inference cycle |
| Trade lifecycle | ds3m_paper_trades | Writing per trade |
| Settlement outcomes | event_settlements + market_settlements | Collecting |

**Gap:** market_snapshots at 10-min intervals may be too coarse for microstructure analysis. Consider increasing to 1-min for the orderbook data once the trading bot goes live.
