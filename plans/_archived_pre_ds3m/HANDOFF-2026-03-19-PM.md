# Handoff — 2026-03-19 Afternoon Session

## What Was Done This Session

Massive infrastructure build for regime-aware probability estimation. Everything built, tested, and documented.

### New Modules Created (9 files)

| File | Purpose |
|------|---------|
| `src/analyzer/daily_data_builder.py` | Full day data extraction with OHLC hourly aggregation + RAPID_CHANGE injection |
| `src/analyzer/prompt_builder.py` | LLM prompt assembly with calibration context |
| `src/analyzer/prompts/daily_review.md` | LLM instruction template (7-section YAML output) |
| `src/analyzer/ai_review_parser.py` | Parse AI output, store to DB + reviews/ai/ |
| `src/analyzer/train_skf.py` | Train SKF from regime-labeled data via OLS |
| `src/analyzer/hdp_regime_discovery.py` | Shadow-mode HDP-Sticky HMM (nightly batch, regime discovery) |
| `src/analyzer/post_settlement_calibrator.py` | Auto-tune CUSUM + SKF via EMA after settlement |
| `src/engine/kalman_regimes.py` | Switching Kalman Filter (5D state, per-regime models, Bayesian updates) |
| `src/engine/changepoint_detector.py` | CUSUM + SPECI-style threshold change detection |

### Modified Files

| File | Change |
|------|--------|
| `src/collector/store/schema.py` | Added `regime_labels` + `regime_labels_hdp_test` tables, v12 |
| `src/collector/store/db.py` | Added regime_labels methods |
| `src/engine/regime_classifier.py` | Extended `RegimeState` with SKF fields |
| `src/engine/residual_estimator.py` | Added SKF adjustment pathway (trust-gated) |
| `src/engine/signal_engine.py` | Fixed forward_curve_extremes unpacking (6 return values) |
| `pyproject.toml` | Added numpy, pyyaml, 4 new CLI entrypoints |

### Key Decisions Made

1. **SKF for regime tracking** — trajectory-aware, forward prediction, mu/sigma native. Evaluated 15+ alternatives (HMM, HDP-HMM, GP, particle filter, VAE, CRF, GARCH, SOM, etc.). SKF won on: forward prediction + minimal training data + direct mu/sigma + cheap compute.

2. **CUSUM + SPECI thresholds for change detection** — from industrial process monitoring, not Bayesian statistics. BOCD with NIW failed on Miami weather data (diurnal trends + discretized obs broke its assumptions). CUSUM on detrended Fourier residuals works. Verified on March 16 frontal passage — fires at 22:00 LST with temp+dew crash, escalates to P(cp)=1.0 by 23:00.

3. **HDP-Sticky in shadow mode** — runs nightly, discovers regime count automatically, writes to separate table. Goal: eventually replace AI labeling for API independence.

4. **Self-learning loops** — statistical (CUSUM/SKF auto-tune via EMA after settlement) + AI review (sanity check, can override) + SKF batch refit (after AI review) + HDP shadow (silent comparison).

5. **Daily schedule:** calibrate at 09:00 LST, AI review at 09:30 LST, HDP shadow at 09:00 LST.

### Verification

- 35 tests passing
- All modules import cleanly
- CUSUM+SKF verified on March 16 frontal passage
- Data builder produces 55-74K token prompts (fits in context)
- 4 human reviews backfilled into regime_labels
- Signal engine extracts live data for today (2026-03-19)

### Current Bot Output (raw, untuned)

Ran inference for 2026-03-19 at 12:31 LST. Raw consensus output (no residual adjustments):

```
HIGH: consensus 73.5°F ± 1.7°F  |  Obs: 77.0°F (OBS_RUNNING_HOT flag)
LOW:  consensus 61.7°F ± 1.4°F  |  Running low: 62.1°F

Bot output is stale — doesn't incorporate obs divergence. Market has B77.5 at 77-79¢,
bot prices it at 1¢. The full residual pipeline would fix this.
```

---

## What Needs to Be Done Next

### Priority 1: Orchestrator Module

**File:** `src/engine/orchestrator.py`

Build a single module that wires the full inference pipeline end-to-end:

```
signal_engine.extract()
    → baseline_engine.build_baseline()
    → short_range_updater
    → station_transfer
    → changepoint_detector.update()
    → SKF.notify_changepoint() + SKF.update()
    → regime_classifier.classify_regime()
    → signal_families.compute_signal_families()
    → residual_estimator.apply_residual_adjustment()
    → bracket_pricer.bracket_yes_probability() for each active bracket
```

**Key challenges:**

1. **baseline_engine needs `ForecastSourceSnapshot` objects.** Currently model_forecasts are in the DB as raw rows. Need a function that reads `model_forecasts` for today, deduplicates to latest per model/source, and wraps them in `ForecastSourceSnapshot` objects with the correct `family_name` mapping (openmeteo, wethr, nws).

2. **short_range_updater needs `ShortRangeInput`.** This is a thin wrapper around obs deltas — build from signal_engine's extracted obs data.

3. **station_transfer needs obs from nearby stations.** Already in signal_engine's extract output (fawn_temp_f, nearby_divergence_f, etc.).

4. **Brackets come from `active_brackets` table** (populated by trader) or fall back to `market_snapshots` (populated by collector).

5. **Output format:** For each bracket, output `{ticker, market_type, floor, cap, model_probability, market_price, edge}`. Print to stdout for debugging. Later: write to `bracket_estimates` table.

**The orchestrator should be runnable standalone:**
```bash
python -m engine.orchestrator --db miami_collector.db --date 2026-03-19
```

And also importable for the live loop:
```python
from engine.orchestrator import run_inference_cycle
results = run_inference_cycle(db_path, target_date)
```

### Priority 2: Trader ↔ Miami Bot Bracket Interface

The trader (separate repo: `weather-trader/`) needs to push active brackets to the miami bot. Two approaches:

**Option A: Shared SQLite table** (simplest)
- Trader writes to `active_brackets` table in `miami_collector.db`
- Miami bot reads from `active_brackets` in the orchestrator
- Trader uses Kalshi WS to discover markets, parses tickers, writes brackets

**Option B: REST API** (cleaner separation)
- Miami bot exposes a simple HTTP endpoint: `POST /brackets` accepts a list of brackets, `GET /probabilities` returns current estimates
- Trader calls the API
- More work, but proper for multi-city scaling

For now, go with Option A. The `active_brackets` table already exists in the schema:
```sql
CREATE TABLE IF NOT EXISTS active_brackets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    target_date TEXT NOT NULL,
    market_type TEXT NOT NULL,
    ticker TEXT NOT NULL,
    floor_strike REAL,
    cap_strike REAL,
    directional TEXT,
    ...
)
```

### Priority 3: Kalshi WS Bracket Discovery

The collector already has `sources/kalshi_ws.py` and `sources/kalshi_rest.py` with authentication and market discovery. What's needed:

1. **Auto-populate `active_brackets`** from the Kalshi WS feed. When a new KMIA high/low market appears, parse the ticker to extract floor/cap strikes and write to `active_brackets`.

2. **Auto-remove stale brackets** — when the climate day rolls over (05:00Z), clear yesterday's brackets and discover today's.

3. **Ticker parsing:** Kalshi tickers follow patterns like:
   - `KXHIGHMIA-26MAR19-B77.5` → high market, bracket [77.0, 77.5]
   - `KXHIGHMIA-26MAR19-T73` → high market, under 73.0 (directional)
   - `KXLOWTMIA-26MAR19-B61.5` → low market, bracket [61.0, 61.5]
   - `KXLOWTMIA-26MAR19-T57` → low market, under 57.0 (directional)
   - `KXLOWTMIA-26MAR19-T64` → low market, over 64.0 (directional)

   The `T` prefix with the lower number = "under" (threshold below). The `T` prefix with the higher number = "over". The `B` prefix = exact bracket. Floor is always 0.5 below the number (B77.5 = [77.0, 77.5]).

   Wait — actually check the `market_snapshots` table, it already has `floor_strike` and `cap_strike` parsed. The collector is already doing this parsing in `kalshi_rest.py` or `kalshi_ws.py`. Reuse that logic.

4. **The WS feed already sends market data.** Check `sources/kalshi_ws.py` for how it handles orderbook updates. The discovery part (which markets exist for KMIA today) is in `sources/kalshi_rest.py` via `discover_markets()` or similar.

### Priority 4: Live Loop

Once the orchestrator and bracket discovery work:

```python
# In collector/runner.py, add a new loop:
async def inference_loop(self):
    """Run inference every 5 minutes on active brackets."""
    while True:
        try:
            results = run_inference_cycle(self.db_path, target_date)
            for r in results:
                store.insert_bracket_estimate(r)
                if r.edge > MIN_EDGE:
                    log.info("EDGE: %s %.1f¢ (model=%.1f, market=%.1f)",
                             r.ticker, r.edge, r.model_prob * 100, r.market_price)
        except Exception as e:
            log.error("Inference cycle failed: %s", e)
        await asyncio.sleep(300)  # 5 minutes
```

---

## Key Files to Read First

1. **`plans/SYSTEM-MAP.md`** — complete system reference (updated today with all new modules)
2. **`plans/regime-classification-ai-labeling.md`** — full regime system plan with operational schedule
3. **`src/engine/signal_engine.py`** — signal extraction (already works, tested today)
4. **`src/engine/baseline_engine.py`** — needs ForecastSourceSnapshot wiring
5. **`src/engine/source_registry.py`** — ForecastSourceSnapshot interface
6. **`src/engine/bracket_pricer.py`** — bracket → probability (42 lines, simple)
7. **`src/collector/sources/kalshi_rest.py`** — market discovery
8. **`src/collector/sources/kalshi_ws.py`** — live orderbook feed

## Current State

- **Bot is running** — collector is live, ingesting data continuously
- **DB is populated** — 385MB+ of obs, forecasts, market data
- **Signal extraction works** — tested on 2026-03-19 live data
- **Bracket pricer works** — produces probabilities from distributions
- **Change detector works** — CUSUM fires on frontal passages
- **SKF works** — identifies regimes from obs stream
- **What's missing:** the orchestrator that connects signal_engine output to baseline_engine to residual_estimator to bracket_pricer, and the bracket discovery from Kalshi WS that populates active_brackets

## DB Tables Relevant to Next Steps

- `active_brackets` — exists but empty. Needs to be populated by trader/WS discovery.
- `bracket_estimates` — exists. Orchestrator should write estimates here.
- `market_snapshots` — populated by collector. Has tickers with floor/cap strikes already parsed. Can be used as bracket source until active_brackets is populated.
- `model_forecasts` — all model forecasts. Orchestrator needs to read and wrap in ForecastSourceSnapshot.
- `model_consensus` — bias-adjusted consensus. Can be used as quick baseline.
