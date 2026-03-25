# DS3M Evolution Plan: From Weather Forecaster to Profit-First Money Machine

**Version:** 1.0  
**Date:** 2026-03-23  
**Author:** BMO  
**Goal:** Transform DS3M into a money-making agent that uses weather forecasting as one tool among many.

## Vision

DS3M's **only terminal objective** is to maximize long-term risk-adjusted profit on Kalshi temperature bracket markets.

Weather forecasting is a powerful instrument in its toolkit — not the purpose of its existence. DS3M should be free to win through superior forecasting, sentiment exploitation, timing, liquidity awareness, or any other edge.

Being right on the weather is a **bonus**, not the scorecard.

## Current Limitations

- Entry logic is dominated by forecast edge.
- Optimization targets are still heavily forecast-oriented (CRPS, calibration).
- Limited signals for sentiment, timing, and liquidity.
- Daily review evaluates it primarily as a forecasting system.

## Target Architecture

Introduce a **Composite Trade Attractiveness Score**:

```python
attractiveness = (
    w1 * forecast_edge +
    w2 * sentiment_mispricing +
    w3 * timing_score +
    w4 * liquidity_bonus
)
```

The entry tuner will learn the optimal weights (w1–w4).

### New Signals

- **Sentiment Mispricing**: Deviation from recent volume-weighted fair value.
- **Timing Score**: Time-of-day profitability patterns.
- **Liquidity Bonus**: Bonus for entering when real depth is present (target ~500 contracts).
- **Forecast Edge**: Retained but no longer dominant.

## Files to Create

1. `src/ds3m/profit_objective.py` — Core attractiveness score calculation.
2. `src/ds3m/sentiment_detector.py` — Mispricing and momentum detection.
3. `src/ds3m/timing_model.py` — Time-of-day bias learning.

## Files to Modify

- `src/ds3m/paper_trader.py` — Replace `_evaluate_entry()` with new profit objective.
- `src/engine/entry_tuner.py` — Expand to tune the four weights + liquidity logic.
- `src/engine/exit_tuner.py` — Integrate liquidity-adjusted pricing in replay.
- `src/analyzer/daily_post_settlement.py` — Call entry tuner and feed results to review.
- `src/analyzer/prompts/daily_review.md` — Add "Money Machine Assessment" section.

## Success Criteria

The daily review should be able to clearly answer:

- What % of DS3M profit came from forecast vs non-forecast edges?
- Is DS3M becoming less dependent on accurate weather prediction?
- Are the tuners producing stable, profitable policies?

## Implementation Order (Recommended)

1. Create the three new `ds3m/` files.
2. Update `paper_trader.py` to use the new score.
3. Fully implement the entry tuner with liquidity-adjusted pricing.
4. Update exit tuner and daily review prompt.
5. Integrate into `daily_post_settlement.py`.

## Open Questions for Blake

- Initial weight distribution for the four components?
- Should we keep a "pure forecast mode" for comparison?
- Should attractiveness score also influence position sizing?

---

This document is ready to hand to Opus. It contains everything needed to implement the full transformation.

Would you like me to spawn a coding agent (Codex or Opus) to begin implementation based on this plan?
