# BOA Family Weight Audit (openmeteo vs wethr)

Date: 2026-03-22
Scope: `/Users/blakebot/blakebot/miami-project` + `/Users/blakebot/blakebot/weather-trader`

## Summary

- Claim checked: "BOA downweighted wethr while valuing openmeteo more"
- Result: **partially true**
  - **LOW market BOA** favors openmeteo overall
  - **HIGH market BOA** currently favors wethr (MOS experts dominate)
- Separate from BOA, the source-trust layer (`source_trust_state.json`) currently favors openmeteo over wethr.

## Evidence

### 1) Source-trust state (non-BOA)
File: `analysis_data/source_trust_state.json`

```json
{
  "family_multipliers": {
    "openmeteo": 1.3,
    "wethr": 0.7
  }
}
```

This layer explicitly upweights openmeteo vs wethr.

### 2) BOA state (normalized by market side)
File: `analysis_data/boa_state.json`

Using `engine.boa.BOAManager.get_weights()` normalization:

- **HIGH** family sums:
  - openmeteo: **0.0000**
  - wethr: **1.0000**
- **LOW** family sums:
  - openmeteo: **0.7167**
  - wethr: **0.2833**

Top HIGH experts are wethr MOS variants (`wethr:GFS-MOS`, `wethr:NAM-MOS`, `wethr:NBS-MOS`).
Top LOW expert is `openmeteo:ECMWF-AIFS`.

## Weather-trader repo check

Path: `/Users/blakebot/blakebot/weather-trader`

- No BOA module/state/history found there.
- `trader.db` tables exist (`signal_snapshots`, `parameter_history`, etc.) but currently have no BOA evidence.

## Conclusion

- If the statement is about the **trust multipliers** or **LOW-side BOA**, it is correct.
- If the statement is about BOA behavior across **both** high+low markets, it is **not fully correct** right now.
