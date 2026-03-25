# Canonical Replay Bundle

- Station: KMIA
- Dates: None → None
- Resolved dates: 16

## Core metrics
- Brier (all estimates): 0.161996
- Log-loss (all estimates): 0.903573
- Trade Brier: None
- Sharpness |p-0.5| mean: 0.401129

## Remaining-target metrics
- high: n=4, mae=0.51825, rmse=0.730034, crps=0.512011
- low: n=4, mae=1.43825, rmse=2.303997, crps=0.863318
- CRPS method: quantile-grid approximation over remaining-target transformed predictive distribution

## Trading outcomes
- Trades: 0
- Contracts: 0
- Realized PnL cents: 0
- Expected value cents total: 0
- Expected - realized cents: 0

## Trade-quality cuts

## Regime cuts
- high / advective_cooling: n=1, mae=0.582, rmse=0.582, crps=0.4047
- high / mixed_uncertain: n=3, mae=0.497, rmse=0.773105, crps=0.547781
- low / advective_cooling: n=1, mae=0.104, rmse=0.104, crps=0.244957
- low / mixed_uncertain: n=3, mae=1.883, rmse=2.659749, crps=1.069438

## Worst-day diagnostics
- Highest Brier day: {'date': '2026-03-19', 'estimates': 12, 'brier': 0.293842, 'logloss': 1.407264, 'remaining_high_mae': None, 'remaining_high_rmse': None, 'remaining_low_mae': None, 'remaining_low_rmse': None, 'trades': 0, 'wins': 0, 'win_rate': None, 'contracts': 0, 'expected_value_cents': 0.0, 'realized_pnl_cents': 0.0}
- Lowest realized PnL day: {'date': '2026-03-14', 'estimates': 1, 'brier': 0.0, 'logloss': 1e-06, 'remaining_high_mae': 0.014, 'remaining_high_rmse': 0.014, 'remaining_low_mae': 0.083, 'remaining_low_rmse': 0.083, 'trades': 0, 'wins': 0, 'win_rate': None, 'contracts': 0, 'expected_value_cents': 0.0, 'realized_pnl_cents': 0.0}

## TTL cuts
- 6-12h: n=25, brier=0.116999, logloss=0.758851, rem_high_mae=0.247333, rem_low_mae=1.554
- 12-24h: n=30, brier=0.199493, logloss=1.024175, rem_high_mae=1.331, rem_low_mae=1.091

## Changepoint compare (BOCPD vs CUSUM)
- Events: 74
- BOCPD fires: 1786
- CUSUM fires: 1786
- Delta false-positive proxy: 0
