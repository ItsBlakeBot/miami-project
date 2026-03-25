# Source Trust Backfill Summary

- Station: KMIA
- Window: 2026-02-05 → 2026-03-22
- Records: 9988
- Covered days: 9
- Sufficient days (target 15): False

## Per-family metrics
- openmeteo: n=5934, mae=2.9792, rmse=3.868057, bias=-0.907882
- wethr: n=4054, mae=6.392617, rmse=7.402208, bias=-1.194512

## Reliability by family / time-to-settlement bucket
### openmeteo
- 0-3h: n=110, mae=1.948909, rmse=2.470814, bias=-0.594182
- 12-24h: n=1092, mae=2.98631, rmse=3.956102, bias=-0.663929
- 24h+: n=3036, mae=3.372839, rmse=4.341413, bias=-0.784499
- 3-6h: n=186, mae=2.425591, rmse=2.977863, bias=-0.400215
- 6-12h: n=1510, mae=2.325854, rmse=2.836616, bias=-1.417762

### wethr
- 0-3h: n=50, mae=7.456, rmse=8.339376, bias=-2.204
- 12-24h: n=916, mae=6.241103, rmse=7.392199, bias=-2.225055
- 24h+: n=970, mae=5.50733, rmse=6.625585, bias=0.035186
- 3-6h: n=126, mae=6.904762, rmse=7.821475, bias=-1.968254
- 6-12h: n=1992, mae=6.834292, rmse=7.706602, bias=-1.245146
