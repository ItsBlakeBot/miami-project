# Archive Audit Report

- DB: `miami_collector.db`
- Generated (UTC): `2026-03-21T17:13:45Z`
- Station: `KMIA`
- Required tables: `11`
- Missing tables: `0`
- Replay-ready (schema): `True`

## Table details

### `active_brackets`
- present: `True`
- rows: `48`
- missing required columns: `0`
- timestamp column: `updated_at`
- latest timestamp: `2026-03-21T17:07:11.461Z`
- freshness mode: `realtime`
- freshness status: `fresh`
- freshness minutes: `6.57`

### `atmospheric_data`
- present: `True`
- rows: `29809`
- missing required columns: `0`
- timestamp column: `valid_time_utc`
- latest timestamp: `2026-03-27T23:00`
- freshness mode: `forecast_validity`
- freshness status: `future_expected`
- freshness minutes: `-8986.24`

### `bracket_estimates`
- present: `True`
- rows: `304194`
- missing required columns: `0`
- timestamp column: `timestamp_utc`
- latest timestamp: `2026-03-21T17:13:45Z`
- freshness mode: `realtime`
- freshness status: `fresh`
- freshness minutes: `0.01`

### `event_settlements`
- present: `True`
- rows: `16`
- missing required columns: `0`
- timestamp column: `received_at`
- latest timestamp: `2026-03-21T16:57:31Z`
- freshness mode: `historical`
- freshness status: `historical`
- freshness minutes: `16.25`

### `fawn_observations`
- present: `True`
- rows: `531`
- missing required columns: `0`
- timestamp column: `timestamp_utc`
- latest timestamp: `2026-03-21T17:00:00Z`
- freshness mode: `realtime`
- freshness status: `fresh`
- freshness minutes: `13.76`

### `forward_curves`
- present: `True`
- rows: `2472`
- missing required columns: `0`
- timestamp column: `snapshot_time_utc`
- latest timestamp: `2026-03-21T16:56:59Z`
- freshness mode: `forecast_validity`
- freshness status: `fresh`
- freshness minutes: `16.78`

### `market_settlements`
- present: `True`
- rows: `178`
- missing required columns: `0`
- timestamp column: `settled_at`
- latest timestamp: `2026-03-21T04:59:00Z`
- freshness mode: `historical`
- freshness status: `historical`
- freshness minutes: `734.76`

### `market_snapshots`
- present: `True`
- rows: `418049`
- missing required columns: `0`
- timestamp column: `snapshot_time`
- latest timestamp: `2026-03-21T17:13:42Z`
- freshness mode: `realtime`
- freshness status: `fresh`
- freshness minutes: `0.06`

### `model_forecasts`
- present: `True`
- rows: `428335`
- missing required columns: `0`
- timestamp column: `fetch_time_utc`
- latest timestamp: `2026-03-21T17:09:37Z`
- freshness mode: `realtime`
- freshness status: `fresh`
- freshness minutes: `4.15`

### `nearby_observations`
- present: `True`
- rows: `18139`
- missing required columns: `0`
- timestamp column: `timestamp_utc`
- latest timestamp: `2026-03-21T17:55:00Z`
- freshness mode: `realtime`
- freshness status: `future`
- freshness minutes: `-41.24`

### `observations`
- present: `True`
- rows: `3656`
- missing required columns: `0`
- timestamp column: `timestamp_utc`
- latest timestamp: `2026-03-21T17:10:00+00:00`
- freshness mode: `realtime`
- freshness status: `fresh`
- freshness minutes: `3.76`
