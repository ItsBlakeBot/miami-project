# Changepoint Replay Comparison

- Station: KMIA
- Rows: 3729
- Proxy events: 64

## BOCPD enabled
- Fires: 1568 (L1=538, L2=1030, L3=0)
- Matched/missed events: 64/0
- Median latency (min): 0.0
- False-positive proxy: 641

## CUSUM only
- Fires: 1568 (L1=538, L2=1030)
- Matched/missed events: 64/0
- Median latency (min): 0.0
- False-positive proxy: 641

## Delta (BOCPD - CUSUM)
- Fires: 0
- Missed events: 0
- Median latency minutes: 0.0
- False-positive proxy: 0
