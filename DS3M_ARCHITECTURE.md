# DS3M v2 Architecture

**Deep Switching State Space Model with Normalizing Flows**
**Version:** 2.0 (Production)
**Last updated:** 2026-03-25

---

## What Is DS3M?

DS3M is a unified, end-to-end differentiable inference system for weather bracket trading. It replaces the old multi-component pipeline (EMOS + BOA + regime catalog + LETKF + sigma climatology + truncated Normal CDF) with a single coherent generative model that learns weather regimes, temporal dynamics, and bracket probabilities autonomously from data.

---

## Model Summary (4.4M parameters)

| Component | Architecture | Params | Role |
|-----------|-------------|--------|------|
| **Mamba Encoder** | 8-layer Mamba2, d_model=256, d_state=32 | 3.6M | Extract temporal patterns from 33-feature weather vectors |
| **Differentiable PF** | 500 particles, d_latent=32, HDP regimes | 55K | Track temperature state + discover weather regimes |
| **Neural Spline Flow** | 10 coupling layers, 24 spline bins | 784K | Convert particle cloud into continuous temperature density |

---

## Inference Pipeline (5-second cycles)

```
33-Feature Vector                    Mamba Encoder                 DPF
┌──────────────┐     ┌──────────────────────────────┐     ┌───────────────┐
│ NWP [0-10]   │     │ 8× MambaBlock                │     │ 500 particles │
│ Obs [11-16]  │────▶│   SelectiveSSM (d=256, s=32) │────▶│ HDP regimes   │
│ Atmos [17-22]│     │   LayerNorm + residual        │     │ Skew-normal   │
│ Time [23-28] │     │   Dropout 0.15                │     │ emission      │
│ Deriv [29-32]│     │                                │     │               │
└──────────────┘     │ Output: h_t ∈ ℝ²⁵⁶           │     │ Output:       │
                     └──────────────────────────────┘     │  particles    │
                                                           │  regime post  │
                                                           │  ESS          │
                                                           └───────┬───────┘
                                                                   │
     ┌─────────────────────────────────────────────────────────────┘
     │
     ▼
Neural Spline Flow              Bracket Pricer              Paper Trader
┌─────────────────┐     ┌──────────────────────┐     ┌──────────────────┐
│ 10 coupling     │     │ CDF(bracket_edges)   │     │ Regime sizing    │
│ layers, 24 bins │────▶│ = bracket probs      │────▶│ Kelly fraction   │
│                 │     │ vs market price      │     │ Maker/taker      │
│ Cond: h_t +    │     │ = edge per bracket   │     │ Risk limits      │
│  stats + regime │     │                      │     │                  │
│ = 280-dim       │     │ Bracket KF Bank:     │     │ Max 8 open       │
│                 │     │ 10 parallel filters  │     │ 15 daily limit   │
│ Output: CDF(T) │     │ Innovation monitor   │     │ 10-min cooldown  │
└─────────────────┘     └──────────────────────┘     └──────────────────┘
```

---

## HDP Regime Discovery

The Hierarchical Dirichlet Process discovers weather regimes without manual specification.

**Physical priors (5 seed regimes):**
| ID | Regime | Signature | KF Priors (Q, R) |
|----|--------|-----------|------------------|
| 0 | Continental | NW wind, low dew, clear sky | 0.4, 1.0 |
| 1 | Sea Breeze | E/SE wind shift, dew spike, Cu buildup | 0.2, 0.6 |
| 2 | Frontal | N wind, pressure rise, temp drop | 1.5, 2.0 |
| 3 | Tropical | S/SW flow, high CAPE, SST-driven | 0.3, 0.8 |
| 4 | Nocturnal | Calm, radiational cooling, stable | 0.6, 1.2 |

**v2 Enhancements:**
- **Learnable concentration α**: Controls regime birth rate, optimized during training
- **Merge-split moves**: Automatically merge redundant regimes (KL < 0.05) and split bimodal ones
- **Per-regime emission noise**: Each regime learns its own observation noise scale

---

## Kalman Filter Architecture

### Regime-Conditioned Scalar KF
One KF tracking the temperature point estimate, with Q/R blended across regimes by posterior probability.

### Bracket KF Bank (NEW in v2)
10 parallel scalar KFs, one per temperature bracket:
- Tail brackets (<60°F, 100+°F): higher Q (volatile), higher R (uncertain)
- Center brackets (75-85°F): lower Q (stable), lower R (certain)
- **Innovation monitor**: tracks residual statistics per bracket, fires `regime_change_signal` when innovations become non-Gaussian (early warning before HDP catches the shift)

---

## Training Architecture

### Data (262K samples, augmented to ~80K effective independent)
| Source | Records | Features |
|--------|---------|----------|
| IEM ASOS hourly (10 stations) | 269K | temp, dew, wind, sky, pressure |
| Open-Meteo historical forecast | 28K hours | CAPE, CIN, cloud cover layers |
| NDBC buoy (VAKF1) | 255K | SST, air temp, marine wind |
| CLI daily backfill | 11.4K | Daily max/min (05Z-05Z UTC) |

### Augmentation Pipeline
| Method | Probability | Effect |
|--------|------------|--------|
| Temporal jitter | 0.5 | Shift ±1-3 timesteps |
| Feature noise | 0.7 | Calibrated Gaussian per feature |
| Station dropout | 0.15 | Zero one feature group |
| Regime splicing | 0.1 | Splice halves from different samples |

### Phase 1: Mamba Pre-training (50 epochs)
- **Loss**: 0.3×MSE(next_temp) + 0.3×MSE(remaining_move) + 0.4×CRPS(daily_max)
- **Optimizer**: AdamW (lr=1e-3, weight_decay=1e-3)
- **Schedule**: Cosine annealing, early stopping (patience=10)

### Phase 2: NSF Fine-tuning (100 epochs)
- **Loss**: CRPS + bracket Brier score
- **Data**: CLI settlements + market settlement outcomes

### Nightly Retraining
- Post-settlement: 120-day rolling window incremental update
- Incorporates new day's data, adjusts to drift

---

## Key Design Decisions

1. **UTC everywhere**: All timestamps, time encoding, and CLI day boundaries use UTC. Climate day = 05Z to 05Z.

2. **IEM 3-letter mapping**: IEM ASOS uses 3-letter codes (MIA, FLL). We maintain STATION_MAP for 4-letter ICAO conversion.

3. **Differentiable throughout**: Particle filter uses optimal transport resampling (Corenflos et al. 2021), enabling gradient flow from CRPS loss through the entire pipeline.

4. **MPS first, CUDA second**: Development on Apple Silicon (MPS), production-ready for CUDA. Pure-PyTorch SSM implementation — no mamba-ssm CUDA dependency required.

5. **Augmentation justifies depth**: 8 Mamba layers (3.6M params) would overfit on raw 11K station-days. Augmentation pipeline inflates effective independent samples to ~50-80K, supporting the deeper architecture.
