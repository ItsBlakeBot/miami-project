# DS3M v3 Implementation Plan
## Deep Switching State Space Model — Weather Perception Engine

**Status**: Active development
**Target**: Production-grade probabilistic weather forecasting for Kalshi bracket markets

> **CRITICAL MIGRATION NOTES (DO NOT FORGET):**
> 1. **Mamba-2 → Mamba-3**: Current 5070 training uses Mamba-2. Once complete and validated,
>    retrain with Mamba-3 (complex-valued states, MIMO, exp-trapezoidal discretization).
>    Install `mamba-ssm>=2.3.1` from source on 5070. Write pure-PyTorch fallback for Mac.
> 2. **NSF → Flow Matching (OT-CFM)**: Replace Neural Spline Flow with Conditional
>    Optimal Transport Flow Matching. Better tails, no bins/transforms to tune,
>    cleaner gradients. Uses `torchdiffeq` for ODE solve (works on both CUDA and MPS).
> 3. **Order of operations**: Finish current Mamba-2 + NSF training → validate pipeline
>    end-to-end → THEN upgrade to Mamba-3 + Flow Matching and retrain on 5070.
**Primary station**: KMIA (Miami International)
**Supporting stations**: 35 SE Florida ASOS/mesonet stations

---

## Architecture Overview

```
Raw Observations (35 stations, 5-sec cycle)
    │
    ▼
┌─────────────────────────────────────────────────┐
│  FEATURE ENGINEERING (feature_vector.py)          │
│  33→40 features per station per timestep          │
│  NWP[0-10], Obs[11-16], Atmos[17-22],           │
│  Temporal[23-28], Derived[29-32], SST[33-35],    │
│  Pressure tendency[36-37], Visibility[38-39]      │
└─────────────────────┬───────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│  GRAPH-MAMBA-3 ENCODER (graph_mamba3.py)         │
│                                                   │
│  Per-station Mamba-3 temporal encoder:            │
│    d_model=384, d_state=24 (MIMO), 8 layers      │
│    Complex-valued state updates                   │
│    Exponential-trapezoidal discretization         │
│    No causal convolution (Mamba-3 removes it)     │
│                                                   │
│  Graph attention layers (4 layers):               │
│    35 station nodes, dynamic edges               │
│    Wind-direction-aware adjacency                 │
│    Multi-head attention (8 heads)                 │
│    Propagates spatial signals:                    │
│      - Sea breeze front detection                 │
│      - Frontal passage timing                     │
│      - Urban heat island gradients                │
│                                                   │
│  Output: 384-dim embedding per station            │
│          + 384-dim global context                  │
└─────────────────────┬───────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│  DIFFERENTIABLE PARTICLE FILTER (dpf.py)         │
│  1000 particles, d_latent=64                      │
│  Learned transition + emission networks           │
│  Mamba-3 embedding conditions the proposal        │
│                                                   │
│  Output: particle cloud + weights                 │
│          → posterior over temperature states       │
└─────────────────────┬───────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│  HDP REGIME DYNAMICS (hdp_regime.py)              │
│  Hierarchical Dirichlet Process                   │
│  5-8 weather regimes with merge/split hysteresis  │
│  Concentration parameter learning                 │
│  Innovation monitoring for regime birth/death     │
│                                                   │
│  Regimes: continental, sea_breeze, frontal,       │
│           tropical, nocturnal, (+discovered)       │
│                                                   │
│  Output: regime posterior probabilities            │
│          regime transition matrix                  │
└─────────────────────┬───────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│  FLOW MATCHING DENSITY ESTIMATOR (flow_match.py) │
│  Replaces Neural Spline Flow (NSF)               │
│                                                   │
│  Conditional OT-CFM:                             │
│    Conditioning: Mamba embedding + regime +       │
│                  particle summary + KF state      │
│    Learns P(T_max | context) as continuous flow   │
│    No bins/transforms to tune                     │
│    Better tail calibration than NSF               │
│                                                   │
│  Output: full probability density over temp       │
│          → bracket probability vector             │
└─────────────────────┬───────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│  REGIME-CONDITIONED KF BANK (realtime_updater.py)│
│  Per-regime Q/R parameters                        │
│  Per-bracket tracking KFs                         │
│  Soft-blended by HDP posterior                    │
│  Adapts from settlement errors                    │
│                                                   │
│  Output: filtered temp estimate + uncertainty     │
│          running max/min tracking                 │
└─────────────────────┬───────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│  CONFORMAL CALIBRATOR (conformal_calibrator.py)  │
│  Distribution-free prediction intervals           │
│  Adaptive coverage targeting (90%, 95%)           │
│                                                   │
│  Output: calibrated bracket probabilities         │
│          prediction intervals                     │
└─────────────────────┬───────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────┐
│  BRACKET PRICER (bracket_pricer_v2.py)           │
│  Integrates flow density over bracket ranges      │
│  Applies conformal calibration                    │
│  Outputs: P(bracket_i wins) for all brackets     │
│                                                   │
│  → Sent to Weather Trader TSSM as frozen input   │
│  → Displayed on live dashboard                    │
└─────────────────────────────────────────────────┘
```

---

## Implementation Phases

### Phase 1: Mamba-2 → Mamba-3 Upgrade
**Files**: `mamba_encoder.py` → `mamba3_encoder.py`, `graph_mamba.py` → `graph_mamba3.py`
**Priority**: HIGH — do this AFTER current 5070 training completes

1. **Install mamba-ssm v2.3.1 on 5070**
   - `MAMBA_FORCE_BUILD=TRUE pip install git+https://github.com/state-spaces/mamba.git --no-build-isolation`
   - Requires CUDA, causal-conv1d >= 1.4.0

2. **Write Mamba3Block wrapper**
   - Use official `Mamba3` class from `mamba_ssm.modules.mamba3`
   - Complex-valued states (set `complex_ssm=True`)
   - MIMO formulation (set `mimo=True`, `mimo_groups=4`)
   - Exponential-trapezoidal discretization (default in Mamba-3)
   - d_model=384, d_state=24 (MIMO halves effective state size)
   - 8 layers with dropout=0.15, LayerNorm

3. **Write pure-PyTorch Mamba-3 fallback for Mac inference**
   - Implement the exponential-trapezoidal recurrence in ~50 lines
   - Complex-valued state updates: `h_new = exp(A * dt) * h + B_trap * x`
   - MIMO: reshape d_model into groups, independent SSMs per group
   - Load same weights as CUDA version
   - ~10ms per step on M4 (inference only, single timestep)

4. **Update GraphMamba to use Mamba-3 backbone**
   - Replace MambaBlock with Mamba3Block in per-station encoder
   - Graph attention layers stay the same (PyTorch native)
   - Test on 5070: verify graph + Mamba-3 train without issues

5. **Expand feature vector to 40 dimensions**
   - Add SST from NDBC buoy [33-35]: sst_f, sst_anomaly, sst_trend
   - Add pressure tendency [36-37]: 1h_change, 3h_change
   - Add visibility [38]: current visibility (mi)
   - Add precip indicator [39]: boolean rain/no-rain
   - Update `feature_vector.py` and `training_pipeline.py`

### Phase 2: NSF → Flow Matching
**Files**: `neural_spline_flow.py` → `flow_matching.py`
**Priority**: HIGH — cleaner density estimation

1. **Implement Conditional OT-CFM**
   - Vector field network: 3-layer MLP (512 hidden)
   - Input: (t, x, condition) where t ∈ [0,1], x = temperature, condition = Mamba embedding
   - Target: optimal transport velocity field
   - Loss: MSE between predicted and target velocity
   - No ODE solve during training (simulation-free)

2. **ODE solver for inference**
   - Use `torchdiffeq` or simple Euler/RK4
   - 20-50 steps for density evaluation
   - Works on both CUDA and MPS

3. **Bracket probability integration**
   - Sample N=1000 points from base distribution (Gaussian)
   - Push through learned flow
   - Histogram into bracket ranges
   - Or: use change-of-variables formula for exact density

4. **Training**
   - Same conditioning as NSF: Mamba embedding + regime + particle summary
   - Minibatch OT for straighter paths (faster inference)
   - Validate against NSF on held-out settlement data

### Phase 3: HDP Hardening
**Files**: `hdp_regime.py`
**Priority**: MEDIUM

1. **Concentration parameter learning**
   - Online update of Dirichlet concentration α
   - Controls how readily new regimes are born

2. **Merge/split hysteresis**
   - Don't merge regimes that just split (cooldown period)
   - Don't split regimes below minimum occupancy
   - Exponential moving average on merge/split criteria

3. **Innovation monitoring**
   - Track KF innovation sequence per regime
   - If innovation exceeds 3σ consistently → propose new regime
   - If two regimes' innovations are indistinguishable → merge

### Phase 4: Enhanced Particle Filter
**Files**: `diff_particle_filter.py`
**Priority**: MEDIUM-LOW (current DPF works, this is refinement)

1. **Diffusion-based proposal sampling**
   - Replace Gaussian proposal with learned diffusion model
   - Better coverage of multimodal posteriors
   - Based on DiffPF (2025) paper

2. **Adaptive particle count**
   - More particles during regime transitions
   - Fewer during stable periods
   - ESS-based dynamic allocation

### Phase 5: Extended Station Network
**Files**: `graph_mamba3.py`, `enriched_backfill.py`, `training_pipeline.py`
**Priority**: MEDIUM

1. **Expand from 35 to 50+ stations**
   - Add Tampa Bay area (KTPA, KPIE, KMCF)
   - Add Orlando area (KMCO, KORL, KSFB)
   - Add Keys (KEYW, KMTH)
   - Each new station needs IEM backfill + enriched data

2. **Dynamic graph structure**
   - Wind-direction weighted edges (upwind stations get higher weight)
   - Time-lagged edges (station X 30 min ago → station Y now)
   - Learned edge weights from training

---

## Training Pipeline

### Pre-training (Phase 1 — Mamba-3)
- **Data**: 35 stations × 3 years × hourly = ~900K samples
- **Augmentation**: temporal jitter, station dropout, feature noise, synthetic regime transitions
- **Effective samples**: ~4M with augmentation
- **Loss**: 0.3 MSE_next + 0.3 MSE_remaining + 0.4 CRPS_daily
- **Hardware**: RTX 5070 (12GB VRAM)
- **Estimated time**: 3-5 hours for 50 epochs

### Flow Matching Training (Phase 2)
- **Data**: daily settlements × conformal calibration windows
- **Loss**: Flow matching objective (MSE on velocity field)
- **Hardware**: RTX 5070 or Mac MPS (lighter model)
- **Estimated time**: 1-2 hours

### Nightly Incremental Update
- **Data**: 1 new day of observations across all stations
- **Procedure**: 1-2 epochs fine-tune on new day's data
- **Hardware**: Mac Mini M4 (sufficient for incremental)
- **Time**: 5-10 minutes

---

## File Map

```
src/engine/ds3m/
├── __init__.py                  # Module docstring
├── config.py                    # Unified DS3MConfig
├── feature_vector.py            # 40-feature construction (live)
├── graph_mamba3.py              # NEW: Mamba-3 + graph attention encoder
├── mamba3_encoder.py            # NEW: Pure Mamba-3 (no graph, for ablation)
├── mamba3_inference.py          # NEW: Pure-PyTorch Mac fallback
├── flow_matching.py             # NEW: OT-CFM density estimator
├── diff_particle_filter.py      # Differentiable particle filter
├── hdp_regime.py                # HDP regime dynamics
├── realtime_updater.py          # Regime-conditioned KF bank
├── conformal_calibrator.py      # Distribution-free calibration
├── bracket_pricer_v2.py         # Bracket probability integration
├── orchestrator_v2.py           # Main loop: collect → infer → price → emit
├── paper_trader_v2.py           # DS3M-level paper trading (being replaced by TSSM)
├── training_pipeline.py         # Dataset + training loops
├── train.py                     # CLI training runner
├── augmentation.py              # Data augmentation transforms
├── enriched_backfill.py         # Historical data enrichment
├── cli_backfill.py              # CLI settlement backfill
├── kalshi_data_puller.py        # Kalshi historical trade data
├── graph_mamba.py               # DEPRECATED: Mamba-2 graph encoder
├── mamba_encoder.py             # DEPRECATED: Mamba-2 encoder
├── neural_spline_flow.py        # DEPRECATED: NSF density estimator
└── _deprecated_v1/              # Old v1 files
```

---

## Outputs to Weather Trader

The DS3M emits a **frozen signal packet** every 5 seconds to the Weather Trader TSSM:

```python
@dataclass
class DS3MSignal:
    timestamp_utc: str
    station: str                          # "KMIA"
    target_date: str                      # "2026-03-26"

    # Bracket probabilities (6 floats, sum to 1)
    bracket_probs: list[float]            # [0.02, 0.08, 0.35, 0.40, 0.12, 0.03]
    bracket_ranges: list[tuple[float,float]]  # [(0,80), (80,85), (85,90), (90,95), (95,100), (100,inf)]

    # Regime
    regime_posterior: list[float]          # [0.1, 0.6, 0.05, 0.2, 0.05]
    regime_names: list[str]               # ["continental", "sea_breeze", ...]
    dominant_regime: str                   # "sea_breeze"

    # GraphMamba hidden embedding (384-dim)
    mamba_embedding: list[float]          # Frozen, not to be modified by trader

    # KF state
    filtered_temp_f: float                # 87.3
    filtered_uncertainty: float           # 1.2
    running_max_f: float                  # 89.1
    running_min_f: float                  # 72.4
    predicted_max_f: float                # 91.2
    predicted_min_f: float                # 71.8

    # Conformal
    prediction_interval_90: tuple[float,float]  # (88.5, 93.1)
    prediction_interval_95: tuple[float,float]  # (87.2, 94.0)

    # Metadata
    hours_to_settlement: float            # 6.3
    n_obs_today: int                      # 47
    model_confidence: float               # 0.82 (based on ESS, regime stability)
```

This is the interface contract between the weather brain and the trading brain.

---

## Dependencies

```toml
[project]
dependencies = [
    "torch>=2.4",
    "scipy>=1.14",
    "torchdiffeq>=0.2.4",      # ODE solver for flow matching
    # mamba-ssm installed from source on CUDA machines
]

[project.optional-dependencies]
cuda = [
    "mamba-ssm>=2.3.1",        # Official Mamba-3 (CUDA only)
    "causal-conv1d>=1.4.0",
]
```
