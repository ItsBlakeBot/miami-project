"""DS3M v2: Deep Switching State Space Model — Production Inference.

Full-stack probabilistic inference for KMIA temperature bracket markets on Kalshi.

Architecture:
  Layer 1: Mamba3 selective SSM encoder (33-feature → 64-dim hidden state)
  Layer 2: Differentiable Particle Filter (500 particles, Gumbel-Softmax resample)
  Layer 3: Neural Spline Flow density estimator (conditional RQ-NSF → bracket P)
  Layer 4: HDP-Sticky regime discovery (nonparametric, auto-naming)
  Layer 5: Conformal calibration (distribution-free coverage guarantee)
  Layer 6: Bracket pricer with 5 microstructure signals + maker/taker routing
  Layer 7: Regime-conditioned paper trader (fractional Kelly, per-regime caps)

Inter-cycle updates:
  - Regime-conditioned scalar Kalman filter (per-regime Q/R, soft-blended)
  - Weather event detectors (sea breeze, convective outflow, overnight warm)

Training:
  - Multi-task pre-training on 255K IEM observations (10 SE Florida stations)
  - NSF calibration via CRPS + bracket Brier loss
  - Nightly fine-tuning after CLI settlement
"""
