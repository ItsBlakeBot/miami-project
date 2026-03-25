# DS3M v2 Migration — Status

**Date:** 2026-03-25
**Status:** Training in progress

---

## Completed

- [x] Mamba encoder (8L, d_model=256, d_state=32) — 3.6M params
- [x] Differentiable particle filter (500p, d_latent=32) + HDP regime discovery
- [x] Neural spline flow (10 transforms, 24 bins, hidden=128) — 784K params
- [x] 33-feature vector extraction from collector DB
- [x] Regime-conditioned scalar Kalman filter (per-regime Q/R)
- [x] Bracket KF bank (10 parallel KFs with innovation monitoring)
- [x] Paper trader v2 (regime sizing, maker/taker, Kelly)
- [x] Orchestrator v2 (main loop wiring all components)
- [x] Dashboard (FastAPI + SSE, dark theme)
- [x] HDP hardening (learnable α, merge-split, per-regime emission noise)
- [x] Training data: 269K ASOS + 28K atmosphere + 255K SST + 11.4K CLI targets
- [x] UTC time encoding throughout (training + live)
- [x] CLI 05Z-05Z climate day alignment
- [x] Data augmentation pipeline (jitter, noise, dropout, splicing)
- [x] Old engine deprecated and archived (30 files → _deprecated_v1/)
- [x] Old DS3M v1 deprecated and archived (10 files → _deprecated_v1/)
- [x] Plan docs cleaned up, outdated plans archived

## In Progress

- [ ] Full training (Phase 1: Mamba 50 epochs + Phase 2: NSF 100 epochs) — running on MPS
- [ ] Model verification and smoke test

## Remaining

- [ ] Wire paper_trader_v2 into live orchestrator loop
- [ ] Nightly retraining pipeline
- [ ] Conformal calibration from settlement accumulation
- [ ] Trading center bot (see TRADING_BOT_DESIGN.md)
- [ ] Multi-station expansion
- [ ] Live Kalshi execution (graduate from paper)

## Architecture

See DS3M_ARCHITECTURE.md for full technical details.
See SCHEMA.md for annotated data flow diagram.
See TRADING_BOT_DESIGN.md for trading intelligence layer design.
