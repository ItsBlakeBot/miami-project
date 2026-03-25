# Deprecated DS3M v1 (Shadow Mode) Files

These files comprised the original DS3M shadow-mode system.
Replaced by DS3M v2 production modules as of 2026-03-25.

## Replacement mapping

| Old file | Replaced by |
|----------|-------------|
| orchestrator.py | orchestrator_v2.py |
| paper_trader.py | paper_trader_v2.py |
| particle_filter.py | diff_particle_filter.py |
| regime_dynamics.py | hdp_regime.py |
| regime_discovery.py | hdp_regime.py |
| observation_model.py | diff_particle_filter.py (DifferentiableObservation) |
| density_estimator.py | neural_spline_flow.py |
| market_comparator.py | bracket_pricer_v2.py |
| state.py | diff_particle_filter.py (ParticleCloud) |
| trainer.py | training_pipeline.py |
