# Deprecated v1 Engine Files

These files comprised the original production inference pipeline.
They have been replaced by DS3M v2 (`engine/ds3m/`) as of 2026-03-25.

**Do not import from this directory.** Kept for reference only.

## Replacement mapping

| Old file | Replaced by |
|----------|-------------|
| orchestrator.py | ds3m/orchestrator_v2.py |
| baseline_engine.py | ds3m/mamba_encoder.py |
| emos.py | ds3m/neural_spline_flow.py |
| boa.py | ds3m/neural_spline_flow.py |
| bracket_pricer.py | ds3m/bracket_pricer_v2.py |
| regime_catalog.py | ds3m/hdp_regime.py |
| regime_classifier.py | ds3m/hdp_regime.py |
| letkf.py | ds3m/diff_particle_filter.py |
| changepoint_detector.py | ds3m/realtime_updater.py |
| signal_engine.py | ds3m/bracket_pricer_v2.py (MicrostructureSignals) |
