#!/bin/bash
set -e

echo "=== 4x H200 Training Orchestrator ==="
echo "Phase 1: Weather brains in parallel (GPU 0-2)"

CUDA_VISIBLE_DEVICES=0 python train_h200.py --phase weather-only --db miami_collector.db --seeds "42,137" --use-muon --batch-size 64 --output-dir trained_weights &
PID0=$!

CUDA_VISIBLE_DEVICES=1 python train_h200.py --phase weather-only --db miami_collector.db --seeds "256,512" --use-muon --batch-size 64 --output-dir trained_weights &
PID1=$!

CUDA_VISIBLE_DEVICES=2 python train_h200.py --phase weather-only --db miami_collector.db --seeds "1024" --use-muon --batch-size 64 --output-dir trained_weights &
PID2=$!

echo "Waiting for weather training to complete..."
wait $PID0 $PID1 $PID2
echo "Phase 1 complete!"

echo "Phase 2: Backtest (single GPU)"
CUDA_VISIBLE_DEVICES=0 python train_h200.py --phase backtest --db miami_collector.db --output-dir trained_weights

echo "Phase 3: Trading brain (single GPU)"
CUDA_VISIBLE_DEVICES=0 python train_h200.py --phase trading-only --db miami_collector.db --use-muon --batch-size 64 --output-dir trained_weights

echo "=== ALL TRAINING COMPLETE ==="
