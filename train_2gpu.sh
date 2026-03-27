#!/bin/bash
set -e

echo "============================================================"
echo "  2x H200 Training Orchestrator"
echo "  Weather Brain: 248M params x 5 ensemble"
echo "  Trading Brain: 128M params"
echo "  Batch size: 128 (fit test: first epoch will validate)"
echo "============================================================"

BATCH=128
DB=miami_collector.db
OUT=trained_weights
mkdir -p $OUT

# Phase 1: Weather brains in parallel (2 GPUs)
echo ""
echo "=== PHASE 1: Weather Brain Training ==="
echo "  GPU 0: seeds 42, 137, 256 (3 models)"
echo "  GPU 1: seeds 512, 1024 (2 models)"

CUDA_VISIBLE_DEVICES=0 python train_h200.py \
    --phase weather-only \
    --db $DB \
    --seeds "42,137,256" \
    --use-muon \
    --batch-size $BATCH \
    --output-dir $OUT \
    2>&1 | tee ${OUT}/gpu0_weather.log &
PID0=$!

CUDA_VISIBLE_DEVICES=1 python train_h200.py \
    --phase weather-only \
    --db $DB \
    --seeds "512,1024" \
    --use-muon \
    --batch-size $BATCH \
    --output-dir $OUT \
    2>&1 | tee ${OUT}/gpu1_weather.log &
PID1=$!

echo "  Waiting for weather training..."
wait $PID0 $PID1
echo "  Weather training COMPLETE"

# Phase 2: Backtest all 5 ensemble members
echo ""
echo "=== PHASE 2: Backtest (generate frozen DS3M signals) ==="
CUDA_VISIBLE_DEVICES=0 python train_h200.py \
    --phase backtest \
    --db $DB \
    --batch-size $BATCH \
    --output-dir $OUT \
    2>&1 | tee ${OUT}/backtest.log

echo "  Backtest COMPLETE — ds3m_frozen_signals.pt generated"

# Phase 3: Trading brain
echo ""
echo "=== PHASE 3: Trading Brain Training ==="
CUDA_VISIBLE_DEVICES=0 python train_h200.py \
    --phase trading-only \
    --db $DB \
    --use-muon \
    --batch-size $BATCH \
    --output-dir $OUT \
    2>&1 | tee ${OUT}/trading.log

echo ""
echo "============================================================"
echo "  ALL TRAINING COMPLETE"
echo "  Weights saved to: $OUT/"
echo "  Files:"
ls -la $OUT/*.pt 2>/dev/null || echo "  (no .pt files found)"
echo "============================================================"
