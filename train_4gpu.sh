#!/bin/bash
set -e

echo "============================================================"
echo "  4x H200 Training Orchestrator"
echo "  Weather Brain: 243M params x 5 ensemble"
echo "  Trading Brain: 126M params"
echo "  Batch size: 64 | Mamba2 CUDA | BF16 | --no-compile"
echo "  Strategy: 1 seed per GPU (4 parallel + 1 sequential)"
echo "============================================================"

# Activate venv if available
source /venv/main/bin/activate 2>/dev/null || true

BATCH=64
DB=miami_collector.db
OUT=trained_weights
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p $OUT

# Phase 1: Weather brains — 4 GPUs, 4 seeds parallel, then 1 more
echo ""
echo "=== PHASE 1: Weather Brain Training ==="
echo "  GPU 0: seed 42"
echo "  GPU 1: seed 137"
echo "  GPU 2: seed 256"
echo "  GPU 3: seed 512"
echo "  Then GPU 0: seed 1024"

CUDA_VISIBLE_DEVICES=0 python train_h200.py \
    --phase weather-only \
    --db $DB \
    --seeds "42" \
    --use-muon \
    --no-compile \
    --batch-size $BATCH \
    --output-dir $OUT \
    2>&1 | tee ${OUT}/gpu0_weather.log &
PID0=$!

CUDA_VISIBLE_DEVICES=1 python train_h200.py \
    --phase weather-only \
    --db $DB \
    --seeds "137" \
    --use-muon \
    --no-compile \
    --batch-size $BATCH \
    --output-dir $OUT \
    2>&1 | tee ${OUT}/gpu1_weather.log &
PID1=$!

CUDA_VISIBLE_DEVICES=2 python train_h200.py \
    --phase weather-only \
    --db $DB \
    --seeds "256" \
    --use-muon \
    --no-compile \
    --batch-size $BATCH \
    --output-dir $OUT \
    2>&1 | tee ${OUT}/gpu2_weather.log &
PID2=$!

CUDA_VISIBLE_DEVICES=3 python train_h200.py \
    --phase weather-only \
    --db $DB \
    --seeds "512" \
    --use-muon \
    --no-compile \
    --batch-size $BATCH \
    --output-dir $OUT \
    2>&1 | tee ${OUT}/gpu3_weather.log &
PID3=$!

echo "  Waiting for first 4 seeds..."
wait $PID0; EXIT0=$?
wait $PID1; EXIT1=$?
wait $PID2; EXIT2=$?
wait $PID3; EXIT3=$?

if [ $EXIT0 -ne 0 ] || [ $EXIT1 -ne 0 ] || [ $EXIT2 -ne 0 ] || [ $EXIT3 -ne 0 ]; then
    echo "  ERROR: Weather training failed (GPU0=$EXIT0, GPU1=$EXIT1, GPU2=$EXIT2, GPU3=$EXIT3)"
    echo "  Check logs in ${OUT}/"
    exit 1
fi

echo "  First 4 seeds COMPLETE. Training seed 1024 on GPU 0..."

CUDA_VISIBLE_DEVICES=0 python train_h200.py \
    --phase weather-only \
    --db $DB \
    --seeds "1024" \
    --use-muon \
    --no-compile \
    --batch-size $BATCH \
    --output-dir $OUT \
    2>&1 | tee ${OUT}/gpu0_weather_1024.log
EXIT4=$?

if [ $EXIT4 -ne 0 ]; then
    echo "  ERROR: Seed 1024 failed"
    exit 1
fi

# Validate checkpoints
MISSING=0
for SEED in 42 137 256 512 1024; do
    if ! ls ${OUT}/weather_brain_seed${SEED}_*.pt 1>/dev/null 2>&1; then
        echo "  WARNING: No checkpoint found for seed $SEED"
        MISSING=$((MISSING + 1))
    fi
done

if [ $MISSING -eq 5 ]; then
    echo "  ERROR: No weather brain checkpoints found. Aborting."
    exit 1
fi

echo "  Weather training COMPLETE ($((5 - MISSING))/5 models checkpointed)"

# Phase 2: Backtest all 5 ensemble members
echo ""
echo "=== PHASE 2: Backtest (generate frozen DS3M signals) ==="
CUDA_VISIBLE_DEVICES=0 python train_h200.py \
    --phase backtest \
    --db $DB \
    --no-compile \
    --batch-size $BATCH \
    --output-dir $OUT \
    2>&1 | tee ${OUT}/backtest.log

if [ ! -f ${OUT}/ds3m_frozen_signals.pt ]; then
    echo "  ERROR: ds3m_frozen_signals.pt not generated. Aborting."
    exit 1
fi

echo "  Backtest COMPLETE — ds3m_frozen_signals.pt generated"

# Phase 3: Trading brain
echo ""
echo "=== PHASE 3: Trading Brain Training ==="
CUDA_VISIBLE_DEVICES=0 python train_h200.py \
    --phase trading-only \
    --db $DB \
    --use-muon \
    --no-compile \
    --batch-size $BATCH \
    --output-dir $OUT \
    2>&1 | tee ${OUT}/trading.log

echo ""
echo "============================================================"
echo "  ALL TRAINING COMPLETE"
echo "  Weights saved to: $OUT/"
echo "  Files:"
ls -lh $OUT/*.pt 2>/dev/null || echo "  (no .pt files found)"
echo "============================================================"
