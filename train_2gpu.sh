#!/bin/bash
set -e

echo "============================================================"
echo "  2x H200 Training Orchestrator"
echo "  Weather Brain: 243M params x 5 ensemble"
echo "  Trading Brain: 126M params"
echo "  Batch size: 64 | Mamba2 CUDA | BF16"
echo "============================================================"

BATCH=64
DB=miami_collector.db
OUT=trained_weights
mkdir -p $OUT

# Phase 1: Weather brains in parallel (2 GPUs)
echo ""
echo "=== PHASE 1: Weather Brain Training ==="
echo "  GPU 0: seeds 42, 137, 256 (3 models sequential)"
echo "  GPU 1: seeds 512, 1024 (2 models sequential)"

CUDA_VISIBLE_DEVICES=0 python train_h200.py \
    --phase weather-only \
    --db $DB \
    --seeds "42,137,256" \
    --use-muon \
    --no-compile \
    --batch-size $BATCH \
    --output-dir $OUT \
    2>&1 | tee ${OUT}/gpu0_weather.log &
PID0=$!

CUDA_VISIBLE_DEVICES=1 python train_h200.py \
    --phase weather-only \
    --db $DB \
    --seeds "512,1024" \
    --use-muon \
    --no-compile \
    --batch-size $BATCH \
    --output-dir $OUT \
    2>&1 | tee ${OUT}/gpu1_weather.log &
PID1=$!

echo "  Waiting for weather training..."
wait $PID0
EXIT0=$?
wait $PID1
EXIT1=$?

if [ $EXIT0 -ne 0 ] || [ $EXIT1 -ne 0 ]; then
    echo "  ERROR: Weather training failed (GPU0=$EXIT0, GPU1=$EXIT1)"
    echo "  Check logs: ${OUT}/gpu0_weather.log, ${OUT}/gpu1_weather.log"
    exit 1
fi

# Validate at least one checkpoint per seed exists
MISSING=0
for SEED in 42 137 256 512 1024; do
    if ! ls ${OUT}/weather_brain_seed${SEED}_*.pt 1>/dev/null 2>&1; then
        echo "  WARNING: No checkpoint found for seed $SEED"
        MISSING=$((MISSING + 1))
    fi
done

if [ $MISSING -eq 5 ]; then
    echo "  ERROR: No weather brain checkpoints found at all. Aborting."
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
