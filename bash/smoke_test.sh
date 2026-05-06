#!/bin/bash
# 1-epoch smoke test: CRIB on ETTh1, point-missing 0.2.
# Verifies the pipeline end-to-end (data load -> model -> train -> eval -> csv write).

set -e
cd "$(dirname "$0")/.."

DATA_PATH="/home/ubuntu/work/CRIB-Rebutal/data"
OUT_DIR="./result/smoke"
mkdir -p "$OUT_DIR"

python train.py \
    --model CRIB \
    --dataset ETTh1 \
    --data_path "$DATA_PATH" \
    --train_epochs 1 \
    --batch_size 16 \
    --num_workers 0 \
    --seq_len 24 \
    --pred_len 24 \
    --model_dim 32 \
    --learning_rate 0.001 \
    --missing_pattern point \
    --missing_rate 0.2 \
    --seed 0 \
    --iter smoke \
    --exp_type Smoke \
    --csv_path "$OUT_DIR/smoke_result.csv"

echo "--- smoke test passed; results at $OUT_DIR/smoke_result.csv ---"
