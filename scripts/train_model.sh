#!/bin/bash
# Train the LSTM model with default config
set -e
cd "$(dirname "$0")/.."
python src/training/train.py \
    --data data/sample_data.csv \
    --config configs/model_config.yaml \
    --epochs 50 \
    --batch-size 32 \
    --save-dir results/models/
