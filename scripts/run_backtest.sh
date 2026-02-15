#!/bin/bash
# Run full pipeline: train model, run backtest, generate plots
set -e
cd "$(dirname "$0")/.."
echo "Training model..."
python src/training/train.py --data data/sample_data.csv --epochs 50 --batch-size 32 --save-dir results/models/
echo "Running backtest..."
python scripts/run_backtest.py
echo "Plots in results/figures/"
