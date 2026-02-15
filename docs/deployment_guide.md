# Deployment Guide

## Prerequisites

- Python 3.8+
- Install: `pip install -r requirements.txt`

## Training

1. **Data:** Place OHLCV CSV in `data/` or use `data/sample_data.csv`. Optionally download with:
   ```bash
   python scripts/download_data.py --ticker SPY --start 2020-01-01 --end 2024-12-31 --output data/ohlcv.csv
   ```
2. **Train:**
   ```bash
   python src/training/train.py --data data/sample_data.csv --config configs/model_config.yaml --epochs 50 --batch-size 32 --save-dir results/models/
   ```
3. Checkpoint is saved to `results/models/best_model.pt`.

## Backtest

```bash
python scripts/run_backtest.py
```

Uses `results/models/best_model.pt` and `data/sample_data.csv`, writes:
- `results/figures/` (equity curve, drawdown, monthly heatmap, returns distribution)
- `results/metrics/backtest_results.json`, `results/metrics/trade_log.csv`

## Inference (production-style)

1. Load model and scaler (if you saved it during training).
2. For each new bar: append OHLCV, compute 45 features, take last `sequence_length` rows, scale, run model forward, threshold output (e.g. ≥ 0.5 for long).
3. Use signal for execution (with your own risk and execution layer).

## Logging and errors

- Training uses tqdm and print. For file logging, configure `src.utils.logger.get_logger(name, log_file="logs/train.log")`.
- Ensure paths are correct when running from different working directories (use `Path(__file__).resolve().parent.parent` or set `PYTHONPATH` to project root).

## Optional: Docker

Example Dockerfile (minimal):

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "src/training/train.py", "--data", "data/sample_data.csv"]
```

Build: `docker build -t qt-ml .`  
Run: `docker run -v $(pwd)/results:/app/results qt-ml`
