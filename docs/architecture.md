# System Architecture

## Overview

The quantitative trading ML system consists of: data loading and validation, feature engineering (45 features), LSTM model training, and a backtesting engine with metrics and visualization.

## Components

### Data Flow

1. **Raw OHLCV** (CSV) → `data_loader.load_ohlcv()` → validated DataFrame
2. **DataFrame** → `feature_engineering.compute_all_features()` → 45 feature columns
3. **Features + target** → `data_preprocessing.create_sequences()` → (X, y) sequences for LSTM
4. **Sequences** → `scale_features()` → normalized inputs
5. **Train/Val/Test** → `Trainer.train_batched()` → trained model checkpoint
6. **Model + prices** → `Backtester.run()` → equity curve, trade log, metrics
7. **Results** → `visualization.generate_all_plots()` → figures and JSON metrics

### Model Architecture

- **Input**: `(batch, sequence_length=60, num_features=45)`
- **LSTM**: 2 layers, 128 hidden units each, dropout 0.2
- **Dense**: 64 units, ReLU, dropout 0.3
- **Output**: 1 unit, sigmoid (direction probability)
- **Training**: SGD (momentum 0.9), BCE loss, cosine annealing LR, gradient clipping, early stopping

### Feature Engineering Pipeline

- Returns (10): 1d–60d returns, log return, relative return, cumulative returns, momentum
- Technical (15): SMA 5/10/20/50/200, EMA 12/26, RSI, MACD + signal + hist, Stochastic, Williams %R, price vs SMA20
- Volatility (8): historical vol 10/20/30d, Bollinger Bands (upper/middle/lower/width), ATR
- Volume (6): volume SMA 10/30, volume ratio, OBV (normalized), volume ROC, MFI
- Pattern (6): higher highs, lower lows, trend strength, price momentum 5/10, ROC 5

### Backtesting Methodology

- Long-only; position when model probability ≥ threshold (default 0.5)
- Transaction cost and slippage applied on entry/exit
- Metrics: total return, annualized return, Sharpe, Sortino, max drawdown, Calmar, win rate, profit factor, trade count

### Deployment Considerations

- Run training offline; save `best_model.pt` and optional `scaler` for inference
- Inference: load model, compute features on latest window, run forward pass, threshold output for signal
- For production: add API layer, schedule daily retrain or rolling validation
