# API Reference

## Data

### `data.data_loader.load_ohlcv(path, date_column="date", parse_dates=True)`
Load OHLCV CSV. Returns DataFrame with columns `date`, `open`, `high`, `low`, `close`, `volume`. Raises if required columns are missing.

### `data.data_loader.validate_ohlcv(df)`
Returns `(is_valid, list of error strings)`.

---

## Feature Engineering

### `src.feature_engineering.compute_all_features(df)`
Computes all 45 features. `df` must have `open`, `high`, `low`, `close`, `volume`. Returns DataFrame with `date` (if present) and feature columns. NaNs may remain from rolling windows.

### `src.feature_engineering.FEATURE_COLUMNS`
List of 45 feature column names.

---

## Preprocessing

### `src.data_preprocessing.drop_missing_or_fill(df, strategy="drop")`
`strategy`: `"drop"` | `"forward_fill"` | `"zero"`. Returns cleaned DataFrame.

### `src.data_preprocessing.create_sequences(features, targets, sequence_length)`
Returns `(X, y)` with `X` shape `(n_samples, sequence_length, n_features)` and `y` aligned to last time step.

### `src.data_preprocessing.train_val_test_split_temporal(X, y, train_ratio, val_ratio, test_ratio)`
Returns `(X_train, y_train), (X_val, y_val), (X_test, y_test)` in time order.

### `src.data_preprocessing.scale_features(X_train, X_val=None, X_test=None, scaler=None)`
Returns `(X_train_scaled, [X_val_scaled], [X_test_scaled], scaler)`.

---

## Models

### `src.models.lstm_model.LSTMModel(input_size=45, hidden_size=128, num_layers=2, dropout=0.2, dense_units=64, output_size=1)`
PyTorch Module. `forward(x)` expects `x` shape `(batch, seq_len, input_size)`; returns `(batch, 1)` in [0, 1].

### `src.models.baseline.BaselineModel(kind="logistic"|"random_forest", **kwargs)`
`.fit(X, y)`, `.predict_proba(X)`, `.predict(X)`. For sequences, use last time step (e.g. `X[:, -1, :]`).

### `src.models.model_utils.init_weights(module)`, `count_parameters(model)`, `save_checkpoint(...)`, `load_checkpoint(...)`
Standard model utilities.

---

## Backtesting

### `src.backtesting.backtester.Backtester(initial_capital=1e6, transaction_cost_pct=0.1, slippage_pct=0.05, position_size_pct=1.0, stop_loss_pct=None, take_profit_pct=None)`
`.run(prices, signals, threshold=0.5)` returns `(equity_df, trade_log_df, metrics_dict)`.

### `src.backtesting.metrics.compute_metrics(returns, trades=None, risk_free_rate=0, periods_per_year=252)`
Returns dict with `total_return_pct`, `sharpe_ratio`, `max_drawdown_pct`, `win_rate_pct`, `total_trades`, etc.

### `src.backtesting.visualization.generate_all_plots(equity_df, dates=None, buy_and_hold=None, figures_dir=None)`
Writes equity curve, drawdown, monthly heatmap, returns distribution to `figures_dir`.

---

## Training

### `src.training.trainer.Trainer(model, device, learning_rate=0.001, momentum=0.9, weight_decay=0.0001, max_epochs=50, early_stopping_patience=10, save_dir=None, grad_clip=1.0)`
`.train_batched(X_train, y_train, X_val, y_val, batch_size=32)` returns `(train_losses, train_accs, val_losses, val_accs)`.
`.evaluate(X, y, batch_size=32)` returns `(loss, accuracy)`.

### `src.training.train` (script)
CLI: `--data`, `--config`, `--epochs`, `--batch-size`, `--save-dir`, `--seed`.
