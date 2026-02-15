#!/usr/bin/env python3


import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import json

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.data_loader import load_ohlcv
from src.feature_engineering import compute_all_features, FEATURE_COLUMNS
from src.data_preprocessing import drop_missing_or_fill, create_sequences, scale_features
from src.models.lstm_model import LSTMModel
from src.backtesting.backtester import Backtester
from src.backtesting.metrics import compute_metrics
from src.backtesting.visualization import generate_all_plots


def main():
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "sample_data.csv"
    model_path = project_root / "results" / "models" / "best_model.pt"
    figures_dir = project_root / "results" / "figures"
    metrics_dir = project_root / "results" / "metrics"

    df = load_ohlcv(data_path)
    df_feat = compute_all_features(df)
    df_feat = drop_missing_or_fill(df_feat, strategy="drop")
    feature_cols = [c for c in FEATURE_COLUMNS if c in df_feat.columns]
    X = df_feat[feature_cols].values.astype(np.float32)
    seq_len = 60
    X_seq, _ = create_sequences(X, np.zeros(len(X)), seq_len)
    res = scale_features(X_seq, None, None)
    X_scaled = res[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMModel(input_size=X_scaled.shape[2], hidden_size=128, num_layers=2, dropout=0.2)
    if model_path.exists():
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_scaled, dtype=torch.float32, device=device)).cpu().numpy().ravel()

    # Align with price index: predictions correspond to indices [seq_len, ..., n-1]
    n = len(df)
    signals_full = np.full(n, np.nan)
    signals_full[seq_len : seq_len + len(preds)] = preds
    signals_full = np.nan_to_num(signals_full, nan=0.5)
    prices = df["close"].values

    bt = Backtester(initial_capital=1_000_000, transaction_cost_pct=0.1, slippage_pct=0.05)
    equity_df, trade_log, metrics = bt.run(prices, signals_full, threshold=0.5)

    # Buy-and-hold for comparison
    bh = 1_000_000 * (prices / prices[0])
    equity_df["equity"] = equity_df["equity"].values

    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    dates = df["date"].iloc[: len(equity_df)].values if len(equity_df) <= len(df) else df["date"].values
    if len(dates) > len(equity_df):
        dates = dates[: len(equity_df)]
    generate_all_plots(
        equity_df,
        dates=pd.to_datetime(dates) if len(dates) else None,
        buy_and_hold=bh[: len(equity_df)] if len(bh) >= len(equity_df) else None,
        figures_dir=figures_dir,
    )

    with open(metrics_dir / "backtest_results.json", "w") as f:
        json.dump(metrics, f, indent=2)
    trade_log.to_csv(metrics_dir / "trade_log.csv", index=False)

    # Confusion matrix and ROC on test-range predictions (use same preds we already have)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix, roc_curve, auc
        # Recompute y_test from data for alignment with preds
        close = df["close"].values
        fwd_ret = np.roll(close, -1) / close - 1
        y_all = (fwd_ret >= 0).astype(np.float32)
        y_test = y_all[seq_len : seq_len + len(preds)]
        pred_bin = (preds >= 0.5).astype(int)
        cm = confusion_matrix(y_test, pred_bin)
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Down", "Up"])
        ax.set_yticklabels(["Down", "Up"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")
        plt.colorbar(im, ax=ax)
        ax.set_title("Confusion matrix")
        fig.savefig(figures_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        fpr, tpr, _ = roc_curve(y_test, preds)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        ax.plot([0, 1], [0, 1], "k--")
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.legend()
        ax.set_title("ROC curve")
        fig.savefig(figures_dir / "roc_curve.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print("Could not save confusion/ROC plots:", e)

    print("=== Backtesting Results ===")
    print(f"Total Return: {metrics['total_return_pct']:.2f}%")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
    print(f"Win Rate: {metrics['win_rate_pct']:.2f}%")
    print(f"Total Trades: {metrics['total_trades']}")
    print(f"\nPlots saved to {figures_dir}")
    print(f"Metrics saved to {metrics_dir / 'backtest_results.json'}")


if __name__ == "__main__":
    main()
