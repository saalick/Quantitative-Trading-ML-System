"""
Main training script: load data, preprocess, train LSTM, evaluate.
Usage:
  python src/training/train.py --data data/sample_data.csv --config configs/model_config.yaml --epochs 50 --batch-size 32 --save-dir results/models/
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from data.data_loader import load_ohlcv
from src.feature_engineering import compute_all_features, FEATURE_COLUMNS
from src.data_preprocessing import (
    drop_missing_or_fill,
    create_sequences,
    train_val_test_split_temporal,
    scale_features,
)
from src.models.lstm_model import LSTMModel
from src.training.trainer import Trainer
from src.utils.helpers import set_seed, ensure_dir


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="data/sample_data.csv", help="Path to OHLCV CSV")
    p.add_argument("--config", type=str, default="configs/model_config.yaml", help="Model config YAML")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--save-dir", type=str, default="results/models")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_config(path: str) -> dict:
    path = Path(path)
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def main():
    args = parse_args()
    set_seed(args.seed)
    project_root = Path(__file__).resolve().parent.parent.parent
    data_path = project_root / args.data
    config_path = project_root / args.config
    config = load_config(config_path)
    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})
    data_cfg = config.get("data", {})

    seq_len = data_cfg.get("sequence_length", 60)
    train_ratio = data_cfg.get("train_split", 0.7)
    val_ratio = data_cfg.get("val_split", 0.15)
    test_ratio = data_cfg.get("test_split", 0.15)

    print("=== Loading data ===")
    df = load_ohlcv(data_path)
    print("=== Computing features ===")
    df = compute_all_features(df)
    df = drop_missing_or_fill(df, strategy=data_cfg.get("missing_value_strategy", "drop"))

    feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    if len(feature_cols) < 45:
        print(f"Warning: using {len(feature_cols)} features (expected 45)")
    X = df[feature_cols].values.astype(np.float32)
    # Target: next-day return direction (1 if close goes up, 0 else)
    close = load_ohlcv(data_path)["close"].reindex(df.index).ffill().bfill()
    fwd_ret = close.shift(-1) / close - 1
    y = (fwd_ret >= 0).astype(np.float32).values
    # Align length
    valid = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X = X[valid]
    y = y[valid]

    print("=== Creating sequences ===")
    X_seq, y_seq = create_sequences(X, y, seq_len)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = train_val_test_split_temporal(
        X_seq, y_seq, train_ratio, val_ratio, test_ratio
    )

    print("=== Scaling ===")
    X_train_s, X_val_s, X_test_s, scaler = scale_features(X_train, X_val, X_test)

    input_size = X_train_s.shape[2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMModel(
        input_size=input_size,
        hidden_size=model_cfg.get("hidden_size", 128),
        num_layers=model_cfg.get("num_layers", 2),
        dropout=model_cfg.get("dropout", 0.2),
        dense_units=64,
        output_size=1,
    )
    print(model.summary())

    save_dir = project_root / args.save_dir
    ensure_dir(save_dir)
    trainer = Trainer(
        model=model,
        device=device,
        learning_rate=training_cfg.get("learning_rate", 0.001),
        momentum=training_cfg.get("momentum", 0.9),
        weight_decay=training_cfg.get("weight_decay", 0.0001),
        max_epochs=args.epochs,
        early_stopping_patience=training_cfg.get("early_stopping_patience", 10),
        save_dir=save_dir,
        grad_clip=1.0,
    )

    print("=== Training LSTM Model ===")
    train_losses, train_accs, val_losses, val_accs = trainer.train_batched(
        X_train_s, y_train, X_val_s, y_val, batch_size=args.batch_size
    )
    for epoch in range(len(train_losses)):
        print(f"Epoch {epoch+1}/{len(train_losses)}: train_loss={train_losses[epoch]:.4f}, val_loss={val_losses[epoch]:.4f}, val_acc={val_accs[epoch]:.4f}")
    print("Early stopping triggered!" if len(train_losses) < args.epochs else "Training completed.")
    print(f"Best model saved to: {save_dir / 'best_model.pt'}")

    print("=== Final evaluation on test set ===")
    test_loss, test_acc = trainer.evaluate(X_test_s, y_test, batch_size=args.batch_size)
    print(f"Test loss: {test_loss:.4f}, Test accuracy (directional): {test_acc:.4f}")

    # Save loss curves
    fig_dir = project_root / "results" / "figures"
    if fig_dir.exists() or save_dir.exists():
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot(train_losses, label="Train")
        ax[0].plot(val_losses, label="Val")
        ax[0].set_xlabel("Epoch")
        ax[0].set_ylabel("Loss")
        ax[0].legend()
        ax[0].set_title("Loss curves")
        ax[1].plot(train_accs, label="Train")
        ax[1].plot(val_accs, label="Val")
        ax[1].set_xlabel("Epoch")
        ax[1].set_ylabel("Accuracy")
        ax[1].legend()
        ax[1].set_title("Accuracy curves")
        fig.savefig(fig_dir / "loss_curves.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    return 0


if __name__ == "__main__":
    main()
