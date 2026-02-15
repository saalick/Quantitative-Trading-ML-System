"""
Data cleaning and preparation for the ML pipeline.
Handles missing values, scaling, and time-series train/val/test splits.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Literal, Optional, Tuple
from sklearn.preprocessing import StandardScaler


def drop_missing_or_fill(
    df: pd.DataFrame,
    strategy: Literal["drop", "forward_fill", "zero"] = "drop",
) -> pd.DataFrame:
    """
    Handle missing values in feature DataFrame.

    Args:
        df: DataFrame with possible NaNs (e.g. from rolling features).
        strategy: 'drop' removes rows with any NaN; 'forward_fill' then drop remainder; 'zero' fills with 0.

    Returns:
        Cleaned DataFrame.
    """
    if strategy == "drop":
        return df.dropna().copy()
    if strategy == "forward_fill":
        return df.ffill().dropna().copy()
    if strategy == "zero":
        return df.fillna(0).copy()
    raise ValueError(f"Unknown strategy: {strategy}")


def train_val_test_split_temporal(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Split time-series data chronologically (no shuffle).

    Ratios should sum to 1.0.

    Returns:
        (X_train, y_train), (X_val, y_val), (X_test, y_test)
    """
    n = len(X)
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    t1 = int(n * train_ratio)
    t2 = int(n * (train_ratio + val_ratio))
    return (
        (X[:t1], y[:t1]),
        (X[t1:t2], y[t1:t2]),
        (X[t2:], y[t2:]),
    )


def create_sequences(
    features: np.ndarray,
    targets: np.ndarray,
    sequence_length: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for LSTM: each sample is (sequence_length, n_features).

    Args:
        features: (n_timesteps, n_features).
        targets: (n_timesteps,) or (n_timesteps, 1). Target at time t aligns with last step of sequence.

    Returns:
        X: (n_samples, sequence_length, n_features), y: (n_samples,) or (n_samples, 1).
    """
    n = len(features)
    if n != len(targets):
        raise ValueError("features and targets length mismatch")
    n_samples = n - sequence_length
    if n_samples <= 0:
        raise ValueError("Not enough rows for sequence_length")

    X = np.zeros((n_samples, sequence_length, features.shape[1]), dtype=np.float32)
    for i in range(n_samples):
        X[i] = features[i : i + sequence_length]
    y = targets[sequence_length:]
    if y.ndim == 1:
        y = y.astype(np.float32)
    else:
        y = y.astype(np.float32)
    return X, y


def scale_features(
    X_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    X_test: Optional[np.ndarray] = None,
    scaler: Optional[StandardScaler] = None,
) -> Tuple[np.ndarray, ...]:
    """
    Scale features with StandardScaler fit on train.
    X_train is (n_samples, seq_len, n_features) or (n_samples, n_features).

    Returns:
        X_train_scaled, [X_val_scaled], [X_test_scaled], scaler
        (val/test only if provided).
    """
    orig_ndim = X_train.ndim
    if X_train.ndim == 3:
        n_samples, seq_len, n_f = X_train.shape
        X_flat = X_train.reshape(-1, n_f)
    else:
        X_flat = X_train

    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(X_flat)

    out = [scaler.transform(X_flat)]
    if orig_ndim == 3:
        out[0] = out[0].reshape(n_samples, seq_len, n_f)

    for X_other in (X_val, X_test):
        if X_other is None:
            continue
        if X_other.ndim == 3:
            n_s, s_l, n_f = X_other.shape
            flat = X_other.reshape(-1, n_f)
            scaled = scaler.transform(flat).reshape(n_s, s_l, n_f)
        else:
            scaled = scaler.transform(X_other)
        out.append(scaled)

    return (*out, scaler)
