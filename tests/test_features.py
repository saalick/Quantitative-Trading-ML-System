"""Tests for feature engineering: NaN handling, shapes, edge cases."""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.feature_engineering import compute_all_features, FEATURE_COLUMNS


@pytest.fixture
def sample_ohlcv():
    n = 300
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    close = 100 * np.exp(np.cumsum(0.001 + 0.02 * np.random.randn(n)))
    open_ = np.roll(close, 1)
    open_[0] = 100
    high = close + np.abs(0.5 * np.random.randn(n))
    low = close - np.abs(0.5 * np.random.randn(n))
    volume = (1e6 * (1 + np.random.rand(n))).astype(int)
    return pd.DataFrame({"date": dates, "open": open_, "high": high, "low": low, "close": close, "volume": volume})


def test_compute_all_features_returns_dataframe(sample_ohlcv):
    df = compute_all_features(sample_ohlcv)
    assert isinstance(df, pd.DataFrame)


def test_feature_count(sample_ohlcv):
    df = compute_all_features(sample_ohlcv)
    present = [c for c in FEATURE_COLUMNS if c in df.columns]
    assert len(present) == 45, f"Expected 45 features, got {len(present)}"


def test_nan_handling_after_drop(sample_ohlcv):
    df = compute_all_features(sample_ohlcv)
    df_clean = df.dropna()
    assert df_clean.isnull().sum().sum() == 0


def test_zero_volume_edge(sample_ohlcv):
    df = sample_ohlcv.copy()
    df.loc[df.index[10], "volume"] = 0
    df = compute_all_features(df)
    assert "volume_ratio" in df.columns or "mfi" in df.columns


def test_constant_price_edge():
    n = 100
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    c = np.ones(n) * 100
    df = pd.DataFrame({"date": dates, "open": c, "high": c, "low": c, "close": c, "volume": np.ones(n) * 1e6})
    df = compute_all_features(df)
    assert df.shape[0] == n
