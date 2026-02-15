"""Tests for backtester: trade logic, metrics, edge cases."""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backtesting.backtester import Backtester
from src.backtesting.metrics import compute_metrics


def test_backtester_all_long():
    n = 100
    prices = 100 * np.ones(n)
    prices[1:] = 100 + np.cumsum(np.ones(n - 1) * 0.5)  # always up
    signals = np.ones(n) * 0.9  # always long
    bt = Backtester(initial_capital=1000, transaction_cost_pct=0, slippage_pct=0)
    equity_df, trade_log, metrics = bt.run(prices, signals, threshold=0.5)
    assert len(equity_df) == n
    assert metrics["total_return_pct"] > 0


def test_backtester_no_trades():
    n = 50
    prices = np.linspace(100, 101, n)
    signals = np.zeros(n)  # never long
    bt = Backtester(initial_capital=1000, transaction_cost_pct=0, slippage_pct=0)
    equity_df, trade_log, metrics = bt.run(prices, signals, threshold=0.5)
    assert metrics["total_trades"] == 0
    assert equity_df["equity"].iloc[-1] == 1000


def test_metrics_all_wins():
    returns = np.ones(252) * 0.001
    metrics = compute_metrics(returns)
    assert metrics["sharpe_ratio"] > 0
    assert metrics["total_return_pct"] > 0


def test_metrics_all_losses():
    returns = np.ones(252) * -0.001
    metrics = compute_metrics(returns)
    assert metrics["total_return_pct"] < 0


def test_metrics_with_trades():
    returns = np.random.randn(100) * 0.01
    trades = pd.DataFrame({"pnl": [0.01, -0.005, 0.02], "entry_time": [0, 10, 20], "exit_time": [5, 15, 25]})
    metrics = compute_metrics(returns, trades=trades)
    assert "win_rate_pct" in metrics
    assert metrics["total_trades"] == 3
