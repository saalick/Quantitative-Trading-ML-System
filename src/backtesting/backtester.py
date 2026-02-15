"""
Backtesting engine: transaction costs, slippage, position sizing, equity curve.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Callable, Optional, Tuple
from src.backtesting.metrics import compute_metrics


class Backtester:
    """
    Vectorized-style backtester: for each bar we have a signal (e.g. model prob).
    Entry/exit: long when signal > threshold, flat otherwise (or short if enabled).
    """

    def __init__(
        self,
        initial_capital: float = 1_000_000,
        transaction_cost_pct: float = 0.1,
        slippage_pct: float = 0.05,
        position_size_pct: float = 1.0,
        stop_loss_pct: Optional[float] = None,
        take_profit_pct: Optional[float] = None,
    ):
        self.initial_capital = initial_capital
        self.transaction_cost_pct = transaction_cost_pct / 100
        self.slippage_pct = slippage_pct / 100
        self.position_size_pct = position_size_pct
        self.stop_loss_pct = stop_loss_pct / 100 if stop_loss_pct is not None else None
        self.take_profit_pct = take_profit_pct / 100 if take_profit_pct is not None else None

    def run(
        self,
        prices: np.ndarray,
        signals: np.ndarray,
        threshold: float = 0.5,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
        """
        prices: (n,) close prices. signals: (n,) model probability or score.
        Returns: (equity_curve_df, trade_log_df, metrics_dict).
        """
        n = len(prices)
        if len(signals) != n:
            raise ValueError("prices and signals length mismatch")

        equity = self.initial_capital
        position = 0
        entry_price = 0.0
        entry_idx = 0
        equity_curve = [equity]
        trades = []
        cost = self.transaction_cost_pct + self.slippage_pct

        for i in range(1, n):
            signal = signals[i - 1]
            if np.isnan(signal):
                signal = 0.0
            want_long = 1 if signal >= threshold else 0

            if want_long != position:
                if position == 1:
                    equity *= 1 - cost
                    pnl = (prices[i] - entry_price) / entry_price if entry_price else 0
                    trades.append({
                        "entry_idx": entry_idx, "entry_price": entry_price, "entry_time": entry_idx,
                        "exit_idx": i, "exit_price": prices[i], "exit_time": i, "pnl": pnl,
                    })
                position = want_long
                if position == 1:
                    entry_price = prices[i] * (1 + self.slippage_pct)
                    entry_idx = i
                    equity *= 1 - cost

            if position == 1:
                ret_position = (prices[i] - prices[i - 1]) / prices[i - 1]
                equity *= 1 + ret_position * self.position_size_pct
                if self.stop_loss_pct and (prices[i] / entry_price - 1) <= -self.stop_loss_pct:
                    equity *= 1 - cost
                    trades.append({
                        "entry_idx": entry_idx, "entry_price": entry_price, "entry_time": entry_idx,
                        "exit_idx": i, "exit_price": prices[i], "exit_time": i, "pnl": -self.stop_loss_pct,
                    })
                    position = 0
                    entry_price = 0
                elif self.take_profit_pct and (prices[i] / entry_price - 1) >= self.take_profit_pct:
                    equity *= 1 - cost
                    trades.append({
                        "entry_idx": entry_idx, "entry_price": entry_price, "entry_time": entry_idx,
                        "exit_idx": i, "exit_price": prices[i], "exit_time": i, "pnl": self.take_profit_pct,
                    })
                    position = 0
                    entry_price = 0

            equity_curve.append(equity)

        if position == 1:
            equity *= 1 - cost
            trades.append({
                "entry_idx": entry_idx, "entry_price": entry_price, "entry_time": entry_idx,
                "exit_idx": n - 1, "exit_price": prices[-1], "exit_time": n - 1,
                "pnl": (prices[-1] - entry_price) / entry_price,
            })

        trade_log = pd.DataFrame(trades) if trades else pd.DataFrame(
            columns=["entry_idx", "entry_price", "exit_idx", "exit_price", "pnl", "entry_time", "exit_time"]
        )

        # Period returns from equity curve
        eq = np.array(equity_curve)
        period_returns = np.diff(eq) / eq[:-1]
        period_returns = np.nan_to_num(period_returns, nan=0.0)

        metrics = compute_metrics(period_returns, trade_log)

        equity_df = pd.DataFrame({
            "equity": equity_curve,
            "return": np.concatenate([[0], period_returns]),
        })
        return equity_df, trade_log, metrics
