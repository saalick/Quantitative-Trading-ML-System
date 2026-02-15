"""
Performance metrics for backtesting: Sharpe, Sortino, drawdown, win rate, etc.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional


def compute_metrics(
    returns: np.ndarray,
    trades: Optional[pd.DataFrame] = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> Dict[str, float]:
    """
    Compute standard backtest metrics.
    returns: array of period returns (e.g. daily).
    trades: optional DataFrame with columns 'pnl' or 'return', 'entry_time', 'exit_time'.
    """
    if len(returns) == 0:
        return _empty_metrics()

    returns = np.asarray(returns, dtype=float)
    excess = returns - risk_free_rate / periods_per_year
    total_return = np.prod(1 + returns) - 1
    n_periods = len(returns)
    years = n_periods / periods_per_year
    ann_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0
    vol = np.std(returns)
    ann_vol = vol * np.sqrt(periods_per_year) if vol > 0 else 0.0
    sharpe = (np.mean(excess) / vol * np.sqrt(periods_per_year)) if vol > 0 else 0.0
    downside = returns[returns < 0]
    downside_vol = np.std(downside) * np.sqrt(periods_per_year) if len(downside) > 0 else 0.0
    sortino = (ann_return - risk_free_rate) / downside_vol if downside_vol > 0 else 0.0

    # Max drawdown
    cum = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cum)
    drawdowns = (cum - running_max) / running_max
    max_dd = np.min(drawdowns)
    calmar = ann_return / abs(max_dd) if max_dd != 0 else 0.0

    # Trade-based (if trades provided)
    win_rate = np.nan
    avg_win = np.nan
    avg_loss = np.nan
    profit_factor = np.nan
    total_trades = 0
    avg_trade_duration = np.nan

    if trades is not None and len(trades) > 0:
        total_trades = len(trades)
        pnl = trades["pnl"] if "pnl" in trades.columns else trades.get("return", pd.Series([0] * len(trades)))
        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]
        win_rate = 100 * len(wins) / total_trades if total_trades > 0 else 0
        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = losses.mean() if len(losses) > 0 else 0
        gross_profit = wins.sum()
        gross_loss = abs(losses.sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0)
        if "entry_time" in trades.columns and "exit_time" in trades.columns:
            dur = (pd.to_datetime(trades["exit_time"]) - pd.to_datetime(trades["entry_time"])).dt.days
            avg_trade_duration = dur.mean()

    return {
        "total_return_pct": total_return * 100,
        "annualized_return_pct": ann_return * 100,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown_pct": max_dd * 100,
        "calmar_ratio": calmar,
        "win_rate_pct": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "total_trades": total_trades,
        "avg_trade_duration_days": avg_trade_duration if not np.isnan(avg_trade_duration) else None,
        "volatility_annual_pct": ann_vol * 100,
    }


def _empty_metrics() -> Dict[str, float]:
    return {
        "total_return_pct": 0,
        "annualized_return_pct": 0,
        "sharpe_ratio": 0,
        "sortino_ratio": 0,
        "max_drawdown_pct": 0,
        "calmar_ratio": 0,
        "win_rate_pct": 0,
        "avg_win": 0,
        "avg_loss": 0,
        "profit_factor": 0,
        "total_trades": 0,
        "avg_trade_duration_days": None,
        "volatility_annual_pct": 0,
    }
