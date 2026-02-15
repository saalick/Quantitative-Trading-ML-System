"""
Backtest visualization: equity curve, drawdown, monthly heatmap, returns distribution.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional


def plot_equity_curve(
    equity_df: pd.DataFrame,
    dates: Optional[pd.DatetimeIndex] = None,
    buy_and_hold: Optional[np.ndarray] = None,
    save_path: Optional[Path] = None,
) -> None:
    """Cumulative equity and optional buy-and-hold comparison."""
    fig, ax = plt.subplots(figsize=(10, 5))
    x = dates if dates is not None else np.arange(len(equity_df))
    ax.plot(x, equity_df["equity"].values, label="Strategy", color="C0")
    if buy_and_hold is not None and len(buy_and_hold) == len(equity_df):
        ax.plot(x, buy_and_hold, label="Buy & Hold", color="C1", alpha=0.8)
    ax.set_xlabel("Time")
    ax.set_ylabel("Equity")
    ax.set_title("Equity Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_drawdown(
    equity_df: pd.DataFrame,
    dates: Optional[pd.DatetimeIndex] = None,
    save_path: Optional[Path] = None,
) -> None:
    """Underwater (drawdown) chart."""
    eq = equity_df["equity"].values
    running_max = np.maximum.accumulate(eq)
    drawdown = (eq - running_max) / running_max * 100
    fig, ax = plt.subplots(figsize=(10, 4))
    x = dates if dates is not None else np.arange(len(equity_df))
    ax.fill_between(x, drawdown, 0, color="red", alpha=0.3)
    ax.plot(x, drawdown, color="darkred")
    ax.set_xlabel("Time")
    ax.set_ylabel("Drawdown (%)")
    ax.set_title("Underwater (Drawdown) Chart")
    ax.grid(True, alpha=0.3)
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_monthly_returns_heatmap(
    returns: np.ndarray,
    dates: pd.DatetimeIndex,
    save_path: Optional[Path] = None,
) -> None:
    """Calendar heatmap of monthly returns."""
    s = pd.Series(returns, index=dates)
    monthly = s.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    pivot = monthly.groupby([monthly.index.year, monthly.index.month]).sum().unstack(level=0)
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(pivot.values * 100, aspect="auto", cmap="RdYlGn", vmin=-10, vmax=10)
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(12))
    ax.set_yticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    plt.colorbar(im, ax=ax, label="Return (%)")
    ax.set_title("Monthly Returns Heatmap")
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_returns_distribution(
    returns: np.ndarray,
    save_path: Optional[Path] = None,
) -> None:
    """Histogram of returns and optional normal fit."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(returns * 100, bins=50, density=True, alpha=0.7, color="steelblue", edgecolor="white")
    mu, sigma = returns.mean() * 100, returns.std() * 100
    x = np.linspace(returns.min() * 100, returns.max() * 100, 200)
    if sigma > 0:
        ax.plot(x, (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2), "r-", lw=2, label="Normal fit")
    ax.set_xlabel("Return (%)")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of Returns")
    ax.legend()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_all_plots(
    equity_df: pd.DataFrame,
    dates: Optional[pd.DatetimeIndex] = None,
    buy_and_hold: Optional[np.ndarray] = None,
    figures_dir: Optional[Path] = None,
) -> None:
    """Generate equity, drawdown, monthly heatmap, returns distribution."""
    figures_dir = Path(figures_dir) if figures_dir else Path("results/figures")
    returns = equity_df["return"].values
    if dates is None and "date" in equity_df.columns:
        dates = pd.to_datetime(equity_df["date"])
    elif dates is None:
        dates = pd.RangeIndex(len(equity_df))

    plot_equity_curve(equity_df, dates, buy_and_hold, figures_dir / "equity_curve.png")
    plot_drawdown(equity_df, dates, figures_dir / "drawdown_chart.png")
    if hasattr(dates, "year"):
        plot_monthly_returns_heatmap(returns, dates, figures_dir / "monthly_returns_heatmap.png")
    plot_returns_distribution(returns, figures_dir / "returns_distribution.png")
