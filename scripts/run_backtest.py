"""
Enhanced Risk-Adjusted Performance Metrics

Extends the backtesting metrics with institutional-grade risk measures:
- Sortino Ratio (downside-deviation adjusted returns)
- Calmar Ratio (return / max drawdown)
- Omega Ratio (probability-weighted gains vs losses)
- Tail Risk metrics (VaR, CVaR at 95% and 99%)
- Gain-to-Pain Ratio

These complement the existing Sharpe Ratio and Max Drawdown metrics
with measures commonly used in institutional portfolio evaluation.

References:
    - Sortino & Price (1994), "Performance Measurement in a Downside Risk Framework"
    - Keating & Shadwick (2002), "A Universal Performance Measure"
    - Rockafellar & Uryasev (2000), "Optimization of Conditional Value-at-Risk"
"""

import numpy as np
from typing import Dict, Optional


def compute_sortino_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    annualization_factor: int = 252,
) -> float:
    """
    Sortino Ratio: excess return / downside deviation.

    Unlike Sharpe, only penalizes negative volatility, making it
    more appropriate for asymmetric return distributions common
    in directional trading strategies.

    Args:
        returns: Array of periodic returns.
        risk_free_rate: Risk-free rate per period (default 0).
        annualization_factor: Trading days per year (default 252).

    Returns:
        Annualized Sortino Ratio.
    """
    if len(returns) < 2:
        return 0.0

    excess = returns - risk_free_rate
    downside = np.minimum(excess, 0.0)
    downside_std = np.sqrt(np.mean(downside ** 2))

    if downside_std < 1e-10:
        return 0.0

    return (np.mean(excess) / downside_std) * np.sqrt(annualization_factor)


def compute_calmar_ratio(
    returns: np.ndarray,
    annualization_factor: int = 252,
) -> float:
    """
    Calmar Ratio: annualized return / max drawdown.

    Measures how well the strategy compensates for its worst
    peak-to-trough decline. Values > 1.0 are generally considered good.

    Args:
        returns: Array of periodic returns.
        annualization_factor: Trading days per year.

    Returns:
        Calmar Ratio (positive value; 0 if max drawdown is ~0).
    """
    if len(returns) < 2:
        return 0.0

    cumulative = np.cumprod(1.0 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - running_max) / running_max
    max_dd = np.abs(np.min(drawdowns))

    if max_dd < 1e-10:
        return 0.0

    annualized_return = np.mean(returns) * annualization_factor
    return annualized_return / max_dd


def compute_omega_ratio(
    returns: np.ndarray,
    threshold: float = 0.0,
) -> float:
    """
    Omega Ratio: probability-weighted gains / probability-weighted losses.

    A more complete measure than Sharpe as it considers the entire
    return distribution, not just the first two moments.

    Args:
        returns: Array of periodic returns.
        threshold: Minimum acceptable return (default 0).

    Returns:
        Omega Ratio (values > 1 indicate net positive performance).
    """
    if len(returns) < 2:
        return 0.0

    excess = returns - threshold
    gains = np.sum(excess[excess > 0])
    losses = np.abs(np.sum(excess[excess <= 0]))

    if losses < 1e-10:
        return float("inf") if gains > 0 else 0.0

    return gains / losses


def compute_tail_risk(
    returns: np.ndarray,
    confidence_levels: tuple = (0.95, 0.99),
) -> Dict[str, float]:
    """
    Value-at-Risk (VaR) and Conditional VaR (Expected Shortfall).

    VaR: maximum expected loss at a given confidence level.
    CVaR: expected loss given that loss exceeds VaR (tail average).

    Args:
        returns: Array of periodic returns.
        confidence_levels: Tuple of confidence levels.

    Returns:
        Dictionary with VaR and CVaR at each confidence level.
    """
    results = {}
    for cl in confidence_levels:
        alpha = 1 - cl
        var = np.percentile(returns, alpha * 100)
        cvar = np.mean(returns[returns <= var]) if np.any(returns <= var) else var
        pct = int(cl * 100)
        results[f"VaR_{pct}"] = float(var)
        results[f"CVaR_{pct}"] = float(cvar)
    return results


def compute_gain_to_pain_ratio(returns: np.ndarray) -> float:
    """
    Gain-to-Pain Ratio: sum of all returns / sum of absolute negative returns.

    A simple but effective measure of whether cumulative profits
    justify the pain of drawdowns experienced along the way.

    Args:
        returns: Array of periodic returns.

    Returns:
        Gain-to-Pain ratio.
    """
    if len(returns) < 2:
        return 0.0

    total_return = np.sum(returns)
    total_pain = np.sum(np.abs(returns[returns < 0]))

    if total_pain < 1e-10:
        return float("inf") if total_return > 0 else 0.0

    return total_return / total_pain


def compute_enhanced_metrics(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    annualization_factor: int = 252,
) -> Dict[str, float]:
    """
    Compute all enhanced risk metrics in a single call.

    Designed to be called alongside existing backtest metrics.
    Returns a flat dictionary that can be merged with existing
    backtest_results.json output.

    Args:
        returns: Array of daily strategy returns.
        risk_free_rate: Daily risk-free rate.
        annualization_factor: Trading days per year.

    Returns:
        Dictionary containing all enhanced metrics.

    Example:
        >>> returns = np.array([0.01, -0.005, 0.008, -0.002, 0.003])
        >>> metrics = compute_enhanced_metrics(returns)
        >>> print(f"Sortino: {metrics['sortino_ratio']:.2f}")
    """
    metrics = {
        "sortino_ratio": compute_sortino_ratio(
            returns, risk_free_rate, annualization_factor
        ),
        "calmar_ratio": compute_calmar_ratio(returns, annualization_factor),
        "omega_ratio": compute_omega_ratio(returns),
        "gain_to_pain_ratio": compute_gain_to_pain_ratio(returns),
    }

    tail = compute_tail_risk(returns)
    metrics.update({k.lower(): v for k, v in tail.items()})

    return metrics
