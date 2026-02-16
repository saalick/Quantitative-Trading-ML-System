"""
Backtesting engine: transaction costs, slippage, position sizing, equity curve.
Enhanced with advanced risk metrics and dynamic position sizing strategies.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Callable, Optional, Tuple, Dict
from src.backtesting.metrics import compute_metrics


class Backtester:
    """
    Vectorized-style backtester: for each bar we have a signal (e.g. model prob).
    Entry/exit: long when signal > threshold, flat otherwise (or short if enabled).
    
    Enhanced features:
    - Dynamic position sizing (Kelly Criterion, volatility-based, confidence-based)
    - Advanced risk metrics (Sortino, Calmar, Omega, VaR, CVaR)
    - Trailing stops and better risk management
    - Per-trade analytics and drawdown tracking
    """

    def __init__(
        self,
        initial_capital: float = 1_000_000,
        transaction_cost_pct: float = 0.1,
        slippage_pct: float = 0.05,
        position_size_pct: float = 1.0,
        position_sizing_method: str = 'fixed',  # 'fixed', 'kelly', 'volatility', 'confidence'
        max_position_size: float = 1.0,  # Maximum position as fraction of equity
        stop_loss_pct: Optional[float] = None,
        take_profit_pct: Optional[float] = None,
        trailing_stop_pct: Optional[float] = None,  # New: trailing stop
        max_drawdown_limit: Optional[float] = None,  # New: circuit breaker
        risk_free_rate: float = 0.02,  # For Sharpe/Sortino calculations
    ):
        """
        Initialize backtester with enhanced risk management.
        
        Args:
            initial_capital: Starting capital
            transaction_cost_pct: Transaction cost as percentage
            slippage_pct: Slippage as percentage
            position_size_pct: Base position size (0-1)
            position_sizing_method: Method for position sizing
                - 'fixed': Constant position size
                - 'kelly': Kelly Criterion based on win rate and avg returns
                - 'volatility': Size inversely proportional to volatility
                - 'confidence': Size proportional to signal strength
            max_position_size: Maximum allowed position size (risk limit)
            stop_loss_pct: Hard stop loss percentage
            take_profit_pct: Take profit target percentage
            trailing_stop_pct: Trailing stop percentage (dynamic)
            max_drawdown_limit: Maximum allowed drawdown before stopping
            risk_free_rate: Annual risk-free rate for metrics
        """
        self.initial_capital = initial_capital
        self.transaction_cost_pct = transaction_cost_pct / 100
        self.slippage_pct = slippage_pct / 100
        self.position_size_pct = position_size_pct
        self.position_sizing_method = position_sizing_method
        self.max_position_size = max_position_size
        self.stop_loss_pct = stop_loss_pct / 100 if stop_loss_pct is not None else None
        self.take_profit_pct = take_profit_pct / 100 if take_profit_pct is not None else None
        self.trailing_stop_pct = trailing_stop_pct / 100 if trailing_stop_pct is not None else None
        self.max_drawdown_limit = max_drawdown_limit / 100 if max_drawdown_limit is not None else None
        self.risk_free_rate = risk_free_rate
        
        # Track historical performance for Kelly Criterion
        self.historical_trades = []

    def _calculate_position_size(
        self, 
        signal: float, 
        current_equity: float,
        recent_volatility: float,
        prices: np.ndarray,
        current_idx: int,
    ) -> float:
        """
        Calculate position size based on selected method.
        
        Returns: Position size as fraction of equity (0-1)
        """
        if self.position_sizing_method == 'fixed':
            return min(self.position_size_pct, self.max_position_size)
        
        elif self.position_sizing_method == 'kelly':
            # Kelly Criterion: f* = (p*b - q) / b
            # where p = win rate, q = loss rate, b = avg_win/avg_loss
            if len(self.historical_trades) < 10:
                return min(self.position_size_pct * 0.5, self.max_position_size)
            
            trades_df = pd.DataFrame(self.historical_trades)
            wins = trades_df[trades_df['pnl'] > 0]
            losses = trades_df[trades_df['pnl'] < 0]
            
            if len(wins) == 0 or len(losses) == 0:
                return min(self.position_size_pct * 0.5, self.max_position_size)
            
            win_rate = len(wins) / len(trades_df)
            avg_win = wins['pnl'].mean()
            avg_loss = abs(losses['pnl'].mean())
            
            if avg_loss == 0:
                kelly_fraction = 0.5
            else:
                b = avg_win / avg_loss
                kelly_fraction = (win_rate * b - (1 - win_rate)) / b
                # Use fractional Kelly (e.g., 0.5x Kelly) for safety
                kelly_fraction = max(0, min(kelly_fraction * 0.5, self.max_position_size))
            
            return kelly_fraction
        
        elif self.position_sizing_method == 'volatility':
            # Inverse volatility: size inversely proportional to recent volatility
            # Target risk: constant volatility of returns
            if recent_volatility == 0 or np.isnan(recent_volatility):
                return min(self.position_size_pct, self.max_position_size)
            
            # Target vol of 2% per day, scale position accordingly
            target_vol = 0.02
            vol_scaling = target_vol / recent_volatility
            position_size = self.position_size_pct * vol_scaling
            return min(position_size, self.max_position_size)
        
        elif self.position_sizing_method == 'confidence':
            # Scale position by signal confidence (distance from threshold)
            # Signal 0.5 = 0% position, Signal 1.0 = 100% position
            confidence = abs(signal - 0.5) * 2  # Scale 0-1
            position_size = self.position_size_pct * confidence
            return min(position_size, self.max_position_size)
        
        else:
            return min(self.position_size_pct, self.max_position_size)

    def _calculate_recent_volatility(self, prices: np.ndarray, current_idx: int, window: int = 20) -> float:
        """Calculate recent volatility for position sizing."""
        if current_idx < window:
            window = current_idx
        if window < 2:
            return 0.0
        
        recent_prices = prices[max(0, current_idx - window):current_idx + 1]
        returns = np.diff(recent_prices) / recent_prices[:-1]
        return np.std(returns) if len(returns) > 0 else 0.0

    def run(
        self,
        prices: np.ndarray,
        signals: np.ndarray,
        threshold: float = 0.5,
        dates: Optional[np.ndarray] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
        """
        Run backtest with enhanced risk management and position sizing.
        
        Args:
            prices: (n,) close prices
            signals: (n,) model probability or score
            threshold: Signal threshold for entry
            dates: Optional date array for time-based analysis
            
        Returns:
            (equity_curve_df, trade_log_df, metrics_dict)
        """
        n = len(prices)
        if len(signals) != n:
            raise ValueError("prices and signals length mismatch")

        # Initialize tracking variables
        equity = self.initial_capital
        position = 0
        position_size = 0.0  # Actual position size being used
        entry_price = 0.0
        entry_idx = 0
        highest_price = 0.0  # For trailing stop
        peak_equity = self.initial_capital  # For drawdown tracking
        
        equity_curve = [equity]
        trades = []
        cost = self.transaction_cost_pct + self.slippage_pct
        
        # Enhanced tracking
        daily_position_sizes = [0.0]
        daily_volatilities = [0.0]
        
        for i in range(1, n):
            signal = signals[i - 1]
            if np.isnan(signal):
                signal = 0.0
            
            # Check drawdown circuit breaker
            current_drawdown = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0
            if self.max_drawdown_limit and current_drawdown >= self.max_drawdown_limit:
                # Force exit all positions if drawdown limit hit
                if position == 1:
                    equity *= (1 - cost)
                    pnl = (prices[i] - entry_price) / entry_price if entry_price else 0
                    trade_record = {
                        "entry_idx": entry_idx, 
                        "entry_price": entry_price, 
                        "entry_time": entry_idx,
                        "exit_idx": i, 
                        "exit_price": prices[i], 
                        "exit_time": i, 
                        "pnl": pnl,
                        "exit_reason": "max_drawdown_limit",
                        "position_size": position_size,
                    }
                    trades.append(trade_record)
                    self.historical_trades.append(trade_record)
                    position = 0
                    position_size = 0.0
                # Stop trading
                equity_curve.append(equity)
                daily_position_sizes.append(0.0)
                daily_volatilities.append(self._calculate_recent_volatility(prices, i))
                continue
            
            # Update peak equity for drawdown tracking
            if equity > peak_equity:
                peak_equity = equity
            
            want_long = 1 if signal >= threshold else 0
            
            # Calculate recent volatility for position sizing
            recent_vol = self._calculate_recent_volatility(prices, i)
            
            # Entry/Exit logic with dynamic position sizing
            if want_long != position:
                # Exit existing position
                if position == 1:
                    equity *= (1 - cost)
                    pnl = (prices[i] - entry_price) / entry_price if entry_price else 0
                    trade_record = {
                        "entry_idx": entry_idx, 
                        "entry_price": entry_price, 
                        "entry_time": entry_idx,
                        "exit_idx": i, 
                        "exit_price": prices[i], 
                        "exit_time": i, 
                        "pnl": pnl,
                        "exit_reason": "signal",
                        "position_size": position_size,
                    }
                    trades.append(trade_record)
                    self.historical_trades.append(trade_record)
                
                position = want_long
                
                # Enter new position with dynamic sizing
                if position == 1:
                    entry_price = prices[i] * (1 + self.slippage_pct)
                    entry_idx = i
                    highest_price = entry_price
                    equity *= (1 - cost)
                    
                    # Calculate dynamic position size
                    position_size = self._calculate_position_size(
                        signal, equity, recent_vol, prices, i
                    )
                else:
                    position_size = 0.0

            # Position P&L with dynamic sizing
            if position == 1:
                ret_position = (prices[i] - prices[i - 1]) / prices[i - 1]
                equity *= (1 + ret_position * position_size)
                
                # Update highest price for trailing stop
                if prices[i] > highest_price:
                    highest_price = prices[i]
                
                # Check stops
                exit_triggered = False
                exit_reason = None
                
                # Hard stop loss
                if self.stop_loss_pct and (prices[i] / entry_price - 1) <= -self.stop_loss_pct:
                    exit_triggered = True
                    exit_reason = "stop_loss"
                    final_pnl = -self.stop_loss_pct
                
                # Take profit
                elif self.take_profit_pct and (prices[i] / entry_price - 1) >= self.take_profit_pct:
                    exit_triggered = True
                    exit_reason = "take_profit"
                    final_pnl = self.take_profit_pct
                
                # Trailing stop
                elif self.trailing_stop_pct and (prices[i] / highest_price - 1) <= -self.trailing_stop_pct:
                    exit_triggered = True
                    exit_reason = "trailing_stop"
                    final_pnl = (prices[i] - entry_price) / entry_price
                
                if exit_triggered:
                    equity *= (1 - cost)
                    trade_record = {
                        "entry_idx": entry_idx, 
                        "entry_price": entry_price, 
                        "entry_time": entry_idx,
                        "exit_idx": i, 
                        "exit_price": prices[i], 
                        "exit_time": i, 
                        "pnl": final_pnl,
                        "exit_reason": exit_reason,
                        "position_size": position_size,
                    }
                    trades.append(trade_record)
                    self.historical_trades.append(trade_record)
                    position = 0
                    position_size = 0.0
                    entry_price = 0

            equity_curve.append(equity)
            daily_position_sizes.append(position_size if position == 1 else 0.0)
            daily_volatilities.append(recent_vol)

        # Close any remaining position
        if position == 1:
            equity *= (1 - cost)
            trade_record = {
                "entry_idx": entry_idx, 
                "entry_price": entry_price, 
                "entry_time": entry_idx,
                "exit_idx": n - 1, 
                "exit_price": prices[-1], 
                "exit_time": n - 1,
                "pnl": (prices[-1] - entry_price) / entry_price,
                "exit_reason": "end_of_data",
                "position_size": position_size,
            }
            trades.append(trade_record)
            self.historical_trades.append(trade_record)

        # Create trade log with enhanced information
        trade_log = pd.DataFrame(trades) if trades else pd.DataFrame(
            columns=["entry_idx", "entry_price", "exit_idx", "exit_price", 
                    "pnl", "entry_time", "exit_time", "exit_reason", "position_size"]
        )

        # Calculate returns
        eq = np.array(equity_curve)
        period_returns = np.diff(eq) / eq[:-1]
        period_returns = np.nan_to_num(period_returns, nan=0.0)

        # Compute standard metrics
        metrics = compute_metrics(period_returns, trade_log)
        
        # Add enhanced risk metrics
        enhanced_metrics = self._compute_enhanced_metrics(
            period_returns, eq, trade_log, prices
        )
        metrics.update(enhanced_metrics)

        # Create enhanced equity dataframe
        equity_df = pd.DataFrame({
            "equity": equity_curve,
            "return": np.concatenate([[0], period_returns]),
            "position_size": daily_position_sizes,
            "volatility": daily_volatilities,
        })
        
        # Add dates if provided
        if dates is not None and len(dates) == n:
            equity_df['date'] = dates
        
        return equity_df, trade_log, metrics

    def _compute_enhanced_metrics(
        self, 
        returns: np.ndarray, 
        equity_curve: np.ndarray,
        trade_log: pd.DataFrame,
        prices: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute advanced risk-adjusted performance metrics.
        
        Returns metrics including:
        - Sortino Ratio (downside deviation)
        - Calmar Ratio (return/max drawdown)
        - Omega Ratio (probability-weighted gains/losses)
        - Value at Risk (95% and 99%)
        - Conditional VaR (Expected Shortfall)
        - Tail Ratio
        - Max Drawdown Duration
        """
        if len(returns) == 0:
            return {}
        
        metrics = {}
        periods_per_year = 252  # Assuming daily data
        
        # Sortino Ratio (downside risk adjusted)
        excess_returns = returns - self.risk_free_rate / periods_per_year
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) > 0:
            downside_std = np.sqrt(np.mean(downside_returns**2)) * np.sqrt(periods_per_year)
            if downside_std > 0:
                annual_return = returns.mean() * periods_per_year
                metrics['sortino_ratio'] = (annual_return - self.risk_free_rate) / downside_std
            else:
                metrics['sortino_ratio'] = np.nan
        else:
            metrics['sortino_ratio'] = np.inf
        
        # Calmar Ratio (return / max drawdown)
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - running_max) / running_max
        max_dd = abs(drawdown.min())
        if max_dd > 0:
            annual_return = returns.mean() * periods_per_year
            metrics['calmar_ratio'] = annual_return / max_dd
        else:
            metrics['calmar_ratio'] = np.nan
        
        # Omega Ratio (gains vs losses at threshold=0)
        threshold = 0.0
        excess = returns - threshold
        gains = excess[excess > 0].sum()
        losses = -excess[excess < 0].sum()
        if losses > 0:
            metrics['omega_ratio'] = gains / losses
        else:
            metrics['omega_ratio'] = np.inf if gains > 0 else np.nan
        
        # Value at Risk (95% and 99%)
        metrics['var_95'] = np.percentile(returns, 5)
        metrics['var_99'] = np.percentile(returns, 1)
        
        # Conditional VaR (Expected Shortfall)
        var_95 = metrics['var_95']
        cvar_95 = returns[returns <= var_95].mean()
        metrics['cvar_95'] = cvar_95 if not np.isnan(cvar_95) else var_95
        
        var_99 = metrics['var_99']
        cvar_99 = returns[returns <= var_99].mean()
        metrics['cvar_99'] = cvar_99 if not np.isnan(cvar_99) else var_99
        
        # Tail Ratio (avg top 5% / avg bottom 5%)
        top_tail = returns[returns >= np.percentile(returns, 95)]
        bottom_tail = returns[returns <= np.percentile(returns, 5)]
        if len(bottom_tail) > 0 and abs(bottom_tail.mean()) > 1e-10:
            metrics['tail_ratio'] = abs(top_tail.mean() / bottom_tail.mean())
        else:
            metrics['tail_ratio'] = np.nan
        
        # Max Drawdown Duration
        in_drawdown = drawdown < 0
        max_duration = 0
        current_duration = 0
        for is_dd in in_drawdown:
            if is_dd:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0
        metrics['max_dd_duration'] = max_duration
        
        # Position sizing statistics
        if 'position_size' in trade_log.columns and len(trade_log) > 0:
            metrics['avg_position_size'] = trade_log['position_size'].mean()
            metrics['max_position_size'] = trade_log['position_size'].max()
            metrics['min_position_size'] = trade_log['position_size'].min()
        
        # Exit reason breakdown
        if 'exit_reason' in trade_log.columns and len(trade_log) > 0:
            exit_counts = trade_log['exit_reason'].value_counts()
            for reason, count in exit_counts.items():
                metrics[f'exits_{reason}'] = count
        
        return metrics
