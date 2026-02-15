"""
Feature engineering for quantitative trading: 45 features.
Returns-based, technical indicators, volatility, volume, and pattern features.
All use pandas for efficiency and handle NaNs appropriately.
"""

import numpy as np
import pandas as pd
from typing import List


# --- Returns-based features (10) ---

def add_return_features(df: pd.DataFrame, close: str = "close") -> pd.DataFrame:
    """
    Returns at multiple horizons. Predictive: momentum and mean-reversion
    signals (e.g. short-term reversal, intermediate momentum).
    """
    c = df[close]
    df = df.copy()
    for w in [1, 5, 10, 20, 60]:
        df[f"return_{w}d"] = c.pct_change(w)
    # Log return (symmetric, better for volatility modeling)
    df["log_return_1d"] = np.log(c / c.shift(1))
    # Relative to 20d mean (deviation from recent trend)
    df["relative_return_20d"] = c.pct_change(1) - c.pct_change(20).rolling(20).mean()
    # Cumulative over windows
    df["cum_return_5d"] = (c / c.shift(5) - 1)
    df["cum_return_20d"] = (c / c.shift(20) - 1)
    # Return momentum (5d change in 5d return)
    df["return_momentum_5d"] = c.pct_change(5) - c.pct_change(5).shift(5)
    return df


# --- Technical indicators (15) ---

def add_sma(df: pd.DataFrame, close: str = "close") -> pd.DataFrame:
    """Simple moving averages. Trend and support/resistance levels."""
    c = df[close]
    for period in [5, 10, 20, 50, 200]:
        df[f"sma_{period}"] = c.rolling(period, min_periods=1).mean()
    return df


def add_ema(df: pd.DataFrame, close: str = "close") -> pd.DataFrame:
    """Exponential moving averages. More weight to recent prices."""
    c = df[close]
    for span in [12, 26]:
        df[f"ema_{span}"] = c.ewm(span=span, adjust=False).mean()
    return df


def add_rsi(df: pd.DataFrame, close: str = "close", period: int = 14) -> pd.DataFrame:
    """
    Relative Strength Index. Overbought/oversold; predictive of short-term reversals.
    """
    c = df[close]
    delta = c.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))
    return df


def add_macd(df: pd.DataFrame, close: str = "close", fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """
    MACD line, signal line, and histogram. Trend and momentum.
    """
    c = df[close]
    ema_fast = c.ewm(span=fast, adjust=False).mean()
    ema_slow = c.ewm(span=slow, adjust=False).mean()
    df["macd"] = ema_fast - ema_slow
    df["macd_signal"] = df["macd"].ewm(span=signal, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    return df


def add_stochastic(df: pd.DataFrame, high: str = "high", low: str = "low", close: str = "close", period: int = 14) -> pd.DataFrame:
    """Stochastic %K and %D. Momentum and overbought/oversold."""
    h = df[high]
    l = df[low]
    c = df[close]
    lowest = l.rolling(period).min()
    highest = h.rolling(period).max()
    df["stoch_k"] = 100 * (c - lowest) / (highest - lowest).replace(0, np.nan)
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()
    return df


def add_williams_r(df: pd.DataFrame, high: str = "high", low: str = "low", close: str = "close", period: int = 14) -> pd.DataFrame:
    """Williams %R. Similar to stochastic, inverted scale."""
    h = df[high]
    l = df[low]
    c = df[close]
    highest = h.rolling(period).max()
    lowest = l.rolling(period).min()
    df["williams_r"] = -100 * (highest - c) / (highest - lowest).replace(0, np.nan)
    return df


def add_price_vs_sma(df: pd.DataFrame, close: str = "close") -> pd.DataFrame:
    """Price position relative to SMA(20). Mean reversion / trend strength."""
    c = df[close]
    sma20 = c.rolling(20).mean()
    df["price_vs_sma20"] = (c - sma20) / sma20.replace(0, np.nan)
    return df


# --- Volatility indicators (8) ---

def add_historical_volatility(df: pd.DataFrame, close: str = "close") -> pd.DataFrame:
    """Annualized volatility over 10, 20, 30 days. Risk and regime."""
    log_ret = np.log(df[close] / df[close].shift(1))
    for w in [10, 20, 30]:
        df[f"hist_vol_{w}d"] = log_ret.rolling(w).std() * np.sqrt(252) * 100
    return df


def add_bollinger_bands(df: pd.DataFrame, close: str = "close", period: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """Bollinger Bands. Volatility and mean reversion (price relative to bands)."""
    c = df[close]
    df["bb_middle"] = c.rolling(period).mean()
    std = c.rolling(period).std()
    df["bb_upper"] = df["bb_middle"] + num_std * std
    df["bb_lower"] = df["bb_middle"] - num_std * std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"].replace(0, np.nan)
    return df


def add_atr(df: pd.DataFrame, high: str = "high", low: str = "low", close: str = "close", period: int = 14) -> pd.DataFrame:
    """Average True Range. Volatility and stop-loss sizing."""
    h = df[high]
    l = df[low]
    c = df[close]
    prev_close = c.shift(1)
    tr = np.maximum(h - l, np.maximum((h - prev_close).abs(), (l - prev_close).abs()))
    df["atr_14"] = tr.rolling(period).mean()
    return df


# --- Volume features (6) ---

def add_volume_features(df: pd.DataFrame, volume: str = "volume", high: str = "high", low: str = "low", close: str = "close") -> pd.DataFrame:
    """Volume SMAs, ratio, OBV, rate of change, Money Flow Index."""
    vol = df[volume]
    c = df[close]
    df["volume_sma_10"] = vol.rolling(10).mean()
    df["volume_sma_30"] = vol.rolling(30).mean()
    df["volume_ratio"] = vol / df["volume_sma_30"].replace(0, np.nan)
    # On-Balance Volume (cumulative signed volume)
    obv = (np.sign(c.diff()) * vol).fillna(0).cumsum()
    df["obv"] = obv
    # Normalize OBV for scale (optional: use as-is or scale by rolling max)
    df["obv_norm"] = obv / (obv.rolling(100).std().replace(0, np.nan))
    df["volume_roc"] = vol.pct_change(10)
    # Money Flow Index (14 period)
    typical = (df[high] + df[low] + c) / 3
    mf = typical * vol
    pos = mf.where(typical > typical.shift(1), 0).rolling(14).sum()
    neg = mf.where(typical < typical.shift(1), 0).rolling(14).sum()
    df["mfi"] = 100 - (100 / (1 + pos / neg.replace(0, np.nan)))
    return df


# --- Pattern features (6) ---

def add_higher_highs_lower_lows(df: pd.DataFrame, high: str = "high", low: str = "low", window: int = 5) -> pd.DataFrame:
    """Higher highs and lower lows (count or binary). Trend structure."""
    h = df[high]
    l = df[low]
    df["higher_highs"] = (h > h.shift(1)).rolling(window).sum()
    df["lower_lows"] = (l < l.shift(1)).rolling(window).sum()
    return df


def add_trend_strength(df: pd.DataFrame, close: str = "close", window: int = 20) -> pd.DataFrame:
    """Trend strength as linear regression R^2 of price over window."""
    c = df[close]
    def _r2(s):
        x = np.arange(len(s))
        if np.var(s) == 0:
            return 0
        return np.corrcoef(x, s)[0, 1] ** 2
    df["trend_strength"] = c.rolling(window).apply(_r2, raw=False)
    return df


def add_price_momentum(df: pd.DataFrame, close: str = "close") -> pd.DataFrame:
    """Price momentum: rate of change over 5 and 10 periods."""
    c = df[close]
    df["price_momentum_5"] = c / c.shift(5) - 1
    df["price_momentum_10"] = c / c.shift(10) - 1
    return df


def add_roc(df: pd.DataFrame, close: str = "close") -> pd.DataFrame:
    """Rate of change at 5 and 10 periods. Momentum feature."""
    c = df[close]
    df["roc_5"] = (c - c.shift(5)) / c.shift(5).replace(0, np.nan)
    df["roc_10"] = (c - c.shift(10)) / c.shift(10).replace(0, np.nan)
    return df


# --- Feature list and main API (45 features) ---

FEATURE_COLUMNS: List[str] = [
    "return_1d", "return_5d", "return_10d", "return_20d", "return_60d",
    "log_return_1d", "relative_return_20d", "cum_return_5d", "cum_return_20d", "return_momentum_5d",
    "sma_5", "sma_10", "sma_20", "sma_50", "sma_200", "ema_12", "ema_26", "rsi_14",
    "macd", "macd_signal", "macd_hist", "stoch_k", "stoch_d", "williams_r", "price_vs_sma20",
    "hist_vol_10d", "hist_vol_20d", "hist_vol_30d", "bb_upper", "bb_middle", "bb_lower", "bb_width", "atr_14",
    "volume_sma_10", "volume_sma_30", "volume_ratio", "obv_norm", "volume_roc", "mfi",
    "higher_highs", "lower_lows", "trend_strength", "price_momentum_5", "price_momentum_10", "roc_5",
]


def compute_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all 45 features. Assumes df has columns: open, high, low, close, volume.
    NaNs from rolling calculations are left for caller to handle (e.g. drop or fill).
    """
    df = df.copy()
    df = add_return_features(df)
    df = add_sma(df)
    df = add_ema(df)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_stochastic(df)
    df = add_williams_r(df)
    df = add_price_vs_sma(df)
    df = add_historical_volatility(df)
    df = add_bollinger_bands(df)
    df = add_atr(df)
    df = add_volume_features(df)
    df = add_higher_highs_lower_lows(df)
    df = add_trend_strength(df)
    df = add_price_momentum(df)
    df = add_roc(df)

    # Ensure we only return columns that exist and are in FEATURE_COLUMNS
    available = [c for c in FEATURE_COLUMNS if c in df.columns]
    return df[["date"] + available].copy() if "date" in df.columns else df[available].copy()
