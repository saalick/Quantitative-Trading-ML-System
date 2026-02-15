#!/usr/bin/env python3

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

def generate_sample_ohlcv(n_days: int = 1200, seed: int = 42) -> pd.DataFrame:
    """
    Generate daily OHLCV data with realistic dynamics.
    
    - Geometric Brownian motion for price drift
    - GARCH-like volatility clustering
    - Occasional trends and reversals
    - Realistic volume patterns
    - Some extreme events (crashes/rallies)
    """
    np.random.seed(seed)
    start_date = datetime(2020, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_days)]
    
    # Base parameters
    S0 = 100.0
    mu = 0.0003  # daily drift
    sigma_base = 0.012  # base volatility
    
    # Volatility clustering: volatility follows an AR(1)-like process
    vol_state = np.zeros(n_days)
    vol_state[0] = sigma_base
    for t in range(1, n_days):
        vol_state[t] = 0.9 * vol_state[t - 1] + 0.1 * sigma_base + 0.02 * np.random.randn()
        vol_state[t] = np.clip(vol_state[t], 0.005, 0.05)
    
    # Add occasional regime shifts (crashes/rallies)
    regime = np.ones(n_days)
    for _ in range(8):  # 8 events
        t = np.random.randint(50, n_days - 50)
        length = np.random.randint(5, 25)
        if np.random.rand() > 0.5:
            regime[t : t + length] = -1.5  # crash
        else:
            regime[t : t + length] = 1.8   # rally
    
    # Generate log returns
    log_returns = np.zeros(n_days)
    for t in range(1, n_days):
        shock = vol_state[t] * np.random.randn()
        drift = mu + 0.3 * (regime[t] - 1) * vol_state[t]
        log_returns[t] = drift + shock
    
    # Build OHLC from close and returns
    close = S0 * np.exp(np.cumsum(log_returns))
    
    # Open = previous close
    open_ = np.roll(close, 1)
    open_[0] = S0
    
    # High/Low: add intraday range
    daily_range = np.abs(vol_state * close * np.random.randn(n_days) * 0.5)
    high = np.maximum(close, open_) + daily_range * np.random.rand(n_days)
    low = np.minimum(close, open_) - daily_range * (1 - np.random.rand(n_days))
    # Ensure OHLC consistency
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))
    
    # Volume: base + correlation with absolute return
    base_vol = 1_000_000
    vol_scale = 0.5 + np.abs(log_returns) * 50 + np.random.lognormal(0, 0.5, n_days)
    volume = (base_vol * vol_scale).astype(int)
    volume = np.maximum(volume, 100_000)
    
    df = pd.DataFrame({
        "date": dates,
        "open": np.round(open_, 2),
        "high": np.round(high, 2),
        "low": np.round(low, 2),
        "close": np.round(close, 2),
        "volume": volume,
    })
    return df


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "sample_data.csv"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_sample_ohlcv(1200)
    df.to_csv(data_path, index=False)
    print(f"Generated {len(df)} rows -> {data_path}")
