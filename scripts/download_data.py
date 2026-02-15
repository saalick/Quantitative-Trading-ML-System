#!/usr/bin/env python3
"""
Download OHLCV data from Yahoo Finance.
Usage:
  python scripts/download_data.py --ticker SPY --start 2020-01-01 --end 2024-12-31 --output data/ohlcv.csv
"""

import argparse
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ticker", type=str, default="SPY")
    p.add_argument("--start", type=str, default="2020-01-01")
    p.add_argument("--end", type=str, default="2024-12-31")
    p.add_argument("--output", type=str, default="data/downloaded.csv")
    args = p.parse_args()
    try:
        import yfinance as yf
    except ImportError:
        print("Install yfinance: pip install yfinance")
        return 1
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        data = yf.download(args.ticker, start=args.start, end=args.end, progress=False)
        if data.empty:
            print("No data returned. Check ticker and date range.")
            return 1
        data = data.reset_index()
        data.columns = [c.lower() for c in data.columns]
        rename = {"date": "date", "open": "open", "high": "high", "low": "low", "close": "close", "adj close": "close", "volume": "volume"}
        cols = ["date", "open", "high", "low", "close", "volume"]
        if "adj close" in data.columns:
            data["close"] = data["adj close"]
        data = data[[c for c in cols if c in data.columns]]
        data.to_csv(out, index=False)
        print(f"Saved {len(data)} rows to {out}")
    except Exception as e:
        print(f"Error: {e}")
        return 1
    return 0


if __name__ == "__main__":
    exit(main())
