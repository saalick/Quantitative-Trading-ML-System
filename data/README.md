# Data Directory

## Contents

- **sample_data.csv**: 1000+ rows of daily OHLCV (Open, High, Low, Close, Volume) data generated with realistic price dynamics (geometric Brownian motion, volatility clustering, trends). Date range: 2020-01-01 to recent. Use this for development and demos.

- **data_loader.py**: Utilities to load and validate OHLCV data from CSV or other sources.

## Obtaining Real Data

Use the script in `scripts/download_data.py` to download historical data from Yahoo Finance:

```bash
python scripts/download_data.py --ticker SPY --start 2020-01-01 --end 2024-12-31 --output data/ohlcv.csv
```

## Data Schema

| Column | Type   | Description        |
|--------|--------|--------------------|
| date   | datetime | Trading date     |
| open   | float  | Opening price      |
| high   | float  | High price         |
| low    | float  | Low price          |
| close  | float  | Closing price      |
| volume | int/float | Trading volume |

All prices should be in the same currency; volume in shares/contracts.
