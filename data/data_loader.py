"""
Data loading utilities for OHLCV data.
Loads from CSV, validates schema, and returns pandas DataFrame.
"""

import pandas as pd
from pathlib import Path
from typing import Union, Optional


REQUIRED_COLUMNS = ["date", "open", "high", "low", "close", "volume"]


def load_ohlcv(
    path: Union[str, Path],
    date_column: str = "date",
    parse_dates: bool = True,
) -> pd.DataFrame:
    """
    Load OHLCV data from CSV.

    Args:
        path: Path to CSV file.
        date_column: Name of date column.
        parse_dates: Whether to parse date column to datetime.

    Returns:
        DataFrame with columns: date, open, high, low, close, volume.

    Raises:
        FileNotFoundError: If path does not exist.
        ValueError: If required columns are missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path)

    # Normalize column names (strip whitespace, lowercase for comparison)
    col_map = {c.strip(): c for c in df.columns}
    df = df.rename(columns={v: k.strip().lower() for k, v in col_map.items()})

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns)}")

    df = df[REQUIRED_COLUMNS].copy()

    if parse_dates:
        df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Coerce numeric
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

    return df


def validate_ohlcv(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """
    Validate OHLCV DataFrame for consistency.

    Returns:
        (is_valid, list of error messages).
    """
    errors = []

    if df["close"].isna().any():
        errors.append("Missing close prices")
    if (df[["open", "high", "low", "close"]].min(axis=1) < 0).any():
        errors.append("Negative prices found")
    if (df["high"] < df["low"]).any():
        errors.append("High < Low for some rows")
    if (df["high"] < df["close"]).any() or (df["high"] < df["open"]).any():
        errors.append("High should be >= open and close")
    if (df["low"] > df["close"]).any() or (df["low"] > df["open"]).any():
        errors.append("Low should be <= open and close")
    if (df["volume"] < 0).any():
        errors.append("Negative volume found")

    return (len(errors) == 0, errors)
