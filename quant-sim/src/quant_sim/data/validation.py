from __future__ import annotations
import pandas as pd

REQUIRED_COLS = ["open", "high", "low", "close", "volume"]

def validate_ohlcv(df: pd.DataFrame, name: str = "data") -> None:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing columns {missing}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"{name}: index must be DatetimeIndex")
    if df.index.has_duplicates:
        raise ValueError(f"{name}: duplicate timestamps found")
    if df[REQUIRED_COLS].isna().any().any():
        raise ValueError(f"{name}: NaNs found in required columns")
    if (df["high"] < df[["open","close","low"]].max(axis=1)).any():
        raise ValueError(f"{name}: invalid OHLC (high too low)")
    if (df["low"] > df[["open","close","high"]].min(axis=1)).any():
        raise ValueError(f"{name}: invalid OHLC (low too high)")
