from __future__ import annotations
import pandas as pd
from typing import Dict, List, Optional
from .validation import validate_ohlcv

def load_csv_ohlcv(
    csv_files: List[str],
    timestamp_col: str = "timestamp",
    tz: str = "UTC",
) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for path in csv_files:
        df = pd.read_csv(path)
        if timestamp_col not in df.columns:
            raise ValueError(f"{path}: missing timestamp col '{timestamp_col}'")
        # Parse timestamp robustly
        ts = df[timestamp_col]
        if pd.api.types.is_numeric_dtype(ts):
            # assume epoch seconds
            dt = pd.to_datetime(ts, unit="s", utc=True)
        else:
            dt = pd.to_datetime(ts, utc=True, errors="coerce")
        if dt.isna().any():
            raise ValueError(f"{path}: some timestamps could not be parsed")
        df = df.drop(columns=[timestamp_col])
        df.index = dt.tz_convert(tz) if tz else dt
        df = df.sort_index()
        validate_ohlcv(df, path)
        # Infer symbol from filename like BTC-USD_1m.csv
        fname = path.split("/")[-1]
        symbol = fname.split("_")[0]
        out[symbol] = df
    return out
