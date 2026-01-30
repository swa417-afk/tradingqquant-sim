from __future__ import annotations
import pandas as pd

RULE_MAP = {
    "1s":"1S","10s":"10S","30s":"30S",
    "1m":"1min","5m":"5min","15m":"15min","30m":"30min",
    "1h":"1H","4h":"4H",
    "1d":"1D",
}

def to_rule(tf: str) -> str:
    tf = tf.strip().lower()
    if tf not in RULE_MAP:
        raise ValueError(f"Unsupported timeframe: {tf}. Supported: {list(RULE_MAP)}")
    return RULE_MAP[tf]

def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    rule = to_rule(timeframe)
    agg = {
        "open":"first",
        "high":"max",
        "low":"min",
        "close":"last",
        "volume":"sum",
    }
    out = df.resample(rule).agg(agg).dropna()
    return out
