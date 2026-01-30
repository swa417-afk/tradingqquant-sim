from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

def make_features(df: pd.DataFrame, lookbacks: List[int], normalize: bool = True) -> pd.DataFrame:
    # Features are strictly based on past values (shifted where needed)
    close = df["close"].astype(float)
    ret1 = close.pct_change().fillna(0.0)
    feats = {}
    for lb in lookbacks:
        feats[f"ret_{lb}"] = close.pct_change(lb).shift(1)
        feats[f"vol_{lb}"] = ret1.rolling(lb).std().shift(1)
        feats[f"mom_{lb}"] = (close / close.rolling(lb).mean()).shift(1) - 1.0
    X = pd.DataFrame(feats, index=df.index).dropna()
    if normalize:
        mu = X.mean()
        sd = X.std().replace(0.0, 1.0)
        X = (X - mu) / sd
    return X

def make_labels(df: pd.DataFrame, kind: str = "p_up") -> pd.Series:
    close = df["close"].astype(float)
    fwd_ret = close.pct_change().shift(-1)  # next bar return
    if kind == "p_up":
        y = (fwd_ret > 0).astype(int)
    elif kind == "expected_return":
        y = fwd_ret.astype(float)
    else:
        raise ValueError(f"Unknown label kind: {kind}")
    return y
