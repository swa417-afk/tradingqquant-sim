from __future__ import annotations
import pandas as pd

def compute_metrics(equity_curve: pd.DataFrame) -> dict:
    if equity_curve.empty:
        return {}
    eq = equity_curve["equity"].astype(float)
    rets = eq.pct_change().fillna(0.0)
    cum_ret = (eq.iloc[-1] / eq.iloc[0]) - 1.0 if eq.iloc[0] != 0 else 0.0
    peak = eq.cummax()
    dd = (peak - eq) / peak.replace(0.0, 1.0)
    max_dd = float(dd.max())
    vol = float(rets.std())
    return {
        "start_equity": float(eq.iloc[0]),
        "end_equity": float(eq.iloc[-1]),
        "cumulative_return": float(cum_ret),
        "max_drawdown": max_dd,
        "return_volatility": vol,
        "num_points": int(len(eq)),
    }
