from __future__ import annotations
from typing import Dict
import pandas as pd
from .base import StrategyBase, Signal

class MeanReversionStrategy(StrategyBase):
    def generate_signals(self, bars: Dict[str, pd.DataFrame]) -> Dict[str, Signal]:
        lb = int(self.params.get("lookback", 50))
        z_enter = float(self.params.get("z_enter", 1.5))
        out: Dict[str, Signal] = {}
        for sym, df in bars.items():
            c = df["close"].astype(float)
            if len(c) < lb + 2:
                out[sym] = Signal(0.0)
                continue
            m = c.rolling(lb).mean().iloc[-1]
            s = c.rolling(lb).std().iloc[-1]
            if s == 0 or pd.isna(s):
                out[sym] = Signal(0.0)
                continue
            z = (c.iloc[-1] - m) / s
            if z > z_enter:
                out[sym] = Signal(-0.10)
            elif z < -z_enter:
                out[sym] = Signal(+0.10)
            else:
                out[sym] = Signal(0.0)
        return out
