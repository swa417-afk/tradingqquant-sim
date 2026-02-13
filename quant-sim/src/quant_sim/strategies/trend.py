from __future__ import annotations
from typing import Dict
import pandas as pd
from .base import StrategyBase, Signal

class TrendStrategy(StrategyBase):
    def generate_signals(self, bars: Dict[str, pd.DataFrame]) -> Dict[str, Signal]:
        fast = int(self.params.get("fast", 20))
        slow = int(self.params.get("slow", 60))
        out: Dict[str, Signal] = {}
        for sym, df in bars.items():
            c = df["close"].astype(float)
            if len(c) < slow + 2:
                out[sym] = Signal(0.0)
                continue
            ma_fast = c.rolling(fast).mean().iloc[-1]
            ma_slow = c.rolling(slow).mean().iloc[-1]
            # Simple: long if fast > slow, short if fast < slow
            if ma_fast > ma_slow:
                out[sym] = Signal(+0.10)
            elif ma_fast < ma_slow:
                out[sym] = Signal(-0.10)
            else:
                out[sym] = Signal(0.0)
        return out
