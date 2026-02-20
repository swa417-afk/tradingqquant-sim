from __future__ import annotations
from typing import Dict
import pandas as pd
from .base import StrategyBase, Signal

class PairsStrategy(StrategyBase):
    """
    Stub: relative value/pairs trading.
    Extend with cointegration/spread z-score logic.
    """
    def generate_signals(self, bars: Dict[str, pd.DataFrame]) -> Dict[str, Signal]:
        # default: flat
        return {sym: Signal(0.0) for sym in bars.keys()}
