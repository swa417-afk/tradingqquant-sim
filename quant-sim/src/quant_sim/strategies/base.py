from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
import pandas as pd

@dataclass
class Signal:
    # desired target position as fraction of equity (e.g., 0.1 long, -0.1 short)
    target_pct: float

class StrategyBase:
    def __init__(self, params: Dict):
        self.params = params

    def generate_signals(self, bars: Dict[str, pd.DataFrame]) -> Dict[str, Signal]:
        raise NotImplementedError
