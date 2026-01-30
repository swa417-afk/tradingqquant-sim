from __future__ import annotations
from typing import Dict
import pandas as pd
from .base import StrategyBase, Signal

class EnsembleStrategy(StrategyBase):
    """
    Combine multiple child strategy signals (simple average).
    params:
    children: list of {name: "trend"/"meanrev"/..., params: {...}}
    """
    def __init__(self, params: Dict, registry: Dict[str, type[StrategyBase]]):
        super().__init__(params)
        self.registry = registry
        self.children = []
        for child in params.get("children", []):
            name = child["name"]
            cls = registry[name]
            self.children.append(cls(child.get("params", {})))

    def generate_signals(self, bars: Dict[str, pd.DataFrame]) -> Dict[str, Signal]:
        if not self.children:
            return {sym: Signal(0.0) for sym in bars}
        combined = {sym: 0.0 for sym in bars}
        for strat in self.children:
            sigs = strat.generate_signals(bars)
            for sym, s in sigs.items():
                combined[sym] += s.target_pct
        k = float(len(self.children))
        return {sym: Signal(v / k) for sym, v in combined.items()}
