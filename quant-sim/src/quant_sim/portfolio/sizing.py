from __future__ import annotations
from typing import Dict
import math

def target_pct_to_qty(target_pct: float, equity: float, price: float) -> float:
    if price <= 0:
        return 0.0
    target_notional = target_pct * equity
    return target_notional / price
