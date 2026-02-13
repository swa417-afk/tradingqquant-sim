from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import pandas as pd

@dataclass
class RiskLimits:
    max_gross_leverage: float
    max_position_pct: float
    max_daily_loss_pct: float
    kill_switch: bool = True

def clamp_target(
    symbol: str,
    target_pct: float,
    equity: float,
    price: float,
    limits: RiskLimits
) -> float:
    # clamp by max position pct
    target_pct = max(min(target_pct, limits.max_position_pct), -limits.max_position_pct)
    # leverage is enforced at portfolio level later; here just return symbol clamp
    return target_pct

def check_daily_loss(equity_start: float, equity_now: float, limits: RiskLimits) -> bool:
    if equity_start <= 0:
        return False
    dd = (equity_start - equity_now) / equity_start
    return dd >= limits.max_daily_loss_pct
