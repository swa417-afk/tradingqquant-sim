from __future__ import annotations

def apply_bps(price: float, bps: float, side: float) -> float:
    # side > 0 buy pays more; side < 0 sell receives less
    # bps are basis points (1 bps = 0.01%)
    adj = (bps / 10000.0) * price
    return price + (adj if side > 0 else -adj)

def commission_cost(notional: float, commission_bps: float) -> float:
    return abs(notional) * (commission_bps / 10000.0)
