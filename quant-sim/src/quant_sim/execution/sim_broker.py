from __future__ import annotations
from typing import Dict
import pandas as pd
from .broker_base import BrokerBase
from .orders import Order
from .costs import apply_bps, commission_cost

class SimBroker(BrokerBase):
    def __init__(self, bars: Dict[str, pd.DataFrame], commission_bps: float, slippage_bps: float):
        self.bars = bars
        self.commission_bps = float(commission_bps)
        self.slippage_bps = float(slippage_bps)

    def get_price(self, symbol: str, ts: pd.Timestamp | None = None) -> float:
        df = self.bars[symbol]
        if ts is None:
            return float(df["close"].iloc[-1])
        # use last known close <= ts
        s = df.loc[:ts, "close"]
        if len(s) == 0:
            raise ValueError(f"No price for {symbol} at {ts}")
        return float(s.iloc[-1])

    def place_order(self, order: Order, ts: pd.Timestamp | None = None) -> Dict:
        px = self.get_price(order.symbol, ts)
        fill_px = apply_bps(px, self.slippage_bps, order.qty)
        notional = fill_px * order.qty
        fee = commission_cost(notional, self.commission_bps)
        return {
            "symbol": order.symbol,
            "qty": float(order.qty),
            "fill_price": float(fill_px),
            "notional": float(notional),
            "commission": float(fee),
            "timestamp": (ts.isoformat() if ts is not None else None),
        }
