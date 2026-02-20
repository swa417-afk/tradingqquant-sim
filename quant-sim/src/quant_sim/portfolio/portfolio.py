from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List
import pandas as pd

@dataclass
class Position:
    qty: float = 0.0
    avg_price: float = 0.0

@dataclass
class Portfolio:
    cash: float
    positions: Dict[str, Position] = field(default_factory=dict)
    equity_curve: List[Dict] = field(default_factory=list)

    def position_qty(self, symbol: str) -> float:
        return self.positions.get(symbol, Position()).qty

    def update_from_fill(self, fill: Dict) -> None:
        sym = fill["symbol"]
        qty = float(fill["qty"])
        px = float(fill["fill_price"])
        fee = float(fill.get("commission", 0.0))
        notional = px * qty

        pos = self.positions.get(sym, Position())
        new_qty = pos.qty + qty

        # cash decreases on buy (qty>0), increases on sell (qty<0)
        self.cash -= notional
        self.cash -= fee

        # update average price for remaining position (simple weighted)
        if new_qty == 0:
            pos = Position(0.0, 0.0)
        else:
            # If adding in same direction, update avg; if reducing/flip, keep it simple
            if (pos.qty == 0) or (pos.qty * qty > 0):
                pos.avg_price = (pos.avg_price * pos.qty + px * qty) / new_qty
            else:
                # reduction or flip: set avg to px if flipped, else keep prior avg
                if (pos.qty > 0 and new_qty < 0) or (pos.qty < 0 and new_qty > 0):
                    pos.avg_price = px
            pos.qty = new_qty

        self.positions[sym] = pos

    def mark_to_market(self, prices: Dict[str, float], ts: pd.Timestamp) -> Dict:
        pos_val = 0.0
        for sym, pos in self.positions.items():
            px = prices.get(sym)
            if px is None:
                continue
            pos_val += pos.qty * float(px)
        equity = self.cash + pos_val
        row = {"timestamp": ts.isoformat(), "cash": self.cash, "positions_value": pos_val, "equity": equity}
        self.equity_curve.append(row)
        return row
