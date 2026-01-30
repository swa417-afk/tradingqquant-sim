from __future__ import annotations
from typing import Dict, Optional
import time
import requests
import pandas as pd
from .broker_base import BrokerBase
from .orders import Order
from .costs import apply_bps, commission_cost

class CoinbasePaperBroker(BrokerBase):
    """
    Paper broker using Coinbase public endpoints for prices.
    Default uses best bid/ask; falls back to last trade.
    This does NOT place real orders (live mode is intentionally not implemented here).
    """
    def __init__(self, products: list[str], commission_bps: float, slippage_bps: float, use_public_api: bool = True):
        self.products = products
        self.commission_bps = float(commission_bps)
        self.slippage_bps = float(slippage_bps)
        self.use_public_api = use_public_api
        self.base = "https://api.exchange.coinbase.com"
        self.session = requests.Session()
        self._cache: Dict[str, tuple[float, float]] = {}  # symbol -> (price, unix_time)

    def _get(self, path: str, timeout: float = 10.0) -> Dict:
        url = self.base + path
        r = self.session.get(url, timeout=timeout, headers={"Accept": "application/json"})
        r.raise_for_status()
        return r.json()

    def get_price(self, symbol: str, ts: pd.Timestamp | None = None) -> float:
        # cache for 1 second to avoid hammering
        now = time.time()
        if symbol in self._cache and (now - self._cache[symbol][1] < 1.0):
            return float(self._cache[symbol][0])

        # best bid/ask
        try:
            book = self._get(f"/products/{symbol}/book?level=1")
            bid = float(book["bids"][0][0]) if book.get("bids") else None
            ask = float(book["asks"][0][0]) if book.get("asks") else None
            if bid and ask:
                mid = (bid + ask) / 2.0
                self._cache[symbol] = (mid, now)
                return mid
        except Exception:
            pass

        # fallback: last trade
        ticker = self._get(f"/products/{symbol}/ticker")
        px = float(ticker["price"])
        self._cache[symbol] = (px, now)
        return px

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
            "mode": "paper",
            "venue": "coinbase_public",
        }
