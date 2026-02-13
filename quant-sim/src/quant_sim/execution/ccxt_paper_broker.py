from __future__ import annotations

from typing import Any, Dict, Optional
import time
import pandas as pd

import ccxt

from .broker_base import BrokerBase
from .orders import Order
from .costs import apply_bps, commission_cost


class CCXTPaperBroker(BrokerBase):
    """Unified paper broker using CCXT for market prices.

    - Uses public endpoints (no API keys) for quotes.
    - Simulates fills with slippage + commission.
    - Works with many exchanges (binance, coinbase, kraken, etc.)
    - Symbols are CCXT style: "BTC/USDT", "ETH/USD", etc.
    """

    def __init__(
        self,
        exchange_id: str,
        symbols: list[str],
        *,
        commission_bps: float,
        slippage_bps: float,
        quote_cache_seconds: float = 1.0,
        ccxt_options: Optional[Dict[str, Any]] = None,
    ):
        self.exchange_id = exchange_id
        self.symbols = symbols
        self.commission_bps = float(commission_bps)
        self.slippage_bps = float(slippage_bps)
        self.quote_cache_seconds = float(quote_cache_seconds)

        opts = dict(ccxt_options or {})
        opts.setdefault("enableRateLimit", True)
        ex_cls = getattr(ccxt, exchange_id)
        self.ex = ex_cls(opts)

        self._cache: Dict[str, tuple[float, float]] = {}

    def get_price(self, symbol: str, ts: pd.Timestamp | None = None) -> float:
        now = time.time()
        cached = self._cache.get(symbol)
        if cached is not None and (now - cached[1]) < self.quote_cache_seconds:
            return float(cached[0])

        t = self.ex.fetch_ticker(symbol)
        px = t.get("last") or t.get("close")
        if px is None:
            bid = t.get("bid")
            ask = t.get("ask")
            if bid is not None and ask is not None:
                px = (float(bid) + float(ask)) / 2.0
        if px is None:
            raise ValueError(f"No price for {symbol} from {self.exchange_id}")

        px_f = float(px)
        self._cache[symbol] = (px_f, now)
        return px_f

    def place_order(self, order: Order, ts: pd.Timestamp | None = None) -> dict:
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
            "venue": f"{self.exchange_id}_ccxt_paper",
        }
