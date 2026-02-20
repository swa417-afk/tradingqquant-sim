from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Tuple
import pandas as pd
from .orders import Order

class BrokerBase(ABC):
    @abstractmethod
    def get_price(self, symbol: str, ts: pd.Timestamp | None = None) -> float:
        ...

    @abstractmethod
    def place_order(self, order: Order, ts: pd.Timestamp | None = None) -> Dict:
        ...
