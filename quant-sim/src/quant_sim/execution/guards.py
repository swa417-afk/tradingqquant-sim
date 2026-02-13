from __future__ import annotations

import time
from dataclasses import dataclass


class OrderRateLimiter:
    """Simple rolling-window order rate limiter (kill-switch)."""

    def __init__(self, max_orders: int, window_seconds: int = 60):
        self.max_orders = int(max_orders)
        self.window_seconds = int(window_seconds)
        self._ts: list[float] = []

    def allow(self) -> bool:
        if self.max_orders <= 0:
            return True
        now = time.time()
        cutoff = now - self.window_seconds
        self._ts = [t for t in self._ts if t >= cutoff]
        if len(self._ts) >= self.max_orders:
            return False
        self._ts.append(now)
        return True


@dataclass
class DisconnectGuard:
    """Trips if quotes go stale or repeated quote failures occur."""

    max_stale_seconds: float = 20.0
    max_failures: int = 5

    last_ok_ts: float = 0.0
    failures: int = 0

    def ok(self):
        self.last_ok_ts = time.time()
        self.failures = 0

    def fail(self):
        self.failures += 1

    def tripped(self) -> bool:
        now = time.time()
        if self.failures >= self.max_failures:
            return True
        if self.last_ok_ts and (now - self.last_ok_ts) > self.max_stale_seconds:
            return True
        return False
