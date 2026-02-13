
import * as fs from 'fs';
import * as path from 'path';

const REPO = "quant-sim";

function mkdir(p: string) {
    if (!fs.existsSync(p)) {
        fs.mkdirSync(p, { recursive: true });
    }
}

function write(p: string, content: string) {
    const dir = path.dirname(p);
    mkdir(dir);
    fs.writeFileSync(p, content.trimStart());
}

mkdir(REPO);
mkdir(path.join(REPO, "configs"));
mkdir(path.join(REPO, "src/quant_sim/data"));
mkdir(path.join(REPO, "src/quant_sim/strategies"));
mkdir(path.join(REPO, "src/quant_sim/execution"));
mkdir(path.join(REPO, "src/quant_sim/portfolio"));
mkdir(path.join(REPO, "src/quant_sim/analytics"));
mkdir(path.join(REPO, "src/quant_sim/utils"));
mkdir(path.join(REPO, "tests"));
mkdir(path.join(REPO, "runs"));

// pyproject.toml
write(path.join(REPO, "pyproject.toml"), `
[build-system]
requires = ["setuptools>=69.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "quant-sim"
version = "0.1.0"
description = "Quant Simulator"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
  "numpy>=1.26",
  "pandas>=2.2",
  "pyyaml>=6.0",
  "requests>=2.31",
  "matplotlib>=3.8",
  "jinja2>=3.1",
  "torch>=2.2",
]

[project.scripts]
quant-sim = "quant_sim.cli:main"

[tool.setuptools]
package-dir = {"" = "src"}
`);

// README.md
write(path.join(REPO, "README.md"), `
# Quant Simulator
`);

// Configs
write(path.join(REPO, "configs/backtest.yaml"), `
run:
  name: "backtest_demo"
  out_dir: "runs"
  seed: 42

data:
  csv_files: [] 
  timestamp_col: "timestamp"
  tz: "UTC"
  base_timeframe: "1m"

universe:
  symbols: []

execution:
  commission_bps: 2.0
  slippage_bps: 1.0

risk:
  max_gross_leverage: 2.0
  max_position_pct: 0.25
  max_daily_loss_pct: 0.05

portfolio:
  starting_cash: 100000.0

strategy:
  name: "trend"
  timeframe: "5m"
  params:
    fast: 20
    slow: 60

ml:
  enabled: false
  config: "configs/pytorch.yaml"

reports:
  csv: true
  html: true
`);

write(path.join(REPO, "configs/paper_coinbase.yaml"), `
run:
  name: "paper_coinbase_demo"
  out_dir: "runs"
  seed: 42

paper:
  mode: "paper"
  poll_seconds: 3
  decision_timeframe: "1m"

coinbase:
  products: ["BTC-USD", "ETH-USD"]
  use_public_api: true

execution:
  commission_bps: 2.0
  slippage_bps: 1.0

risk:
  max_gross_leverage: 2.0
  max_position_pct: 0.25
  max_daily_loss_pct: 0.05
  kill_switch: true

portfolio:
  starting_cash: 100000.0

strategy:
  name: "trend"
  timeframe: "5m"
  params:
    fast: 20
    slow: 60

ml:
  enabled: false
  config: "configs/pytorch.yaml"

reports:
  csv: true
  html: true
`);

write(path.join(REPO, "configs/pytorch.yaml"), `
features:
  lookbacks: [5, 10, 20, 60]
  use_returns: true
  use_vol: true
  normalize: true

model:
  kind: "mlp"
  hidden: [64, 64]
  dropout: 0.1
  out: "p_up"

training:
  walk_forward: true
  train_bars: 2000
  test_bars: 500
  epochs: 10
  lr: 0.001
  batch_size: 256
  early_stop_patience: 2
  seed: 42
`);

// Python Source

write(path.join(REPO, "src/quant_sim/init.py"), `
__all__ = ["cli"]
version = "0.1.0"
`);

write(path.join(REPO, "src/quant_sim/utils/logging.py"), \`
from __future__ import annotations
import logging
import os
from typing import Optional

def setup_logger(name: str, log_path: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger.propagate = False
    return logger
\`);

write(path.join(REPO, "src/quant_sim/config.py"), \`
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict
import yaml

def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

@dataclass(frozen=True)
class Config:
    raw: Dict[str, Any]

    def get(self, *keys: str, default=None):
        cur = self.raw
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur
\`);

write(path.join(REPO, "src/quant_sim/run_artifacts.py"), \`
from __future__ import annotations
import os
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional
import uuid

@dataclass
class RunArtifacts:
    run_id: str
    run_dir: str
    logs_path: str

def make_run_dir(out_dir: str, run_name: str) -> RunArtifacts:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    run_id = f"{ts}_runid-{uuid.uuid4().hex[:8]}"
    run_dir = os.path.join(out_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    logs_path = os.path.join(run_dir, "logs.txt")
    return RunArtifacts(run_id=run_id, run_dir=run_dir, logs_path=logs_path)

def snapshot_config(config_path: str, run_dir: str) -> str:
    dst = os.path.join(run_dir, "config_snapshot.yaml")
    shutil.copyfile(config_path, dst)
    return dst

def save_json(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
\`);

write(path.join(REPO, "src/quant_sim/data/provider.py"), \`
from __future__ import annotations
import pandas as pd
from typing import Dict, List, Optional
from .validation import validate_ohlcv

def load_csv_ohlcv(
    csv_files: List[str],
    timestamp_col: str = "timestamp",
    tz: str = "UTC",
) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for path in csv_files:
        df = pd.read_csv(path)
        if timestamp_col not in df.columns:
            raise ValueError(f"{path}: missing timestamp col '{timestamp_col}'")
        # Parse timestamp robustly
        ts = df[timestamp_col]
        if pd.api.types.is_numeric_dtype(ts):
            # assume epoch seconds
            dt = pd.to_datetime(ts, unit="s", utc=True)
        else:
            dt = pd.to_datetime(ts, utc=True, errors="coerce")
        if dt.isna().any():
            raise ValueError(f"{path}: some timestamps could not be parsed")
        df = df.drop(columns=[timestamp_col])
        df.index = dt.tz_convert(tz) if tz else dt
        df = df.sort_index()
        validate_ohlcv(df, path)
        # Infer symbol from filename like BTC-USD_1m.csv
        fname = path.split("/")[-1]
        symbol = fname.split("_")[0]
        out[symbol] = df
    return out
\`);

write(path.join(REPO, "src/quant_sim/data/validation.py"), \`
from __future__ import annotations
import pandas as pd

REQUIRED_COLS = ["open", "high", "low", "close", "volume"]

def validate_ohlcv(df: pd.DataFrame, name: str = "data") -> None:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing columns {missing}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"{name}: index must be DatetimeIndex")
    if df.index.has_duplicates:
        raise ValueError(f"{name}: duplicate timestamps found")
    if df[REQUIRED_COLS].isna().any().any():
        raise ValueError(f"{name}: NaNs found in required columns")
    if (df["high"] < df[["open","close","low"]].max(axis=1)).any():
        raise ValueError(f"{name}: invalid OHLC (high too low)")
    if (df["low"] > df[["open","close","high"]].min(axis=1)).any():
        raise ValueError(f"{name}: invalid OHLC (low too high)")
\`);

write(path.join(REPO, "src/quant_sim/data/resample.py"), \`
from __future__ import annotations
import pandas as pd

RULE_MAP = {
    "1s":"1S","10s":"10S","30s":"30S",
    "1m":"1min","5m":"5min","15m":"15min","30m":"30min",
    "1h":"1H","4h":"4H",
    "1d":"1D",
}

def to_rule(tf: str) -> str:
    tf = tf.strip().lower()
    if tf not in RULE_MAP:
        raise ValueError(f"Unsupported timeframe: {tf}. Supported: {list(RULE_MAP)}")
    return RULE_MAP[tf]

def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    rule = to_rule(timeframe)
    agg = {
        "open":"first",
        "high":"max",
        "low":"min",
        "close":"last",
        "volume":"sum",
    }
    out = df.resample(rule).agg(agg).dropna()
    return out
\`);

write(path.join(REPO, "src/quant_sim/data/features.py"), \`
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

def make_features(df: pd.DataFrame, lookbacks: List[int], normalize: bool = True) -> pd.DataFrame:
    # Features are strictly based on past values (shifted where needed)
    close = df["close"].astype(float)
    ret1 = close.pct_change().fillna(0.0)
    feats = {}
    for lb in lookbacks:
        feats[f"ret_{lb}"] = close.pct_change(lb).shift(1)
        feats[f"vol_{lb}"] = ret1.rolling(lb).std().shift(1)
        feats[f"mom_{lb}"] = (close / close.rolling(lb).mean()).shift(1) - 1.0
    X = pd.DataFrame(feats, index=df.index).dropna()
    if normalize:
        mu = X.mean()
        sd = X.std().replace(0.0, 1.0)
        X = (X - mu) / sd
    return X

def make_labels(df: pd.DataFrame, kind: str = "p_up") -> pd.Series:
    close = df["close"].astype(float)
    fwd_ret = close.pct_change().shift(-1)  # next bar return
    if kind == "p_up":
        y = (fwd_ret > 0).astype(int)
    elif kind == "expected_return":
        y = fwd_ret.astype(float)
    else:
        raise ValueError(f"Unknown label kind: {kind}")
    return y
\`);

write(path.join(REPO, "src/quant_sim/strategies/base.py"), \`
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
import pandas as pd

@dataclass
class Signal:
    # desired target position as fraction of equity (e.g., 0.1 long, -0.1 short)
    target_pct: float

class StrategyBase:
    def __init__(self, params: Dict):
        self.params = params

    def generate_signals(self, bars: Dict[str, pd.DataFrame]) -> Dict[str, Signal]:
        raise NotImplementedError
\`);

write(path.join(REPO, "src/quant_sim/strategies/trend.py"), \`
from __future__ import annotations
from typing import Dict
import pandas as pd
from .base import StrategyBase, Signal

class TrendStrategy(StrategyBase):
    def generate_signals(self, bars: Dict[str, pd.DataFrame]) -> Dict[str, Signal]:
        fast = int(self.params.get("fast", 20))
        slow = int(self.params.get("slow", 60))
        out: Dict[str, Signal] = {}
        for sym, df in bars.items():
            c = df["close"].astype(float)
            if len(c) < slow + 2:
                out[sym] = Signal(0.0)
                continue
            ma_fast = c.rolling(fast).mean().iloc[-1]
            ma_slow = c.rolling(slow).mean().iloc[-1]
            # Simple: long if fast > slow, short if fast < slow
            if ma_fast > ma_slow:
                out[sym] = Signal(+0.10)
            elif ma_fast < ma_slow:
                out[sym] = Signal(-0.10)
            else:
                out[sym] = Signal(0.0)
        return out
\`);

write(path.join(REPO, "src/quant_sim/strategies/meanrev.py"), \`
from __future__ import annotations
from typing import Dict
import pandas as pd
from .base import StrategyBase, Signal

class MeanReversionStrategy(StrategyBase):
    def generate_signals(self, bars: Dict[str, pd.DataFrame]) -> Dict[str, Signal]:
        lb = int(self.params.get("lookback", 50))
        z_enter = float(self.params.get("z_enter", 1.5))
        out: Dict[str, Signal] = {}
        for sym, df in bars.items():
            c = df["close"].astype(float)
            if len(c) < lb + 2:
                out[sym] = Signal(0.0)
                continue
            m = c.rolling(lb).mean().iloc[-1]
            s = c.rolling(lb).std().iloc[-1]
            if s == 0 or pd.isna(s):
                out[sym] = Signal(0.0)
                continue
            z = (c.iloc[-1] - m) / s
            if z > z_enter:
                out[sym] = Signal(-0.10)
            elif z < -z_enter:
                out[sym] = Signal(+0.10)
            else:
                out[sym] = Signal(0.0)
        return out
\`);

write(path.join(REPO, "src/quant_sim/strategies/pairs.py"), \`
from __future__ import annotations
from typing import Dict
import pandas as pd
from .base import StrategyBase, Signal

class PairsStrategy(StrategyBase):
    """
    Stub: relative value/pairs trading.
    Extend with cointegration/spread z-score logic.
    """
    def generate_signals(self, bars: Dict[str, pd.DataFrame]) -> Dict[str, Signal]:
        # default: flat
        return {sym: Signal(0.0) for sym in bars.keys()}
\`);

write(path.join(REPO, "src/quant_sim/strategies/ml_pytorch.py"), \`
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from quant_sim.data.features import make_features, make_labels

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: List[int], dropout: float, out_kind: str):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers.append(nn.Linear(d, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)
        self.out_kind = out_kind

    def forward(self, x):
        return self.net(x).squeeze(-1)

@dataclass
class MLConfig:
    lookbacks: List[int]
    normalize: bool
    hidden: List[int]
    dropout: float
    out: str
    train_bars: int
    test_bars: int
    epochs: int
    lr: float
    batch_size: int
    early_stop_patience: int
    seed: int

def _to_cfg(raw: Dict) -> MLConfig:
    f = raw.get("features", {})
    m = raw.get("model", {})
    t = raw.get("training", {})
    return MLConfig(
        lookbacks=list(f.get("lookbacks", [5,10,20,60])),
        normalize=bool(f.get("normalize", True)),
        hidden=list(m.get("hidden", [64,64])),
        dropout=float(m.get("dropout", 0.1)),
        out=str(m.get("out", "p_up")),
        train_bars=int(t.get("train_bars", 2000)),
        test_bars=int(t.get("test_bars", 500)),
        epochs=int(t.get("epochs", 10)),
        lr=float(t.get("lr", 1e-3)),
        batch_size=int(t.get("batch_size", 256)),
        early_stop_patience=int(t.get("early_stop_patience", 2)),
        seed=int(t.get("seed", 42)),
    )

class PyTorchMLSignal:
    """
    Walk-forward ML signal generator.
    Produces per-timestamp prediction series aligned to bars index.
    """
    def __init__(self, ml_raw_cfg: Dict):
        self.cfg = _to_cfg(ml_raw_cfg)
        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)

    def walk_forward_predict(self, df: pd.DataFrame) -> pd.Series:
        cfg = self.cfg
        X = make_features(df, cfg.lookbacks, normalize=cfg.normalize)
        y = make_labels(df.loc[X.index], kind=cfg.out).loc[X.index]
        # drop last label NaN (since it uses fwd return)
        valid = y.dropna().index
        X = X.loc[valid]
        y = y.loc[valid]

        if len(X) < (cfg.train_bars + cfg.test_bars):
            raise ValueError(f"Not enough bars for walk-forward. Need >= {cfg.train_bars+cfg.test_bars}, got {len(X)}")

        preds = pd.Series(index=X.index, dtype=float)

        start = 0
        while (start + cfg.train_bars + cfg.test_bars) <= len(X):
            tr = slice(start, start + cfg.train_bars)
            te = slice(start + cfg.train_bars, start + cfg.train_bars + cfg.test_bars)

            Xtr = torch.tensor(X.iloc[tr].values, dtype=torch.float32)
            Xte = torch.tensor(X.iloc[te].values, dtype=torch.float32)
            ytr_np = y.iloc[tr].values

            if cfg.out == "p_up":
                ytr = torch.tensor(ytr_np.astype(np.float32), dtype=torch.float32)
                loss_fn = nn.BCEWithLogitsLoss()
            else:
                ytr = torch.tensor(ytr_np.astype(np.float32), dtype=torch.float32)
                loss_fn = nn.MSELoss()

            model = MLP(in_dim=Xtr.shape[1], hidden=cfg.hidden, dropout=cfg.dropout, out_kind=cfg.out)
            opt = optim.Adam(model.parameters(), lr=cfg.lr)

            best_loss = float("inf")
            patience = 0

            # mini-batch training
            n = Xtr.shape[0]
            idx = np.arange(n)

            for epoch in range(cfg.epochs):
                np.random.shuffle(idx)
                model.train()
                total = 0.0
                for i in range(0, n, cfg.batch_size):
                    j = idx[i:i+cfg.batch_size]
                    xb = Xtr[j]
                    yb = ytr[j]
                    opt.zero_grad()
                    logits = model(xb)
                    loss = loss_fn(logits, yb)
                    loss.backward()
                    opt.step()
                    total += float(loss.item())

                # simple early stop based on training loss
                if total < best_loss - 1e-6:
                    best_loss = total
                    patience = 0
                else:
                    patience += 1
                    if patience >= cfg.early_stop_patience:
                        break

            model.eval()
            with torch.no_grad():
                raw = model(Xte).cpu().numpy()
                if cfg.out == "p_up":
                    p = 1.0 / (1.0 + np.exp(-raw))
                    preds.iloc[start + cfg.train_bars : start + cfg.train_bars + cfg.test_bars] = p
                else:
                    preds.iloc[start + cfg.train_bars : start + cfg.train_bars + cfg.test_bars] = raw

            start += cfg.test_bars

        return preds
\`);

write(path.join(REPO, "src/quant_sim/strategies/ensemble.py"), \`
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
\`);

write(path.join(REPO, "src/quant_sim/execution/orders.py"), \`
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum

class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"

@dataclass
class Order:
    symbol: str
    qty: float               # positive buy, negative sell
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    stop_price: float | None = None
\`);

write(path.join(REPO, "src/quant_sim/execution/costs.py"), \`
from __future__ import annotations

def apply_bps(price: float, bps: float, side: float) -> float:
    # side > 0 buy pays more; side < 0 sell receives less
    # bps are basis points (1 bps = 0.01%)
    adj = (bps / 10000.0) * price
    return price + (adj if side > 0 else -adj)

def commission_cost(notional: float, commission_bps: float) -> float:
    return abs(notional) * (commission_bps / 10000.0)
\`);

write(path.join(REPO, "src/quant_sim/execution/broker_base.py"), \`
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
\`);

write(path.join(REPO, "src/quant_sim/execution/sim_broker.py"), \`
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
\`);

write(path.join(REPO, "src/quant_sim/execution/coinbase_broker.py"), \`
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
\`);

write(path.join(REPO, "src/quant_sim/portfolio/portfolio.py"), \`
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
\`);

write(path.join(REPO, "src/quant_sim/portfolio/risk.py"), \`
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
\`);

write(path.join(REPO, "src/quant_sim/portfolio/sizing.py"), \`
from __future__ import annotations
from typing import Dict
import math

def target_pct_to_qty(target_pct: float, equity: float, price: float) -> float:
    if price <= 0:
        return 0.0
    target_notional = target_pct * equity
    return target_notional / price
\`);

write(path.join(REPO, "src/quant_sim/analytics/metrics.py"), \`
from __future__ import annotations
import pandas as pd

def compute_metrics(equity_curve: pd.DataFrame) -> dict:
    if equity_curve.empty:
        return {}
    eq = equity_curve["equity"].astype(float)
    rets = eq.pct_change().fillna(0.0)
    cum_ret = (eq.iloc[-1] / eq.iloc[0]) - 1.0 if eq.iloc[0] != 0 else 0.0
    peak = eq.cummax()
    dd = (peak - eq) / peak.replace(0.0, 1.0)
    max_dd = float(dd.max())
    vol = float(rets.std())
    return {
        "start_equity": float(eq.iloc[0]),
        "end_equity": float(eq.iloc[-1]),
        "cumulative_return": float(cum_ret),
        "max_drawdown": max_dd,
        "return_volatility": vol,
        "num_points": int(len(eq)),
    }
\`);

write(path.join(REPO, "src/quant_sim/analytics/report_csv.py"), \`
from __future__ import annotations
import os
import pandas as pd

def write_csv(run_dir: str, name: str, df: pd.DataFrame) -> str:
    path = os.path.join(run_dir, f"{name}.csv")
    df.to_csv(path, index=False)
    return path
\`);

write(path.join(REPO, "src/quant_sim/analytics/report_html.py"), \`
from __future__ import annotations
import os
import pandas as pd
import matplotlib.pyplot as plt
from jinja2 import Template

HTML_TMPL = """
<!doctype html>

<html>
<head>
  <meta charset="utf-8"/>
  <title>Quant Sim Report</title>
  <style>
    body { font-family: -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 24px; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
    .card { border: 1px solid #ddd; border-radius: 10px; padding: 14px; }
    table { width: 100%; border-collapse: collapse; }
    th, td { border-bottom: 1px solid #eee; text-align: left; padding: 6px; font-size: 13px; }
    h1 { margin: 0 0 8px 0; }
    .muted { color: #666; font-size: 13px; }
    img { max-width: 100%; border-radius: 10px; border: 1px solid #eee; }
    code { background: #f6f6f6; padding: 2px 6px; border-radius: 6px; }
  </style>
</head>
<body>
  <h1>Quant Simulator Report</h1>
  <div class="muted">Run: <code>{{ run_id }}</code></div>
  <div class="grid" style="margin-top:16px;">
    <div class="card">
      <h3>Metrics</h3>
      <table>
        {% for k, v in metrics.items() %}
        <tr><th>{{ k }}</th><td>{{ v }}</td></tr>
        {% endfor %}
      </table>
    </div>
    <div class="card">
      <h3>Equity Curve</h3>
      <img src="{{ equity_png }}" alt="Equity Curve"/>
    </div>
  </div>


  <div class="card" style="margin-top:16px;">
    <h3>Recent Trades</h3>
    <table>
      <tr>
        {% for col in trades_cols %}<th>{{ col }}</th>{% endfor %}
      </tr>
      {% for row in trades_rows %}
      <tr>
        {% for col in trades_cols %}<td>{{ row[col] }}</td>{% endfor %}
      </tr>
      {% endfor %}
    </table>
  </div>
</body>
</html>
"""


def _plot_equity(equity_curve: pd.DataFrame, png_path: str) -> None:
    plt.figure()
    plt.plot(pd.to_datetime(equity_curve["timestamp"]), equity_curve["equity"].astype(float))
    plt.xlabel("Time")
    plt.ylabel("Equity")
    plt.tight_layout()
    plt.savefig(png_path, dpi=140)
    plt.close()

def write_html_report(run_dir: str, run_id: str, metrics: dict, equity_curve: pd.DataFrame, trades: pd.DataFrame) -> str:
    png_path = os.path.join(run_dir, "equity_curve.png")
    _plot_equity(equity_curve, png_path)

    tmpl = Template(HTML_TMPL)
    trades_cols = list(trades.columns) if not trades.empty else ["(no trades)"]
    trades_rows = trades.tail(50).to_dict(orient="records") if not trades.empty else [{"(no trades)": ""}]

    html = tmpl.render(
        run_id=run_id,
        metrics=metrics,
        equity_png=os.path.basename(png_path),
        trades_cols=trades_cols,
        trades_rows=trades_rows,
    )
    out = os.path.join(run_dir, "report.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    return out
\`);

write(path.join(REPO, "src/quant_sim/cli.py"), \`
from __future__ import annotations
import argparse
import os
import time
from typing import Dict, Optional, Tuple
import pandas as pd

from quant_sim.config import load_yaml, Config
from quant_sim.run_artifacts import make_run_dir, snapshot_config, save_json
from quant_sim.utils.logging import setup_logger

from quant_sim.data.provider import load_csv_ohlcv
from quant_sim.data.resample import resample_ohlcv

from quant_sim.strategies.trend import TrendStrategy
from quant_sim.strategies.meanrev import MeanReversionStrategy
from quant_sim.strategies.pairs import PairsStrategy
from quant_sim.strategies.ensemble import EnsembleStrategy
from quant_sim.strategies.ml_pytorch import PyTorchMLSignal

from quant_sim.execution.orders import Order
from quant_sim.execution.sim_broker import SimBroker
from quant_sim.execution.coinbase_broker import CoinbasePaperBroker

from quant_sim.portfolio.portfolio import Portfolio
from quant_sim.portfolio.risk import RiskLimits, clamp_target, check_daily_loss
from quant_sim.portfolio.sizing import target_pct_to_qty

from quant_sim.analytics.metrics import compute_metrics
from quant_sim.analytics.report_csv import write_csv
from quant_sim.analytics.report_html import write_html_report

STRAT_REGISTRY = {
    "trend": TrendStrategy,
    "meanrev": MeanReversionStrategy,
    "pairs": PairsStrategy,
    "ensemble": EnsembleStrategy,
}

def _build_strategy(cfg: Config):
    name = str(cfg.get("strategy", "name", default="trend")).lower()
    params = cfg.get("strategy", "params", default={}) or {}
    if name == "ensemble":
        return EnsembleStrategy(params, registry=STRAT_REGISTRY)
    if name not in STRAT_REGISTRY:
        raise ValueError(f"Unknown strategy: {name}. Options: {list(STRAT_REGISTRY.keys())}")
    return STRAT_REGISTRY[name](params)

def _risk_limits(cfg: Config) -> RiskLimits:
    r = cfg.get("risk", default={}) or {}
    return RiskLimits(
        max_gross_leverage=float(r.get("max_gross_leverage", 2.0)),
        max_position_pct=float(r.get("max_position_pct", 0.25)),
        max_daily_loss_pct=float(r.get("max_daily_loss_pct", 0.05)),
        kill_switch=bool(r.get("kill_switch", True)),
    )

def _prices_from_bars(bars: Dict[str, pd.DataFrame], ts: pd.Timestamp) -> Dict[str, float]:
    out = {}
    for sym, df in bars.items():
        s = df.loc[:ts, "close"]
        if len(s) > 0:
            out[sym] = float(s.iloc[-1])
    return out

def _align_bars_to_timeframe(bars_1m: Dict[str, pd.DataFrame], tf: str) -> Dict[str, pd.DataFrame]:
    if tf.lower() == "1m":
        return bars_1m
    return {sym: resample_ohlcv(df, tf) for sym, df in bars_1m.items()}

def run_backtest(config_path: str, ml_path_override: Optional[str] = None) -> str:
    raw = load_yaml(config_path)
    cfg = Config(raw)

    run_name = str(cfg.get("run", "name", default="run"))
    out_dir = str(cfg.get("run", "out_dir", default="runs"))
    artifacts = make_run_dir(out_dir, run_name)
    snapshot_config(config_path, artifacts.run_dir)
    logger = setup_logger("quant_sim", artifacts.logs_path)
    logger.info(f"Starting backtest: {run_name} -> {artifacts.run_dir}")

    # data
    csv_files = cfg.get("data", "csv_files", default=[]) or []
    if not csv_files:
        raise ValueError("No CSV files configured. Set data.csv_files in config.")
    bars_1m = load_csv_ohlcv(
        csv_files=csv_files,
        timestamp_col=str(cfg.get("data", "timestamp_col", default="timestamp")),
        tz=str(cfg.get("data", "tz", default="UTC")),
    )
    universe = cfg.get("universe", "symbols", default=list(bars_1m.keys())) or list(bars_1m.keys())
    bars_1m = {sym: bars_1m[sym] for sym in universe if sym in bars_1m}

    decision_tf = str(cfg.get("strategy", "timeframe", default="5m"))
    bars = _align_bars_to_timeframe(bars_1m, decision_tf)

    # broker
    commission_bps = float(cfg.get("execution", "commission_bps", default=2.0))
    slippage_bps = float(cfg.get("execution", "slippage_bps", default=1.0))
    broker = SimBroker(bars_1m, commission_bps=commission_bps, slippage_bps=slippage_bps)

    # portfolio
    starting_cash = float(cfg.get("portfolio", "starting_cash", default=100000.0))
    pf = Portfolio(cash=starting_cash)
    limits = _risk_limits(cfg)

    # ML
    ml_enabled = bool(cfg.get("ml", "enabled", default=False))
    ml_cfg_path = ml_path_override or cfg.get("ml", "config", default=None)
    ml_models: Dict[str, PyTorchMLSignal] = {}
    ml_preds: Dict[str, pd.Series] = {}

    if ml_enabled:
        if not ml_cfg_path:
            raise ValueError("ml.enabled is true but no ml.config provided.")
        ml_raw = load_yaml(str(ml_cfg_path))
        logger.info(f"ML enabled. Using config: {ml_cfg_path}")
        for sym, df1m in bars_1m.items():
            sig = PyTorchMLSignal(ml_raw)
            preds = sig.walk_forward_predict(df1m)
            ml_models[sym] = sig
            ml_preds[sym] = preds
        logger.info("ML walk-forward predictions generated.")

    strat = _build_strategy(cfg)

    # iterate
    common_index = None
    for sym, df in bars.items():
        common_index = df.index if common_index is None else common_index.intersection(df.index)
    if common_index is None or len(common_index) < 5:
        raise ValueError("Not enough aligned bars to run backtest.")

    trades = []
    equity_start_day = starting_cash
    day_anchor = pd.to_datetime(common_index[0]).date()

    for ts in common_index:
        if pd.to_datetime(ts).date() != day_anchor:
            last_eq = pf.equity_curve[-1]["equity"] if pf.equity_curve else starting_cash
            equity_start_day = float(last_eq)
            day_anchor = pd.to_datetime(ts).date()

        prices = _prices_from_bars(bars_1m, ts)
        row = pf.mark_to_market(prices, ts)

        if limits.kill_switch and check_daily_loss(equity_start_day, row["equity"], limits):
            logger.warning("Kill-switch triggered: max daily loss reached. Flattening positions and stopping.")
            for sym, pos in list(pf.positions.items()):
                if pos.qty != 0:
                    fill = broker.place_order(Order(sym, qty=-pos.qty), ts=ts)
                    pf.update_from_fill(fill)
                    trades.append(fill)
            break

        current_bars = {sym: df.loc[:ts] for sym, df in bars.items()}
        sigs = strat.generate_signals(current_bars)

        equity = float(row["equity"])
        for sym, sig in sigs.items():
            px = prices.get(sym)
            if px is None:
                continue

            target_pct = clamp_target(sym, sig.target_pct, equity, px, limits)

            if ml_enabled and sym in ml_preds:
                p = ml_preds[sym].loc[:ts].tail(1)
                if len(p) == 1:
                    val = float(p.iloc[0])
                    if str(load_yaml(str(ml_cfg_path)).get("model", {}).get("out", "p_up")) == "p_up":
                        scale = max(min((val - 0.5) * 2.0, 1.0), -1.0)
                        target_pct = target_pct * scale
                    else:
                        scale = max(min(val * 10.0, 1.0), -1.0)
                        target_pct = target_pct * scale

            current_qty = pf.position_qty(sym)
            target_qty = target_pct_to_qty(target_pct, equity, px)
            delta = target_qty - current_qty

            if abs(delta) * px < 10.0:
                continue

            fill = broker.place_order(Order(sym, qty=delta), ts=ts)
            pf.update_from_fill(fill)
            trades.append(fill)

    equity_df = pd.DataFrame(pf.equity_curve)
    trades_df = pd.DataFrame(trades)

    metrics = compute_metrics(equity_df) if not equity_df.empty else {}
    save_json(os.path.join(artifacts.run_dir, "metrics.json"), metrics)

    if bool(cfg.get("reports", "csv", default=True)):
        if not trades_df.empty:
            write_csv(artifacts.run_dir, "trades", trades_df)
        if not equity_df.empty:
            write_csv(artifacts.run_dir, "equity_curve", equity_df)

    if bool(cfg.get("reports", "html", default=True)) and not equity_df.empty:
        write_html_report(artifacts.run_dir, artifacts.run_id, metrics, equity_df, trades_df)

    logger.info(f"Backtest complete. Metrics: {metrics}")
    logger.info(f"Artifacts: {artifacts.run_dir}")
    return artifacts.run_dir

def run_paper(config_path: str, ml_path_override: Optional[str] = None) -> str:
    raw = load_yaml(config_path)
    cfg = Config(raw)

    run_name = str(cfg.get("run", "name", default="paper"))
    out_dir = str(cfg.get("run", "out_dir", default="runs"))
    artifacts = make_run_dir(out_dir, run_name)
    snapshot_config(config_path, artifacts.run_dir)
    logger = setup_logger("quant_sim", artifacts.logs_path)
    logger.info(f"Starting paper mode: {run_name} -> {artifacts.run_dir}")

    products = cfg.get("coinbase", "products", default=["BTC-USD"]) or ["BTC-USD"]
    commission_bps = float(cfg.get("execution", "commission_bps", default=2.0))
    slippage_bps = float(cfg.get("execution", "slippage_bps", default=1.0))
    broker = CoinbasePaperBroker(products, commission_bps=commission_bps, slippage_bps=slippage_bps)

    starting_cash = float(cfg.get("portfolio", "starting_cash", default=100000.0))
    pf = Portfolio(cash=starting_cash)
    limits = _risk_limits(cfg)
    strat = _build_strategy(cfg)

    poll_seconds = float(cfg.get("paper", "poll_seconds", default=3))
    decision_tf = str(cfg.get("strategy", "timeframe", default="5m")).lower()
    decision_rule = {"1m":60, "5m":300, "15m":900, "1h":3600}.get(decision_tf, 300)

    ml_enabled = bool(cfg.get("ml", "enabled", default=False))
    ml_cfg_path = ml_path_override or cfg.get("ml", "config", default=None)
    ml_raw = load_yaml(str(ml_cfg_path)) if (ml_enabled and ml_cfg_path) else None
    ml_signal = PyTorchMLSignal(ml_raw) if ml_enabled else None

    bars_1m: Dict[str, pd.DataFrame] = {p: pd.DataFrame(columns=["open","high","low","close","volume"]) for p in products}
    trades = []

    equity_start_day = starting_cash
    day_anchor = pd.Timestamp.utcnow().date()
    last_decision = time.time()

    logger.info("Paper mode running. Press Ctrl+C to stop.")
    try:
        while True:
            now = pd.Timestamp.utcnow().tz_localize("UTC")
            if now.date() != day_anchor:
                last_eq = pf.equity_curve[-1]["equity"] if pf.equity_curve else starting_cash
                equity_start_day = float(last_eq)
                day_anchor = now.date()

            prices = {}
            for sym in products:
                px = broker.get_price(sym)
                prices[sym] = float(px)
                df = bars_1m[sym]
                minute = now.floor("min")
                if df.empty or (df.index[-1] != minute):
                    new = pd.DataFrame(
                        {"open":[px], "high":[px], "low":[px], "close":[px], "volume":[0.0]},
                        index=pd.DatetimeIndex([minute], tz="UTC")
                    )
                    bars_1m[sym] = pd.concat([df, new]).tail(5000)
                else:
                    bars_1m[sym].loc[minute, "high"] = max(float(bars_1m[sym].loc[minute, "high"]), px)
                    bars_1m[sym].loc[minute, "low"] = min(float(bars_1m[sym].loc[minute, "low"]), px)
                    bars_1m[sym].loc[minute, "close"] = px

            row = pf.mark_to_market(prices, now)

            if limits.kill_switch and check_daily_loss(equity_start_day, row["equity"], limits):
                logger.warning("Kill-switch triggered: max daily loss reached. Flattening positions and stopping.")
                for sym, pos in list(pf.positions.items()):
                    if pos.qty != 0:
                        fill = broker.place_order(Order(sym, qty=-pos.qty), ts=now)
                        pf.update_from_fill(fill)
                        trades.append(fill)
                break

            if time.time() - last_decision >= decision_rule:
                last_decision = time.time()

                decision_bars = {}
                for sym, df in bars_1m.items():
                    if len(df) < 120:
                        continue
                    if decision_tf == "1m":
                        decision_bars[sym] = df.copy()
                    else:
                        decision_bars[sym] = resample_ohlcv(df, decision_tf)

                if not decision_bars:
                    time.sleep(poll_seconds)
                    continue

                sigs = strat.generate_signals(decision_bars)
                equity = float(row["equity"])

                for sym, sig in sigs.items():
                    px = prices.get(sym)
                    if px is None:
                        continue
                    target_pct = clamp_target(sym, sig.target_pct, equity, px, limits)

                    current_qty = pf.position_qty(sym)
                    target_qty = target_pct_to_qty(target_pct, equity, px)
                    delta = target_qty - current_qty
                    if abs(delta) * px < 10.0:
                        continue

                    fill = broker.place_order(Order(sym, qty=delta), ts=now)
                    pf.update_from_fill(fill)
                    trades.append(fill)

                logger.info(f"Decision executed at {now.isoformat()} | equity={equity:.2f}")

            time.sleep(poll_seconds)

    except KeyboardInterrupt:
        logger.info("Stopping (Ctrl+C). Writing artifacts...")

    equity_df = pd.DataFrame(pf.equity_curve)
    trades_df = pd.DataFrame(trades)

    metrics = compute_metrics(equity_df) if not equity_df.empty else {}
    save_json(os.path.join(artifacts.run_dir, "metrics.json"), metrics)

    if bool(cfg.get("reports", "csv", default=True)):
        if not trades_df.empty:
            write_csv(artifacts.run_dir, "trades", trades_df)
        if not equity_df.empty:
            write_csv(artifacts.run_dir, "equity_curve", equity_df)

    if bool(cfg.get("reports", "html", default=True)) and not equity_df.empty:
        write_html_report(artifacts.run_dir, artifacts.run_id, metrics, equity_df, trades_df)

    logger.info(f"Paper run complete. Metrics: {metrics}")
    logger.info(f"Artifacts: {artifacts.run_dir}")
    return artifacts.run_dir

def run_report(run_dir: str) -> str:
    eq_path = os.path.join(run_dir, "equity_curve.csv")
    tr_path = os.path.join(run_dir, "trades.csv")
    m_path = os.path.join(run_dir, "metrics.json")

    equity_df = pd.read_csv(eq_path) if os.path.exists(eq_path) else pd.DataFrame()
    trades_df = pd.read_csv(tr_path) if os.path.exists(tr_path) else pd.DataFrame()

    import json
    metrics = {}
    if os.path.exists(m_path):
        with open(m_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)

    run_id = os.path.basename(run_dir.rstrip("/"))
    if equity_df.empty:
        raise ValueError("No equity_curve.csv found to render report.")
    path = write_html_report(run_dir, run_id, metrics, equity_df, trades_df)
    return path

def main():
    ap = argparse.ArgumentParser(prog="quant-sim")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_bt = sub.add_parser("backtest", help="Run historical backtest from CSV OHLCV files")
    ap_bt.add_argument("--config", required=True, help="Path to YAML config")
    ap_bt.add_argument("--ml", required=False, help="Optional ML YAML config override")

    ap_pp = sub.add_parser("paper", help="Run paper trading loop using Coinbase public price feed")
    ap_pp.add_argument("--config", required=True, help="Path to YAML config")
    ap_pp.add_argument("--ml", required=False, help="Optional ML YAML config override")

    ap_rp = sub.add_parser("report", help="Re-render HTML report for an existing run directory")
    ap_rp.add_argument("--run", required=True, help="Path to run directory (runs/<run_id>/)")

    args = ap.parse_args()

    if args.cmd == "backtest":
        run_backtest(args.config, ml_path_override=args.ml)
    elif args.cmd == "paper":
        run_paper(args.config, ml_path_override=args.ml)
    elif args.cmd == "report":
        path = run_report(args.run)
        print(path)

if __name__ == "__main__":
    main()
\`);
