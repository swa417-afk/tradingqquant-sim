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
