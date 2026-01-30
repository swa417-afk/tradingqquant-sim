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
