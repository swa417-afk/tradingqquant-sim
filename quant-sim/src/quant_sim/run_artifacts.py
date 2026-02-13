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
