from __future__ import annotations
import os
import pandas as pd

def write_csv(run_dir: str, name: str, df: pd.DataFrame) -> str:
    path = os.path.join(run_dir, f"{name}.csv")
    df.to_csv(path, index=False)
    return path
