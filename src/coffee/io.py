from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

SENTINEL = -2147483648  # your R sentinel for missing

def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    df = pd.read_csv(path)
    df.replace(SENTINEL, np.nan, inplace=True)
    return df
