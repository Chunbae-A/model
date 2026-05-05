from __future__ import annotations

from pathlib import Path
import pandas as pd

def load_model_input(path: Path) -> pd.DataFrame:
    """Read preprocessed model input table (csv or excel)."""
    if not path.exists():
        raise FileNotFoundError(f"Model input file not found: {path}")

    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    return pd.read_csv(path)
