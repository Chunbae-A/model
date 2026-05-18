from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def resolve_project_path(path: Path) -> Path:
    """Resolve repo-relative paths even when scripts are launched from elsewhere."""
    path = Path(path)
    if path.is_absolute():
        return path
    return ROOT / path


def load_model_input(path: Path) -> pd.DataFrame:
    """Read preprocessed model input table (csv or excel)."""
    path = resolve_project_path(path)
    if not path.exists():
        raise FileNotFoundError(
            "Model input file not found: "
            f"{path}\nRun `python src/pipeline.py --skip-train` first, or provide data/Final.csv."
        )

    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    return pd.read_csv(path, encoding="utf-8-sig")
