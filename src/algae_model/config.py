from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


TARGET_COL = "유해남조류 세포수 (cells/㎖) (1+2+3+4)"
SITE_ORDER = {"회남": 0, "추동": 1, "문의": 2}
SITE_TO_STATION = {"회남": "보은", "추동": "대전", "문의": "청주"}
UPSTREAM = {"회남": None, "추동": "회남", "문의": "추동"}


@dataclass(frozen=True)
class DataPaths:
    quality: Path
    dam: Path
    kma: Path | None = None
    geum_dam: Path | None = None


@dataclass(frozen=True)
class TrainConfig:
    data_paths: DataPaths
    output_dir: Path
    train_end_year: int = 2022
    valid_year: int = 2023
    test_start_year: int = 2024
    random_state: int = 42


def default_config() -> TrainConfig:
    """Build config from environment variables.

    Required:
      QUALITY_CSV, DAM_CSV

    Optional:
      KMA_CSV, GEUM_DAM_CSV, MODEL_OUTPUT_DIR
    """
    quality = os.getenv("QUALITY_CSV")
    dam = os.getenv("DAM_CSV")
    if not quality or not dam:
        raise ValueError(
            "QUALITY_CSV and DAM_CSV are required. "
            "Example: QUALITY_CSV=data/processed/... DAM_CSV=data/processed/... python scripts/train_model.py"
        )
    return TrainConfig(
        data_paths=DataPaths(
            quality=Path(quality),
            dam=Path(dam),
            kma=Path(os.getenv("KMA_CSV")) if os.getenv("KMA_CSV") else None,
            geum_dam=Path(os.getenv("GEUM_DAM_CSV")) if os.getenv("GEUM_DAM_CSV") else None,
        ),
        output_dir=Path(os.getenv("MODEL_OUTPUT_DIR", "outputs/model_pipeline")),
    )
