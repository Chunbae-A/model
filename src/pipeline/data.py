from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import model_config as config


def load_model_input(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Model input file not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Model input file is empty: {path}")
    return df


def check_required_columns(df: pd.DataFrame) -> None:
    required = set(config.ID_COLUMNS + [config.SPLIT_COLUMN, config.REGRESSION_TARGET, config.CLASSIFICATION_TARGET])
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"Required columns are missing: {missing}")


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    check_required_columns(df)
    drop_columns = [col for col in config.DROP_COLUMNS if col in df.columns]
    feature_columns = [col for col in df.columns if col not in drop_columns]

    suspicious = [
        col
        for col in feature_columns
        if any(keyword.lower() in col.lower() for keyword in config.FORBIDDEN_FEATURE_KEYWORDS)
    ]
    if suspicious:
        raise ValueError(f"Leakage-prone feature columns found: {suspicious}")

    if config.REQUIRE_NUMERIC_FEATURES:
        non_numeric = [col for col in feature_columns if not pd.api.types.is_numeric_dtype(df[col])]
        if non_numeric:
            raise TypeError(f"Non-numeric feature columns found: {non_numeric}")

    return feature_columns


def time_based_train_valid_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    check_required_columns(df)

    if config.SPLIT_COLUMN in df.columns:
        train_df = df[df[config.SPLIT_COLUMN].eq("train")].copy()
        valid_df = df[df[config.SPLIT_COLUMN].isin(["valid", "validation"])].copy()
        if train_df.empty or valid_df.empty:
            raise ValueError("split column exists, but train or valid rows are empty.")
        return train_df, valid_df

    sorted_df = df.copy()
    sorted_df[config.DATE_COLUMN] = pd.to_datetime(sorted_df[config.DATE_COLUMN])
    unique_dates = pd.Series(sorted_df[config.DATE_COLUMN].sort_values().unique())
    split_idx = int(len(unique_dates) * (1 - config.VALID_SIZE_RATIO))
    valid_start = unique_dates.iloc[split_idx]
    return (
        sorted_df[sorted_df[config.DATE_COLUMN] < valid_start].copy(),
        sorted_df[sorted_df[config.DATE_COLUMN] >= valid_start].copy(),
    )
