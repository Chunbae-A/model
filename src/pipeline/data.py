from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import model_config as config


def load_model_input(path: Path) -> pd.DataFrame:
    """모델 입력 CSV를 읽고 최소한의 유효성만 확인한다.

    세부 전처리는 `src/utils/build_model_datasets.py`에서 이미 끝난 상태를 가정한다.
    여기서는 파일 존재 여부와 빈 파일 여부만 확인해, 이후 학습 단계에서
    원인을 알기 어려운 에러가 나지 않도록 빠르게 실패시킨다.
    """

    if not path.exists():
        raise FileNotFoundError(f"Model input file not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Model input file is empty: {path}")
    return df


def check_required_columns(df: pd.DataFrame) -> None:
    """학습 파이프라인이 반드시 필요로 하는 공통 컬럼을 검사한다."""

    required = set(config.ID_COLUMNS + [config.SPLIT_COLUMN, config.REGRESSION_TARGET, config.CLASSIFICATION_TARGET])
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"Required columns are missing: {missing}")


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """학습에 사용할 feature 목록을 만든다.

    핵심은 정답 누수를 막는 것이다. 날짜, split, target 컬럼은 feature에서
    제외하고, target 이름이 포함된 의심 컬럼도 한 번 더 차단한다.
    현재 모델 후보는 모두 수치형 입력을 기대하므로 non-numeric feature도
    여기서 명시적으로 실패시킨다.
    """

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
    """학습/검증 데이터를 나눈다.

    우선 입력 데이터에 이미 만들어둔 `split` 컬럼을 신뢰한다. 이 프로젝트의
    데이터는 한 조사일이 여러 loc/station 행으로 확장되므로 랜덤 split을 쓰면
    같은 날짜의 사건이 train과 valid에 동시에 들어갈 수 있다. 그래서 split이
    없을 때도 날짜 순서 기준 holdout을 만들어, 미래 구간 검증에 가깝게 나눈다.
    """

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
