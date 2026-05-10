from pathlib import Path
import numpy as np
import pandas as pd
from src.config import (
    SCHEMA_COLUMN_NAME,
    SCHEMA_ROLE_NAME,
    FEATURE_ROLE_VALUE,
    REGRESSION_TARGET, 
    CLASSIFICATION_TARGET,
    FORBIDDEN_FEATURE_KEYWORDS,
    REQUIRE_NUMERIC_FEATURES
)

def load_model_input(path: Path) -> pd.DataFrame:
    """전처리 완료 모델 입력 테이블을 읽습니다.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Model input file not found: {path}. "
        )

    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path)

    return pd.read_csv(path)


def load_model_input_schema(schema_path: Path) -> pd.DataFrame | None:
    """선택 입력값인 모델 입력 schema 파일을 읽습니다.

    schema는 feature engineering이 아니라, 전처리팀과 모델링팀 사이에서
    어떤 컬럼을 feature로 사용할지 명시하는 안전 계약입니다.
    """
    if not schema_path.exists():
        return None
    return pd.read_csv(schema_path)


def get_feature_columns_from_schema(
    df: pd.DataFrame,
    schema_df: pd.DataFrame,
    column_name_col: str = "column_name",
    role_col: str = "role",
    feature_role_value: str = "feature",
) -> list[str]:
    required_cols = {column_name_col, role_col}
    missing = required_cols - set(schema_df.columns)
    if missing:
        raise KeyError(f"schema 파일에 필수 컬럼이 없습니다: {sorted(missing)}")

    feature_columns = (
        schema_df.loc[schema_df[role_col].eq(feature_role_value), column_name_col]
        .dropna()
        .astype(str)
        .tolist()
    )

    missing_in_df = [col for col in feature_columns if col not in df.columns]
    if missing_in_df:
        raise KeyError(f"schema에서 feature로 승인된 컬럼이 모델 입력 테이블에 없습니다: {missing_in_df}")

    return feature_columns


def get_feature_columns_by_drop_rule(df: pd.DataFrame, drop_columns: list[str]) -> list[str]:
    existing_drop_columns = [col for col in drop_columns if col in df.columns]
    feature_columns = [col for col in df.columns if col not in existing_drop_columns]
    return feature_columns


def check_leakage_columns(
    feature_columns: list[str],
    forbidden_keywords: list[str],
) -> None:
    suspicious = [
        col for col in feature_columns
        if any(keyword in col.lower() for keyword in forbidden_keywords)
    ]
    if suspicious:
        raise ValueError(
            "누수 의심 컬럼이 feature에 포함되어 있습니다. "
            f"schema 또는 DROP_COLUMNS를 확인하세요: {suspicious}"
        )
    

def check_required_targets(
    df: pd.DataFrame,
    regression_target: str,
    classification_target: str,
) -> None:
    missing = [col for col in [regression_target, classification_target] if col not in df.columns]
    if missing:
        raise KeyError(f"모델 입력 테이블에 target 컬럼이 없습니다: {missing}")


def check_numeric_features(df: pd.DataFrame, feature_columns: list[str]) -> None:
    non_numeric = [col for col in feature_columns if not pd.api.types.is_numeric_dtype(df[col])]
    if non_numeric:
        raise TypeError(
            "모델링 노트북은 전처리 완료 feature table을 입력으로 받습니다. "
            "문자형/categorical feature는 전처리팀에서 encoding 후 전달해야 합니다. "
            f"Non-numeric feature columns: {non_numeric}"
        )


def get_feature_columns(
    df: pd.DataFrame,
    drop_columns: list[str],
    schema_path: Path | None = None,
) -> list[str]:
    schema_df = load_model_input_schema(schema_path) if schema_path is not None else None

    if schema_df is not None:
        feature_columns = get_feature_columns_from_schema(
            df,
            schema_df,
            column_name_col=SCHEMA_COLUMN_NAME,
            role_col=SCHEMA_ROLE_NAME,
            feature_role_value=FEATURE_ROLE_VALUE,
        )
    else:
        feature_columns = get_feature_columns_by_drop_rule(df, drop_columns)

    check_required_targets(df, REGRESSION_TARGET, CLASSIFICATION_TARGET)
    check_leakage_columns(feature_columns, FORBIDDEN_FEATURE_KEYWORDS)

    if REQUIRE_NUMERIC_FEATURES:
        check_numeric_features(df, feature_columns)

    return feature_columns


def split_features_targets(
    df: pd.DataFrame,
    feature_columns: list[str],
    regression_target: str,
    classification_target: str,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    check_required_targets(df, regression_target, classification_target)

    x = df[feature_columns]
    y_reg = df[regression_target]
    y_cls = df[classification_target]
    return x, y_reg, y_cls


def time_based_train_valid_split(
    df: pd.DataFrame,
    date_column: str,
    split_column: str | None = None,
    valid_size_ratio: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if split_column and split_column in df.columns:
        train_df = df[df[split_column] == "train"].copy()
        valid_df = df[df[split_column].isin(["valid", "validation"])].copy()
        if len(train_df) == 0 or len(valid_df) == 0:
            raise ValueError("split column exists, but train/valid rows are empty. Check split labels.")
        return train_df, valid_df

    if date_column not in df.columns:
        raise KeyError(f"Date column not found: {date_column}")

    sorted_df = df.copy()
    sorted_df[date_column] = pd.to_datetime(sorted_df[date_column])
    sorted_df = sorted_df.sort_values(date_column).reset_index(drop=True)

    split_idx = int(len(sorted_df) * (1 - valid_size_ratio))
    train_df = sorted_df.iloc[:split_idx].copy()
    valid_df = sorted_df.iloc[split_idx:].copy()
    return train_df, valid_df

def get_year_based_cv_splits(df: pd.DataFrame, date_column: str, n_splits: int = 5, gap_days: int = 14) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    [고도화된 시계열 교차 검증 분할기]
    연도(Year)를 기준으로 과거 데이터를 훈련(Train)하고, 미래의 온전한 1년 전체를 검증(Valid)하도록 인덱스를 나눕니다.
    데이터 누수 방지를 위해 Train과 Valid 사이에 gap_days 만큼의 간격을 둡니다.
    """
    # 1. 날짜 데이터 추출 및 연도(Year) 확인
    df_dates = pd.to_datetime(df[date_column]).reset_index(drop=True)
    years = sorted(df_dates.dt.year.unique())
    
    # 5개의 폴드를 만들려면 최소 6년 치 데이터가 필요함. 부족할 경우 경고.
    if len(years) <= n_splits:
        print(f"⚠️ [경고] 데이터 연도 수({len(years)}년)가 n_splits({n_splits})보다 적거나 같아 완벽한 연도 분할이 어렵습니다.")
        # 데이터가 부족해도 가능한 만큼만 폴드를 생성하도록 n_splits 강제 조정
        n_splits = max(1, len(years) - 1)
        print(f"⚠️ -> n_splits를 {n_splits}로 조정하여 진행합니다.")

    splits = []
    # 가장 최근 연도부터 역순으로 n_splits 개수만큼 검증(Valid) 연도로 지정
    valid_years = years[-n_splits:]

    for valid_year in valid_years:
        # Train: 검증 연도보다 과거인 모든 데이터
        train_mask = df_dates.dt.year < valid_year
        # Valid: 검증 연도에 해당하는 1년 치 데이터 전체
        valid_mask = df_dates.dt.year == valid_year

        train_indices = np.where(train_mask)[0]
        valid_indices = np.where(valid_mask)[0]

        # 2. Gap (간격) 적용: Train 세트의 마지막에서 gap_days 만큼 제거하여 컨닝 방지
        if gap_days > 0 and len(train_indices) > gap_days:
            train_indices = train_indices[:-gap_days]

        splits.append((train_indices, valid_indices))

    print(f"✅ 연도 기반 CV 분할 완료 (총 {len(splits)}개 Fold, Gap: {gap_days}일)")
    return splits