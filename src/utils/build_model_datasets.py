from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "src" / "data"

RAW_DIR = DATA_DIR / "team-raw"
PROCESSED_DIR = DATA_DIR / "processed" / "model_input"
TREE_DIR = PROCESSED_DIR / "tree_gradient_boosting"
NON_TREE_DIR = PROCESSED_DIR / "non_tree_scaled"

SOURCE_FILES = {
    "algae_merged": RAW_DIR / "ALGAE_DATA.csv",
    "weather_scaled": RAW_DIR / "ALGAE_MODEL_DATA_SCALED.csv",
    "daechung_base": RAW_DIR / "daechung_for_merge_v1.csv",
}

TREE_OUTPUT = TREE_DIR / "algae_tree_station_expanded.csv"
NON_TREE_OUTPUT = NON_TREE_DIR / "algae_non_tree_scaled_station_expanded.csv"

DATE_COLUMN = "date"
TARGET_REGRESSION = "next_log_cells"
TARGET_CURRENT_LOG = "log_target"
TARGET_ALERT_NEXT = "target_alert_next"
ALERT_CELL_THRESHOLD = 1000

ID_COLUMNS = [DATE_COLUMN, "station", "loc_encoded"]
TARGET_COLUMNS = [TARGET_REGRESSION, TARGET_CURRENT_LOG, "alert_encoded", TARGET_ALERT_NEXT]

WEATHER_ROBUST_COLS = [
    "daily_rain",
    "rain_3d_sum",
    "rain_7d_sum_y",
    "rain_14d_sum",
]

WEATHER_MINMAX_COLS = [
    "avg_temp",
    "min_temp",
    "max_temp",
    "avg_wind",
    "air_temp_7d_mean",
    "wind_7d_mean",
    "sunshine",
    "solar_rad",
    "cloud_cover",
    "sunshine_7d_sum",
    "solar_7d_sum",
    "cloud_7d_mean",
    "max_wind_gust",
    "sin_season",
    "cos_season",
]

WATER_QUALITY_MINMAX_COLS = [
    "water_temp",
    "pH",
    "DO",
    "transparency",
    "acc_temp_7d",
    "TSI_Chla",
    "TSI_SD",
    "water_level",
    "storage_vol",
    "storage_rate",
]

WATER_HYDRO_ROBUST_COLS = [
    "level_change_7d",
    "inflow",
    "outflow",
    "rainfall",
    "rain_7d_sum_x",
    "inflow_7d_sum",
    "outflow_7d_sum",
    "residence_proxy",
    "nutrient_stagnation_index",
]

LOG_THEN_ROBUST_COLS = [
    "turbidity",
    "Chl_a",
    "cyano_cells",
    "Microcystis",
    "Anabaena",
    "Oscillatoria",
    "Aphanizomenon",
]

PASS_THROUGH_FEATURE_COLS = [
    "microcystis_ratio",
    "anabaena_ratio",
    "oscillatoria_ratio",
    "aphanizomenon_ratio",
    "hot_days_30c_7d",
    "low_wind_days_2ms_7d",
]


def ensure_dirs() -> None:
    """원본 보관 폴더와 모델 입력 폴더를 만든다."""

    for path in [RAW_DIR, TREE_DIR, NON_TREE_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def copy_raw_files() -> None:
    """원본 CSV를 team-raw 영역에 보존한다.

    이미 같은 파일이 있으면 덮어쓰지 않는다. 전처리 재현성을 위해 원본을
    보존하는 목적이며, 모델 학습은 processed/model_input 아래 파일을 사용한다.
    """

    for source in SOURCE_FILES.values():
        if source.exists():
            destination = RAW_DIR / source.name
            if not destination.exists():
                shutil.copy2(source, destination)


def add_model_columns(df: pd.DataFrame) -> pd.DataFrame:
    """공통 모델 컬럼을 추가한다.

    - `target_alert_next`: 다음 조사 시점 세포수가 1,000 cells/mL 이상인지 여부
    - `split`: 날짜 기준 train/valid 구분

    같은 날짜가 여러 loc/station 행으로 반복되므로 정렬 기준도 date, loc, station으로
    고정해 결과 CSV가 매번 같은 순서로 생성되게 한다.
    """

    output = df.copy()
    output[DATE_COLUMN] = pd.to_datetime(output[DATE_COLUMN])
    threshold_log10 = np.log10(ALERT_CELL_THRESHOLD + 1)
    output[TARGET_ALERT_NEXT] = (output[TARGET_REGRESSION] >= threshold_log10).astype(int)
    output["split"] = make_time_split(output[DATE_COLUMN])
    output = output.sort_values([DATE_COLUMN, "loc_encoded", "station"]).reset_index(drop=True)
    output[DATE_COLUMN] = output[DATE_COLUMN].dt.strftime("%Y-%m-%d")
    return output


def make_time_split(date_series: pd.Series, valid_ratio: float = 0.2) -> pd.Series:
    """날짜 순서를 기준으로 마지막 20% 날짜를 valid로 둔다."""

    unique_dates = pd.Series(pd.to_datetime(date_series).sort_values().unique())
    split_idx = int(len(unique_dates) * (1 - valid_ratio))
    valid_start = unique_dates.iloc[split_idx]
    return np.where(pd.to_datetime(date_series) >= valid_start, "valid", "train")


def write_tree_dataset(source_df: pd.DataFrame) -> pd.DataFrame:
    """트리 기반 모델용 입력 파일을 만든다.

    트리/부스팅 계열은 feature의 단위 차이에 덜 민감하므로 수질·조류·수문
    feature의 원 단위를 유지한다. 모델 target과 split만 추가한다.
    """

    tree_df = add_model_columns(source_df)
    tree_df.to_csv(TREE_OUTPUT, index=False)
    return tree_df


def fit_transform_column_group(
    df: pd.DataFrame,
    columns: list[str],
    scaler,
    train_mask: pd.Series,
    suffix: str = "",
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    """지정된 컬럼 묶음에 scaler를 fit/transform하고 요약 통계를 반환한다.

    scaler는 반드시 train 구간에만 fit한다. valid 데이터까지 함께 fit하면
    미래 분포 정보를 전처리 단계에서 미리 본 셈이 되어 validation leakage가 된다.
    """

    output = pd.DataFrame(index=df.index)
    fitted = scaler.fit(df.loc[train_mask, columns])
    values = fitted.transform(df[columns])
    output_columns = [f"{col}{suffix}" for col in columns]
    output[output_columns] = values
    stats = {}
    for idx, col in enumerate(output_columns):
        stats[col] = {
            "train_min": float(output.loc[train_mask, col].min()),
            "train_median": float(output.loc[train_mask, col].median()),
            "train_max": float(output.loc[train_mask, col].max()),
        }
    return output, stats


def write_non_tree_dataset(source_df: pd.DataFrame) -> pd.DataFrame:
    """비트리/딥러닝 모델용 스케일링 입력 파일을 만든다.

    선형 모델, SVM, KNN, MLP는 feature scale에 민감하다. 그래서 station/location은
    one-hot encoding하고, 강수/유입/방류처럼 극단값이 많은 변수는 RobustScaler,
    물리적 범위가 비교적 제한된 변수는 MinMaxScaler를 적용한다.
    """

    base = add_model_columns(source_df)
    train_mask = base["split"].eq("train")

    parts = [
        base[[DATE_COLUMN, "split", "station", "loc_encoded"]].copy(),
        pd.get_dummies(base["station"], prefix="station", dtype=int),
        pd.get_dummies(base["loc_encoded"], prefix="loc", dtype=int),
    ]
    scaling_summary: dict[str, dict[str, float]] = {}

    for columns, scaler, suffix in [
        (WEATHER_ROBUST_COLS, RobustScaler(), "_robust"),
        (WEATHER_MINMAX_COLS, MinMaxScaler(), "_minmax"),
        (WATER_QUALITY_MINMAX_COLS, MinMaxScaler(), "_minmax"),
        (WATER_HYDRO_ROBUST_COLS, RobustScaler(), "_robust"),
    ]:
        existing = [col for col in columns if col in base.columns]
        transformed, stats = fit_transform_column_group(base, existing, scaler, train_mask, suffix)
        parts.append(transformed)
        scaling_summary.update(stats)

    log_existing = [col for col in LOG_THEN_ROBUST_COLS if col in base.columns]
    # 세포수와 Chl-a는 0이 많고 큰 폭증 구간이 있어 먼저 log10(x + 1)로 압축한다.
    logged = np.log10(base[log_existing] + 1)
    logged.columns = [f"{col}_log10p1" for col in log_existing]
    transformed, stats = fit_transform_column_group(
        logged,
        list(logged.columns),
        RobustScaler(),
        train_mask,
        "_robust",
    )
    parts.append(transformed)
    scaling_summary.update(stats)

    pass_through_existing = [col for col in PASS_THROUGH_FEATURE_COLS if col in base.columns]
    parts.append(base[pass_through_existing].copy())
    parts.append(base[TARGET_COLUMNS].copy())

    non_tree_df = pd.concat(parts, axis=1)
    non_tree_df.to_csv(NON_TREE_OUTPUT, index=False)

    metadata = {
        "source": str(SOURCE_FILES["algae_merged"].relative_to(ROOT)),
        "output": str(NON_TREE_OUTPUT.relative_to(ROOT)),
        "fit_policy": "Scalers are fit on rows where split == 'train' only, then applied to valid rows.",
        "target_regression": TARGET_REGRESSION,
        "target_alert_next": TARGET_ALERT_NEXT,
        "alert_cell_threshold": ALERT_CELL_THRESHOLD,
        "feature_policy": {
            "tree_gradient_boosting": "Keep original units except already-scaled weather columns from the merged source.",
            "non_tree_scaled": "One-hot encode station/location; log-transform skewed algae columns; scale feature groups.",
        },
        "scaling_summary": scaling_summary,
    }
    (NON_TREE_DIR / "preprocessing_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2)
    )
    return non_tree_df


def write_readme(tree_df: pd.DataFrame, non_tree_df: pd.DataFrame) -> None:
    """생성된 모델 입력 데이터의 README를 갱신한다."""

    readme = f"""# 모델 입력 데이터 구조

이 폴더는 원본 데이터를 보존하면서 모델 종류별 최종 입력 CSV를 분리한다.

## 폴더

- `team-raw/`: 전처리팀이 제공한 원본 CSV 보존 영역.
- `processed/model_input/tree_gradient_boosting/`: LightGBM, XGBoost, HistGradientBoosting 같은 트리 기반 Gradient Boosting용 입력.
- `processed/model_input/non_tree_scaled/`: 선형모델, SVM, KNN, 신경망, PCA 등 스케일에 민감한 모델용 입력.

## 생성 파일

| 파일 | 행 | 열 | 용도 |
| --- | ---: | ---: | --- |
| `processed/model_input/tree_gradient_boosting/{TREE_OUTPUT.name}` | {len(tree_df):,} | {tree_df.shape[1]} | 트리 기반 모델용. 원 단위 수질/조류/댐 feature를 유지한다. |
| `processed/model_input/non_tree_scaled/{NON_TREE_OUTPUT.name}` | {len(non_tree_df):,} | {non_tree_df.shape[1]} | 비트리 모델용. 수질/조류/댐 feature까지 로그/스케일링한다. |

## 공통 주의사항

- 두 파일 모두 `ALGAE_DATA.csv`의 station-expanded 구조를 유지한다.
- 같은 `date + loc_encoded` 수질 행이 `station` 604, 643, 648, 888별로 4번 존재한다.
- 검증 성능 과대평가를 막으려면 반드시 날짜 기준 split을 사용한다.
- 회귀 target은 `next_log_cells`다.
- 분류 target은 `target_alert_next = next_log_cells >= log10(1000 + 1)`로 생성했다.

## 트리 기반 버전

트리 기반 Gradient Boosting은 feature scale에 둔감하므로 원 단위 수질/조류/댐 값을 유지한다. 기상 컬럼은 병합 원본에 들어 있던 스케일링 상태를 그대로 사용한다.

## 비트리 스케일 버전

비트리 모델용 파일은 아래 정책을 적용한다.

- `station`, `loc_encoded`: 원값 보존 + one-hot 컬럼 추가
- 조류 세포수/탁도/Chl-a: `log10(x + 1)` 후 `RobustScaler`
- 강우/유입/방류/정체 지표: `RobustScaler`
- 수온, pH, DO, 수위, 저수량 등 범위가 비교적 제한적인 컬럼: `MinMaxScaler`
- 기상 컬럼: 전처리팀의 Robust/MinMax 의도를 유지하되, 비트리 파일에서는 train split 기준으로 다시 스케일링
- scaler는 `split == train` 행에만 fit하고 valid 행에는 transform만 적용한다.

"""
    (DATA_DIR / "README.md").write_text(readme)


def main() -> None:
    ensure_dirs()
    copy_raw_files()
    source_df = pd.read_csv(SOURCE_FILES["algae_merged"])
    tree_df = write_tree_dataset(source_df)
    non_tree_df = write_non_tree_dataset(source_df)
    write_readme(tree_df, non_tree_df)


if __name__ == "__main__":
    main()
