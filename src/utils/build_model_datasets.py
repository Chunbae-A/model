from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "src" / "data"

RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed" / "model_input"
TREE_DIR = PROCESSED_DIR / "tree_gradient_boosting"
NON_TREE_DIR = PROCESSED_DIR / "non_tree_scaled"

SOURCE_FILES = {
    "algae_merged": DATA_DIR / "ALGAE_DATA.csv",
    "weather_scaled": DATA_DIR / "ALGAE_MODEL_DATA_SCALED.csv",
    "daechung_base": DATA_DIR / "daechung_for_merge_v1.csv",
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
    for path in [RAW_DIR, TREE_DIR, NON_TREE_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def copy_raw_files() -> None:
    for source in SOURCE_FILES.values():
        if source.exists():
            destination = RAW_DIR / source.name
            if not destination.exists():
                shutil.copy2(source, destination)


def add_model_columns(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    output[DATE_COLUMN] = pd.to_datetime(output[DATE_COLUMN])
    threshold_log10 = np.log10(ALERT_CELL_THRESHOLD + 1)
    output[TARGET_ALERT_NEXT] = (output[TARGET_REGRESSION] >= threshold_log10).astype(int)
    output["split"] = make_time_split(output[DATE_COLUMN])
    output = output.sort_values([DATE_COLUMN, "loc_encoded", "station"]).reset_index(drop=True)
    output[DATE_COLUMN] = output[DATE_COLUMN].dt.strftime("%Y-%m-%d")
    return output


def make_time_split(date_series: pd.Series, valid_ratio: float = 0.2) -> pd.Series:
    unique_dates = pd.Series(pd.to_datetime(date_series).sort_values().unique())
    split_idx = int(len(unique_dates) * (1 - valid_ratio))
    valid_start = unique_dates.iloc[split_idx]
    return np.where(pd.to_datetime(date_series) >= valid_start, "valid", "train")


def write_tree_dataset(source_df: pd.DataFrame) -> pd.DataFrame:
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
    readme = f"""# 모델 입력 데이터 구조

이 폴더는 원본 데이터를 보존하면서 모델 종류별 최종 입력 CSV를 분리한다.

## 폴더

- `raw/`: 원본 CSV 복사본. 재현성과 비교를 위한 보존 영역.
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
