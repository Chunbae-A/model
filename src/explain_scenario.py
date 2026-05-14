from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    BLOOM_CELL_THRESHOLD,
    DATE_COLUMN,
    HIGH_RISK_THRESHOLD,
    MEDIUM_RISK_THRESHOLD,
    SCENARIO_ACTION_CATEGORY,
    SCENARIO_FEATURE_GROUPS,
    SITE_COLUMN,
    WARNING_CELL_THRESHOLD,
    WATCH_CELL_THRESHOLD,
)


SCENARIO_CONTEXT_COLUMNS = [
    "location_name",
    "loc_flow_order",
    "water_temp",
    "수온",
    "pH",
    "Chl_a",
    "avg_temp",
    "max_temp",
    "air_temp_7d_mean",
    "hot_days_30c_7d",
    "sunshine_7d_sum",
    "solar_7d_sum",
    "rain_3d_sum",
    "rain_7d_sum_x",
    "rain_7d_sum_y",
    "rain_14d_sum",
    "inflow_7d_sum",
    "outflow_7d_sum",
    "outflow",
    "방류량",
    "residence_proxy",
    "nutrient_stagnation_index",
    "wind_7d_mean",
    "low_wind_days_2ms_7d",
    "cyano_cells",
    "유해남조류_세포수",
    "microcystis",
    "Microcystis",
    "anabaena",
    "Anabaena",
    "oscillatoria",
    "Oscillatoria",
    "aphanizomenon",
    "Aphanizomenon",
    "previous_observed_cells",
    "previous_exceeded",
    "current_exceeded",
    "cell_change_since_previous",
    "cell_growth_ratio_since_previous",
    "hoenam_cells_same_date",
    "hoenam_pressure_for_downstream",
    "upstream_cells_same_date",
    "operational_alert_target",
]


def get_feature_importance_table(model: Any, feature_columns: list[str]) -> pd.DataFrame:
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
        return (
            pd.DataFrame({"feature": feature_columns, "importance": importance})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
    return pd.DataFrame(columns=["feature", "importance"])


def _extract_2d_shap_values(shap_values: Any) -> np.ndarray:
    values = np.asarray(shap_values.values)
    if values.ndim == 3:
        values = values[:, :, -1]
    return values


def _make_global_top_reasons_table(
    trained: dict[str, Any],
    input_df: pd.DataFrame,
    best_prediction_df: pd.DataFrame,
    top_n: int,
) -> pd.DataFrame:
    feature_columns = trained["feature_columns"]
    x = input_df[feature_columns]
    model = trained["classification_model"]

    try:
        probabilities = model.predict_proba(x)[:, 1]
    except Exception:
        probabilities = np.zeros(len(x))

    scores = []
    for feature in feature_columns:
        feature_values = pd.to_numeric(x[feature], errors="coerce").fillna(0).to_numpy()
        if np.nanstd(feature_values) == 0 or np.nanstd(probabilities) == 0:
            score = 0.0
        else:
            score = float(abs(np.corrcoef(feature_values, probabilities)[0, 1]))
            if np.isnan(score):
                score = 0.0
        scores.append(score)

    top_indices = np.argsort(scores)[::-1][:top_n]
    rows = []
    for row_idx in range(len(x)):
        row: dict[str, Any] = {}
        for col in [DATE_COLUMN, SITE_COLUMN]:
            if col in input_df.columns:
                row[col] = input_df.iloc[row_idx][col]

        row["predicted_cells"] = best_prediction_df.iloc[row_idx].get("predicted_cells")
        row["alert_probability"] = best_prediction_df.iloc[row_idx].get("alert_probability")
        row["predicted_alert_label"] = best_prediction_df.iloc[row_idx].get("predicted_alert_label")
        row["predicted_alert_stage"] = best_prediction_df.iloc[row_idx].get("predicted_alert_stage")

        for col in SCENARIO_CONTEXT_COLUMNS:
            if col in input_df.columns:
                row[col] = input_df.iloc[row_idx].get(col)
            elif col in best_prediction_df.columns:
                row[col] = best_prediction_df.iloc[row_idx].get(col)

        for rank, feature_idx in enumerate(top_indices, start=1):
            row[f"shap_top_{rank}"] = feature_columns[feature_idx]
            row[f"shap_top_{rank}_value"] = scores[feature_idx]

        rows.append(row)
    return pd.DataFrame(rows)


def make_shap_top_reasons_table(
    trained: dict[str, Any],
    input_df: pd.DataFrame,
    best_prediction_df: pd.DataFrame,
    top_n: int = 3,
) -> pd.DataFrame:
    try:
        import shap
    except Exception:
        return pd.DataFrame()

    feature_columns = trained["feature_columns"]
    x = input_df[feature_columns]
    model = trained["classification_model"]
    try:
        explainer = shap.Explainer(model, x)
        try:
            shap_values = explainer(x, check_additivity=False)
        except TypeError:
            shap_values = explainer(x)
    except Exception:
        return _make_global_top_reasons_table(trained, input_df, best_prediction_df, top_n)
    values_2d = _extract_2d_shap_values(shap_values)

    rows = []
    for row_idx in range(len(x)):
        row_values = values_2d[row_idx]
        top_indices = np.argsort(np.abs(row_values))[::-1][:top_n]
        row: dict[str, Any] = {}

        for col in [DATE_COLUMN, SITE_COLUMN]:
            if col in input_df.columns:
                row[col] = input_df.iloc[row_idx][col]

        row["predicted_cells"] = best_prediction_df.iloc[row_idx].get("predicted_cells")
        row["alert_probability"] = best_prediction_df.iloc[row_idx].get("alert_probability")
        row["predicted_alert_label"] = best_prediction_df.iloc[row_idx].get("predicted_alert_label")
        row["predicted_alert_stage"] = best_prediction_df.iloc[row_idx].get("predicted_alert_stage")

        for col in SCENARIO_CONTEXT_COLUMNS:
            if col in input_df.columns:
                row[col] = input_df.iloc[row_idx].get(col)
            elif col in best_prediction_df.columns:
                row[col] = best_prediction_df.iloc[row_idx].get(col)

        for rank, feature_idx in enumerate(top_indices, start=1):
            row[f"shap_top_{rank}"] = feature_columns[feature_idx]
            row[f"shap_top_{rank}_value"] = row_values[feature_idx]

        rows.append(row)
    return pd.DataFrame(rows)


def _feature_matches_group(feature_name: Any, group_features: list[str]) -> bool:
    if feature_name is None or pd.isna(feature_name):
        return False
    feature_name_lower = str(feature_name).lower()
    return any(group_feature.lower() in feature_name_lower for group_feature in group_features)


def _count_top_feature_groups(top_features: list[Any]) -> dict[str, int]:
    counts = {group_name: 0 for group_name in SCENARIO_FEATURE_GROUPS}
    for feature_name in top_features:
        for group_name, group_features in SCENARIO_FEATURE_GROUPS.items():
            if _feature_matches_group(feature_name, group_features):
                counts[group_name] += 1
    return counts


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _flag(row: pd.Series, name: str) -> bool:
    return bool(int(_as_float(row.get(name), 0.0)))


def _high(series: pd.Series, floor: float = 0.0) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").fillna(0)
    threshold = max(float(numeric.quantile(0.75)), floor)
    return numeric.ge(threshold)


def _low(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").fillna(0)
    return numeric.le(float(numeric.quantile(0.25)))


def _max_across(output: pd.DataFrame, columns: list[str]) -> pd.Series:
    existing = [col for col in columns if col in output.columns]
    if not existing:
        return pd.Series(0.0, index=output.index)
    return output[existing].apply(pd.to_numeric, errors="coerce").fillna(0).max(axis=1)


def alert_stage_from_cells(cells: Any) -> str:
    cells_value = _as_float(cells, 0.0)
    if cells_value >= BLOOM_CELL_THRESHOLD:
        return "조류대발생"
    if cells_value >= WARNING_CELL_THRESHOLD:
        return "경계"
    if cells_value >= WATCH_CELL_THRESHOLD:
        return "관심"
    return "미발령"


def add_scenario_flags(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()

    rain = _max_across(output, ["rain_3d_sum", "rain_7d_sum", "rain_7d_sum_x", "rain_7d_sum_y", "rain_14d_sum"])
    temp = _max_across(output, ["water_temp", "수온", "avg_temp", "air_temp_7d_mean", "max_temp"])
    light = _max_across(output, ["sunshine_7d_sum", "solar_7d_sum"])
    hoenam = pd.to_numeric(output.get("hoenam_cells_same_date", 0), errors="coerce").fillna(0)
    month = (
        pd.to_datetime(output[DATE_COLUMN], errors="coerce").dt.month
        if DATE_COLUMN in output.columns
        else pd.Series(0, index=output.index)
    )

    residence_high = _high(output["residence_proxy"], 0) if "residence_proxy" in output.columns else False
    outflow_low = _low(output["outflow_7d_sum"]) if "outflow_7d_sum" in output.columns else False
    wind_low = _low(output["wind_7d_mean"]) if "wind_7d_mean" in output.columns else False
    low_wind_days = (
        pd.to_numeric(output["low_wind_days_2ms_7d"], errors="coerce").fillna(0).ge(2)
        if "low_wind_days_2ms_7d" in output.columns
        else False
    )

    growth_ratio = pd.to_numeric(output.get("cell_growth_ratio_since_previous", 0), errors="coerce").fillna(0)
    growth_delta = pd.to_numeric(output.get("cell_change_since_previous", 0), errors="coerce").fillna(0)
    current_cells = pd.to_numeric(
        output["cyano_cells"] if "cyano_cells" in output.columns else output.get("유해남조류_세포수", 0),
        errors="coerce",
    ).fillna(0)
    predicted_cells = pd.to_numeric(output.get("predicted_cells", 0), errors="coerce").fillna(0)

    output["current_alert_stage"] = current_cells.map(alert_stage_from_cells)
    output["predicted_alert_stage"] = predicted_cells.map(alert_stage_from_cells)
    output["scenario_flag_rainfall"] = (_high(rain, 10) | rain.ge(30)).astype(int)
    output["scenario_flag_stagnation"] = (residence_high | outflow_low | wind_low | low_wind_days).astype(int)
    output["scenario_flag_season_20c"] = (temp.ge(20) & month.isin([6, 7, 8, 9, 10, 11])).astype(int)
    output["scenario_flag_heat_light"] = (
        temp.ge(25)
        | (_high(temp, 25) & month.isin([6, 7, 8, 9, 10]))
        | (_high(light, 0) & month.isin([6, 7, 8, 9, 10]))
    ).astype(int)
    output["scenario_flag_growth"] = (growth_ratio.ge(2.0) | growth_delta.ge(500)).astype(int)
    output["scenario_flag_hoenam_pressure"] = (
        (pd.to_numeric(output.get(SITE_COLUMN, 0), errors="coerce").fillna(0) != 1)
        & hoenam.ge(WATCH_CELL_THRESHOLD)
    ).astype(int)
    return output


def classify_scenario(row: pd.Series) -> tuple[str, str, str]:
    top_features = [row.get("shap_top_1"), row.get("shap_top_2"), row.get("shap_top_3")]
    group_counts = _count_top_feature_groups(top_features)
    active_groups = [group_name for group_name, count in group_counts.items() if count > 0]

    probability = _as_float(row.get("alert_probability"), 0.0)
    predicted_cells = _as_float(row.get("predicted_cells"), 0.0)
    current_cells = _as_float(row.get("cyano_cells", row.get("유해남조류_세포수")), 0.0)
    previous_cells = _as_float(row.get("previous_observed_cells"), 0.0)

    previous_watch = previous_cells >= WATCH_CELL_THRESHOLD
    previous_warning = previous_cells >= WARNING_CELL_THRESHOLD
    predicted_watch = predicted_cells >= WATCH_CELL_THRESHOLD or _flag(row, "predicted_alert_label")
    predicted_warning = predicted_cells >= WARNING_CELL_THRESHOLD
    predicted_bloom = predicted_cells >= BLOOM_CELL_THRESHOLD
    current_watch = current_cells >= WATCH_CELL_THRESHOLD

    rainfall_signal = _flag(row, "scenario_flag_rainfall")
    stagnation_signal = _flag(row, "scenario_flag_stagnation")
    season_20c_signal = _flag(row, "scenario_flag_season_20c")
    heat_light_signal = _flag(row, "scenario_flag_heat_light")
    growth_signal = _flag(row, "scenario_flag_growth")
    hoenam_signal = _flag(row, "scenario_flag_hoenam_pressure")

    if predicted_bloom:
        scenario_type = "조류대발생 감시 시나리오"
    elif previous_warning and predicted_warning:
        scenario_type = "경계 발령 후보 시나리오"
    elif previous_watch and predicted_watch:
        scenario_type = "관심 발령 후보 시나리오"
    elif previous_watch and not current_watch and not predicted_watch:
        scenario_type = "하향·해제 관찰 시나리오"
    elif predicted_watch and not previous_watch:
        scenario_type = "관심 기준 돌파 예측 시나리오"
    elif predicted_cells >= 500 or current_cells >= 500:
        scenario_type = "관심 기준 접근 시나리오"
    elif hoenam_signal and (probability >= 0.10 or predicted_cells >= 300):
        scenario_type = "회남 선행 전파 관찰 시나리오"
    elif probability >= HIGH_RISK_THRESHOLD and rainfall_signal and stagnation_signal:
        scenario_type = "강우 유입 후 정체 고위험 시나리오"
    elif probability >= HIGH_RISK_THRESHOLD and len(active_groups) >= 2:
        scenario_type = "복합 고위험 시나리오"
    elif stagnation_signal and (probability >= 0.15 or predicted_cells >= 300):
        scenario_type = "수체 정체·성층 위험 시나리오"
    elif rainfall_signal and (probability >= 0.15 or predicted_cells >= 300):
        scenario_type = "강우 이후 영양염류 유입 시나리오"
    elif heat_light_signal and (probability >= 0.05 or predicted_cells >= 200):
        scenario_type = "고온·고일사 성장 촉진 시나리오"
    elif growth_signal and (probability >= 0.05 or predicted_cells >= 200):
        scenario_type = "과거 증식 추세 지속 시나리오"
    elif season_20c_signal:
        scenario_type = "수온 20도 이상 계절 감시 시나리오"
    elif probability >= MEDIUM_RISK_THRESHOLD:
        scenario_type = "관찰 강화 시나리오"
    else:
        scenario_type = "일반 안정 시나리오"

    signals = {
        "season_20c": season_20c_signal,
        "rainfall": rainfall_signal,
        "stagnation": stagnation_signal,
        "heat_light": heat_light_signal,
        "growth": growth_signal,
        "hoenam_pressure": hoenam_signal,
        "previous_watch": previous_watch,
        "predicted_watch": predicted_watch,
    }
    scenario_reason = (
        f"alert_probability={probability:.3f}, current_cells={current_cells:.1f}, "
        f"previous_cells={previous_cells:.1f}, predicted_cells={predicted_cells:.1f}, "
        f"current_stage={alert_stage_from_cells(current_cells)}, "
        f"predicted_stage={alert_stage_from_cells(predicted_cells)}, "
        f"signals={signals}, top_features={[str(feature) for feature in top_features if pd.notna(feature)]}, "
        f"matched_groups={group_counts}"
    )
    action_category = SCENARIO_ACTION_CATEGORY.get(scenario_type, "모니터링 강화")
    return scenario_type, scenario_reason, action_category


def build_scenario_results(shap_top_reasons_df: pd.DataFrame) -> pd.DataFrame:
    required = ["predicted_cells", "alert_probability", "predicted_alert_label"]
    missing = [col for col in required if col not in shap_top_reasons_df.columns]
    if missing:
        raise KeyError(f"시나리오 입력에 필수 컬럼이 없습니다: {missing}")

    output = add_scenario_flags(shap_top_reasons_df)
    scenario_values = output.apply(classify_scenario, axis=1, result_type="expand")
    output["scenario_type"] = scenario_values[0]
    output["scenario_reason"] = scenario_values[1]
    output["recommended_action_category"] = scenario_values[2]

    keep_cols = [
        DATE_COLUMN,
        SITE_COLUMN,
        "location_name",
        "loc_flow_order",
        "predicted_cells",
        "alert_probability",
        "predicted_alert_label",
        "current_alert_stage",
        "predicted_alert_stage",
        "cyano_cells",
        "유해남조류_세포수",
        "previous_observed_cells",
        "previous_exceeded",
        "hoenam_cells_same_date",
        "scenario_flag_season_20c",
        "scenario_flag_rainfall",
        "scenario_flag_stagnation",
        "scenario_flag_heat_light",
        "scenario_flag_growth",
        "scenario_flag_hoenam_pressure",
        "shap_top_1",
        "shap_top_2",
        "shap_top_3",
        "scenario_type",
        "scenario_reason",
        "recommended_action_category",
    ]
    existing_cols = [col for col in keep_cols if col in output.columns]
    return output[existing_cols]


def save_scenario_outputs(scenario_results_df: pd.DataFrame, scenario_dir: Path) -> list[dict[str, Any]]:
    scenario_dir.mkdir(parents=True, exist_ok=True)
    scenario_results_df.to_csv(scenario_dir / "scenario_results.csv", index=False)
    records = []
    for _, row in scenario_results_df.iterrows():
        top_reasons = []
        for rank in range(1, 4):
            feature = row.get(f"shap_top_{rank}")
            if pd.notna(feature):
                top_reasons.append({"rank": rank, "feature": feature})
        records.append({
            "date": str(row.get(DATE_COLUMN)) if DATE_COLUMN in row else None,
            "site": row.get(SITE_COLUMN) if SITE_COLUMN in row else None,
            "forecast_horizon": None,
            "predicted_cells": row.get("predicted_cells"),
            "alert_probability": row.get("alert_probability"),
            "predicted_alert_label": row.get("predicted_alert_label"),
            "current_alert_stage": row.get("current_alert_stage"),
            "predicted_alert_stage": row.get("predicted_alert_stage"),
            "top_reasons": top_reasons,
            "scenario_type": row.get("scenario_type"),
            "scenario_reason": row.get("scenario_reason"),
            "recommended_action_category": row.get("recommended_action_category"),
        })
    return records
