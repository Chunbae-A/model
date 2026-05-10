from pathlib import Path
import pandas as pd
import numpy as np
from typing import Any
from src.config import (
    DATE_COLUMN, 
    SITE_COLUMN,
    FEATURE_IMPORTANCE_FILE,
    SHAP_TOP_REASONS_FILE
)

def get_feature_importance_table(model: Any, feature_columns: list[str]) -> pd.DataFrame:
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
        return (
            pd.DataFrame({"feature": feature_columns, "importance": importance})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    return pd.DataFrame(columns=["feature", "importance"])


def get_feature_importance_for_all_models(
    trained: dict[str, Any],
) -> pd.DataFrame:
    """변수 중요도 속성이 있는 모든 후보 모델의 중요도 표를 반환합니다."""
    rows = []
    feature_columns = trained["feature_columns"]

    for task, model_group_key in [
        ("regression", "regression_models"),
        ("classification", "classification_models"),
    ]:
        for model_name, model in trained.get(model_group_key, {}).items():
            importance_df = get_feature_importance_table(model, feature_columns)
            if importance_df.empty:
                rows.append({
                    "task": task,
                    "model_name": model_name,
                    "feature": None,
                    "importance": None,
                    "note": "이 모델은 feature_importances_ 속성을 제공하지 않습니다",
                })
                continue

            for _, row in importance_df.iterrows():
                rows.append({
                    "task": task,
                    "model_name": model_name,
                    "feature": row["feature"],
                    "importance": row["importance"],
                    "note": None,
                })

    return pd.DataFrame(rows)


def _extract_2d_shap_values(shap_values: Any) -> np.ndarray:
    values = np.asarray(shap_values.values)
    # 일부 이진 분류 모델은 (샘플 수, 입력 변수 수, 클래스 수) 형태의 SHAP 값을 반환할 수 있습니다.
    # 이 경우 관례적으로 양성 위험 클래스의 SHAP 값을 사용합니다.
    if values.ndim == 3:
        values = values[:, :, -1]
    return values


def _prediction_value_for_row(
    prediction_df: pd.DataFrame | None,
    row_idx: int,
    model_name: str,
    task: str,
) -> dict[str, Any]:
    if prediction_df is None:
        return {}

    result: dict[str, Any] = {}
    if task == "classification":
        proba_col = f"{model_name}_alert_risk_probability"
        label_col = f"{model_name}_alert_pred_label"
        if proba_col in prediction_df.columns:
            result["alert_probability"] = prediction_df.iloc[row_idx][proba_col]
        if label_col in prediction_df.columns:
            result["predicted_alert_label"] = prediction_df.iloc[row_idx][label_col]
    elif task == "regression":
        pred_col = f"{model_name}_pred_cells"
        if pred_col in prediction_df.columns:
            result["predicted_cells"] = prediction_df.iloc[row_idx][pred_col]
    return result


def make_shap_summary_table_for_all_models(
    trained: dict[str, Any],
    input_df: pd.DataFrame,
    prediction_df: pd.DataFrame | None = None,
    task: str = "classification",
    top_n: int = 3,
) -> pd.DataFrame:
    """모든 후보 모델에 대해 행별 상위 SHAP feature를 반환합니다.

    이 함수는 SHAP 설명 결과의 구조를 정의합니다.
    실제 SHAP 값은 실제 데이터로 코드를 실행했을 때 계산됩니다.
    """
    try:
        import shap
    except ImportError as exc:
        raise ImportError("shap이 설치되어 있지 않습니다. shap을 설치하거나 feature importance 기반 대체 결과를 사용하세요.") from exc

    if task == "classification":
        model_group = trained.get("classification_models", {})
    elif task == "regression":
        model_group = trained.get("regression_models", {})
    else:
        raise ValueError("task는 'classification' 또는 'regression'이어야 합니다.")

    feature_columns = trained["feature_columns"]
    x = input_df[feature_columns]
    all_rows = []

    for model_name, model in model_group.items():
        explainer = shap.Explainer(model, x)
        shap_values = explainer(x, check_additivity=False)
        values_2d = _extract_2d_shap_values(shap_values)

        for row_idx in range(len(x)):
            row_values = values_2d[row_idx]
            top_indices = np.argsort(np.abs(row_values))[::-1][:top_n]

            row = {"task": task, "model_name": model_name}
            for col in [DATE_COLUMN, SITE_COLUMN]:
                if col in input_df.columns:
                    row[col] = input_df.iloc[row_idx][col]

            row.update(_prediction_value_for_row(prediction_df, row_idx, model_name, task))

            for rank, feature_idx in enumerate(top_indices, start=1):
                row[f"top_{rank}_feature"] = feature_columns[feature_idx]
                row[f"top_{rank}_shap_value"] = row_values[feature_idx]
            all_rows.append(row)

    return pd.DataFrame(all_rows)


def make_shap_top_reasons_table(
    trained: dict[str, Any],
    input_df: pd.DataFrame,
    best_prediction_df: pd.DataFrame,
    top_n: int = 3,
) -> pd.DataFrame:
    """선택된 best 분류 모델 기준으로 필수 SHAP 상위 원인표를 만듭니다."""
    try:
        import shap
    except ImportError:
        return make_feature_importance_fallback_reasons(trained, input_df, best_prediction_df, top_n=top_n)

    feature_columns = trained["feature_columns"]
    x = input_df[feature_columns]
    model = trained["classification_model"]

    explainer = shap.Explainer(model, x)
    shap_values = explainer(x, check_additivity=False)
    values_2d = _extract_2d_shap_values(shap_values)

    rows = []
    for row_idx in range(len(x)):
        row_values = values_2d[row_idx]
        top_indices = np.argsort(np.abs(row_values))[::-1][:top_n]

        row = {}
        for col in [DATE_COLUMN, SITE_COLUMN]:
            if col in input_df.columns:
                row[col] = input_df.iloc[row_idx][col]

        row["predicted_cells"] = best_prediction_df.iloc[row_idx].get("predicted_cells")
        row["alert_probability"] = best_prediction_df.iloc[row_idx].get("alert_probability")
        row["predicted_alert_label"] = best_prediction_df.iloc[row_idx].get("predicted_alert_label")

        for rank, feature_idx in enumerate(top_indices, start=1):
            row[f"shap_top_{rank}"] = feature_columns[feature_idx]
            row[f"shap_top_{rank}_value"] = row_values[feature_idx]
        rows.append(row)

    return pd.DataFrame(rows)


def make_feature_importance_fallback_reasons(
    trained: dict[str, Any],
    input_df: pd.DataFrame,
    best_prediction_df: pd.DataFrame,
    top_n: int = 3,
) -> pd.DataFrame:
    """SHAP을 사용할 수 없을 때 변수 중요도 기반 대체 원인표를 만듭니다.

    각 예측 행에 동일한 전역 변수 중요도를 붙입니다.
    행별 SHAP보다 설명력은 약하지만, 다음 에이전트가 기대하는 출력 구조는 유지합니다.
    """
    importance_df = get_feature_importance_table(trained["classification_model"], trained["feature_columns"])
    top_features = importance_df.head(top_n).to_dict(orient="records")

    rows = []
    for row_idx in range(len(input_df)):
        row = {}
        for col in [DATE_COLUMN, SITE_COLUMN]:
            if col in input_df.columns:
                row[col] = input_df.iloc[row_idx][col]

        row["predicted_cells"] = best_prediction_df.iloc[row_idx].get("predicted_cells")
        row["alert_probability"] = best_prediction_df.iloc[row_idx].get("alert_probability")
        row["predicted_alert_label"] = best_prediction_df.iloc[row_idx].get("predicted_alert_label")

        for rank in range(1, top_n + 1):
            if rank <= len(top_features):
                row[f"shap_top_{rank}"] = top_features[rank - 1].get("feature")
                row[f"shap_top_{rank}_value"] = top_features[rank - 1].get("importance")
            else:
                row[f"shap_top_{rank}"] = None
                row[f"shap_top_{rank}_value"] = None
        rows.append(row)

    return pd.DataFrame(rows)


def save_explain_outputs(
    importance_df: pd.DataFrame,
    shap_top_reasons_df: pd.DataFrame,
    explain_dir: Path,
) -> None:
    explain_dir.mkdir(parents=True, exist_ok=True)
    importance_df.to_csv(explain_dir / FEATURE_IMPORTANCE_FILE, index=False, encoding='utf-8-sig')
    shap_top_reasons_df.to_csv(explain_dir / SHAP_TOP_REASONS_FILE, index=False, encoding='utf-8-sig')


