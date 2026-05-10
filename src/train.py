import pandas as pd
from sklearn.base import clone
from typing import Any
from src.config import DATE_COLUMN, PROBABILITY_THRESHOLD, SITE_COLUMN
from src.builder import build_regression_model_candidates, build_classification_model_candidates
from src.utils import inverse_transform_cells

def train_candidate_models(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    regression_target: str,
    classification_target: str,
    random_state: int = 42,
    optuna_cls_params: dict = None, 
    optuna_reg_params: dict = None   
) -> dict[str, Any]:
    x_train = train_df[feature_columns]
    y_reg_train = train_df[regression_target]
    y_cls_train = train_df[classification_target]

    regression_candidates = build_regression_model_candidates(
        random_state=random_state, optuna_best_params_reg=optuna_reg_params
    )
    classification_candidates = build_classification_model_candidates(
        random_state=random_state, optuna_best_params=optuna_cls_params
    )

    trained_regression_models = {}
    for model_name, model in regression_candidates.items():
        fitted_model = clone(model)
        fitted_model.fit(x_train, y_reg_train)
        trained_regression_models[model_name] = fitted_model

    trained_classification_models = {}
    for model_name, model in classification_candidates.items():
        fitted_model = clone(model)
        fitted_model.fit(x_train, y_cls_train)
        trained_classification_models[model_name] = fitted_model

    return {
        "regression_models": trained_regression_models,
        "classification_models": trained_classification_models,
        "feature_columns": feature_columns,
    }


def predict_with_all_models(
    trained: dict[str, Any],
    input_df: pd.DataFrame,
    restore_cells: bool = True,
) -> pd.DataFrame:
    """모든 후보 모델의 예측 컬럼을 반환합니다."""
    feature_columns = trained["feature_columns"]
    x = input_df[feature_columns]

    output = input_df[[col for col in [DATE_COLUMN, SITE_COLUMN] if col in input_df.columns]].copy()

    for model_name, model in trained.get("regression_models", {}).items():
        pred_reg = model.predict(x)
        output[f"{model_name}_pred_regression_target"] = pred_reg

        if restore_cells:
            output[f"{model_name}_pred_cells"] = inverse_transform_cells(pred_reg)

    threshold = trained.get("metadata", {}).get("probability_threshold", PROBABILITY_THRESHOLD)
    for model_name, model in trained.get("classification_models", {}).items():
        pred_proba = model.predict_proba(x)[:, 1]
        output[f"{model_name}_alert_risk_probability"] = pred_proba
        output[f"{model_name}_alert_pred_label"] = (pred_proba >= threshold).astype(int)

    return output


def predict_with_best_models(
    trained: dict[str, Any],
    input_df: pd.DataFrame,
    restore_cells: bool = True,
) -> pd.DataFrame:
    """선택된 best model의 핵심 예측 컬럼만 반환합니다."""
    feature_columns = trained["feature_columns"]
    x = input_df[feature_columns]

    pred_reg = trained["regression_model"].predict(x)
    pred_proba = trained["classification_model"].predict_proba(x)[:, 1]
    threshold = trained.get("metadata", {}).get("probability_threshold", PROBABILITY_THRESHOLD)

    output = input_df[[col for col in [DATE_COLUMN, SITE_COLUMN] if col in input_df.columns]].copy()
    output["pred_regression_target"] = pred_reg

    if restore_cells:
        output["predicted_cells"] = inverse_transform_cells(pred_reg)

    output["alert_probability"] = pred_proba
    output["predicted_alert_label"] = (pred_proba >= threshold).astype(int)
    return output