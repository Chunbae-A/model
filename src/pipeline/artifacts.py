from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.config import model_config as config
from src.pipeline.evaluation import classification_probability, inverse_transform_cells


def workflow_artifact_dirs(workflow: config.WorkflowConfig) -> dict[str, Path]:
    root = config.ARTIFACT_DIR / workflow.artifact_subdir
    return {
        "root": root,
        "models": root / "models",
        "metrics": root / "metrics",
        "predictions": root / "predictions",
        "explain": root / "explain",
    }


def make_predictions(
    trained: dict[str, Any],
    valid_df: pd.DataFrame,
    best_models: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_columns = trained["feature_columns"]
    ids = valid_df[config.ID_COLUMNS].reset_index(drop=True)
    x_valid = valid_df[feature_columns]

    reg_model = trained["regression_models"][best_models["regression"]]
    cls_model = trained["classification_models"][best_models["classification"]]

    reg_pred = reg_model.predict(x_valid)
    cls_prob = classification_probability(cls_model, x_valid)
    cls_label = (cls_prob >= config.PROBABILITY_THRESHOLD).astype(int)

    regression_df = ids.copy()
    regression_df["actual_log_cells"] = valid_df[config.REGRESSION_TARGET].to_numpy()
    regression_df["predicted_log_cells"] = reg_pred
    regression_df["actual_cells"] = inverse_transform_cells(regression_df["actual_log_cells"])
    regression_df["predicted_cells"] = np.maximum(inverse_transform_cells(reg_pred), 0)
    regression_df["model_name"] = best_models["regression"]

    classification_df = ids.copy()
    classification_df["actual_alert"] = valid_df[config.CLASSIFICATION_TARGET].to_numpy()
    classification_df["alert_probability"] = cls_prob
    classification_df["predicted_alert_label"] = cls_label
    classification_df["model_name"] = best_models["classification"]

    return regression_df, classification_df


def get_feature_importance(
    trained: dict[str, Any],
    valid_df: pd.DataFrame,
    best_models: dict[str, str],
) -> pd.DataFrame:
    feature_columns = trained["feature_columns"]
    x_valid = valid_df[feature_columns]
    y_reg = valid_df[config.REGRESSION_TARGET]
    model = trained["regression_models"][best_models["regression"]]

    if hasattr(model, "feature_importances_"):
        importance = np.asarray(model.feature_importances_, dtype=float)
    else:
        result = permutation_importance(
            model,
            x_valid,
            y_reg,
            n_repeats=5,
            random_state=config.RANDOM_STATE,
            scoring="neg_root_mean_squared_error",
        )
        importance = result.importances_mean

    return (
        pd.DataFrame({"feature": feature_columns, "importance": importance})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def save_artifacts(
    workflow: config.WorkflowConfig,
    trained: dict[str, Any],
    metrics_df: pd.DataFrame,
    threshold_df: pd.DataFrame,
    best_models: dict[str, str],
    regression_predictions: pd.DataFrame,
    classification_predictions: pd.DataFrame,
    feature_importance: pd.DataFrame,
) -> dict[str, Path]:
    dirs = workflow_artifact_dirs(workflow)
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    bundle = {
        "trained": trained,
        "best_models": best_models,
        "workflow": workflow.key,
        "config": {
            "model_input_path": str(workflow.model_input_path),
            "regression_target": config.REGRESSION_TARGET,
            "classification_target": config.CLASSIFICATION_TARGET,
            "id_columns": config.ID_COLUMNS,
        },
    }
    joblib.dump(bundle, dirs["models"] / config.MODEL_BUNDLE_FILE)
    joblib.dump(trained["regression_models"][best_models["regression"]], dirs["models"] / config.REGRESSION_MODEL_FILE)
    joblib.dump(trained["classification_models"][best_models["classification"]], dirs["models"] / config.CLASSIFICATION_MODEL_FILE)

    regression_metrics = metrics_df[metrics_df["task"].eq("regression")].to_dict(orient="records")
    classification_metrics_rows = metrics_df[metrics_df["task"].eq("classification")].to_dict(orient="records")
    (dirs["metrics"] / config.REGRESSION_METRIC_FILE).write_text(
        json.dumps(regression_metrics, ensure_ascii=False, indent=2, default=_json_default)
    )
    (dirs["metrics"] / config.CLASSIFICATION_METRIC_FILE).write_text(
        json.dumps(classification_metrics_rows, ensure_ascii=False, indent=2, default=_json_default)
    )
    threshold_df.to_csv(dirs["metrics"] / config.THRESHOLD_CANDIDATE_FILE, index=False)
    regression_predictions.to_csv(dirs["predictions"] / config.REGRESSION_PREDICTION_FILE, index=False)
    classification_predictions.to_csv(dirs["predictions"] / config.CLASSIFICATION_PREDICTION_FILE, index=False)
    feature_importance.to_csv(dirs["explain"] / config.FEATURE_IMPORTANCE_FILE, index=False)
    return dirs
