from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from src.config import model_config as config


def inverse_transform_cells(values: np.ndarray | pd.Series) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if config.REGRESSION_TARGET_TRANSFORM == "log10_plus_1":
        return (10**arr) - 1
    if config.REGRESSION_TARGET_TRANSFORM == "log1p":
        return np.expm1(arr)
    if config.REGRESSION_TARGET_TRANSFORM == "none":
        return arr
    raise ValueError(f"Unsupported target transform: {config.REGRESSION_TARGET_TRANSFORM}")


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float | None]:
    y_true_cells = inverse_transform_cells(y_true)
    y_pred_cells = np.maximum(inverse_transform_cells(y_pred), 0)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "r2": float(r2_score(y_true, y_pred)),
        "mae_cells": float(mean_absolute_error(y_true_cells, y_pred_cells)),
        "rmse_cells": float(mean_squared_error(y_true_cells, y_pred_cells) ** 0.5),
    }


def classification_probability(model: Any, x: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(x)
        return 1 / (1 + np.exp(-scores))
    return np.asarray(model.predict(x), dtype=float)


def classification_metrics(
    y_true: pd.Series,
    risk_probability: np.ndarray,
    threshold: float = config.PROBABILITY_THRESHOLD,
) -> dict[str, Any]:
    y_pred = (risk_probability >= threshold).astype(int)
    result: dict[str, Any] = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, zero_division=0, output_dict=True),
    }
    if pd.Series(y_true).nunique() == 2:
        result["roc_auc"] = float(roc_auc_score(y_true, risk_probability))
        result["pr_auc"] = float(average_precision_score(y_true, risk_probability))
    else:
        result["roc_auc"] = None
        result["pr_auc"] = None
    return result


def evaluate_threshold_candidates(y_true: pd.Series, risk_probability: np.ndarray) -> pd.DataFrame:
    rows = []
    for threshold in config.THRESHOLD_CANDIDATES:
        y_pred = (risk_probability >= threshold).astype(int)
        rows.append(
            {
                "threshold": float(threshold),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                "predicted_positive": int(y_pred.sum()),
            }
        )
    return pd.DataFrame(rows)


def evaluate_candidate_models(
    trained: dict[str, Any],
    valid_df: pd.DataFrame,
    workflow_key: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_columns = trained["feature_columns"]
    x_valid = valid_df[feature_columns]
    y_reg = valid_df[config.REGRESSION_TARGET]
    y_cls = valid_df[config.CLASSIFICATION_TARGET]

    rows = []
    threshold_frames = []

    for name, model in trained["regression_models"].items():
        pred = model.predict(x_valid)
        metrics = regression_metrics(y_reg, pred)
        rows.append({"workflow": workflow_key, "task": "regression", "model_name": name, **metrics})

    for name, model in trained["classification_models"].items():
        prob = classification_probability(model, x_valid)
        metrics = classification_metrics(y_cls, prob)
        rows.append({"workflow": workflow_key, "task": "classification", "model_name": name, **metrics})
        frame = evaluate_threshold_candidates(y_cls, prob)
        frame.insert(0, "model_name", name)
        frame.insert(0, "workflow", workflow_key)
        threshold_frames.append(frame)

    metrics_df = pd.DataFrame(rows)
    threshold_df = pd.concat(threshold_frames, ignore_index=True) if threshold_frames else pd.DataFrame()
    return metrics_df, threshold_df


def select_best_models(metrics_df: pd.DataFrame) -> dict[str, str]:
    reg_df = metrics_df[metrics_df["task"].eq("regression")].copy()
    cls_df = metrics_df[metrics_df["task"].eq("classification")].copy()
    best_reg = reg_df.sort_values(config.MODEL_SELECTION_METRIC_REGRESSION, ascending=True).iloc[0]["model_name"]
    best_cls = cls_df.sort_values(config.MODEL_SELECTION_METRIC_CLASSIFICATION, ascending=False).iloc[0]["model_name"]
    return {"regression": str(best_reg), "classification": str(best_cls)}
