from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, classification_report, confusion_matrix, f1_score, mean_absolute_error, mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score

from src.config import MIN_PRECISION_FOR_THRESHOLD, PROBABILITY_THRESHOLD, THRESHOLD_CANDIDATES
from src.utils import _safe_rmsle, inverse_transform_cells


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float | None]:
    """회귀 예측 성능을 평가합니다.

    MAE/RMSE/R2는 모델이 학습한 target 스케일에서 계산합니다.
    REGRESSION_TARGET_TRANSFORM을 알 수 있으면, 원 단위로 복원한 *_cells 지표도 함께 계산합니다.
    """
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    metrics: dict[str, float | None] = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse,
        "r2": float(r2_score(y_true, y_pred)),
    }

    y_true_cells = inverse_transform_cells(y_true.to_numpy())
    y_pred_cells = inverse_transform_cells(y_pred)
    metrics.update({
        "mae_cells": float(mean_absolute_error(y_true_cells, y_pred_cells)),
        "rmse_cells": float(np.sqrt(mean_squared_error(y_true_cells, y_pred_cells))),
        "rmsle_cells": _safe_rmsle(y_true_cells, y_pred_cells),
    })
    return metrics


def classification_metrics(
    y_true: pd.Series,
    risk_probability: np.ndarray,
    probability_threshold: float = 0.5,
) -> dict[str, Any]:
    y_pred = (risk_probability >= probability_threshold).astype(int)

    metrics = {
        "threshold": float(probability_threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, zero_division=0, output_dict=True),
    }

    if len(pd.Series(y_true).dropna().unique()) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, risk_probability))
        metrics["pr_auc"] = float(average_precision_score(y_true, risk_probability))
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None

    return metrics


def evaluate_threshold_candidates(
    y_true: pd.Series,
    risk_probability: np.ndarray,
    thresholds: list[float],
) -> pd.DataFrame:
    """기준값 후보별 정밀도/재현율/F1을 계산합니다.

    주의: threshold는 validation 기간에서만 탐색하세요.
    test 데이터에 threshold를 맞추면 운영 성능을 과대평가할 수 있습니다.
    """
    rows = []
    for threshold in thresholds:
        y_pred = (risk_probability >= threshold).astype(int)
        rows.append({
            "threshold": float(threshold),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        })
    return pd.DataFrame(rows)


def select_operational_threshold(
    threshold_df: pd.DataFrame,
    min_precision: float = 0.30,
) -> float:
    """최소 Precision 조건을 지키면서 Recall을 우선하는 threshold를 선택합니다."""
    if threshold_df.empty:
        return float(PROBABILITY_THRESHOLD)

    candidates = threshold_df[threshold_df["precision"].ge(min_precision)].copy()
    if candidates.empty:
        candidates = threshold_df.copy()

    selected = candidates.sort_values(
        ["recall", "f1", "precision"],
        ascending=[False, False, False],
    ).iloc[0]
    return float(selected["threshold"])


def evaluate_candidate_models(
    trained: dict[str, Any],
    valid_df: pd.DataFrame,
    regression_target: str,
    classification_target: str,
    probability_threshold: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """모든 후보 모델을 평가하고 성능표와 threshold 후보표를 반환합니다."""
    feature_columns = trained["feature_columns"]
    x_valid = valid_df[feature_columns]

    metric_rows = []
    threshold_rows = []

    for model_name, model in trained["regression_models"].items():
        pred = model.predict(x_valid)
        metric = regression_metrics(valid_df[regression_target], pred)
        metric_rows.append({
            "task": "regression",
            "model_name": model_name,
            **metric,
        })

    for model_name, model in trained["classification_models"].items():
        risk_probability = model.predict_proba(x_valid)[:, 1]
        threshold_df = evaluate_threshold_candidates(
            valid_df[classification_target],
            risk_probability,
            THRESHOLD_CANDIDATES,
        )
        threshold_df.insert(0, "model_name", model_name)
        threshold_rows.append(threshold_df)

        selected_threshold = select_operational_threshold(threshold_df, MIN_PRECISION_FOR_THRESHOLD)
        metric = classification_metrics(valid_df[classification_target], risk_probability, selected_threshold)
        metric_rows.append({
            "task": "classification",
            "model_name": model_name,
            "selected_threshold": selected_threshold,
            **metric,
        })

    metrics_df = pd.DataFrame(metric_rows)
    threshold_result_df = pd.concat(threshold_rows, ignore_index=True) if threshold_rows else pd.DataFrame()
    return metrics_df, threshold_result_df


def select_best_models(
    trained: dict[str, Any],
    metrics_df: pd.DataFrame,
    regression_metric: str = "rmse",
    classification_metric: str = "recall",
) -> dict[str, Any]:
    """검증 성능표를 기준으로 최종 회귀 모델과 최종 분류 모델을 선택합니다."""
    reg_metrics = metrics_df[metrics_df["task"].eq("regression")].copy()
    cls_metrics = metrics_df[metrics_df["task"].eq("classification")].copy()

    if reg_metrics.empty:
        raise ValueError("회귀 모델 평가 지표가 없습니다.")
    if cls_metrics.empty:
        raise ValueError("분류 모델 평가 지표가 없습니다.")

    # 회귀 오차 지표는 낮을수록 좋습니다.
    best_reg_name = reg_metrics.sort_values(regression_metric, ascending=True).iloc[0]["model_name"]

    # 분류 지표는 높을수록 좋습니다.
    best_cls_row = cls_metrics.sort_values(classification_metric, ascending=False).iloc[0]
    best_cls_name = best_cls_row["model_name"]

    return {
        "regression_model": trained["regression_models"][best_reg_name],
        "classification_model": trained["classification_models"][best_cls_name],
        "feature_columns": trained["feature_columns"],
        "best_regression_model_name": best_reg_name,
        "best_classification_model_name": best_cls_name,
        "best_classification_threshold": float(best_cls_row.get("selected_threshold", PROBABILITY_THRESHOLD)),
        "candidate_metrics": metrics_df,
    }