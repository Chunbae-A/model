from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    StackingClassifier,
    StackingRegressor,
)
from sklearn.linear_model import LogisticRegression, RidgeCV
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

from .config import (
    MIN_PRECISION_FOR_THRESHOLD,
    PROBABILITY_THRESHOLD,
    REGRESSION_TARGET_TRANSFORM,
    THRESHOLD_CANDIDATES,
)
from .model_config import get_enabled_models, get_model_params, load_model_config


def _require_package(import_error: Exception, package_name: str, model_name: str) -> None:
    raise ImportError(
        f"'{model_name}' is enabled in config/model_config.yaml, but package '{package_name}' "
        "is not installed or cannot be imported. Install the dependency or remove the model "
        "from enabled_models."
    ) from import_error


def _build_single_regression_model(
    model_name: str,
    model_config: dict[str, Any],
    random_state: int = 42,
) -> Any:
    params = get_model_params(model_config, model_name, "regression")
    params.setdefault("random_state", random_state)

    if model_name == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(**params)

    if model_name == "random_forest":
        return RandomForestRegressor(**params)

    if model_name == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
        except Exception as exc:
            _require_package(exc, "lightgbm", model_name)
        return LGBMRegressor(**params)

    if model_name == "xgboost":
        try:
            from xgboost import XGBRegressor
        except Exception as exc:
            _require_package(exc, "xgboost", model_name)
        return XGBRegressor(**params)

    if model_name == "catboost":
        try:
            from catboost import CatBoostRegressor
        except Exception as exc:
            _require_package(exc, "catboost", model_name)
        params.pop("random_state", None)
        return CatBoostRegressor(**params)

    raise ValueError(f"Unsupported regression model in YAML: {model_name}")


def _build_single_classification_model(
    model_name: str,
    model_config: dict[str, Any],
    random_state: int = 42,
) -> Any:
    params = get_model_params(model_config, model_name, "classification")
    params.setdefault("random_state", random_state)

    if model_name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(**params)

    if model_name == "random_forest":
        return RandomForestClassifier(**params)

    if model_name == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except Exception as exc:
            _require_package(exc, "lightgbm", model_name)
        return LGBMClassifier(**params)

    if model_name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except Exception as exc:
            _require_package(exc, "xgboost", model_name)
        return XGBClassifier(**params)

    if model_name == "catboost":
        try:
            from catboost import CatBoostClassifier
        except Exception as exc:
            _require_package(exc, "catboost", model_name)
        params.pop("random_state", None)
        return CatBoostClassifier(**params)

    raise ValueError(f"Unsupported classification model in YAML: {model_name}")


def _stacking_base_estimators(
    task: str,
    model_config: dict[str, Any],
    random_state: int,
) -> list[tuple[str, Any]]:
    params = get_model_params(model_config, "stacking_ensemble", task)
    estimator_names = params.get("estimators", [])
    if not estimator_names:
        raise ValueError("stacking_ensemble requires at least one base estimator in YAML.")

    estimators = []
    for model_name in estimator_names:
        if model_name == "stacking_ensemble":
            raise ValueError("stacking_ensemble cannot include itself as a base estimator.")
        if task == "regression":
            estimator = _build_single_regression_model(model_name, model_config, random_state)
        else:
            estimator = _build_single_classification_model(model_name, model_config, random_state)
        estimators.append((model_name, estimator))
    return estimators


def _build_stacking_regressor(model_config: dict[str, Any], random_state: int = 42) -> StackingRegressor:
    params = get_model_params(model_config, "stacking_ensemble", "regression")
    estimators = _stacking_base_estimators("regression", model_config, random_state)
    final_estimator_name = params.get("final_estimator", "ridge")
    if final_estimator_name != "ridge":
        raise ValueError(f"Unsupported stacking regression final_estimator: {final_estimator_name}")

    return StackingRegressor(
        estimators=estimators,
        final_estimator=RidgeCV(),
        cv=int(params.get("cv", 5)),
        passthrough=bool(params.get("passthrough", False)),
        n_jobs=params.get("n_jobs", 1),
    )


def _build_stacking_classifier(model_config: dict[str, Any], random_state: int = 42) -> StackingClassifier:
    params = get_model_params(model_config, "stacking_ensemble", "classification")
    estimators = _stacking_base_estimators("classification", model_config, random_state)
    final_estimator_name = params.get("final_estimator", "logistic_regression")
    if final_estimator_name != "logistic_regression":
        raise ValueError(f"Unsupported stacking classification final_estimator: {final_estimator_name}")

    return StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state),
        cv=int(params.get("cv", 5)),
        passthrough=bool(params.get("passthrough", False)),
        n_jobs=params.get("n_jobs", 1),
        stack_method="predict_proba",
    )


def build_regression_model_candidates(
    random_state: int = 42,
    model_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    model_config = model_config or load_model_config()
    candidates = {}
    for model_name in get_enabled_models(model_config, "regression"):
        if model_name == "stacking_ensemble":
            candidates[model_name] = _build_stacking_regressor(model_config, random_state)
        else:
            candidates[model_name] = _build_single_regression_model(model_name, model_config, random_state)
    return candidates


def build_classification_model_candidates(
    random_state: int = 42,
    model_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    model_config = model_config or load_model_config()
    candidates = {}
    for model_name in get_enabled_models(model_config, "classification"):
        if model_name == "stacking_ensemble":
            candidates[model_name] = _build_stacking_classifier(model_config, random_state)
        else:
            candidates[model_name] = _build_single_classification_model(model_name, model_config, random_state)
    return candidates


def train_candidate_models(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    regression_target: str,
    classification_target: str,
    random_state: int = 42,
    model_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    model_config = model_config or load_model_config()
    x_train = train_df[feature_columns]
    y_reg_train = train_df[regression_target]
    y_cls_train = train_df[classification_target]

    regression_candidates = build_regression_model_candidates(random_state=random_state, model_config=model_config)
    classification_candidates = build_classification_model_candidates(random_state=random_state, model_config=model_config)

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
        "model_config": model_config,
    }


def inverse_transform_cells(pred_target: np.ndarray) -> np.ndarray:
    pred_target = np.asarray(pred_target, dtype=float)
    if REGRESSION_TARGET_TRANSFORM == "log1p":
        cells = np.expm1(pred_target)
        return np.maximum(cells, 0)
    if REGRESSION_TARGET_TRANSFORM == "log10_plus_1":
        cells = (10 ** pred_target) - 1
        return np.maximum(cells, 0)
    if REGRESSION_TARGET_TRANSFORM == "none":
        return np.maximum(pred_target, 0)
    raise ValueError(f"Unsupported REGRESSION_TARGET_TRANSFORM: {REGRESSION_TARGET_TRANSFORM}")


def _safe_rmsle(y_true_cells: np.ndarray, y_pred_cells: np.ndarray) -> float | None:
    y_true_cells = np.asarray(y_true_cells, dtype=float)
    y_pred_cells = np.asarray(y_pred_cells, dtype=float)
    if np.any(y_true_cells < 0) or np.any(y_pred_cells < 0):
        return None
    return float(np.sqrt(mean_squared_error(np.log1p(y_true_cells), np.log1p(y_pred_cells))))


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float | None]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    metrics = {
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
    probability_threshold: float = PROBABILITY_THRESHOLD,
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
    min_precision: float = MIN_PRECISION_FOR_THRESHOLD,
) -> float:
    if threshold_df.empty:
        return float(PROBABILITY_THRESHOLD)
    candidates = threshold_df[threshold_df["precision"].ge(min_precision)].copy()
    if candidates.empty:
        candidates = threshold_df.copy()
    selected = candidates.sort_values(["recall", "f1", "precision"], ascending=[False, False, False]).iloc[0]
    return float(selected["threshold"])


def evaluate_candidate_models(
    trained: dict[str, Any],
    valid_df: pd.DataFrame,
    regression_target: str,
    classification_target: str,
    probability_threshold: float = PROBABILITY_THRESHOLD,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_columns = trained["feature_columns"]
    x_valid = valid_df[feature_columns]
    metric_rows = []
    threshold_rows = []

    for model_name, model in trained.get("regression_models", {}).items():
        pred = model.predict(x_valid)
        metric = regression_metrics(valid_df[regression_target], pred)
        metric_rows.append({"task": "regression", "model_name": model_name, **metric})

    if "log_target" in valid_df.columns:
        metric = regression_metrics(valid_df[regression_target], valid_df["log_target"].to_numpy())
        metric_rows.append({"task": "regression", "model_name": "persistence_baseline", **metric})

    for model_name, model in trained.get("classification_models", {}).items():
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

    if "alert_encoded" in valid_df.columns:
        baseline_probability = (valid_df["alert_encoded"] >= 1).astype(float).to_numpy()
        threshold_df = evaluate_threshold_candidates(
            valid_df[classification_target],
            baseline_probability,
            THRESHOLD_CANDIDATES,
        )
        threshold_df.insert(0, "model_name", "persistence_baseline")
        threshold_rows.append(threshold_df)
        selected_threshold = select_operational_threshold(threshold_df, MIN_PRECISION_FOR_THRESHOLD)
        metric = classification_metrics(valid_df[classification_target], baseline_probability, selected_threshold)
        metric_rows.append({
            "task": "classification",
            "model_name": "persistence_baseline",
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
    reg_metrics = metrics_df[metrics_df["task"].eq("regression")].copy()
    cls_metrics = metrics_df[metrics_df["task"].eq("classification")].copy()
    reg_metrics = reg_metrics[~reg_metrics["model_name"].eq("persistence_baseline")].copy()
    cls_metrics = cls_metrics[~cls_metrics["model_name"].eq("persistence_baseline")].copy()
    if reg_metrics.empty:
        raise ValueError("No fitted regression candidates are available.")
    if cls_metrics.empty:
        raise ValueError("No fitted classification candidates are available.")

    best_reg_name = reg_metrics.sort_values(regression_metric, ascending=True).iloc[0]["model_name"]
    best_cls_row = cls_metrics.sort_values(
        [classification_metric, "f1", "precision", "pr_auc"],
        ascending=[False, False, False, False],
    ).iloc[0]
    best_cls_name = best_cls_row["model_name"]
    return {
        "regression_model": trained["regression_models"][best_reg_name],
        "classification_model": trained["classification_models"][best_cls_name],
        "feature_columns": trained["feature_columns"],
        "best_regression_model_name": best_reg_name,
        "best_classification_model_name": best_cls_name,
        "best_classification_threshold": float(best_cls_row.get("selected_threshold", PROBABILITY_THRESHOLD)),
        "candidate_metrics": metrics_df,
        "model_config": trained.get("model_config"),
    }
