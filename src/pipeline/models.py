from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR

from src.config import model_config as config


def _expand_model_candidates(candidates: list[str]) -> list[str]:
    if "auto" in candidates:
        return ["lightgbm", "xgboost", "hist_gradient_boosting"]
    return candidates


def build_regression_model_candidates(
    candidates: list[str],
    random_state: int = config.RANDOM_STATE,
) -> dict[str, Any]:
    models: dict[str, Any] = {}
    for name in _expand_model_candidates(candidates):
        if name == "lightgbm":
            try:
                from lightgbm import LGBMRegressor

                models[name] = LGBMRegressor(
                    n_estimators=400,
                    learning_rate=0.03,
                    num_leaves=31,
                    random_state=random_state,
                    verbosity=-1,
                )
            except Exception:
                continue
        elif name == "xgboost":
            try:
                from xgboost import XGBRegressor

                models[name] = XGBRegressor(
                    n_estimators=400,
                    learning_rate=0.03,
                    max_depth=4,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    random_state=random_state,
                    objective="reg:squarederror",
                )
            except Exception:
                continue
        elif name == "hist_gradient_boosting":
            models[name] = HistGradientBoostingRegressor(
                max_iter=300,
                learning_rate=0.05,
                l2_regularization=0.01,
                random_state=random_state,
            )
        elif name == "ridge":
            models[name] = Ridge(alpha=1.0)
        elif name == "elasticnet":
            models[name] = ElasticNet(alpha=0.001, l1_ratio=0.2, random_state=random_state, max_iter=10000)
        elif name == "svr_rbf":
            models[name] = SVR(kernel="rbf", C=10.0, epsilon=0.05)
        elif name == "knn_regressor":
            models[name] = KNeighborsRegressor(n_neighbors=12, weights="distance")
        else:
            raise ValueError(f"Unsupported regression candidate: {name}")
    if not models:
        raise RuntimeError("No regression model candidates are available.")
    return models


def build_classification_model_candidates(
    candidates: list[str],
    random_state: int = config.RANDOM_STATE,
) -> dict[str, Any]:
    models: dict[str, Any] = {}
    for name in _expand_model_candidates(candidates):
        if name == "lightgbm":
            try:
                from lightgbm import LGBMClassifier

                models[name] = LGBMClassifier(
                    n_estimators=400,
                    learning_rate=0.03,
                    num_leaves=31,
                    random_state=random_state,
                    verbosity=-1,
                )
            except Exception:
                continue
        elif name == "xgboost":
            try:
                from xgboost import XGBClassifier

                models[name] = XGBClassifier(
                    n_estimators=400,
                    learning_rate=0.03,
                    max_depth=4,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    random_state=random_state,
                    eval_metric="logloss",
                )
            except Exception:
                continue
        elif name == "hist_gradient_boosting":
            models[name] = HistGradientBoostingClassifier(
                max_iter=300,
                learning_rate=0.05,
                l2_regularization=0.01,
                random_state=random_state,
            )
        elif name == "logistic_regression":
            models[name] = LogisticRegression(
                max_iter=10000,
                solver="liblinear",
                class_weight="balanced",
                random_state=random_state,
            )
        elif name == "svc_rbf":
            models[name] = SVC(kernel="rbf", C=5.0, probability=True, class_weight="balanced", random_state=random_state)
        elif name == "knn_classifier":
            models[name] = KNeighborsClassifier(n_neighbors=12, weights="distance")
        else:
            raise ValueError(f"Unsupported classification candidate: {name}")
    if not models:
        raise RuntimeError("No classification model candidates are available.")
    return models


def train_candidate_models(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    workflow: config.WorkflowConfig,
) -> dict[str, Any]:
    x_train = train_df[feature_columns]
    y_reg = train_df[config.REGRESSION_TARGET]
    y_cls = train_df[config.CLASSIFICATION_TARGET]

    regression_models = {}
    for name, model in build_regression_model_candidates(workflow.regression_candidates).items():
        fitted = clone(model)
        fitted.fit(x_train, y_reg)
        regression_models[name] = fitted

    classification_models = {}
    for name, model in build_classification_model_candidates(workflow.classification_candidates).items():
        fitted = clone(model)
        fitted.fit(x_train, y_cls)
        classification_models[name] = fitted

    return {
        "regression_models": regression_models,
        "classification_models": classification_models,
        "feature_columns": feature_columns,
    }
