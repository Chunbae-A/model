from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, HuberRegressor, LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR

from src.config import model_config as config


def _expand_model_candidates(candidates: list[str]) -> list[str]:
    """`auto` 별칭을 실제 모델 후보 목록으로 확장한다."""

    if "auto" in candidates:
        return ["lightgbm", "xgboost", "hist_gradient_boosting"]
    return candidates


def build_regression_model_candidates(
    candidates: list[str],
    random_state: int = config.RANDOM_STATE,
) -> dict[str, Any]:
    """회귀 후보 모델을 생성한다.

    optional 라이브러리인 LightGBM, XGBoost, CatBoost는 설치되어 있을 때만
    후보에 포함한다. 이렇게 해두면 실험 환경이 조금 달라도 파이프라인이
    완전히 멈추지 않고, 사용 가능한 후보끼리 비교할 수 있다.
    """

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
            # scikit-learn 내장 gradient boosting. 외부 패키지 없이 재현 가능한 기준 모델이다.
            models[name] = HistGradientBoostingRegressor(
                max_iter=300,
                learning_rate=0.05,
                l2_regularization=0.01,
                random_state=random_state,
            )
        elif name == "random_forest":
            # 여러 tree의 평균으로 예측해 단일 tree보다 분산을 줄인다.
            models[name] = RandomForestRegressor(
                n_estimators=400,
                max_depth=None,
                min_samples_leaf=3,
                n_jobs=-1,
                random_state=random_state,
            )
        elif name == "catboost":
            try:
                from catboost import CatBoostRegressor

                models[name] = CatBoostRegressor(
                    iterations=500,
                    learning_rate=0.03,
                    depth=5,
                    loss_function="RMSE",
                    random_seed=random_state,
                    verbose=False,
                    allow_writing_files=False,
                )
            except Exception:
                continue
        elif name == "ridge":
            # L2 규제로 상관된 feature가 많은 상황에서 계수를 안정화하는 선형 baseline.
            models[name] = Ridge(alpha=1.0)
        elif name == "elasticnet":
            # L1+L2 규제 조합. 중복 feature를 줄이면서도 완전히 불안정한 선택을 피한다.
            models[name] = ElasticNet(alpha=0.001, l1_ratio=0.2, random_state=random_state, max_iter=10000)
        elif name == "huber_regressor":
            # 큰 오차에 덜 민감한 robust 회귀. 폭우/조류 폭증 같은 튀는 구간을 고려한다.
            models[name] = HuberRegressor(epsilon=1.35, alpha=0.0001, max_iter=5000)
        elif name == "svr_rbf":
            # 비선형 margin 기반 회귀. 스케일링된 non-tree 입력에서만 의미 있게 비교한다.
            models[name] = SVR(kernel="rbf", C=10.0, epsilon=0.05)
        elif name == "knn_regressor":
            # 가까운 관측치의 target을 참조하는 거리 기반 모델. feature scale에 매우 민감하다.
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
    """분류 후보 모델을 생성한다.

    조류경보 분류에서는 위험을 놓치지 않는 것이 중요하므로 이후 평가에서
    recall을 우선 지표로 사용한다. 여기서는 각 후보의 기본 파라미터를
    비교 가능한 수준으로 맞춰 생성만 담당한다.
    """

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
        elif name == "random_forest":
            # class_weight를 적용해 경보/비경보 비율이 달라져도 양성 class를 덜 놓치게 한다.
            models[name] = RandomForestClassifier(
                n_estimators=400,
                max_depth=None,
                min_samples_leaf=3,
                class_weight="balanced",
                n_jobs=-1,
                random_state=random_state,
            )
        elif name == "catboost":
            try:
                from catboost import CatBoostClassifier

                models[name] = CatBoostClassifier(
                    iterations=500,
                    learning_rate=0.03,
                    depth=5,
                    loss_function="Logloss",
                    eval_metric="Recall",
                    auto_class_weights="Balanced",
                    random_seed=random_state,
                    verbose=False,
                    allow_writing_files=False,
                )
            except Exception:
                continue
        elif name == "logistic_regression":
            # 확률 해석이 쉽고 운영 threshold 조정이 쉬운 조기경보 baseline.
            models[name] = LogisticRegression(
                max_iter=10000,
                solver="liblinear",
                class_weight="balanced",
                random_state=random_state,
            )
        elif name == "calibrated_logistic_regression":
            # 로지스틱 회귀 확률을 한 번 더 보정해, "위험 확률"로 보고하기 적합한지 확인한다.
            base_model = LogisticRegression(
                max_iter=10000,
                solver="liblinear",
                class_weight="balanced",
                random_state=random_state,
            )
            models[name] = CalibratedClassifierCV(
                estimator=base_model,
                method="sigmoid",
                cv=3,
            )
        elif name == "svc_rbf":
            # 비선형 결정경계를 확인하는 SVM 후보. probability=True로 ROC/PR 계산을 가능하게 한다.
            models[name] = SVC(kernel="rbf", C=5.0, probability=True, class_weight="balanced", random_state=random_state)
        elif name == "knn_classifier":
            # 유사한 수질/수문 조건의 과거 사례를 참조하는 거리 기반 분류 baseline.
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
    """회귀와 분류 후보 모델을 모두 학습한다.

    동일한 feature matrix에서 회귀 target(`next_log_cells`)과 분류 target
    (`target_alert_next`)을 각각 학습한다. clone을 사용해 config에 정의된
    원본 estimator가 fit 과정에서 변하지 않도록 한다.
    """

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
