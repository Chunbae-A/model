from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import TrainConfig
from .features import build_master_table


EXCLUDE_PREFIXES = ("target_", "next_")
EXCLUDE_COLUMNS = {
    "조사일",
    "채수위치",
    "sample_date",
    "prev_sample_date",
    "next_sample_date",
    "cells",
    "log_cells",
    "alert",
    "발령단계",
}


def split_by_time(df: pd.DataFrame, config: TrainConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[df["sample_date"].dt.year <= config.train_end_year].copy()
    valid = df[df["sample_date"].dt.year == config.valid_year].copy()
    test = df[df["sample_date"].dt.year >= config.test_start_year].copy()
    return train, valid, test


def select_features(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    candidates = []
    for col in df.columns:
        if col in EXCLUDE_COLUMNS or col.startswith(EXCLUDE_PREFIXES):
            continue
        if pd.api.types.is_numeric_dtype(df[col]) or col == "site":
            candidates.append(col)
    numeric = [col for col in candidates if pd.api.types.is_numeric_dtype(df[col])]
    categorical = [col for col in candidates if col not in numeric]
    return numeric, categorical


def make_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    transformers = []
    if numeric:
        transformers.append(
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "cat",
                Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]),
                categorical,
            )
        )
    return ColumnTransformer(transformers)


def rmse(y_true: pd.Series, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def regression_metrics(split: str, model: str, y_true: pd.Series, y_pred: np.ndarray) -> dict:
    return {
        "task": "regression",
        "split": split,
        "model": model,
        "n": int(len(y_true)),
        "rmse_log": rmse(y_true, y_pred),
        "mae_log": float(mean_absolute_error(y_true, y_pred)),
        "bias_log": float(np.mean(y_pred - y_true)),
    }


def classification_metrics(split: str, model: str, y_true: pd.Series, proba: np.ndarray, threshold: float) -> dict:
    pred = (proba >= threshold).astype(int)
    return {
        "task": "classification",
        "split": split,
        "model": model,
        "n": int(len(y_true)),
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision_alert": float(precision_score(y_true, pred, zero_division=0)),
        "recall_alert": float(recall_score(y_true, pred, zero_division=0)),
        "f1_alert": float(f1_score(y_true, pred, zero_division=0)),
    }


def choose_threshold(y_true: pd.Series, proba: np.ndarray) -> float:
    thresholds = np.linspace(0.1, 0.9, 33)
    best = (0.0, 0.5)
    for threshold in thresholds:
        pred = (proba >= threshold).astype(int)
        recall = recall_score(y_true, pred, zero_division=0)
        f1 = f1_score(y_true, pred, zero_division=0)
        score = 0.7 * recall + 0.3 * f1
        if score > best[0]:
            best = (score, float(threshold))
    return best[1]


def train_pipeline(config: TrainConfig) -> dict:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "models").mkdir(exist_ok=True)
    (config.output_dir / "tables").mkdir(exist_ok=True)

    master = build_master_table(config.data_paths)
    model_df = master.dropna(subset=["target_log_cells_next", "target_alert_next"]).copy()
    train, valid, test = split_by_time(model_df, config)
    numeric, categorical = select_features(model_df)
    numeric = [col for col in numeric if train[col].notna().any()]
    categorical = [col for col in categorical if train[col].notna().any()]
    features = numeric + categorical
    preprocessor = make_preprocessor(numeric, categorical)

    x_train, y_reg_train, y_cls_train = train[features], train["target_log_cells_next"], train["target_alert_next"].astype(int)
    x_valid, y_reg_valid, y_cls_valid = valid[features], valid["target_log_cells_next"], valid["target_alert_next"].astype(int)
    x_test, y_reg_test, y_cls_test = test[features], test["target_log_cells_next"], test["target_alert_next"].astype(int)

    regressors = {
        "hist_gradient_boosting": HistGradientBoostingRegressor(random_state=config.random_state, max_iter=300, learning_rate=0.04),
    }
    classifiers = {
        "hist_gradient_boosting": HistGradientBoostingClassifier(random_state=config.random_state, max_iter=300, learning_rate=0.04),
        "random_forest": RandomForestClassifier(
            random_state=config.random_state,
            n_estimators=500,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            n_jobs=-1,
        ),
    }

    metrics = []
    predictions = test[["site", "sample_date", "target_cells_next", "target_log_cells_next", "target_alert_next"]].copy()

    for name, estimator in regressors.items():
        pipe = Pipeline([("prep", preprocessor), ("model", estimator)])
        pipe.fit(x_train, y_reg_train)
        for split_name, x, y in [("valid", x_valid, y_reg_valid), ("test", x_test, y_reg_test)]:
            metrics.append(regression_metrics(split_name, name, y, pipe.predict(x)))
        predictions[f"{name}_pred_log_cells"] = pipe.predict(x_test)
        predictions[f"{name}_pred_cells"] = np.power(10, predictions[f"{name}_pred_log_cells"]) - 1
        joblib.dump(pipe, config.output_dir / "models" / f"{name}_regressor.joblib")

    for name, estimator in classifiers.items():
        pipe = Pipeline([("prep", preprocessor), ("model", estimator)])
        pipe.fit(x_train, y_cls_train)
        valid_proba = pipe.predict_proba(x_valid)[:, 1]
        threshold = choose_threshold(y_cls_valid, valid_proba)
        for split_name, x, y in [("valid", x_valid, y_cls_valid), ("test", x_test, y_cls_test)]:
            proba = pipe.predict_proba(x)[:, 1]
            metrics.append(classification_metrics(split_name, name, y, proba, threshold))
        predictions[f"{name}_alert_proba"] = pipe.predict_proba(x_test)[:, 1]
        predictions[f"{name}_alert_pred"] = (predictions[f"{name}_alert_proba"] >= threshold).astype(int)
        joblib.dump(pipe, config.output_dir / "models" / f"{name}_classifier.joblib")

    master.to_csv(config.output_dir / "tables" / "master_table.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"feature": features, "type": ["numeric"] * len(numeric) + ["categorical"] * len(categorical)}).to_csv(
        config.output_dir / "tables" / "feature_list.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(metrics).to_csv(config.output_dir / "tables" / "metrics.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(config.output_dir / "tables" / "predictions.csv", index=False, encoding="utf-8-sig")

    manifest = {
        "output_dir": str(config.output_dir),
        "rows_master": int(len(master)),
        "rows_model": int(len(model_df)),
        "features": int(len(features)),
        "train_rows": int(len(train)),
        "valid_rows": int(len(valid)),
        "test_rows": int(len(test)),
    }
    (config.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest
