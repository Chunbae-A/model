from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.config import model_config as config
from src.pipeline import artifacts, data, evaluation, models
from src.pipeline.runner import get_workflow, run_training_pipeline, run_workflow_comparison


def load_model_input(path: Path | None = None) -> pd.DataFrame:
    workflow = get_workflow()
    return data.load_model_input(path or workflow.model_input_path)


def time_based_train_valid_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return data.time_based_train_valid_split(df)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return data.get_feature_columns(df)


def train_candidate_models(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    workflow_key: str = config.DEFAULT_WORKFLOW,
) -> dict[str, Any]:
    return models.train_candidate_models(train_df, feature_columns, get_workflow(workflow_key))


def evaluate_candidate_models(
    trained: dict[str, Any],
    valid_df: pd.DataFrame,
    workflow_key: str = config.DEFAULT_WORKFLOW,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return evaluation.evaluate_candidate_models(trained, valid_df, workflow_key)


def select_best_models(metrics_df: pd.DataFrame) -> dict[str, str]:
    return evaluation.select_best_models(metrics_df)


def make_predictions(
    trained: dict[str, Any],
    valid_df: pd.DataFrame,
    best_models: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return artifacts.make_predictions(trained, valid_df, best_models)


def get_feature_importance(
    trained: dict[str, Any],
    valid_df: pd.DataFrame,
    best_models: dict[str, str],
) -> pd.DataFrame:
    return artifacts.get_feature_importance(trained, valid_df, best_models)


def save_artifacts(
    trained: dict[str, Any],
    metrics_df: pd.DataFrame,
    threshold_df: pd.DataFrame,
    best_models: dict[str, str],
    regression_predictions: pd.DataFrame,
    classification_predictions: pd.DataFrame,
    feature_importance: pd.DataFrame,
    workflow_key: str = config.DEFAULT_WORKFLOW,
) -> dict[str, Path]:
    return artifacts.save_artifacts(
        get_workflow(workflow_key),
        trained,
        metrics_df,
        threshold_df,
        best_models,
        regression_predictions,
        classification_predictions,
        feature_importance,
    )
