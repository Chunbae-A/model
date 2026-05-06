from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

@dataclass(frozen=True)
class WorkflowConfig:
    key: str
    label: str
    model_input_path: Path
    regression_candidates: list[str]
    classification_candidates: list[str]
    artifact_subdir: str


TREE_MODEL_INPUT_PATH = (
    PROJECT_ROOT
    / "src/data/processed/model_input/tree_gradient_boosting/algae_tree_station_expanded.csv"
)
NON_TREE_MODEL_INPUT_PATH = (
    PROJECT_ROOT
    / "src/data/processed/model_input/non_tree_scaled/algae_non_tree_scaled_station_expanded.csv"
)

WORKFLOWS = {
    "tree": WorkflowConfig(
        key="tree",
        label="Tree Gradient Boosting",
        model_input_path=TREE_MODEL_INPUT_PATH,
        regression_candidates=["lightgbm", "xgboost", "hist_gradient_boosting", "random_forest", "catboost"],
        classification_candidates=["lightgbm", "xgboost", "hist_gradient_boosting", "random_forest", "catboost"],
        artifact_subdir="tree_gradient_boosting",
    ),
    "non_tree": WorkflowConfig(
        key="non_tree",
        label="Non-tree Scaled",
        model_input_path=NON_TREE_MODEL_INPUT_PATH,
        regression_candidates=["ridge", "elasticnet", "huber_regressor", "svr_rbf", "knn_regressor"],
        classification_candidates=["logistic_regression", "calibrated_logistic_regression", "svc_rbf", "knn_classifier"],
        artifact_subdir="non_tree_scaled",
    ),
}

DEFAULT_WORKFLOW = "tree"
MODEL_INPUT_PATH = WORKFLOWS[DEFAULT_WORKFLOW].model_input_path

# ---------------------------------------------------------------------------
# Column contract
# ---------------------------------------------------------------------------
DATE_COLUMN = "date"
SITE_COLUMN = "loc_encoded"
STATION_COLUMN = "station"
SPLIT_COLUMN = "split"
ID_COLUMNS = [DATE_COLUMN, SITE_COLUMN, STATION_COLUMN]

REGRESSION_TARGET = "next_log_cells"
CLASSIFICATION_TARGET = "target_alert_next"
REGRESSION_TARGET_TRANSFORM = "log10_plus_1"

# Columns that must never be used as model features.
DROP_COLUMNS = [
    DATE_COLUMN,
    SPLIT_COLUMN,
    REGRESSION_TARGET,
    CLASSIFICATION_TARGET,
]

FORBIDDEN_FEATURE_KEYWORDS = [
    "target_alert_next",
    "next_log_cells",
]

REQUIRE_NUMERIC_FEATURES = True

# ---------------------------------------------------------------------------
# Modeling
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
VALID_SIZE_RATIO = 0.2

REGRESSION_MODEL_CANDIDATES = WORKFLOWS[DEFAULT_WORKFLOW].regression_candidates
CLASSIFICATION_MODEL_CANDIDATES = WORKFLOWS[DEFAULT_WORKFLOW].classification_candidates

MODEL_SELECTION_METRIC_REGRESSION = "rmse"
MODEL_SELECTION_METRIC_CLASSIFICATION = "recall"

ALERT_CELL_THRESHOLD = 1000
PROBABILITY_THRESHOLD = 0.5
THRESHOLD_CANDIDATES = [round(x, 2) for x in np.arange(0.10, 0.91, 0.05)]
MIN_PRECISION_FOR_THRESHOLD = 0.30

# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
OUTPUT_DIR = ARTIFACT_DIR / "models"
PREDICTION_DIR = ARTIFACT_DIR / "predictions"
METRIC_DIR = ARTIFACT_DIR / "metrics"
EXPLAIN_DIR = ARTIFACT_DIR / "explain"

REGRESSION_MODEL_FILE = "regression_model.pkl"
CLASSIFICATION_MODEL_FILE = "classification_model.pkl"
MODEL_BUNDLE_FILE = "model_bundle.pkl"

REGRESSION_PREDICTION_FILE = "regression_predictions.csv"
CLASSIFICATION_PREDICTION_FILE = "classification_predictions.csv"
REGRESSION_METRIC_FILE = "regression_metrics.json"
CLASSIFICATION_METRIC_FILE = "classification_metrics.json"
THRESHOLD_CANDIDATE_FILE = "classification_threshold_candidates.csv"
FEATURE_IMPORTANCE_FILE = "feature_importance.csv"
