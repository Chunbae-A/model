from __future__ import annotations

import json
from typing import Any
from pathlib import Path

import joblib
import pandas as pd

from .config import (
    REGRESSION_MODEL_FILE,
    CLASSIFICATION_MODEL_FILE,
    PREPROCESSING_PIPELINE_FILE,
    REGRESSION_PREDICTION_FILE,
    CLASSIFICATION_PREDICTION_FILE,
    REGRESSION_METRIC_FILE,
    CLASSIFICATION_METRIC_FILE,
)


def _json_default(obj: Any) -> Any:
    import numpy as _np
    import pandas as _pd

    if isinstance(obj, (_np.integer, _np.floating)):
        return obj.item()
    if isinstance(obj, _np.ndarray):
        return obj.tolist()
    if _pd.isna(obj):
        return None
    return str(obj)


def save_models(
    trained: dict[str, Any],
    best_models: dict[str, Any],
    output_dir: Path,
    preprocessing_pipeline: Any | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    regression_dir = output_dir / "regression_candidates"
    classification_dir = output_dir / "classification_candidates"
    regression_dir.mkdir(exist_ok=True)
    classification_dir.mkdir(exist_ok=True)

    for model_name, model in trained.get("regression_models", {}).items():
        joblib.dump(model, regression_dir / f"{model_name}.joblib")
    for model_name, model in trained.get("classification_models", {}).items():
        joblib.dump(model, classification_dir / f"{model_name}.joblib")

    joblib.dump(best_models["regression_model"], output_dir / REGRESSION_MODEL_FILE)
    joblib.dump(best_models["classification_model"], output_dir / CLASSIFICATION_MODEL_FILE)
    joblib.dump(preprocessing_pipeline, output_dir / PREPROCESSING_PIPELINE_FILE)

    if "candidate_metrics" in best_models:
        best_models["candidate_metrics"].to_csv(output_dir / "candidate_model_metrics.csv", index=False)

    metadata = {
        "feature_columns": best_models["feature_columns"],
        "regression_target": best_models.get("regression_target"),
        "classification_target": best_models.get("classification_target"),
        "best_regression_model_name": best_models.get("best_regression_model_name"),
        "best_classification_model_name": best_models.get("best_classification_model_name"),
        "best_classification_threshold": best_models.get("best_classification_threshold"),
        "regression_model_candidates": list(trained.get("regression_models", {}).keys()),
        "classification_model_candidates": list(trained.get("classification_models", {}).keys()),
        "model_config": best_models.get("model_config"),
    }
    with open(output_dir / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2, default=_json_default)


def load_models(model_dir: Path) -> dict[str, Any]:
    with open(model_dir / "model_metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    regression_models = {}
    for model_name in metadata.get("regression_model_candidates", []):
        model_path = model_dir / "regression_candidates" / f"{model_name}.joblib"
        if model_path.exists():
            regression_models[model_name] = joblib.load(model_path)

    classification_models = {}
    for model_name in metadata.get("classification_model_candidates", []):
        model_path = model_dir / "classification_candidates" / f"{model_name}.joblib"
        if model_path.exists():
            classification_models[model_name] = joblib.load(model_path)

    return {
        "regression_model": joblib.load(model_dir / REGRESSION_MODEL_FILE),
        "classification_model": joblib.load(model_dir / CLASSIFICATION_MODEL_FILE),
        "preprocessing_pipeline": joblib.load(model_dir / PREPROCESSING_PIPELINE_FILE),
        "regression_models": regression_models,
        "classification_models": classification_models,
        "feature_columns": metadata.get("feature_columns"),
        "metadata": metadata,
    }


def save_metrics(metrics_df: pd.DataFrame, metric_dir: Path) -> None:
    metric_dir.mkdir(parents=True, exist_ok=True)
    regression_metrics_df = metrics_df[metrics_df["task"].eq("regression")].copy()
    classification_metrics_df = metrics_df[metrics_df["task"].eq("classification")].copy()
    regression_payload = regression_metrics_df.to_dict(orient="records")
    classification_payload = classification_metrics_df.to_dict(orient="records")
    with open(metric_dir / REGRESSION_METRIC_FILE, "w", encoding="utf-8") as f:
        json.dump(regression_payload, f, ensure_ascii=False, indent=2, default=_json_default)
    with open(metric_dir / CLASSIFICATION_METRIC_FILE, "w", encoding="utf-8") as f:
        json.dump(classification_payload, f, ensure_ascii=False, indent=2, default=_json_default)


def save_threshold_results(threshold_df: pd.DataFrame, metric_dir: Path) -> None:
    metric_dir.mkdir(parents=True, exist_ok=True)
    threshold_df.to_csv(metric_dir / "classification_threshold_candidates.csv", index=False)


def save_prediction_outputs(
    prediction_df: pd.DataFrame,
    original_df: pd.DataFrame,
    prediction_dir: Path,
    regression_target: str,
    classification_target: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_dir.mkdir(parents=True, exist_ok=True)
    id_cols = [
        col
        for col in [
            "date",
            "sample_date",
            "site",
            "loc_encoded",
            "location_name",
            "loc_flow_order",
            "sampling_gap_days",
            "previous_observed_cells",
            "previous_exceeded",
        ]
        if col in prediction_df.columns
    ]
    regression_df = prediction_df[id_cols].copy()
    if regression_target in original_df.columns:
        regression_df["y_true_target"] = original_df[regression_target].to_numpy()
        from .models import inverse_transform_cells

        regression_df["y_true_cells"] = inverse_transform_cells(original_df[regression_target].to_numpy())

    if "pred_regression_target" in prediction_df.columns:
        regression_df["y_pred_target"] = prediction_df["pred_regression_target"].to_numpy()
    elif "predicted_cells" in prediction_df.columns:
        regression_df["y_pred_target"] = prediction_df["predicted_cells"].to_numpy()
    else:
        regression_df["y_pred_target"] = None

    if "predicted_cells" in prediction_df.columns:
        regression_df["y_pred_cells"] = prediction_df["predicted_cells"].to_numpy()
    regression_df.to_csv(prediction_dir / REGRESSION_PREDICTION_FILE, index=False)

    classification_df = prediction_df[id_cols].copy()
    if classification_target in original_df.columns:
        classification_df["y_true_alert"] = original_df[classification_target].to_numpy()
    for extra_col in [
        "operational_alert_target",
        "predicted_alert_stage",
        "predicted_cell_exceeded",
        "operational_alert_candidate",
    ]:
        if extra_col in prediction_df.columns:
            classification_df[extra_col] = prediction_df[extra_col].to_numpy()
    classification_df["y_pred_alert"] = (
        prediction_df["predicted_alert_label"] if "predicted_alert_label" in prediction_df.columns else None
    )
    classification_df["y_pred_probability"] = (
        prediction_df["alert_probability"] if "alert_probability" in prediction_df.columns else None
    )
    classification_df.to_csv(prediction_dir / CLASSIFICATION_PREDICTION_FILE, index=False)
    return regression_df, classification_df


def save_run_reports(trained: dict[str, Any], metrics_df: pd.DataFrame, output_root: Path) -> Path:
    """Save a per-run report directory with per-model text files and a run metadata file.

    - output_root: base folder (relative to cwd) where runs are created; e.g. `output/`.
    Returns the Path to the created run directory.
    """
    from datetime import datetime

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"run_{run_id}"
    models_dir = run_dir / "models"
    run_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "run_id": run_id,
        "num_regression_candidates": len(trained.get("regression_models", {})),
        "num_classification_candidates": len(trained.get("classification_models", {})),
    }
    with open(run_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2, default=_json_default)

    if metrics_df is not None and not metrics_df.empty:
        for model_name in metrics_df["model_name"].unique():
            mdf = metrics_df[metrics_df["model_name"].eq(model_name)].copy()
            txt_path = models_dir / f"{model_name}.txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(f"Model: {model_name}\n")
                for _, row in mdf.iterrows():
                    f.write("- " + json.dumps(row.dropna().to_dict(), ensure_ascii=False, default=_json_default) + "\n")

        try:
            reg_df = metrics_df[metrics_df["task"].eq("regression")].copy()
            cls_df = metrics_df[metrics_df["task"].eq("classification")].copy()

            best_reg = None
            if not reg_df.empty:
                if "rmse" in reg_df.columns:
                    best_reg = reg_df.loc[reg_df["rmse"].idxmin()].to_dict()
                elif "mae" in reg_df.columns:
                    best_reg = reg_df.loc[reg_df["mae"].idxmin()].to_dict()

            best_cls = None
            if not cls_df.empty:
                if "f1" in cls_df.columns:
                    best_cls = cls_df.loc[cls_df["f1"].idxmax()].to_dict()
                elif "roc_auc" in cls_df.columns:
                    best_cls = cls_df.loc[cls_df["roc_auc"].idxmax()].to_dict()
                elif "accuracy" in cls_df.columns:
                    best_cls = cls_df.loc[cls_df["accuracy"].idxmax()].to_dict()

            def _reg_score(d: dict[str, Any]) -> float:
                if d is None:
                    return 0.0
                if "rmse" in d and d["rmse"] is not None:
                    return 1.0 / (float(d["rmse"]) + 1e-8)
                if "mae" in d and d["mae"] is not None:
                    return 1.0 / (float(d["mae"]) + 1e-8)
                return 0.0

            def _cls_score(d: dict[str, Any]) -> float:
                if d is None:
                    return 0.0
                for k in ("f1", "roc_auc", "accuracy"):
                    if k in d and d[k] is not None:
                        return float(d[k])
                return 0.0

            reg_score = _reg_score(best_reg)
            cls_score = _cls_score(best_cls)

            overall = None
            if reg_score <= 0 and cls_score <= 0:
                overall = None
            elif cls_score >= reg_score:
                overall = {"task": "classification", "best": best_cls}
            else:
                overall = {"task": "regression", "best": best_reg}

            summary = {
                "best_regression": best_reg,
                "best_classification": best_cls,
                "overall_best": overall,
            }

            with open(run_dir / "best_model_summary.txt", "w", encoding="utf-8") as f:
                f.write(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default))
        except Exception:
            # best-effort only; don't fail the run reporting on summary errors
            pass

    return run_dir
