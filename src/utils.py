import pandas as pd
import numpy as np
from typing import Any
from sklearn.base import clone
from pathlib import Path
import joblib
import json
from sklearn.metrics import mean_squared_error
from src.config import (
    REGRESSION_TARGET, 
    CLASSIFICATION_TARGET,
    REGRESSION_TARGET_TRANSFORM,
    PROBABILITY_THRESHOLD,
    REGRESSION_MODEL_FILE,
    CLASSIFICATION_MODEL_FILE,
    PREPROCESSING_PIPELINE_FILE,
    ALERT_CELL_THRESHOLD,
    MODEL_SELECTION_METRIC_REGRESSION,
    MODEL_SELECTION_METRIC_CLASSIFICATION,
    REGRESSION_METRIC_FILE,
    CLASSIFICATION_METRIC_FILE,
    DATE_COLUMN, SITE_COLUMN,
    REGRESSION_PREDICTION_FILE,
    CLASSIFICATION_PREDICTION_FILE
)



def inverse_transform_cells(pred_target: np.ndarray) -> np.ndarray:
    """회귀 예측값을 target 변환 방식에 맞춰 cells/mL 단위로 복원합니다."""
    pred_target = np.asarray(pred_target, dtype=float)

    if REGRESSION_TARGET_TRANSFORM == "log1p":
        return np.expm1(pred_target)
    if REGRESSION_TARGET_TRANSFORM == "log10_plus_1":
        return (10 ** pred_target) - 1
    if REGRESSION_TARGET_TRANSFORM == "none":
        return pred_target

    raise ValueError(f"지원하지 않는 REGRESSION_TARGET_TRANSFORM 값입니다: {REGRESSION_TARGET_TRANSFORM}")


def _safe_rmsle(y_true_cells: np.ndarray, y_pred_cells: np.ndarray) -> float | None:
    """실제값과 예측값이 모두 음수가 아닐 때만 RMSLE를 계산합니다."""
    y_true_cells = np.asarray(y_true_cells, dtype=float)
    y_pred_cells = np.asarray(y_pred_cells, dtype=float)

    if np.any(y_true_cells < 0) or np.any(y_pred_cells < 0):
        return None

    return float(np.sqrt(mean_squared_error(np.log1p(y_true_cells), np.log1p(y_pred_cells))))


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if pd.isna(obj):
        return None
    return str(obj)


def save_models(
    trained: dict[str, Any],
    best_models: dict[str, Any],
    output_dir: Path,
    preprocessing_pipeline: Any | None = None,
) -> None:
    """모든 후보 모델, 선택된 best model, 비교 지표를 저장합니다."""
    output_dir.mkdir(parents=True, exist_ok=True)

    regression_dir = output_dir / "regression_candidates"
    classification_dir = output_dir / "classification_candidates"
    regression_dir.mkdir(exist_ok=True)
    classification_dir.mkdir(exist_ok=True)

    for model_name, model in trained["regression_models"].items():
        joblib.dump(model, regression_dir / f"{model_name}.joblib")

    for model_name, model in trained["classification_models"].items():
        joblib.dump(model, classification_dir / f"{model_name}.joblib")

    # 추론 편의성과 최종 산출물 계약을 위해 최종 모델 별칭을 저장합니다.
    joblib.dump(best_models["regression_model"], output_dir / REGRESSION_MODEL_FILE)
    joblib.dump(best_models["classification_model"], output_dir / CLASSIFICATION_MODEL_FILE)

    # 모델링 에이전트는 전처리를 만들지 않습니다.
    # 전처리팀이 학습 완료된 파이프라인을 제공하면 이 인자로 넘기고, 없으면 None을 저장합니다.
    joblib.dump(preprocessing_pipeline, output_dir / PREPROCESSING_PIPELINE_FILE)

    if "candidate_metrics" in best_models:
        best_models["candidate_metrics"].to_csv(output_dir / "candidate_model_metrics.csv", index=False, encoding='utf-8-sig')

    metadata = {
        "feature_columns": best_models["feature_columns"],
        "regression_target": REGRESSION_TARGET,
        "classification_target": CLASSIFICATION_TARGET,
        "probability_threshold": best_models.get("best_classification_threshold", PROBABILITY_THRESHOLD),
        "alert_cell_threshold": ALERT_CELL_THRESHOLD,
        "regression_target_transform": REGRESSION_TARGET_TRANSFORM,
        "best_regression_model_name": best_models.get("best_regression_model_name"),
        "best_classification_model_name": best_models.get("best_classification_model_name"),
        "regression_model_candidates": list(trained["regression_models"].keys()),
        "classification_model_candidates": list(trained["classification_models"].keys()),
        "model_selection_metric_regression": MODEL_SELECTION_METRIC_REGRESSION,
        "model_selection_metric_classification": MODEL_SELECTION_METRIC_CLASSIFICATION,
    }
    with open(output_dir / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2, default=_json_default)


def load_models(model_dir: Path) -> dict[str, Any]:
    """저장된 후보 모델과 best model alias를 불러옵니다."""
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
        "feature_columns": metadata["feature_columns"],
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
    """분류 모델 검토용 threshold 후보표를 저장합니다."""
    metric_dir.mkdir(parents=True, exist_ok=True)
    threshold_df.to_csv(metric_dir / "classification_threshold_candidates.csv", index=False, encoding='utf-8-sig')




def save_prediction_outputs(
    prediction_df: pd.DataFrame,
    original_df: pd.DataFrame,
    prediction_dir: Path,
    regression_target: str,
    classification_target: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """회귀/분류 예측 결과를 간결한 CSV 파일로 저장합니다."""
    prediction_dir.mkdir(parents=True, exist_ok=True)

    id_cols = [col for col in [DATE_COLUMN, SITE_COLUMN] if col in prediction_df.columns]

    regression_df = prediction_df[id_cols].copy()
    if regression_target in original_df.columns:
        regression_df["y_true_target"] = original_df[regression_target].to_numpy()
        regression_df["y_true_cells"] = inverse_transform_cells(original_df[regression_target].to_numpy())
    regression_df["y_pred_target"] = prediction_df["pred_regression_target"].to_numpy()
    regression_df["y_pred_cells"] = prediction_df["predicted_cells"].to_numpy()

    classification_df = prediction_df[id_cols].copy()
    if classification_target in original_df.columns:
        classification_df["y_true_alert"] = original_df[classification_target].to_numpy()
    classification_df["y_pred_alert"] = prediction_df["predicted_alert_label"].to_numpy()
    classification_df["y_pred_probability"] = prediction_df["alert_probability"].to_numpy()

    regression_df.to_csv(prediction_dir / REGRESSION_PREDICTION_FILE, index=False, encoding='utf-8-sig')
    classification_df.to_csv(prediction_dir / CLASSIFICATION_PREDICTION_FILE, index=False, encoding='utf-8-sig')
    return regression_df, classification_df


def apply_operational_alert_logic(
    prediction_df: pd.DataFrame,
    previous_cells_column: str = "previous_observed_cells",  # 수정 필요: 실제 컬럼명에 맞게 수정
    pred_cells_column: str = "predicted_cells",
    probability_column: str = "alert_probability",
    alert_cell_threshold: int = ALERT_CELL_THRESHOLD,
    probability_threshold: float = PROBABILITY_THRESHOLD,
) -> pd.DataFrame:
    # 참고:
    # 이 함수는 조류경보 공식 발령을 자동 결정하는 코드가 아닙니다.
    # 예측 결과를 바탕으로 관리자가 사전 점검할 수 있는
    # '관심 단계 후보'를 표시하는 운영 보조 로직입니다.
    output = prediction_df.copy()

    has_previous_cells = previous_cells_column in output.columns
    has_pred_cells = pred_cells_column in output.columns
    has_probability = probability_column in output.columns

    previous_exceeded = output[previous_cells_column] >= alert_cell_threshold if has_previous_cells else False
    predicted_cells_exceeded = output[pred_cells_column] >= alert_cell_threshold if has_pred_cells else False
    predicted_probability_exceeded = output[probability_column] >= probability_threshold if has_probability else False

    output["alert_candidate"] = previous_exceeded & (predicted_cells_exceeded | predicted_probability_exceeded)
    return output


def get_station_output_dir(base_dir, station_id):
    return base_dir / str(station_id)


def prepare_station_directories(station_id: str):
    """
    지점명(예: '문의(청남대)')을 받아 해당 지점 전용 폴더들을 생성하고 경로들을 반환합니다.
    """
    # 1. 폴더명 정리: 괄호나 공백을 언더바(_)로 바꿔서 안전한 경로 생성
    # 예: '문의(청남대)' -> '문의_청남대'
    folder_name = station_id.replace("(", "_").replace(")", "").replace(" ", "_")
    
    # 2. 각 목적별 경로 정의
    from src import config
    paths = {
        "model": config.OUTPUT_DIR / folder_name,
        "metric": config.METRIC_DIR / folder_name,
        "prediction": config.PREDICTION_DIR / folder_name,
        "explain": config.EXPLAIN_DIR / folder_name,
        "scenario": config.SCENARIO_DIR / folder_name
    }
    
    # 3. 폴더가 없으면 생성 (parents=True로 중간 경로까지 한 번에 생성)
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
        
    return paths