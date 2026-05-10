from __future__ import annotations

from typing import Any

import pandas as pd

from src.config import model_config as config
from src.pipeline import artifacts, data, evaluation, models


def get_workflow(workflow_key: str = config.DEFAULT_WORKFLOW) -> config.WorkflowConfig:
    """문자열 key로 workflow 설정을 가져온다."""

    if workflow_key not in config.WORKFLOWS:
        raise KeyError(f"Unknown workflow '{workflow_key}'. Available: {sorted(config.WORKFLOWS)}")
    return config.WORKFLOWS[workflow_key]


def run_training_pipeline(
    workflow_key: str = config.DEFAULT_WORKFLOW,
    save: bool = True,
) -> dict[str, Any]:
    """단일 workflow의 end-to-end 학습 파이프라인을 실행한다.

    흐름은 데이터 로드 -> 시간 기준 split -> feature 선택 -> 후보 모델 학습 ->
    전체 후보 평가 -> best model 선택 -> 예측/해석 산출물 저장 순서다. 이 함수가
    프로젝트의 공식 실행 진입점이다.
    """

    workflow = get_workflow(workflow_key)
    df = data.load_model_input(workflow.model_input_path)
    train_df, valid_df = data.time_based_train_valid_split(df)
    feature_columns = data.get_feature_columns(train_df)
    trained = models.train_candidate_models(train_df, feature_columns, workflow)
    metrics_df, threshold_df = evaluation.evaluate_candidate_models(trained, valid_df, workflow.key)
    best_models = evaluation.select_best_models(metrics_df)
    regression_predictions, classification_predictions = artifacts.make_predictions(trained, valid_df, best_models)
    feature_importance = artifacts.get_feature_importance(trained, valid_df, best_models)

    artifact_dirs = None
    if save:
        artifact_dirs = artifacts.save_artifacts(
            workflow,
            trained,
            metrics_df,
            threshold_df,
            best_models,
            regression_predictions,
            classification_predictions,
            feature_importance,
        )

    return {
        "workflow": workflow,
        "artifact_dirs": artifact_dirs,
        "df": df,
        "train_df": train_df,
        "valid_df": valid_df,
        "feature_columns": feature_columns,
        "trained": trained,
        "metrics_df": metrics_df,
        "threshold_df": threshold_df,
        "best_models": best_models,
        "regression_predictions": regression_predictions,
        "classification_predictions": classification_predictions,
        "feature_importance": feature_importance,
    }


def build_workflow_summary(result: dict[str, Any]) -> pd.DataFrame:
    """한 workflow에서 선택된 best 회귀/분류 모델만 요약한다."""

    metrics = result["metrics_df"].copy()
    best = result["best_models"]
    rows = []
    for task, model_name in best.items():
        task_name = "regression" if task == "regression" else "classification"
        row = metrics[(metrics["task"].eq(task_name)) & (metrics["model_name"].eq(model_name))].iloc[0].to_dict()
        row["selected_for"] = task
        rows.append(row)
    return pd.DataFrame(rows)


def run_workflow_comparison(
    workflow_keys: list[str] | None = None,
    save: bool = True,
) -> dict[str, Any]:
    """tree와 non_tree workflow를 같은 조건에서 비교 실행한다."""

    keys = workflow_keys or list(config.WORKFLOWS)
    results = {key: run_training_pipeline(key, save=save) for key in keys}
    summary = pd.concat([build_workflow_summary(result) for result in results.values()], ignore_index=True)

    if save:
        config.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        summary.to_csv(config.ARTIFACT_DIR / "workflow_comparison_summary.csv", index=False)

    return {"results": results, "summary": summary}
