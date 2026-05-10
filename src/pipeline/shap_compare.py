from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import seaborn as sns

from src.config import model_config as config
from src.pipeline import data
from src.pipeline.runner import get_workflow


SHAP_DIR = config.ARTIFACT_DIR / "shap"


def _classification_shap_values(model, workflow_key: str, x_background: pd.DataFrame, x_sample: pd.DataFrame) -> np.ndarray:
    """workflow에 맞는 SHAP explainer로 분류 모델의 SHAP 값을 계산한다.

    tree 모델은 TreeExplainer, Logistic Regression 같은 선형 모델은
    LinearExplainer가 적합하다. SHAP 라이브러리는 모델/버전에 따라 반환 shape가
    조금씩 다를 수 있어 마지막에 2차원 배열로 정리한다.
    """

    if workflow_key == "tree":
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(x_sample)
    else:
        explainer = shap.LinearExplainer(model, x_background)
        values = explainer.shap_values(x_sample)

    if isinstance(values, list):
        values = values[-1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, -1]
    return values


def _load_workflow_data(workflow_key: str) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """SHAP 계산에 필요한 train background와 valid sample 후보를 로드한다."""

    workflow = get_workflow(workflow_key)
    df = data.load_model_input(workflow.model_input_path)
    train_df, valid_df = data.time_based_train_valid_split(df)
    feature_columns = data.get_feature_columns(train_df)
    return train_df, valid_df, feature_columns


def _load_classification_model(workflow_key: str):
    """저장된 model bundle에서 best classification model만 꺼낸다."""

    workflow = get_workflow(workflow_key)
    bundle_path = config.ARTIFACT_DIR / workflow.artifact_subdir / "models" / config.MODEL_BUNDLE_FILE
    bundle = joblib.load(bundle_path)
    best_name = bundle["best_models"]["classification"]
    model = bundle["trained"]["classification_models"][best_name]
    return best_name, model


def create_shap_comparison(max_background: int = 250, max_sample: int = 500) -> list[Path]:
    """tree/non_tree 분류 모델의 SHAP bar, beeswarm, 비교표를 생성한다.

    background와 sample 크기를 제한하는 이유는 SHAP 계산 비용을 관리하기 위해서다.
    전체 valid를 모두 쓰지 않아도 주요 feature 방향성과 상대 중요도를 설명하는
    데에는 충분한 표본을 확보할 수 있다.
    """

    SHAP_DIR.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    importance_frames = []

    for workflow_key in ["tree", "non_tree"]:
        train_df, valid_df, feature_columns = _load_workflow_data(workflow_key)
        model_name, model = _load_classification_model(workflow_key)

        # LinearExplainer는 background 분포를 사용하므로 train에서 표본을 뽑는다.
        background = train_df[feature_columns].sample(
            min(max_background, len(train_df)),
            random_state=config.RANDOM_STATE,
        )
        # 해석 그림은 valid 구간에서 뽑아 실제 미래 검증 사례 중심으로 만든다.
        sample = valid_df[feature_columns].sample(
            min(max_sample, len(valid_df)),
            random_state=config.RANDOM_STATE,
        )

        values = _classification_shap_values(model, workflow_key, background, sample)
        mean_abs = np.abs(values).mean(axis=0)
        importance = (
            pd.DataFrame(
                {
                    "workflow": workflow_key,
                    "model_name": model_name,
                    "feature": feature_columns,
                    "mean_abs_shap": mean_abs,
                }
            )
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )
        importance.to_csv(SHAP_DIR / f"{workflow_key}_classification_shap_importance.csv", index=False)
        importance_frames.append(importance)

        plt.figure(figsize=(9, 5.5))
        shap.summary_plot(values, sample, plot_type="bar", max_display=15, show=False)
        plt.title(f"SHAP Mean Impact: {workflow_key} / {model_name}")
        path = SHAP_DIR / f"{workflow_key}_classification_shap_bar.png"
        plt.tight_layout()
        plt.savefig(path, dpi=180, bbox_inches="tight")
        plt.close()
        outputs.append(path)

        plt.figure(figsize=(9, 6))
        shap.summary_plot(values, sample, max_display=15, show=False)
        plt.title(f"SHAP Directional Impact: {workflow_key} / {model_name}")
        path = SHAP_DIR / f"{workflow_key}_classification_shap_beeswarm.png"
        plt.tight_layout()
        plt.savefig(path, dpi=180, bbox_inches="tight")
        plt.close()
        outputs.append(path)

    combined = pd.concat(importance_frames, ignore_index=True)
    combined.to_csv(SHAP_DIR / "classification_shap_importance_comparison.csv", index=False)

    top = combined.groupby("workflow").head(12).copy()
    plt.figure(figsize=(10.5, 6))
    sns.barplot(data=top, y="feature", x="mean_abs_shap", hue="workflow")
    plt.title("Classification SHAP Importance Comparison")
    plt.xlabel("Mean absolute SHAP value")
    plt.ylabel("")
    path = SHAP_DIR / "classification_shap_importance_comparison.png"
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    outputs.append(path)

    return outputs


if __name__ == "__main__":
    for output in create_shap_comparison():
        print(output)
