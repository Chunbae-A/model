from __future__ import annotations

import ast
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import model_config as config


FIGURE_DIR = config.ARTIFACT_DIR / "figures"


def _save_current_figure(path: Path) -> Path:
    """현재 matplotlib figure를 저장하고 메모리를 정리한다."""

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    return path


def load_comparison_summary() -> pd.DataFrame:
    """workflow 비교 요약 CSV를 읽는다."""

    path = config.ARTIFACT_DIR / "workflow_comparison_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Comparison summary not found: {path}")
    return pd.read_csv(path)


def plot_regression_rmse(summary: pd.DataFrame) -> Path:
    """회귀 best 모델의 RMSE를 workflow별 막대그래프로 저장한다."""

    reg = summary[summary["selected_for"].eq("regression")].copy()
    reg["label"] = reg["workflow"] + "\n" + reg["model_name"]
    plt.figure(figsize=(7.5, 4.5))
    ax = sns.barplot(data=reg, x="label", y="rmse", hue="workflow", dodge=False)
    ax.set_title("Regression RMSE by Workflow")
    ax.set_xlabel("")
    ax.set_ylabel("RMSE on log10(cells + 1)")
    ax.legend_.remove()
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=3)
    return _save_current_figure(FIGURE_DIR / "regression_rmse_by_workflow.png")


def plot_classification_metrics(summary: pd.DataFrame) -> Path:
    """분류 best 모델의 precision/recall/f1을 비교한다."""

    cls = summary[summary["selected_for"].eq("classification")].copy()
    long = cls.melt(
        id_vars=["workflow", "model_name"],
        value_vars=["precision", "recall", "f1"],
        var_name="metric",
        value_name="score",
    )
    long["label"] = long["workflow"] + "\n" + long["model_name"]
    plt.figure(figsize=(8.5, 4.8))
    ax = sns.barplot(data=long, x="label", y="score", hue="metric")
    ax.set_title("Classification Metrics by Workflow")
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.set_ylim(0.80, 1.01)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=2, fontsize=8)
    return _save_current_figure(FIGURE_DIR / "classification_metrics_by_workflow.png")


def plot_confusion_matrices(summary: pd.DataFrame) -> Path:
    """분류 best 모델의 confusion matrix를 시각화한다."""

    cls = summary[summary["selected_for"].eq("classification")].copy()
    fig, axes = plt.subplots(1, len(cls), figsize=(5.2 * len(cls), 4.5))
    if len(cls) == 1:
        axes = [axes]
    for ax, (_, row) in zip(axes, cls.iterrows()):
        matrix = ast.literal_eval(row["confusion_matrix"])
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
        ax.set_title(f"{row['workflow']} / {row['model_name']}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticklabels(["stable", "alert"])
        ax.set_yticklabels(["stable", "alert"], rotation=0)
    return _save_current_figure(FIGURE_DIR / "classification_confusion_matrices.png")


def plot_prediction_scatter() -> Path:
    """회귀 예측값과 실제값의 일치 정도를 산점도로 확인한다."""

    paths = {
        "tree": config.ARTIFACT_DIR
        / "tree_gradient_boosting/predictions/regression_predictions.csv",
        "non_tree": config.ARTIFACT_DIR
        / "non_tree_scaled/predictions/regression_predictions.csv",
    }
    frames = []
    for workflow, path in paths.items():
        df = pd.read_csv(path)
        df["workflow"] = workflow
        frames.append(df)
    pred = pd.concat(frames, ignore_index=True)

    g = sns.FacetGrid(pred, col="workflow", height=4.5, aspect=1)
    g.map_dataframe(
        sns.scatterplot,
        x="actual_log_cells",
        y="predicted_log_cells",
        alpha=0.35,
        s=24,
    )
    for ax in g.axes.flat:
        min_value = min(ax.get_xlim()[0], ax.get_ylim()[0])
        max_value = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([min_value, max_value], [min_value, max_value], color="crimson", linestyle="--", linewidth=1)
        ax.set_xlim(min_value, max_value)
        ax.set_ylim(min_value, max_value)
        ax.set_xlabel("Actual log10(cells + 1)")
        ax.set_ylabel("Predicted log10(cells + 1)")
    g.fig.suptitle("Regression Prediction Scatter", y=1.03)
    return _save_current_figure(FIGURE_DIR / "regression_prediction_scatter.png")


def plot_feature_importance_top10() -> Path:
    """workflow별 상위 10개 feature importance를 비교한다."""

    paths = {
        "tree": config.ARTIFACT_DIR / "tree_gradient_boosting/explain/feature_importance.csv",
        "non_tree": config.ARTIFACT_DIR / "non_tree_scaled/explain/feature_importance.csv",
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, (workflow, path) in zip(axes, paths.items()):
        imp = pd.read_csv(path).head(10).copy()
        sns.barplot(data=imp, y="feature", x="importance", ax=ax, color="#4C78A8")
        ax.set_title(f"Top 10 Feature Importance: {workflow}")
        ax.set_xlabel("Importance")
        ax.set_ylabel("")
    return _save_current_figure(FIGURE_DIR / "feature_importance_top10.png")


def create_all_visualizations() -> list[Path]:
    """모델 비교용 주요 그림을 한 번에 생성한다."""

    summary = load_comparison_summary()
    return [
        plot_regression_rmse(summary),
        plot_classification_metrics(summary),
        plot_confusion_matrices(summary),
        plot_prediction_scatter(),
        plot_feature_importance_top10(),
    ]


if __name__ == "__main__":
    for path in create_all_visualizations():
        print(path)
