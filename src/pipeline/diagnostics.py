from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn.metrics import brier_score_loss
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson, jarque_bera

from src.config import model_config as config
from src.pipeline import data
from src.pipeline.runner import get_workflow


DIAGNOSTIC_DIR = config.ARTIFACT_DIR / "diagnostics"


def _savefig(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    return path


def _load_non_tree_valid_features() -> pd.DataFrame:
    workflow = get_workflow("non_tree")
    df = data.load_model_input(workflow.model_input_path)
    _, valid_df = data.time_based_train_valid_split(df)
    feature_columns = data.get_feature_columns(valid_df)
    return valid_df[feature_columns]


def run_non_tree_diagnostics() -> dict[str, object]:
    regression_path = config.ARTIFACT_DIR / "non_tree_scaled/predictions/regression_predictions.csv"
    classification_path = config.ARTIFACT_DIR / "non_tree_scaled/predictions/classification_predictions.csv"
    if not regression_path.exists() or not classification_path.exists():
        raise FileNotFoundError("Run non_tree workflow before diagnostics.")

    reg = pd.read_csv(regression_path).sort_values(config.ID_COLUMNS).reset_index(drop=True)
    cls = pd.read_csv(classification_path).sort_values(config.ID_COLUMNS).reset_index(drop=True)
    x_valid = _load_non_tree_valid_features().reset_index(drop=True)

    reg["residual"] = reg["actual_log_cells"] - reg["predicted_log_cells"]
    exog = sm.add_constant(x_valid, has_constant="add")
    bp_lm, bp_lm_pvalue, bp_fvalue, bp_f_pvalue = het_breuschpagan(reg["residual"], exog)
    jb_stat, jb_pvalue, skew, kurtosis = jarque_bera(reg["residual"])
    dw_stat = durbin_watson(reg["residual"])

    brier = brier_score_loss(cls["actual_alert"], cls["alert_probability"])

    result = {
        "workflow": "non_tree",
        "regression_model": "elasticnet",
        "classification_model": "logistic_regression",
        "regression_residuals": {
            "mean": float(reg["residual"].mean()),
            "std": float(reg["residual"].std()),
            "durbin_watson": float(dw_stat),
            "breusch_pagan_lm": float(bp_lm),
            "breusch_pagan_lm_pvalue": float(bp_lm_pvalue),
            "breusch_pagan_f": float(bp_fvalue),
            "breusch_pagan_f_pvalue": float(bp_f_pvalue),
            "jarque_bera": float(jb_stat),
            "jarque_bera_pvalue": float(jb_pvalue),
            "skew": float(skew),
            "kurtosis": float(kurtosis),
        },
        "classification_calibration": {
            "brier_score": float(brier),
        },
        "interpretation": {
            "durbin_watson": "2에 가까울수록 잔차 자기상관이 약하다.",
            "breusch_pagan": "p-value가 작으면 등분산성 가정이 약하다는 신호다.",
            "jarque_bera": "p-value가 작으면 잔차 정규성이 약하다는 신호다.",
            "brier_score": "0에 가까울수록 확률 예측이 잘 보정되어 있다.",
        },
    }

    DIAGNOSTIC_DIR.mkdir(parents=True, exist_ok=True)
    (DIAGNOSTIC_DIR / "non_tree_diagnostics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2)
    )

    plt.figure(figsize=(6.5, 4.5))
    sns.scatterplot(data=reg, x="predicted_log_cells", y="residual", alpha=0.35, s=24)
    plt.axhline(0, color="crimson", linestyle="--", linewidth=1)
    plt.title("Non-tree Regression Residuals vs Fitted")
    plt.xlabel("Predicted log10(cells + 1)")
    plt.ylabel("Residual")
    _savefig(DIAGNOSTIC_DIR / "non_tree_residuals_vs_fitted.png")

    fig = sm.qqplot(reg["residual"], line="45", fit=True)
    fig.set_size_inches(6.5, 4.5)
    plt.title("Non-tree Regression Residual Q-Q Plot")
    _savefig(DIAGNOSTIC_DIR / "non_tree_residual_qq.png")

    plt.figure(figsize=(6.5, 4.5))
    bins = pd.qcut(cls["alert_probability"], q=10, duplicates="drop")
    calib = cls.groupby(bins, observed=True).agg(
        mean_probability=("alert_probability", "mean"),
        observed_rate=("actual_alert", "mean"),
        count=("actual_alert", "size"),
    )
    sns.lineplot(data=calib, x="mean_probability", y="observed_rate", marker="o")
    plt.plot([0, 1], [0, 1], color="crimson", linestyle="--", linewidth=1)
    plt.title("Non-tree Logistic Regression Calibration")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed alert rate")
    _savefig(DIAGNOSTIC_DIR / "non_tree_logistic_calibration.png")

    return result


if __name__ == "__main__":
    diagnostics = run_non_tree_diagnostics()
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))
