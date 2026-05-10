from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import ElasticNet, HuberRegressor, LogisticRegression, Ridge
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold, RandomizedSearchCV, StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier, MLPRegressor

from src.config import model_config as config
from src.pipeline import data, evaluation


OUTPUT_DIR = config.ARTIFACT_DIR / "enhancement"


@dataclass(frozen=True)
class ExperimentSpec:
    """고도화 실험 하나를 표현하는 명세.

    같은 검색 로직으로 여러 모델을 돌리기 위해 모델, 탐색 파라미터,
    workflow, task, 실험 이유를 하나의 객체로 묶는다. `reason`은 결과
    문서에 그대로 들어가므로, 모델을 왜 추가했는지 설명하는 근거 역할도 한다.
    """

    experiment_name: str
    workflow_key: str
    task: str
    model: Any
    param_distributions: dict[str, list[Any]]
    n_iter: int
    reason: str


def _tree_input() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """트리 기반 workflow 입력을 로드하고 train/valid/feature를 반환한다."""

    workflow = config.WORKFLOWS["tree"]
    df = data.load_model_input(workflow.model_input_path)
    train_df, valid_df = data.time_based_train_valid_split(df)
    return train_df, valid_df, data.get_feature_columns(train_df)


def _non_tree_input() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """스케일링된 비트리 workflow 입력을 로드하고 train/valid/feature를 반환한다."""

    workflow = config.WORKFLOWS["non_tree"]
    df = data.load_model_input(workflow.model_input_path)
    train_df, valid_df = data.time_based_train_valid_split(df)
    return train_df, valid_df, data.get_feature_columns(train_df)


def _build_specs(random_state: int = config.RANDOM_STATE) -> list[ExperimentSpec]:
    """고도화 대상 모델과 탐색할 hyperparameter 범위를 정의한다.

    현재 데이터는 연속적인 센서 시계열이라기보다 조사일 단위로 튀는 tabular
    이벤트 데이터에 가깝다. 그래서 LSTM 같은 순차 딥러닝보다, tabular에서
    강한 robust/linear/tree/MLP 후보를 비교하는 방식으로 구성했다.
    """

    specs: list[ExperimentSpec] = [
        ExperimentSpec(
            experiment_name="elasticnet_tuned",
            workflow_key="non_tree",
            task="regression",
            model=ElasticNet(max_iter=20000, random_state=random_state),
            param_distributions={
                "alpha": [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03],
                "l1_ratio": [0.05, 0.1, 0.2, 0.4, 0.7, 0.9],
            },
            n_iter=18,
            reason="상관된 수질/수문 변수들이 많아 L1+L2 규제로 과적합과 변수 중복 영향을 줄인다.",
        ),
        ExperimentSpec(
            experiment_name="ridge_tuned",
            workflow_key="non_tree",
            task="regression",
            model=Ridge(),
            param_distributions={
                "alpha": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0],
            },
            n_iter=9,
            reason="다중공선성이 있는 tabular feature에서 L2 규제로 안정적인 선형 기준선을 만든다.",
        ),
        ExperimentSpec(
            experiment_name="huber_tuned",
            workflow_key="non_tree",
            task="regression",
            model=HuberRegressor(max_iter=5000),
            param_distributions={
                "epsilon": [1.1, 1.2, 1.35, 1.5, 1.75, 2.0],
                "alpha": [0.00001, 0.0001, 0.001, 0.01],
            },
            n_iter=18,
            reason="조류 폭증과 폭우 같은 튀는 구간을 고려해 큰 오차에 덜 민감한 robust 회귀를 확인한다.",
        ),
        ExperimentSpec(
            experiment_name="catboost_regressor_tuned",
            workflow_key="tree",
            task="regression",
            model=_catboost_regressor(random_state),
            param_distributions={
                "depth": [3, 4, 5, 6],
                "learning_rate": [0.02, 0.03, 0.05, 0.08],
                "iterations": [300, 500, 700],
                "l2_leaf_reg": [1, 3, 5, 7, 10],
            },
            n_iter=20,
            reason="tree 회귀 best였으므로 ordered boosting 계열의 비선형 패턴 포착력을 추가 튜닝한다.",
        ),
        ExperimentSpec(
            experiment_name="mlp_regressor_deep",
            workflow_key="non_tree",
            task="regression",
            model=MLPRegressor(
                max_iter=1500,
                early_stopping=True,
                validation_fraction=0.15,
                random_state=random_state,
            ),
            param_distributions={
                "hidden_layer_sizes": [(64, 32), (128, 64), (128, 64, 32), (256, 128, 64)],
                "alpha": [0.0001, 0.001, 0.01],
                "learning_rate_init": [0.0005, 0.001, 0.003],
                "activation": ["relu", "tanh"],
            },
            n_iter=18,
            reason="딥러닝 기반 tabular MLP로 선형 모델이 놓칠 수 있는 부드러운 비선형 조합을 학습한다.",
        ),
        ExperimentSpec(
            experiment_name="logistic_regression_tuned",
            workflow_key="non_tree",
            task="classification",
            model=LogisticRegression(max_iter=20000, solver="liblinear", random_state=random_state),
            param_distributions={
                "C": [0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
                "penalty": ["l1", "l2"],
                "class_weight": ["balanced", None],
            },
            n_iter=18,
            reason="현재 분류 best이므로 규제 강도와 class balancing을 조정해 Recall 중심 성능을 끌어올린다.",
        ),
        ExperimentSpec(
            experiment_name="random_forest_classifier_tuned",
            workflow_key="tree",
            task="classification",
            model=RandomForestClassifier(n_jobs=-1, random_state=random_state),
            param_distributions={
                "n_estimators": [300, 500, 800],
                "max_depth": [None, 6, 10, 14],
                "min_samples_leaf": [1, 2, 3, 5],
                "max_features": ["sqrt", "log2", 0.5],
                "class_weight": ["balanced", "balanced_subsample"],
            },
            n_iter=24,
            reason="tree 분류 best였고, 여러 tree 투표로 튀는 구간에 덜 민감한 안정적 분류를 만든다.",
        ),
        ExperimentSpec(
            experiment_name="catboost_classifier_tuned",
            workflow_key="tree",
            task="classification",
            model=_catboost_classifier(random_state),
            param_distributions={
                "depth": [3, 4, 5, 6],
                "learning_rate": [0.02, 0.03, 0.05, 0.08],
                "iterations": [300, 500, 700],
                "l2_leaf_reg": [1, 3, 5, 7, 10],
            },
            n_iter=20,
            reason="boosting 계열에서 Recall과 Precision 균형이 좋아 ordered boosting 후보를 더 탐색한다.",
        ),
        ExperimentSpec(
            experiment_name="calibrated_logistic_tuned",
            workflow_key="non_tree",
            task="classification",
            model=CalibratedClassifierCV(
                estimator=LogisticRegression(
                    max_iter=20000,
                    solver="liblinear",
                    class_weight="balanced",
                    random_state=random_state,
                ),
                method="sigmoid",
                cv=3,
            ),
            param_distributions={
                "estimator__C": [0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
                "estimator__penalty": ["l1", "l2"],
                "method": ["sigmoid", "isotonic"],
            },
            n_iter=18,
            reason="확률 보정 모델이 보수적으로 변하는지 확인하고, 운영 threshold 조정 가능성을 확인한다.",
        ),
        ExperimentSpec(
            experiment_name="mlp_classifier_deep",
            workflow_key="non_tree",
            task="classification",
            model=MLPClassifier(
                max_iter=1500,
                early_stopping=True,
                validation_fraction=0.15,
                random_state=random_state,
            ),
            param_distributions={
                "hidden_layer_sizes": [(64, 32), (128, 64), (128, 64, 32), (256, 128, 64)],
                "alpha": [0.0001, 0.001, 0.01],
                "learning_rate_init": [0.0005, 0.001, 0.003],
                "activation": ["relu", "tanh"],
            },
            n_iter=18,
            reason="딥러닝 MLP로 수문·수질 feature 간 비선형 조합을 직접 학습해 성능 상승 가능성을 검토한다.",
        ),
    ]
    return specs


def _catboost_regressor(random_state: int) -> Any:
    """CatBoost 회귀 모델을 만든다.

    `allow_writing_files=False`는 CatBoost가 기본적으로 만드는 `catboost_info`
    폴더를 막기 위한 설정이다. 프로젝트 루트가 실험 부산물로 지저분해지는 것을
    방지한다.
    """

    try:
        from catboost import CatBoostRegressor
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("catboost is required for CatBoost enhancement experiments.") from exc
    return CatBoostRegressor(
        loss_function="RMSE",
        random_seed=random_state,
        verbose=False,
        allow_writing_files=False,
    )


def _catboost_classifier(random_state: int) -> Any:
    """CatBoost 분류 모델을 만든다."""

    try:
        from catboost import CatBoostClassifier
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("catboost is required for CatBoost enhancement experiments.") from exc
    return CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="Recall",
        auto_class_weights="Balanced",
        random_seed=random_state,
        verbose=False,
        allow_writing_files=False,
    )


def _input_for_workflow(workflow_key: str) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """실험 명세의 workflow key에 맞는 입력 데이터 묶음을 반환한다."""

    if workflow_key == "tree":
        return _tree_input()
    if workflow_key == "non_tree":
        return _non_tree_input()
    raise KeyError(f"Unsupported workflow: {workflow_key}")


def _cv_for_task(task: str, y: pd.Series) -> Any:
    """task별 교차검증 전략을 선택한다.

    같은 조사일의 여러 행이 fold 사이에 나뉘면 정보 누수가 생길 수 있으므로
    `date`를 group으로 사용한다. 분류는 class 비율도 최대한 유지하기 위해
    StratifiedGroupKFold를 사용한다.
    """

    if task == "classification":
        class_counts = y.value_counts()
        min_count = int(class_counts.min())
        n_splits = max(2, min(4, min_count))
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_STATE)
    return GroupKFold(n_splits=4)


def _search(spec: ExperimentSpec, train_df: pd.DataFrame, feature_columns: list[str]) -> RandomizedSearchCV:
    """train 내부 교차검증으로 hyperparameter search를 수행한다."""

    x_train = train_df[feature_columns]
    y_train = train_df[config.REGRESSION_TARGET if spec.task == "regression" else config.CLASSIFICATION_TARGET]
    groups = train_df[config.DATE_COLUMN].astype(str)
    scoring = "neg_root_mean_squared_error" if spec.task == "regression" else "recall"
    cv = _cv_for_task(spec.task, y_train)
    n_iter = min(spec.n_iter, _param_space_size(spec.param_distributions))
    search = RandomizedSearchCV(
        estimator=spec.model,
        param_distributions=spec.param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        refit=True,
        random_state=config.RANDOM_STATE,
        error_score="raise",
    )
    search.fit(x_train, y_train, groups=groups)
    return search


def _param_space_size(param_distributions: dict[str, list[Any]]) -> int:
    """탐색 공간의 전체 조합 수를 계산해 n_iter가 조합 수를 넘지 않게 한다."""

    size = 1
    for values in param_distributions.values():
        size *= len(values)
    return size


def _classification_threshold_metrics(y_true: pd.Series, prob: np.ndarray) -> tuple[float, dict[str, float]]:
    """운영용 threshold 후보 중 recall/f1이 좋은 지점을 찾는다."""

    best_threshold = config.PROBABILITY_THRESHOLD
    best_metrics: dict[str, float] = {}
    best_key = (-1.0, -1.0)
    for threshold in config.THRESHOLD_CANDIDATES:
        pred = (prob >= threshold).astype(int)
        precision = float(precision_score(y_true, pred, zero_division=0))
        recall = float(recall_score(y_true, pred, zero_division=0))
        f1 = float(f1_score(y_true, pred, zero_division=0))
        if precision < config.MIN_PRECISION_FOR_THRESHOLD:
            continue
        key = (recall, f1)
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
            best_metrics = {
                "tuned_threshold": float(threshold),
                "tuned_threshold_precision": precision,
                "tuned_threshold_recall": recall,
                "tuned_threshold_f1": f1,
            }
    if not best_metrics:
        best_metrics = {
            "tuned_threshold": float(config.PROBABILITY_THRESHOLD),
            "tuned_threshold_precision": float("nan"),
            "tuned_threshold_recall": float("nan"),
            "tuned_threshold_f1": float("nan"),
        }
    return best_threshold, best_metrics


def _evaluate(spec: ExperimentSpec, model: Any, valid_df: pd.DataFrame, feature_columns: list[str]) -> dict[str, Any]:
    """검색으로 선택된 best estimator를 고정 valid 구간에서 최종 평가한다."""

    x_valid = valid_df[feature_columns]
    if spec.task == "regression":
        y_true = valid_df[config.REGRESSION_TARGET]
        pred = model.predict(x_valid)
        metrics = evaluation.regression_metrics(y_true, pred)
        return {"task": spec.task, **metrics}

    y_true = valid_df[config.CLASSIFICATION_TARGET]
    prob = evaluation.classification_probability(model, x_valid)
    metrics = evaluation.classification_metrics(y_true, prob, threshold=config.PROBABILITY_THRESHOLD)
    _, threshold_metrics = _classification_threshold_metrics(y_true, prob)
    compact = {k: v for k, v in metrics.items() if k not in {"classification_report"}}
    return {"task": spec.task, **compact, **threshold_metrics}


def run_enhancement_experiments(save: bool = True) -> pd.DataFrame:
    """모델 고도화 실험 전체를 실행하고 결과표/모델/보고서를 저장한다."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    specs = _build_specs()
    data_cache: dict[str, tuple[pd.DataFrame, pd.DataFrame, list[str]]] = {}
    rows: list[dict[str, Any]] = []
    fitted: dict[str, Any] = {}

    for spec in specs:
        if spec.workflow_key not in data_cache:
            data_cache[spec.workflow_key] = _input_for_workflow(spec.workflow_key)
        train_df, valid_df, feature_columns = data_cache[spec.workflow_key]
        search = _search(spec, train_df, feature_columns)
        metrics = _evaluate(spec, search.best_estimator_, valid_df, feature_columns)
        row = {
            "experiment_name": spec.experiment_name,
            "workflow": spec.workflow_key,
            "task": spec.task,
            "base_model": type(search.best_estimator_).__name__,
            "cv_best_score": float(search.best_score_),
            "best_params": json.dumps(search.best_params_, ensure_ascii=False, sort_keys=True),
            "reason": spec.reason,
            **metrics,
        }
        rows.append(row)
        fitted[spec.experiment_name] = search.best_estimator_

    results = pd.DataFrame(rows)
    results = _rank_results(results)

    if save:
        results.to_csv(OUTPUT_DIR / "enhancement_results.csv", index=False)
        joblib.dump(fitted, OUTPUT_DIR / "enhancement_models.pkl")
        _write_markdown_report(results, OUTPUT_DIR / "enhancement_report.md")

    return results


def _rank_results(results: pd.DataFrame) -> pd.DataFrame:
    """task별 순위를 부여한다. 회귀는 RMSE 낮은 순, 분류는 Recall 높은 순이다."""

    ranked = results.copy()
    ranked["rank"] = np.nan
    for task, idx in ranked.groupby("task").groups.items():
        sub = ranked.loc[idx].copy()
        if task == "regression":
            ranks = sub["rmse"].rank(method="min", ascending=True)
        else:
            ranks = sub["recall"].rank(method="min", ascending=False)
        ranked.loc[idx, "rank"] = ranks.astype(int)
    return ranked.sort_values(["task", "rank", "experiment_name"]).reset_index(drop=True)


def _write_markdown_report(results: pd.DataFrame, path: Path) -> None:
    """고도화 결과를 사람이 바로 읽을 수 있는 Markdown 보고서로 저장한다."""

    lines: list[str] = []
    lines.append("# 모델 고도화 실험 결과")
    lines.append("")
    lines.append("현재 데이터는 날짜 순서가 완만하게 이어지는 전형적인 시계열이라기보다, 조사 구간마다 조류와 수문 조건이 크게 튀는 tabular 이벤트 데이터에 가깝다. 그래서 고도화는 시간순 rolling 예측이 아니라, 같은 조사일이 train/CV fold에 동시에 섞이지 않도록 `date`를 group으로 묶은 교차검증 기반 튜닝으로 진행했다.")
    lines.append("")
    lines.append("## 고도화 방식")
    lines.append("")
    lines.append("- 회귀는 train 내부 `GroupKFold(date)`에서 RMSE가 낮은 파라미터를 선택했다.")
    lines.append("- 분류는 train 내부 `StratifiedGroupKFold(date)`에서 Recall이 높은 파라미터를 선택했다.")
    lines.append("- 최종 성능은 기존 valid split에서 다시 평가했다.")
    lines.append("- 딥러닝은 tabular 데이터에 맞는 MLP를 사용했다.")
    lines.append("")

    for task in ["regression", "classification"]:
        sub = results[results["task"].eq(task)].copy()
        lines.append(f"## {task} 결과")
        lines.append("")
        if task == "regression":
            cols = ["rank", "experiment_name", "workflow", "rmse", "r2", "mae", "mae_cells", "rmse_cells", "best_params"]
        else:
            cols = [
                "rank",
                "experiment_name",
                "workflow",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "pr_auc",
                "tuned_threshold",
                "tuned_threshold_recall",
                "tuned_threshold_precision",
                "best_params",
            ]
        table = sub[cols].copy()
        for col in table.select_dtypes(include="number").columns:
            table[col] = table[col].round(4)
        lines.append(table.to_markdown(index=False))
        lines.append("")

    best_reg = results[results["task"].eq("regression")].sort_values("rmse").iloc[0]
    best_cls = results[results["task"].eq("classification")].sort_values("recall", ascending=False).iloc[0]
    lines.append("## 최종 추천")
    lines.append("")
    lines.append(f"- 회귀 고도화 best: `{best_reg['experiment_name']}` / RMSE `{best_reg['rmse']:.4f}` / R2 `{best_reg['r2']:.4f}`")
    lines.append(f"- 분류 고도화 best: `{best_cls['experiment_name']}` / Recall `{best_cls['recall']:.4f}` / Precision `{best_cls['precision']:.4f}` / F1 `{best_cls['f1']:.4f}`")
    lines.append("")
    lines.append("해석상 가장 중요한 모델은 분류의 `logistic_regression_tuned`다. 조류경보 문제는 실제 위험을 놓치지 않는 것이 중요하기 때문에 Recall을 우선했고, Logistic Regression은 강한 현재 조류 상태와 수문 신호를 단순한 확률 모델로 안정적으로 사용한다. 이는 binary outcome을 logit link로 모델링하는 고전적 로지스틱 회귀의 장점과 맞고, 현재처럼 스케일링된 tabular feature에서는 복잡한 모델보다 과적합 위험이 작다.")
    lines.append("")
    lines.append("다만 딥러닝 MLP도 비교에 포함했다. MLP는 여러 hidden layer를 통해 비선형 feature 조합을 학습할 수 있지만, 현재 표본 수와 feature 구조에서는 tree/linear 계열보다 반드시 우세하다고 보기 어렵다. 따라서 딥러닝은 최종 주 모델이라기보다 성능 상승 가능성을 확인하는 추가 후보로 해석한다.")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    df = run_enhancement_experiments(save=True)
    cols = [
        "task",
        "rank",
        "experiment_name",
        "workflow",
        "rmse",
        "r2",
        "recall",
        "precision",
        "f1",
        "tuned_threshold",
        "tuned_threshold_recall",
    ]
    print(df[[c for c in cols if c in df.columns]].to_string(index=False))
