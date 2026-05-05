# 대청댐 조류경보 예측 모델링

대청댐 수질·조류·댐 운영·기상 데이터를 활용해 **다음 조사 시점의 유해남조류 세포수와 경보 위험 여부를 예측**하는 모델링 프로젝트입니다. 전처리팀이 만든 병합 데이터를 기반으로 트리 기반 모델과 비트리 스케일 모델을 모두 학습하고, 성능·해석·시각화 결과를 비교합니다.

## 빠른 실행

패키지 설치:

```bash
python -m pip install -r requirements.txt
```

트리/비트리 모델 전체 비교:

```bash
python - <<'PY'
from src.pipeline.runner import run_workflow_comparison

out = run_workflow_comparison(save=True)
print(out["summary"][["workflow", "selected_for", "model_name", "rmse", "recall", "precision", "f1"]])
PY
```

시각화와 진단 산출물 생성:

```bash
python -m src.pipeline.visualization
python -m src.pipeline.diagnostics
python -m src.pipeline.shap_compare
```

노트북으로 실행:

```bash
jupyter notebook model_training.ipynb
```

## 과제 목표

예측 target은 두 가지입니다.

| 구분 | Target | 의미 |
| --- | --- | --- |
| 회귀 | `next_log_cells` | 다음 조사 시점 유해남조류 세포수의 `log10(cells + 1)` 값 |
| 분류 | `target_alert_next` | 다음 조사 시점 세포수가 1,000 이상인지 여부 |

`log10(cells + 1)`을 쓰는 이유는 유해남조류 세포수가 0이 많고 일부 시점에서 수만~수십만까지 튀는 right-skew 분포이기 때문입니다. 로그 변환은 극단값 영향을 줄이고, 1,000/10,000 같은 경보 기준을 10배 단위로 설명하기 쉽게 만듭니다.

## 데이터와 전처리

전처리팀 원본은 `src/data/team-raw/`에 보관합니다.

| 파일 | 역할 |
| --- | --- |
| `daechung_for_merge_v1.csv` | 수질·조류·댐 운영 중심 base 데이터 |
| `ALGAE_MODEL_DATA_SCALED.csv` | station별 일 단위 기상 feature 데이터 |
| `ALGAE_DATA.csv` | 수질·조류·댐 운영 데이터와 station별 기상 데이터를 병합한 데이터 |

`ALGAE_DATA.csv`는 `date x loc_encoded x station` 구조입니다. 즉 하나의 조사일에 3개 수질 지점과 4개 기상 station이 결합되어 최대 12행이 만들어집니다. 이 구조 때문에 랜덤 split을 쓰면 같은 사건이 train과 valid에 동시에 들어갈 수 있어, 현재는 **date-level chronological holdout**을 사용합니다.

모델 입력은 두 가지로 나눴습니다.

| 입력 데이터 | 목적 | 처리 방식 |
| --- | --- | --- |
| `tree_gradient_boosting/algae_tree_station_expanded.csv` | 트리 기반 모델용 | 원 단위 feature 유지, `split`과 `target_alert_next`만 추가 |
| `non_tree_scaled/algae_non_tree_scaled_station_expanded.csv` | 선형/거리 기반 모델용 | 로그 변환, RobustScaler, MinMaxScaler, one-hot 적용 |

비트리 데이터에서는 조류 세포수·탁도·Chl-a는 `log10(x + 1)` 후 robust scaling하고, 강우/유입/방류/정체 지표는 `RobustScaler`, 수온/pH/DO/수위 등 제한 범위 변수는 `MinMaxScaler`를 적용했습니다. scaler는 train 구간에만 fit하고 valid 구간에는 transform만 적용해 preprocessing leakage를 막았습니다.

## 모델 선정과 학습

두 workflow를 같은 검증 기준으로 비교합니다.

| workflow | 회귀 후보 | 분류 후보 |
| --- | --- | --- |
| `tree` | LightGBM, XGBoost, HistGradientBoosting | LightGBM, XGBoost, HistGradientBoosting |
| `non_tree` | Ridge, ElasticNet, SVR-RBF, KNN Regressor | Logistic Regression, SVC-RBF, KNN Classifier |

LightGBM과 XGBoost도 실제 후보에 포함되어 학습됩니다. 설치되어 있지 않은 환경에서는 sklearn의 HistGradientBoosting으로 fallback될 수 있으므로, 실행 전 `python -m pip install -r requirements.txt`를 권장합니다.

학습 흐름은 `src/pipeline/` 아래에 간단히 나누었습니다.

```text
data.py          데이터 로드, feature 선택, split
models.py        후보 모델 정의와 학습
evaluation.py    metric, threshold, best model 선택
artifacts.py     예측, feature importance, 모델/지표 저장
runner.py        단일 workflow 실행과 트리/비트리 비교
visualization.py 결과 시각화
diagnostics.py   비트리 잔차/가정 진단
shap_compare.py  Tree/Non-tree SHAP 비교
```

## 모델 비교 결과

현재 holdout 검증 결과는 다음과 같습니다.

| workflow | task | best model | 주요 성능 |
| --- | --- | --- | --- |
| tree | regression | LightGBM | RMSE 0.7339 |
| tree | classification | XGBoost | Recall 0.8960 / Precision 0.9781 / F1 0.9352 |
| non_tree | regression | ElasticNet | RMSE 0.6773 / R2 0.8364 |
| non_tree | classification | Logistic Regression | Recall 0.9599 / Precision 0.9427 / F1 0.9512 |

현재 기준에서는 비트리 스케일 workflow가 회귀와 분류 모두에서 더 좋습니다. 특히 조류경보 운영에서는 미탐을 줄이는 Recall이 중요하므로 `non_tree + Logistic Regression`이 1차 추천 분류 모델입니다. 반면 `tree + XGBoost`는 Precision이 높아 오경보를 줄이는 비교군으로 의미가 있습니다.

![Classification Metrics](artifacts/figures/classification_metrics_by_workflow.png)

![Confusion Matrices](artifacts/figures/classification_confusion_matrices.png)

## 비트리 모델 신뢰성 진단

비트리 workflow는 예측 성능은 좋지만, 선형 회귀 계열을 통계적 추론 모델처럼 해석하려면 잔차 가정을 확인해야 합니다.

| 진단 | 결과 | 해석 |
| --- | ---: | --- |
| Durbin-Watson | 0.4624 | 2에서 멀어 잔차 자기상관 가능성이 큼 |
| Breusch-Pagan p-value | 2.57e-34 | 등분산성 가정이 약함 |
| Jarque-Bera p-value | 6.55e-22 | 잔차 정규성 가정이 약함 |
| Logistic Brier Score | 0.0323 | 분류 확률 보정은 비교적 양호 |

따라서 `ElasticNet`은 예측 모델로는 유효하지만, 계수 p-value나 인과적 해석에는 적합하지 않습니다. 이 모델은 “설명 가능한 예측 baseline”으로 보는 것이 안전합니다. Logistic Regression 분류 모델은 Recall/F1과 calibration이 좋아 조기경보 후보로 유지할 수 있습니다.

![Non-tree Residuals](artifacts/diagnostics/non_tree_residuals_vs_fitted.png)

## SHAP 기반 원인 해석

Tree workflow의 best classification model인 XGBoost와 Non-tree workflow의 best classification model인 Logistic Regression에 대해 SHAP 비교를 수행했습니다.

| workflow | 주요 SHAP feature |
| --- | --- |
| tree / XGBoost | `cyano_cells`, `sin_season`, `turbidity`, `acc_temp_7d`, `water_temp` |
| non_tree / Logistic Regression | `outflow_7d_sum_robust`, `rain_7d_sum_x_robust`, `log_target`, `level_change_7d_robust`, `alert_encoded` |

Tree 모델은 현재 조류량·수질·계절 조건을 강하게 보고, Non-tree 모델은 수문 변화량·누적 강우/방류·현재 조류 상태를 강하게 봅니다. 두 모델이 서로 다른 관점의 신호를 사용하므로, 운영 보고에서는 두 해석을 함께 제시하는 것이 좋습니다.

![SHAP Comparison](artifacts/shap/classification_shap_importance_comparison.png)

## 산출물

```text
artifacts/workflow_comparison_summary.csv
artifacts/figures/
artifacts/diagnostics/
artifacts/shap/

artifacts/tree_gradient_boosting/
  models/
  metrics/
  predictions/
  explain/

artifacts/non_tree_scaled/
  models/
  metrics/
  predictions/
  explain/
```

자세한 문서:

- [데이터/컬럼 사전](docs/data_dictionary.md)
- [파이프라인 도식 HTML](docs/modeling_pipeline_diagram.html)
- [파이프라인 도식 Mermaid](docs/modeling_pipeline_diagram.md)
- [워크플로 상세 분석](docs/modeling_workflow_analysis.md)
- [모델 결과 최종 정리](docs/model_results_summary.md)

## 다음 과제

현재 결과는 단일 chronological holdout 기준입니다. 최종 보고 전에는 `walk-forward validation` 또는 연도별 out-of-time validation으로 안정성을 확인해야 합니다. 또한 `log_target` 포함/제외 ablation, station-expanded 구조와 대표 station 집계 방식 비교, 운영 목표에 맞춘 threshold 조정이 필요합니다.

