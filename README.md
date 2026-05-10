# 대청댐 유해남조류 조기경보 예측 모델링

## 실행 가이드

이 프로젝트는 대청댐 수질·조류·댐 운영·기상 데이터를 이용해 **다음 조사 시점의 유해남조류 세포수**와 **경보 위험 여부**를 예측한다. 먼저 아래 순서로 실행하면 현재 결과를 재현할 수 있다.

```bash
python -m pip install -r requirements.txt
```

트리 기반 모델과 비트리 스케일 모델을 함께 학습하고 비교한다.

```bash
python - <<'PY'
from src.pipeline.runner import run_workflow_comparison

out = run_workflow_comparison(save=True)
print(out["summary"][["workflow", "selected_for", "model_name", "rmse", "recall", "precision", "f1"]])
PY
```

결과 시각화, 진단, SHAP 비교, 고도화 실험은 각각 아래 명령어로 실행한다.

```bash
python -m src.pipeline.visualization
python -m src.pipeline.diagnostics
python -m src.pipeline.shap_compare
python -m src.pipeline.enhancement
python eda/scripts/01_algae_data_eda.py
```

## 초록

본 프로젝트의 목적은 대청댐에서 관측된 수질, 조류, 수문, 기상 정보를 통합해 다음 조사 시점의 유해남조류 발생 위험을 예측하는 것이다. 회귀 문제에서는 다음 조사 시점의 유해남조류 세포수를 `log10(cells + 1)`로 변환한 `next_log_cells`를 예측하고, 분류 문제에서는 세포수가 1,000 cells/mL 이상인지 여부를 나타내는 `target_alert_next`를 예측한다.

데이터는 하나의 조사일이 여러 조사 지점과 여러 기상 관측소 조합으로 확장된 tabular event 데이터다. 따라서 임의 랜덤 분할은 같은 날짜의 정보가 train과 valid에 동시에 들어갈 위험이 있다. 본 프로젝트는 이를 막기 위해 날짜 기준 holdout을 사용하고, 트리 기반 모델과 스케일링된 비트리 모델을 동일한 검증 조건에서 비교했다.

현재 holdout 검증에서는 비트리 스케일 workflow가 가장 우수했다. 회귀는 `ElasticNet`, 분류는 `Logistic Regression`이 기본 비교에서 가장 좋은 성능을 보였고, 추가 고도화에서는 `HuberRegressor`와 tuned `Logistic Regression`이 가장 좋은 결과를 보였다. 이는 현재 feature set에서 로그 변환, 누적 수문 변수, 현재 조류 상태가 강한 신호를 제공하며, 복잡한 모델보다 안정적인 전처리와 누수 없는 검증 설계가 더 중요하다는 점을 시사한다.

## 연구 문제와 Target 정의

본 연구는 두 가지 예측 문제로 구성된다.

| 구분 | target | 설명 |
| --- | --- | --- |
| 회귀 | `next_log_cells` | 다음 조사 시점 유해남조류 세포수의 `log10(cells + 1)` 값 |
| 분류 | `target_alert_next` | 다음 조사 시점 세포수가 1,000 cells/mL 이상인지 여부 |

유해남조류 세포수는 0이 많고 특정 시점에는 매우 큰 값으로 급증한다. 원 단위 세포수를 그대로 학습하면 극단값이 손실 함수와 모델 계수에 과도한 영향을 줄 수 있다. 그래서 `log10(cells + 1)` 변환을 사용했다. `+1`은 세포수가 0일 때 로그 계산이 불가능한 문제를 해결하고, `log10`은 1,000과 10,000처럼 10배 단위로 해석되는 조류경보 기준과 설명 방식이 잘 맞는다.

분류 target은 아래 기준으로 만들었다.

```text
target_alert_next = next_log_cells >= log10(1000 + 1)
```

즉 다음 조사 시점의 유해남조류 세포수가 1,000 cells/mL 이상이면 1, 아니면 0이다. 이 target은 회귀 target인 `next_log_cells`를 경보 운영 관점으로 변환한 것이다.

## 데이터

원본 및 팀 전처리 데이터는 `src/data/team-raw/`에 보관한다.

| 파일 | 역할 |
| --- | --- |
| `daechung_for_merge_v1.csv` | 수질, 조류, 댐 운영 feature가 포함된 base 데이터 |
| `ALGAE_MODEL_DATA_SCALED.csv` | 기상 관측소별 weather feature 데이터 |
| `ALGAE_DATA.csv` | 수질·조류·댐 운영 데이터와 기상 데이터를 결합한 최종 병합 데이터 |

`ALGAE_DATA.csv`는 본 프로젝트에서 새로 병합한 파일이 아니라, 전처리팀으로부터 이미 병합된 상태로 제공된 데이터다. 구조 검증 결과, `daechung_for_merge_v1.csv`의 각 `date + loc_encoded` 행이 station 4개와 결합되어 `date x loc_encoded x station` 구조를 이룬다. 한 조사일에 보통 3개 조사 지점과 4개 기상 관측소가 결합되므로 최대 12행이 생성된다.

조사 위치와 기상 관측소 인코딩은 다음처럼 해석한다.

| 구분 | 코드 | 의미 |
| --- | ---: | --- |
| 조사 위치 | 0 | 문의 |
| 조사 위치 | 1 | 추동 |
| 조사 위치 | 2 | 하남 |
| 기상 station | 604 | 옥천 |
| 기상 station | 643 | 세천 |
| 기상 station | 648 | 장동 |
| 기상 station | 888 | 청남대 |

## 전처리 방법

모델 입력은 두 가지 workflow로 나뉜다.

| workflow | 입력 파일 | 전처리 원칙 |
| --- | --- | --- |
| `tree` | `src/data/processed/model_input/tree_gradient_boosting/algae_tree_station_expanded.csv` | 트리 기반 모델용. 원 단위 feature를 최대한 유지한다. |
| `non_tree` | `src/data/processed/model_input/non_tree_scaled/algae_non_tree_scaled_station_expanded.csv` | 선형, SVM, KNN, MLP용. 로그 변환, 스케일링, one-hot encoding을 적용한다. |

트리 기반 모델은 feature scale에 비교적 둔감하므로 수질, 조류, 수문 feature의 원 단위를 유지했다. 반면 비트리 모델은 변수 단위 차이에 민감하다. 예를 들어 수온은 수십 단위지만 세포수나 유입량은 수천에서 수십만까지 커질 수 있다. 이런 상태에서 선형 모델이나 거리 기반 모델을 학습하면 큰 단위의 feature가 모델을 지배할 수 있으므로 별도 스케일링 데이터가 필요하다.

비트리 입력의 전처리 원칙은 다음과 같다.

| feature 계열 | 처리 |
| --- | --- |
| 세포수, Chl-a, 탁도 | `log10(x + 1)` 후 `RobustScaler` |
| 강우, 유입, 방류, 체류시간, 정체 지수 | `RobustScaler` |
| 수온, pH, DO, 투명도, 수위, 저수량 | `MinMaxScaler` |
| station, loc_encoded | one-hot encoding |

중요한 점은 scaler를 train 구간에만 fit하고 valid 구간에는 transform만 적용했다는 것이다. valid 데이터까지 사용해 scaler를 fit하면 미래 분포 정보를 전처리 단계에서 미리 본 것이 되어 검증 성능이 과대평가될 수 있다.

## 학습 및 검증 설계

현재 데이터는 센서처럼 매 시점이 촘촘하게 이어지는 전형적인 시계열이라기보다, 조사일 단위로 관측되고 구간별로 값이 크게 튀는 tabular event 데이터에 가깝다. 그럼에도 날짜 구조는 중요하다. 같은 조사일의 행이 여러 station과 loc 조합으로 반복되기 때문이다.

따라서 랜덤 split 대신 날짜 기준 holdout을 사용했다.

| split | 기간 | 행 수 |
| --- | --- | ---: |
| train | 2016-04-04 ~ 2024-04-29 | 5,120 |
| valid | 2024-05-07 ~ 2025-12-15 | 1,184 |

이 방식은 과거 데이터로 학습하고 미래 구간에서 검증하는 out-of-time validation에 가깝다. 실제 운영 환경에서도 미래의 조류 상태를 예측해야 하므로, 무작위 검증보다 이 설정이 더 현실적인 성능 추정에 가깝다.

## 모델 방법론

비교 대상은 크게 tree workflow와 non_tree workflow다.

| workflow | 회귀 후보 | 분류 후보 |
| --- | --- | --- |
| `tree` | LightGBM, XGBoost, HistGradientBoosting, RandomForest, CatBoost | LightGBM, XGBoost, HistGradientBoosting, RandomForest, CatBoost |
| `non_tree` | Ridge, ElasticNet, HuberRegressor, SVR-RBF, KNN Regressor | Logistic Regression, Calibrated Logistic Regression, SVC-RBF, KNN Classifier |

트리 계열 모델을 포함한 이유는 조류 발생이 수온, 강우, 체류시간, 현재 조류 상태, 계절성의 비선형 상호작용으로 나타날 가능성이 크기 때문이다. LightGBM, XGBoost, CatBoost는 gradient boosting 방식으로 오차를 순차적으로 보정하며 복잡한 상호작용을 포착할 수 있고, RandomForest는 여러 tree의 평균 또는 투표로 예측 분산을 줄인다.

비트리 모델을 포함한 이유는 단순한 baseline을 넘어서, 스케일링과 로그 변환이 잘 설계된 경우 더 단순한 모델이 미래 holdout에서 더 안정적일 수 있는지 확인하기 위해서다. 특히 Logistic Regression은 확률 출력과 threshold 조정이 쉬워 조기경보 운영 모델로 설명하기 좋다.

## 기본 모델 결과

기본 후보 전체 비교 결과의 best model은 다음과 같다.

| workflow | task | best model | 주요 성능 |
| --- | --- | --- | --- |
| tree | regression | CatBoost | RMSE 0.7200 / R2 0.8152 |
| tree | classification | RandomForest | Recall 0.9197 / Precision 0.9692 / F1 0.9438 |
| non_tree | regression | ElasticNet | RMSE 0.6773 / R2 0.8364 |
| non_tree | classification | Logistic Regression | Recall 0.9599 / Precision 0.9427 / F1 0.9512 |

현재 holdout 기준에서는 non_tree workflow가 회귀와 분류 모두에서 가장 우수했다. 회귀에서 ElasticNet이 좋은 결과를 낸 것은, 로그 변환과 스케일링으로 극단적인 조류 폭증 구간의 영향이 줄어들고, 상관된 feature가 많은 상황에서 L1/L2 규제가 과적합을 완화했기 때문으로 해석할 수 있다.

분류에서는 Logistic Regression이 가장 높은 Recall을 보였다. 조류경보 문제에서는 실제 위험 상황을 놓치는 미탐이 운영상 큰 문제이므로 Recall이 특히 중요하다. 동시에 Precision도 0.94 이상으로 유지되어, 위험을 지나치게 많이 찍어서 Recall만 높인 모델은 아니라고 볼 수 있다.

전체 후보별 결과는 아래 파일에 저장되어 있다.

```text
artifacts/all_model_results.csv
```

## 모델 고도화 결과

기본 비교 이후에는 각 task의 상위 모델을 중심으로 hyperparameter tuning과 MLP 딥러닝 후보를 추가했다. 교차검증은 같은 날짜가 fold 사이에 섞이지 않도록 `date`를 group으로 사용했다.

| task | 고도화 best | 주요 성능 | 주요 파라미터 |
| --- | --- | --- | --- |
| regression | HuberRegressor | RMSE 0.6691 / R2 0.8403 | `epsilon=2.0`, `alpha=1e-05` |
| classification | Logistic Regression tuned | Recall 0.9818 / Precision 0.9181 / F1 0.9489 | `C=0.03`, `penalty=l2`, `class_weight=balanced` |

고도화 후 회귀는 ElasticNet보다 HuberRegressor가 더 낮은 RMSE를 보였다. HuberRegressor는 일반적인 구간에서는 제곱 오차처럼 학습하지만, 큰 오차 구간에는 덜 민감하게 반응한다. 현재 데이터처럼 폭우, 유입량 급증, 조류 폭증이 함께 나타나는 구간에서는 이런 robust 특성이 유리하게 작용할 수 있다.

분류는 tuned Logistic Regression이 Recall 0.9818을 달성했다. 기존 Logistic Regression보다 실제 위험을 놓치는 false negative가 줄었다. 다만 Precision은 약간 낮아졌으므로, 운영에서는 “미탐을 줄이는 대신 일부 과잉 경보를 감수한다”는 의사결정으로 설명해야 한다.

딥러닝 후보로는 MLPRegressor와 MLPClassifier를 추가했다. MLP는 hidden layer를 통해 feature 간 비선형 조합을 학습할 수 있으나, 현재 데이터는 표본 수가 크지 않고 이미 도메인 기반 파생 feature가 많이 포함된 tabular 데이터다. 결과적으로 MLP는 최종 best가 되지 못했다. 따라서 본 프로젝트에서 딥러닝은 주 모델이 아니라, 복잡한 비선형 학습을 시도했지만 현재 데이터 조건에서는 규제 선형/robust 모델이 더 안정적이었다는 비교 근거로 해석한다.

고도화 결과는 아래 위치에 저장된다.

```text
artifacts/enhancement/enhancement_results.csv
artifacts/enhancement/enhancement_report.md
```

## 비트리 모델 신뢰성 진단

비트리 모델은 현재 예측 성능이 우수하지만, 선형 회귀 계수를 인과 효과처럼 해석하기에는 주의가 필요하다.

| 진단 | 결과 | 해석 |
| --- | ---: | --- |
| Durbin-Watson | 0.4624 | 잔차 자기상관 가능성이 큼 |
| Breusch-Pagan p-value | 2.57e-34 | 등분산성 가정이 약함 |
| Jarque-Bera p-value | 6.55e-22 | 잔차 정규성 가정이 약함 |
| Logistic Brier Score | 0.0323 | 분류 확률 보정은 비교적 양호 |

정리하면 non_tree workflow는 예측 모델로는 유효하지만, 전통적인 선형 회귀 가정에 기반한 인과 해석에는 제한이 있다. ElasticNet과 HuberRegressor는 로그 세포수 예측 baseline 및 robust 예측 모델로 사용하고, Logistic Regression은 높은 Recall을 가진 조기경보 분류 모델로 활용하는 것이 적절하다.

## SHAP 해석

SHAP 분석은 tree classification best model과 non_tree classification best model을 비교하기 위해 수행했다.

| workflow | 주요 SHAP feature |
| --- | --- |
| tree / RandomForest | `cyano_cells`(현재 유해남조류 세포수), `log_target`(현재 세포수 로그값), `alert_encoded`(현재 경보 단계), `water_temp`(수온), `turbidity`(탁도) |
| non_tree / Logistic Regression | `outflow_7d_sum_robust`(최근 7일 누적 방류량), `rain_7d_sum_x_robust`(최근 7일 누적 강우량), `log_target`(현재 세포수 로그값), `level_change_7d_robust`(최근 7일 수위 변화), `alert_encoded`(현재 경보 단계) |

Tree 모델은 현재 조류 상태와 수질 조건을 강하게 보고, non_tree 모델은 수문 변화와 현재 조류 상태를 강하게 본다. 이 차이는 두 workflow가 서로 다른 관점에서 위험을 판단한다는 의미가 있다. 운영 보고에서는 Logistic Regression을 주 모델로 두되, RandomForest/CatBoost의 SHAP 결과를 보조 설명으로 함께 제시하는 방식이 설득력 있다.

```text
artifacts/shap/tree_classification_shap_beeswarm.png
artifacts/shap/non_tree_classification_shap_beeswarm.png
artifacts/shap/classification_shap_importance_comparison.png
```

## EDA 산출물

EDA는 `eda/` 폴더에서 관리한다. 각 그림은 데이터 구조, target 분포, 수질, 조류 종별 특성, 수문 변수, 기상 station 차이, 상관관계를 설명한다.

```text
eda/figures/
eda/tables/
eda/figure_interpretation.md
```

그림 해석 문서는 각 그래프를 서론, 본론, 결론, 요약 형식으로 정리했다.

## 프로젝트 구조

```text
.
├── README.md
├── requirements.txt
├── src/
│   ├── config/
│   │   └── model_config.py
│   ├── data/
│   │   ├── team-raw/
│   │   └── processed/model_input/
│   ├── schema/
│   │   ├── model_input_schema.csv
│   │   └── model_input_schema.md
│   ├── pipeline/
│   │   ├── data.py
│   │   ├── models.py
│   │   ├── evaluation.py
│   │   ├── artifacts.py
│   │   ├── runner.py
│   │   ├── visualization.py
│   │   ├── diagnostics.py
│   │   ├── shap_compare.py
│   │   └── enhancement.py
│   └── utils/
│       └── build_model_datasets.py
├── artifacts/
│   ├── tree_gradient_boosting/
│   ├── non_tree_scaled/
│   ├── enhancement/
│   ├── diagnostics/
│   ├── figures/
│   └── shap/
└── eda/
```

핵심 코드는 `src/pipeline/`에 있다. `runner.py`가 전체 실행 진입점이고, `data.py`, `models.py`, `evaluation.py`, `artifacts.py`가 각각 데이터 로드, 모델 학습, 평가, 저장을 맡는다. `enhancement.py`는 추가 고도화 실험을 담당하며, `visualization.py`, `diagnostics.py`, `shap_compare.py`는 결과 분석 산출물을 만든다.

## 주요 산출물

| 경로 | 내용 |
| --- | --- |
| `src/data/README.md` | 모델 입력 데이터 생성 기준 |
| `artifacts/all_model_results.csv` | 모든 후보 모델의 결과표 |
| `artifacts/workflow_comparison_summary.csv` | workflow별 best 모델 요약 |
| `artifacts/enhancement/enhancement_report.md` | 고도화 실험 결과와 최종 추천 |
| `artifacts/figures/` | 모델 성능 비교 시각화 |
| `artifacts/shap/` | SHAP 해석 결과 |
| `eda/figure_interpretation.md` | EDA 그래프별 해석 |

## 결론

현재 프로젝트의 가장 중요한 결론은 세 가지다. 첫째, `ALGAE_DATA.csv`는 수질·조류·댐 운영 데이터와 기상 station 데이터가 결합된 station-expanded 병합 데이터이며, 모델링에서는 날짜 기준 split이 필수다. 둘째, 현재 holdout 검증에서는 트리 기반 모델보다 스케일링된 비트리 모델이 더 안정적인 성능을 보였다. 셋째, 조기경보 운영 목적에서는 회귀 RMSE보다 분류 Recall이 더 중요한 의사결정 지표이며, tuned Logistic Regression이 현재 가장 적합한 1차 운영 후보로 판단된다.

다만 단일 holdout 결과만으로 최종 운영 모델을 확정하기에는 부족하다. 다음 단계에서는 walk-forward validation으로 여러 미래 구간에서 성능 안정성을 확인하고, threshold tuning으로 미탐과 과잉 경보의 균형을 운영 목적에 맞게 조정해야 한다. 트리 기반 모델은 최종 best가 아니더라도 SHAP 기반 원인 해석과 비선형 비교 모델로 계속 유지하는 것이 좋다.
