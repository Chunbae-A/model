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
| `non_tree` | Ridge, ElasticNet, HuberRegressor, SVR-RBF, KNN Regressor | Logistic Regression, SGDClassifier, Calibrated Logistic Regression, SVC-RBF, KNN Classifier |

트리 계열 모델을 포함한 이유는 조류 발생이 수온, 강우, 체류시간, 현재 조류 상태, 계절성의 비선형 상호작용으로 나타날 가능성이 크기 때문이다. LightGBM, XGBoost, CatBoost는 gradient boosting 방식으로 오차를 순차적으로 보정하며 복잡한 상호작용을 포착할 수 있고, RandomForest는 여러 tree의 평균 또는 투표로 예측 분산을 줄인다.

비트리 모델을 포함한 이유는 단순한 baseline을 넘어서, 스케일링과 로그 변환이 잘 설계된 경우 더 단순한 모델이 미래 holdout에서 더 안정적일 수 있는지 확인하기 위해서다. 특히 Logistic Regression은 확률 출력과 threshold 조정이 쉬워 조기경보 운영 모델로 설명하기 좋다. SGDClassifier는 Logistic Regression과 같은 log-loss 기반 선형 분류기를 확률적 경사하강법으로 학습하는 후보로, 향후 데이터가 커지거나 온라인 업데이트 구조로 확장할 때의 비교 기준으로 추가했다.

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

기본 회귀 후보 전체 결과는 다음과 같다. `best=Y`는 각 workflow 내부에서 선택된 모델을 의미한다.

| workflow | model | rank | RMSE | MAE | R2 | MAE_cells | RMSE_cells | best |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| non_tree | ElasticNet | 1 | 0.6773 | 0.4932 | 0.8364 | 4,764.3 | 13,977.2 | Y |
| non_tree | Ridge | 2 | 0.6786 | 0.4967 | 0.8358 | 4,974.0 | 14,442.5 |  |
| non_tree | HuberRegressor | 3 | 0.7153 | 0.4446 | 0.8176 | 4,204.5 | 13,200.8 |  |
| non_tree | SVR-RBF | 4 | 1.1172 | 1.0193 | 0.5550 | 7,626.5 | 17,347.6 |  |
| non_tree | KNN Regressor | 5 | 1.1790 | 0.9188 | 0.5044 | 7,252.7 | 16,978.3 |  |
| tree | CatBoost | 1 | 0.7200 | 0.5471 | 0.8152 | 5,272.2 | 14,413.6 | Y |
| tree | RandomForest | 2 | 0.7318 | 0.5683 | 0.8091 | 5,777.9 | 14,834.6 |  |
| tree | LightGBM | 3 | 0.7339 | 0.5510 | 0.8080 | 4,976.0 | 14,045.1 |  |
| tree | XGBoost | 4 | 0.7385 | 0.5525 | 0.8056 | 5,096.8 | 14,158.1 |  |
| tree | HistGradientBoosting | 5 | 0.7437 | 0.5534 | 0.8028 | 4,925.8 | 13,892.0 |  |

기본 분류 후보 전체 결과는 다음과 같다. 조류경보 목적에서는 실제 위험을 놓치지 않는 것이 중요하므로 rank는 Recall 기준이다.

| workflow | model | rank | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | TN | FP | FN | TP | best |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| non_tree | Logistic Regression | 1 | 0.9544 | 0.9427 | 0.9599 | 0.9512 | 0.9924 | 0.9911 | 604 | 32 | 22 | 526 | Y |
| non_tree | SGDClassifier | 2 | 0.9341 | 0.9304 | 0.9270 | 0.9287 | 0.9433 | 0.9068 | 598 | 38 | 40 | 508 |  |
| non_tree | Calibrated Logistic Regression | 3 | 0.9316 | 0.9587 | 0.8905 | 0.9234 | 0.9910 | 0.9892 | 615 | 21 | 60 | 488 |  |
| non_tree | KNN Classifier | 4 | 0.7965 | 0.9528 | 0.5894 | 0.7283 | 0.9222 | 0.9140 | 620 | 16 | 225 | 323 |  |
| non_tree | SVC-RBF | 5 | 0.7458 | 0.8847 | 0.5182 | 0.6536 | 0.9258 | 0.8300 | 599 | 37 | 264 | 284 |  |
| tree | RandomForest | 1 | 0.9493 | 0.9692 | 0.9197 | 0.9438 | 0.9941 | 0.9931 | 620 | 16 | 44 | 504 | Y |
| tree | CatBoost | 2 | 0.9392 | 0.9542 | 0.9124 | 0.9328 | 0.9919 | 0.9904 | 612 | 24 | 48 | 500 |  |
| tree | XGBoost | 3 | 0.9426 | 0.9781 | 0.8960 | 0.9352 | 0.9941 | 0.9933 | 625 | 11 | 57 | 491 |  |
| tree | HistGradientBoosting | 4 | 0.9324 | 0.9756 | 0.8759 | 0.9231 | 0.9923 | 0.9910 | 624 | 12 | 68 | 480 |  |
| tree | LightGBM | 5 | 0.9307 | 0.9755 | 0.8723 | 0.9210 | 0.9938 | 0.9929 | 624 | 12 | 70 | 478 |  |

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

고도화 후 회귀는 ElasticNet보다 HuberRegressor가 더 낮은 RMSE를 보였다. HuberRegressor는 일반적인 구간에서는 제곱 오차처럼 학습하지만, 큰 오차 구간에는 덜 민감하게 반응한다. 현재 데이터처럼 폭우, 유입량 급증, 조류 폭증이 함께 나타나는 구간에서는 이런 robust 특성이 유리하게 작용할 수 있다. 자세한 적용 방식은 [HuberRegressor 적용 설명](docs/huber_regressor_explanation.md)에 정리했다.

분류는 tuned Logistic Regression이 Recall 0.9818을 달성했다. 기존 Logistic Regression보다 실제 위험을 놓치는 false negative가 줄었다. 다만 Precision은 약간 낮아졌으므로, 운영에서는 “미탐을 줄이는 대신 일부 과잉 경보를 감수한다”는 의사결정으로 설명해야 한다.

딥러닝 후보로는 MLPRegressor와 MLPClassifier를 추가했다. MLP는 hidden layer를 통해 feature 간 비선형 조합을 학습할 수 있으나, 현재 데이터는 표본 수가 크지 않고 이미 도메인 기반 파생 feature가 많이 포함된 tabular 데이터다. 결과적으로 MLP는 최종 best가 되지 못했다. 따라서 본 프로젝트에서 딥러닝은 주 모델이 아니라, 복잡한 비선형 학습을 시도했지만 현재 데이터 조건에서는 규제 선형/robust 모델이 더 안정적이었다는 비교 근거로 해석한다.

고도화 회귀 후보 전체 결과는 다음과 같다.

| rank | experiment | workflow | model | RMSE | MAE | R2 | MAE_cells | RMSE_cells | selected params |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | huber_tuned | non_tree | HuberRegressor | 0.6691 | 0.4694 | 0.8403 | 5,312.7 | 15,376.0 | `epsilon=2.0`, `alpha=1e-05` |
| 2 | elasticnet_tuned | non_tree | ElasticNet | 0.6777 | 0.4935 | 0.8363 | 4,759.2 | 13,944.1 | `alpha=0.001`, `l1_ratio=0.4` |
| 3 | ridge_tuned | non_tree | Ridge | 0.6779 | 0.4923 | 0.8361 | 4,688.3 | 13,867.8 | `alpha=10.0` |
| 4 | catboost_regressor_tuned | tree | CatBoostRegressor | 0.6991 | 0.5305 | 0.8257 | 5,252.0 | 14,372.9 | `depth=5`, `iterations=500`, `l2_leaf_reg=7`, `learning_rate=0.02` |
| 5 | mlp_regressor_deep | non_tree | MLPRegressor | 0.7800 | 0.5939 | 0.7831 | 6,349.9 | 16,039.2 | `activation=tanh`, `hidden_layer_sizes=(64, 32)`, `alpha=0.01`, `learning_rate_init=0.003` |

고도화 분류 후보 전체 결과는 다음과 같다. `tuned_th`는 Recall을 더 높이기 위해 탐색한 운영 threshold 후보 중 선택된 값이다.

| rank | experiment | workflow | model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | tuned_th | tuned_precision | tuned_recall | selected params |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | logistic_regression_tuned | non_tree | LogisticRegression | 0.9510 | 0.9181 | 0.9818 | 0.9489 | 0.9896 | 0.9872 | 0.15 | 0.8176 | 0.9982 | `C=0.03`, `penalty=l2`, `class_weight=balanced` |
| 2 | catboost_classifier_tuned | tree | CatBoostClassifier | 0.9493 | 0.9420 | 0.9489 | 0.9455 | 0.9906 | 0.9889 | 0.10 | 0.8523 | 1.0000 | `depth=3`, `iterations=300`, `l2_leaf_reg=5`, `learning_rate=0.03` |
| 3 | calibrated_logistic_tuned | non_tree | CalibratedClassifierCV | 0.9468 | 0.9450 | 0.9398 | 0.9424 | 0.9884 | 0.9815 | 0.10 | 0.8558 | 0.9964 | `C=0.03`, `penalty=l1`, `method=isotonic` |
| 4 | random_forest_classifier_tuned | tree | RandomForestClassifier | 0.9468 | 0.9499 | 0.9343 | 0.9420 | 0.9930 | 0.9918 | 0.20 | 0.9013 | 1.0000 | `n_estimators=800`, `max_depth=6`, `max_features=log2`, `min_samples_leaf=3`, `class_weight=balanced` |
| 5 | mlp_classifier_deep | non_tree | MLPClassifier | 0.9451 | 0.9600 | 0.9197 | 0.9394 | 0.9886 | 0.9863 | 0.10 | 0.8390 | 0.9891 | `activation=relu`, `hidden_layer_sizes=(128, 64, 32)`, `alpha=0.01`, `learning_rate_init=0.001` |

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

## 운영 시나리오 흐름

최종 운영 목적은 단순히 모델 점수를 출력하는 것이 아니라, 대청댐 관리자가 현재 상황을 보고 어떤 대응을 해야 하는지 **시나리오 기반으로 설명 가능한 판단 근거를 제공하는 것**이다. 따라서 본 프로젝트의 운영 흐름은 아래처럼 구성하는 것이 적절하다.

```text
입력 데이터
↓
Logistic Regression / HuberRegressor 예측
↓
위험 단계 산정
↓
SHAP으로 위험 원인 설명
↓
시나리오 기반 대응안 출력
↓
추후 운영 데이터 축적 후 Hvt-UCB로 대응 정책 최적화
```

여기서 `Logistic Regression`은 다음 조사 시점의 경보 위험 여부를 판단하는 **분류 주 모델**이다. 운영상 중요한 것은 실제 위험 상황을 놓치지 않는 것이므로 Recall이 높은 Logistic Regression을 조기경보 판단의 중심 모델로 둔다. `HuberRegressor`는 다음 조사 시점의 유해남조류 세포수 로그값을 예측하는 **회귀 주 모델**이다. 폭우, 유입량 급증, 조류 폭증처럼 값이 튀는 구간에 덜 민감한 robust 회귀라는 점에서 보조적인 수치 예측 근거로 활용한다.

위험 단계는 두 모델의 출력을 함께 사용해 산정할 수 있다. 예를 들어 Logistic Regression의 경보 확률이 높고, HuberRegressor의 예상 세포수도 1,000 cells/mL 기준을 넘는다면 `고위험` 또는 `경계` 단계로 볼 수 있다. 반대로 분류 확률은 다소 높지만 예상 세포수가 낮거나 SHAP 원인이 약하면 `주의 관찰` 단계로 낮춰 볼 수 있다. 이처럼 분류 모델은 “위험 여부”, 회귀 모델은 “예상 규모”를 담당한다.

SHAP은 이 과정에서 관리자에게 설명 가능한 원인을 제공한다. 예측 결과가 `고위험`으로 나왔을 때 단순히 “위험 확률 0.92”라고 출력하는 것보다, “최근 7일 방류량 감소, 누적 강우 증가, 현재 세포수 증가, 경보 단계 상승이 위험도를 높였다”처럼 원인을 함께 제공해야 실제 대응으로 연결된다. 따라서 SHAP은 모델 해석용 부가 기능이 아니라, 시나리오 기반 대응안을 만드는 핵심 근거다.

시나리오 출력은 다음처럼 구성할 수 있다.

| 위험 단계 | 판단 예시 | 관리자 대응 시나리오 |
| --- | --- | --- |
| 안정 | 경보 확률 낮음, 예상 세포수 낮음 | 기존 모니터링 유지 |
| 주의 | 경보 확률 상승, 일부 수문·수질 위험 요인 존재 | 채수 빈도 확대 검토, 원인 feature 추적 |
| 경계 | 경보 확률 높음, 예상 세포수 기준 근접 또는 초과 | 현장 점검, 관계기관 공유, 취수·방류 운영 검토 |
| 고위험 | 경보 확률 매우 높음, 예상 세포수 기준 초과, SHAP 위험 요인 다수 | 즉시 경보 대응 체계 가동, 집중 모니터링, 운영 조치 우선순위 검토 |

Hvt-UCB는 현재 예측 모델로 직접 사용하지 않는다. 다만 운영 데이터가 충분히 쌓여 “어떤 상황에서 어떤 대응을 했고, 그 결과 세포수와 경보 상태가 어떻게 바뀌었는지”를 reward로 정의할 수 있다면, 향후에는 여러 대응 조치 중 장기적으로 효과적인 행동을 선택하는 **실시간 대응 정책 최적화 레이어**로 확장할 수 있다.

## SHAP 해석

SHAP 분석은 tree classification best model과 non_tree classification best model을 비교하기 위해 수행했다. 본 프로젝트에서 SHAP의 역할은 단순히 feature importance를 보여주는 것이 아니라, 운영 시나리오에서 **왜 해당 위험 단계가 나왔는지 설명하는 것**이다.

| workflow | 주요 SHAP feature |
| --- | --- |
| tree / RandomForest | `cyano_cells`(현재 유해남조류 세포수), `log_target`(현재 세포수 로그값), `alert_encoded`(현재 경보 단계), `water_temp`(수온), `turbidity`(탁도) |
| non_tree / Logistic Regression | `outflow_7d_sum_robust`(최근 7일 누적 방류량), `rain_7d_sum_x_robust`(최근 7일 누적 강우량), `log_target`(현재 세포수 로그값), `level_change_7d_robust`(최근 7일 수위 변화), `alert_encoded`(현재 경보 단계) |

Tree 모델은 현재 조류 상태와 수질 조건을 강하게 보고, non_tree 모델은 수문 변화와 현재 조류 상태를 강하게 본다. 이 차이는 두 workflow가 서로 다른 관점에서 위험을 판단한다는 의미가 있다. 운영 보고에서는 Logistic Regression을 주 모델로 두되, RandomForest/CatBoost의 SHAP 결과를 보조 설명으로 함께 제시하는 방식이 설득력 있다.

예를 들어 non_tree SHAP에서 `outflow_7d_sum_robust`, `rain_7d_sum_x_robust`, `level_change_7d_robust`, `log_target`이 크게 나타난다면, 모델은 최근 방류·강우·수위 변화와 현재 조류 상태를 근거로 위험도를 높게 판단한 것이다. 이 경우 대응 시나리오는 단순 관찰보다 수문 운영 상태 확인, 채수 빈도 확대, 관계기관 공유 쪽으로 강화될 수 있다. 반대로 tree SHAP에서 `cyano_cells`, `water_temp`, `turbidity`, `Microcystis`가 크게 나타난다면 현재 조류량과 수질 조건 자체가 위험 판단의 핵심 근거라는 뜻이므로 현장 수질 점검과 조류 종별 모니터링이 중요해진다.

단, tree 모델과 Logistic Regression의 raw SHAP 값은 같은 단위가 아니다. Tree SHAP은 확률 변화량에 가까운 작은 값으로 나오고, Linear SHAP은 log-odds 기준으로 더 크게 나올 수 있다. 따라서 workflow 간 비교 그림은 raw SHAP 절대값을 그대로 쓰지 않고, 각 workflow 내부 최대 중요도를 1.0으로 맞춘 `normalized_mean_abs_shap` 기준으로 저장한다.

```text
artifacts/shap/tree_classification_shap_beeswarm.png
artifacts/shap/non_tree_classification_shap_beeswarm.png
artifacts/shap/classification_shap_importance_comparison.png
```

<table>
  <tr>
    <td align="center" width="33%">
      <img src="artifacts/shap/tree_classification_shap_beeswarm.png" alt="Tree SHAP beeswarm" width="100%">
      <br>
      <sub>Tree SHAP beeswarm</sub>
    </td>
    <td align="center" width="33%">
      <img src="artifacts/shap/non_tree_classification_shap_beeswarm.png" alt="Non-tree SHAP beeswarm" width="100%">
      <br>
      <sub>Non-tree SHAP beeswarm</sub>
    </td>
    <td align="center" width="33%">
      <img src="artifacts/shap/classification_shap_importance_comparison.png" alt="Normalized SHAP comparison" width="100%">
      <br>
      <sub>Normalized SHAP comparison</sub>
    </td>
  </tr>
</table>

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
├── README.md                         # 프로젝트 목적, 실행 방법, 모델 결과를 정리한 메인 문서
├── requirements.txt                  # 모델 학습/시각화/해석에 필요한 Python 패키지 목록
├── docs/                             # 모델 선택 근거와 세부 설명 문서
│   └── huber_regressor_explanation.md # HuberRegressor 적용 방식과 해석
├── src/                              # 재사용 가능한 모델링 소스 코드와 데이터 정의
│   ├── config/                       # 경로, target, 후보 모델, 평가 기준 설정
│   │   └── model_config.py           # tree/non_tree workflow와 공통 상수 정의
│   ├── data/                         # 원본 데이터와 모델 입력 데이터 보관
│   │   ├── team-raw/                 # 전처리팀이 제공한 원본 CSV 보존 영역
│   │   └── processed/model_input/    # tree/non_tree 학습에 바로 쓰는 최종 입력 CSV
│   ├── schema/                       # 모델 입력 컬럼 구조와 스키마 설명
│   │   ├── model_input_schema.csv    # 입력 컬럼 목록과 타입/역할 요약
│   │   └── model_input_schema.md     # 모델 입력 스키마 설명 문서
│   ├── pipeline/                     # 모델 학습, 평가, 저장, 해석의 핵심 파이프라인
│   │   ├── data.py                   # 데이터 로드, feature 선택, train/valid split
│   │   ├── models.py                 # 후보 회귀/분류 모델 정의와 학습
│   │   ├── evaluation.py             # RMSE, Recall, Precision 등 성능 평가
│   │   ├── artifacts.py              # 모델, 예측값, metric, feature importance 저장
│   │   ├── runner.py                 # 전체 workflow 실행 진입점
│   │   ├── visualization.py          # 모델 비교 결과 시각화 생성
│   │   ├── diagnostics.py            # 비트리 모델 잔차와 확률 보정 진단
│   │   ├── shap_compare.py           # tree/non_tree SHAP 해석 비교
│   │   └── enhancement.py            # 튜닝, MLP 등 모델 고도화 실험
│   └── utils/                        # 데이터셋 재생성 등 보조 유틸리티
│       └── build_model_datasets.py   # ALGAE_DATA 기반 tree/non_tree 입력 생성
├── artifacts/                        # 학습 후 생성된 모델, 지표, 예측, 그림 산출물
│   ├── tree_gradient_boosting/       # tree workflow의 모델/metric/prediction/explain
│   ├── non_tree_scaled/              # non_tree workflow의 모델/metric/prediction/explain
│   ├── enhancement/                  # 고도화 실험 결과와 튜닝 모델
│   ├── diagnostics/                  # 잔차 진단, Q-Q plot, calibration plot
│   ├── figures/                      # 모델 성능 비교 그래프
│   └── shap/                         # SHAP importance, bar, beeswarm 결과
└── eda/                              # 탐색적 데이터 분석 스크립트, 표, 시각화
```

핵심 코드는 `src/pipeline/`에 있다. `runner.py`가 전체 실행 진입점이고, `data.py`, `models.py`, `evaluation.py`, `artifacts.py`가 각각 데이터 로드, 모델 학습, 평가, 저장을 맡는다. `enhancement.py`는 추가 고도화 실험을 담당하며, `visualization.py`, `diagnostics.py`, `shap_compare.py`는 결과 분석 산출물을 만든다.

## 주요 산출물

| 경로 | 내용 |
| --- | --- |
| `src/data/README.md` | 모델 입력 데이터 생성 기준 |
| `docs/huber_regressor_explanation.md` | HuberRegressor 적용 방식과 회귀 성능 해석 |
| `artifacts/all_model_results.csv` | 모든 후보 모델의 결과표 |
| `artifacts/workflow_comparison_summary.csv` | workflow별 best 모델 요약 |
| `artifacts/enhancement/enhancement_report.md` | 고도화 실험 결과와 최종 추천 |
| `artifacts/figures/` | 모델 성능 비교 시각화 |
| `artifacts/shap/` | SHAP 해석 결과 |
| `eda/figure_interpretation.md` | EDA 그래프별 해석 |

## 결론

현재 프로젝트의 가장 중요한 결론은 세 가지다. 첫째, `ALGAE_DATA.csv`는 수질·조류·댐 운영 데이터와 기상 station 데이터가 결합된 station-expanded 병합 데이터이며, 모델링에서는 날짜 기준 split이 필수다. 둘째, 현재 holdout 검증에서는 트리 기반 모델보다 스케일링된 비트리 모델이 더 안정적인 성능을 보였다. 셋째, 조기경보 운영 목적에서는 회귀 RMSE보다 분류 Recall이 더 중요한 의사결정 지표이며, tuned Logistic Regression이 현재 가장 적합한 1차 운영 후보로 판단된다.

다만 단일 holdout 결과만으로 최종 운영 모델을 확정하기에는 부족하다. 다음 단계에서는 walk-forward validation으로 여러 미래 구간에서 성능 안정성을 확인하고, threshold tuning으로 미탐과 과잉 경보의 균형을 운영 목적에 맞게 조정해야 한다. 트리 기반 모델은 최종 best가 아니더라도 SHAP 기반 원인 해석과 비선형 비교 모델로 계속 유지하는 것이 좋다.
