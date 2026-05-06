# 모델 결과 최종 정리

실행 기준:

- 입력 데이터: `ALGAE_DATA.csv` 기반 모델 입력 2종
- 검증 방식: `date-level chronological holdout`
- 평가 구간: `valid` split
- 회귀 target: `next_log_cells`
- 분류 target: `target_alert_next`
- 분류 threshold: `0.5`

## 1. 비교 대상

| workflow | 입력 데이터 | 모델군 |
| --- | --- | --- |
| `tree` | `tree_gradient_boosting/algae_tree_station_expanded.csv` | LightGBM, XGBoost, HistGradientBoosting, RandomForest, CatBoost |
| `non_tree` | `non_tree_scaled/algae_non_tree_scaled_station_expanded.csv` | Ridge, ElasticNet, HuberRegressor, SVR, KNN, Logistic Regression, Calibrated Logistic Regression, SVC |

## 2. 핵심 결과

| workflow | task | best model | 핵심 metric |
| --- | --- | --- | --- |
| `tree` | regression | `catboost` | RMSE `0.7200`, R2 `0.8152` |
| `tree` | classification | `random_forest` | Recall `0.9197`, Precision `0.9692`, F1 `0.9438` |
| `non_tree` | regression | `elasticnet` | RMSE `0.6773`, R2 `0.8364` |
| `non_tree` | classification | `logistic_regression` | Recall `0.9599`, Precision `0.9427`, F1 `0.9512` |

현재 holdout 기준으로는 `non_tree_scaled` workflow가 회귀와 분류 모두에서 더 좋은 결과를 보였다.

## 3. 회귀 결과 해석

회귀는 다음 조사 시점의 유해남조류 세포수 로그값인 `next_log_cells`를 예측한다.

| workflow | model | MAE | RMSE | R2 | MAE cells | RMSE cells |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `tree` | `catboost` | 0.5471 | 0.7200 | 0.8152 | 5,272.2 | 14,413.6 |
| `non_tree` | `elasticnet` | 0.4932 | 0.6773 | 0.8364 | 4,764.3 | 13,977.2 |

해석:

- 로그 스케일 RMSE는 `non_tree`가 더 낮다.
- 원 세포수 단위 MAE도 `non_tree`가 조금 더 낮다.
- RMSE cells는 두 workflow가 비슷하지만 `non_tree`가 근소하게 낮다.
- 현재 split에서는 스케일링과 로그 변환을 적용한 비트리 입력이 선형 회귀 계열 모델에 잘 맞았다.

시각화:

![Regression RMSE](/Users/hywznn/Desktop/model/artifacts/figures/regression_rmse_by_workflow.png)

![Regression Scatter](/Users/hywznn/Desktop/model/artifacts/figures/regression_prediction_scatter.png)

## 4. 분류 결과 해석

분류는 다음 조사 시점의 유해남조류 세포수가 1,000 이상인지 예측한다.

| workflow | model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `tree` | `random_forest` | 0.9493 | 0.9692 | 0.9197 | 0.9438 | 0.9941 | 0.9931 |
| `non_tree` | `logistic_regression` | 0.9544 | 0.9427 | 0.9599 | 0.9512 | 0.9924 | 0.9911 |

해석:

- `tree`의 RandomForest는 tree 계열 중 Recall/F1이 가장 높고 Precision도 높다.
- 하지만 Recall은 `non_tree`의 Logistic Regression이 더 높다.
- 조류경보 예측에서는 미탐을 줄이는 것이 중요하므로 Recall 관점에서는 `non_tree`가 더 유리하다.
- F1도 `non_tree`가 더 높아 precision-recall 균형이 더 좋았다.
- ROC-AUC/PR-AUC는 두 workflow 모두 매우 높고 차이는 작다.

시각화:

![Classification Metrics](/Users/hywznn/Desktop/model/artifacts/figures/classification_metrics_by_workflow.png)

![Confusion Matrices](/Users/hywznn/Desktop/model/artifacts/figures/classification_confusion_matrices.png)

## 5. Confusion Matrix 기준 비교

`tree` classification best model: `random_forest`

```text
TN 620 / FP 16
FN 44  / TP 504
```

`non_tree` classification best model: `logistic_regression`

```text
TN 604 / FP 32
FN 22  / TP 526
```

해석:

- `tree`는 false positive가 비교적 적다. 즉 불필요한 경보 후보를 덜 만든다.
- `non_tree`는 false negative가 적다. 즉 실제 위험을 놓치는 경우가 훨씬 적다.
- 운영 목적이 조기 경보라면 false negative를 줄이는 `non_tree` 분류 모델이 더 적합하다.
- 단, 현장 대응 비용이 매우 높고 오경보를 줄이는 것이 더 중요하면 `tree` 분류 모델도 후보로 남길 수 있다.

## 6. Feature Importance

### Tree Workflow

상위 feature:

| rank | feature | importance |
| ---: | --- | ---: |
| 1 | `Chl_a` | 798.0 |
| 2 | `DO` | 747.0 |
| 3 | `turbidity` | 727.0 |
| 4 | `acc_temp_7d` | 650.0 |
| 5 | `cyano_cells` | 628.0 |
| 6 | `water_temp` | 590.0 |
| 7 | `transparency` | 517.0 |
| 8 | `water_level` | 439.0 |
| 9 | `pH` | 375.0 |
| 10 | `rainfall` | 368.0 |

해석:

- 트리 모델은 현재 조류량, 수온/누적수온, DO, 탁도, Chl-a 같은 생물학적/수질 상태를 강하게 사용했다.
- 이는 조류 증식 조건과 직접 연결되는 변수들이므로 도메인적으로 자연스럽다.

### Non-tree Workflow

상위 feature:

| rank | feature | importance |
| ---: | --- | ---: |
| 1 | `log_target` | 1.3356 |
| 2 | `rain_7d_sum_x_robust` | 0.7657 |
| 3 | `outflow_7d_sum_robust` | 0.4958 |
| 4 | `level_change_7d_robust` | 0.4267 |
| 5 | `microcystis_ratio` | 0.2909 |
| 6 | `sin_season_minmax` | 0.0883 |
| 7 | `aphanizomenon_ratio` | 0.0568 |
| 8 | `hot_days_30c_7d` | 0.0428 |
| 9 | `inflow_robust` | 0.0403 |
| 10 | `inflow_7d_sum_robust` | 0.0372 |

해석:

- 비트리 모델은 현재 세포수 로그값인 `log_target`을 가장 강하게 사용했다.
- 강우/방류/수위 변화 지표가 상위에 있어, 수문 메커니즘을 꽤 반영한다.
- `sin_season_minmax`, `hot_days_30c_7d`도 포함되어 계절성과 고온 조건이 일부 작동한다.

시각화:

![Feature Importance](/Users/hywznn/Desktop/model/artifacts/figures/feature_importance_top10.png)

## 7. 현재 결론

현재 holdout 검증 기준의 1차 결론:

1. 회귀 예측은 `non_tree_scaled + ElasticNet`이 가장 좋다.
2. 분류 예측은 `non_tree_scaled + Logistic Regression`이 가장 좋다.
3. 조류경보 운영 관점에서는 Recall이 높은 `non_tree` 분류 모델이 더 안전하다.
4. 트리 모델은 Precision이 높아 오경보를 줄이는 방향에서는 여전히 의미가 있다.
5. 두 workflow 모두 ROC-AUC/PR-AUC가 높아 분류 신호 자체는 충분히 강하다.

따라서 현재 추천 후보는 다음과 같다.

| 목적 | 추천 모델 |
| --- | --- |
| 다음 세포수 정량 예측 | `non_tree_scaled / ElasticNet` |
| 관심 이상 조기 탐지 | `non_tree_scaled / Logistic Regression` |
| 비선형 보조 비교군 | `tree / RandomForest`, `tree / CatBoost` |

## 8. 다음 검증 과제

현재 결과를 최종 모델로 확정하기 전에 아래를 추가로 확인하는 것이 좋다.

1. `walk-forward validation` 또는 연도별 out-of-time validation
2. `log_target` 포함/제외 ablation
3. station-expanded 구조를 대표 station 또는 station 평균 집계 방식과 비교
4. 분류 threshold를 0.5 고정이 아니라 recall/precision 목표 기준으로 조정
5. SHAP 기반 개별 예측 원인 분석 추가

## 9. 비트리 모델 진단

비트리 workflow는 현재 holdout 성능이 가장 좋지만, 선형 회귀 계열 모델을 **통계적 추론 모델**로 해석하기 전에 잔차 가정을 확인해야 한다.

진단 파일:

```text
artifacts/diagnostics/non_tree_diagnostics.json
artifacts/diagnostics/non_tree_residuals_vs_fitted.png
artifacts/diagnostics/non_tree_residual_qq.png
artifacts/diagnostics/non_tree_logistic_calibration.png
```

주요 결과:

| 항목 | 값 | 해석 |
| --- | ---: | --- |
| Durbin-Watson | 0.4624 | 2에서 멀어 잔차 자기상관 가능성이 높다. |
| Breusch-Pagan p-value | 2.57e-34 | 등분산성 가정이 약하다. |
| Jarque-Bera p-value | 6.55e-22 | 잔차 정규성 가정이 약하다. |
| Logistic Brier Score | 0.0323 | 분류 확률 보정은 비교적 양호하다. |

결론:

- `ElasticNet` 회귀는 예측 성능 기준으로는 유효하지만, 잔차 독립성·등분산성·정규성 가정이 약하다.
- 따라서 계수의 통계적 유의성이나 p-value 중심 해석에는 적합하지 않다.
- 현재 용도는 설명 가능한 예측 baseline으로 보는 것이 안전하다.
- 분류 모델인 `Logistic Regression`은 Recall/F1과 Brier score가 양호해 조기경보 분류 후보로 유지할 수 있다.

![Non-tree Residuals](/Users/hywznn/Desktop/model/artifacts/diagnostics/non_tree_residuals_vs_fitted.png)

![Non-tree Calibration](/Users/hywznn/Desktop/model/artifacts/diagnostics/non_tree_logistic_calibration.png)

## 10. SHAP 비교

Tree workflow의 best classification model인 `RandomForest`와 Non-tree workflow의 best classification model인 `Logistic Regression`에 대해 SHAP 분석을 수행했다.

생성 파일:

```text
artifacts/shap/tree_classification_shap_bar.png
artifacts/shap/tree_classification_shap_beeswarm.png
artifacts/shap/non_tree_classification_shap_bar.png
artifacts/shap/non_tree_classification_shap_beeswarm.png
artifacts/shap/classification_shap_importance_comparison.png
artifacts/shap/classification_shap_importance_comparison.csv
```

상위 SHAP feature:

| workflow | model | 주요 feature |
| --- | --- | --- |
| tree | RandomForest | `cyano_cells`, `log_target`, `alert_encoded`, `water_temp`, `turbidity` |
| non_tree | Logistic Regression | `outflow_7d_sum_robust`, `rain_7d_sum_x_robust`, `log_target`, `level_change_7d_robust`, `alert_encoded` |

해석:

- Tree 모델은 현재 조류량, 현재 경보 단계, 수질 조건을 강하게 사용한다.
- Non-tree 모델은 수문 변화량, 누적 강우/방류, 현재 조류 상태를 강하게 사용한다.
- 두 모델이 서로 다른 관점의 신호를 사용하므로, 운영 보고에서는 두 해석을 함께 제시하는 것이 좋다.

![SHAP Comparison](/Users/hywznn/Desktop/model/artifacts/shap/classification_shap_importance_comparison.png)
