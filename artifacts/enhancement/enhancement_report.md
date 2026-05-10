# 모델 고도화 실험 결과

현재 데이터는 날짜 순서가 완만하게 이어지는 전형적인 시계열이라기보다, 조사 구간마다 조류와 수문 조건이 크게 튀는 tabular 이벤트 데이터에 가깝다. 그래서 고도화는 시간순 rolling 예측이 아니라, 같은 조사일이 train/CV fold에 동시에 섞이지 않도록 `date`를 group으로 묶은 교차검증 기반 튜닝으로 진행했다.

## 고도화 방식

- 회귀는 train 내부 `GroupKFold(date)`에서 RMSE가 낮은 파라미터를 선택했다.
- 분류는 train 내부 `StratifiedGroupKFold(date)`에서 Recall이 높은 파라미터를 선택했다.
- 최종 성능은 기존 valid split에서 다시 평가했다.
- 딥러닝은 tabular 데이터에 맞는 MLP를 사용했다.

## regression 결과

|   rank | experiment_name          | workflow   |   rmse |     r2 |    mae |   mae_cells |   rmse_cells | best_params                                                                                        |
|-------:|:-------------------------|:-----------|-------:|-------:|-------:|------------:|-------------:|:---------------------------------------------------------------------------------------------------|
|      1 | huber_tuned              | non_tree   | 0.6691 | 0.8403 | 0.4694 |     5312.73 |      15376   | {"alpha": 1e-05, "epsilon": 2.0}                                                                   |
|      2 | elasticnet_tuned         | non_tree   | 0.6777 | 0.8363 | 0.4935 |     4759.17 |      13944.1 | {"alpha": 0.001, "l1_ratio": 0.4}                                                                  |
|      3 | ridge_tuned              | non_tree   | 0.6779 | 0.8361 | 0.4923 |     4688.27 |      13867.8 | {"alpha": 10.0}                                                                                    |
|      4 | catboost_regressor_tuned | tree       | 0.6991 | 0.8257 | 0.5305 |     5252.01 |      14372.9 | {"depth": 5, "iterations": 500, "l2_leaf_reg": 7, "learning_rate": 0.02}                           |
|      5 | mlp_regressor_deep       | non_tree   | 0.78   | 0.7831 | 0.5939 |     6349.86 |      16039.2 | {"activation": "tanh", "alpha": 0.01, "hidden_layer_sizes": [64, 32], "learning_rate_init": 0.003} |

## classification 결과

|   rank | experiment_name                | workflow   |   accuracy |   precision |   recall |     f1 |   roc_auc |   pr_auc |   tuned_threshold |   tuned_threshold_recall |   tuned_threshold_precision | best_params                                                                                                      |
|-------:|:-------------------------------|:-----------|-----------:|------------:|---------:|-------:|----------:|---------:|------------------:|-------------------------:|----------------------------:|:-----------------------------------------------------------------------------------------------------------------|
|      1 | logistic_regression_tuned      | non_tree   |     0.951  |      0.9181 |   0.9818 | 0.9489 |    0.9896 |   0.9872 |              0.15 |                   0.9982 |                      0.8176 | {"C": 0.03, "class_weight": "balanced", "penalty": "l2"}                                                         |
|      2 | catboost_classifier_tuned      | tree       |     0.9493 |      0.942  |   0.9489 | 0.9455 |    0.9906 |   0.9889 |              0.1  |                   1      |                      0.8523 | {"depth": 3, "iterations": 300, "l2_leaf_reg": 5, "learning_rate": 0.03}                                         |
|      3 | calibrated_logistic_tuned      | non_tree   |     0.9468 |      0.945  |   0.9398 | 0.9424 |    0.9884 |   0.9815 |              0.1  |                   0.9964 |                      0.8558 | {"estimator__C": 0.03, "estimator__penalty": "l1", "method": "isotonic"}                                         |
|      4 | random_forest_classifier_tuned | tree       |     0.9468 |      0.9499 |   0.9343 | 0.942  |    0.993  |   0.9918 |              0.2  |                   1      |                      0.9013 | {"class_weight": "balanced", "max_depth": 6, "max_features": "log2", "min_samples_leaf": 3, "n_estimators": 800} |
|      5 | mlp_classifier_deep            | non_tree   |     0.9451 |      0.96   |   0.9197 | 0.9394 |    0.9886 |   0.9863 |              0.1  |                   0.9891 |                      0.839  | {"activation": "relu", "alpha": 0.01, "hidden_layer_sizes": [128, 64, 32], "learning_rate_init": 0.001}          |

## 최종 추천

- 회귀 고도화 best: `huber_tuned` / RMSE `0.6691` / R2 `0.8403`
- 분류 고도화 best: `logistic_regression_tuned` / Recall `0.9818` / Precision `0.9181` / F1 `0.9489`

해석상 가장 중요한 모델은 분류의 `logistic_regression_tuned`다. 조류경보 문제는 실제 위험을 놓치지 않는 것이 중요하기 때문에 Recall을 우선했고, Logistic Regression은 강한 현재 조류 상태와 수문 신호를 단순한 확률 모델로 안정적으로 사용한다. 이는 binary outcome을 logit link로 모델링하는 고전적 로지스틱 회귀의 장점과 맞고, 현재처럼 스케일링된 tabular feature에서는 복잡한 모델보다 과적합 위험이 작다.

다만 딥러닝 MLP도 비교에 포함했다. MLP는 여러 hidden layer를 통해 비선형 feature 조합을 학습할 수 있지만, 현재 표본 수와 feature 구조에서는 tree/linear 계열보다 반드시 우세하다고 보기 어렵다. 따라서 딥러닝은 최종 주 모델이라기보다 성능 상승 가능성을 확인하는 추가 후보로 해석한다.