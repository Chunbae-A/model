# HuberRegressor 적용 설명

## 핵심 요약

본 프로젝트에서 `HuberRegressor`는 **다음 조사 시점의 유해남조류 세포수 로그값(`next_log_cells`)을 예측하는 회귀 모델**로 사용된다.

핵심 목적은 다음과 같다.

> 다음 조사 시점의 유해남조류 세포수 로그값을 예측하되, 폭우·유입량 급증·조류 폭증처럼 값이 크게 튀는 구간에 모델이 과도하게 끌려가지 않도록 robust 회귀를 적용한다.

## 코드상 적용 위치

`HuberRegressor`는 두 위치에서 사용된다.

첫 번째는 기본 non-tree 회귀 후보 모델이다.

```text
src/pipeline/models.py
```

기본 설정은 다음과 같다.

```python
HuberRegressor(epsilon=1.35, alpha=0.0001, max_iter=5000)
```

두 번째는 모델 고도화 실험이다.

```text
src/pipeline/enhancement.py
```

고도화에서는 여러 파라미터를 탐색했고, 최종 선택된 값은 다음과 같다.

```text
epsilon = 2.0
alpha = 1e-05
```

## 사용 데이터

`HuberRegressor`는 `non_tree` workflow에 포함되어 있다. 따라서 아래 모델 입력 데이터를 사용한다.

```text
src/data/processed/model_input/non_tree_scaled/algae_non_tree_scaled_station_expanded.csv
```

이 데이터는 선형·거리 기반·딥러닝 계열 모델에 맞게 전처리된 입력이다.

| feature 계열 | 처리 방식 |
| --- | --- |
| station, loc_encoded | one-hot encoding |
| 세포수, Chl-a, 탁도 | `log10(x + 1)` 후 RobustScaler |
| 강우, 유입, 방류, 정체 지수 | RobustScaler |
| 수온, pH, DO, 수위 등 | MinMaxScaler |

즉 HuberRegressor는 원본 단위의 값을 그대로 보는 것이 아니라, **스케일링된 feature들**을 입력으로 사용한다.

## 예측 Target

예측 target은 다음과 같다.

```text
next_log_cells
```

이는 다음 조사 시점 유해남조류 세포수의 로그 변환값이다.

```text
next_log_cells = log10(다음 세포수 + 1)
```

모델은 로그 스케일에서 예측하고, 평가 단계에서는 다시 원래 세포수 단위로 되돌려 `MAE_cells`, `RMSE_cells`도 함께 계산한다.

## HuberRegressor를 사용한 이유

일반적인 MSE 기반 회귀 모델은 큰 오차를 매우 강하게 벌준다. 그런데 본 프로젝트의 데이터는 다음과 같은 특징이 있다.

- 유해남조류 세포수가 특정 조사 시점에 급격히 폭증한다.
- 강우량, 유입량, 방류량이 일부 기간에 크게 튄다.
- 평상시 조건과 이벤트 조건의 차이가 크다.

이런 데이터에서는 극단적인 몇 개 관측치가 모델 전체의 회귀식을 과도하게 흔들 수 있다.

`HuberRegressor`는 작은 오차에는 일반적인 제곱 오차처럼 반응하지만, 일정 기준 이상으로 큰 오차에는 선형적으로 반응한다. 따라서 이상치가 많은 데이터에서 일반 선형회귀보다 더 안정적인 학습이 가능하다.

## 성능 결과

기본 모델 비교에서 HuberRegressor의 결과는 다음과 같았다.

| model | RMSE | R2 |
| --- | ---: | ---: |
| HuberRegressor | 0.7153 | 0.8176 |

기본 설정에서는 `ElasticNet`보다 성능이 낮았다.

하지만 고도화 실험에서 파라미터 튜닝 후 결과는 다음과 같이 개선되었다.

| model | RMSE | R2 | selected params |
| --- | ---: | ---: | --- |
| HuberRegressor tuned | 0.6691 | 0.8403 | `epsilon=2.0`, `alpha=1e-05` |

고도화 후 HuberRegressor는 회귀 후보 중 가장 좋은 RMSE와 R2를 보였다.

## 해석

기본 설정에서는 ElasticNet이 더 안정적인 결과를 보였지만, HuberRegressor의 이상치 대응 강도와 규제 강도를 조정하자 최종 회귀 성능이 가장 좋아졌다.

이는 본 데이터에서 조류 폭증, 폭우, 유입량 급증처럼 값이 튀는 구간이 실제로 모델 성능에 영향을 주고 있으며, 큰 오차에 덜 민감한 robust loss가 유효하게 작동했음을 의미한다.

## 보고서용 문장

보고서나 발표에서는 다음과 같이 정리할 수 있다.

> 본 프로젝트에서 HuberRegressor는 다음 조사 시점 유해남조류 세포수의 로그값을 예측하는 robust 회귀 모델로 사용되었다. 조류 데이터는 폭우, 유입량 급증, 조류 대발생 등으로 인해 극단값이 많기 때문에 일반 MSE 기반 회귀는 일부 이상치에 과도하게 영향을 받을 수 있다. HuberRegressor는 큰 오차에 덜 민감한 Huber loss를 사용하여 이러한 문제를 완화하며, 고도화 실험에서 RMSE 0.6691, R2 0.8403으로 가장 우수한 회귀 성능을 보였다.
