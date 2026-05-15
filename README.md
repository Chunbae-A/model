# 대청댐 수역별 유해남조류 선행 예측 모델

<img width="1693" height="929" alt="img" src="https://github.com/user-attachments/assets/db890506-92e3-41da-9345-874791e1bc23" />

대청댐의 문의, 회남, 추동 수역 데이터를 이용해 다음 채수 시점의 유해남조류 세포수를 예측하고, 조류경보 운영 기준에 맞춘 의사결정 시나리오를 생성하는 모델링 프로젝트입니다.

## 목차

1. [타깃 선정](#타깃-선정)
2. [디렉토리 구조](#디렉토리-구조)
3. [데이터와 feature](#데이터와-feature)
4. [Feature 선택 방식](#feature-선택-방식)
5. [수역 인코딩](#수역-인코딩)
6. [경보 기준](#경보-기준)
7. [학습 실행](#학습-실행)
8. [사용 모델](#사용-모델)
9. [평가 지표](#평가-지표)
10. [회귀 결과](#회귀-결과)
11. [분류 결과](#분류-결과)
12. [모델 비교 그래프](#모델-비교-그래프)
13. [시나리오 분류 기준](#시나리오-분류-기준)
14. [시나리오 결과](#시나리오-결과)
15. [주요 산출물](#주요-산출물)
16. [LLM 브리핑 payload 생성](#llm-브리핑-payload-생성)
17. [시나리오별 대응 초안](#시나리오별-대응-초안)

## 타깃 선정

이 프로젝트의 핵심 타깃은 `유해남조류_세포수`입니다. 이 값을 타깃으로 잡은 이유는 조류경보제의 행정 판단 기준이 결국 유해남조류 세포수이기 때문입니다. 사용자가 제공한 운영 기준에서도 관심, 경계, 조류대발생 단계는 남조류 유해 항목 4종의 세포수 기준으로 나뉩니다. 즉 모델이 실제 의사결정에 연결되려면 Chl-a나 수온 같은 간접 지표가 아니라, 경보 발령 기준과 직접 연결되는 `유해남조류_세포수`를 먼저 예측해야 합니다.

`유해남조류_세포수`는 Microcystis, Anabaena, Oscillatoria, Aphanizomenon 네 종의 세포수를 합산한 값으로 봅니다. 종별 세포수는 왜 위험한지 설명하는 데 중요하지만, 최종 행정 기준은 종별 하나하나가 아니라 유해남조류 총세포수 기준으로 판단됩니다. 그래서 종별 세포수는 타깃이 아니라 설명 변수 또는 원인 해석 변수로 두고, 최종 예측 타깃은 총 유해남조류 세포수로 정했습니다.

회귀 모델의 정답은 현재 세포수가 아니라 `next_log_cells`입니다. 이 컬럼은 다음 채수 시점의 유해남조류 세포수를 `log10(cells + 1)`로 변환한 값입니다. 현재 시점의 세포수를 맞히는 모델은 이미 관측된 상태를 재확인하는 수준에 그칠 수 있습니다. 이 프로젝트의 목적은 오늘까지 확인 가능한 수질, 수문, 기상 조건으로 다음 채수 시점 또는 약 7일 뒤 위험을 먼저 알려주는 것이므로, 다음 시점의 세포수인 `next_log_cells`를 회귀 타깃으로 사용합니다.

로그 변환을 쓰는 이유는 유해남조류 세포수가 대부분 기간에는 낮거나 0에 가깝다가 여름철 특정 시기에 수천, 수만 단위로 급증하는 치우친 분포를 갖기 때문입니다. 원본 세포수를 그대로 학습하면 일부 큰 값에 모델이 과하게 끌릴 수 있습니다. `log10(cells + 1)`을 쓰면 0도 안전하게 처리하면서 급증 구간의 영향을 완화할 수 있고, 예측 후에는 다시 cells/mL 단위로 복원해 현장 기준과 비교할 수 있습니다.

분류 모델의 정답은 학습 중 `next_alert_binary`로 새로 만듭니다. `next_alert_binary`는 다음 채수 시점의 예측 세포수가 관심 기준인 `1,000 cells/mL` 이상인지 나타내는 이진 타깃입니다. 관심 단계 기준을 넘는지 먼저 맞히는 이진 분류로 둔 이유는, 미발령/관심/경계/대발생 전체를 한 번에 나누기에는 경계와 대발생 표본이 상대적으로 적고, 실제 운영에서 가장 중요한 첫 질문이 “다음 채수 때 관심 이상으로 넘어갈 가능성이 있는가?”이기 때문입니다.

마지막으로 운영 판단에는 `previous_exceeded`와 예측 결과를 함께 봅니다. 조류경보제는 단순히 한 번 기준을 넘었다고 바로 모든 상황이 확정되는 구조가 아니라, 2회 연속 채수 기준을 중요하게 봅니다. 그래서 코드에서는 직전 실측이 기준을 넘었는지와 다음 시점 예측이 기준을 넘는지를 함께 확인해 `operational_alert_candidate` 또는 시나리오 판단에 반영합니다.

정리하면 이 프로젝트의 타깃 구조는 다음과 같습니다.

| 구분 | 사용 컬럼 | 의미 | 선정 이유 |
| --- | --- | --- | --- |
| 원본 핵심 타깃 | `유해남조류_세포수` | 유해남조류 4종 총세포수 | 조류경보 발령 기준과 직접 연결 |
| 회귀 타깃 | `next_log_cells` | 다음 채수 시점 세포수의 로그값 | 7일 선행 예측과 급증 분포 완화 |
| 분류 타깃 | `next_alert_binary` | 다음 채수 시점 관심 이상 여부 | 관심 기준 1,000 cells/mL 조기 탐지 |
| 운영 판단 보조 | `previous_exceeded` + 예측 초과 여부 | 2회 연속 기준 반영 | 실제 조류경보 운영 로직과 연결 |
| 설명 변수 | 종별 세포수, 수온, Chl-a, 강우, 방류량, 체류시간 등 | 위험 원인 해석 | 왜 위험한지 설명하고 시나리오 생성 |

## 디렉토리 구조

```text
model/
  train.py                         # 전체 학습, 평가, 예측, 설명, 시나리오 생성
  plot_model_comparison.py          # 모델별 성능 비교 SVG 생성
  model_training.ipynb              # 실험 기록용 노트북
  requirements.txt                  # 실행 의존성
  README.md                         # 프로젝트 설명 문서

  data/
    daechung_for_merge_v1.csv       # 수질 + 수문 중심 원천/중간 테이블
    WEATHER.csv                     # 기상 일 단위 feature 테이블
    processed/model_input/
      algae_model_input.csv         # 최종 모델 입력 테이블

  src/
    config.py                       # 경로, 컬럼명, 임계값, 모델 후보, 시나리오 설정
    loader.py                       # CSV/Excel 로더
    data_prep.py                    # 간단 병합 유틸리티
    features.py                     # 시간 간격, 위치 순서, 회남 선행 압력 feature 생성
    models.py                       # 모델 후보 생성, 학습, 평가, best model 선택
    persistence.py                  # 모델, 메트릭, 예측 결과 저장
    explain_scenario.py             # SHAP 상위 요인, 경보 단계, 시나리오 생성
    llm_publisher.py                # LLM 브리핑용 payload/sample 생성

  artifacts/
    models/                         # 학습된 모델과 메타데이터
    metrics/                        # 회귀/분류 평가 JSON, threshold 결과
    predictions/                    # validation 예측 결과
    explain/                        # feature importance, SHAP top reasons
    scenario/                       # 전체 기간 시나리오 결과
    figures/                        # 모델 성능 비교 SVG

  output/
    run_YYYYMMDD_HHMMSS/            # 실행별 결과 백업과 LLM sample
```

## 데이터와 feature

최종 모델 입력은 다음 파일을 사용합니다.

| 파일 | 역할 |
| --- | --- |
| `data/processed/model_input/algae_model_input.csv` | 모델 학습에 직접 투입되는 최종 테이블 |
| `data/daechung_for_merge_v1.csv` | 수질, 수문 중심 데이터 |
| `data/WEATHER.csv` | 기온, 강우, 풍속, 일조, 일사 등 기상 feature |

주요 feature 그룹은 다음과 같습니다.

| 그룹 | 예시 컬럼 | 의미 |
| --- | --- | --- |
| 시간/공간 | `date`, `loc_encoded` | 채수일과 수역 |
| 수질 | `수온`, `pH`, `DO`, `탁도`, `투명도`, `Chl_a` | 채수 당시 수질 상태 |
| 남조류 | `유해남조류_세포수`, `Microcystis`, `Anabaena`, `Oscillatoria`, `Aphanizomenon` | 현재 생물학적 상태 |
| 수문 | `수위`, `저수량`, `저수율`, `유입량`, `방류량` | 댐 운영과 체류 조건 |
| 수문 파생 | `level_change_7d`, `inflow_7d_sum`, `outflow_7d_sum`, `residence_proxy`, `nutrient_stagnation_index` | 정체, 희석, 영양염 유입 조건 |
| 기상 | `avg_temp`, `daily_rain`, `avg_wind`, `sunshine`, `solar_rad`, `cloud_cover` | 조류 성장 환경 |
| 기상 파생 | `air_temp_7d_mean`, `rain_3d_sum`, `rain_14d_sum`, `wind_7d_mean`, `sunshine_7d_sum`, `solar_7d_sum` | 누적 고온, 강우, 저풍속, 광합성 조건 |
| 타깃 | `log_target`, `next_log_cells`, `alert_encoded` | 회귀/분류 정답과 경보 단계 |

## Feature 선택 방식

feature 선택은 [src/features.py](src/features.py)의 `get_feature_columns()`에서 수행합니다.

1. [src/config.py](src/config.py)의 `DROP_COLUMNS`에 들어 있는 날짜, 타깃, 행정 라벨, 누수 위험 컬럼을 제외합니다.
2. `FORBIDDEN_FEATURE_KEYWORDS`로 `next`, `target`, `future`처럼 미래 정답을 직접 암시할 수 있는 컬럼이 feature에 섞였는지 검사합니다.
3. `REQUIRE_NUMERIC_FEATURES=True`이면 학습에 들어가는 모든 feature가 숫자형인지 검사합니다.

따라서 이 프로젝트는 별도 입력 스키마 파일을 쓰지 않고, 코드 안의 drop rule과 누수 검사로 feature를 통제합니다. 이 방식은 처음 보는 사람이 확인하기 쉽고, 모델에 들어가는 컬럼이 `model_metadata.json`에도 저장되어 재현성이 유지됩니다.

## 수역 인코딩

현재 수역 인코딩은 사용자가 정리한 흐름 구조를 반영합니다.

| `loc_encoded` | 수역 |
| ---: | --- |
| 0 | 문의 |
| 1 | 회남 |
| 2 | 추동 |

모델은 단순히 0, 1, 2라는 숫자만 쓰는 것이 아니라 `loc_flow_order`, `hoenam_cells_same_date`, `hoenam_pressure_for_downstream`을 추가해 회남 선행 압력과 수역 차이를 반영합니다.

## 경보 기준

조류경보 기준은 유해남조류 4종의 세포수 합계를 기준으로 봅니다.

| 단계 | 기준 |
| --- | --- |
| 미발령 | 유해남조류 세포수 < 1,000 cells/mL |
| 관심 | 유해남조류 세포수 >= 1,000 cells/mL |
| 경계 | 유해남조류 세포수 >= 10,000 cells/mL |
| 조류대발생 | 유해남조류 세포수 >= 1,000,000 cells/mL |

운영 시나리오는 단순 세포수 예측만 보지 않고 `previous_exceeded`와 예측 결과를 함께 봅니다. 즉 직전 채수와 다음 채수 예측이 연속으로 기준을 넘는 경우 `operational_alert_candidate`로 표시합니다.

## 학습 실행

```powershell
cd model
$env:LOKY_MAX_CPU_COUNT='1'
$env:OMP_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
python train.py
```

학습 흐름은 다음과 같습니다.

1. `algae_model_input.csv` 로드
2. 시간/공간 파생 feature 생성
3. 다음 채수 타깃이 없는 마지막 row 제거
4. `next_alert_binary` 생성
5. 시간 기준 train/validation split
6. 회귀 후보 모델 학습
7. 분류 후보 모델 학습
8. persistence baseline과 비교
9. best regression / best classification 선택
10. 예측 CSV, 모델 메타데이터, SHAP, 시나리오 생성

현재 split은 시간 기반 validation입니다. `split` 컬럼이 있으면 그 값을 따르고, 없으면 날짜순 앞 80%를 train, 뒤 20%를 validation으로 사용합니다.

## 사용 모델

현재 후보 모델은 `config/model_config.yaml`에서 지정합니다. 기본 설정은 6개 모델입니다.

| 모델 | 사용 목적 | 주요 설정 |
| --- | --- | --- |
| `LightGBM` | tabular gradient boosting 비교 모델 | `n_estimators=500`, `learning_rate=0.03`, `num_leaves=31` |
| `XGBoost` | 대표적인 gradient boosting 비교 모델 | `n_estimators=500`, `learning_rate=0.03`, `max_depth=5` |
| `HistGradientBoosting` | scikit-learn 기반 boosting fallback | `max_iter=500`, `learning_rate=0.03` |
| `RandomForest` | 안정적인 bagging ensemble, 관심 이상 탐지 | `n_estimators=500`, `min_samples_leaf=2`, `max_features=sqrt` |
| `CatBoost` | 범주/수치 tabular data에 강한 ordered boosting 후보 | `iterations=500`, `learning_rate=0.03`, `depth=6` |
| `Stacking Ensemble` | 위 5개 모델 예측을 결합하는 메타 모델 | base models + `RidgeCV` 또는 `LogisticRegression` |

이 모델들은 딥러닝처럼 epoch 단위로 학습하지 않습니다. 대신 boosting 계열은 tree 또는 iteration 수가 epoch와 비슷한 반복 학습 단위입니다.

| 모델 | 반복 단위 | 학습률 |
| --- | --- | --- |
| LightGBM | `n_estimators=500` trees | `0.03` |
| XGBoost | `n_estimators=500` trees | `0.03` |
| HistGradientBoosting | `max_iter=500` boosting iterations | `0.03` |
| RandomForest | `n_estimators=500` independent trees | 없음 |
| CatBoost | `iterations=500` ordered boosting iterations | `0.03` |
| Stacking Ensemble | base model fitting + meta model fitting | base model 설정을 따름 |

모델 후보와 하이퍼파라미터를 바꾸려면 [config/model_config.yaml](config/model_config.yaml)을 수정하면 됩니다. `enabled_models.regression`과 `enabled_models.classification`에서 후보를 켜고 끌 수 있고, `models.*.regression`, `models.*.classification` 아래에서 모델별 설정을 바꿀 수 있습니다.

## 평가 지표

회귀 평가는 로그 스케일과 실제 cells/mL 스케일을 함께 봅니다.

| 지표 | 의미 |
| --- | --- |
| `MAE` | log target 기준 평균 절대 오차 |
| `RMSE` | log target 기준 큰 오차를 더 강하게 반영 |
| `R2` | 설명력 |
| `MAE_cells` | cells/mL로 복원한 평균 절대 오차 |
| `RMSE_cells` | cells/mL로 복원한 RMSE |
| `RMSLE_cells` | 세포수 규모 차이를 완화해 보는 로그 오차 |

분류 평가는 경보 미탐지를 줄이는 관점에서 `Recall`을 중요하게 봅니다.

| 지표 | 의미 |
| --- | --- |
| `Accuracy` | 전체 정답률 |
| `Precision` | 경보라고 예측한 것 중 실제 경보 비율 |
| `Recall` | 실제 경보 상황을 놓치지 않은 비율 |
| `F1` | Precision과 Recall의 균형 |
| `ROC AUC` | threshold 전반의 구분 능력 |
| `PR AUC` | 경보 class가 적을 때 더 민감한 분류 품질 |

## 회귀 결과

아래 결과는 `artifacts/models/candidate_model_metrics.csv`에 저장된 현재 실행 결과입니다. 회귀 타깃은 다음 채수 시점의 `next_log_cells`이며, 모델 비교는 validation 구간에서 수행했습니다.

| 모델 | RMSE(log) | RMSE(cells/mL) | MAE(log) | MAE(cells/mL) | R2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stacking Ensemble | 0.6678 | 13,316.0 | 0.4860 | 4,463.0 | 0.8403 |
| LightGBM | 0.6765 | 12,654.5 | 0.4844 | 3,963.4 | 0.8361 |
| RandomForest | 0.6788 | 14,073.2 | 0.5123 | 4,829.3 | 0.8349 |
| HistGradientBoosting | 0.6850 | 12,601.8 | 0.4964 | 3,990.6 | 0.8319 |
| XGBoost | 0.6871 | 13,398.1 | 0.5107 | 4,493.1 | 0.8309 |
| CatBoost | 0.6962 | 13,832.1 | 0.5317 | 5,062.6 | 0.8264 |
| Persistence baseline | 0.7987 | 14,604.1 | 0.4759 | 3,978.7 | 0.7715 |

회귀 기준 best model은 `Stacking Ensemble`입니다. log-scale RMSE가 0.6678로 가장 낮아 전체적인 로그 세포수 예측 오차가 가장 작았습니다. 다만 실제 cells/mL 단위 RMSE는 `HistGradientBoosting`이 12,601.8로 가장 낮아, 현장 단위 해석에서는 두 지표를 함께 확인하는 것이 좋습니다.

## 분류 결과

분류 타깃은 다음 채수 시점의 관심 이상 여부인 `next_alert_binary`입니다. 운영 관점에서는 실제 위험 상황을 놓치는 미탐지가 가장 위험하므로, best classification model은 Recall과 F1을 함께 보고 선택합니다.

| 모델 | Threshold | Accuracy | Precision | Recall | F1 | ROC AUC | PR AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RandomForest | 0.20 | 0.9587 | 0.9133 | 1.0000 | 0.9547 | 0.9956 | 0.9944 |
| Stacking Ensemble | 0.20 | 0.9524 | 0.9013 | 1.0000 | 0.9481 | 0.9952 | 0.9938 |
| CatBoost | 0.20 | 0.9619 | 0.9433 | 0.9708 | 0.9568 | 0.9934 | 0.9909 |
| LightGBM | 0.10 | 0.9651 | 0.9701 | 0.9489 | 0.9594 | 0.9918 | 0.9866 |
| XGBoost | 0.10 | 0.9619 | 0.9630 | 0.9489 | 0.9559 | 0.9954 | 0.9943 |
| HistGradientBoosting | 0.10 | 0.9556 | 0.9556 | 0.9416 | 0.9485 | 0.9928 | 0.9892 |
| Persistence baseline | 0.10 | 0.9492 | 0.9416 | 0.9416 | 0.9416 | 0.9483 | 0.9120 |

분류 기준 best model은 `RandomForest`입니다. Recall이 1.0000으로 실제 관심 이상 사례를 놓치지 않았고, 같은 Recall 1.0000인 `Stacking Ensemble`보다 F1이 더 높습니다. 따라서 경보 미탐지 최소화라는 운영 목적에 가장 잘 맞습니다.

## 모델 비교 그래프

<p align="center">
  <img src="./artifacts/figures/model_comparison.svg" alt="Model Comparison" width="800"/>
</p>


## 시나리오 분류 기준

시나리오는 단순히 위험/안전으로만 나누지 않았습니다. 실제 수질 관리자는 예측 세포수 하나만 받으면 다시 매뉴얼, 채수 기준, 수문 상태, 기상 조건을 따로 확인해야 합니다. 그래서 시나리오는 모델 예측값을 현장 대응 언어로 바꾸기 위해 다음 네 가지 기준으로 나눴습니다.

첫째, 행정 경보 기준입니다. 관심, 경계, 조류대발생은 각각 1,000, 10,000, 1,000,000 cells/mL 기준과 연결됩니다. 따라서 `관심 발령 후보`, `경계 발령 후보`, `조류대발생 감시`는 예측 세포수가 실제 경보 단계에 가까워졌을 때 바로 드러나도록 만든 분류입니다. 특히 `관심 발령 후보`와 `경계 발령 후보`는 직전 채수값과 다음 예측값을 함께 보아 2회 연속 기준을 반영합니다.

둘째, 발생 메커니즘입니다. 사용자가 제공한 자료에서 녹조는 수온 20도 이상인 여름부터 초가을에 주로 발생하고, 수온, 일조, 강우 후 영양염 유입, 방류량 부족, 체류시간 증가, 저풍속에 따른 성층과 정체가 중요한 원인으로 정리되어 있습니다. 그래서 `수온 20도 이상 계절 감시`, `고온·고일사 성장 촉진`, `강우 이후 영양염류 유입`, `수체 정체·성층 위험`, `강우 유입 후 정체 고위험`처럼 원인별 시나리오를 분리했습니다.

셋째, 공간 전파 구조입니다. 대청댐은 수역별로 경보가 따로 발령되고, 회남, 문의, 추동의 수역 특성이 다릅니다. 특히 회남의 세포수 상승이 다른 지점의 위험 신호가 될 수 있으므로 `회남 선행 전파 관찰` 시나리오를 따로 둡니다. 이 시나리오는 현재 지점이 회남이 아니고, 같은 날짜 회남 세포수가 관심 기준 이상일 때 다른 수역의 선제 관찰 필요성을 표시합니다.

넷째, 운영 단계입니다. 모든 상황이 곧바로 경보 발령 후보는 아닙니다. 예측값이 기준에는 못 미치지만 500 cells/mL 이상으로 접근하거나, 확률이 중간 이상이면 `관심 기준 접근` 또는 `관찰 강화`로 분류합니다. 반대로 직전에는 기준을 넘었지만 현재와 예측이 모두 낮아지면 `하향·해제 관찰`로 분류해 2회 연속 하위 기준 확인이 필요하다는 흐름을 남깁니다. 아무 위험 신호가 없으면 `일반 안정`으로 둡니다.

시나리오 판단 우선순위는 행정 위험이 큰 순서로 배치했습니다. 조류대발생 예측이 가장 먼저 잡히고, 그다음 경계/관심 발령 후보, 하향·해제 관찰, 관심 기준 돌파/접근, 회남 전파, 복합 고위험, 원인별 위험, 계절 감시, 일반 안정 순서입니다. 이렇게 한 이유는 같은 row가 여러 조건에 걸릴 때 가장 중요한 운영 메시지가 묻히지 않게 하기 위해서입니다.

| 시나리오 축 | 대표 시나리오 | 분류 이유 |
| --- | --- | --- |
| 행정 경보 기준 | 관심 발령 후보, 경계 발령 후보, 조류대발생 감시 | 실제 조류경보 기준과 직접 연결 |
| 기준 접근/해제 | 관심 기준 접근, 관심 기준 돌파 예측, 하향·해제 관찰 | 발령 전후의 운영 판단 지원 |
| 기상·수질 성장 조건 | 수온 20도 이상 계절 감시, 고온·고일사 성장 촉진 | 여름철 고수온·광합성 조건 반영 |
| 수문·정체 조건 | 수체 정체·성층 위험, 강우 유입 후 정체 고위험 | 체류시간, 방류량, 저풍속, 성층 위험 반영 |
| 강우·영양염 유입 | 강우 이후 영양염류 유입 | 강우 뒤 영양염 유입과 지연 효과 반영 |
| 공간 전파 | 회남 선행 전파 관찰 | 수역별 예측과 회남 선행 위험 반영 |
| 추세/설명 | 과거 증식 추세 지속, 복합 고위험 | 직전 증가세와 SHAP 주요 원인 결합 |
| 기본 상태 | 관찰 강화, 일반 안정 | 위험 신호가 약하거나 안정적인 상태 구분 |

## 시나리오 결과

시나리오는 validation 일부가 아니라 전체 모델 입력 기간을 대상으로 생성합니다.

```text
artifacts/scenario/scenario_results.csv
output/run_*/llm_scenario_sample.csv
```

최근 `llm_scenario_sample.csv`는 최대 1,500행까지 생성되도록 확장했습니다. 따라서 일반 안정 시나리오만 보는 문제가 줄고, 관심 접근, 관심 발령 후보, 경계 발령 후보, 수온 20도 이상 계절 감시, 수체 정체와 성층 위험, 회남 선행 전파 관찰 같은 다양한 상황을 샘플링할 수 있습니다.

최근 시나리오 분포는 다음과 같습니다.

| 시나리오 | 행 수 |
| --- | ---: |
| 일반 안정 시나리오 | 823 |
| 관심 발령 후보 시나리오 | 271 |
| 관심 기준 접근 시나리오 | 142 |
| 관심 기준 돌파 예측 시나리오 | 90 |
| 수온 20도 이상 계절 감시 시나리오 | 71 |
| 경계 발령 후보 시나리오 | 63 |
| 수체 정체·성층 위험 시나리오 | 41 |
| 하향·해제 관찰 시나리오 | 27 |
| 고온·고일사 성장 촉진 시나리오 | 22 |
| 과거 증식 추세 지속 시나리오 | 15 |
| 회남 선행 전파 관찰 시나리오 | 8 |

## 주요 산출물

| 파일 | 내용 |
| --- | --- |
| `artifacts/models/regression_model.pkl` | best 회귀 모델 |
| `artifacts/models/classification_model.pkl` | best 분류 모델 |
| `artifacts/models/model_metadata.json` | feature 목록, best model 이름, threshold |
| `artifacts/models/candidate_model_metrics.csv` | 후보 모델과 baseline 성능 비교 |
| `artifacts/predictions/regression_predictions.csv` | validation 회귀 예측 결과 |
| `artifacts/predictions/classification_predictions.csv` | validation 분류 예측 결과 |
| `artifacts/explain/feature_importance.csv` | best 분류 모델 feature importance |
| `artifacts/explain/shap_top_reasons.csv` | validation SHAP 상위 요인 |
| `artifacts/explain/shap_top_reasons_all.csv` | 전체 기간 SHAP 상위 요인 |
| `artifacts/scenario/scenario_results.csv` | 전체 기간 시나리오 결과 |
| `artifacts/figures/model_comparison.svg` | 모델 성능 비교 SVG |

## LLM 브리핑 payload 생성

학습 결과를 LLM에 넘기기 쉬운 형태로 정리하려면 다음을 실행합니다.

```powershell
cd model
python src\llm_publisher.py
```

특정 run을 지정할 수도 있습니다.

```powershell
python src\llm_publisher.py --run run_20260505_161211
```

생성 파일:

```text
output/run_*/llm_payload.json
output/run_*/llm_prompt.md
output/run_*/llm_models_summary.csv
output/run_*/llm_prediction_sample.csv
output/run_*/llm_scenario_sample.csv
```

## 시나리오별 대응 초안

아래 대응은 모델이 자동으로 행정 명령을 내리는 것이 아니라, 수자원공사 또는 관계기관 담당자가 빠르게 검토할 수 있는 의사결정 보조 문장입니다. 실제 조치는 기상 상황, 취수장 운영 조건, 댐 방류 가능량, 관계기관 협의, 현장 채수 결과를 함께 확인한 뒤 결정해야 합니다.

| 시나리오 | 수자원공사 대응 초안 |
| --- | --- |
| 일반 안정 시나리오 | 평시 모니터링을 유지하고, 수온·Chl-a·유해남조류 세포수의 완만한 상승 여부만 관찰합니다. 별도 비상 조치보다는 다음 정기 채수 결과를 확인합니다. |
| 수온 20도 이상 계절 감시 시나리오 | 여름철부터 초가을까지 녹조 발생 가능성이 커지는 기본 조건으로 보고, 관심 단계 전이라도 수온, 일조, 풍속, 체류시간 지표를 매주 점검합니다. |
| 고온·고일사 성장 촉진 시나리오 | 고수온과 강한 일조가 지속되는 구간이므로 표층 남조류 증식 여부를 집중 확인합니다. 물순환설비 가동 가능성, 취수 수심 조정 필요성, 정수처리 강화 준비 상태를 사전에 점검합니다. |
| 강우 이후 영양염류 유입 시나리오 | 강우 뒤 유입량 증가로 영양염류가 들어왔을 가능성을 보고, 강우 후 3~7일 사이 세포수 증가 여부를 추가 관찰합니다. 상류 유입부와 회남 지점의 수질 변화를 우선 확인합니다. |
| 수체 정체·성층 위험 시나리오 | 방류량 부족, 낮은 풍속, 체류시간 증가로 표층 정체와 성층 가능성이 있는 상황입니다. 물순환설비 가동, 선택 취수 수심 점검, 방류 운영 여력 검토를 우선 검토합니다. |
| 강우 유입 후 정체 고위험 시나리오 | 비로 영양염류가 유입됐지만 물이 충분히 빠지지 않는 복합 위험 상황으로 봅니다. 유입부와 정체 수역의 추가 채수, 방류/순환 조건 점검, 관계기관 사전 공유를 권장합니다. |
| 과거 증식 추세 지속 시나리오 | 직전 채수 대비 세포수 증가율이 높거나 증가량이 큰 상황입니다. 단발성 상승인지 연속 증가인지 확인하기 위해 다음 채수 전 현장 확인과 추가 모니터링을 검토합니다. |
| 회남 선행 전파 관찰 시나리오 | 회남에서 관심 기준 이상 위험이 먼저 감지된 경우로 보고, 문의·추동 지점의 선제 관찰을 강화합니다. 상류 변화가 다른 수역으로 이어지는지 lag 관점에서 추적합니다. |
| 관심 기준 접근 시나리오 | 예측 세포수가 1,000 cells/mL에 근접하는 단계입니다. 아직 발령 후보는 아니더라도 현장 육안 확인, 채수 일정 앞당김, 관계기관 사전 공유를 준비합니다. |
| 관심 기준 돌파 예측 시나리오 | 다음 채수 시점에 관심 기준을 넘을 가능성이 있는 상태입니다. 채수 및 분석 지연을 고려해 선제적으로 주 1회 이상 모니터링 체계를 준비하고 정수처리 대응 자원을 점검합니다. |
| 관심 발령 후보 시나리오 | 직전 실측과 다음 예측이 모두 관심 기준을 넘는 흐름입니다. 조류경보 발령 가능성을 관계기관과 공유하고, 주 1회 이상 채수, 취수장 수질 감시, 활성탄 등 정수처리 준비를 검토합니다. |
| 경계 발령 후보 시나리오 | 경계 기준 이상이 연속될 가능성이 있는 고위험 상황입니다. 주 2회 이상 채수 체계, 정수처리 강화, 취수 수심 조정, 물순환설비 가동, 조류 제거 장비 준비 여부를 우선 확인합니다. |
| 조류대발생 감시 시나리오 | 매우 높은 세포수 예측이 나온 비상 감시 상황입니다. 관계기관 긴급 공유, 현장 추가 채수, 독성 및 냄새물질 분석 준비, 정수장 운영 리스크 점검을 즉시 검토합니다. |
| 하향·해제 관찰 시나리오 | 직전에는 기준을 넘었지만 현재와 예측이 낮아지는 흐름입니다. 즉시 해제 판단보다 2회 연속 하위 기준 충족 여부를 확인하고, 모니터링 강도를 단계적으로 낮추는 방안을 검토합니다. |
| 복합 고위험 시나리오 | 고수온, 강우, 정체, 과거 증식, 회남 선행 위험 등 여러 신호가 동시에 나타난 상황입니다. 단일 조치보다 채수 강화, 관계기관 공유, 수문 운영 검토, 정수처리 준비를 묶은 통합 대응이 필요합니다. |
| 관찰 강화 시나리오 | 명확한 발령 후보는 아니지만 확률 또는 일부 원인 지표가 상승한 상태입니다. 다음 정기 채수 전까지 주요 원인 지표를 일 단위로 확인하고, 위험이 커지면 관심 접근 또는 발령 후보 단계로 전환합니다. |

LLM 브리핑 문장으로 만들 때는 위 대응 초안을 그대로 명령문처럼 쓰기보다, “권장”, “검토”, “사전 준비”, “관계기관 공유”처럼 의사결정 보조 표현으로 변환하는 것이 안전합니다.

## 2026-05-14 자동화 구조 추가 

이번 정리 이후 Python 실행 코드는 모두 `src/` 아래에 모았습니다. 루트에는 실행 스크립트, README, requirements만 남기는 구조입니다.

```text
model/
  run_pipeline.ps1                 # Windows 자동 실행 진입점
  run_pipeline.sh                  # Bash/WSL 자동 실행 진입점
  requirements.txt
  README.md

  src/
    pipeline.py                    # 전체 파이프라인 오케스트레이션
    water_gate.py                  # 대청댐 수문 Selenium 수집 및 원본 CSV 병합
    water_quality.py               # 대청호 조류/수질 Selenium 수집 및 원본 CSV 병합
    weather_api.py                 # 기상 API 수집 및 WEATHER.csv 생성
    fetch_10y.py                   # KMA ASOS 요청 helper
    preprocess_data.py             # 수문+수질+기상 병합, 파생변수 생성, Final.csv 저장
    train_models.py                # 모델 학습/평가/예측 산출
    plot_metrics.py                # 모델 성능 비교 SVG 생성
    config.py
    features.py
    models.py
    persistence.py
    explain_scenario.py
    loader.py
    model_config.py
    llm_publisher.py
```

병합과 전처리는 `src/preprocess_data.py`에서 수행합니다. 이 파일이 `data/대청수문_10년치_통합데이터.csv`, `data/수질_10년치_통합데이터.csv`, `data/WEATHER.csv`를 읽어서 수문/수질/기상을 합치고, TSI, 적산수온, 종별 비율, 수문 정체 지수, lag/diff 변수, `log_target`, `next_log_cells`를 만든 뒤 아래 파일을 저장합니다.

```text
data/Final.csv
data/daechung_final_clean_dataset.csv
data/combined_weather_water_10y.csv
data/processed/model_input/algae_model_input.csv
```

한 번에 실행하려면 Windows에서는 아래 명령을 사용합니다. PowerShell 실행 정책 때문에 `.\run_pipeline.ps1`이 막히는 경우가 있어 `-ExecutionPolicy Bypass` 형태를 권장합니다.

```powershell
cd C:\Users\User\Desktop\content\model
powershell -ExecutionPolicy Bypass -File .\run_pipeline.ps1
```

기존 데이터로 전처리까지만 확인하려면:

```powershell
.\venv\Scripts\python.exe src\pipeline.py --skip-train
```

이미 만든 `Final.csv`와 모델 입력 파일로 학습만 다시 돌리려면:

```powershell
.\venv\Scripts\python.exe src\pipeline.py --skip-preprocess --skip-plot
```

새 데이터를 다시 받으려면:

```powershell
.\venv\Scripts\python.exe src\pipeline.py --fetch all
```

`--fetch water`, `--fetch weather`, `--fetch all`을 선택할 수 있습니다. `--fetch water`는 `src/water_gate.py`와 `src/water_quality.py`를 순서대로 실행합니다. 기상 API는 `.env` 또는 환경변수 `KMA_SERVICE_KEY`가 있으면 `src/weather_api.py`에서 갱신합니다.

---

## Windows / macOS 실행 방법

기본 실행은 `--fetch auto`로 동작합니다. 새로 클론한 뒤 `data/` 폴더가 비어 있어도, Git에 포함된 `water_data/*.xlsx` 파일에서 수문/수질 원천 CSV를 다시 만들고, `WEATHER.csv`가 없으면 날씨 데이터를 생성한 뒤 정해진 최종 컬럼으로 병합합니다.

생성되는 주요 파일은 아래와 같습니다.

```text
data/Final.csv
data/daechung_final_clean_dataset.csv
data/combined_weather_water_10y.csv
data/processed/model_input/algae_model_input.csv
artifacts/
output/
```

### Windows PowerShell

```powershell
git clone <repo-url>
cd <repo-folder>\model
powershell -ExecutionPolicy Bypass -File .\run_pipeline.ps1
```

전처리와 데이터 병합만 확인하려면:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_pipeline.ps1 --skip-train
```

이미 만들어진 `Final.csv`로 학습만 다시 돌리려면:

```powershell
.\venv\Scripts\python.exe src\pipeline.py --skip-preprocess
```

### macOS / Linux

```bash
git clone <repo-url>
cd <repo-folder>/model
bash run_pipeline.sh
```

전처리와 데이터 병합만 확인하려면:

```bash
bash run_pipeline.sh --skip-train
```

이미 만들어진 `Final.csv`로 학습만 다시 돌리려면:

```bash
./venv/bin/python src/pipeline.py --skip-preprocess
```

### 데이터 갱신 옵션

기본값은 `--fetch auto`입니다. 클론 직후에는 이 옵션만으로 로컬 엑셀 재생성, 날씨 생성, 최종 병합을 자동 처리합니다.

```bash
python src/pipeline.py --fetch auto
python src/pipeline.py --fetch water
python src/pipeline.py --fetch weather
python src/pipeline.py --fetch all
```

`--fetch water`와 `--fetch all`은 Selenium으로 수문/수질 데이터를 다시 크롤링합니다. Chrome/ChromeDriver 환경이 필요합니다.

날씨 상세 AWS API까지 강제로 가져오려면 환경변수를 추가합니다. 기본 실행에서는 안정성을 위해 AWS 상세 호출을 건너뛰고 표준 `WEATHER.csv`를 생성합니다.

```bash
export KMA_SERVICE_KEY="your-key"
export KMA_FETCH_AWS=1
python src/pipeline.py --fetch weather
```

PowerShell에서는 아래처럼 설정합니다.

```powershell
$env:KMA_SERVICE_KEY="your-key"
$env:KMA_FETCH_AWS="1"
python src\pipeline.py --fetch weather
```

## ALGAE_FINAL_v2 기준 재생성 / API 원천 재생성

`data/ALGAE_FINAL_v2.csv`는 수동 전처리까지 끝난 기준 최종 데이터입니다. 기본 실행은 이 파일이 있으면 이 파일을 기준으로 `Final.csv`, `daechung_final_clean_dataset.csv`, `combined_weather_water_10y.csv`, `processed/model_input/algae_model_input.csv`를 다시 만듭니다.

```bash
cd <repo-folder>/model
bash run_pipeline.sh --skip-train
```

API와 크롤링 원천 데이터부터 다시 받아서 같은 전처리 규칙으로 재생성하려면 `--fetch` 옵션을 사용합니다. KMA 기상 API 키가 있으면 AWS 상세 기상값도 함께 가져옵니다.

```bash
cd <repo-folder>/model
export KMA_SERVICE_KEY="your-kma-api-key"
export KMA_FETCH_AWS=1
bash run_pipeline.sh --fetch all --skip-train
```

기상 데이터만 다시 만들 때는 아래처럼 실행합니다.

```bash
export KMA_SERVICE_KEY="your-kma-api-key"
export KMA_FETCH_AWS=1
bash run_pipeline.sh --fetch weather --skip-train
```

macOS에서 LightGBM 실행 중 `libomp.dylib` 오류가 나면 OpenMP 런타임이 없는 상태입니다.

```bash
brew install libomp
```

`--fetch water`, `--fetch weather`, `--fetch all`을 쓰면 기준 파일 복사 모드가 아니라 원천 데이터 재생성 모드로 동작합니다. 실행 후 `Final.csv`는 `date + loc_encoded` 중복, 결측치, object 컬럼 검사를 통과해야 합니다.

