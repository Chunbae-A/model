# model

대청댐 조류경보 예측 AI 프로젝트의 **모델링 코드 전용 레포지토리**입니다.

이 레포지토리는 전처리팀이 만든 최종 모델 입력 테이블을 받아서 Gradient Boosting 기반 모델을 학습, 평가, 저장, 추론하고, 다음 에이전트가 사용할 수 있는 구조화 결과 파일을 생성하는 코드만 포함합니다.

## 핵심 파일

```text
model_training.ipynb   # 모델 학습/평가/저장/추론 노트북
requirements.txt       # 실행 패키지
```

## 입력 데이터 가정

노트북은 아래 최종 모델 입력 테이블이 이미 준비되어 있다고 가정합니다.

```text
data/processed/model_input/algae_model_input.csv
```

선택적으로 아래 컬럼 정의서를 함께 사용할 수 있습니다.

```text
data/processed/model_input/model_input_schema.csv
```

권장 schema 형식:

```text
column_name,role,available_at_prediction,description
rain_7d_sum,feature,yes,최근 7일 누적강수량
target_alert_t_plus_7,target,no,7일 뒤 관심 이상 여부
sample_date,id,yes,기준 날짜
split,meta,yes,학습/검증 구분
```

이 테이블에는 다음 작업이 모두 완료되어 있어야 합니다.

- 수질·기상·댐 운영 데이터 정리 및 병합
- 결측치/이상치 처리
- lag feature 생성
- rolling feature 생성
- 누적 강수량 feature 생성
- 변화량 feature 생성
- 계절성 feature 생성
- hydro-topology feature 생성
- target label 생성

따라서 이 레포지토리에서는 전처리나 feature engineering 코드를 작성하지 않습니다.

## 포함하는 코드

- 최종 모델 입력 테이블 로드
- feature column / target column 분리
- schema 기반 feature whitelist
- target/future/next/t_plus 등 누수 의심 컬럼 자동 차단
- 시간 기준 train / validation split
- Gradient Boosting 회귀 모델 학습
- Gradient Boosting 분류 모델 학습
- 모델 평가 지표 계산
- classification threshold 후보 탐색
- 모델 저장 및 로드
- 추론 함수
- 조류경보 운영 보조 로직
- SHAP 또는 feature importance 기반 설명 함수
- 시나리오 기반 후처리 결과 생성

## 포함하지 않는 코드

- 원천 데이터 전처리
- 데이터 병합
- 결측치/이상치 처리
- target 생성
- lag / rolling / 누적강수 / 변화량 / 계절성 feature 생성
- hydro-topology feature 생성
- Granger Causality / VAR 분석
- UI, API, DB, Docker, 배포 코드

## 모델 방향

최종 예측 모델은 Gradient Boosting 계열을 기본으로 합니다.

비교 후보:

1. LightGBM
2. XGBoost
3. HistGradientBoosting

노트북은 세 후보 모델을 가능한 범위에서 모두 학습하고, validation 성능표를 만든 뒤 회귀/분류 각각의 best model을 선택합니다.

- 회귀 모델 선택 기준 기본값: `RMSE` 최소
- 분류 모델 선택 기준 기본값: `Recall` 최대

LightGBM 또는 XGBoost가 설치되어 있지 않으면 해당 후보만 건너뛰고, sklearn의 `HistGradientBoosting`은 기본 후보로 사용합니다.

저장 대상:

- 후보 모델별 validation 성능표
- threshold 후보별 Precision / Recall / F1 표
- 후보 회귀 모델 전체
- 후보 분류 모델 전체
- 회귀 best model alias
- 분류 best model alias
- 후보 모델별 feature importance table
- 후보 모델별 SHAP summary table 생성 함수

## 예측 출력

1. 회귀 출력
   - 7일 뒤 또는 다음 채수일 유해남조류 세포수 예측
   - 기본 가정은 `log1p(cells)` target을 학습하고 `expm1`으로 원 단위 복원
   - 전처리팀이 `log10(cells + 1)`을 썼다면 config의 `REGRESSION_TARGET_TRANSFORM`을 수정

2. 분류 출력
   - 관심 이상 확률 예측
   - 예: `P(유해남조류 세포수 >= 1,000)`
   - 기본 threshold는 0.5이지만 validation 기준으로 후보 threshold를 비교할 수 있음
   - 조류경보 예측에서는 Accuracy보다 Recall, Precision, F1-score를 우선 확인

3. 설명 출력
   - 후보 모델별 SHAP summary table 또는 feature importance table
   - 날짜·지점별 위험 예측에 영향을 준 상위 feature를 모델별로 비교

4. 시나리오 출력
   - 예측 세포수, 관심 이상 확률, SHAP 상위 feature를 바탕으로 위험 유형을 분류
   - 공식 조류경보 발령이나 행정 명령이 아니라 다음 에이전트가 활용할 구조화 데이터입니다.

## 최종 저장 파일

노트북의 실행 예시는 아래 파일을 생성하도록 구성되어 있습니다.

```text
artifacts/predictions/regression_predictions.csv
artifacts/predictions/classification_predictions.csv
artifacts/metrics/regression_metrics.json
artifacts/metrics/classification_metrics.json
artifacts/metrics/classification_threshold_candidates.csv
artifacts/explain/feature_importance.csv
artifacts/explain/shap_top_reasons.csv
artifacts/scenario/scenario_results.csv
artifacts/scenario/scenario_input_for_llm.json
artifacts/models/regression_model.pkl
artifacts/models/classification_model.pkl
artifacts/models/preprocessing_pipeline.pkl
```

`preprocessing_pipeline.pkl`은 전처리팀이 fitted pipeline을 제공하면 함께 저장할 수 있도록 둔 자리입니다. 모델링 노트북은 전처리 pipeline을 새로 만들지 않습니다.

## 시나리오 후처리

시나리오 후처리는 별도 예측 모델이 아니라 rule-based post-processing입니다.

사용 입력:

- `predicted_cells`
- `alert_probability`
- `predicted_alert_label`
- `shap_top_1`, `shap_top_2`, `shap_top_3`

기본 시나리오:

- 수체 정체·성층 위험 시나리오
- 강우 이후 영양염류 유입 시나리오
- 고온·고일사 성장 촉진 시나리오
- 과거 증식 추세 지속 시나리오
- 복합 고위험 시나리오
- 관찰 강화 시나리오
- 일반 안정 시나리오

`HIGH_RISK_THRESHOLD`, `MEDIUM_RISK_THRESHOLD`, 시나리오별 feature group은 노트북 상단 config에서 수정합니다.

## 실행 방법

```bash
python -m pip install -r requirements.txt
jupyter notebook model_training.ipynb
```

노트북 상단의 config에서 실제 전처리 산출물에 맞게 아래 값을 수정합니다.

- `MODEL_INPUT_PATH`
- `SCHEMA_PATH`
- `DATE_COLUMN`
- `SITE_COLUMN`
- `REGRESSION_TARGET`
- `CLASSIFICATION_TARGET`
- `DROP_COLUMNS`
- `PROBABILITY_THRESHOLD`
- `ALERT_CELL_THRESHOLD`
- `REGRESSION_TARGET_TRANSFORM`
- `THRESHOLD_CANDIDATES`
- `HIGH_RISK_THRESHOLD`
- `MEDIUM_RISK_THRESHOLD`
- `SCENARIO_FEATURE_GROUPS`

## 주의사항

- 실제 성능 수치와 SHAP 결과는 데이터를 실행한 뒤 확인해야 합니다.
- 이 레포지토리에서는 임의의 성능 결과나 p-value를 작성하지 않습니다.
- threshold는 validation 기준으로만 탐색해야 하며, test 데이터에 맞춰 조정하면 성능이 과대평가될 수 있습니다.
- Granger Causality / VAR는 최종 모델이 아니며, 요청이 있을 때 별도 사전 분석 모듈로만 작성합니다.
- 가장 안전한 방식은 전처리팀이 `model_input_schema.csv`에 `role=feature`로 승인한 컬럼만 모델 입력으로 사용하는 것입니다.
- schema가 없을 경우 노트북은 `DROP_COLUMNS` 기반으로 feature 후보를 만들고, 누수 의심 키워드가 포함된 컬럼을 자동 차단합니다.
- 시나리오 결과는 대응 유형을 구조화할 뿐, 방류량 조정처럼 구체적인 운영 명령을 자동 생성하지 않습니다.
