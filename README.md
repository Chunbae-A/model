# model
대청호 조류경보 예측을 위한 수질·수문·기상 기반 AI 모델링 레포지토리

## 모델 개요

본 레포지토리는 대청호 조류경보 예측을 위한 모델 학습 코드를 관리합니다.

핵심 방향은 **PH-GBM**, 즉 수질·수문·기상 데이터를 채수일 기준으로 통합하고 도메인 지식 기반 피처를 생성한 뒤 Gradient Boosting 계열 모델로 다음 채수일 유해남조류 세포수와 관심 이상 확률을 예측하는 구조입니다.

## 주요 피처

| 구분 | 피처 예시 | 의미 |
| --- | --- | --- |
| 수질 | `tsi_chla`, `tsi_transparency`, `tsi_proxy_mean` | Chl-a와 투명도 기반 TSI proxy |
| 수질 | `temp_light_growth_proxy` | 수온과 일조시간을 결합한 광합성 성장 조건 |
| 수문 | `net_flow` | 유입량 - 방류량 |
| 수문 | `residence_proxy` | 저수량 / 방류량 기반 체류시간 proxy |
| 수문 | `level_change_3d`, `level_change_7d` | 최근 수위 변화 |
| 기상 | `rain_3d_sum`, `rain_7d_sum`, `rain_14d_sum` | 강우 누적 및 lag 효과 |
| 기상 | `hot_days_30c_7d`, `low_wind_days_2ms_7d` | 고온·저풍속 지속 조건 |
| 공간 | `graph_decay_signal` | 회남 -> 추동 -> 문의 상류 전파 신호 |

주의: 현재 데이터에 TP가 없으므로 TSI는 완전한 Carlson TSI가 아니라 `Chl-a`와 `투명도`를 활용한 proxy입니다.

## 레포 구조

```text
src/algae_model/
  config.py      # 데이터 경로와 학습 설정
  features.py    # 수질·수문·기상 feature engineering
  train.py       # 모델 학습, 평가, 저장

scripts/
  train_model.py # 학습 실행 엔트리포인트
```

## 설치

```bash
python -m pip install -r requirements.txt
```

## 실행 방법

데이터 파일은 레포에 포함하지 않습니다. 로컬 데이터 경로를 환경변수로 넘겨 실행합니다.

```bash
export QUALITY_CSV="/path/to/daecheong_algae_monitoring_daily.csv"
export DAM_CSV="/path/to/daecheong_dam_operations_daily.csv"
export KMA_CSV="/path/to/OBS_ASOS_TIM_20260424114201.csv"
export GEUM_DAM_CSV="/path/to/dmList_2016-01-01_2026-04-25_0000.csv"
export MODEL_OUTPUT_DIR="outputs/model_pipeline"

PYTHONPATH=src python scripts/train_model.py
```

필수 입력은 `QUALITY_CSV`, `DAM_CSV`입니다. `KMA_CSV`, `GEUM_DAM_CSV`는 있으면 자동으로 추가 피처를 생성합니다.

## 산출물

```text
outputs/model_pipeline/
  manifest.json
  models/
    hist_gradient_boosting_regressor.joblib
    hist_gradient_boosting_classifier.joblib
    random_forest_classifier.joblib
  tables/
    master_table.csv
    feature_list.csv
    metrics.csv
    predictions.csv
```
