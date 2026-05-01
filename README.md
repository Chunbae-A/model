# model

대청호 조류경보 예측을 위한 **모델 학습 노트북**입니다.

## 파일 구성

```text
model_training.ipynb   # 모델 학습 코드 전체
requirements.txt       # 실행 패키지
```

데이터와 학습 산출물은 레포에 올리지 않습니다.

## 실행 방법

```bash
python -m pip install -r requirements.txt
jupyter notebook model_training.ipynb
```

노트북 상단의 데이터 경로만 본인 환경에 맞게 수정하면 됩니다.

기본 입력 데이터:

- K-water 수질/조류 CSV
- K-water 댐 운영 CSV
- KMA ASOS CSV
- 금강 홍수통제소 대청댐 수문 CSV

## 모델 개요

수질·수문·기상 데이터를 채수일 기준으로 통합하고, 도메인 지식 기반 파생변수를 생성한 뒤 Gradient Boosting 계열 모델로 다음 채수일의 유해남조류 세포수와 관심 이상 확률을 예측합니다.

주요 파생 피처:

- `tsi_chla`, `tsi_transparency`, `tsi_proxy_mean`
- `net_flow`
- `residence_proxy`
- `level_change_3d`, `level_change_7d`
- 강우·일조·풍속 rolling feature
- `graph_decay_signal`

주의: 현재 데이터에 TP가 없으므로 TSI는 완전한 Carlson TSI가 아니라 `Chl-a`와 `투명도` 기반 proxy입니다.

## 산출물

노트북 실행 시 아래 폴더에 결과가 생성됩니다.

```text
outputs/model_pipeline/
  models/
  tables/
    master_table.csv
    feature_list.csv
    metrics.csv
    predictions.csv
```
