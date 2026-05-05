# 모델 입력 데이터 구조

이 폴더는 원본 데이터를 보존하면서 모델 종류별 최종 입력 CSV를 분리한다.

## 폴더

- `raw/`: 원본 CSV 복사본. 재현성과 비교를 위한 보존 영역.
- `processed/model_input/tree_gradient_boosting/`: LightGBM, XGBoost, HistGradientBoosting 같은 트리 기반 Gradient Boosting용 입력.
- `processed/model_input/non_tree_scaled/`: 선형모델, SVM, KNN, 신경망, PCA 등 스케일에 민감한 모델용 입력.

## 생성 파일

| 파일 | 행 | 열 | 용도 |
| --- | ---: | ---: | --- |
| `processed/model_input/tree_gradient_boosting/algae_tree_station_expanded.csv` | 6,304 | 59 | 트리 기반 모델용. 원 단위 수질/조류/댐 feature를 유지한다. |
| `processed/model_input/non_tree_scaled/algae_non_tree_scaled_station_expanded.csv` | 6,304 | 66 | 비트리 모델용. 수질/조류/댐 feature까지 로그/스케일링한다. |

비트리 모델용 데이터 생성 과정은 `notebooks/features/build_non_tree_model_input.ipynb`에도 정리되어 있다.

## 공통 주의사항

- 두 파일 모두 `ALGAE_DATA.csv`의 station-expanded 구조를 유지한다.
- 같은 `date + loc_encoded` 수질 행이 `station` 604, 643, 648, 888별로 4번 존재한다.
- 검증 성능 과대평가를 막으려면 반드시 날짜 기준 split을 사용한다.
- 회귀 target은 `next_log_cells`다.
- 분류 target은 `target_alert_next = next_log_cells >= log10(1000 + 1)`로 생성했다.

## 트리 기반 버전

트리 기반 Gradient Boosting은 feature scale에 둔감하므로 원 단위 수질/조류/댐 값을 유지한다. 기상 컬럼은 병합 원본에 들어 있던 스케일링 상태를 그대로 사용한다.

## 비트리 스케일 버전

비트리 모델용 파일은 아래 정책을 적용한다.

- `station`, `loc_encoded`: 원값 보존 + one-hot 컬럼 추가
- 조류 세포수/탁도/Chl-a: `log10(x + 1)` 후 `RobustScaler`
- 강우/유입/방류/정체 지표: `RobustScaler`
- 수온, pH, DO, 수위, 저수량 등 범위가 비교적 제한적인 컬럼: `MinMaxScaler`
- 기상 컬럼: 전처리팀의 Robust/MinMax 의도를 유지하되, 비트리 파일에서는 train split 기준으로 다시 스케일링
- scaler는 `split == train` 행에만 fit하고 valid 행에는 transform만 적용한다.
