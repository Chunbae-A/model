# 모델링 워크플로 상세 분석

이 문서는 현재 모델링 코드 구조, 트리/비트리 워크플로 차이, 산출물 저장 위치, 현재 실행 결과를 정리한다.

## 1. 현재 코드 구조

현재 구조는 너무 잘게 쪼개지 않되, 한 파일에 모든 책임이 몰리지 않도록 아래 정도로만 나누었다.

```text
src/config/model_config.py
src/pipeline/data.py
src/pipeline/models.py
src/pipeline/evaluation.py
src/pipeline/artifacts.py
src/pipeline/runner.py
src/model_pipeline.py
model_training.ipynb
```

역할은 다음과 같다.

| 파일 | 역할 |
| --- | --- |
| `src/config/model_config.py` | 데이터셋 경로, workflow 정의, target, split, 모델 후보, artifact 경로 설정 |
| `src/pipeline/data.py` | 데이터 로드, 필수 컬럼 검사, feature 선택, train/valid split |
| `src/pipeline/models.py` | 트리/비트리 후보 모델 정의 및 학습 |
| `src/pipeline/evaluation.py` | 회귀/분류 metric 계산, threshold 후보 평가, best model 선택 |
| `src/pipeline/artifacts.py` | 예측 결과, feature importance, 모델/지표 저장 |
| `src/pipeline/runner.py` | 단일 workflow 실행 및 트리/비트리 비교 실행 |
| `src/model_pipeline.py` | 기존 노트북 호환용 facade |
| `model_training.ipynb` | 실행 순서와 결과 확인용 노트북 |

## 2. 전체 실행 흐름

단일 workflow 학습 흐름:

```text
1. 데이터 로드
2. train/valid split
3. feature 선택
4. 후보 모델 학습
5. 모델 평가
6. best model 선택
7. 예측 결과 생성
8. feature importance 생성
9. artifact 저장
```

트리/비트리 비교 실행 흐름:

```text
tree workflow 실행
non_tree workflow 실행
두 workflow의 best regression / best classification 결과를 summary로 저장
```

## 3. Workflow 정의

workflow는 `src/config/model_config.py`의 `WORKFLOWS`에 정의되어 있다.

### Tree Workflow

키:

```text
tree
```

입력:

```text
src/data/processed/model_input/tree_gradient_boosting/algae_tree_station_expanded.csv
```

후보 모델:

| task | candidates |
| --- | --- |
| regression | LightGBM, XGBoost, HistGradientBoosting |
| classification | LightGBM, XGBoost, HistGradientBoosting |

의도:

- 원 단위 수질/조류/댐 feature를 유지한다.
- 트리 기반 모델은 feature scale에 둔감하므로 추가 스케일링을 하지 않는다.
- 현재 프로젝트의 기본 기준선이다.

### Non-tree Workflow

키:

```text
non_tree
```

입력:

```text
src/data/processed/model_input/non_tree_scaled/algae_non_tree_scaled_station_expanded.csv
```

후보 모델:

| task | candidates |
| --- | --- |
| regression | Ridge, ElasticNet, SVR-RBF, KNN Regressor |
| classification | Logistic Regression, SVC-RBF, KNN Classifier |

의도:

- 비트리 모델은 값의 크기에 민감하므로 스케일링된 입력을 사용한다.
- `station`, `loc_encoded`는 one-hot 컬럼을 함께 사용한다.
- 조류 세포수/탁도/Chl-a는 `log10(x + 1)` 후 robust scaling되어 있다.

## 4. Artifact 저장 구조

모델이 생성되는 위치는 workflow별로 분리한다.

```text
artifacts/
  workflow_comparison_summary.csv

  tree_gradient_boosting/
    models/
      model_bundle.pkl
      regression_model.pkl
      classification_model.pkl
    metrics/
      regression_metrics.json
      classification_metrics.json
      classification_threshold_candidates.csv
    predictions/
      regression_predictions.csv
      classification_predictions.csv
    explain/
      feature_importance.csv

  non_tree_scaled/
    models/
      model_bundle.pkl
      regression_model.pkl
      classification_model.pkl
    metrics/
      regression_metrics.json
      classification_metrics.json
      classification_threshold_candidates.csv
    predictions/
      regression_predictions.csv
      classification_predictions.csv
    explain/
      feature_importance.csv
```

이렇게 분리한 이유:

- 트리 모델과 비트리 모델의 산출물이 섞이지 않는다.
- 같은 파일명이라도 workflow 폴더가 달라서 비교가 쉽다.
- `workflow_comparison_summary.csv`에서 두 workflow의 best model만 빠르게 비교할 수 있다.

## 5. 현재 실행 결과

명령:

```bash
python - <<'PY'
from src.pipeline.runner import run_workflow_comparison

out = run_workflow_comparison(save=True)
print(out["summary"][["workflow", "selected_for", "model_name", "rmse", "recall", "precision", "f1"]])
PY
```

결과 요약:

| workflow | selected_for | model_name | RMSE | Precision | Recall | F1 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| tree | regression | lightgbm | 0.7339 |  |  |  |
| tree | classification | xgboost |  | 0.9781 | 0.8960 | 0.9352 |
| non_tree | regression | elasticnet | 0.6773 |  |  |  |
| non_tree | classification | logistic_regression |  | 0.9427 | 0.9599 | 0.9512 |

해석:

- 회귀 RMSE 기준으로는 현재 비트리 스케일 workflow의 `ElasticNet`이 더 낮다.
- 분류 Recall/F1 기준으로도 현재 비트리 스케일 workflow의 `Logistic Regression`이 더 높다.
- 다만 현재 데이터는 station-expanded 구조이므로, 이 결과는 date-level holdout 기준의 1차 비교로 봐야 한다.
- 최종 모델 확정 전에는 walk-forward validation 또는 연도별 out-of-time validation으로 안정성을 확인하는 것이 좋다.

## 6. 현재 결과를 볼 때의 주의사항

1. `non_tree_scaled`가 더 좋아 보이지만, 이는 현재 split에서의 결과다.
2. `tree` workflow는 원 단위 feature를 유지하므로 해석이 쉽고, 비선형/상호작용을 잘 잡는 장점이 있다.
3. `non_tree` workflow는 스케일링과 로그 변환 효과가 커서 선형 모델이 안정적으로 작동한다.
4. `target_alert_next`는 `next_log_cells >= log10(1000 + 1)`로 만든 분류 target이다.
5. `next_log_cells`, `target_alert_next`, `split`, `date`는 feature로 넣지 않는다.
6. 최종 보고용 성능은 단일 holdout보다 rolling-origin evaluation으로 보강하는 편이 좋다.

