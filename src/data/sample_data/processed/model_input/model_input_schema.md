# 모델 입력 Schema 설명

`model_input_schema.csv`는 최종 모델 입력 테이블의 컬럼 사용 설명서입니다.

모델 코드는 이 파일에서 `role == "feature"`인 컬럼만 학습 feature로 사용합니다. 이를 통해 target 컬럼이나 미래 정보 컬럼이 실수로 모델 입력에 들어가는 데이터 누수를 방지합니다.

## 권장 경로

```text
data/processed/model_input/model_input_schema.csv
```

## 예시 파일

```text
examples/model_input_schema_example.csv
```

## 컬럼 설명

| 컬럼                      | 의미                                         |
| ------------------------- | -------------------------------------------- |
| `column_name`             | 최종 모델 입력 테이블의 컬럼명               |
| `role`                    | 컬럼 역할: `feature`, `target`, `id`, `meta` |
| `available_at_prediction` | 실제 예측 시점에 알 수 있는 값인지 여부      |
| `description`             | 컬럼 설명                                    |

## role 기준

| role      | 의미                    | 모델 feature 사용 여부 |
| --------- | ----------------------- | ---------------------- |
| `feature` | 모델 입력 변수          | 사용                   |
| `target`  | 모델이 맞춰야 하는 정답 | 사용 금지              |
| `id`      | 날짜, 지점 등 식별자    | 기본 사용 금지         |
| `meta`    | split 등 관리용 컬럼    | 사용 금지              |

## 작성 예시

```csv
column_name,role,available_at_prediction,description
rain_7d_sum,feature,yes,최근 7일 누적강수량
target_alert_t_plus_7,target,no,7일 뒤 관심 이상 여부
sample_date,id,yes,기준 날짜
split,meta,yes,학습/검증 구분
```

## 주의사항

- `target`, `future`, `next`, `t_plus`, `label` 의미가 있는 컬럼은 feature로 지정하지 않습니다.
- 현재 시점에 알 수 있는 값만 `feature`로 지정합니다.
- 7일 뒤 실제 세포수, 7일 뒤 발령 여부, 미래 채수 결과는 `target` 또는 제외 대상입니다.
