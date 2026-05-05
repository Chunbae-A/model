# 데이터셋 설명 및 컬럼 사전

분석 대상 파일:

- `/Users/hywznn/Desktop/model/src/data/ALGAE_DATA.csv`
- `/Users/hywznn/Desktop/model/src/data/daechung_for_merge_v1.csv`

## 1. 전체 관계

`daechung_for_merge_v1.csv`는 대청댐 조류/수질/댐 운영 feature 중심의 기본 학습 테이블이다.

`ALGAE_DATA.csv`는 `daechung_for_merge_v1.csv`의 기본 행에 기상 관측소별 일 단위 기상 feature를 붙인 확장 테이블이다. 기상 관측소가 4개라서 기본 행 하나가 관측소별로 4번 반복된다.

| 항목 | ALGAE_DATA.csv | daechung_for_merge_v1.csv |
| --- | ---: | ---: |
| 행 수 | 6,304 | 1,576 |
| 열 수 | 57 | 35 |
| 날짜 범위 | 2016-04-04 ~ 2025-12-15 | 2016-04-04 ~ 2025-12-15 |
| 고유 날짜 수 | 546 | 546 |
| 결측치 | 0 | 0 |
| 완전 중복 행 | 0 | 0 |
| 지점 수 | `loc_encoded` 3개 | `loc_encoded` 3개 |
| 기상 관측소 수 | `station` 4개 | 없음 |

`ALGAE_DATA.csv`의 행 수는 `daechung_for_merge_v1.csv`의 정확히 4배다.

```text
1,576 rows * 4 stations = 6,304 rows
```

station별 행 수:

| station | 행 수 |
| ---: | ---: |
| 604 | 1,576 |
| 643 | 1,576 |
| 648 | 1,576 |
| 888 | 1,576 |

## 2. 샘플 구조

`daechung_for_merge_v1.csv`의 한 행은 특정 조사일과 특정 지점의 조류/수질/댐 운영 상태를 나타낸다.

예시:

| 조사일 | 수온 | pH | DO | Chl_a | 유해남조류_세포수 | 수위 | 저수량 | loc_encoded | log_target | next_log_cells |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2016-04-04 | 8.8 | 8.2 | 13.3 | 4.5 | 0 | 67.29 | 664.255 | 0 | 0.0 | 0.0 |
| 2016-04-11 | 9.1 | 7.7 | 12.1 | 4.4 | 0 | 67.98 | 696.142 | 0 | 0.0 | 0.0 |

`ALGAE_DATA.csv`는 같은 조사일/지점 행에 `station`별 기상값이 붙는다.

예시:

| date | loc_encoded | station | water_temp | cyano_cells | water_level | avg_temp | daily_rain | rain_7d_sum_y |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2016-04-04 | 0 | 604 | 8.8 | 0 | 67.29 | 0.578288 | 2.0 | 0.107143 |
| 2016-04-04 | 0 | 643 | 8.8 | 0 | 67.29 | 0.586639 | 1.0 | -0.071429 |
| 2016-04-04 | 0 | 648 | 8.8 | 0 | 67.29 | 0.590814 | 0.0 | -0.160714 |
| 2016-04-04 | 0 | 888 | 8.8 | 0 | 67.29 | 0.594990 | 0.0 | -0.160714 |

## 3. daechung_for_merge_v1.csv 컬럼 사전

| 컬럼 | 타입 | 설명 |
| --- | --- | --- |
| `조사일` | object/date | 조류 및 수질 조사 기준일. 모델링 시 날짜형으로 변환해서 시간순 split에 사용한다. |
| `수온` | float | 현장 수온. 조류 증식 조건을 설명하는 핵심 수질 feature다. |
| `pH` | float | 수소이온농도 지수. 수계 상태와 조류 활성 변화와 관련될 수 있다. |
| `DO` | float | 용존산소. 조류 성장, 분해, 수체 상태를 반영하는 수질 feature다. |
| `투명도` | float | 물의 투명도. 탁도 및 조류 농도와 반대 방향으로 움직일 수 있다. |
| `탁도` | float | 물의 탁한 정도. 강우 유입, 부유물질, 조류 상태와 관련된다. |
| `Chl_a` | float | 클로로필-a 농도. 조류 생물량의 대표 proxy다. |
| `유해남조류_세포수` | int | 현재 조사 시점의 유해남조류 총 세포수. 가장 직접적인 현재 상태 feature다. |
| `Microcystis` | int | Microcystis 속 세포수. 유해남조류 구성 중 주요 우점종 후보. |
| `Anabaena` | int | Anabaena 속 세포수. 유해남조류 구성 feature. |
| `Oscillatoria` | int | Oscillatoria 속 세포수. 유해남조류 구성 feature. |
| `Aphanizomenon` | int | Aphanizomenon 속 세포수. 유해남조류 구성 feature. |
| `acc_temp_7d` | float | 최근 7일 누적 수온 또는 수온 누적 지표. 고온 지속 효과를 반영한다. |
| `TSI_Chla` | float | Chl-a 기반 부영양화 지수. 조류량 기반 영양상태 proxy다. |
| `TSI_SD` | float | 투명도 기반 부영양화 지수. 수체 탁도/투명도 기반 영양상태 proxy다. |
| `microcystis_ratio` | float | 총 유해남조류 중 Microcystis 비율. 0~1 범위. |
| `anabaena_ratio` | float | 총 유해남조류 중 Anabaena 비율. 0~1 범위. |
| `oscillatoria_ratio` | float | 총 유해남조류 중 Oscillatoria 비율. 0~1 범위. |
| `aphanizomenon_ratio` | float | 총 유해남조류 중 Aphanizomenon 비율. 0~1 범위. |
| `수위` | float | 대청댐 수위. 저수 상태와 수체 체류 조건을 설명한다. |
| `저수량` | float | 댐 저수량. 수체 규모 및 희석/체류 조건과 관련된다. |
| `저수율` | float | 저수율. 저수량을 비율 형태로 표현한 feature다. |
| `유입량` | float | 유입량. 강우 후 외부 유입 및 영양염류 유입 가능성과 관련된다. |
| `방류량` | float | 방류량. 체류시간과 수체 교체 정도에 영향을 준다. |
| `강우량` | float | 강우량 관련 댐/유역 feature. 비가 온 뒤 유입 효과를 설명한다. |
| `level_change_7d` | float | 최근 7일 수위 변화량. 수문 상태 변화 feature다. |
| `rain_7d_sum` | float | 최근 7일 누적 강우량. 강우 유입 시나리오의 핵심 feature다. |
| `inflow_7d_sum` | float | 최근 7일 누적 유입량. 영양염류 유입 및 수문 변화 proxy다. |
| `outflow_7d_sum` | float | 최근 7일 누적 방류량. 수체 교체 및 체류시간 proxy다. |
| `residence_proxy` | float | 저수량/방류량 등을 이용한 체류시간 proxy. 값이 클수록 정체 가능성이 높다고 해석할 수 있다. |
| `nutrient_stagnation_index` | float | 영양염류 유입과 정체 조건을 합성한 지표로 보인다. 정체성 위험 feature다. |
| `loc_encoded` | int | 조사 지점 인코딩 값. 현재 0, 1, 2 세 지점이 있다. |
| `alert_encoded` | int | 현재 경보 또는 관심 이상 여부로 보이는 이진 라벨. 0/1 값. |
| `log_target` | float | 현재 `유해남조류_세포수`를 `log10(cells + 1)`로 변환한 값. |
| `next_log_cells` | float | 같은 지점의 다음 조사 시점 `log_target`. 다음 시점 세포수 예측용 회귀 target으로 쓸 수 있다. |

## 4. ALGAE_DATA.csv 컬럼 사전

`ALGAE_DATA.csv`의 1~35번째 컬럼은 `daechung_for_merge_v1.csv`의 컬럼을 영어 이름으로 바꾼 뒤, `rain_7d_sum`만 기상 데이터와 이름 충돌을 피하려고 `rain_7d_sum_x`가 된 형태다.

| ALGAE 컬럼 | 대응 daechung 컬럼 | 타입 | 설명 |
| --- | --- | --- | --- |
| `date` | `조사일` | object/date | 조류 및 수질 조사 기준일. |
| `water_temp` | `수온` | float | 현장 수온. |
| `pH` | `pH` | float | 수소이온농도 지수. |
| `DO` | `DO` | float | 용존산소. |
| `transparency` | `투명도` | float | 투명도. |
| `turbidity` | `탁도` | float | 탁도. |
| `Chl_a` | `Chl_a` | float | 클로로필-a 농도. |
| `cyano_cells` | `유해남조류_세포수` | int | 현재 유해남조류 총 세포수. |
| `Microcystis` | `Microcystis` | int | Microcystis 세포수. |
| `Anabaena` | `Anabaena` | int | Anabaena 세포수. |
| `Oscillatoria` | `Oscillatoria` | int | Oscillatoria 세포수. |
| `Aphanizomenon` | `Aphanizomenon` | int | Aphanizomenon 세포수. |
| `acc_temp_7d` | `acc_temp_7d` | float | 최근 7일 누적 수온 지표. |
| `TSI_Chla` | `TSI_Chla` | float | Chl-a 기반 부영양화 지수. |
| `TSI_SD` | `TSI_SD` | float | 투명도 기반 부영양화 지수. |
| `microcystis_ratio` | `microcystis_ratio` | float | Microcystis 구성비. |
| `anabaena_ratio` | `anabaena_ratio` | float | Anabaena 구성비. |
| `oscillatoria_ratio` | `oscillatoria_ratio` | float | Oscillatoria 구성비. |
| `aphanizomenon_ratio` | `aphanizomenon_ratio` | float | Aphanizomenon 구성비. |
| `water_level` | `수위` | float | 댐 수위. |
| `storage_vol` | `저수량` | float | 댐 저수량. |
| `storage_rate` | `저수율` | float | 댐 저수율. |
| `inflow` | `유입량` | float | 유입량. |
| `outflow` | `방류량` | float | 방류량. |
| `rainfall` | `강우량` | float | 댐/유역 강우량 feature. |
| `level_change_7d` | `level_change_7d` | float | 최근 7일 수위 변화량. |
| `rain_7d_sum_x` | `rain_7d_sum` | float | 대청댐 기본 테이블에서 온 최근 7일 누적 강우량. |
| `inflow_7d_sum` | `inflow_7d_sum` | float | 최근 7일 누적 유입량. |
| `outflow_7d_sum` | `outflow_7d_sum` | float | 최근 7일 누적 방류량. |
| `residence_proxy` | `residence_proxy` | float | 체류시간 proxy. |
| `nutrient_stagnation_index` | `nutrient_stagnation_index` | float | 영양염류/정체 조건 합성 지표. |
| `loc_encoded` | `loc_encoded` | int | 조사 지점 인코딩. |
| `alert_encoded` | `alert_encoded` | int | 현재 경보 또는 관심 이상 여부로 보이는 이진 라벨. |
| `log_target` | `log_target` | float | 현재 세포수의 `log10(cells + 1)` 변환값. |
| `next_log_cells` | `next_log_cells` | float | 같은 지점의 다음 조사 시점 로그 세포수. |

`ALGAE_DATA.csv`의 36~57번째 컬럼은 기상 관측소별 기상 feature다.

| 컬럼 | 타입 | 설명 |
| --- | --- | --- |
| `station` | int | 기상 관측소 ID. 현재 604, 643, 648, 888 네 개. |
| `avg_temp` | float | 일 평균기온. 0~1 부근으로 스케일링된 값으로 보인다. |
| `min_temp` | float | 일 최저기온. 스케일링된 값으로 보인다. |
| `max_temp` | float | 일 최고기온. 스케일링된 값으로 보인다. |
| `daily_rain` | float | 일 강수량. 이 컬럼은 원 단위 강수량에 가까우며 최대 419.0까지 나타난다. |
| `max_wind_gust` | float | 최대 순간풍속 또는 최대 풍속 관련 feature. 스케일링된 값으로 보인다. |
| `avg_wind` | float | 평균 풍속. 스케일링된 값으로 보인다. |
| `air_temp_7d_mean` | float | 최근 7일 평균 기온. 스케일링된 값으로 보인다. |
| `hot_days_30c_7d` | float | 최근 7일 중 30도 이상 고온일 수. 0~7 범위. |
| `rain_3d_sum` | float | 최근 3일 누적 강수량 feature. 스케일링 또는 변환된 값으로 보인다. |
| `rain_7d_sum_y` | float | 기상 관측소 데이터에서 온 최근 7일 누적 강수량. `rain_7d_sum_x`와 구분해야 한다. |
| `rain_14d_sum` | float | 최근 14일 누적 강수량 feature. |
| `wind_7d_mean` | float | 최근 7일 평균 풍속. |
| `low_wind_days_2ms_7d` | float | 최근 7일 중 저풍속일 수. 1~7 범위. |
| `sunshine` | float | 일조시간 또는 일조 관련 feature. 스케일링된 값으로 보인다. |
| `solar_rad` | float | 일사량 feature. 스케일링된 값으로 보인다. |
| `cloud_cover` | float | 운량. 0~1 범위. |
| `sunshine_7d_sum` | float | 최근 7일 누적 일조 feature. |
| `solar_7d_sum` | float | 최근 7일 누적 일사 feature. |
| `cloud_7d_mean` | float | 최근 7일 평균 운량. |
| `sin_season` | float | 계절성을 sin 주기로 인코딩한 값. |
| `cos_season` | float | 계절성을 cos 주기로 인코딩한 값. |

## 5. 모델링 관점에서의 권장 사용

### daechung_for_merge_v1.csv

기본 모델 학습에는 이 파일이 더 깔끔하다. 한 조사일/지점당 한 행이므로 중복 확대 문제가 없다.

추천 target:

- 회귀: `next_log_cells`
- 분류: `next_log_cells >= log10(1000 + 1)`로 새 target 생성

주의:

- `log_target`은 현재 세포수 기반 feature로 쓸 수 있지만, 현재 세포수를 이미 `유해남조류_세포수`로 쓰는 경우 정보가 중복된다.
- `next_log_cells`는 미래 target이므로 feature로 넣으면 데이터 누수다.
- `alert_encoded`가 현재 경보인지 다음 경보인지 의미를 명확히 해야 한다.

### ALGAE_DATA.csv

기상 feature까지 함께 쓰려면 이 파일을 사용할 수 있다. 다만 station별로 같은 조류/댐 행이 4번 반복되므로 모델링 전에 전략을 정해야 한다.

가능한 전략:

1. 특정 대표 관측소 하나만 선택한다.
2. 4개 관측소 기상값을 날짜별로 평균/최대/최소 집계해서 한 행으로 줄인다.
3. `station`을 feature로 포함하고 station별 행을 모두 쓰되, train/validation split을 날짜 기준으로 엄격히 묶는다.

가장 안전한 기본 전략은 2번이다. 그렇지 않으면 같은 조사 결과가 station만 바뀐 채 여러 번 들어가서 검증 성능이 과대평가될 수 있다.

