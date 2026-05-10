# ALGAE_DATA EDA 산출물

그래프별 해석:

[figure_interpretation.md](figure_interpretation.md)

## 생성된 시각화

| 파일 | 내용 |
| --- | --- |
| `figures/00_dataset_structure.png` | 데이터 구조, station/위치/조사일 행 수 |
| `figures/01_target_distribution.png` | target 및 세포수 분포 |
| `figures/02_target_time_series_by_location.png` | 위치별 조류 로그값 시간 변화 |
| `figures/03_water_quality_boxplot_by_location.png` | 위치별 수질 피처 분포 |
| `figures/04_algae_species_summary.png` | 종별 누적 세포수와 0값 비율 |
| `figures/05_hydrology_log_boxplot.png` | 수문 피처 log 분포 |
| `figures/06_weather_boxplot_by_station.png` | station별 기상 피처 분포 |
| `figures/07_feature_correlation_heatmap.png` | 주요 피처 상관관계 |

## 생성된 표

| 파일 | 내용 |
| --- | --- |
| `tables/00_dataset_structure.csv` | 데이터 구조 요약 |
| `tables/01_numeric_feature_summary.csv` | 수치형 피처 요약 통계 |
| `tables/02_corr_with_next_log_cells.csv` | 다음 세포수 로그값과의 상관계수 |
