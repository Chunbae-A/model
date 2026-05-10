# EDA 작업 공간

이 폴더는 `ALGAE_DATA.csv`를 비롯한 모델 입력 데이터의 탐색적 데이터 분석 결과를 관리한다.

## 폴더 구조

```text
eda/
  README.md
  figures/    EDA 시각화 이미지 저장
  tables/     EDA 요약 CSV/Markdown 저장
  scripts/    재사용 가능한 EDA 스크립트 저장
```

## 기본 데이터

주요 분석 대상:

```text
src/data/team-raw/ALGAE_DATA.csv
```

이 데이터는 수질·조류·댐 운영 데이터와 station별 기상 데이터가 결합된 최종 병합 데이터다.

## 앞으로의 EDA 진행 순서

1. 전체 구조와 결측 확인
2. target 분포 확인
3. 수질 피처 분포 시각화
4. 조류 세포수와 종별 분포 시각화
5. 댐·수문 피처 분포와 이상치 확인
6. 기상 피처 분포 확인
7. 위치별·station별 차이 확인
8. 주요 피처와 target 관계 확인
9. 상관관계 및 SHAP과 비교

## 산출물 이름 규칙

가능하면 아래처럼 저장한다.

```text
eda/figures/01_target_distribution.png
eda/tables/01_target_summary.csv
eda/scripts/01_make_target_distribution.py
```

## 현재 생성된 산출물

상세 목록:

[EDA_INDEX.md](EDA_INDEX.md)

재생성 명령어:

```bash
python eda/scripts/01_algae_data_eda.py
```
