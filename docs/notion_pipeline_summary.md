# 모델 파이프라인 & 날씨·수문 통합 자동화 (v1)

## 개요
- 목적: 기상청(KMA) + KMA AWS + 수질/수문 데이터를 자동으로 수집·병합하여 모델 입력 파일을 생성합니다.
- 범위(v1):
  - ASOS(대전(133), 보은(226), 청주(131))에 대한 KMA 연도별 이력 수집
  - AWS(청남대 888, 장동 648, 세천 643, 옥천 604) 월별 데이터 병합(ASOS 매핑)
  - 수문/수질 CSV 병합 및 결측 규칙 적용
  - 간단한 파생변수(rolling 평균/합) 생성

## 현재 구조 (파일)
- `src/weather_api.py`: KMA + AWS 수집, 병합, 결측 처리, 파생변수 생성
- `src/fetch_10y.py`: KMA 연단위 fetch 유틸
- `run_pipeline.sh`: Unix/WSL 실행 스크립트 (v1)
- `run_pipeline.ps1`: Windows PowerShell 실행 스크립트 (v1)
- `.env`: KMA API 키 저장(로컬 비공개)
- 출력: `data/weather_{ASOS}_10y.csv`, `data/combined_weather_water_10y.csv`

## 실행 가이드 (Notion에 붙여넣기용)
1. `.env`에 `KMA_SERVICE_KEY` 설정
2. 가상환경 생성 및 활성화

```bash
python -m venv .venv
# Unix / WSL
source .venv/bin/activate
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

3. 의존성 설치

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. 전체 파이프라인 실행

- Unix/WSL/Git Bash

```bash
./run_pipeline.sh
```

- Windows PowerShell

```powershell
.\run_pipeline.ps1
```

## 핵심 동작 요약
- KMA: 연 단위로 쪼개어 `fetch_kma_range()`를 호출해 과거 데이터를 안전하게 수집
- AWS: 월별 API를 호출하며 5회 재시도(backoff) 적용. AWS→ASOS 매핑 후 평균 집계
- 병합: 수문 CSV 컬럼 충돌을 방지하기 위해 `water_` 접두사로 rename
- 결측: 강수량은 기본 0으로 채우고, 기온/풍속은 station별 선형보간 후 전후채움
- 파생변수: 7일 평균, 3/7/14일 합계 등

## 남은 작업(권고)
- AWS 컬럼 필터링(사용자 지정 필드만 보존) 적용
- 스케일링 단계(Scalers) 및 학습 파이프라인 연동
- 예외 로그 및 재실행 안내서(운영용)

## 변경 이력
- v1: `weather_api.py` 작성 — KMA/AWS 수집, 병합, reindex, impute, feature 생성
- run scripts 추가 (Unix/Windows)

## 예시 출력 위치
- 종합 CSV: `data/combined_weather_water_10y.csv`
- ASOS별 파일: `data/weather_133_10y.csv`, `data/weather_226_10y.csv`, `data/weather_131_10y.csv`

---

*작성일: 2026-05-11*
