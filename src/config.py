from pathlib import Path
import numpy as np

# ============================================================
# 경로 설정
# ============================================================
MODEL_INPUT_PATH = Path("data/ALGAE_FINAL.csv")
OUTPUT_DIR = Path("artifacts/models")
PREDICTION_DIR = Path("artifacts/predictions")
METRIC_DIR = Path("artifacts/metrics")
EXPLAIN_DIR = Path("artifacts/explain")
SCENARIO_DIR = Path("artifacts/scenario")

# 최종 산출물 파일명입니다.
REGRESSION_PREDICTION_FILE = "regression_predictions.csv"
CLASSIFICATION_PREDICTION_FILE = "classification_predictions.csv"
REGRESSION_METRIC_FILE = "regression_metrics.json"
CLASSIFICATION_METRIC_FILE = "classification_metrics.json"
FEATURE_IMPORTANCE_FILE = "feature_importance.csv"
SHAP_TOP_REASONS_FILE = "shap_top_reasons.csv"
SCENARIO_RESULT_FILE = "scenario_results.csv"
SCENARIO_LLM_INPUT_FILE = "scenario_input_for_llm.json"
REGRESSION_MODEL_FILE = "regression_model.pkl"
CLASSIFICATION_MODEL_FILE = "classification_model.pkl"
PREPROCESSING_PIPELINE_FILE = "preprocessing_pipeline.pkl"

# ============================================================
# 컬럼 설정
# ============================================================
DATE_COLUMN = "date"      
SITE_COLUMN = "station"
STATION_IDS = ["문의(청남대)", "추동(세천)", "회남(옥천)"]
SPLIT_COLUMN = "split"
FORECAST_HORIZON = "t_plus_7"                          
REGRESSION_TARGET = "log_target"  
CLASSIFICATION_TARGET = "alert_encoded"  

# 모델 입력에서 제외할 컬럼입니다.
# 정답 컬럼, 식별자, 날짜, 분할/관리 컬럼, 미래 정답, 운영 라벨은 입력 변수로 쓰면 데이터 누수가 생길 수 있습니다.
# 가장 안전한 방식은 아래 제외 컬럼 목록보다 schema 기반 허용 목록을 우선 사용하는 것입니다.
DROP_COLUMNS = [
    DATE_COLUMN,
    SITE_COLUMN,
    SPLIT_COLUMN,
    REGRESSION_TARGET,
    CLASSIFICATION_TARGET,
    
    # 🚨 [데이터 누수 방지] 조류 관련 직접 지표 및 파생/미래 타겟 제거
    "Chl_a", "Microcystis", "Anabaena", "Oscillatoria", "Aphanizomenon",
    "TSI_Chla", "TSI_SD", "microcystis_ratio", "anabaena_ratio", 
    "oscillatoria_ratio", "aphanizomenon_ratio",
    "cyano_cells", "next_log_cells",

    # 노이즈
    "cloud_7d_mean", 
    "nutrient_stagnation_index", 
    "avg_wind", 
    "low_wind_days_2ms_7d",
    "hot_days_30c_7d"
]

STATION_DROP_COLUMNS = {
    # 1. 회남: 에어백(강수, 유입량)은 살려두고, 저수율/바람만 쳐냄
    "회남(옥천)": DROP_COLUMNS + [
        "storage_rate", "wind_7d_mean"
    ],
    
    # 2. 추동: 호수 안쪽이라 바람, 최저기온, 유입량이 쓸모없음
    "추동(세천)": DROP_COLUMNS + [
        "daily_rain", "rain_3d_sum", "inflow", # 기존 공통 노이즈였던 것들
        "inflow_7d_sum", "min_temp", "max_wind_gust"
    ],
    
    # 3. 문의: 댐 앞 깊은 곳이라 저수율, 방류량, 비의 영향이 미미함
    "문의(청남대)": DROP_COLUMNS + [
        "daily_rain", "rain_3d_sum", "inflow", # 기존 공통 노이즈였던 것들
        "storage_rate", "min_temp", "outflow_7d_sum"
    ]
}

# ============================================================
# 입력 변수 안전 설정
# ============================================================
SCHEMA_COLUMN_NAME = "column_name"
SCHEMA_ROLE_NAME = "role"
FEATURE_ROLE_VALUE = "feature"

FORBIDDEN_FEATURE_KEYWORDS = [
    "target",
    "future",
    "next",
    "t_plus",
    "label",
    "answer",
    "ground_truth",
]

# 문자열/범주형 입력 변수는 전처리팀이 미리 인코딩했다고 가정합니다.
# 따라서 모델링 노트북에서는 입력 변수가 숫자형인지 검사만 합니다.
REQUIRE_NUMERIC_FEATURES = True

# 비교할 그래디언트 부스팅 후보 모델입니다.
# "auto"는 설치된 LightGBM, XGBoost, sklearn HistGradientBoosting을 모두 비교합니다.
# 특정 모델만 비교하고 싶으면 후보 모델 목록을 직접 수정합니다.
REGRESSION_MODEL_CANDIDATES = ["lightgbm", "xgboost", "hist_gradient_boosting", "randomforest", "catboost"]
CLASSIFICATION_MODEL_CANDIDATES = ["lightgbm", "xgboost", "hist_gradient_boosting", "randomforest", "catboost"]
MODEL_SELECTION_METRIC_REGRESSION = "rmse"      # 낮을수록 좋음
MODEL_SELECTION_METRIC_CLASSIFICATION = "recall"  # 조류경보는 미탐 방지를 위해 recall 우선

# ============================================================
# 모델링 설정
# ============================================================
RANDOM_STATE = 42
VALID_SIZE_RATIO = 0.2
PROBABILITY_THRESHOLD = 0.5  # 기본 threshold. validation 기반 후보 탐색 후 조정 가능
ALERT_CELL_THRESHOLD = 1000  # 수정 필요: 공식 기준 확인 후 config에서 확정

# 회귀 정답값 변환 방식입니다.
# 과제 기본 가정은 log1p(세포수)로 학습한 뒤 expm1로 원 단위를 복원하는 방식입니다.
# 전처리팀이 log10(세포수 + 1)을 사용했다면 "log10_plus_1"로 바꾸세요.
REGRESSION_TARGET_TRANSFORM = "log10_plus_1"  # one of: "log1p", "log10_plus_1", "none"

# 기준값 후보 탐색 설정입니다.
# 주의: 최종 운영 기준값은 별도 검증 기간에서 정하고, 테스트 성능 부풀리기에 사용하면 안 됩니다.
THRESHOLD_CANDIDATES = [round(x, 2) for x in np.arange(0.05, 0.81, 0.05)]
MIN_PRECISION_FOR_THRESHOLD = 0.30

# ============================================================
# 조류경보제 기준 및 원인별 대응 시나리오 통합 설정
# ============================================================
ALERT_ATTENTION = 1000      # 관심 단계
ALERT_WARNING = 10000       # 경계 단계
ALERT_OUTBREAK = 1000000    # 대발생 단계

# 원인 분석을 위한 변수 그룹 (유저님이 만든 파생변수 반영)
SCENARIO_FEATURE_GROUPS = {
    "stagnation": ["residence_proxy", "outflow", "outflow_7d", "low_wind_days", "wind_mean_7d", "storage", "water_level"],
    "rainfall_inflow": ["daily_rain", "rain_3d_sum", "rain_7d_sum", "rain_14d_sum", "inflow", "inflow_7d_sum"],
    "growth_condition": ["water_temp", "acc_temp_7d", "air_temp_7d_mean", "hot_days_30c_7d", "solar_rad", "sunshine", "solar_7d_sum", "turbidity"],
    "past_bloom": ["current_cells", "cyano_cells", "prev_cells"],
    "seasonal": ["sin_season", "cos_season"] 
}

# 기본 단계별 대응 + 원인별 추가 액션
BASE_ACTION = {
    "대발생": "관계기관 긴급 전파 및 취수구 이동 검토",
    "경계": "수질 모니터링 주 2회 이상 강화",
    "관심": "수질 모니터링 주 1회 강화",
    "정상": "일반 관찰 유지"
}

SPECIFIC_ISSUE_ACTION = {
    "stagnation": ("수체 정체에 의한 조류 번식", "물순환설비(수류발생장치 등) 가동 검토"),
    "rainfall_inflow": ("강우 및 영양염류 유입", "상류 오염원 단속 및 현장 예찰"),
    "growth_condition": ("고온 및 일사량 증가", "표층 수온 모니터링 및 정수처리 주의"),
    "past_bloom": ("과거 증식 추세 지속", "현장 채수 확인"),
    "seasonal": ("계절적 요인", "정기 모니터링 유지")
}