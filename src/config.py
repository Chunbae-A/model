from __future__ import annotations

from pathlib import Path
import numpy as np

# Paths (all data assumed under /data)
MODEL_INPUT_PATH = Path("data/processed/model_input/algae_model_input.csv")
MODEL_CONFIG_PATH = Path("config/model_config.yaml")
OUTPUT_DIR = Path("artifacts/models")
PREDICTION_DIR = Path("artifacts/predictions")
METRIC_DIR = Path("artifacts/metrics")
EXPLAIN_DIR = Path("artifacts/explain")
SCENARIO_DIR = Path("artifacts/scenario")

# Filenames
REGRESSION_MODEL_FILE = "regression_model.pkl"
CLASSIFICATION_MODEL_FILE = "classification_model.pkl"
PREPROCESSING_PIPELINE_FILE = "preprocessing_pipeline.pkl"
REGRESSION_PREDICTION_FILE = "regression_predictions.csv"
CLASSIFICATION_PREDICTION_FILE = "classification_predictions.csv"
REGRESSION_METRIC_FILE = "regression_metrics.json"
CLASSIFICATION_METRIC_FILE = "classification_metrics.json"
FEATURE_IMPORTANCE_FILE = "feature_importance.csv"
SHAP_TOP_REASONS_FILE = "shap_top_reasons.csv"
SCENARIO_RESULT_FILE = "scenario_results.csv"
SCENARIO_LLM_INPUT_FILE = "scenario_input_for_llm.json"

# Columns
DATE_COLUMN = "date"
SITE_COLUMN = "loc_encoded"
SPLIT_COLUMN = "split"
FORECAST_HORIZON = "t_plus_7"

REGRESSION_TARGET = "next_log_cells"
CLASSIFICATION_TARGET = "alert_encoded"
CELL_COUNT_COLUMN = "cyano_cells"
LOG_TARGET_COLUMN = "log_target"

# User-confirmed location coding for Daecheong Dam samples.
# Keep both loc_encoded and loc_flow_order as numeric model features so the
# model can learn spatial differences between Munui, Hoenam, and Chudong.
LOCATION_NAME_BY_CODE = {
    0: "문의",
    1: "추동",
    2: "회남",
}
LOCATION_FLOW_ORDER = {
    0: 0,  # 문의
    1: 1,  # 추동
    2: 2,  # 회남
}

# Drop and safety rules
DROP_COLUMNS = [
    DATE_COLUMN,
    SPLIT_COLUMN,
    REGRESSION_TARGET,
    CLASSIFICATION_TARGET,
    LOG_TARGET_COLUMN,
    "next_alert_binary",
    "operational_alert_target",
    "next_sample_available",
]

FORBIDDEN_FEATURE_KEYWORDS = [
    "target",
    "future",
    "next",
    "t_plus",
    "label",
    "answer",
    "ground_truth",
]

REQUIRE_NUMERIC_FEATURES = True

MODEL_SELECTION_METRIC_REGRESSION = "rmse"
MODEL_SELECTION_METRIC_CLASSIFICATION = "recall"

RANDOM_STATE = 42
VALID_SIZE_RATIO = 0.2
PROBABILITY_THRESHOLD = 0.5
ALERT_CELL_THRESHOLD = 1000
WATCH_CELL_THRESHOLD = 1000
WARNING_CELL_THRESHOLD = 10000
BLOOM_CELL_THRESHOLD = 1000000

REGRESSION_TARGET_TRANSFORM = "log10_plus_1"

THRESHOLD_CANDIDATES = [round(x, 2) for x in np.arange(0.10, 0.91, 0.05)]
MIN_PRECISION_FOR_THRESHOLD = 0.30

HIGH_RISK_THRESHOLD = 0.80
MEDIUM_RISK_THRESHOLD = 0.40

SCENARIO_FEATURE_GROUPS = {
    "stagnation": ["residence_proxy", "outflow", "outflow_7d", "low_wind_days", "wind_7d_mean", "storage", "방류량", "저수량", "저수율"],
    "rainfall_inflow": ["rain_3d", "rain_7d", "rain_14d", "rainfall", "inflow", "inflow_7d", "강우량", "유입량"],
    "growth_condition": ["water_temp", "air_temp_mean_7d", "avg_temp", "acc_temp_7d", "solar_sum_7d", "sunshine", "ph", "수온", "pH", "Chl_a", "탁도"],
    "past_bloom": ["current_cells", "prev_cells", "delta_cells", "growth_rate_cells", "유해남조류", "Microcystis", "Anabaena", "Oscillatoria", "Aphanizomenon"],
}

SCENARIO_ACTION_CATEGORY = {
    "관심 발령 후보 시나리오": "주 1회 이상 채수 및 관계기관 공유",
    "경계 발령 후보 시나리오": "주 2회 이상 채수 및 정수처리 강화 준비",
    "조류대발생 감시 시나리오": "관계기관 긴급 공유 및 비상 대응 검토",
    "관심 기준 접근 시나리오": "다음 채수 전 현장 확인 및 모니터링 강화",
    "기준 초과 지속 시나리오": "강화 모니터링 및 대응 자원 준비",
    "하향·해제 관찰 시나리오": "2회 연속 하위 기준 여부 확인",
    "강우 유입 후 정체 고위험 시나리오": "방류/순환 조건 점검",
    "회남 선행 전파 관찰 시나리오": "문의·추동 지점 선제 관찰",
    "수체 정체·성층 위험 시나리오": "물순환설비 가동 검토",
    "강우 이후 영양염류 유입 시나리오": "추가 채수 및 현장 확인",
    "수온 20도 이상 계절 감시 시나리오": "하절기 상시 모니터링 강화",
    "고온·고일사 성장 촉진 시나리오": "모니터링 강화",
    "과거 증식 추세 지속 시나리오": "추가 채수 및 현장 확인",
    "복합 고위험 시나리오": "관계기관 공유 필요",
    "관찰 강화 시나리오": "모니터링 강화",
    "저농도 상승 관찰 시나리오": "추세 관찰 및 다음 채수 확인",
    "계절 위험 준비 시나리오": "하절기 예비 대응 점검",
    "일반 안정 시나리오": "일반 관찰 유지",
}
