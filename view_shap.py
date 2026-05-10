import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt

from src.config import CLASSIFICATION_MODEL_FILE, CLASSIFICATION_TARGET, DATE_COLUMN, DROP_COLUMNS, EXPLAIN_DIR, MODEL_INPUT_PATH, OUTPUT_DIR, RANDOM_STATE, REGRESSION_TARGET, SITE_COLUMN, SPLIT_COLUMN, STATION_IDS, VALID_SIZE_RATIO
from src.data_loader import get_feature_columns, load_model_input, time_based_train_valid_split
from src.train import train_candidate_models


plt.rcParams['font.family'] = 'Malgun Gothic' # 맥은 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

def plot_station_shap(station_name: str, save_csv: bool = True):
    print(f"📊 [{station_name}] 지점의 SHAP 분석을 시작합니다...")
    
    # 1. 지점별 폴더 경로 설정 (유저님의 폴더명에 맞게 수정)
    station_dir = station_name.replace("(", "_").replace(")", "").replace(" ", "_")
    model_path = OUTPUT_DIR / station_dir / CLASSIFICATION_MODEL_FILE
    
    # 2. 모델 로드
    try:
        best_model = joblib.load(model_path)
    except FileNotFoundError:
        print(f"❌ 에러: {model_path} 에서 모델을 찾을 수 없습니다.")
        return

    # 3. 전체 데이터 로드 및 해당 지점 필터링
    df = load_model_input(MODEL_INPUT_PATH)
    station_df = df[df[SITE_COLUMN] == station_name].copy()
    
    # 사용할 피처만 추출 (DROP_COLUMNS 적용)
    feature_columns = get_feature_columns(station_df, DROP_COLUMNS)
    X = station_df[feature_columns]

    # 4. SHAP 값 계산
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X)

    # 이진 분류 모델 SHAP 값 처리 (경보 발생인 '1' 클래스 기준)
    if isinstance(shap_values, list):
        shap_target = shap_values[1] 
    elif len(shap_values.shape) == 3:
        shap_target = shap_values[:, :, 1]
    else:
        shap_target = shap_values
    
    if save_csv:
        # 각 피처별 '평균 절대 기여도'를 계산합니다. (값이 클수록 중요한 피처)
        mean_abs_shap = np.abs(shap_target).mean(axis=0)
        
        # 데이터프레임으로 만들기
        importance_df = pd.DataFrame({
            'feature': feature_columns,
            'mean_abs_shap': mean_abs_shap
        })
        
        # 기여도가 높은 순서대로 내림차순 정렬
        importance_df = importance_df.sort_values(by='mean_abs_shap', ascending=False).reset_index(drop=True)
        
        # 저장 경로 설정 (EXPLAIN_DIR 아래에 지점별 폴더 생성)
        save_dir = EXPLAIN_DIR / station_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        
        csv_path = save_dir / "shap_feature_importance.csv"
        
        # 한글 깨짐 방지 옵션을 넣어서 저장!
        importance_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"💾 SHAP 중요도 수치 파일 저장 완료: {csv_path}")

    # 5. 시각화 (Summary Plot)
    plt.figure(figsize=(10, 8))
    plt.title(f"[{station_name}] 조류 경보 예측 핵심 요인 (SHAP)")
    shap.summary_plot(shap_target, X, max_display=40, show=False)
    plt.tight_layout()
    plt.show()

# ==========================================
# 실행하기 (보고 싶은 지점명을 입력하세요)
# ==========================================
# 예시: "회남(옥천)", "문의(청남대)", "추동(세천)"
target_station = "회남(옥천)"
#target_station = "문의(청남대)"
#target_station = "추동(세천)" 
plot_station_shap(target_station)



