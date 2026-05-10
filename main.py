from __future__ import annotations

import pandas as pd

from src.preprocessing import apply_feature_engineering
from src.evaluate import evaluate_candidate_models, select_best_models
from src.config import CLASSIFICATION_TARGET, DATE_COLUMN, DROP_COLUMNS, EXPLAIN_DIR, METRIC_DIR, MODEL_INPUT_PATH, MODEL_SELECTION_METRIC_CLASSIFICATION, MODEL_SELECTION_METRIC_REGRESSION, OUTPUT_DIR, PREDICTION_DIR, PROBABILITY_THRESHOLD, RANDOM_STATE, REGRESSION_TARGET, SCENARIO_DIR, SITE_COLUMN, SPLIT_COLUMN, STATION_DROP_COLUMNS, STATION_IDS, VALID_SIZE_RATIO
from src.data_loader import load_model_input, time_based_train_valid_split, get_feature_columns, split_features_targets
from src.train import predict_with_all_models, train_candidate_models, predict_with_best_models
from src.explanation import get_feature_importance_for_all_models, get_feature_importance_table, make_shap_summary_table_for_all_models, make_shap_top_reasons_table, save_explain_outputs
from src.scenario import build_scenario_results, save_scenario_outputs
from src.utils import load_models, prepare_station_directories, save_metrics, save_models, save_prediction_outputs, save_threshold_results
from src.tune import tune_hyperparameters


def main():

    print("🚀 대청호 조류 예측: 지점별 개별 모델링 파이프라인 시작...\n")

    # 1. 전체 데이터 한 번에 로드
    df = load_model_input(MODEL_INPUT_PATH)
    df = apply_feature_engineering(df, date_column=DATE_COLUMN, site_column=SITE_COLUMN)

    # 2. 지점별로 반복 (Loop) 시작
    for station_id in STATION_IDS:
        print("="*60)
        print(f"📍 [지점명: {station_id}] 모델링 시작")
        print("="*60)

        station_df = df[df[SITE_COLUMN] == station_id].copy()

        if station_df.empty:
            print(f"⚠️ 경고: 지점 {station_id}에 해당하는 데이터가 없습니다. 건너뜁니다.")
            continue
        
        st_paths = prepare_station_directories(station_id)

        # 3. 시간 기반 Train / Valid 스플릿 (지점 데이터 내에서 분할)
        train_df, valid_df = time_based_train_valid_split(
            station_df,
            DATE_COLUMN,
            SPLIT_COLUMN,
            VALID_SIZE_RATIO,
        )

        # 4. 사용할 피처 선정 (나중에 DROP 리스트가 업데이트되면 여기서 걸러집니다)
        current_drop_list = STATION_DROP_COLUMNS.get(station_id, [])
        feature_columns = get_feature_columns(
            train_df,
            current_drop_list,
        )

        # 타겟 분리 (옵션: train 함수 안에서 분리해도 되지만 기존 로직 유지)
        x_train, y_reg_train, y_cls_train = split_features_targets(
            train_df,
            feature_columns,
            REGRESSION_TARGET,
            CLASSIFICATION_TARGET,
        )

        x_valid, y_reg_valid, y_cls_valid = split_features_targets(
            valid_df,
            feature_columns,
            REGRESSION_TARGET,
            CLASSIFICATION_TARGET,
        )

        print(f"[{station_id}] Optuna 하이퍼파라미터 튜닝 시작 (n_trials=30)...")
        # 튜닝 속도를 위해 n_trials=30 정도로 시작해보세요. (나중에 50~100으로 늘림)
        best_reg_params, best_cls_params = tune_hyperparameters(
            x_train, y_reg_train, y_cls_train, n_trials=30, random_state=RANDOM_STATE
        )
        print(f"  > 회귀 베스트 모델/파라미터: {best_reg_params}")
        print(f"  > 분류 베스트 모델/파라미터: {best_cls_params}")

        


        # 5. [학습] 모델 튜닝 및 학습 진행
        print(f"[{station_id}] 최적 파라미터 적용 후 학습 진행 중...")
        # (주의) train_candidate_models 함수가 optuna 파라미터를 받도록 살짝 수정해야 합니다!
        trained = train_candidate_models(
            train_df,
            feature_columns,
            REGRESSION_TARGET,
            CLASSIFICATION_TARGET,
            RANDOM_STATE,
            optuna_cls_params=best_cls_params, # <- 추가
            optuna_reg_params=best_reg_params  # <- 추가
        )

        # 6. [평가] 평가지표 채점 및 베스트 모델 선정
        metrics_df, threshold_df = evaluate_candidate_models(
            trained,
            valid_df,
            REGRESSION_TARGET,
            CLASSIFICATION_TARGET,
            PROBABILITY_THRESHOLD,
        )

        best_models = select_best_models(
            trained,
            metrics_df,
            MODEL_SELECTION_METRIC_REGRESSION,
            MODEL_SELECTION_METRIC_CLASSIFICATION,
        )

        # 7. [저장] 지점별 폴더에 모델과 지표 저장
        print(f"[{station_id}] 모델 및 지표 저장 중...")
        save_models(
            trained,
            best_models,
            st_paths["model"],
        )
        save_metrics(metrics_df, st_paths["metric"])
        save_threshold_results(threshold_df, st_paths["metric"])

        # 다시 불러와서 예측 결과 저장
        loaded = load_models(st_paths["model"])
        all_model_prediction_df = predict_with_all_models(
            loaded,
            valid_df,
        )
        best_model_prediction_df = predict_with_best_models(
            loaded,
            valid_df,
        )

        regression_prediction_df, classification_prediction_df = save_prediction_outputs(
            best_model_prediction_df,
            valid_df,
            st_paths["prediction"],
            REGRESSION_TARGET,
            CLASSIFICATION_TARGET,
        )

        # 8. [해석] SHAP 등 XAI 분석 및 시나리오 생성
        print(f"[{station_id}] SHAP 분석 및 시나리오 도출 중...")
        importance_df = get_feature_importance_for_all_models(loaded)

        try:
            all_model_shap_cls_df = make_shap_summary_table_for_all_models(
                loaded,
                valid_df,
                all_model_prediction_df,
                task="classification",
            )

            all_model_shap_reg_df = make_shap_summary_table_for_all_models(
                loaded,
                valid_df,
                all_model_prediction_df,
                task="regression",
            )
        except ImportError:
            all_model_shap_cls_df = pd.DataFrame()
            all_model_shap_reg_df = pd.DataFrame()

        shap_top_reasons_df = make_shap_top_reasons_table(
            loaded,
            valid_df,
            best_model_prediction_df,
        )

        save_explain_outputs(
            importance_df,
            shap_top_reasons_df,
            st_paths["explain"],
        )

        scenario_results_df = build_scenario_results(shap_top_reasons_df)
        scenario_input_for_llm = save_scenario_outputs(
            scenario_results_df,
            st_paths["scenario"],
        )

        print(f"✅ [{station_id}] 지점의 모든 파이프라인이 완료되었습니다.\n")
    
    print("🏁 대청호 3개 지점의 개별 모델링이 모두 성공적으로 종료되었습니다!")

if __name__ == "__main__":
    main()
