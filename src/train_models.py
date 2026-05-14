from __future__ import annotations

import logging

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.loader import load_model_input
from src.features import (
    add_temporal_spatial_features,
    drop_rows_without_future_target,
    time_based_train_valid_split,
    get_feature_columns,
)
from src.models import (
    train_candidate_models,
    evaluate_candidate_models,
    select_best_models,
)
from src.model_config import get_selection_metric, load_model_config
from src.persistence import (
    save_models,
    save_metrics,
    save_threshold_results,
    save_prediction_outputs,
)
from src.explain_scenario import (
    get_feature_importance_table,
    make_shap_top_reasons_table,
    build_scenario_results,
    save_scenario_outputs,
    alert_stage_from_cells,
)
from src.config import (
    MODEL_INPUT_PATH,
    OUTPUT_DIR,
    METRIC_DIR,
    PREDICTION_DIR,
    EXPLAIN_DIR,
    SCENARIO_DIR,
    MODEL_CONFIG_PATH,
    DATE_COLUMN,
    SPLIT_COLUMN,
    DROP_COLUMNS,
    VALID_SIZE_RATIO,
    REGRESSION_TARGET,
    CLASSIFICATION_TARGET,
    ALERT_CELL_THRESHOLD,
    LOCATION_NAME_BY_CODE,
)

logging.basicConfig(level=logging.INFO)


FUTURE_ALERT_TARGET = "next_alert_binary"


def add_future_alert_target(df):
    df = df.copy()
    if REGRESSION_TARGET not in df.columns:
        raise KeyError(f"Regression target not found: {REGRESSION_TARGET}")
    alert_log_threshold = __import__("math").log10(ALERT_CELL_THRESHOLD + 1)
    df[FUTURE_ALERT_TARGET] = (df[REGRESSION_TARGET] >= alert_log_threshold).astype(int)
    if "previous_exceeded" in df.columns:
        df["operational_alert_target"] = (
            df["previous_exceeded"].astype(bool) & df[FUTURE_ALERT_TARGET].astype(bool)
        ).astype(int)
    return df


def make_prediction_frame(input_df, best_models):
    from src.models import inverse_transform_cells

    x = input_df[best_models["feature_columns"]]
    pred_reg = best_models["regression_model"].predict(x)
    pred_proba = best_models["classification_model"].predict_proba(x)[:, 1]
    threshold = best_models.get("best_classification_threshold")

    id_cols = [
        DATE_COLUMN,
        "sample_date",
        "site",
        "loc_encoded",
        "loc_flow_order",
        "sampling_gap_days",
        "previous_observed_cells",
        "previous_exceeded",
        "operational_alert_target",
    ]
    pred_df = input_df[[col for col in id_cols if col in input_df.columns]].copy()
    if "loc_encoded" in pred_df.columns:
        pred_df["location_name"] = pred_df["loc_encoded"].map(LOCATION_NAME_BY_CODE)
    pred_df["pred_regression_target"] = pred_reg
    pred_df["predicted_cells"] = inverse_transform_cells(pred_reg)
    pred_df["alert_probability"] = pred_proba
    pred_df["predicted_alert_label"] = (pred_proba >= threshold).astype(int)
    pred_df["predicted_cell_exceeded"] = (pred_df["predicted_cells"] >= ALERT_CELL_THRESHOLD).astype(int)
    pred_df["predicted_alert_stage"] = pred_df["predicted_cells"].map(alert_stage_from_cells)
    if "previous_exceeded" in pred_df.columns:
        pred_df["operational_alert_candidate"] = (
            pred_df["previous_exceeded"].astype(bool)
            & (
                pred_df["predicted_cell_exceeded"].astype(bool)
                | pred_df["predicted_alert_label"].astype(bool)
            )
        ).astype(int)
    return pred_df


def main():
    logging.info("Loading model input")
    df = load_model_input(MODEL_INPUT_PATH)
    model_config = load_model_config(MODEL_CONFIG_PATH)

    logging.info("Adding temporal, location-order, and upstream features")
    df = add_temporal_spatial_features(df)
    df = drop_rows_without_future_target(df)
    df = add_future_alert_target(df)

    logging.info("Splitting train/valid")
    train_df, valid_df = time_based_train_valid_split(df, DATE_COLUMN, SPLIT_COLUMN, VALID_SIZE_RATIO)

    logging.info("Selecting features")
    feature_columns = get_feature_columns(train_df, DROP_COLUMNS)

    logging.info("Training candidate models")
    trained = train_candidate_models(
        train_df,
        feature_columns,
        REGRESSION_TARGET,
        FUTURE_ALERT_TARGET,
        model_config=model_config,
    )

    logging.info("Evaluating candidates")
    metrics_df, threshold_df = evaluate_candidate_models(trained, valid_df, REGRESSION_TARGET, FUTURE_ALERT_TARGET)

    logging.info("Selecting best models")
    best_models = select_best_models(
        trained,
        metrics_df,
        regression_metric=get_selection_metric(model_config, "regression", "rmse"),
        classification_metric=get_selection_metric(model_config, "classification", "recall"),
    )
    best_models["regression_target"] = REGRESSION_TARGET
    best_models["classification_target"] = FUTURE_ALERT_TARGET

    logging.info("Saving models and metrics")
    save_models(trained, best_models, OUTPUT_DIR)
    save_metrics(metrics_df, METRIC_DIR)
    save_threshold_results(threshold_df, METRIC_DIR)

    # save per-run reports under `output/` so user can inspect run-by-run model txts
    from pathlib import Path
    from src.persistence import save_run_reports
    run_reports_dir = Path("output")
    try:
        run_dir = save_run_reports(trained, metrics_df, run_reports_dir)
        logging.info(f"Saved run reports to: {run_dir}")
    except Exception as e:
        logging.warning(f"Failed to save run reports: {e}")

    logging.info("Predicting with best models")
    pred_df = make_prediction_frame(valid_df, best_models)

    save_prediction_outputs(pred_df, valid_df, PREDICTION_DIR, REGRESSION_TARGET, FUTURE_ALERT_TARGET)

    logging.info("Generating explainability outputs")
    importance_df = get_feature_importance_table(best_models["classification_model"], best_models["feature_columns"]) 
    try:
        shap_top = make_shap_top_reasons_table(best_models, valid_df, pred_df)
    except Exception:
        shap_top = make_shap_top_reasons_table(best_models, valid_df, pred_df)

    # save explain outputs
    EXPLAIN_DIR.mkdir(parents=True, exist_ok=True)
    importance_df.to_csv(EXPLAIN_DIR / "feature_importance.csv", index=False)
    shap_top.to_csv(EXPLAIN_DIR / "shap_top_reasons.csv", index=False)

    logging.info("Building full-period scenario results")
    all_pred_df = make_prediction_frame(df, best_models)
    all_shap_top = make_shap_top_reasons_table(best_models, df, all_pred_df)
    all_shap_top.to_csv(EXPLAIN_DIR / "shap_top_reasons_all.csv", index=False)
    scenario_results = build_scenario_results(all_shap_top)
    save_scenario_outputs(scenario_results, SCENARIO_DIR)

    logging.info("Pipeline finished")


if __name__ == "__main__":
    main()
