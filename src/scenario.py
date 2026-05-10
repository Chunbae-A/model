from typing import Any
import pandas as pd
import json
from pathlib import Path
from src.utils import _json_default
from src.config import (
    DATE_COLUMN, SITE_COLUMN,
    SCENARIO_FEATURE_GROUPS,
    ALERT_OUTBREAK, ALERT_WARNING, ALERT_ATTENTION,
    BASE_ACTION, SPECIFIC_ISSUE_ACTION,
    FORECAST_HORIZON,
    SCENARIO_RESULT_FILE,
    SCENARIO_LLM_INPUT_FILE
)
def _feature_matches_group(feature_name: Any, group_features: list[str]) -> bool:
    if feature_name is None or pd.isna(feature_name):
        return False

    feature_name_lower = str(feature_name).lower()
    return any(group_feature.lower() in feature_name_lower for group_feature in group_features)


def _count_top_feature_groups(top_features: list[Any]) -> dict[str, int]:
    counts = {group_name: 0 for group_name in SCENARIO_FEATURE_GROUPS}
    for feature_name in top_features:
        for group_name, group_features in SCENARIO_FEATURE_GROUPS.items():
            if _feature_matches_group(feature_name, group_features):
                counts[group_name] += 1
    return counts


def classify_scenario(row: pd.Series) -> tuple[str, str, str]:
    """경보 수치와 SHAP 원인을 통합하여 구체적인 시나리오를 만든다 냥!"""
    predicted_cells = float(row.get("predicted_cells", 0.0))
    probability = float(row.get("alert_probability", 0.0))
    top_features = [str(row.get(f"shap_top_{i}")) for i in range(1, 4) if pd.notna(row.get(f"shap_top_{i}"))]

    # 1. 조류경보제 수치 기준 등급 판정
    if predicted_cells >= ALERT_OUTBREAK:
        level, icon = "대발생", "🚨"
    elif predicted_cells >= ALERT_WARNING:
        level, icon = "경계", "🔴"
    elif predicted_cells >= ALERT_ATTENTION:
        level, icon = "관심", "🟡"
    else:
        level, icon = "정상", "🟢"

    # 2. SHAP 기반 주요 원인(Issue) 찾기
    group_counts = {k: 0 for k in SCENARIO_FEATURE_GROUPS}
    for feature in top_features:
        for group, features in SCENARIO_FEATURE_GROUPS.items():
            if any(f in feature for f in features):
                group_counts[group] += 1

    # 가장 많이 걸린 원인 그룹 찾기 (계절성은 기본으로 깔리니 다른 심각한 원인을 우선함)
    active_groups = {k: v for k, v in group_counts.items() if v > 0 and k != 'seasonal'}
    
    if level == "정상":
        issue_name = "안정 상태"
        action_category = BASE_ACTION["정상"]
    else:
        if active_groups:
            dominant_group = max(active_groups, key=active_groups.get)
            issue_name, specific_action = SPECIFIC_ISSUE_ACTION[dominant_group]
            action_category = f"{BASE_ACTION[level]} 및 {specific_action}"
        else:
            issue_name, specific_action = SPECIFIC_ISSUE_ACTION["seasonal"]
            action_category = f"{BASE_ACTION[level]} 및 {specific_action}"

    # 3. 최종 문자열 조립
    scenario_type = f"[{icon} {level}] {issue_name} 시나리오"
    scenario_reason = f"예측 세포수: {predicted_cells:.1f} cells/mL (위험도: {probability:.1%}). 주요 탐지 요인: {top_features}"

    return scenario_type, scenario_reason, action_category


def build_scenario_results(shap_top_reasons_df: pd.DataFrame) -> pd.DataFrame:
    """시나리오 유형, 시나리오 근거, 권장 대응 범주 컬럼을 추가합니다."""
    required = ["predicted_cells", "alert_probability", "predicted_alert_label"]
    missing = [col for col in required if col not in shap_top_reasons_df.columns]
    if missing:
        raise KeyError(f"시나리오 입력에 필수 컬럼이 없습니다: {missing}")

    output = shap_top_reasons_df.copy()
    scenario_values = output.apply(classify_scenario, axis=1, result_type="expand")
    output["scenario_type"] = scenario_values[0]
    output["scenario_reason"] = scenario_values[1]
    output["recommended_action_category"] = scenario_values[2]

    keep_cols = [
        DATE_COLUMN,
        SITE_COLUMN,
        "predicted_cells",
        "alert_probability",
        "predicted_alert_label",
        "shap_top_1",
        "shap_top_2",
        "shap_top_3",
        "scenario_type",
        "scenario_reason",
        "recommended_action_category",
    ]
    existing_cols = [col for col in keep_cols if col in output.columns]
    return output[existing_cols]


def build_scenario_input_for_llm(scenario_results_df: pd.DataFrame) -> list[dict[str, Any]]:
    """다음 에이전트가 사용할 구조화 JSON 객체를 만듭니다.

    이 함수는 브리핑 문장을 만들지 않습니다.
    다음 LLM Briefing Agent가 필요한 구조화 데이터만 만듭니다.
    """
    records: list[dict[str, Any]] = []
    for _, row in scenario_results_df.iterrows():
        top_reasons = []
        for rank in range(1, 4):
            feature = row.get(f"shap_top_{rank}")
            if pd.notna(feature):
                top_reasons.append({"rank": rank, "feature": feature})

        records.append({
            "date": str(row.get(DATE_COLUMN)) if DATE_COLUMN in row else None,
            "site": row.get(SITE_COLUMN) if SITE_COLUMN in row else None,
            "forecast_horizon": FORECAST_HORIZON,
            "predicted_cells": row.get("predicted_cells"),
            "alert_probability": row.get("alert_probability"),
            "predicted_alert_label": row.get("predicted_alert_label"),
            "top_reasons": top_reasons,
            "scenario_type": row.get("scenario_type"),
            "scenario_reason": row.get("scenario_reason"),
            "recommended_action_category": row.get("recommended_action_category"),
        })
    return records


def save_scenario_outputs(scenario_results_df: pd.DataFrame, scenario_dir: Path) -> list[dict[str, Any]]:
    scenario_dir.mkdir(parents=True, exist_ok=True)
    scenario_results_df.to_csv(scenario_dir / SCENARIO_RESULT_FILE, index=False, encoding='utf-8-sig')

    llm_input = build_scenario_input_for_llm(scenario_results_df)
    with open(scenario_dir / SCENARIO_LLM_INPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(llm_input, f, ensure_ascii=False, indent=2, default=_json_default)
    return llm_input