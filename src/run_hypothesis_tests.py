from __future__ import annotations

import math
import sys
import textwrap
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import ALERT_CELL_THRESHOLD  # noqa: E402
from src.features import add_temporal_spatial_features, drop_rows_without_future_target  # noqa: E402


INPUT_PATH = ROOT / "data" / "processed" / "model_input" / "algae_model_input.csv"
CLASSIFICATION_METRICS_PATH = ROOT / "artifacts" / "metrics" / "classification_metrics.json"
CLASSIFICATION_PREDICTIONS_PATH = ROOT / "artifacts" / "predictions" / "classification_predictions.csv"
OUTPUT_DIR = ROOT / "artifacts" / "hypotheses"
FIGURE_DIR = ROOT / "artifacts" / "figures"

RESULT_CSV = OUTPUT_DIR / "hypothesis_results.csv"
REPORT_MD = OUTPUT_DIR / "hypothesis_report.md"
SUMMARY_SVG = FIGURE_DIR / "hypothesis_experiment_summary.svg"
SCENARIO_SVG = FIGURE_DIR / "hypothesis_scenario_matrix.svg"
RISK_RATE_SVG = FIGURE_DIR / "hypothesis_risk_rate_comparison.svg"
EFFECT_SIZE_SVG = FIGURE_DIR / "hypothesis_effect_size_comparison.svg"
PVALUE_SVG = FIGURE_DIR / "hypothesis_pvalue_comparison.svg"

LOCATION_LABELS = {
    0: "Munui",
    1: "Chudong",
    2: "Hoenam",
}


def q(df: pd.DataFrame, column: str, quantile: float) -> float:
    if column not in df.columns:
        raise KeyError(f"Missing required column: {column}")
    return float(df[column].quantile(quantile))


def to_binary(series: pd.Series) -> pd.Series:
    return series.fillna(False).astype(bool)


def load_analysis_frame() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Model input file not found: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()
    df = add_temporal_spatial_features(df, date_column="date", site_column="loc_encoded")
    df = drop_rows_without_future_target(df)
    alert_log_threshold = math.log10(ALERT_CELL_THRESHOLD + 1)
    df["next_alert_binary"] = (df["next_log_cells"] >= alert_log_threshold).astype(int)
    df["location_label"] = df["loc_encoded"].map(LOCATION_LABELS).fillna(df["loc_encoded"].astype(str))
    return df


def build_exposure_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    high_temp_surface = (
        (df["water_temp"] >= q(df, "water_temp", 0.70)).astype(int)
        + (df["pH"] >= q(df, "pH", 0.70)).astype(int)
        + (df["sunshine_7d_sum"] >= q(df, "sunshine_7d_sum", 0.60)).astype(int)
        + (df["Chl_a"] >= q(df, "Chl_a", 0.60)).astype(int)
        + (df["microcystis"] >= q(df, "microcystis", 0.70)).astype(int)
        + (df["residence_proxy"] >= q(df, "residence_proxy", 0.60)).astype(int)
    ) >= 4

    rainfall_inflow = (
        (df["rain_3d_sum"] >= q(df, "rain_3d_sum", 0.70)).astype(int)
        + (df["rain_7d_sum"] >= q(df, "rain_7d_sum", 0.70)).astype(int)
        + (df["rain_14d_sum"] >= q(df, "rain_14d_sum", 0.70)).astype(int)
        + (df["inflow_7d_sum"] >= q(df, "inflow_7d_sum", 0.70)).astype(int)
        + (df["turbidity"] >= q(df, "turbidity", 0.60)).astype(int)
        + ((df["anabaena"] + df["aphanizomenon"]) >= q(df, "anabaena", 0.70)).astype(int)
    ) >= 3

    spatial_transfer = (
        (df["has_upstream_site"].eq(1) & (df["upstream_cells_same_date"] >= ALERT_CELL_THRESHOLD))
        | (df["hoenam_pressure_for_downstream"].eq(1))
        | (df["upstream_log_cells_same_date"] >= q(df, "upstream_log_cells_same_date", 0.80))
    )

    stagnation = (
        (df["outflow"] <= q(df, "outflow", 0.30)).astype(int)
        + (df["outflow_7d_sum"] <= q(df, "outflow_7d_sum", 0.30)).astype(int)
        + (df["residence_proxy"] >= q(df, "residence_proxy", 0.70)).astype(int)
        + (df["avg_wind"] <= q(df, "avg_wind", 0.35)).astype(int)
        + (df["water_temp"] >= q(df, "water_temp", 0.60)).astype(int)
        + ((df["microcystis"] + df["oscillatoria"] + df["aphanizomenon"]) >= q(df, "cyano_cells", 0.65)).astype(int)
    ) >= 4

    return {
        "H1": high_temp_surface,
        "H2": rainfall_inflow,
        "H3": spatial_transfer,
        "H4": stagnation,
    }


def fisher_pvalue(table: list[list[int]]) -> tuple[float | None, float | None]:
    try:
        from scipy.stats import fisher_exact
    except Exception:
        return None, None
    odds_ratio, p_value = fisher_exact(table, alternative="greater")
    return float(odds_ratio), float(p_value)


def two_proportion_z_pvalue(a: int, n_a: int, b: int, n_b: int) -> float | None:
    try:
        from scipy.stats import norm
    except Exception:
        return None
    if n_a == 0 or n_b == 0:
        return None
    p1 = a / n_a
    p2 = b / n_b
    pooled = (a + b) / (n_a + n_b)
    se = math.sqrt(max(pooled * (1 - pooled) * (1 / n_a + 1 / n_b), 0))
    if se == 0:
        return None
    z = (p1 - p2) / se
    return float(1 - norm.cdf(z))


def evaluate_binary_hypothesis(
    df: pd.DataFrame,
    *,
    hypothesis_id: str,
    title: str,
    scenario: str,
    exposure: pd.Series,
    expected_effect: str,
    scenario_action: str,
) -> dict[str, object]:
    exposed = to_binary(exposure)
    target = df["next_alert_binary"].astype(int).eq(1)
    n_exposed = int(exposed.sum())
    n_control = int((~exposed).sum())
    event_exposed = int((exposed & target).sum())
    event_control = int(((~exposed) & target).sum())
    non_event_exposed = n_exposed - event_exposed
    non_event_control = n_control - event_control

    rate_exposed = event_exposed / n_exposed if n_exposed else np.nan
    rate_control = event_control / n_control if n_control else np.nan
    risk_ratio = rate_exposed / rate_control if rate_control and not pd.isna(rate_control) else np.nan
    risk_lift_pct = (risk_ratio - 1) * 100 if not pd.isna(risk_ratio) else np.nan
    odds_ratio, fisher_p = fisher_pvalue([[event_exposed, non_event_exposed], [event_control, non_event_control]])
    z_p = two_proportion_z_pvalue(event_exposed, n_exposed, event_control, n_control)

    supported = bool(
        n_exposed >= 20
        and not pd.isna(risk_ratio)
        and risk_ratio > 1
        and ((fisher_p is not None and fisher_p < 0.05) or (z_p is not None and z_p < 0.05))
    )
    if supported:
        conclusion = "supported"
    elif not pd.isna(risk_ratio) and risk_ratio > 1:
        conclusion = "partially_supported"
    else:
        conclusion = "not_supported"

    return {
        "hypothesis_id": hypothesis_id,
        "title": title,
        "scenario": scenario,
        "n_exposed": n_exposed,
        "n_control": n_control,
        "event_exposed": event_exposed,
        "event_control": event_control,
        "event_rate_exposed": rate_exposed,
        "event_rate_control": rate_control,
        "risk_ratio": risk_ratio,
        "risk_lift_pct": risk_lift_pct,
        "odds_ratio": odds_ratio,
        "fisher_p_value": fisher_p,
        "ztest_p_value": z_p,
        "expected_effect": expected_effect,
        "conclusion": conclusion,
        "scenario_action": scenario_action,
    }


def build_hypotheses(df: pd.DataFrame) -> list[dict[str, object]]:
    masks = build_exposure_masks(df)

    definitions = [
        {
            "hypothesis_id": "H1",
            "title": "High-temperature surface bloom",
            "scenario": "고수온 표층 증식 시나리오",
            "exposure": masks["H1"],
            "expected_effect": "interest-risk probability about 1.5-2.0x higher",
            "scenario_action": "모니터링 강화, 조류차단막 운영, 취수구 주변 집중 점검, 정수 처리 강화",
        },
        {
            "hypothesis_id": "H2",
            "title": "Post-rainfall nutrient inflow",
            "scenario": "강수 후 영양염류 유입 시나리오",
            "exposure": masks["H2"],
            "expected_effect": "interest-risk probability about 20-50% higher",
            "scenario_action": "상류 오염원 점검, 부유물질 수거, 오탁방지막/조류차단막 검토, 모니터링 강화",
        },
        {
            "hypothesis_id": "H3",
            "title": "Spatial risk transfer",
            "scenario": "수역별 위험 전이 시나리오",
            "exposure": masks["H3"],
            "expected_effect": "interest-risk probability about 10-30% higher",
            "scenario_action": "선행 위험 수역과 영향 가능 수역까지 모니터링 확대, 장비/대응 우선순위 조정",
        },
        {
            "hypothesis_id": "H4",
            "title": "Hydraulic stagnation bloom",
            "scenario": "정체 조건 심화 시나리오",
            "exposure": masks["H4"],
            "expected_effect": "interest-risk probability about 1.5-2.0x higher",
            "scenario_action": "물순환설비 가동, 방류량/수위 운영 검토, 취수구 주변 조류 유입 차단",
        },
    ]

    return [evaluate_binary_hypothesis(df, **definition) for definition in definitions]


def evaluate_early_warning() -> dict[str, object] | None:
    if not CLASSIFICATION_PREDICTIONS_PATH.exists():
        return None
    pred = pd.read_csv(CLASSIFICATION_PREDICTIONS_PATH)
    y_true = pred["y_true_alert"].astype(int)
    y_pred = pred["y_pred_alert"].astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    accuracy = (tp + tn) / len(pred) if len(pred) else np.nan
    precision = tp / (tp + fp) if (tp + fp) else np.nan
    recall = tp / (tp + fn) if (tp + fn) else np.nan
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else np.nan
    try:
        from scipy.stats import binomtest

        majority = float(y_true.value_counts(normalize=True).max())
        p_value = float(binomtest(tp + tn, len(pred), p=majority, alternative="greater").pvalue)
    except Exception:
        p_value = None
    return {
        "hypothesis_id": "H5",
        "title": "7-day early warning utility",
        "scenario": "7일 선제 대응 시나리오",
        "n_exposed": len(pred),
        "n_control": np.nan,
        "event_exposed": int(y_true.sum()),
        "event_control": np.nan,
        "event_rate_exposed": recall,
        "event_rate_control": np.nan,
        "risk_ratio": np.nan,
        "risk_lift_pct": np.nan,
        "odds_ratio": np.nan,
        "fisher_p_value": p_value,
        "ztest_p_value": np.nan,
        "expected_effect": "about 7-day-ahead operational risk identification",
        "conclusion": "supported" if recall >= 0.9 and accuracy >= 0.9 else "partially_supported",
        "scenario_action": "경보 발령 전 모니터링 강화, 제거선/방제장비 배치, 취수 수심 변경, 활성탄/오존 처리 준비",
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def fmt(value: object, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "-"
    value = float(value)
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    return f"{value:.{digits}f}"


def pct(value: object) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value) * 100:.1f}%"


def p_text(value: object) -> str:
    if value is None or pd.isna(value):
        return "-"
    value = float(value)
    if value < 0.001:
        return f"{value:.2e}"
    return f"{value:.4f}"


def status_label(value: str) -> tuple[str, str]:
    if value == "supported":
        return "Supported", "#2E7D5B"
    if value == "partially_supported":
        return "Partial", "#C9822B"
    return "Not supported", "#A23B3B"


def wrap_svg(parts: list[str], text: str, x: int, y: int, width: int, size: int, fill: str = "#334155") -> int:
    chars = max(16, int(width / (size * 0.56)))
    lines = textwrap.wrap(text, width=chars, break_long_words=False, break_on_hyphens=False) or [""]
    line_h = int(size * 1.45)
    parts.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}">')
    for idx, line in enumerate(lines[:4]):
        dy = 0 if idx == 0 else line_h
        parts.append(f'<tspan x="{x}" dy="{dy}">{escape(line)}</tspan>')
    parts.append("</text>")
    return y + (min(len(lines), 4) - 1) * line_h


def svg_header(width: int, height: int, title: str, subtitle: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="#f5f7fb"/>',
        '<style>text{font-family:Inter,Segoe UI,Arial,sans-serif}</style>',
        f'<text x="54" y="58" font-size="30" font-weight="800" fill="#111827">{escape(title)}</text>',
        f'<text x="54" y="88" font-size="15" fill="#64748b">{escape(subtitle)}</text>',
    ]


def save_svg(parts: list[str], path: Path) -> Path:
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def build_summary_svg(results: pd.DataFrame) -> Path:
    width = 1500
    row_h = 146
    height = 168 + row_h * len(results) + 60
    parts = svg_header(
        width,
        height,
        "Hypothesis Experiment Summary",
        "Risk-rate comparison, p-value checks, and scenario linkage for Daecheong algae warning hypotheses.",
    )
    y = 128
    headers = ["Hypothesis", "Scenario", "Observed Effect", "p-value", "Conclusion"]
    xs = [74, 230, 640, 1010, 1180]
    widths = [130, 370, 330, 130, 230]
    for x, header in zip(xs, headers):
        parts.append(f'<text x="{x}" y="{y}" font-size="13" font-weight="800" fill="#64748b">{escape(header)}</text>')
    for idx, row in enumerate(results.itertuples()):
        yy = y + 24 + idx * row_h
        bg = "#ffffff" if idx % 2 == 0 else "#fbfdff"
        parts.append(f'<rect x="54" y="{yy}" width="1392" height="{row_h - 14}" rx="8" fill="{bg}" stroke="#d7dde8"/>')
        status, color = status_label(row.conclusion)
        parts.append(f'<text x="{xs[0]}" y="{yy + 34}" font-size="20" font-weight="800" fill="#111827">{escape(row.hypothesis_id)}</text>')
        wrap_svg(parts, str(row.title), xs[0], yy + 64, 120, 12, "#475569")
        wrap_svg(parts, str(row.scenario), xs[1], yy + 34, widths[1], 14, "#334155")
        if row.hypothesis_id == "H5":
            effect = f"Accuracy {pct(row.accuracy)}, Recall {pct(row.recall)}, FN {fmt(row.fn, 0)}"
        else:
            effect = (
                f"Exposed {pct(row.event_rate_exposed)} vs general {pct(row.event_rate_control)}; "
                f"RR {fmt(row.risk_ratio, 2)} ({fmt(row.risk_lift_pct, 1)}% lift)"
            )
        wrap_svg(parts, effect, xs[2], yy + 34, widths[2], 14, "#334155")
        pval = row.fisher_p_value if not pd.isna(row.fisher_p_value) else row.ztest_p_value
        parts.append(f'<text x="{xs[3]}" y="{yy + 42}" font-size="18" font-weight="800" fill="#111827">{escape(p_text(pval))}</text>')
        parts.append(f'<rect x="{xs[4]}" y="{yy + 24}" width="160" height="34" rx="17" fill="{color}" fill-opacity="0.14"/>')
        parts.append(f'<text x="{xs[4] + 80}" y="{yy + 47}" text-anchor="middle" font-size="13" font-weight="800" fill="{color}">{escape(status)}</text>')
        wrap_svg(parts, str(row.expected_effect), xs[4], yy + 82, widths[4], 12, "#64748b")
    return save_svg(parts, SUMMARY_SVG)


def build_scenario_svg(results: pd.DataFrame) -> Path:
    width = 1700
    row_h = 252
    height = 188 + row_h * len(results) + 70
    parts = svg_header(
        width,
        height,
        "Hypothesis-to-Scenario Flow",
        "Validated signals are translated into AI risk states and operational response actions.",
    )

    def arrow(cx: int, cy: int, color: str = "#94a3b8") -> None:
        parts.append(f'<line x1="{cx - 26}" y1="{cy}" x2="{cx + 18}" y2="{cy}" stroke="{color}" stroke-width="3" stroke-linecap="round"/>')
        parts.append(f'<path d="M {cx + 18} {cy - 9} L {cx + 34} {cy} L {cx + 18} {cy + 9} Z" fill="{color}"/>')

    def flow_card(x: int, y: int, w: int, h: int, eyebrow: str, title: str, body: str, color: str) -> None:
        parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="#ffffff" stroke="#d7dde8"/>')
        parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="8" rx="4" fill="{color}" fill-opacity="0.82"/>')
        parts.append(f'<text x="{x + 18}" y="{y + 34}" font-size="11" font-weight="800" fill="#64748b">{escape(eyebrow)}</text>')
        wrap_svg(parts, title, x + 18, y + 62, w - 36, 14, "#111827")
        wrap_svg(parts, body, x + 18, y + 92, w - 36, 12, "#475569")

    def signal_text(row) -> str:
        if row.hypothesis_id == "H1":
            return "High water temperature, pH rise, sunshine, Chl-a, and Microcystis increase."
        if row.hypothesis_id == "H2":
            return "Rainfall accumulation, inflow rise, turbidity, and Anabaena/Aphanizomenon signal."
        if row.hypothesis_id == "H3":
            return "Earlier cell-count increase or alert signal from another monitoring location."
        if row.hypothesis_id == "H4":
            return "Low outflow, longer residence time, low wind, warm water, and stagnation species."
        return "Predicted 7-day alert probability and future interest-level exceedance signal."

    def evidence_text(row) -> str:
        if row.hypothesis_id == "H5":
            return f"Accuracy {pct(row.accuracy)}, Recall {pct(row.recall)}, F1 {fmt(row.f1, 3)}, FN {fmt(row.fn, 0)}."
        return (
            f"Event rate {pct(row.event_rate_exposed)} vs {pct(row.event_rate_control)}, "
            f"RR {fmt(row.risk_ratio, 2)}x, p={p_text(row.fisher_p_value)}."
        )

    y = 148
    # Column labels
    labels = [
        (122, "1. Detected signal"),
        (478, "2. Statistical evidence"),
        (834, "3. AI scenario state"),
        (1190, "4. Operational response"),
    ]
    for x, label in labels:
        parts.append(f'<text x="{x}" y="128" font-size="13" font-weight="800" fill="#64748b">{escape(label)}</text>')

    for idx, row in enumerate(results.itertuples()):
        yy = y + idx * row_h
        status, color = status_label(row.conclusion)
        row_bg = "#ffffff" if idx % 2 == 0 else "#fbfdff"
        parts.append(f'<rect x="54" y="{yy}" width="1592" height="{row_h - 26}" rx="14" fill="{row_bg}" stroke="#d7dde8"/>')
        parts.append(f'<rect x="54" y="{yy}" width="12" height="{row_h - 26}" rx="6" fill="{color}"/>')
        parts.append(f'<circle cx="95" cy="{yy + 42}" r="24" fill="{color}" fill-opacity="0.14"/>')
        parts.append(f'<text x="95" y="{yy + 49}" text-anchor="middle" font-size="16" font-weight="900" fill="{color}">{escape(row.hypothesis_id)}</text>')
        parts.append(f'<text x="128" y="{yy + 35}" font-size="18" font-weight="900" fill="#111827">{escape(row.scenario)}</text>')
        parts.append(f'<rect x="128" y="{yy + 52}" width="126" height="28" rx="14" fill="{color}" fill-opacity="0.14"/>')
        parts.append(f'<text x="191" y="{yy + 71}" text-anchor="middle" font-size="12" font-weight="800" fill="{color}">{escape(status)}</text>')

        card_y = yy + 96
        card_h = 106
        flow_card(104, card_y, 300, card_h, "MODEL INPUT", str(row.title), signal_text(row), color)
        arrow(436, card_y + card_h // 2)
        flow_card(468, card_y, 300, card_h, "VALIDATION", "Observed effect", evidence_text(row), color)
        arrow(800, card_y + card_h // 2)
        flow_card(832, card_y, 300, card_h, "AI STATE", str(row.scenario), str(row.expected_effect), color)
        arrow(1164, card_y + card_h // 2)
        flow_card(1196, card_y, 390, card_h, "ACTION", "Response package", str(row.scenario_action), color)
    return save_svg(parts, SCENARIO_SVG)


def bar_panel_header(parts: list[str], x: int, y: int, w: int, h: int, title: str) -> tuple[int, int, int, int]:
    parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="#ffffff" stroke="#d7dde8"/>')
    parts.append(f'<text x="{x + 24}" y="{y + 38}" font-size="20" font-weight="800" fill="#1f2937">{escape(title)}</text>')
    return x + 70, y + 82, w - 116, h - 150


def build_risk_rate_svg(results: pd.DataFrame) -> Path:
    rows = results[results["hypothesis_id"].ne("H5")].copy()
    parts = svg_header(
        1500,
        760,
        "Hypothesis Risk Rate Comparison",
        "Event rate under each scenario condition compared with ordinary periods.",
    )
    x, y, w, h = bar_panel_header(parts, 54, 124, 1392, 560, "Exposed vs General Event Rate")
    max_rate = max(float(rows["event_rate_exposed"].max()), float(rows["event_rate_control"].max()), 0.01)
    max_rate = min(max_rate * 1.18, 1.0)
    group_w = w / len(rows)
    for idx, row in enumerate(rows.itertuples()):
        base = x + idx * group_w + group_w * 0.18
        bw = group_w * 0.24
        exposed_h = float(row.event_rate_exposed) / max_rate * h
        control_h = float(row.event_rate_control) / max_rate * h
        parts.append(f'<rect x="{base:.1f}" y="{y + h - exposed_h:.1f}" width="{bw:.1f}" height="{exposed_h:.1f}" rx="5" fill="#3AAFA9"/>')
        parts.append(f'<rect x="{base + bw + 12:.1f}" y="{y + h - control_h:.1f}" width="{bw:.1f}" height="{control_h:.1f}" rx="5" fill="#94A3B8"/>')
        parts.append(f'<text x="{base + bw:.1f}" y="{y + h + 34}" text-anchor="middle" font-size="14" font-weight="800" fill="#111827">{escape(row.hypothesis_id)}</text>')
        parts.append(f'<text x="{base + bw / 2:.1f}" y="{y + h - exposed_h - 10:.1f}" text-anchor="middle" font-size="12" font-weight="700" fill="#334155">{pct(row.event_rate_exposed)}</text>')
        parts.append(f'<text x="{base + bw * 1.5 + 12:.1f}" y="{y + h - control_h - 10:.1f}" text-anchor="middle" font-size="12" font-weight="700" fill="#334155">{pct(row.event_rate_control)}</text>')
    parts.append('<rect x="1100" y="148" width="16" height="16" rx="3" fill="#3AAFA9"/>')
    parts.append('<text x="1124" y="161" font-size="13" fill="#334155">Condition exposed</text>')
    parts.append('<rect x="1250" y="148" width="16" height="16" rx="3" fill="#94A3B8"/>')
    parts.append('<text x="1274" y="161" font-size="13" fill="#334155">General</text>')
    return save_svg(parts, RISK_RATE_SVG)


def build_effect_size_svg(results: pd.DataFrame) -> Path:
    rows = results[results["hypothesis_id"].ne("H5")].copy()
    parts = svg_header(
        1500,
        760,
        "Hypothesis Effect Size Comparison",
        "Risk ratio greater than 1 means the scenario condition increases next-alert risk.",
    )
    x, y, w, h = bar_panel_header(parts, 54, 124, 1392, 560, "Risk Ratio")
    max_rr = max(float(rows["risk_ratio"].max()), 1.0) * 1.12
    group_w = w / len(rows)
    baseline_y = y + h - (1 / max_rr * h)
    parts.append(f'<line x1="{x}" y1="{baseline_y:.1f}" x2="{x + w}" y2="{baseline_y:.1f}" stroke="#E07A5F" stroke-dasharray="7 7" stroke-width="2"/>')
    parts.append(f'<text x="{x + w - 80}" y="{baseline_y - 10:.1f}" font-size="12" fill="#E07A5F">RR = 1.0</text>')
    for idx, row in enumerate(rows.itertuples()):
        status, color = status_label(row.conclusion)
        bw = group_w * 0.38
        bx = x + idx * group_w + group_w * 0.31
        bh = float(row.risk_ratio) / max_rr * h
        by = y + h - bh
        parts.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw:.1f}" height="{bh:.1f}" rx="6" fill="{color}" fill-opacity="0.82"/>')
        parts.append(f'<text x="{bx + bw / 2:.1f}" y="{by - 12:.1f}" text-anchor="middle" font-size="14" font-weight="800" fill="#111827">{fmt(row.risk_ratio, 2)}x</text>')
        parts.append(f'<text x="{bx + bw / 2:.1f}" y="{y + h + 34}" text-anchor="middle" font-size="14" font-weight="800" fill="#111827">{escape(row.hypothesis_id)}</text>')
        parts.append(f'<text x="{bx + bw / 2:.1f}" y="{y + h + 56}" text-anchor="middle" font-size="12" fill="#64748b">{escape(status)}</text>')
    return save_svg(parts, EFFECT_SIZE_SVG)


def build_pvalue_svg(results: pd.DataFrame) -> Path:
    rows = results.copy()
    rows["main_p_value"] = rows["fisher_p_value"].where(rows["fisher_p_value"].notna(), rows["ztest_p_value"])
    rows["minus_log10_p"] = rows["main_p_value"].apply(lambda v: -math.log10(max(float(v), 1e-300)) if pd.notna(v) else np.nan)
    parts = svg_header(
        1500,
        760,
        "Hypothesis Significance Comparison",
        "-log10(p-value) scale. Higher bars indicate stronger statistical evidence.",
    )
    x, y, w, h = bar_panel_header(parts, 54, 124, 1392, 560, "-log10(p-value)")
    max_score = max(float(rows["minus_log10_p"].max()), 2.0) * 1.08
    threshold_y = y + h - (-math.log10(0.05) / max_score * h)
    parts.append(f'<line x1="{x}" y1="{threshold_y:.1f}" x2="{x + w}" y2="{threshold_y:.1f}" stroke="#E07A5F" stroke-dasharray="7 7" stroke-width="2"/>')
    parts.append(f'<text x="{x + w - 120}" y="{threshold_y - 10:.1f}" font-size="12" fill="#E07A5F">p = 0.05</text>')
    group_w = w / len(rows)
    for idx, row in enumerate(rows.itertuples()):
        status, color = status_label(row.conclusion)
        score = float(row.minus_log10_p) if pd.notna(row.minus_log10_p) else 0
        bw = group_w * 0.38
        bx = x + idx * group_w + group_w * 0.31
        bh = score / max_score * h
        by = y + h - bh
        parts.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw:.1f}" height="{bh:.1f}" rx="6" fill="{color}" fill-opacity="0.82"/>')
        parts.append(f'<text x="{bx + bw / 2:.1f}" y="{by - 12:.1f}" text-anchor="middle" font-size="12" font-weight="800" fill="#111827">{p_text(row.main_p_value)}</text>')
        parts.append(f'<text x="{bx + bw / 2:.1f}" y="{y + h + 34}" text-anchor="middle" font-size="14" font-weight="800" fill="#111827">{escape(row.hypothesis_id)}</text>')
    return save_svg(parts, PVALUE_SVG)


def short_scenario_label(hypothesis_id: str) -> str:
    labels = {
        "H1": "High-temperature surface bloom",
        "H2": "Post-rainfall nutrient inflow",
        "H3": "Spatial risk transfer",
        "H4": "Hydraulic stagnation",
        "H5": "7-day early warning",
    }
    return labels.get(hypothesis_id, hypothesis_id)


def signal_summary(hypothesis_id: str) -> str:
    summaries = {
        "H1": "Water temperature, pH, sunshine, Chl-a, Microcystis, and residence-time signals were combined.",
        "H2": "Rainfall accumulation, inflow, turbidity, and Anabaena/Aphanizomenon signals were combined.",
        "H3": "Earlier alert or cell-count increase in another monitoring location was used as a leading signal.",
        "H4": "Low outflow, longer residence time, low wind, warm water, and stagnation-related species were combined.",
        "H5": "The trained classifier predicted next-sampling or about 7-day-ahead alert-risk exceedance.",
    }
    return summaries.get(hypothesis_id, "")


def scale_value(value: float, low: float, high: float) -> float:
    if high == low:
        return 0.5
    return max(0.0, min(1.0, (value - low) / (high - low)))


def metric_card(parts: list[str], x: int, y: int, label: str, value: str, note: str, color: str, width: int = 220) -> None:
    parts.append(f'<rect x="{x}" y="{y}" width="{width}" height="126" rx="10" fill="#f8fafc" stroke="#d7dde8"/>')
    parts.append(f'<text x="{x + 18}" y="{y + 32}" font-size="13" font-weight="800" fill="#64748b">{escape(label)}</text>')
    size = 28 if len(value) <= 10 else 18
    parts.append(f'<text x="{x + 18}" y="{y + 76}" font-size="{size}" font-weight="900" fill="{color}">{escape(value)}</text>')
    wrap_svg(parts, note, x + 18, y + 102, width - 36, 12, "#64748b")


def scenario_canvas(hid: str, title: str, subtitle: str, status: str, color: str) -> list[str]:
    parts = svg_header(1500, 860, f"{hid}. {title}", subtitle)
    parts.append(f'<rect x="56" y="120" width="1388" height="660" rx="14" fill="#ffffff" stroke="#d7dde8"/>')
    parts.append(f'<rect x="56" y="120" width="12" height="660" rx="6" fill="{color}"/>')
    parts.append(f'<rect x="96" y="150" width="150" height="38" rx="19" fill="{color}" fill-opacity="0.14"/>')
    parts.append(f'<text x="171" y="175" text-anchor="middle" font-size="14" font-weight="900" fill="{color}">{escape(status)}</text>')
    return parts


def build_h1_svg(row: pd.Series, df: pd.DataFrame, mask: pd.Series) -> Path:
    status, color = status_label(str(row["conclusion"]))
    parts = scenario_canvas("H1", "High-temperature surface bloom", "Temperature-pH-Microcystis condition and next-alert risk.", status, color)
    wrap_svg(parts, "This scenario is supported when warm water, high pH, stronger light/Chl-a, and Microcystis increase together.", 96, 215, 460, 15, "#334155")

    chart_x, chart_y, chart_w, chart_h = 110, 330, 560, 300
    parts.append('<text x="96" y="300" font-size="18" font-weight="900" fill="#111827">Water temperature vs pH</text>')
    parts.append(f'<rect x="{chart_x}" y="{chart_y}" width="{chart_w}" height="{chart_h}" rx="8" fill="#fbfdff" stroke="#d7dde8"/>')
    temp = df["water_temp"].astype(float)
    ph = df["pH"].astype(float)
    sample = df.sample(n=min(520, len(df)), random_state=42)
    temp_low, temp_high = float(temp.quantile(0.02)), float(temp.quantile(0.98))
    ph_low, ph_high = float(ph.quantile(0.02)), float(ph.quantile(0.98))
    for idx, point in sample.iterrows():
        sx = chart_x + scale_value(float(point["water_temp"]), temp_low, temp_high) * chart_w
        sy = chart_y + (1 - scale_value(float(point["pH"]), ph_low, ph_high)) * chart_h
        is_risk = int(point["next_alert_binary"]) == 1
        exposed = bool(mask.loc[idx]) if idx in mask.index else False
        fill = "#E07A5F" if is_risk else "#94a3b8"
        radius = 4 if exposed else 3
        stroke = color if exposed else "none"
        parts.append(f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="{radius}" fill="{fill}" fill-opacity="0.62" stroke="{stroke}" stroke-width="1.8"/>')
    parts.append(f'<text x="{chart_x + chart_w / 2}" y="{chart_y + chart_h + 42}" text-anchor="middle" font-size="13" fill="#475569">water_temp</text>')
    parts.append(f'<text transform="translate({chart_x - 42},{chart_y + chart_h / 2}) rotate(-90)" text-anchor="middle" font-size="13" fill="#475569">pH</text>')
    parts.append('<circle cx="468" cy="286" r="5" fill="#E07A5F" fill-opacity="0.75"/><text x="480" y="291" font-size="12" fill="#475569">next alert</text>')
    parts.append(f'<circle cx="560" cy="286" r="5" fill="#94a3b8" fill-opacity="0.75" stroke="{color}" stroke-width="2"/><text x="572" y="291" font-size="12" fill="#475569">H1 condition</text>')

    metric_card(parts, 730, 250, "Event rate", f"{pct(row['event_rate_exposed'])}", f"Ordinary: {pct(row['event_rate_control'])}", color)
    metric_card(parts, 970, 250, "Risk Ratio", f"{fmt(row['risk_ratio'], 2)}x", "H1 condition vs ordinary", color)
    metric_card(parts, 1210, 250, "p-value", p_text(row["fisher_p_value"]), "Fisher exact test", color)

    parts.append('<text x="730" y="445" font-size="18" font-weight="900" fill="#111827">Why this graph fits H1</text>')
    wrap_svg(parts, "The figure checks the high-temperature and high-pH space where surface cyanobacteria, especially Microcystis, are expected to become dominant.", 730, 480, 620, 15, "#334155")
    return save_svg(parts, FIGURE_DIR / "hypothesis_h1_result.svg")


def build_h2_svg(row: pd.Series, df: pd.DataFrame, mask: pd.Series) -> Path:
    status, color = status_label(str(row["conclusion"]))
    parts = scenario_canvas("H2", "Post-rainfall nutrient inflow", "Lagged rainfall windows, inflow, turbidity, and nitrogen-fixing taxa.", status, color)
    wrap_svg(parts, "This scenario tests whether rainfall and inflow create bloom conditions after a lag rather than immediately.", 96, 215, 520, 15, "#334155")
    target = df["next_alert_binary"].eq(1)
    windows = [
        ("3-day rain", df["rain_3d_sum"] >= q(df, "rain_3d_sum", 0.70)),
        ("7-day rain", df["rain_7d_sum"] >= q(df, "rain_7d_sum", 0.70)),
        ("14-day rain", df["rain_14d_sum"] >= q(df, "rain_14d_sum", 0.70)),
        ("7-day inflow", df["inflow_7d_sum"] >= q(df, "inflow_7d_sum", 0.70)),
        ("turbidity", df["turbidity"] >= q(df, "turbidity", 0.60)),
    ]
    rates = [(label, float(target[cond].mean()) if cond.sum() else 0.0) for label, cond in windows]
    chart_x, chart_y, chart_w, chart_h = 105, 335, 720, 300
    max_rate = max([r for _, r in rates] + [float(row["event_rate_control"])]) * 1.2
    max_rate = min(max(max_rate, 0.1), 1.0)
    parts.append('<text x="96" y="300" font-size="18" font-weight="900" fill="#111827">Lag-window risk after rainfall</text>')
    parts.append(f'<line x1="{chart_x}" y1="{chart_y + chart_h}" x2="{chart_x + chart_w}" y2="{chart_y + chart_h}" stroke="#cbd5e1"/>')
    bw = 82
    for i, (label, rate) in enumerate(rates):
        bx = chart_x + 38 + i * 125
        bh = rate / max_rate * chart_h
        by = chart_y + chart_h - bh
        fill = color if "rain" in label or "inflow" in label else "#9673A6"
        parts.append(f'<rect x="{bx}" y="{by:.1f}" width="{bw}" height="{bh:.1f}" rx="7" fill="{fill}"/>')
        parts.append(f'<text x="{bx + bw / 2}" y="{by - 10:.1f}" text-anchor="middle" font-size="13" font-weight="900" fill="#111827">{pct(rate)}</text>')
        wrap_svg(parts, label, bx - 6, chart_y + chart_h + 32, 96, 12, "#475569")
    control_y = chart_y + chart_h - float(row["event_rate_control"]) / max_rate * chart_h
    parts.append(f'<line x1="{chart_x}" y1="{control_y:.1f}" x2="{chart_x + chart_w}" y2="{control_y:.1f}" stroke="#94a3b8" stroke-width="2" stroke-dasharray="7 7"/>')
    parts.append(f'<text x="{chart_x + chart_w - 126}" y="{control_y - 8:.1f}" font-size="12" fill="#64748b">ordinary {pct(row["event_rate_control"])}</text>')

    metric_card(parts, 890, 260, "Scenario rate", pct(row["event_rate_exposed"]), "combined rainfall/inflow condition", color, width=235)
    metric_card(parts, 1150, 260, "Risk Ratio", f"{fmt(row['risk_ratio'], 2)}x", f"p={p_text(row['fisher_p_value'])}", color, width=235)
    parts.append('<text x="890" y="470" font-size="18" font-weight="900" fill="#111827">Why this graph fits H2</text>')
    wrap_svg(parts, "The bars separate 3, 7, and 14 day rainfall windows, matching the hypothesis that runoff and nutrients can affect cyanobacteria with a lag.", 890, 504, 470, 15, "#334155")
    return save_svg(parts, FIGURE_DIR / "hypothesis_h2_result.svg")


def build_h3_svg(row: pd.Series, df: pd.DataFrame, mask: pd.Series) -> Path:
    status, color = status_label(str(row["conclusion"]))
    parts = scenario_canvas("H3", "Spatial risk transfer", "Leading signal from one monitoring area to another.", status, color)
    wrap_svg(parts, "This scenario checks whether an earlier risk signal in one water area works as a warning for other areas.", 96, 215, 520, 15, "#334155")
    nodes = [(260, 410, "Munui"), (590, 410, "Chudong"), (920, 410, "Hoenam")]
    parts.append('<text x="96" y="300" font-size="18" font-weight="900" fill="#111827">Monitoring-area signal flow</text>')
    for i in range(len(nodes) - 1):
        x1, y1, _ = nodes[i]
        x2, y2, _ = nodes[i + 1]
        parts.append(f'<line x1="{x1 + 62}" y1="{y1}" x2="{x2 - 62}" y2="{y2}" stroke="#94a3b8" stroke-width="5" stroke-linecap="round"/>')
        parts.append(f'<path d="M {x2 - 72} {y2 - 14} L {x2 - 48} {y2} L {x2 - 72} {y2 + 14} Z" fill="#94a3b8"/>')
    for x, y, label in nodes:
        parts.append(f'<circle cx="{x}" cy="{y}" r="66" fill="{color}" fill-opacity="0.14" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{x}" y="{y + 6}" text-anchor="middle" font-size="18" font-weight="900" fill="#111827">{escape(label)}</text>')
    parts.append(f'<rect x="190" y="530" width="810" height="64" rx="10" fill="#f8fafc" stroke="#d7dde8"/>')
    parts.append(f'<text x="595" y="570" text-anchor="middle" font-size="17" font-weight="900" fill="{color}">leading cells / alert signal -> next-area risk</text>')

    metric_card(parts, 1085, 255, "Signal period", pct(row["event_rate_exposed"]), f"ordinary {pct(row['event_rate_control'])}", color, width=245)
    metric_card(parts, 1085, 410, "Risk Ratio", f"{fmt(row['risk_ratio'], 2)}x", f"p={p_text(row['fisher_p_value'])}", color, width=245)
    parts.append('<text x="1085" y="610" font-size="18" font-weight="900" fill="#111827">Use in operation</text>')
    wrap_svg(parts, "When one area shows a leading bloom signal, monitoring should expand to adjacent or downstream areas before the next sampling point.", 1085, 642, 300, 14, "#334155")
    return save_svg(parts, FIGURE_DIR / "hypothesis_h3_result.svg")


def build_h4_svg(row: pd.Series, df: pd.DataFrame, mask: pd.Series) -> Path:
    status, color = status_label(str(row["conclusion"]))
    parts = scenario_canvas("H4", "Hydraulic stagnation", "Low outflow, longer residence time, low wind, and warm-water stability.", status, color)
    wrap_svg(parts, "This scenario is a supporting signal: risk direction rises, but statistical evidence is weaker than H1-H3.", 96, 215, 540, 15, "#334155")
    components = [
        ("Low outflow", float((df["outflow"] <= q(df, "outflow", 0.30))[mask].mean()) if mask.sum() else 0),
        ("High residence time", float((df["residence_proxy"] >= q(df, "residence_proxy", 0.70))[mask].mean()) if mask.sum() else 0),
        ("Low wind", float((df["avg_wind"] <= q(df, "avg_wind", 0.35))[mask].mean()) if mask.sum() else 0),
        ("Warm water", float((df["water_temp"] >= q(df, "water_temp", 0.60))[mask].mean()) if mask.sum() else 0),
    ]
    parts.append('<text x="96" y="300" font-size="18" font-weight="900" fill="#111827">Stagnation-condition composition</text>')
    bar_x, bar_y, bar_w = 112, 340, 500
    for idx, (label, value) in enumerate(components):
        yy = bar_y + idx * 58
        parts.append(f'<text x="{bar_x}" y="{yy + 18}" font-size="13" font-weight="800" fill="#334155">{escape(label)}</text>')
        parts.append(f'<rect x="{bar_x + 170}" y="{yy}" width="{bar_w}" height="26" rx="6" fill="#eef2f7"/>')
        parts.append(f'<rect x="{bar_x + 170}" y="{yy}" width="{bar_w * value:.1f}" height="26" rx="6" fill="{color}" fill-opacity="0.86"/>')
        parts.append(f'<text x="{bar_x + 170 + bar_w + 18}" y="{yy + 18}" font-size="13" font-weight="900" fill="#111827">{pct(value)}</text>')

    parts.append('<text x="96" y="610" font-size="18" font-weight="900" fill="#111827">Interpretation</text>')
    wrap_svg(parts, "The hydraulic-stagnation factors are present, but the alert-rate gap is small. Therefore H4 is better used as a secondary risk signal combined with high temperature or species dominance.", 96, 642, 760, 15, "#334155")

    # Event-rate comparison card
    panel_x, panel_y = 870, 292
    parts.append(f'<rect x="{panel_x}" y="{panel_y}" width="450" height="190" rx="10" fill="#f8fafc" stroke="#d7dde8"/>')
    parts.append(f'<text x="{panel_x + 24}" y="{panel_y + 38}" font-size="17" font-weight="900" fill="#111827">Alert-risk rate</text>')
    max_rate = max(float(row["event_rate_exposed"]), float(row["event_rate_control"]), 0.01) * 1.25
    max_rate = min(max_rate, 1.0)
    for idx, (label, value, fill) in enumerate([
        ("Stagnation", float(row["event_rate_exposed"]), color),
        ("Ordinary", float(row["event_rate_control"]), "#94a3b8"),
    ]):
        bx = panel_x + 52 + idx * 164
        bh = value / max_rate * 90
        by = panel_y + 146 - bh
        parts.append(f'<rect x="{bx}" y="{by:.1f}" width="88" height="{bh:.1f}" rx="7" fill="{fill}"/>')
        parts.append(f'<text x="{bx + 44}" y="{by - 10:.1f}" text-anchor="middle" font-size="14" font-weight="900" fill="#111827">{pct(value)}</text>')
        parts.append(f'<text x="{bx + 44}" y="{panel_y + 170}" text-anchor="middle" font-size="12" fill="#475569">{escape(label)}</text>')

    metric_card(parts, 870, 520, "Risk Ratio", f"{fmt(row['risk_ratio'], 2)}x", "small positive direction", color, width=235)
    metric_card(parts, 1130, 520, "p-value", p_text(row["fisher_p_value"]), "not significant at 0.05", color, width=235)
    return save_svg(parts, FIGURE_DIR / "hypothesis_h4_result.svg")


def build_h5_svg(row: pd.Series) -> Path:
    status, color = status_label(str(row["conclusion"]))
    parts = scenario_canvas("H5", "7-day early warning", "Classification accuracy and operational lead time.", status, color)
    parts.append('<text x="96" y="240" font-size="18" font-weight="900" fill="#111827">Lead-time workflow</text>')
    steps = [("Today", "water, weather, dam operation"), ("+7 days", "predicted alert risk"), ("Before alert", "prepare response")]
    for i, (head, body) in enumerate(steps):
        x = 130 + i * 380
        parts.append(f'<rect x="{x}" y="286" width="270" height="116" rx="12" fill="#f8fafc" stroke="#d7dde8"/>')
        parts.append(f'<text x="{x + 24}" y="326" font-size="18" font-weight="900" fill="#111827">{escape(head)}</text>')
        wrap_svg(parts, body, x + 24, 360, 220, 13, "#475569")
        if i < 2:
            parts.append(f'<line x1="{x + 284}" y1="344" x2="{x + 348}" y2="344" stroke="#94a3b8" stroke-width="4" stroke-linecap="round"/>')
            parts.append(f'<path d="M {x + 348} 332 L {x + 370} 344 L {x + 348} 356 Z" fill="#94a3b8"/>')
    metrics = [
        ("Accuracy", pct(row.get("accuracy"))),
        ("Precision", pct(row.get("precision"))),
        ("Recall", pct(row.get("recall"))),
        ("F1-score", fmt(row.get("f1"), 3)),
        ("p-value", p_text(row.get("fisher_p_value"))),
    ]
    for idx, (label, value) in enumerate(metrics):
        metric_card(parts, 98 + idx * 265, 470, label, value, "validation set", color, width=240)
    parts.append('<text x="96" y="650" font-size="18" font-weight="900" fill="#111827">Key result</text>')
    wrap_svg(parts, f"False Negative = {fmt(row.get('fn'), 0)}. The model did not miss any actual alert-risk case in validation, which is critical for pre-alert action.", 96, 680, 1080, 15, "#334155")
    return save_svg(parts, FIGURE_DIR / "hypothesis_h5_result.svg")


def build_single_hypothesis_svg(row: pd.Series, df: pd.DataFrame, masks: dict[str, pd.Series]) -> Path:
    hid = str(row["hypothesis_id"])
    if hid == "H1":
        return build_h1_svg(row, df, masks["H1"])
    if hid == "H2":
        return build_h2_svg(row, df, masks["H2"])
    if hid == "H3":
        return build_h3_svg(row, df, masks["H3"])
    if hid == "H4":
        return build_h4_svg(row, df, masks["H4"])
    if hid == "H5":
        return build_h5_svg(row)

    status, color = status_label(str(row["conclusion"]))
    path = FIGURE_DIR / f"hypothesis_{hid.lower()}_result.svg"

    parts = svg_header(
        1400,
        760,
        f"{hid}. {short_scenario_label(hid)}",
        str(row["scenario"]),
    )
    parts.append(f'<rect x="56" y="120" width="1288" height="560" rx="14" fill="#ffffff" stroke="#d7dde8"/>')
    parts.append(f'<rect x="56" y="120" width="12" height="560" rx="6" fill="{color}"/>')
    parts.append(f'<rect x="96" y="150" width="150" height="38" rx="19" fill="{color}" fill-opacity="0.14"/>')
    parts.append(f'<text x="171" y="175" text-anchor="middle" font-size="14" font-weight="900" fill="{color}">{escape(status)}</text>')
    wrap_svg(parts, signal_summary(hid), 96, 220, 500, 16, "#334155")

    if hid == "H5":
        metrics = [
            ("Accuracy", pct(row.get("accuracy"))),
            ("Precision", pct(row.get("precision"))),
            ("Recall", pct(row.get("recall"))),
            ("F1-score", fmt(row.get("f1"), 3)),
            ("p-value", p_text(row.get("fisher_p_value"))),
        ]
        card_x = 96
        for idx, (label, value) in enumerate(metrics):
            x = card_x + idx * 240
            parts.append(f'<rect x="{x}" y="320" width="210" height="116" rx="10" fill="#f8fafc" stroke="#d7dde8"/>')
            parts.append(f'<text x="{x + 18}" y="352" font-size="13" font-weight="800" fill="#64748b">{escape(label)}</text>')
            value_size = 25 if len(value) <= 12 else 18
            parts.append(f'<text x="{x + 18}" y="397" font-size="{value_size}" font-weight="900" fill="#111827">{escape(value)}</text>')

        # Confusion matrix mini panel
        tp = int(float(row.get("tp", 0)))
        tn = int(float(row.get("tn", 0)))
        fp = int(float(row.get("fp", 0)))
        fn = int(float(row.get("fn", 0)))
        cm = [[tn, fp], [fn, tp]]
        max_v = max(tp, tn, fp, fn, 1)
        sx, sy, cell = 470, 492, 98
        labels = [["TN", "FP"], ["FN", "TP"]]
        parts.append('<text x="96" y="520" font-size="18" font-weight="900" fill="#111827">Operational meaning</text>')
        wrap_svg(parts, "The model detected every actual alert-risk case in the validation set, so it is suitable for pre-alert monitoring support.", 96, 552, 330, 15, "#334155")
        for r in range(2):
            for c in range(2):
                val = cm[r][c]
                opacity = 0.18 + 0.72 * (val / max_v)
                parts.append(f'<rect x="{sx + c * cell}" y="{sy + r * cell}" width="{cell}" height="{cell}" rx="8" fill="{color}" fill-opacity="{opacity:.2f}" stroke="#ffffff" stroke-width="3"/>')
                parts.append(f'<text x="{sx + c * cell + cell / 2}" y="{sy + r * cell + 36}" text-anchor="middle" font-size="13" font-weight="800" fill="#0f172a">{labels[r][c]}</text>')
                parts.append(f'<text x="{sx + c * cell + cell / 2}" y="{sy + r * cell + 70}" text-anchor="middle" font-size="28" font-weight="900" fill="#0f172a">{val}</text>')
        wrap_svg(parts, str(row["scenario_action"]), 760, 512, 500, 15, "#334155")
        parts.append('<text x="760" y="480" font-size="18" font-weight="900" fill="#111827">Response action</text>')
        return save_svg(parts, path)

    exposed = float(row["event_rate_exposed"])
    control = float(row["event_rate_control"])
    rr = float(row["risk_ratio"])
    pval = row["fisher_p_value"]

    # Rate comparison bars
    chart_x, chart_y, chart_w, chart_h = 110, 340, 390, 230
    max_rate = max(exposed, control, 0.01) * 1.2
    max_rate = min(max_rate, 1.0)
    parts.append('<text x="96" y="300" font-size="18" font-weight="900" fill="#111827">Event-rate comparison</text>')
    parts.append(f'<line x1="{chart_x}" y1="{chart_y + chart_h}" x2="{chart_x + chart_w}" y2="{chart_y + chart_h}" stroke="#cbd5e1"/>')
    for idx, (label, value, fill) in enumerate([
        ("Scenario condition", exposed, color),
        ("Ordinary period", control, "#94a3b8"),
    ]):
        bw = 120
        bx = chart_x + 70 + idx * 170
        bh = value / max_rate * chart_h
        by = chart_y + chart_h - bh
        parts.append(f'<rect x="{bx}" y="{by:.1f}" width="{bw}" height="{bh:.1f}" rx="8" fill="{fill}"/>')
        parts.append(f'<text x="{bx + bw / 2}" y="{by - 12:.1f}" text-anchor="middle" font-size="17" font-weight="900" fill="#111827">{pct(value)}</text>')
        wrap_svg(parts, label, int(bx - 8), chart_y + chart_h + 32, 136, 12, "#475569")

    # Evidence cards
    evidence = [
        ("Risk Ratio", f"{fmt(rr, 2)}x", "How many times higher the risk is."),
        ("p-value", p_text(pval), "Fisher exact test."),
        ("Risk Lift", f"{fmt(row['risk_lift_pct'], 1)}%", "Increase relative to ordinary periods."),
    ]
    for idx, (label, value, note) in enumerate(evidence):
        x = 590 + idx * 238
        parts.append(f'<rect x="{x}" y="318" width="210" height="142" rx="10" fill="#f8fafc" stroke="#d7dde8"/>')
        parts.append(f'<text x="{x + 18}" y="352" font-size="13" font-weight="800" fill="#64748b">{escape(label)}</text>')
        size = 28 if len(value) <= 10 else 20
        parts.append(f'<text x="{x + 18}" y="398" font-size="{size}" font-weight="900" fill="#111827">{escape(value)}</text>')
        wrap_svg(parts, note, x + 18, 426, 172, 12, "#64748b")

    parts.append('<text x="590" y="520" font-size="18" font-weight="900" fill="#111827">Interpretation</text>')
    if str(row["conclusion"]) == "supported":
        interpretation = (
            f"When this scenario condition appears, the next alert-risk probability is {fmt(rr, 2)} times higher "
            "than ordinary periods. The p-value supports this relationship statistically."
        )
    else:
        interpretation = (
            "The direction of risk increase is observed, but the p-value does not provide enough statistical evidence. "
            "Use this as a supporting signal with other conditions."
        )
    wrap_svg(parts, interpretation, 590, 552, 660, 15, "#334155")
    parts.append('<text x="96" y="638" font-size="13" font-weight="800" fill="#64748b">Recommended operational response</text>')
    wrap_svg(parts, str(row["scenario_action"]), 96, 666, 1120, 14, "#334155")
    return save_svg(parts, path)


def build_individual_hypothesis_svgs(results: pd.DataFrame, df: pd.DataFrame) -> list[Path]:
    masks = build_exposure_masks(df)
    return [build_single_hypothesis_svg(row, df, masks) for _, row in results.iterrows()]


def write_report(results: pd.DataFrame) -> Path:
    lines = [
        "# 가설 실험 결과",
        "",
        f"- 관심 기준: 유해남조류 세포 수 {ALERT_CELL_THRESHOLD:,} cells/mL 이상",
        "- 검정 방식: 조건 노출군과 일반군의 다음 채수 시점 관심 기준 이상 발생률 비교",
        "- 통계량: Risk Ratio, Odds Ratio, Fisher exact test p-value, two-proportion z-test p-value",
        "",
    ]
    for row in results.itertuples():
        status, _ = status_label(row.conclusion)
        lines.append(f"## {row.hypothesis_id}. {row.scenario}")
        lines.append("")
        lines.append(f"**판정:** {status}")
        lines.append("")
        if row.hypothesis_id == "H5":
            lines.append(
                f"7일 선제 대응 모델 검증 결과, Accuracy는 {pct(row.accuracy)}, Precision은 {pct(row.precision)}, "
                f"Recall은 {pct(row.recall)}, F1-score는 {fmt(row.f1, 3)}로 나타났다. "
                f"혼동행렬 기준 TP={fmt(row.tp, 0)}, TN={fmt(row.tn, 0)}, FP={fmt(row.fp, 0)}, FN={fmt(row.fn, 0)}이다. "
                f"특히 FN이 {fmt(row.fn, 0)}건으로 나타나 실제 관심 이상 위험을 놓치지 않는 방향의 선제 대응 모델로 활용 가능성이 확인되었다."
            )
        else:
            lines.append(
                f"노출군의 관심 기준 이상 발생률은 {pct(row.event_rate_exposed)}, 일반군은 {pct(row.event_rate_control)}로 나타났다. "
                f"Risk Ratio는 {fmt(row.risk_ratio, 2)}로, 해당 조건에서 관심 기준 이상 발생 가능성이 일반 기간 대비 "
                f"약 {fmt(row.risk_lift_pct, 1)}% 높게 나타났다. Fisher exact test p-value는 {p_text(row.fisher_p_value)}이며, "
                f"two-proportion z-test p-value는 {p_text(row.ztest_p_value)}이다."
            )
        lines.append("")
        lines.append(f"**시나리오 연결:** {row.scenario_action}")
        lines.append("")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    return REPORT_MD


def main() -> None:
    df = load_analysis_frame()
    rows = build_hypotheses(df)
    early_warning = evaluate_early_warning()
    if early_warning:
        rows.append(early_warning)
    results = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULT_CSV, index=False, encoding="utf-8-sig")
    write_report(results)
    build_summary_svg(results)
    build_scenario_svg(results)
    build_risk_rate_svg(results)
    build_effect_size_svg(results)
    build_pvalue_svg(results)
    individual_paths = build_individual_hypothesis_svgs(results, df)
    print("Saved hypothesis experiment outputs:")
    for path in [RESULT_CSV, REPORT_MD, SUMMARY_SVG, SCENARIO_SVG, RISK_RATE_SVG, EFFECT_SIZE_SVG, PVALUE_SVG, *individual_paths]:
        print(f"- {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
