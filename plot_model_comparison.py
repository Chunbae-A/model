from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd


METRICS_PATH = Path("artifacts/models/candidate_model_metrics.csv")
OUTPUT_DIR = Path("artifacts/figures")
SVG_OUTPUT_PATH = OUTPUT_DIR / "model_comparison.svg"


MODEL_LABELS = {
    "hist_gradient_boosting": "HistGB",
    "random_forest": "Random Forest",
    "huber_regressor": "Huber",
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
    "catboost": "CatBoost",
    "stacking_ensemble": "Stacking Ensemble",
    "persistence_baseline": "Persistence",
}

MODEL_COLORS = {
    "HistGB": "#6C8EBF",
    "Random Forest": "#82B366",
    "Huber": "#E07A5F",
    "LightGBM": "#B85450",
    "XGBoost": "#9673A6",
    "CatBoost": "#D6B656",
    "Stacking Ensemble": "#3AAFA9",
    "Persistence": "#9E9E9E",
}


def load_metrics(path: Path = METRICS_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")
    df = pd.read_csv(path)
    visible_models = set(MODEL_LABELS)
    df = df[df["model_name"].isin(visible_models)].copy()
    df["model_label"] = df["model_name"].map(MODEL_LABELS).fillna(df["model_name"])
    return df


def fmt_value(value: float, metric: str) -> str:
    if pd.isna(value):
        return ""
    if metric.endswith("cells"):
        return f"{value:,.0f}"
    return f"{value:.3f}"


def draw_bar_panel(
    parts: list[str],
    *,
    x: int,
    y: int,
    width: int,
    height: int,
    title: str,
    rows: list[tuple[str, float, str]],
    lower_is_better: bool,
    value_fmt_metric: str,
) -> None:
    parts.append(f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="14" fill="#ffffff" stroke="#d7dde8"/>')
    parts.append(f'<text x="{x + 20}" y="{y + 32}" font-size="18" font-weight="700" fill="#1f2937">{escape(title)}</text>')

    chart_x = x + 52
    chart_y = y + 62
    chart_w = width - 88
    chart_h = height - 128
    label_y = y + height - 38
    max_value = max([value for _, value, _ in rows if pd.notna(value)] or [1.0])
    min_value = 0.0
    if not lower_is_better and max_value <= 1.05:
        min_value = 0.85
        max_value = 1.02

    parts.append(f'<line x1="{chart_x}" y1="{chart_y + chart_h}" x2="{chart_x + chart_w}" y2="{chart_y + chart_h}" stroke="#cbd5e1"/>')
    parts.append(f'<line x1="{chart_x}" y1="{chart_y}" x2="{chart_x}" y2="{chart_y + chart_h}" stroke="#cbd5e1"/>')

    gap = 14
    bar_w = max(20, int((chart_w - gap * (len(rows) - 1)) / len(rows)))
    for idx, (label, value, color) in enumerate(rows):
        scaled = 0 if max_value == min_value else (value - min_value) / (max_value - min_value)
        scaled = max(0.0, min(1.0, scaled))
        bar_h = scaled * chart_h
        bx = chart_x + idx * (bar_w + gap)
        by = chart_y + chart_h - bar_h
        parts.append(f'<rect x="{bx}" y="{by:.1f}" width="{bar_w}" height="{bar_h:.1f}" rx="5" fill="{color}"/>')
        parts.append(
            f'<text x="{bx + bar_w / 2:.1f}" y="{by - 7:.1f}" text-anchor="middle" '
            f'font-size="11" font-weight="700" fill="#334155">{escape(fmt_value(value, value_fmt_metric))}</text>'
        )
        parts.append(
            f'<text transform="translate({bx + bar_w / 2:.1f},{label_y}) rotate(-24)" '
            f'text-anchor="end" font-size="11" fill="#475569">{escape(label)}</text>'
        )


def build_svg_plot(metrics_df: pd.DataFrame, output_path: Path = SVG_OUTPUT_PATH) -> Path:
    reg = metrics_df[metrics_df["task"].eq("regression")].copy()
    cls = metrics_df[metrics_df["task"].eq("classification")].copy()

    def rows_for(df: pd.DataFrame, metric: str) -> list[tuple[str, float, str]]:
        rows = []
        for _, row in df.iterrows():
            label = row["model_label"]
            rows.append((label, float(row[metric]), MODEL_COLORS.get(label, "#64748b")))
        return rows

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1500" height="950" viewBox="0 0 1500 950">',
        '<rect width="1500" height="950" fill="#f5f7fb"/>',
        '<text x="60" y="58" font-size="30" font-weight="800" fill="#111827">Daecheong Algae Forecast Model Comparison</text>',
        '<text x="60" y="86" font-size="15" fill="#64748b">Validation metrics from artifacts/models/candidate_model_metrics.csv</text>',
    ]

    draw_bar_panel(
        parts,
        x=50,
        y=120,
        width=680,
        height=360,
        title="Regression RMSE, log10(cells + 1)",
        rows=rows_for(reg, "rmse"),
        lower_is_better=True,
        value_fmt_metric="rmse",
    )
    draw_bar_panel(
        parts,
        x=770,
        y=120,
        width=680,
        height=360,
        title="Regression RMSE, cells/mL",
        rows=rows_for(reg, "rmse_cells"),
        lower_is_better=True,
        value_fmt_metric="rmse_cells",
    )
    draw_bar_panel(
        parts,
        x=50,
        y=530,
        width=680,
        height=360,
        title="Classification Recall, alert detection",
        rows=rows_for(cls, "recall"),
        lower_is_better=False,
        value_fmt_metric="recall",
    )
    draw_bar_panel(
        parts,
        x=770,
        y=530,
        width=680,
        height=360,
        title="Classification F1, balance of precision and recall",
        rows=rows_for(cls, "f1"),
        lower_is_better=False,
        value_fmt_metric="f1",
    )

    parts.append(
        '<text x="60" y="925" font-size="13" fill="#64748b">'
        'Regression target: next_log_cells. Classification target: next_alert_binary. Lower RMSE is better; higher Recall/F1 is better.'
        '</text>'
    )
    parts.append("</svg>")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(parts), encoding="utf-8")
    return output_path


def main() -> None:
    metrics = load_metrics()
    output = build_svg_plot(metrics)
    print(f"Saved model comparison figure: {output}")


if __name__ == "__main__":
    main()
