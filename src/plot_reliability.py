from __future__ import annotations

from pathlib import Path
from statistics import mean
import textwrap
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_curve


ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = ROOT / "artifacts" / "models" / "candidate_model_metrics.csv"
THRESHOLD_CANDIDATES_PATH = ROOT / "artifacts" / "metrics" / "classification_threshold_candidates.csv"
CLASSIFICATION_PREDICTIONS_PATH = ROOT / "artifacts" / "predictions" / "classification_predictions.csv"
REGRESSION_PREDICTIONS_PATH = ROOT / "artifacts" / "predictions" / "regression_predictions.csv"
OUTPUT_DIR = ROOT / "artifacts" / "figures"


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

LOCATION_LABELS = {
    0: "Munui",
    1: "Chudong",
    2: "Hoenam",
}


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}\n"
            "Run the training pipeline first, then run this reliability plot script."
        )
    return pd.read_csv(path)


def fmt(value: float, digits: int = 3) -> str:
    if pd.isna(value):
        return "-"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    return f"{value:.{digits}f}"


def percent(value: float) -> str:
    if pd.isna(value):
        return "-"
    return f"{value * 100:.2f}%"


def location_label(value: object) -> str:
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return str(value)
    return LOCATION_LABELS.get(numeric, f"Location {numeric}")


def svg_header(width: int, height: int, title: str, subtitle: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="#f5f7fb"/>',
        '<style>'
        'text{font-family:Inter,Segoe UI,Arial,sans-serif}'
        '.muted{fill:#64748b}'
        '.ink{fill:#111827}'
        '.panel{fill:#ffffff;stroke:#d7dde8}'
        '</style>',
        f'<text x="52" y="58" font-size="30" font-weight="800" fill="#111827">{escape(title)}</text>',
        f'<text x="52" y="86" font-size="15" fill="#64748b">{escape(subtitle)}</text>',
    ]


def save_svg(parts: list[str], path: Path) -> Path:
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def card(parts: list[str], x: int, y: int, w: int, h: int, title: str, value: str, note: str = "") -> None:
    parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="#ffffff" stroke="#d7dde8"/>')
    parts.append(f'<text x="{x + 18}" y="{y + 30}" font-size="14" font-weight="700" fill="#475569">{escape(title)}</text>')
    value_size = 28 if len(value) <= 12 else 21
    parts.append(f'<text x="{x + 18}" y="{y + 70}" font-size="{value_size}" font-weight="800" fill="#111827">{escape(value)}</text>')
    if note:
        add_wrapped_text(parts, note, x + 18, y + 98, max_chars=max(18, int(w / 8)), font_size=12, fill="#64748b")


def add_wrapped_text(
    parts: list[str],
    text: str,
    x: int,
    y: int,
    *,
    max_chars: int,
    font_size: int = 14,
    fill: str = "#334155",
    line_height: int | None = None,
    weight: str | None = None,
) -> int:
    line_height = line_height or int(font_size * 1.45)
    lines = textwrap.wrap(text, width=max_chars, break_long_words=False, break_on_hyphens=False) or [""]
    weight_attr = f' font-weight="{weight}"' if weight else ""
    parts.append(f'<text x="{x}" y="{y}" font-size="{font_size}" fill="{fill}"{weight_attr}>')
    for idx, line in enumerate(lines):
        dy = 0 if idx == 0 else line_height
        parts.append(f'<tspan x="{x}" dy="{dy}">{escape(line)}</tspan>')
    parts.append("</text>")
    return y + (len(lines) - 1) * line_height


def best_row(metrics: pd.DataFrame, task: str, metric: str, higher_is_better: bool) -> pd.Series:
    subset = metrics[metrics["task"].eq(task)].copy()
    subset = subset[subset[metric].notna()]
    if subset.empty:
        raise ValueError(f"No metric rows found for task={task}, metric={metric}")
    idx = subset[metric].idxmax() if higher_is_better else subset[metric].idxmin()
    return subset.loc[idx]


def scaled(value: float, low: float, high: float) -> float:
    if high == low:
        return 0.5
    return max(0.0, min(1.0, (value - low) / (high - low)))


def draw_axis_box(parts: list[str], x: int, y: int, w: int, h: int) -> None:
    parts.append(f'<line x1="{x}" y1="{y + h}" x2="{x + w}" y2="{y + h}" stroke="#cbd5e1"/>')
    parts.append(f'<line x1="{x}" y1="{y}" x2="{x}" y2="{y + h}" stroke="#cbd5e1"/>')
    for tick in [0, 0.25, 0.5, 0.75, 1.0]:
        tx = x + tick * w
        ty = y + (1 - tick) * h
        parts.append(f'<line x1="{tx:.1f}" y1="{y}" x2="{tx:.1f}" y2="{y + h}" stroke="#eef2f7"/>')
        parts.append(f'<line x1="{x}" y1="{ty:.1f}" x2="{x + w}" y2="{ty:.1f}" stroke="#eef2f7"/>')


def draw_horizontal_bars(
    parts: list[str],
    rows: list[tuple[str, float, str]],
    *,
    x: int,
    y: int,
    w: int,
    row_h: int,
    title: str,
    value_suffix: str = "",
    lower_is_better: bool = False,
) -> None:
    panel_h = 76 + len(rows) * row_h
    parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{panel_h}" rx="8" fill="#ffffff" stroke="#d7dde8"/>')
    parts.append(f'<text x="{x + 24}" y="{y + 38}" font-size="20" font-weight="800" fill="#1f2937">{escape(title)}</text>')
    vals = [float(v) for _, v, _ in rows if pd.notna(v)]
    lo = 0 if vals else 0
    hi = max(vals) if vals else 1
    if not lower_is_better and hi <= 1.05:
        lo = min(0.85, min(vals) - 0.02) if vals else 0
        hi = 1.02
    if lower_is_better:
        hi = max(vals) * 1.08 if vals else 1
    bar_x = x + 190
    bar_w = w - 300
    for idx, (label, value, color) in enumerate(rows):
        yy = y + 72 + idx * row_h
        parts.append(f'<text x="{x + 24}" y="{yy + 18}" font-size="13" fill="#334155">{escape(label)}</text>')
        ratio = scaled(float(value), lo, hi)
        width = ratio * bar_w
        if lower_is_better:
            width = max(10, width)
        parts.append(f'<rect x="{bar_x}" y="{yy}" width="{bar_w}" height="24" rx="5" fill="#eef2f7"/>')
        parts.append(f'<rect x="{bar_x}" y="{yy}" width="{width:.1f}" height="24" rx="5" fill="{color}"/>')
        value_text = f"{fmt(float(value), 3)}{value_suffix}"
        parts.append(f'<text x="{bar_x + bar_w + 18}" y="{yy + 17}" font-size="12" font-weight="700" fill="#475569">{escape(value_text)}</text>')


def polyline(points: list[tuple[float, float]], color: str, width: int = 3) -> str:
    if not points:
        return ""
    pairs = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return f'<polyline points="{pairs}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round"/>'


def draw_confusion_matrix(parts: list[str], x: int, y: int, cm: np.ndarray) -> None:
    max_value = max(float(cm.max()), 1.0)
    size = 130
    labels = [["TN", "FP"], ["FN", "TP"]]
    parts.append(f'<text x="{x}" y="{y - 18}" font-size="18" font-weight="800" fill="#1f2937">Confusion Matrix</text>')
    parts.append(f'<text x="{x + 142}" y="{y - 44}" text-anchor="middle" font-size="12" fill="#64748b">Predicted</text>')
    parts.append(f'<text transform="translate({x - 28},{y + 142}) rotate(-90)" text-anchor="middle" font-size="12" fill="#64748b">Actual</text>')
    for row in range(2):
        for col in range(2):
            value = int(cm[row, col])
            intensity = 0.18 + 0.72 * (value / max_value)
            fill = f"rgba(58, 175, 169, {intensity:.2f})"
            bx = x + col * size
            by = y + row * size
            parts.append(f'<rect x="{bx}" y="{by}" width="{size}" height="{size}" fill="{fill}" stroke="#ffffff" stroke-width="3"/>')
            parts.append(f'<text x="{bx + size / 2}" y="{by + 48}" text-anchor="middle" font-size="16" font-weight="700" fill="#0f172a">{labels[row][col]}</text>')
            parts.append(f'<text x="{bx + size / 2}" y="{by + 86}" text-anchor="middle" font-size="34" font-weight="800" fill="#0f172a">{value}</text>')
    parts.append(f'<text x="{x + size / 2}" y="{y + size * 2 + 28}" text-anchor="middle" font-size="12" fill="#64748b">0</text>')
    parts.append(f'<text x="{x + size * 1.5}" y="{y + size * 2 + 28}" text-anchor="middle" font-size="12" fill="#64748b">1</text>')
    parts.append(f'<text x="{x - 18}" y="{y + size / 2}" text-anchor="end" font-size="12" fill="#64748b">0</text>')
    parts.append(f'<text x="{x - 18}" y="{y + size * 1.5}" text-anchor="end" font-size="12" fill="#64748b">1</text>')


def build_classification_summary(metrics: pd.DataFrame, clf_pred: pd.DataFrame) -> Path:
    final_row = metrics[
        metrics["task"].eq("classification") & metrics["model_name"].eq("stacking_ensemble")
    ].iloc[0]
    best_accuracy = best_row(metrics, "classification", "accuracy", True)
    best_recall = best_row(metrics, "classification", "recall", True)

    y_true = clf_pred["y_true_alert"].astype(int).to_numpy()
    y_pred = clf_pred["y_pred_alert"].astype(int).to_numpy()
    cm = np.array(
        [
            [int(((y_true == 0) & (y_pred == 0)).sum()), int(((y_true == 0) & (y_pred == 1)).sum())],
            [int(((y_true == 1) & (y_pred == 0)).sum()), int(((y_true == 1) & (y_pred == 1)).sum())],
        ]
    )

    parts = svg_header(
        1400,
        820,
        "Reliability Check - Classification",
        "Alert-risk prediction metrics on the validation set",
    )
    card(parts, 52, 126, 240, 130, "Final Model", MODEL_LABELS.get(final_row["model_name"], final_row["model_name"]), "selected by recall")
    card(parts, 312, 126, 190, 130, "Accuracy", percent(final_row["accuracy"]), f"n={len(clf_pred)}")
    card(parts, 522, 126, 190, 130, "Precision", percent(final_row["precision"]), "alert class")
    card(parts, 732, 126, 190, 130, "Recall", percent(final_row["recall"]), "missed alerts minimized")
    card(parts, 942, 126, 190, 130, "F1-score", fmt(final_row["f1"], 3), "precision/recall")
    card(parts, 1152, 126, 190, 130, "ROC-AUC", fmt(final_row["roc_auc"], 3), "ranking quality")

    draw_confusion_matrix(parts, 92, 392, cm)

    parts.append('<rect x="470" y="338" width="860" height="360" rx="8" fill="#ffffff" stroke="#d7dde8"/>')
    parts.append('<text x="500" y="382" font-size="21" font-weight="800" fill="#1f2937">Key Reliability Notes</text>')
    notes = [
        f"Final stacking ensemble recall is {percent(final_row['recall'])}; false negatives = {int(cm[1, 0])}.",
        f"Best accuracy model: {MODEL_LABELS.get(best_accuracy['model_name'], best_accuracy['model_name'])} ({percent(best_accuracy['accuracy'])}).",
        f"Best recall model: {MODEL_LABELS.get(best_recall['model_name'], best_recall['model_name'])} ({percent(best_recall['recall'])}).",
        f"Threshold used by final model: {fmt(final_row['threshold'], 2)}.",
        "For algae alerts, recall is prioritized because missing a real warning event is operationally costly.",
    ]
    yy = 426
    for idx, text in enumerate(notes):
        parts.append(f'<circle cx="510" cy="{yy - 5}" r="4" fill="#3AAFA9"/>')
        end_y = add_wrapped_text(parts, text, 526, yy, max_chars=88, font_size=15, fill="#334155")
        yy = end_y + 34

    return save_svg(parts, OUTPUT_DIR / "reliability_classification_summary.svg")


def line_path(points: list[tuple[float, float]], x: int, y: int, w: int, h: int) -> str:
    commands = []
    for idx, (px, py) in enumerate(points):
        sx = x + px * w
        sy = y + (1 - py) * h
        commands.append(("M" if idx == 0 else "L") + f"{sx:.1f},{sy:.1f}")
    return " ".join(commands)


def draw_curve_panel(
    parts: list[str],
    x: int,
    y: int,
    w: int,
    h: int,
    title: str,
    x_label: str,
    y_label: str,
    points: list[tuple[float, float]],
    score_label: str,
    color: str,
    baseline: str | None = None,
) -> None:
    parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="#ffffff" stroke="#d7dde8"/>')
    parts.append(f'<text x="{x + 24}" y="{y + 38}" font-size="20" font-weight="800" fill="#1f2937">{escape(title)}</text>')
    cx, cy, cw, ch = x + 72, y + 78, w - 116, h - 140
    parts.append(f'<line x1="{cx}" y1="{cy + ch}" x2="{cx + cw}" y2="{cy + ch}" stroke="#cbd5e1"/>')
    parts.append(f'<line x1="{cx}" y1="{cy}" x2="{cx}" y2="{cy + ch}" stroke="#cbd5e1"/>')
    for tick in [0, 0.25, 0.5, 0.75, 1.0]:
        tx = cx + tick * cw
        ty = cy + (1 - tick) * ch
        parts.append(f'<line x1="{tx:.1f}" y1="{cy}" x2="{tx:.1f}" y2="{cy + ch}" stroke="#eef2f7"/>')
        parts.append(f'<line x1="{cx}" y1="{ty:.1f}" x2="{cx + cw}" y2="{ty:.1f}" stroke="#eef2f7"/>')
        parts.append(f'<text x="{tx:.1f}" y="{cy + ch + 22}" text-anchor="middle" font-size="11" fill="#64748b">{tick:.2g}</text>')
        parts.append(f'<text x="{cx - 12}" y="{ty + 4:.1f}" text-anchor="end" font-size="11" fill="#64748b">{tick:.2g}</text>')
    if baseline == "diagonal":
        parts.append(f'<line x1="{cx}" y1="{cy + ch}" x2="{cx + cw}" y2="{cy}" stroke="#94a3b8" stroke-dasharray="6 6"/>')
    path = line_path(points, cx, cy, cw, ch)
    parts.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="4" stroke-linejoin="round" stroke-linecap="round"/>')
    parts.append(f'<text x="{cx + cw - 6}" y="{cy + 24}" text-anchor="end" font-size="16" font-weight="800" fill="{color}">{escape(score_label)}</text>')
    parts.append(f'<text x="{cx + cw / 2}" y="{y + h - 26}" text-anchor="middle" font-size="13" fill="#475569">{escape(x_label)}</text>')
    parts.append(f'<text transform="translate({x + 24},{cy + ch / 2}) rotate(-90)" text-anchor="middle" font-size="13" fill="#475569">{escape(y_label)}</text>')


def build_curve_figure(clf_pred: pd.DataFrame) -> Path:
    y_true = clf_pred["y_true_alert"].astype(int)
    proba = clf_pred["y_pred_probability"].astype(float)
    fpr, tpr, _ = roc_curve(y_true, proba)
    precision, recall, _ = precision_recall_curve(y_true, proba)
    roc_auc = auc(fpr, tpr)
    pr_auc = auc(recall, precision)

    parts = svg_header(
        1400,
        680,
        "Reliability Check - ROC / PR Curves",
        "Probability ranking quality for alert-risk prediction",
    )
    draw_curve_panel(
        parts,
        62,
        130,
        610,
        470,
        "ROC Curve",
        "False Positive Rate",
        "True Positive Rate",
        list(zip(fpr, tpr)),
        f"AUC = {roc_auc:.3f}",
        "#3AAFA9",
        baseline="diagonal",
    )
    draw_curve_panel(
        parts,
        728,
        130,
        610,
        470,
        "Precision-Recall Curve",
        "Recall",
        "Precision",
        list(zip(recall, precision)),
        f"PR-AUC = {pr_auc:.3f}",
        "#9673A6",
    )
    return save_svg(parts, OUTPUT_DIR / "reliability_roc_pr_curves.svg")


def build_regression_summary(metrics: pd.DataFrame, reg_pred: pd.DataFrame) -> Path:
    final_row = metrics[
        metrics["task"].eq("regression") & metrics["model_name"].eq("stacking_ensemble")
    ].iloc[0]
    best_rmse = best_row(metrics, "regression", "rmse", False)
    best_cells = best_row(metrics, "regression", "rmse_cells", False)

    actual = reg_pred["y_true_target"].astype(float).to_numpy()
    predicted = reg_pred["y_pred_target"].astype(float).to_numpy()
    residuals = actual - predicted
    x_min, x_max = float(np.nanmin(actual)), float(np.nanmax(actual))
    y_min, y_max = float(np.nanmin(predicted)), float(np.nanmax(predicted))
    lo, hi = min(x_min, y_min), max(x_max, y_max)
    pad = max((hi - lo) * 0.05, 0.1)
    lo, hi = lo - pad, hi + pad

    parts = svg_header(
        1400,
        820,
        "Reliability Check - Regression",
        "Next log10(cells + 1) prediction quality on the validation set",
    )
    card(parts, 52, 126, 240, 130, "Final Model", "Stacking Ensemble", "lowest RMSE")
    card(parts, 312, 126, 180, 130, "MAE(log)", fmt(final_row["mae"], 3), "")
    card(parts, 512, 126, 180, 130, "RMSE(log)", fmt(final_row["rmse"], 3), "")
    card(parts, 712, 126, 180, 130, "R-squared", fmt(final_row["r2"], 3), "")
    card(parts, 912, 126, 190, 130, "MAE(cells)", fmt(final_row["mae_cells"], 0), "")
    card(parts, 1122, 126, 220, 130, "RMSE(cells)", fmt(final_row["rmse_cells"], 0), "")

    # Predicted vs actual scatter.
    x, y, w, h = 78, 376, 590, 340
    parts.append(f'<rect x="{x - 18}" y="{y - 54}" width="{w + 80}" height="{h + 120}" rx="8" fill="#ffffff" stroke="#d7dde8"/>')
    parts.append(f'<text x="{x}" y="{y - 20}" font-size="20" font-weight="800" fill="#1f2937">Predicted vs Actual</text>')
    parts.append(f'<line x1="{x}" y1="{y + h}" x2="{x + w}" y2="{y + h}" stroke="#cbd5e1"/>')
    parts.append(f'<line x1="{x}" y1="{y}" x2="{x}" y2="{y + h}" stroke="#cbd5e1"/>')
    parts.append(f'<line x1="{x}" y1="{y + h}" x2="{x + w}" y2="{y}" stroke="#94a3b8" stroke-dasharray="6 6"/>')
    for a, p in zip(actual, predicted):
        sx = x + (a - lo) / (hi - lo) * w
        sy = y + (1 - (p - lo) / (hi - lo)) * h
        parts.append(f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="3" fill="#3AAFA9" fill-opacity="0.58"/>')
    parts.append(f'<text x="{x + w / 2}" y="{y + h + 40}" text-anchor="middle" font-size="13" fill="#475569">Actual next_log_cells</text>')
    parts.append(f'<text transform="translate({x - 40},{y + h / 2}) rotate(-90)" text-anchor="middle" font-size="13" fill="#475569">Predicted next_log_cells</text>')

    # Residual histogram.
    hx, hy, hw, hh = 760, 376, 520, 340
    counts, bins = np.histogram(residuals, bins=18)
    max_count = max(int(counts.max()), 1)
    parts.append(f'<rect x="{hx - 18}" y="{hy - 54}" width="{hw + 80}" height="{hh + 120}" rx="8" fill="#ffffff" stroke="#d7dde8"/>')
    parts.append(f'<text x="{hx}" y="{hy - 20}" font-size="20" font-weight="800" fill="#1f2937">Residual Distribution</text>')
    parts.append(f'<line x1="{hx}" y1="{hy + hh}" x2="{hx + hw}" y2="{hy + hh}" stroke="#cbd5e1"/>')
    parts.append(f'<line x1="{hx}" y1="{hy}" x2="{hx}" y2="{hy + hh}" stroke="#cbd5e1"/>')
    gap = 4
    bar_w = (hw - gap * (len(counts) - 1)) / len(counts)
    for idx, count in enumerate(counts):
        bh = count / max_count * hh
        bx = hx + idx * (bar_w + gap)
        by = hy + hh - bh
        parts.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w:.1f}" height="{bh:.1f}" rx="3" fill="#9673A6"/>')
    parts.append(f'<text x="{hx + hw / 2}" y="{hy + hh + 40}" text-anchor="middle" font-size="13" fill="#475569">Actual - Predicted residual</text>')
    parts.append(f'<text x="{hx}" y="{hy + hh + 62}" font-size="12" fill="#64748b">Mean residual: {mean(residuals):.3f}</text>')
    parts.append(
        f'<text x="{hx + 220}" y="{hy + hh + 62}" font-size="12" fill="#64748b">'
        f'Best log RMSE: {escape(MODEL_LABELS.get(best_rmse["model_name"], best_rmse["model_name"]))}'
        f'</text>'
    )
    parts.append(
        f'<text x="{hx + 220}" y="{hy + hh + 82}" font-size="12" fill="#64748b">'
        f'Best cell RMSE: {escape(MODEL_LABELS.get(best_cells["model_name"], best_cells["model_name"]))}'
        f'</text>'
    )

    return save_svg(parts, OUTPUT_DIR / "reliability_regression_summary.svg")


def pvalue_text(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "not available"
    if value < 0.001:
        return f"{value:.2e}"
    return f"{value:.4f}"


def compute_significance(clf_pred: pd.DataFrame, reg_pred: pd.DataFrame) -> dict[str, float | None]:
    try:
        from scipy.stats import binomtest, pearsonr, ttest_rel, wilcoxon
    except Exception:
        return {
            "binom_p": None,
            "pearson_r": None,
            "pearson_p": None,
            "paired_t_p": None,
            "wilcoxon_p": None,
            "model_mae_log": None,
            "baseline_mae_log": None,
        }

    y_true = clf_pred["y_true_alert"].astype(int)
    y_pred = clf_pred["y_pred_alert"].astype(int)
    correct = int((y_true == y_pred).sum())
    majority_rate = float(y_true.value_counts(normalize=True).max())
    binom_p = float(binomtest(correct, len(y_true), p=majority_rate, alternative="greater").pvalue)

    actual = reg_pred["y_true_target"].astype(float)
    predicted = reg_pred["y_pred_target"].astype(float)
    model_abs_err = (actual - predicted).abs()
    baseline_pred = np.log10(reg_pred["previous_observed_cells"].astype(float) + 1)
    baseline_abs_err = (actual - baseline_pred).abs()
    paired_t_p = float(ttest_rel(model_abs_err, baseline_abs_err, alternative="less").pvalue)
    wilcoxon_p = float(wilcoxon(model_abs_err, baseline_abs_err, alternative="less").pvalue)
    pearson = pearsonr(actual, predicted)

    return {
        "binom_p": binom_p,
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "paired_t_p": paired_t_p,
        "wilcoxon_p": wilcoxon_p,
        "model_mae_log": float(model_abs_err.mean()),
        "baseline_mae_log": float(baseline_abs_err.mean()),
    }


def build_significance_summary(clf_pred: pd.DataFrame, reg_pred: pd.DataFrame) -> Path:
    stats = compute_significance(clf_pred, reg_pred)
    parts = svg_header(
        1400,
        720,
        "Reliability Check - Statistical Significance",
        "Objective checks against simple baselines and actual-observed relationship",
    )

    items = [
        (
            "Classification binomial test",
            pvalue_text(stats["binom_p"]),
            "Final classifier accuracy vs majority-class baseline",
            "p < 0.05 means the hit rate is unlikely to be explained by class imbalance alone.",
        ),
        (
            "Regression Pearson correlation",
            f"r={fmt(stats['pearson_r'], 3)}, p={pvalue_text(stats['pearson_p'])}",
            "Predicted next_log_cells vs actual next_log_cells",
            "Strong positive correlation supports numerical forecast reliability.",
        ),
        (
            "Paired t-test",
            pvalue_text(stats["paired_t_p"]),
            "Stacking absolute error vs persistence baseline absolute error",
            "p < 0.05 supports lower average error than using the previous observation.",
        ),
        (
            "Wilcoxon signed-rank test",
            pvalue_text(stats["wilcoxon_p"]),
            "Non-parametric paired comparison against persistence baseline",
            "This is stricter for spike-heavy algae data and can be more conservative.",
        ),
    ]

    for idx, (title, value, subtitle, note) in enumerate(items):
        x = 70 + (idx % 2) * 660
        y = 142 + (idx // 2) * 230
        parts.append(f'<rect x="{x}" y="{y}" width="610" height="182" rx="8" fill="#ffffff" stroke="#d7dde8"/>')
        parts.append(f'<text x="{x + 24}" y="{y + 38}" font-size="18" font-weight="800" fill="#1f2937">{escape(title)}</text>')
        value_size = 24 if len(value) <= 24 else 18
        parts.append(f'<text x="{x + 24}" y="{y + 78}" font-size="{value_size}" font-weight="800" fill="#3AAFA9">{escape(value)}</text>')
        end_y = add_wrapped_text(parts, subtitle, x + 24, y + 110, max_chars=70, font_size=13, fill="#475569")
        add_wrapped_text(parts, note, x + 24, end_y + 26, max_chars=76, font_size=12, fill="#64748b")

    if stats["model_mae_log"] is not None:
        parts.append(
            f'<text x="72" y="650" font-size="14" fill="#334155">'
            f'Mean absolute log error: model={stats["model_mae_log"]:.3f}, persistence baseline={stats["baseline_mae_log"]:.3f}'
            f'</text>'
        )

    return save_svg(parts, OUTPUT_DIR / "reliability_significance_summary.svg")


def build_classification_leaderboard(metrics: pd.DataFrame) -> Path:
    cls = metrics[metrics["task"].eq("classification")].copy()
    cls["model_label"] = cls["model_name"].map(MODEL_LABELS).fillna(cls["model_name"])
    cls["color"] = cls["model_label"].map(MODEL_COLORS).fillna("#64748b")
    by_recall = cls.sort_values("recall", ascending=False).head(8)
    by_f1 = cls.sort_values("f1", ascending=False).head(8)
    by_auc = cls.sort_values("roc_auc", ascending=False).head(8)

    parts = svg_header(
        1400,
        1420,
        "Model Leaderboard - Classification",
        "Higher is better. Recall is emphasized for avoiding missed alert events.",
    )
    draw_horizontal_bars(
        parts,
        [(r.model_label, r.recall, r.color) for r in by_recall.itertuples()],
        x=58,
        y=128,
        w=1284,
        row_h=44,
        title="Recall Ranking",
    )
    draw_horizontal_bars(
        parts,
        [(r.model_label, r.f1, r.color) for r in by_f1.itertuples()],
        x=58,
        y=552,
        w=1284,
        row_h=44,
        title="F1-score Ranking",
    )
    draw_horizontal_bars(
        parts,
        [(r.model_label, r.roc_auc, r.color) for r in by_auc.itertuples()],
        x=58,
        y=976,
        w=1284,
        row_h=44,
        title="ROC-AUC Ranking",
    )
    return save_svg(parts, OUTPUT_DIR / "reliability_classification_leaderboard.svg")


def build_regression_leaderboard(metrics: pd.DataFrame) -> Path:
    reg = metrics[metrics["task"].eq("regression")].copy()
    reg["model_label"] = reg["model_name"].map(MODEL_LABELS).fillna(reg["model_name"])
    reg["color"] = reg["model_label"].map(MODEL_COLORS).fillna("#64748b")
    by_rmse = reg.sort_values("rmse", ascending=True).head(8)
    by_cells = reg.sort_values("rmse_cells", ascending=True).head(8)
    by_r2 = reg.sort_values("r2", ascending=False).head(8)

    parts = svg_header(
        1400,
        980,
        "Model Leaderboard - Regression",
        "RMSE is lower-is-better; R-squared is higher-is-better.",
    )
    draw_horizontal_bars(
        parts,
        [(r.model_label, r.rmse, r.color) for r in by_rmse.itertuples()],
        x=58,
        y=128,
        w=610,
        row_h=42,
        title="RMSE(log) Ranking",
        lower_is_better=True,
    )
    draw_horizontal_bars(
        parts,
        [(r.model_label, r.rmse_cells, r.color) for r in by_cells.itertuples()],
        x=732,
        y=128,
        w=610,
        row_h=42,
        title="RMSE(cells) Ranking",
        lower_is_better=True,
    )
    draw_horizontal_bars(
        parts,
        [(r.model_label, r.r2, r.color) for r in by_r2.itertuples()],
        x=58,
        y=540,
        w=1284,
        row_h=42,
        title="R-squared Ranking",
    )
    return save_svg(parts, OUTPUT_DIR / "reliability_regression_leaderboard.svg")


def build_threshold_sensitivity(threshold_df: pd.DataFrame) -> Path:
    model_name = "stacking_ensemble" if "stacking_ensemble" in set(threshold_df["model_name"]) else threshold_df["model_name"].iloc[0]
    df = threshold_df[threshold_df["model_name"].eq(model_name)].sort_values("threshold")
    if df.empty:
        raise ValueError("No threshold candidate rows found.")

    x, y, w, h = 110, 170, 1120, 430
    parts = svg_header(
        1400,
        760,
        "Threshold Sensitivity",
        f"Precision, recall, and F1 changes for {MODEL_LABELS.get(model_name, model_name)}.",
    )
    parts.append(f'<rect x="58" y="124" width="1284" height="560" rx="8" fill="#ffffff" stroke="#d7dde8"/>')
    draw_axis_box(parts, x, y, w, h)
    thresholds = df["threshold"].astype(float).to_numpy()
    low, high = float(thresholds.min()), float(thresholds.max())
    series = [
        ("Precision", df["precision"].astype(float).to_numpy(), "#3AAFA9"),
        ("Recall", df["recall"].astype(float).to_numpy(), "#E07A5F"),
        ("F1-score", df["f1"].astype(float).to_numpy(), "#9673A6"),
    ]
    for name, values, color in series:
        pts = []
        for threshold, value in zip(thresholds, values):
            sx = x + scaled(float(threshold), low, high) * w
            sy = y + (1 - scaled(float(value), 0.0, 1.0)) * h
            pts.append((sx, sy))
        parts.append(polyline(pts, color, 4))
        for sx, sy in pts:
            parts.append(f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="4" fill="{color}" stroke="#ffffff" stroke-width="1.5"/>')
    for idx, (name, _, color) in enumerate(series):
        lx = 1020 + idx * 112
        parts.append(f'<rect x="{lx}" y="138" width="16" height="16" rx="3" fill="{color}"/>')
        parts.append(f'<text x="{lx + 24}" y="151" font-size="13" fill="#334155">{escape(name)}</text>')
    parts.append(f'<text x="{x + w / 2}" y="{y + h + 48}" text-anchor="middle" font-size="14" fill="#475569">Threshold</text>')
    parts.append(f'<text transform="translate(64,{y + h / 2}) rotate(-90)" text-anchor="middle" font-size="14" fill="#475569">Score</text>')
    return save_svg(parts, OUTPUT_DIR / "reliability_threshold_sensitivity.svg")


def build_probability_distribution(clf_pred: pd.DataFrame) -> Path:
    parts = svg_header(
        1400,
        760,
        "Predicted Probability Distribution",
        "Predicted alert probability by actual validation class.",
    )
    x, y, w, h = 110, 166, 1120, 430
    parts.append(f'<rect x="58" y="124" width="1284" height="560" rx="8" fill="#ffffff" stroke="#d7dde8"/>')
    draw_axis_box(parts, x, y, w, h)
    bins = np.linspace(0, 1, 21)
    negatives = clf_pred[clf_pred["y_true_alert"].eq(0)]["y_pred_probability"].astype(float)
    positives = clf_pred[clf_pred["y_true_alert"].eq(1)]["y_pred_probability"].astype(float)
    neg_counts, _ = np.histogram(negatives, bins=bins)
    pos_counts, _ = np.histogram(positives, bins=bins)
    max_count = max(int(neg_counts.max()), int(pos_counts.max()), 1)
    group_w = w / len(neg_counts)
    bar_w = group_w * 0.38
    for idx, (neg, pos) in enumerate(zip(neg_counts, pos_counts)):
        base_x = x + idx * group_w
        neg_h = neg / max_count * h
        pos_h = pos / max_count * h
        parts.append(f'<rect x="{base_x + group_w * 0.12:.1f}" y="{y + h - neg_h:.1f}" width="{bar_w:.1f}" height="{neg_h:.1f}" rx="3" fill="#6C8EBF" fill-opacity="0.85"/>')
        parts.append(f'<rect x="{base_x + group_w * 0.52:.1f}" y="{y + h - pos_h:.1f}" width="{bar_w:.1f}" height="{pos_h:.1f}" rx="3" fill="#E07A5F" fill-opacity="0.85"/>')
    parts.append('<rect x="1012" y="146" width="16" height="16" rx="3" fill="#6C8EBF"/>')
    parts.append('<text x="1036" y="159" font-size="13" fill="#334155">Actual 0</text>')
    parts.append('<rect x="1120" y="146" width="16" height="16" rx="3" fill="#E07A5F"/>')
    parts.append('<text x="1144" y="159" font-size="13" fill="#334155">Actual 1</text>')
    parts.append(f'<text x="{x + w / 2}" y="{y + h + 48}" text-anchor="middle" font-size="14" fill="#475569">Predicted probability</text>')
    parts.append(f'<text transform="translate(64,{y + h / 2}) rotate(-90)" text-anchor="middle" font-size="14" fill="#475569">Count</text>')
    return save_svg(parts, OUTPUT_DIR / "reliability_probability_distribution.svg")


def build_calibration_plot(clf_pred: pd.DataFrame) -> Path:
    df = clf_pred.copy()
    df["bin"] = pd.cut(df["y_pred_probability"].astype(float), bins=np.linspace(0, 1, 11), include_lowest=True)
    grouped = df.groupby("bin", observed=False).agg(
        mean_probability=("y_pred_probability", "mean"),
        observed_rate=("y_true_alert", "mean"),
        count=("y_true_alert", "size"),
    ).dropna()

    x, y, w, h = 130, 166, 1040, 430
    parts = svg_header(
        1400,
        760,
        "Calibration Check",
        "If probabilities are well calibrated, points stay close to the diagonal line.",
    )
    parts.append(f'<rect x="58" y="124" width="1284" height="560" rx="8" fill="#ffffff" stroke="#d7dde8"/>')
    draw_axis_box(parts, x, y, w, h)
    parts.append(f'<line x1="{x}" y1="{y + h}" x2="{x + w}" y2="{y}" stroke="#94a3b8" stroke-dasharray="7 7"/>')
    points = []
    for row in grouped.itertuples():
        sx = x + scaled(float(row.mean_probability), 0, 1) * w
        sy = y + (1 - scaled(float(row.observed_rate), 0, 1)) * h
        radius = max(5, min(18, float(row.count) ** 0.5))
        points.append((sx, sy))
        parts.append(f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="{radius:.1f}" fill="#3AAFA9" fill-opacity="0.78" stroke="#ffffff" stroke-width="2"/>')
    parts.append(polyline(points, "#3AAFA9", 3))
    parts.append(f'<text x="{x + w / 2}" y="{y + h + 48}" text-anchor="middle" font-size="14" fill="#475569">Mean predicted probability</text>')
    parts.append(f'<text transform="translate(78,{y + h / 2}) rotate(-90)" text-anchor="middle" font-size="14" fill="#475569">Observed alert rate</text>')
    parts.append('<text x="955" y="154" font-size="13" fill="#64748b">Bubble size = bin count</text>')
    return save_svg(parts, OUTPUT_DIR / "reliability_calibration_plot.svg")


def build_error_by_location(reg_pred: pd.DataFrame) -> Path:
    df = reg_pred.copy()
    df["abs_log_error"] = (df["y_true_target"].astype(float) - df["y_pred_target"].astype(float)).abs()
    df["abs_cell_error"] = (df["y_true_cells"].astype(float) - df["y_pred_cells"].astype(float)).abs()
    df["location_label"] = df["loc_encoded"].map(location_label)
    grouped = df.groupby("location_label").agg(
        mae_log=("abs_log_error", "mean"),
        mae_cells=("abs_cell_error", "mean"),
        n=("abs_log_error", "size"),
    ).reset_index()

    parts = svg_header(
        1400,
        760,
        "Error by Monitoring Location",
        "Mean absolute error by validation-site group.",
    )
    rows_log = [(str(r[1]["location_label"]), float(r[1]["mae_log"]), "#3AAFA9") for r in grouped.iterrows()]
    rows_cells = [(str(r[1]["location_label"]), float(r[1]["mae_cells"]), "#9673A6") for r in grouped.iterrows()]
    draw_horizontal_bars(parts, rows_log, x=62, y=136, w=610, row_h=58, title="MAE(log) by Location", lower_is_better=True)
    draw_horizontal_bars(parts, rows_cells, x=728, y=136, w=610, row_h=58, title="MAE(cells) by Location", lower_is_better=True)
    yy = 556
    parts.append('<rect x="62" y="520" width="1276" height="126" rx="8" fill="#ffffff" stroke="#d7dde8"/>')
    parts.append('<text x="90" y="560" font-size="18" font-weight="800" fill="#1f2937">Sample Counts</text>')
    for idx, row in enumerate(grouped.itertuples()):
        parts.append(f'<text x="{90 + idx * 220}" y="{yy + 42}" font-size="14" fill="#334155">{escape(str(row.location_label))}: n={int(row.n)}</text>')
    return save_svg(parts, OUTPUT_DIR / "reliability_error_by_location.svg")


def build_timeseries_comparison(reg_pred: pd.DataFrame) -> Path:
    df = reg_pred.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values(["loc_encoded", "date"])
    df["location_label"] = df["loc_encoded"].map(location_label)
    groups = list(df.groupby("loc_encoded"))[:3]

    parts = svg_header(
        1400,
        1080,
        "Time Series Check - Actual vs Predicted",
        "Validation-period next_log_cells by monitoring location.",
    )
    panel_x, panel_w, panel_h = 74, 1250, 238
    colors = {"actual": "#111827", "pred": "#3AAFA9"}
    for idx, (_, group) in enumerate(groups):
        y0 = 136 + idx * 292
        group = group.sort_values("date")
        actual = group["y_true_target"].astype(float).to_numpy()
        pred = group["y_pred_target"].astype(float).to_numpy()
        values = np.concatenate([actual, pred])
        low, high = float(np.nanmin(values)), float(np.nanmax(values))
        pad = max((high - low) * 0.08, 0.1)
        low, high = low - pad, high + pad
        parts.append(f'<rect x="{panel_x}" y="{y0}" width="{panel_w}" height="{panel_h}" rx="8" fill="#ffffff" stroke="#d7dde8"/>')
        loc_label = str(group["location_label"].iloc[0])
        parts.append(f'<text x="{panel_x + 24}" y="{y0 + 36}" font-size="19" font-weight="800" fill="#1f2937">{escape(loc_label)}</text>')
        x, y, w, h = panel_x + 72, y0 + 58, panel_w - 120, panel_h - 100
        parts.append(f'<line x1="{x}" y1="{y + h}" x2="{x + w}" y2="{y + h}" stroke="#cbd5e1"/>')
        parts.append(f'<line x1="{x}" y1="{y}" x2="{x}" y2="{y + h}" stroke="#cbd5e1"/>')
        denom = max(len(group) - 1, 1)
        actual_pts = []
        pred_pts = []
        for i, (a, p) in enumerate(zip(actual, pred)):
            sx = x + i / denom * w
            actual_pts.append((sx, y + (1 - scaled(float(a), low, high)) * h))
            pred_pts.append((sx, y + (1 - scaled(float(p), low, high)) * h))
        parts.append(polyline(actual_pts, colors["actual"], 3))
        parts.append(polyline(pred_pts, colors["pred"], 3))
        for sx, sy in pred_pts[:: max(1, len(pred_pts) // 16)]:
            parts.append(f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="3" fill="{colors["pred"]}"/>')
        parts.append(f'<text x="{x + w - 220}" y="{y0 + 36}" font-size="13" fill="{colors["actual"]}">Actual</text>')
        parts.append(f'<text x="{x + w - 150}" y="{y0 + 36}" font-size="13" fill="{colors["pred"]}">Predicted</text>')
    return save_svg(parts, OUTPUT_DIR / "reliability_timeseries_actual_predicted.svg")


def build_residual_vs_actual(reg_pred: pd.DataFrame) -> Path:
    df = reg_pred.copy()
    actual = df["y_true_target"].astype(float).to_numpy()
    residual = (df["y_true_target"].astype(float) - df["y_pred_target"].astype(float)).to_numpy()
    x_low, x_high = float(np.nanmin(actual)), float(np.nanmax(actual))
    y_abs = max(abs(float(np.nanmin(residual))), abs(float(np.nanmax(residual))), 0.1)
    y_low, y_high = -y_abs * 1.08, y_abs * 1.08

    x, y, w, h = 110, 166, 1120, 430
    parts = svg_header(
        1400,
        760,
        "Residual Check",
        "Residuals around zero indicate balanced over/under prediction across algae levels.",
    )
    parts.append(f'<rect x="58" y="124" width="1284" height="560" rx="8" fill="#ffffff" stroke="#d7dde8"/>')
    draw_axis_box(parts, x, y, w, h)
    zero_y = y + (1 - scaled(0, y_low, y_high)) * h
    parts.append(f'<line x1="{x}" y1="{zero_y:.1f}" x2="{x + w}" y2="{zero_y:.1f}" stroke="#E07A5F" stroke-width="2" stroke-dasharray="7 7"/>')
    for a, r in zip(actual, residual):
        sx = x + scaled(float(a), x_low, x_high) * w
        sy = y + (1 - scaled(float(r), y_low, y_high)) * h
        parts.append(f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="3" fill="#9673A6" fill-opacity="0.58"/>')
    parts.append(f'<text x="{x + w / 2}" y="{y + h + 48}" text-anchor="middle" font-size="14" fill="#475569">Actual next_log_cells</text>')
    parts.append(f'<text transform="translate(64,{y + h / 2}) rotate(-90)" text-anchor="middle" font-size="14" fill="#475569">Actual - Predicted</text>')
    parts.append(f'<text x="{x + w - 170}" y="{zero_y - 10:.1f}" font-size="13" fill="#E07A5F">zero residual</text>')
    return save_svg(parts, OUTPUT_DIR / "reliability_residual_vs_actual.svg")


def build_model_metric_table(metrics: pd.DataFrame) -> Path:
    rows = metrics.copy()
    rows["model_label"] = rows["model_name"].map(MODEL_LABELS).fillna(rows["model_name"])
    rows = rows.sort_values(["task", "model_label"])
    table_path = OUTPUT_DIR / "reliability_metric_table.csv"
    keep_cols = [
        "task",
        "model_name",
        "model_label",
        "mae",
        "rmse",
        "r2",
        "mae_cells",
        "rmse_cells",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "threshold",
    ]
    table_path.parent.mkdir(parents=True, exist_ok=True)
    rows[[col for col in keep_cols if col in rows.columns]].to_csv(table_path, index=False, encoding="utf-8-sig")
    return table_path


def main() -> None:
    metrics = read_csv(METRICS_PATH)
    clf_pred = read_csv(CLASSIFICATION_PREDICTIONS_PATH)
    reg_pred = read_csv(REGRESSION_PREDICTIONS_PATH)

    outputs = [
        build_classification_summary(metrics, clf_pred),
        build_curve_figure(clf_pred),
        build_regression_summary(metrics, reg_pred),
        build_significance_summary(clf_pred, reg_pred),
        build_classification_leaderboard(metrics),
        build_regression_leaderboard(metrics),
        build_probability_distribution(clf_pred),
        build_calibration_plot(clf_pred),
        build_error_by_location(reg_pred),
        build_timeseries_comparison(reg_pred),
        build_residual_vs_actual(reg_pred),
        build_model_metric_table(metrics),
    ]
    if THRESHOLD_CANDIDATES_PATH.exists():
        outputs.insert(6, build_threshold_sensitivity(read_csv(THRESHOLD_CANDIDATES_PATH)))

    print("Saved reliability outputs:")
    for output in outputs:
        print(f"- {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
