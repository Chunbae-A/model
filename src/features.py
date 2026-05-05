from __future__ import annotations

import pandas as pd

from .config import (
    ALERT_CELL_THRESHOLD,
    CELL_COUNT_COLUMN,
    CLASSIFICATION_TARGET,
    DATE_COLUMN,
    FORBIDDEN_FEATURE_KEYWORDS,
    LOCATION_FLOW_ORDER,
    LOG_TARGET_COLUMN,
    REGRESSION_TARGET,
    REQUIRE_NUMERIC_FEATURES,
    SITE_COLUMN,
)


def add_temporal_spatial_features(
    df: pd.DataFrame,
    date_column: str = DATE_COLUMN,
    site_column: str = SITE_COLUMN,
    cell_count_column: str = CELL_COUNT_COLUMN,
    log_target_column: str = LOG_TARGET_COLUMN,
    alert_cell_threshold: int = ALERT_CELL_THRESHOLD,
    location_flow_order: dict[int, int] | None = None,
) -> pd.DataFrame:
    """Add sample cadence, previous-state, and Hoenam-pressure features."""
    required = [date_column, site_column, cell_count_column, log_target_column]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Missing columns for feature engineering: {missing}")

    order_map = location_flow_order or LOCATION_FLOW_ORDER
    output = df.copy()
    output[date_column] = pd.to_datetime(output[date_column])
    output = output.sort_values([site_column, date_column]).reset_index(drop=True)

    output["loc_flow_order"] = output[site_column].map(order_map).fillna(output[site_column]).astype(float)

    grouped = output.groupby(site_column, sort=False)
    prev_date = grouped[date_column].shift(1)
    next_date = grouped[date_column].shift(-1)
    output["sampling_gap_days"] = (output[date_column] - prev_date).dt.days
    output["sampling_gap_days"] = output["sampling_gap_days"].fillna(output["sampling_gap_days"].median()).fillna(7)
    output["next_sample_available"] = next_date.notna().astype(int)

    output["previous_observed_cells"] = grouped[cell_count_column].shift(1).fillna(0)
    output["previous_log_cells"] = grouped[log_target_column].shift(1).fillna(0)
    output["previous_exceeded"] = (output["previous_observed_cells"] >= alert_cell_threshold).astype(int)
    output["current_exceeded"] = (output[cell_count_column] >= alert_cell_threshold).astype(int)
    output["cell_change_since_previous"] = output[cell_count_column] - output["previous_observed_cells"]
    output["cell_growth_ratio_since_previous"] = (
        output[cell_count_column] + 1
    ) / (output["previous_observed_cells"] + 1)

    wide = output.pivot_table(
        index=date_column,
        columns="loc_flow_order",
        values=[cell_count_column, log_target_column],
        aggfunc="first",
    )
    site_wide = output.pivot_table(
        index=date_column,
        columns=site_column,
        values=[cell_count_column, log_target_column],
        aggfunc="first",
    )

    upstream_cells = []
    upstream_log_cells = []
    upstream_available = []
    for _, row in output.iterrows():
        upstream_order = row["loc_flow_order"] - 1
        date_value = row[date_column]
        has_upstream = (
            date_value in wide.index
            and (cell_count_column, upstream_order) in wide.columns
            and pd.notna(wide.loc[date_value, (cell_count_column, upstream_order)])
        )
        upstream_available.append(int(has_upstream))
        if has_upstream:
            upstream_cells.append(float(wide.loc[date_value, (cell_count_column, upstream_order)]))
            upstream_log_cells.append(float(wide.loc[date_value, (log_target_column, upstream_order)]))
        else:
            upstream_cells.append(0.0)
            upstream_log_cells.append(0.0)

    output["has_upstream_site"] = upstream_available
    output["upstream_cells_same_date"] = upstream_cells
    output["upstream_log_cells_same_date"] = upstream_log_cells
    output["upstream_minus_current_log_cells"] = (
        output["upstream_log_cells_same_date"] - output[log_target_column]
    )

    hoenam_code = 1
    hoenam_cells = []
    hoenam_log_cells = []
    for _, row in output.iterrows():
        date_value = row[date_column]
        has_hoenam = (
            date_value in site_wide.index
            and (cell_count_column, hoenam_code) in site_wide.columns
            and pd.notna(site_wide.loc[date_value, (cell_count_column, hoenam_code)])
        )
        if has_hoenam:
            hoenam_cells.append(float(site_wide.loc[date_value, (cell_count_column, hoenam_code)]))
            hoenam_log_cells.append(float(site_wide.loc[date_value, (log_target_column, hoenam_code)]))
        else:
            hoenam_cells.append(0.0)
            hoenam_log_cells.append(0.0)

    output["hoenam_cells_same_date"] = hoenam_cells
    output["hoenam_log_cells_same_date"] = hoenam_log_cells
    output["hoenam_pressure_for_downstream"] = (
        (output[site_column] != hoenam_code)
        & (output["hoenam_cells_same_date"] >= alert_cell_threshold)
    ).astype(int)

    return output


def drop_rows_without_future_target(df: pd.DataFrame) -> pd.DataFrame:
    """Remove terminal rows per location where no next sample exists."""
    if "next_sample_available" not in df.columns:
        return df.copy()
    return df[df["next_sample_available"].eq(1)].copy()


def get_feature_columns_by_drop_rule(df: pd.DataFrame, drop_columns: list[str]) -> list[str]:
    existing_drop_columns = [col for col in drop_columns if col in df.columns]
    return [col for col in df.columns if col not in existing_drop_columns]


def check_leakage_columns(feature_columns: list[str], forbidden_keywords: list[str]) -> None:
    suspicious = [
        col for col in feature_columns if any(keyword in col.lower() for keyword in forbidden_keywords)
    ]
    if suspicious:
        raise ValueError(f"Potential leakage columns are included as features: {suspicious}")


def check_required_targets(
    df: pd.DataFrame,
    regression_target: str = REGRESSION_TARGET,
    classification_target: str = CLASSIFICATION_TARGET,
) -> None:
    missing = [col for col in [regression_target, classification_target] if col not in df.columns]
    if missing:
        raise KeyError(f"Target columns are missing from model input: {missing}")


def check_numeric_features(df: pd.DataFrame, feature_columns: list[str]) -> None:
    non_numeric = [col for col in feature_columns if not pd.api.types.is_numeric_dtype(df[col])]
    if non_numeric:
        raise TypeError(f"Non-numeric feature columns: {non_numeric}")


def get_feature_columns(df: pd.DataFrame, drop_columns: list[str]) -> list[str]:
    """Select features by dropping metadata, target, and leakage columns.

    Feature selection is explicit: DROP_COLUMNS removes metadata and targets,
    then leakage keywords and numeric dtypes are checked before training.
    """
    feature_columns = get_feature_columns_by_drop_rule(df, drop_columns)
    check_required_targets(df)
    check_leakage_columns(feature_columns, FORBIDDEN_FEATURE_KEYWORDS)

    if REQUIRE_NUMERIC_FEATURES:
        check_numeric_features(df, feature_columns)

    return feature_columns


def split_features_targets(
    df: pd.DataFrame,
    feature_columns: list[str],
    regression_target: str,
    classification_target: str,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    check_required_targets(df, regression_target, classification_target)
    x = df[feature_columns]
    y_reg = df[regression_target]
    y_cls = df[classification_target]
    return x, y_reg, y_cls


def time_based_train_valid_split(
    df: pd.DataFrame,
    date_column: str,
    split_column: str | None = None,
    valid_size_ratio: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if split_column and split_column in df.columns:
        train_df = df[df[split_column] == "train"].copy()
        valid_df = df[df[split_column].isin(["valid", "validation"])].copy()
        if len(train_df) == 0 or len(valid_df) == 0:
            raise ValueError("split column exists, but train/valid rows are empty. Check split labels.")
        return train_df, valid_df

    if date_column not in df.columns:
        raise KeyError(f"Date column not found: {date_column}")

    sorted_df = df.copy()
    sorted_df[date_column] = pd.to_datetime(sorted_df[date_column])
    sorted_df = sorted_df.sort_values(date_column).reset_index(drop=True)

    split_idx = int(len(sorted_df) * (1 - valid_size_ratio))
    train_df = sorted_df.iloc[:split_idx].copy()
    valid_df = sorted_df.iloc[split_idx:].copy()
    return train_df, valid_df
