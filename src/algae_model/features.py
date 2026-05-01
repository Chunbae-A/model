from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from .config import SITE_ORDER, SITE_TO_STATION, TARGET_COL, UPSTREAM, DataPaths


def to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.replace({"-": np.nan, "&nbsp;": np.nan, "정량한계미만": np.nan}),
        errors="coerce",
    )


def safe_div(a: pd.Series, b: pd.Series | float) -> pd.Series:
    return a / (b + 1e-6)


def log1p10(values: pd.Series | np.ndarray) -> pd.Series | np.ndarray:
    return np.log10(np.maximum(values, 0) + 1)


def positive_ln(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    return np.log(values.where(values > 0))


def read_csv_if_exists(path: Path | None) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None
    return pd.read_csv(path, low_memory=False)


def load_quality(path: Path) -> pd.DataFrame:
    q = pd.read_csv(path, low_memory=False)
    q["sample_date"] = pd.to_datetime(q["조사일"], errors="coerce")
    q["site"] = q["채수위치"].astype(str)
    numeric_cols = [
        TARGET_COL,
        "(1) Microcystis",
        "(2) Anabaena",
        "(3) Oscillatoria",
        "(4) Aphanizomenon",
        "수온(℃)",
        "pH",
        "DO(㎎/L)",
        "투명도",
        "탁도",
        "Chl-a (㎎/㎥)",
        "일조시간 합계(hr)",
        "일강수량(mm)",
    ]
    for col in numeric_cols:
        if col in q.columns:
            q[col] = to_num(q[col])
    return q.sort_values(["site", "sample_date"]).reset_index(drop=True)


def load_dam(path: Path) -> pd.DataFrame:
    dam = pd.read_csv(path, low_memory=False)
    dam["date"] = pd.to_datetime(dam["일시"], errors="coerce")
    rename = {
        "수위(EL.m)": "dam_level",
        "저수량(백만㎥)": "storage_volume",
        "저수율(%)": "storage_rate",
        "강우량(mm)": "rain",
        "유입량(㎥/s)": "inflow",
        "총방류량(㎥/s)": "outflow",
    }
    dam = dam.rename(columns=rename)
    for col in rename.values():
        dam[col] = to_num(dam[col])
    return dam[["date", *rename.values()]].sort_values("date")


def load_geum_dam(path: Path | None) -> pd.DataFrame | None:
    geum = read_csv_if_exists(path)
    if geum is None:
        return None
    if "OBSNM" in geum.columns:
        geum = geum[geum["OBSNM"].astype(str).eq("대청댐")].copy()
    geum["date"] = pd.to_datetime(geum["snapshot_datetime"].astype(str), format="%Y%m%d%H%M", errors="coerce")
    rename = {
        "SWL": "geum_level",
        "SFW": "geum_storage_volume",
        "VOL": "geum_storage_rate",
        "INF": "geum_inflow",
        "TOTOTF": "geum_outflow",
    }
    geum = geum.rename(columns=rename)
    keep = ["date", *[v for v in rename.values() if v in geum.columns]]
    for col in keep:
        if col != "date":
            geum[col] = to_num(geum[col])
    return geum[keep].sort_values("date")


def load_kma_daily(path: Path | None) -> pd.DataFrame | None:
    kma = read_csv_if_exists(path)
    if kma is None:
        return None
    kma["datetime"] = pd.to_datetime(kma["일시"], errors="coerce")
    kma["date"] = kma["datetime"].dt.floor("D")
    numeric = ["기온(°C)", "강수량(mm)", "풍속(m/s)", "습도(%)", "일조(hr)", "일사(MJ/m2)", "전운량(10분위)"]
    for col in numeric:
        if col in kma.columns:
            kma[col] = to_num(kma[col])
    return (
        kma.groupby(["지점명", "date"], as_index=False)
        .agg(
            avg_temp=("기온(°C)", "mean"),
            max_temp=("기온(°C)", "max"),
            daily_rain=("강수량(mm)", "sum"),
            avg_wind=("풍속(m/s)", "mean"),
            sunshine=("일조(hr)", "sum"),
            solar_rad=("일사(MJ/m2)", "sum"),
            cloud_cover=("전운량(10분위)", "mean"),
        )
        .sort_values(["지점명", "date"])
    )


def add_rolling_features(
    daily: pd.DataFrame,
    prefix: str,
    windows: tuple[int, ...] = (3, 7, 14),
    sum_cols: tuple[str, ...] = (),
    mean_cols: tuple[str, ...] = (),
) -> pd.DataFrame:
    daily = daily.copy().sort_values("date")
    out = daily[["date"]].copy()
    for col in sum_cols:
        for window in windows:
            out[f"{prefix}_{col}_{window}d_sum"] = daily[col].rolling(window, min_periods=1).sum()
        for lag in (1, 3, 7):
            out[f"{prefix}_{col}_lag{lag}"] = daily[col].shift(lag)
    for col in mean_cols:
        for window in windows:
            out[f"{prefix}_{col}_{window}d_mean"] = daily[col].rolling(window, min_periods=1).mean()
    return out


def build_hydro_features(paths: DataPaths) -> pd.DataFrame:
    dam = load_dam(paths.dam)
    dam["net_flow"] = dam["inflow"] - dam["outflow"]
    dam["residence_proxy"] = safe_div(dam["storage_volume"], dam["outflow"])
    dam["nutrient_stagnation"] = safe_div(dam["rain"] * dam["inflow"], dam["outflow"] + 1)
    feat = add_rolling_features(
        dam,
        "hydro",
        sum_cols=("rain", "inflow", "outflow", "net_flow"),
        mean_cols=("dam_level", "storage_rate", "residence_proxy", "nutrient_stagnation"),
    )
    feat["level_change_3d"] = dam["dam_level"] - dam["dam_level"].shift(3)
    feat["level_change_7d"] = dam["dam_level"] - dam["dam_level"].shift(7)
    feat["moderate_rain_days_10_30_7d"] = dam["rain"].between(10, 30).rolling(7, min_periods=1).sum()
    feat["heavy_rain_80_7d"] = dam["rain"].rolling(7, min_periods=1).max().ge(80).astype(float)

    geum = load_geum_dam(paths.geum_dam)
    if geum is not None and not geum.empty:
        geum["geum_net_flow"] = geum["geum_inflow"] - geum["geum_outflow"]
        geum["geum_residence_proxy"] = safe_div(geum["geum_storage_volume"], geum["geum_outflow"])
        geum_feat = add_rolling_features(
            geum,
            "geum",
            sum_cols=("geum_inflow", "geum_outflow", "geum_net_flow"),
            mean_cols=("geum_level", "geum_storage_rate", "geum_residence_proxy"),
        )
        feat = feat.merge(geum_feat, on="date", how="outer")
    return feat.rename(columns={"date": "sample_date"}).sort_values("sample_date")


def build_weather_features(samples: pd.DataFrame, paths: DataPaths) -> pd.DataFrame:
    kma = load_kma_daily(paths.kma)
    if kma is None or kma.empty:
        return samples[["site", "sample_date"]].drop_duplicates().copy()

    rows = []
    for site, station in SITE_TO_STATION.items():
        station_daily = kma[kma["지점명"].eq(station)].copy().sort_values("date")
        if station_daily.empty:
            continue
        roll = add_rolling_features(
            station_daily,
            "weather",
            sum_cols=("daily_rain", "sunshine", "solar_rad"),
            mean_cols=("avg_temp", "max_temp", "avg_wind", "cloud_cover"),
        )
        roll["site"] = site
        roll["hot_days_30c_7d"] = station_daily["max_temp"].ge(30).rolling(7, min_periods=1).sum()
        roll["low_wind_days_2ms_7d"] = station_daily["avg_wind"].le(2).rolling(7, min_periods=1).sum()
        roll["effective_light_7d"] = roll["weather_solar_rad_7d_sum"] * (1 - roll["weather_cloud_cover_7d_mean"] / 10)
        rows.append(roll.rename(columns={"date": "sample_date"}))
    if not rows:
        return samples[["site", "sample_date"]].drop_duplicates().copy()
    return pd.concat(rows, ignore_index=True)


def add_quality_features(q: pd.DataFrame) -> pd.DataFrame:
    df = q.copy()
    rename = {
        "수온(℃)": "water_temp",
        "pH": "ph",
        "DO(㎎/L)": "do",
        "투명도": "transparency",
        "탁도": "turbidity",
        "Chl-a (㎎/㎥)": "chla",
        "일조시간 합계(hr)": "sunshine_obs",
        "일강수량(mm)": "rain_obs",
        "(1) Microcystis": "microcystis",
        "(2) Anabaena": "anabaena",
        "(3) Oscillatoria": "oscillatoria",
        "(4) Aphanizomenon": "aphanizomenon",
    }
    df = df.rename(columns=rename)
    df["site_order"] = df["site"].map(SITE_ORDER)
    df["month"] = df["sample_date"].dt.month
    df["day_of_year"] = df["sample_date"].dt.dayofyear
    df["season_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["season_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)
    df["cells"] = df[TARGET_COL]
    df["log_cells"] = log1p10(df["cells"])
    df["alert"] = (df["cells"] >= 1000).astype(float)

    for col in ["microcystis", "anabaena", "oscillatoria", "aphanizomenon"]:
        if col in df.columns:
            df[f"{col}_ratio"] = safe_div(df[col].fillna(0), df["cells"].fillna(0))

    df["tsi_chla"] = 9.81 * positive_ln(df["chla"]) + 30.6
    df["tsi_transparency"] = 60 - 14.41 * positive_ln(df["transparency"])
    df["tsi_proxy_mean"] = df[["tsi_chla", "tsi_transparency"]].mean(axis=1)
    df["eutrophic_flag"] = df["tsi_proxy_mean"].ge(50).astype(float)
    df["hypertrophic_flag"] = df["tsi_proxy_mean"].ge(70).astype(float)
    df["temp_light_growth_proxy"] = np.maximum(df["water_temp"] - 20, 0) * df["sunshine_obs"].fillna(0)

    sequence_cols = [
        "cells",
        "log_cells",
        "alert",
        "water_temp",
        "ph",
        "do",
        "transparency",
        "turbidity",
        "chla",
        "tsi_proxy_mean",
        "microcystis",
        "anabaena",
        "oscillatoria",
        "aphanizomenon",
    ]
    for col in sequence_cols:
        if col in df.columns:
            df[f"prev_{col}"] = df.groupby("site")[col].shift(1)
            df[f"delta_{col}"] = df[col] - df[f"prev_{col}"]

    df["prev_sample_date"] = df.groupby("site")["sample_date"].shift(1)
    df["time_delta_days"] = (df["sample_date"] - df["prev_sample_date"]).dt.days
    df["next_sample_date"] = df.groupby("site")["sample_date"].shift(-1)
    df["target_cells_next"] = df.groupby("site")["cells"].shift(-1)
    df["target_log_cells_next"] = log1p10(df["target_cells_next"])
    df["target_alert_next"] = (df["target_cells_next"] >= 1000).astype(float)
    df.loc[df["target_cells_next"].isna(), "target_alert_next"] = np.nan
    return df


def add_hydro_topology_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["site", "sample_date"]).copy()
    lookup = {site: group.sort_values("sample_date") for site, group in df.groupby("site")}
    records = []
    for _, row in df.iterrows():
        upstream = UPSTREAM.get(row["site"])
        values = {
            "upstream_log_cells_21d": np.nan,
            "upstream_days_since": np.nan,
            "upstream_alert_21d": np.nan,
            "graph_decay_signal": np.nan,
        }
        if upstream in lookup:
            hist = lookup[upstream]
            hist = hist[
                (hist["sample_date"] < row["sample_date"])
                & (hist["sample_date"] >= row["sample_date"] - pd.Timedelta(days=21))
            ]
            if not hist.empty:
                latest = hist.iloc[-1]
                days = int((row["sample_date"] - latest["sample_date"]).days)
                decay = math.exp(-days / 7)
                flow_weight = 1.0
                if "hydro_inflow_7d_sum" in row and "hydro_outflow_7d_sum" in row:
                    inflow = row.get("hydro_inflow_7d_sum")
                    outflow = row.get("hydro_outflow_7d_sum")
                    if pd.notna(inflow) and pd.notna(outflow):
                        flow_weight = float(np.tanh((inflow + 1) / (outflow + 1)) + 0.5)
                values = {
                    "upstream_log_cells_21d": latest["log_cells"],
                    "upstream_days_since": days,
                    "upstream_alert_21d": latest["alert"],
                    "graph_decay_signal": latest["log_cells"] * decay * flow_weight,
                }
        records.append(values)
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(records)], axis=1)


def build_master_table(paths: DataPaths) -> pd.DataFrame:
    quality = add_quality_features(load_quality(paths.quality))
    hydro = build_hydro_features(paths)
    weather = build_weather_features(quality[["site", "sample_date"]], paths)
    master = quality.merge(hydro, on="sample_date", how="left")
    master = master.merge(weather, on=["site", "sample_date"], how="left")
    master = add_hydro_topology_features(master)
    return master
