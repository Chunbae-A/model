from __future__ import annotations

from pathlib import Path
import unicodedata

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

DAM_RAW = DATA_DIR / "대청수문_10년치_통합데이터.csv"
WATER_RAW = DATA_DIR / "수질_10년치_통합데이터.csv"
WEATHER = DATA_DIR / "WEATHER.csv"
DAM_RAW_CANDIDATES = [
    DAM_RAW,
    DATA_DIR / "daechung_dam_10y.csv",
    DATA_DIR / "dam_10y.csv",
]
WATER_RAW_CANDIDATES = [
    WATER_RAW,
    DATA_DIR / "daechung_water_quality_10y.csv",
    DATA_DIR / "water_quality_10y.csv",
]
WEATHER_CANDIDATES = [
    WEATHER,
    DATA_DIR / "weather.csv",
]
WATER_DATA_DIR = ROOT / "water_data"

WATER_HYDRO_OUT = DATA_DIR / "daechung_water_hydro_10y_clean.csv"
FINAL_OUT = DATA_DIR / "daechung_final_clean_dataset.csv"
FINAL_ALIAS_OUT = DATA_DIR / "Final.csv"
COMBINED_OUT = DATA_DIR / "combined_weather_water_10y.csv"
MODEL_INPUT_OUT = DATA_DIR / "processed" / "model_input" / "algae_model_input.csv"


LOC_MAP = {"문의": 0, "추동": 1, "회남": 2}

# The source weather table has four AWS stations. These are the nearest station
# assignments used to keep one weather record per water-quality observation.
LOC_TO_STATION = {
    0: 888,  # 문의 -> 청남대
    1: 643,  # 추동 -> 세천
    2: 604,  # 회남 -> 옥천
}


def normalize_filename(name: str) -> str:
    return unicodedata.normalize("NFC", name).casefold()


def find_data_file(candidates: list[Path], *, label: str) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate

    if DATA_DIR.exists():
        by_name = {normalize_filename(path.name): path for path in DATA_DIR.iterdir() if path.is_file()}
        for candidate in candidates:
            matched = by_name.get(normalize_filename(candidate.name))
            if matched:
                return matched

    candidate_list = "\n".join(f" - {path}" for path in candidates)
    raise FileNotFoundError(
        f"Missing {label} source file. Looked for:\n{candidate_list}\n"
        "The data/ directory is gitignored, so a fresh clone needs source CSVs copied in "
        "or regenerated with `python src/pipeline.py --fetch water/weather/all`."
    )


def _excel_files() -> list[Path]:
    if not WATER_DATA_DIR.exists():
        return []
    return sorted(
        [
            *WATER_DATA_DIR.glob("*.xls"),
            *WATER_DATA_DIR.glob("*.xlsx"),
            *WATER_DATA_DIR.glob("*/*.xls"),
            *WATER_DATA_DIR.glob("*/*.xlsx"),
        ]
    )


def _classify_excel(file_path: Path) -> str | None:
    try:
        sample = pd.read_excel(file_path, header=None, nrows=4)
    except Exception:
        return None

    text = " ".join(sample.astype("string").fillna("").to_numpy().ravel().tolist())
    if "대청댐" in text or "수위(EL.m)" in text or file_path.name.startswith("excelDataList"):
        return "dam"
    if "조류모니터링" in text or "유해남조류" in text or "Chl-a" in text or "과거수질" in file_path.name:
        return "water"
    return None


def build_source_csvs_from_excels(force: bool = False) -> None:
    """Rebuild ignored raw data CSVs from tracked/downloaded Excel files."""
    need_dam = force or not any(path.exists() for path in DAM_RAW_CANDIDATES)
    need_water = force or not any(path.exists() for path in WATER_RAW_CANDIDATES)
    if not need_dam and not need_water:
        return

    dam_frames: list[pd.DataFrame] = []
    water_frames: list[pd.DataFrame] = []
    for file_path in _excel_files():
        kind = _classify_excel(file_path)
        if kind == "dam" and need_dam:
            dam_frames.append(pd.read_excel(file_path, header=None))
        elif kind == "water" and need_water:
            water_frames.append(pd.read_excel(file_path, header=None))

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if need_dam:
        if not dam_frames:
            raise FileNotFoundError(f"No dam Excel files found under {WATER_DATA_DIR}")
        dam_raw = pd.concat(dam_frames, ignore_index=True, sort=False)
        dam_raw.to_csv(DAM_RAW, index=False, header=False, encoding="utf-8-sig")
        print(f"[preprocess] rebuilt dam raw csv: {DAM_RAW} rows={len(dam_raw):,}")

    if need_water:
        if not water_frames:
            raise FileNotFoundError(f"No water-quality Excel files found under {WATER_DATA_DIR}")
        water_raw = pd.concat(water_frames, ignore_index=True, sort=False)
        water_raw.to_csv(WATER_RAW, index=False, header=False, encoding="utf-8-sig")
        print(f"[preprocess] rebuilt water raw csv: {WATER_RAW} rows={len(water_raw):,}")


def ensure_local_source_csvs() -> None:
    build_source_csvs_from_excels(force=False)


def to_number(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype("string")
        .str.replace(",", "", regex=False)
        .str.replace("정량한계미만", "0", regex=False)
        .str.strip()
    )
    return pd.to_numeric(cleaned, errors="coerce")


def read_raw_csv(path: Path, *, min_columns: int, label: str) -> pd.DataFrame:
    raw = pd.read_csv(path, header=None, dtype=str, low_memory=False, encoding="utf-8-sig")
    if raw.shape[1] < min_columns:
        raise ValueError(
            f"{label} file has {raw.shape[1]} columns, but at least {min_columns} are required: {path}"
        )
    return raw


def parse_dam(path: Path | None = None) -> pd.DataFrame:
    path = path or find_data_file(DAM_RAW_CANDIDATES, label="dam operation")
    raw = read_raw_csv(path, min_columns=7, label="Dam operation")
    date_columns = [0, *range(7, 17)]
    frames: list[pd.DataFrame] = []

    for date_col in date_columns:
        if date_col >= raw.shape[1]:
            continue

        chunk = pd.DataFrame(
            {
                "date": pd.to_datetime(raw[date_col], format="%Y-%m-%d", errors="coerce"),
                "water_level": to_number(raw[1]),
                "storage_vol": to_number(raw[2]),
                "storage_rate": to_number(raw[3]),
                "dam_rain": to_number(raw[4]),
                "inflow": to_number(raw[5]),
                "outflow": to_number(raw[6]),
            }
        )
        chunk = chunk.dropna(subset=["date"])
        frames.append(chunk)

    dam = pd.concat(frames, ignore_index=True)
    dam = dam.drop_duplicates("date", keep="first")
    dam = dam.sort_values("date").reset_index(drop=True)

    numeric_cols = [c for c in dam.columns if c != "date"]
    dam[numeric_cols] = dam[numeric_cols].fillna(0)

    dam["level_change_7d"] = dam["water_level"].diff(7)
    dam["dam_rain_7d_sum"] = dam["dam_rain"].rolling(7, min_periods=1).sum()
    dam["inflow_7d_sum"] = dam["inflow"].rolling(7, min_periods=1).sum()
    dam["outflow_7d_sum"] = dam["outflow"].rolling(7, min_periods=1).sum()
    dam["residence_proxy"] = dam["storage_vol"] / (dam["outflow"] + 1)
    dam["nutrient_stagnation_index"] = (
        dam["dam_rain_7d_sum"] * dam["inflow_7d_sum"] / (dam["outflow_7d_sum"] + 1)
    )
    return dam


def parse_water(path: Path | None = None) -> pd.DataFrame:
    path = path or find_data_file(WATER_RAW_CANDIDATES, label="water quality")
    raw = read_raw_csv(path, min_columns=15, label="Water quality")
    rows = raw[raw[2].isin(LOC_MAP)].copy()

    def value_at(primary_col: int, fallback_col: int) -> pd.Series:
        primary = rows[primary_col] if primary_col in rows.columns else pd.Series(index=rows.index, dtype="string")
        fallback = rows[fallback_col] if fallback_col in rows.columns else pd.Series(index=rows.index, dtype="string")
        return primary.where(primary.notna(), fallback)

    water = pd.DataFrame(
        {
            "date": pd.to_datetime(rows[3], format="%Y.%m.%d", errors="coerce"),
            "loc_encoded": rows[2].map(LOC_MAP),
            "water_temp": to_number(rows[4]),
            "pH": to_number(rows[5]),
            "DO": to_number(rows[6]),
            "transparency": to_number(value_at(7, 18)),
            "turbidity": to_number(value_at(8, 19)),
            "Chl_a": to_number(value_at(9, 20)),
            "cyano_cells": to_number(value_at(10, 21)),
            "microcystis": to_number(value_at(11, 22)),
            "anabaena": to_number(value_at(12, 23)),
            "oscillatoria": to_number(value_at(13, 24)),
            "aphanizomenon": to_number(value_at(14, 25)),
        }
    )
    water = water.dropna(subset=["date", "loc_encoded"])
    water = water[
        water["date"].between(pd.Timestamp("2016-04-01"), pd.Timestamp("2025-12-31"))
    ]

    numeric_cols = [c for c in water.columns if c != "date"]
    water[numeric_cols] = water[numeric_cols].fillna(0)
    water["loc_encoded"] = water["loc_encoded"].astype(int)

    water = (
        water.sort_values(["loc_encoded", "date"])
        .drop_duplicates(["loc_encoded", "date"], keep="first")
        .reset_index(drop=True)
    )
    water["acc_temp_7d"] = water.groupby("loc_encoded")["water_temp"].transform(
        lambda s: s.rolling(7, min_periods=1).sum()
    )
    water["TSI_Chla"] = 9.81 * np.log(water["Chl_a"].replace(0, 0.01)) + 30.6
    water["TSI_SD"] = 60 - 14.41 * np.log(water["transparency"].replace(0, 0.01))

    for col in ["microcystis", "anabaena", "oscillatoria", "aphanizomenon"]:
        water[f"{col}_ratio"] = water[col] / (water["cyano_cells"] + 1)

    water["alert_encoded"] = np.select(
        [
            water["cyano_cells"].ge(1_000_000),
            water["cyano_cells"].ge(10_000),
            water["cyano_cells"].ge(1_000),
        ],
        [3, 2, 1],
        default=0,
    )
    water["log_target"] = np.log10(water["cyano_cells"] + 1)
    water["next_log_cells"] = water.groupby("loc_encoded")["log_target"].shift(-1)

    water["water_temp_diff_1d"] = water.groupby("loc_encoded")["water_temp"].diff(1)
    return water


def merge_water_hydro(water: pd.DataFrame, dam: pd.DataFrame) -> pd.DataFrame:
    merged = pd.merge(water, dam, how="left", on="date")
    hydro_cols = [c for c in dam.columns if c != "date"]
    merged[hydro_cols] = merged[hydro_cols].ffill().bfill()

    merged = merged.sort_values(["loc_encoded", "date"]).reset_index(drop=True)
    merged["inflow_diff_1d"] = merged.groupby("loc_encoded")["inflow"].diff(1)
    merged["water_level_diff_1d"] = merged.groupby("loc_encoded")["water_level"].diff(1)
    merged["heat_stagnation_index"] = merged["water_temp"] * merged["residence_proxy"]
    merged["rain_lag_1d"] = merged.groupby("loc_encoded")["dam_rain"].shift(1)
    merged["rain_lag_2d"] = merged.groupby("loc_encoded")["dam_rain"].shift(2)
    return merged


def merge_weather(water_hydro: pd.DataFrame, weather_path: Path | None = None) -> pd.DataFrame:
    weather_path = weather_path or find_data_file(WEATHER_CANDIDATES, label="weather")
    weather = pd.read_csv(weather_path, encoding="utf-8-sig")
    weather["date"] = pd.to_datetime(weather["date"], errors="coerce")
    weather["station"] = pd.to_numeric(weather["station"], errors="coerce").astype("Int64")

    output = water_hydro.copy()
    output["station"] = output["loc_encoded"].map(LOC_TO_STATION).astype(int)
    output = pd.merge(output, weather, how="left", on=["station", "date"], suffixes=("", "_weather"))

    return output


def finalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    final_columns = [
        "station",
        "date",
        "avg_temp",
        "min_temp",
        "max_temp",
        "daily_rain",
        "max_wind_gust",
        "avg_wind",
        "air_temp_7d_mean",
        "hot_days_30c_7d",
        "rain_3d_sum",
        "rain_7d_sum",
        "rain_14d_sum",
        "wind_7d_mean",
        "low_wind_days_2ms_7d",
        "sunshine",
        "solar_rad",
        "cloud_cover",
        "sunshine_7d_sum",
        "solar_7d_sum",
        "cloud_7d_mean",
        "sin_season",
        "cos_season",
        "loc_encoded",
        "water_temp",
        "pH",
        "DO",
        "transparency",
        "turbidity",
        "Chl_a",
        "cyano_cells",
        "microcystis",
        "anabaena",
        "oscillatoria",
        "aphanizomenon",
        "acc_temp_7d",
        "TSI_Chla",
        "TSI_SD",
        "microcystis_ratio",
        "anabaena_ratio",
        "oscillatoria_ratio",
        "aphanizomenon_ratio",
        "water_level",
        "storage_vol",
        "storage_rate",
        "dam_rain",
        "inflow",
        "outflow",
        "level_change_7d",
        "dam_rain_7d_sum",
        "inflow_7d_sum",
        "outflow_7d_sum",
        "residence_proxy",
        "nutrient_stagnation_index",
        "alert_encoded",
        "log_target",
        "next_log_cells",
        "water_temp_diff_1d",
        "inflow_diff_1d",
        "water_level_diff_1d",
        "heat_stagnation_index",
        "rain_lag_1d",
        "rain_lag_2d",
    ]

    missing_columns = [col for col in final_columns if col not in df.columns]
    if missing_columns:
        available = ", ".join(df.columns)
        raise KeyError(
            f"Missing columns while building Final.csv: {missing_columns}\n"
            f"Available columns: {available}"
        )

    final = df.loc[:, final_columns].copy()
    final = final.dropna(subset=["next_log_cells", "nutrient_stagnation_index"])
    final = final.sort_values(["loc_encoded", "date"]).reset_index(drop=True)

    numeric_cols = [c for c in final.columns if c != "date"]
    final[numeric_cols] = final[numeric_cols].apply(pd.to_numeric, errors="coerce")
    final[numeric_cols] = final[numeric_cols].fillna(0)
    final["station"] = final["station"].astype(int)
    final["loc_encoded"] = final["loc_encoded"].astype(int)
    final["alert_encoded"] = final["alert_encoded"].astype(int)
    final["date"] = pd.to_datetime(final["date"]).dt.strftime("%Y-%m-%d")
    return final


def main() -> None:
    ensure_local_source_csvs()
    dam = parse_dam()
    water = parse_water()
    water_hydro = merge_water_hydro(water, dam)
    final = finalize_columns(merge_weather(water_hydro))

    WATER_HYDRO_OUT.parent.mkdir(parents=True, exist_ok=True)
    MODEL_INPUT_OUT.parent.mkdir(parents=True, exist_ok=True)

    water_hydro.to_csv(WATER_HYDRO_OUT, index=False, encoding="utf-8-sig")
    final.to_csv(FINAL_OUT, index=False, encoding="utf-8-sig")
    final.to_csv(FINAL_ALIAS_OUT, index=False, encoding="utf-8-sig")
    final.to_csv(COMBINED_OUT, index=False, encoding="utf-8-sig")
    final.to_csv(MODEL_INPUT_OUT, index=False, encoding="utf-8-sig")

    non_date_object_cols = [
        col for col in final.select_dtypes(include=["object", "str"]).columns if col != "date"
    ]
    print("--- build complete ---")
    print(f"dam rows: {len(dam):,}")
    print(f"water rows after 2016-04-01: {len(water):,}")
    print(f"water+hydro rows: {len(water_hydro):,}")
    print(f"final shape: {final.shape}")
    print(f"non-date object columns: {non_date_object_cols}")
    print(f"outputs: {FINAL_OUT}, {FINAL_ALIAS_OUT}, {COMBINED_OUT}, {MODEL_INPUT_OUT}")


if __name__ == "__main__":
    main()
