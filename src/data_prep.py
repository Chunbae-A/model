from __future__ import annotations

from pathlib import Path
import pandas as pd

from .config import MODEL_INPUT_PATH


def prepare_and_merge(water_csv: Path, weather_csv: Path, out_path: Path = MODEL_INPUT_PATH, station_to_loc_map: dict | None = None) -> pd.DataFrame:
    """Merge water quality and weather files into a single model input CSV.

    - water_csv: file containing water observations (has 조사일 and loc_encoded)
    - weather_csv: weather file with `station` and `date`
    - station_to_loc_map: optional dict mapping weather station -> loc_encoded

    If station_to_loc_map is provided, weather rows are mapped per station and merged per loc/date.
    Otherwise weather is aggregated by date (mean) and merged on date only.
    """
    wdf = pd.read_csv(water_csv)
    # normalize date column name
    if "조사일" in wdf.columns:
        wdf = wdf.rename(columns={"조사일": "date"})

    wdf["date"] = pd.to_datetime(wdf["date"]).dt.date

    weather = pd.read_csv(weather_csv)
    weather["date"] = pd.to_datetime(weather["date"]).dt.date

    if station_to_loc_map:
        # map station -> loc code
        weather["loc_encoded"] = weather["station"].map(station_to_loc_map)
        merged = pd.merge(wdf, weather, how="left", on=["date", "loc_encoded"])
    else:
        # aggregate weather by date (mean) then merge
        numeric_cols = weather.select_dtypes("number").columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in ("station",)]
        agg = weather.groupby("date").agg({col: "mean" for col in numeric_cols}).reset_index()
        merged = pd.merge(wdf, agg, how="left", on="date")

    # ensure output dir exists and save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    return merged


if __name__ == "__main__":
    # quick CLI: model/data/daechung_for_merge_v1.csv and model/data/WEATHER.csv
    base = Path(__file__).parents[1] / "data"
    water = base / "daechung_for_merge_v1.csv"
    weather = base / "WEATHER.csv"
    out = base / "processed" / "model_input" / "algae_model_input.csv"
    print("Preparing and merging:", water, weather, "->", out)
    df = prepare_and_merge(water, weather, out)
    print("Merged rows:", len(df))
