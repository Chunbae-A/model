import os
import argparse
import requests
import pandas as pd
from datetime import datetime, timedelta

KMA_BASE = "http://apis.data.go.kr/1360000/VilageFcstInfoService/getVilageFcst"


def fetch_vilage_forecast(service_key: str, nx: int, ny: int, base_date: str, base_time: str, num_of_rows: int = 1000):
    params = {
        "serviceKey": service_key,
        "pageNo": "1",
        "numOfRows": str(num_of_rows),
        "dataType": "JSON",
        "base_date": base_date,
        "base_time": base_time,
        "nx": str(nx),
        "ny": str(ny),
    }
    r = requests.get(KMA_BASE, params=params, timeout=30)
    r.raise_for_status()
    j = r.json()
    items = j.get("response", {}).get("body", {}).get("items", {}).get("item", [])
    if not items:
        return pd.DataFrame()

    rows = []
    for it in items:
        # each item has fields: category, fcstDate, fcstTime, fcstValue
        rows.append({
            "category": it.get("category"),
            "fcstDate": it.get("fcstDate"),
            "fcstTime": it.get("fcstTime"),
            "fcstValue": it.get("fcstValue"),
        })

    df = pd.DataFrame(rows)
    # pivot so that categories become columns
    df["datetime"] = pd.to_datetime(df["fcstDate"] + df["fcstTime"], format="%Y%m%d%H%M")
    df_pivot = df.pivot_table(index="datetime", columns="category", values="fcstValue", aggfunc="first")
    # convert numeric columns
    for c in df_pivot.columns:
        df_pivot[c] = pd.to_numeric(df_pivot[c], errors="coerce")

    df_pivot = df_pivot.reset_index()
    return df_pivot


def aggregate_hourly_to_daily(df_hourly: pd.DataFrame):
    df = df_hourly.copy()
    df["date"] = df["datetime"].dt.date
    agg = {
        "T1H": ["mean", "min", "max"],
        "RN1": ["sum"],
        "WSD": ["mean", "max"],
    }
    # only keep cols that exist
    use = {k: v for k, v in agg.items() if k in df.columns}
    if not use:
        return pd.DataFrame()
    g = df.groupby("date").agg(use)
    # flatten columns
    g.columns = ["_" . join([c for c in col if c]) for col in g.columns]
    g = g.reset_index()
    # normalize column names similar to existing WEATHER.csv
    rename_map = {}
    if "T1H_mean" in g.columns:
        rename_map["T1H_mean"] = "avg_temp"
    if "T1H_min" in g.columns:
        rename_map["T1H_min"] = "min_temp"
    if "T1H_max" in g.columns:
        rename_map["T1H_max"] = "max_temp"
    if "RN1_sum" in g.columns:
        rename_map["RN1_sum"] = "daily_rain"
    if "WSD_mean" in g.columns:
        rename_map["WSD_mean"] = "avg_wind"
    if "WSD_max" in g.columns:
        rename_map["WSD_max"] = "max_wind_gust"

    g = g.rename(columns=rename_map)
    return g


def merge_into_weather_csv(daily_df: pd.DataFrame, out_path: str = "data/WEATHER.csv", station: int = None):
    out_path = os.path.abspath(out_path)
    if os.path.exists(out_path):
        base = pd.read_csv(out_path, parse_dates=["date"])
        base["date"] = pd.to_datetime(base["date"]).dt.date
    else:
        base = pd.DataFrame()

    daily_df_local = daily_df.copy()
    daily_df_local["date"] = pd.to_datetime(daily_df_local["date"]).dt.date
    if station is not None:
        daily_df_local["station"] = station

    if base.empty:
        # create minimal WEATHER.csv format
        cols = ["station", "date"] + [c for c in daily_df_local.columns if c not in ["date", "station"]]
        out = daily_df_local[cols]
    else:
        # merge/replace by date & station if provided, otherwise by date
        if "station" in base.columns and "station" in daily_df_local.columns:
            merged = pd.concat([base[~base.set_index(["station", "date"]).index.isin(daily_df_local.set_index(["station", "date"]).index)],], ignore_index=True)
            out = pd.concat([merged, daily_df_local], ignore_index=True)
        else:
            # replace by date
            base = base[~base["date"].isin(daily_df_local["date"])]
            out = pd.concat([base, daily_df_local], ignore_index=True)

    # try to keep original column order if possible
    out = out.sort_values(by=["date"]) if "date" in out.columns else out
    out.to_csv(out_path, index=False)
    return out


def main():
    parser = argparse.ArgumentParser(description="Fetch KMA village forecast and append to WEATHER.csv")
    parser.add_argument("--nx", type=int, required=True, help="KMA grid x (nx)")
    parser.add_argument("--ny", type=int, required=True, help="KMA grid y (ny)")
    parser.add_argument("--date", help="base_date YYYYMMDD (default: today)")
    parser.add_argument("--time", help="base_time HHMM (default: nearest base time)")
    parser.add_argument("--service-key", help="KMA service key (or set KMA_SERVICE_KEY env var)")
    parser.add_argument("--out", default="data/WEATHER_kma_raw.csv", help="raw hourly output path")
    parser.add_argument("--merge-into", default="data/WEATHER.csv", help="path to existing WEATHER.csv to merge daily aggregates into")
    parser.add_argument("--station", type=int, help="station id to tag merged rows")
    args = parser.parse_args()

    key = args.service_key or os.getenv("KMA_SERVICE_KEY")
    if not key:
        raise SystemExit("KMA service key required: set KMA_SERVICE_KEY or pass --service-key")

    # defaults
    base_date = args.date or datetime.utcnow().strftime("%Y%m%d")
    # choose a base_time that KMA accepts; common base_times: 0200,0500,0800,... use 0200 as fallback
    base_time = args.time or datetime.utcnow().strftime("%H00")

    df_hourly = fetch_vilage_forecast(key, args.nx, args.ny, base_date, base_time)
    if df_hourly.empty:
        print("No items returned from KMA for given inputs.")
        return

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df_hourly.to_csv(args.out, index=False)
    print(f"Saved hourly raw forecast to {args.out}")

    daily = aggregate_hourly_to_daily(df_hourly)
    if daily.empty:
        print("Daily aggregation produced no rows.")
        return

    merged = merge_into_weather_csv(daily, out_path=args.merge_into, station=args.station)
    print(f"Merged daily aggregates into {args.merge_into}")


if __name__ == "__main__":
    main()
