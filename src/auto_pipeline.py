import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# support running both as script and as package
try:
    from . import kma_fetcher
    from .data_prep import prepare_and_merge
except Exception:
    # add repo root to sys.path so `import src.*` works
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        import src.kma_fetcher as kma_fetcher
        from src.data_prep import prepare_and_merge
    except Exception:
        # fallback: import by filename if everything else fails
        import importlib.util
        spec = importlib.util.spec_from_file_location("kma_fetcher", os.path.join(repo_root, "src", "kma_fetcher.py"))
        kma_fetcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(kma_fetcher)
        spec2 = importlib.util.spec_from_file_location("data_prep", os.path.join(repo_root, "src", "data_prep.py"))
        data_prep = importlib.util.module_from_spec(spec2)
        spec2.loader.exec_module(data_prep)
        prepare_and_merge = data_prep.prepare_and_merge


def run_pipeline(
    kma_service_key: str | None = None,
    nx: int | None = None,
    ny: int | None = None,
    station: int | None = None,
    water_csv: str | None = None,
    weather_csv: str | None = None,
    out_model_input: str | None = None,
):
    # load environment
    load_dotenv()
    key = kma_service_key or os.getenv("KMA_SERVICE_KEY")
    nx = nx or int(os.getenv("KMA_NX", "60"))
    ny = ny or int(os.getenv("KMA_NY", "127"))
    station = station or int(os.getenv("KMA_STATION", "604"))

    if not key:
        raise SystemExit("KMA_SERVICE_KEY not set. Please set in .env or pass as argument.")

    today = None
    # fetch latest hourly, aggregate to daily and merge into weather CSV
    hourly_out = Path("data/WEATHER_kma_raw.csv")
    merged_weather = Path(weather_csv or "data/WEATHER.csv")

    df_hourly = kma_fetcher.fetch_vilage_forecast(key, nx, ny, datetime_now_str(), default_base_time())
    if df_hourly.empty:
        print("No forecast fetched; aborting pipeline.")
        return

    hourly_out.parent.mkdir(parents=True, exist_ok=True)
    df_hourly.to_csv(hourly_out, index=False)
    print(f"Saved hourly forecast to {hourly_out}")

    daily = kma_fetcher.aggregate_hourly_to_daily(df_hourly)
    if daily.empty:
        print("No daily aggregates produced; aborting.")
        return

    kma_fetcher.merge_into_weather_csv(daily, out_path=str(merged_weather), station=station)
    print(f"Merged daily weather into {merged_weather}")

    # run data_prep to produce model input
    water = Path(water_csv or "data/daechung_for_merge_v1.csv")
    out = Path(out_model_input or "data/processed/model_input/algae_model_input.csv")
    out.parent.mkdir(parents=True, exist_ok=True)

    merged = prepare_and_merge(water, merged_weather, out)
    print(f"Prepared model input at {out} rows={len(merged)}")


def datetime_now_str():
    from datetime import datetime

    return datetime.utcnow().strftime("%Y%m%d")


def default_base_time():
    # crude default: KMA accepts certain base times; use 0500 as common
    return "0500"


if __name__ == "__main__":
    # simple CLI that loads .env and runs pipeline
    run_pipeline()
