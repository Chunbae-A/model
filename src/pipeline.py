from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import date
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
DATA_DIR = ROOT / "data"
sys.path.insert(0, str(ROOT))

RAW_DAM = DATA_DIR / "대청수문_10년치_통합데이터.csv"
RAW_WATER = DATA_DIR / "수질_10년치_통합데이터.csv"
WEATHER = DATA_DIR / "WEATHER.csv"
FINAL = DATA_DIR / "Final.csv"
MODEL_INPUT = DATA_DIR / "processed" / "model_input" / "algae_model_input.csv"


def log(message: str) -> None:
    print(f"[pipeline] {message}", flush=True)


def run_command(args: list[str], *, required: bool = True) -> None:
    log("running: " + " ".join(args))
    result = subprocess.run(args, cwd=ROOT)
    if required and result.returncode != 0:
        raise SystemExit(result.returncode)
    if result.returncode != 0:
        log(f"optional step failed with exit code {result.returncode}")


def run_python(script: str, *, required: bool = True) -> None:
    run_command([sys.executable, str(ROOT / script)], required=required)


def set_runtime_env() -> None:
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")


def ensure_inputs(paths: list[Path], hint: str) -> None:
    missing = [path for path in paths if not path.exists()]
    if missing:
        pretty = "\n".join(f" - {path}" for path in missing)
        raise FileNotFoundError(f"Missing required input files:\n{pretty}\n{hint}")


def fetch_water_sources() -> None:
    log("fetching dam operation data with Selenium crawler")
    run_python("src/water_gate.py")
    log("fetching algae/water-quality data with Selenium crawler")
    run_python("src/water_quality.py")
    ensure_inputs(
        [RAW_DAM, RAW_WATER],
        "Crawler finished, but expected raw CSV names were not found. Check data/ and water_data/ downloads.",
    )


def fetch_weather_sources(start: date, end: date) -> None:
    if not os.environ.get("KMA_SERVICE_KEY"):
        log("KMA_SERVICE_KEY is not set. ASOS fetch can still run, but AWS-only fields may use existing/fallback data.")

    log(f"fetching weather API data: {start.isoformat()} ~ {end.isoformat()}")
    from src.weather_api import run as run_weather

    run_weather(start, end)
    if not WEATHER.exists():
        # Older weather_api.py writes combined_weather_water_10y.csv. Keep the
        # pipeline explicit: the clean builder expects WEATHER.csv as the
        # canonical weather feature table.
        raise FileNotFoundError(
            f"{WEATHER} was not created. Run weather preprocessing or provide the existing WEATHER.csv."
        )


def preprocess() -> None:
    ensure_inputs(
        [RAW_DAM, RAW_WATER, WEATHER],
        "Use --fetch water/weather/all, or place the prepared source files under data/.",
    )
    log("building clean merged dataset and model input")
    run_python("src/preprocess_data.py")
    validate_final_dataset()


def validate_final_dataset() -> None:
    ensure_inputs([FINAL, MODEL_INPUT], "Preprocessing did not finish successfully.")
    df = pd.read_csv(FINAL)
    duplicate_count = int(df.duplicated(["date", "loc_encoded"]).sum())
    missing_count = int(df.isna().sum().sum())
    non_date_objects = [c for c in df.select_dtypes(include=["object", "str"]).columns if c != "date"]

    if duplicate_count:
        raise ValueError(f"Final.csv has duplicate date+loc_encoded keys: {duplicate_count}")
    if missing_count:
        raise ValueError(f"Final.csv has missing values: {missing_count}")
    if non_date_objects:
        raise TypeError(f"Final.csv has non-numeric object columns: {non_date_objects}")

    log(f"Final.csv ready: shape={df.shape}, date={df['date'].min()}~{df['date'].max()}")


def train_model(skip_plot: bool) -> None:
    log("training models")
    run_python("src/train_models.py")
    if not skip_plot and (SRC_DIR / "plot_metrics.py").exists():
        log("creating model comparison plot")
        run_python("src/plot_metrics.py", required=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch data, preprocess it into Final.csv, and train the algae prediction models."
    )
    parser.add_argument(
        "--fetch",
        choices=["none", "water", "weather", "all"],
        default="none",
        help="Fetch fresh source data before preprocessing. Default uses existing data files.",
    )
    parser.add_argument("--start-date", default="2016-01-01", help="Weather fetch start date, YYYY-MM-DD.")
    parser.add_argument("--end-date", default="2025-12-31", help="Weather fetch end date, YYYY-MM-DD.")
    parser.add_argument("--skip-preprocess", action="store_true", help="Skip Final.csv/model input rebuild.")
    parser.add_argument("--skip-train", action="store_true", help="Skip model training.")
    parser.add_argument("--skip-plot", action="store_true", help="Skip optional comparison plot.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_runtime_env()

    start = date.fromisoformat(args.start_date)
    end = date.fromisoformat(args.end_date)
    if start > end:
        raise ValueError("--start-date must be earlier than or equal to --end-date")

    log(f"workspace: {ROOT}")
    log(f"fetch mode: {args.fetch}")

    if args.fetch in {"water", "all"}:
        fetch_water_sources()
    if args.fetch in {"weather", "all"}:
        fetch_weather_sources(start, end)

    if not args.skip_preprocess:
        preprocess()
    else:
        validate_final_dataset()

    if not args.skip_train:
        train_model(args.skip_plot)

    log("done")
    log(f"final data: {FINAL}")
    log(f"model input: {MODEL_INPUT}")
    log(f"artifacts: {ROOT / 'artifacts'}")


if __name__ == "__main__":
    main()
