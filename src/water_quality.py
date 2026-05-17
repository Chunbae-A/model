from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
from selenium import webdriver
from selenium.common.exceptions import NoAlertPresentException, UnexpectedAlertPresentException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select


ROOT = Path(__file__).resolve().parents[1]
DOWNLOAD_DIR = ROOT / "water_data" / "quality"
OUTPUT_PATH = ROOT / "data" / "수질_10년치_통합데이터.csv"

ALGAE_URL = "https://water.nier.go.kr/web/algaePreMeasure?pMENU_NO=111"
TARGET_STATIONS = ["대청호(문의)", "대청호(회남)", "대청호(추동)"]


def make_driver(download_dir: Path, headless: bool = False) -> webdriver.Chrome:
    download_dir.mkdir(parents=True, exist_ok=True)
    options = webdriver.ChromeOptions()
    options.add_experimental_option(
        "prefs",
        {
            "download.default_directory": str(download_dir),
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
        },
    )
    if headless:
        options.add_argument("--headless=new")
    return webdriver.Chrome(options=options)


def list_excel_files(download_dir: Path) -> set[Path]:
    return set(download_dir.glob("*.xlsx")) | set(download_dir.glob("*.xls"))


def wait_for_new_download(download_dir: Path, before_files: set[Path], timeout_seconds: int = 90) -> Path:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        current_files = list_excel_files(download_dir)
        new_files = current_files - before_files
        if new_files and not list(download_dir.glob("*.crdownload")):
            return max(new_files, key=lambda path: path.stat().st_mtime)
        time.sleep(1)
    raise TimeoutError(f"No new Excel download finished within {timeout_seconds} seconds: {download_dir}")


def two_year_ranges(start_year: int, end_year: int) -> list[tuple[str, str, str, str]]:
    ranges = []
    year = start_year
    while year <= end_year:
        ranges.append((str(year), "01", str(min(year + 1, end_year)), "12"))
        year += 2
    return ranges


def select_stations(driver: webdriver.Chrome) -> None:
    driver.execute_script("goStationPop();")
    time.sleep(2)
    for station in TARGET_STATIONS:
        xpath = f"//tr[td[contains(text(), '{station}')]]//input[@type='checkbox']"
        checkbox = driver.find_element(By.XPATH, xpath)
        driver.execute_script("arguments[0].click();", checkbox)
        time.sleep(0.3)

    try:
        confirm_button = driver.find_element(By.XPATH, "//a[contains(@class, 'bgNavy') and contains(text(), '확인')]")
        driver.execute_script("arguments[0].click();", confirm_button)
    except Exception:
        driver.execute_script("goSelectStation();")
    time.sleep(2)


def download_quality_excels(start_year: int, end_year: int, headless: bool = False) -> list[Path]:
    driver = make_driver(DOWNLOAD_DIR, headless=headless)
    try:
        driver.get(ALGAE_URL)
        time.sleep(3)
        select_stations(driver)

        for start_yr, start_mo, end_yr, end_mo in two_year_ranges(start_year, end_year):
            for attempt in range(1, 4):
                try:
                    print(f"[water_quality] download {start_yr}.{start_mo} ~ {end_yr}.{end_mo} (try {attempt})")
                    Select(driver.find_element(By.ID, "s_year")).select_by_value(start_yr)
                    Select(driver.find_element(By.ID, "s_month")).select_by_value(start_mo)
                    Select(driver.find_element(By.ID, "e_year")).select_by_value(end_yr)
                    Select(driver.find_element(By.ID, "e_month")).select_by_value(end_mo)
                    driver.execute_script("goSearch();")
                    time.sleep(5)
                    before_files = list_excel_files(DOWNLOAD_DIR)
                    driver.execute_script("goExcel();")
                    downloaded = wait_for_new_download(DOWNLOAD_DIR, before_files)
                    print(f"[water_quality] downloaded {downloaded.name}")
                    break
                except UnexpectedAlertPresentException as exc:
                    print(f"[water_quality] alert: {exc.alert_text}")
                    try:
                        driver.switch_to.alert.accept()
                    except NoAlertPresentException:
                        pass
                    time.sleep(5)
                except Exception as exc:
                    print(f"[water_quality] failed: {exc}")
                    time.sleep(5)
            else:
                raise RuntimeError(f"Failed to download water-quality data for {start_yr}.{start_mo} ~ {end_yr}.{end_mo}")
    finally:
        driver.quit()

    downloaded_files = sorted(DOWNLOAD_DIR.glob("*.xlsx")) + sorted(DOWNLOAD_DIR.glob("*.xls"))
    expected_count = ((end_year - start_year) // 2) + 1
    if len(downloaded_files) < expected_count:
        raise RuntimeError(
            f"Expected at least {expected_count} water-quality Excel files, but found {len(downloaded_files)} in {DOWNLOAD_DIR}."
        )
    return downloaded_files


def merge_excels(excel_files: list[Path], output_path: Path = OUTPUT_PATH) -> pd.DataFrame:
    if not excel_files:
        raise FileNotFoundError(f"No water-quality Excel files found in {DOWNLOAD_DIR}")

    frames = []
    for file_path in excel_files:
        frames.append(pd.read_excel(file_path, header=None))
        print(f"[water_quality] read {file_path.name}")

    merged = pd.concat(frames, ignore_index=True, sort=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False, header=False, encoding="utf-8-sig")
    print(f"[water_quality] saved {output_path} rows={len(merged)}")
    return merged


def run(start_year: int = 2016, end_year: int = 2025, headless: bool = False) -> Path:
    excel_files = download_quality_excels(start_year, end_year, headless=headless)
    merge_excels(excel_files)
    return OUTPUT_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and merge Daecheong algae monitoring data.")
    parser.add_argument("--start-year", type=int, default=2016)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.start_year, args.end_year, args.headless)


if __name__ == "__main__":
    main()
