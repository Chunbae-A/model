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
DOWNLOAD_DIR = ROOT / "water_data" / "dam"
OUTPUT_PATH = ROOT / "data" / "대청수문_10년치_통합데이터.csv"

DAECHUNG_DAM_URL = "https://www.water.or.kr/kor/menu/sub.do?menuId=13_91_93"


def yearly_ranges(start_year: int, end_year: int) -> list[tuple[str, str]]:
    return [(f"{year}-01-01", f"{year}-12-31") for year in range(start_year, end_year + 1)]


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


def prepare_page(driver: webdriver.Chrome) -> None:
    driver.get(DAECHUNG_DAM_URL)
    time.sleep(4)
    Select(driver.find_element(By.ID, "dam_select0")).select_by_value("1")
    time.sleep(1)
    Select(driver.find_element(By.ID, "dam_select1")).select_by_value("3008110")
    time.sleep(1)
    driver.execute_script("fnObj.linkTab('hydr');")
    time.sleep(2)
    Select(driver.find_element(By.ID, "hydr_param1")).select_by_value("D")
    time.sleep(1)


def download_dam_excels(start_year: int, end_year: int, headless: bool = False) -> list[Path]:
    driver = make_driver(DOWNLOAD_DIR, headless=headless)
    try:
        prepare_page(driver)
        for start_date, end_date in yearly_ranges(start_year, end_year):
            for attempt in range(1, 4):
                try:
                    print(f"[water_gate] download {start_date} ~ {end_date} (try {attempt})")
                    start_input = driver.find_element(By.ID, "hydr_startDate")
                    end_input = driver.find_element(By.ID, "hydr_endDate")
                    driver.execute_script("arguments[0].value = arguments[1];", start_input, start_date)
                    driver.execute_script("arguments[0].value = arguments[1];", end_input, end_date)
                    driver.execute_script("fnObj.linkSearch('hydr');")
                    time.sleep(6)
                    before_files = list_excel_files(DOWNLOAD_DIR)
                    driver.execute_script("fnObj.hydrExcelDown();")
                    downloaded = wait_for_new_download(DOWNLOAD_DIR, before_files)
                    print(f"[water_gate] downloaded {downloaded.name}")
                    break
                except UnexpectedAlertPresentException as exc:
                    print(f"[water_gate] alert: {exc.alert_text}")
                    try:
                        driver.switch_to.alert.accept()
                    except NoAlertPresentException:
                        pass
                    time.sleep(5)
                except Exception as exc:
                    print(f"[water_gate] failed: {exc}")
                    time.sleep(5)
            else:
                raise RuntimeError(f"Failed to download dam data for {start_date} ~ {end_date}")
    finally:
        driver.quit()

    downloaded_files = sorted(DOWNLOAD_DIR.glob("*.xlsx")) + sorted(DOWNLOAD_DIR.glob("*.xls"))
    expected_count = end_year - start_year + 1
    if len(downloaded_files) < expected_count:
        raise RuntimeError(
            f"Expected at least {expected_count} dam Excel files, but found {len(downloaded_files)} in {DOWNLOAD_DIR}."
        )
    return downloaded_files


def merge_excels(excel_files: list[Path], output_path: Path = OUTPUT_PATH) -> pd.DataFrame:
    if not excel_files:
        raise FileNotFoundError(f"No dam Excel files found in {DOWNLOAD_DIR}")

    frames = []
    for file_path in excel_files:
        frames.append(pd.read_excel(file_path, header=None))
        print(f"[water_gate] read {file_path.name}")

    merged = pd.concat(frames, ignore_index=True, sort=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False, header=False, encoding="utf-8-sig")
    print(f"[water_gate] saved {output_path} rows={len(merged)}")
    return merged


def run(start_year: int = 2016, end_year: int = 2025, headless: bool = False) -> Path:
    excel_files = download_dam_excels(start_year, end_year, headless=headless)
    merge_excels(excel_files)
    return OUTPUT_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and merge Daecheong dam operation data.")
    parser.add_argument("--start-year", type=int, default=2016)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.start_year, args.end_year, args.headless)


if __name__ == "__main__":
    main()
