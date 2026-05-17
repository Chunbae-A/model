import os
from datetime import date, datetime, timedelta
import time
import traceback
import pandas as pd
import numpy as np
import requests
import xml.etree.ElementTree as ET

import sys
from pathlib import Path
from typing import Optional

# load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Ensure repo root is on path so this works both as a script and as src.weather_api.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
    from src.fetch_10y import fetch_kma_range
except Exception:
    try:
        from fetch_10y import fetch_kma_range
    except Exception:
        def fetch_kma_range(*a, **k):
            raise RuntimeError('fetch_kma_range not available; ensure fetch_10y.py exists')


ASOS_STATIONS = {
    "133": "daejeon",
}

# AWS <-> ASOS mapping for merging water data
AWS_TO_ASOS = {
    "888": "133",  # 청남대 -> 대전
    "648": "133",  # 장동 -> 대전
    "643": "133",  # 세천 -> 대전
    "604": "133",  # 옥천 -> 대전
}

WEATHER_OUTPUT_COLUMNS = [
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
]

ROBUST_SCALE_COLUMNS = [
    "daily_rain",
    "rain_3d_sum",
    "rain_7d_sum",
    "rain_14d_sum",
]

MINMAX_SCALE_COLUMNS = [
    "avg_temp",
    "min_temp",
    "max_temp",
    "max_wind_gust",
    "avg_wind",
    "air_temp_7d_mean",
    "wind_7d_mean",
    "sunshine",
    "solar_rad",
    "cloud_cover",
    "sunshine_7d_sum",
    "solar_7d_sum",
    "cloud_7d_mean",
]

WEATHER_COLUMN_CANDIDATES = {
    "avg_temp": ["avg_temp", "avg_temp_x", "avg_temp_y", "평균기온(°C)", "평균기온", "TAVG"],
    "min_temp": ["min_temp", "min_temp_x", "min_temp_y", "최저기온(°C)", "최저기온", "TMIN"],
    "max_temp": ["max_temp", "max_temp_x", "max_temp_y", "최고기온(°C)", "최고기온", "TMAX"],
    "daily_rain": ["daily_rain", "daily_rain_x", "daily_rain_y", "일강수량(mm)", "일강수량", "강수량", "RN"],
    "max_wind_gust": ["max_wind_gust", "max_wind_gust_x", "max_wind_gust_y", "최대 순간 풍속(m/s)", "최대풍속", "WS_MAX"],
    "avg_wind": ["avg_wind", "avg_wind_x", "avg_wind_y", "평균 풍속(m/s)", "평균풍속", "WS_AVG"],
    "sunshine": ["sunshine", "sunshine_x", "sunshine_y", "합계 일조 시간(hr)", "합계 일조시간(hr)", "합계 일조 시간", "일조시간"],
    "solar_rad": ["solar_rad", "solar_rad_x", "solar_rad_y", "합계 일사(MJ/m2)", "합계 일사량(MJ/m2)", "합계 일사", "일사량"],
    "cloud_cover": ["cloud_cover", "cloud_cover_x", "cloud_cover_y", "평균 전운량(1/10)", "평균 전운량", "전운량"],
}

OUT_DIR = os.path.join(str(ROOT), 'data')
os.makedirs(OUT_DIR, exist_ok=True)


def build_weather_for_asos(stn: str, start_dt: date, end_dt: date) -> pd.DataFrame:
    df_list = []
    current = start_dt
    while current <= end_dt:
        chunk_end = date(current.year, 12, 31)
        if chunk_end > end_dt:
            chunk_end = end_dt
        kma_sd = current.strftime('%Y%m%d')
        kma_ed = chunk_end.strftime('%Y%m%d')
        try:
            part = fetch_kma_range(kma_sd, kma_ed, stn)
            if isinstance(part, pd.DataFrame) and not part.empty:
                df_list.append(part)
        except Exception:
            traceback.print_exc()
        current = chunk_end + timedelta(days=1)
        time.sleep(0.2)

    if not df_list:
        return pd.DataFrame()
    df = pd.concat(df_list, ignore_index=True, sort=False)
    # Ensure date column exists
    if 'date' not in df.columns:
        if 'TM' in df.columns:
            try:
                df['date'] = pd.to_datetime(df['TM'].astype(str), format='%Y%m%d')
            except Exception:
                df['date'] = pd.to_datetime(df['TM'], errors='coerce')
        else:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['station'] = stn
    return df


def _parse_aws_item(item: ET.Element) -> dict:
    d = {}
    for ch in item:
        tag = ch.tag.lower()
        text = ch.text.strip() if ch.text else None
        d[tag] = text
    return d


def fetch_aws_month(station: str, year: int, month: int, auth_key: str) -> pd.DataFrame:
    base = "https://apihub.kma.go.kr/api/typ02/openApi/AwsMtlyInfoService/getDailyAwsData"
    params = {
        'pageNo': 1,
        'numOfRows': 1000,
        'dataType': 'XML',
        'year': year,
        'month': f"{month:02d}",
        'station': station,
        'authKey': auth_key,
    }
    # retry/backoff for transient failures
    max_attempts = 5
    backoff = 1.0
    for attempt in range(1, max_attempts + 1):
        try:
            r = requests.get(base, params=params, timeout=60)
            r.raise_for_status()
            root = ET.fromstring(r.content)
            break
        except requests.exceptions.HTTPError as e:
            status = getattr(e.response, 'status_code', None)
            body = None
            try:
                body = e.response.text
            except Exception:
                pass
            # retry on 5xx or 429, else abort
            if status and (500 <= status < 600 or status == 429):
                time.sleep(backoff)
                backoff *= 2
                continue
            raise
        except Exception:
            if attempt == max_attempts:
                raise
            time.sleep(backoff)
            backoff *= 2
            continue
    items = root.findall('.//item')
    rows = []
    for it in items:
        row = _parse_aws_item(it)
        # find date-like key
        date_val = None
        for k in row:
            if 'date' in k or 'tm' in k or 'day' in k:
                date_val = row[k]
                break
        if date_val is None:
            continue
        # normalize date
        try:
            if len(date_val) == 8 and date_val.isdigit():
                dt = datetime.strptime(date_val, '%Y%m%d').date()
            else:
                dt = pd.to_datetime(date_val).date()
        except Exception:
            continue
        # find relevant numeric fields (many possible tag names, handle Korean/English)
        vals = {
            'avg_temp': None,
            'min_temp': None,
            'max_temp': None,
            'daily_rain': None,
            'avg_wind': None,
            'max_wind_gust': None,
            'sunshine': None,
            'solar_rad': None,
            'cloud_cover': None,
        }
        for k, v in row.items():
            if v is None:
                continue
            key = k.lower()
            try:
                num = float(v)
            except Exception:
                num = None
            # temperature
            if any(x in key for x in ['avgtemp', '평균기온', 'tavg', 'tmean', 'temperature_2m_mean']):
                vals['avg_temp'] = num
            if any(x in key for x in ['mintemp', '최저기온', 'tmin', 'temperature_2m_min']):
                vals['min_temp'] = num
            if any(x in key for x in ['maxtemp', '최고기온', 'tmax', 'temperature_2m_max']):
                vals['max_temp'] = num
            # rain
            if any(x in key for x in ['precip', '강수', 'rain', 'daily_rain']):
                vals['daily_rain'] = num
            # wind
            if any(x in key for x in ['windspeed', '평균 풍속', '평균풍속', 'avg_wind', 'wind_avg']):
                vals['avg_wind'] = num
            if any(x in key for x in ['gust', 'max_wind', '최대', '최대풍속', 'wind_gust']):
                vals['max_wind_gust'] = num
            # sunshine/solar/cloud
            if 'sun' in key and vals.get('sunshine') is None:
                vals['sunshine'] = num
            if any(x in key for x in ['solar', 'rad']) and vals.get('solar_rad') is None:
                vals['solar_rad'] = num
            if any(x in key for x in ['cloud', 'cld', '전운'] ) and vals.get('cloud_cover') is None:
                vals['cloud_cover'] = num

        row_out = {'date': pd.to_datetime(dt)}
        row_out.update(vals)
        rows.append(row_out)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def fetch_aws_range(start_dt: date, end_dt: date, station: str, auth_key: str) -> pd.DataFrame:
    cur = date(start_dt.year, start_dt.month, 1)
    end_month = date(end_dt.year, end_dt.month, 1)
    parts = []
    while cur <= end_month:
        try:
            dfm = fetch_aws_month(station, cur.year, cur.month, auth_key)
            if not dfm.empty:
                parts.append(dfm)
        except Exception:
            traceback.print_exc()
        # next month
        if cur.month == 12:
            cur = date(cur.year + 1, 1, 1)
        else:
            cur = date(cur.year, cur.month + 1, 1)
        time.sleep(0.2)
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True, sort=False)
    df['date'] = pd.to_datetime(df['date'])
    return df


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    # Expect df contains at least 'date', 'station' and some numeric weather cols
    df = df.copy()
    normalize_weather_columns(df)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['station', 'date']).reset_index(drop=True)
    df = add_weather_features(df)
    return df


def normalize_weather_columns(df: pd.DataFrame) -> pd.DataFrame:
    for target, candidates in WEATHER_COLUMN_CANDIDATES.items():
        values = None
        for candidate in candidates:
            if candidate not in df.columns:
                continue
            candidate_values = pd.to_numeric(df[candidate], errors="coerce")
            values = candidate_values if values is None else values.fillna(candidate_values)
        df[target] = values if values is not None else np.nan

    if "station" not in df.columns and "지점" in df.columns:
        df["station"] = df["지점"]
    if "date" not in df.columns:
        if "일시" in df.columns:
            df["date"] = df["일시"]
        elif "TM" in df.columns:
            df["date"] = df["TM"]

    df["station"] = df["station"].astype(str)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["station", "date"]).sort_values(["station", "date"]).reset_index(drop=True)

    gb = df.groupby("station", sort=False)
    df["air_temp_7d_mean"] = gb["avg_temp"].transform(lambda x: x.rolling(window=7, min_periods=1).mean())
    df["hot_days_30c_7d"] = gb["max_temp"].transform(lambda x: (x >= 30).rolling(window=7, min_periods=1).sum())
    df["rain_3d_sum"] = gb["daily_rain"].transform(lambda x: x.rolling(window=3, min_periods=1).sum())
    df["rain_7d_sum"] = gb["daily_rain"].transform(lambda x: x.rolling(window=7, min_periods=1).sum())
    df["rain_14d_sum"] = gb["daily_rain"].transform(lambda x: x.rolling(window=14, min_periods=1).sum())
    df["wind_7d_mean"] = gb["avg_wind"].transform(lambda x: x.rolling(window=7, min_periods=1).mean())
    df["low_wind_days_2ms_7d"] = gb["avg_wind"].transform(lambda x: (x <= 2.0).rolling(window=7, min_periods=1).sum())
    df["sunshine_7d_sum"] = gb["sunshine"].transform(lambda x: x.rolling(window=7, min_periods=1).sum())
    df["solar_7d_sum"] = gb["solar_rad"].transform(lambda x: x.rolling(window=7, min_periods=1).sum())
    df["cloud_7d_mean"] = gb["cloud_cover"].transform(lambda x: x.rolling(window=7, min_periods=1).mean())

    doy = df["date"].dt.dayofyear
    df["sin_season"] = np.sin(2 * np.pi * doy / 365.25)
    df["cos_season"] = np.cos(2 * np.pi * doy / 365.25)
    return df


def complete_weather_calendar(df: pd.DataFrame, start_dt: date | None = None, end_dt: date | None = None) -> pd.DataFrame:
    frames = []
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for station, group in df.groupby("station", sort=False):
        group = group.sort_values("date").drop_duplicates("date", keep="first")
        start = pd.Timestamp(start_dt) if start_dt else group["date"].min()
        end = pd.Timestamp(end_dt) if end_dt else group["date"].max()
        idx = pd.date_range(start, end, freq="D")
        completed = group.set_index("date").reindex(idx).rename_axis("date").reset_index()
        completed["station"] = station
        frames.append(completed)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else df


def impute_weather_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values(["station", "date"]).reset_index(drop=True)
    df["daily_rain"] = pd.to_numeric(df["daily_rain"], errors="coerce").fillna(0)

    spatial_cols = ["sunshine", "solar_rad", "cloud_cover"]
    for col in spatial_cols:
        daily_mean = df.groupby("date")[col].transform("mean")
        df[col] = df[col].fillna(daily_mean)

    interp_cols = [
        "avg_temp",
        "min_temp",
        "max_temp",
        "max_wind_gust",
        "avg_wind",
        "sunshine",
        "solar_rad",
        "cloud_cover",
    ]
    df[interp_cols] = df.groupby("station")[interp_cols].transform(
        lambda g: g.interpolate(method="linear", limit_direction="both").ffill().bfill()
    )
    df[interp_cols] = df[interp_cols].fillna(0)
    return df


def scale_weather_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ROBUST_SCALE_COLUMNS:
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        median = values.median()
        df[col] = 0.0 if pd.isna(iqr) or iqr == 0 else (values - median) / iqr

    for col in MINMAX_SCALE_COLUMNS:
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        min_value = values.min()
        max_value = values.max()
        span = max_value - min_value
        df[col] = 0.0 if pd.isna(span) or span == 0 else (values - min_value) / span
    return df


def save_standard_weather_csv(weather_df: pd.DataFrame) -> str:
    """Save the canonical WEATHER.csv consumed by build_clean_dataset.py."""
    output = weather_df.copy()
    normalize_weather_columns(output)

    frames = []
    for aws_station, asos_station in AWS_TO_ASOS.items():
        part = output[output["station"].eq(str(asos_station))].copy()
        if part.empty:
            continue
        part["station"] = int(aws_station)
        frames.append(part)

    if frames:
        standard = pd.concat(frames, ignore_index=True, sort=False)
    else:
        standard = output[output["station"].isin(AWS_TO_ASOS.keys())].copy()
        if standard.empty:
            raise ValueError("No AWS or mapped ASOS rows were available to build WEATHER.csv")

    for col in WEATHER_OUTPUT_COLUMNS:
        if col not in standard.columns:
            standard[col] = np.nan

    standard = standard[[c for c in standard.columns if c in set(WEATHER_OUTPUT_COLUMNS)]].copy()
    for col in [c for c in standard.columns if c not in {"date", "station"}]:
        standard[col] = pd.to_numeric(standard[col], errors="coerce")

    standard = complete_weather_calendar(standard)
    standard = impute_weather_values(standard)
    standard = add_weather_features(standard)
    standard = scale_weather_values(standard)
    standard = standard[WEATHER_OUTPUT_COLUMNS].copy()
    standard["station"] = standard["station"].astype(int)
    standard["date"] = standard["date"].dt.strftime("%Y-%m-%d")

    out_path = os.path.join(OUT_DIR, "WEATHER.csv")
    standard.to_csv(out_path, index=False, encoding="utf-8-sig")
    print("Saved canonical weather file:", out_path, "rows=", len(standard))
    return out_path


def merge_with_water(weather_df: pd.DataFrame, water_csv: str, aws_matches: list) -> pd.DataFrame:
    if not os.path.exists(water_csv):
        print("Water CSV not found, skipping merge:", water_csv)
        return weather_df
    w = pd.read_csv(water_csv, parse_dates=['date'])
    if 'station' not in w.columns:
        for c in ['지점', 'site', 'station_id']:
            if c in w.columns:
                w = w.rename(columns={c: 'station'})
                break

    if 'station' not in w.columns:
        copies = []
        for aws in aws_matches:
            tmp = w.copy()
            tmp['station'] = aws
            copies.append(tmp)
        w = pd.concat(copies, ignore_index=True, sort=False)

    weather_df = weather_df.copy()
    w = w.copy()
    weather_df['station'] = weather_df['station'].astype(str)
    w['station'] = w['station'].astype(str)
    weather_df['date'] = pd.to_datetime(weather_df['date'])
    w['date'] = pd.to_datetime(w['date'])

    # avoid duplicate column name conflicts: rename overlapping water columns
    overlap = set(weather_df.columns).intersection(set(w.columns)) - {'station', 'date'}
    if overlap:
        rename_map = {c: f"water_{c}" for c in overlap}
        w = w.rename(columns=rename_map)

    merged = pd.merge(weather_df, w, how='left', on=['station', 'date'])
    return merged


def run(start_dt: Optional[date] = None, end_dt: Optional[date] = None):
    today = date.today()
    if end_dt is None:
        end_dt = today
    if start_dt is None:
        start_dt = today - timedelta(days=3650)

    all_out = []
    for stn, name in ASOS_STATIONS.items():
        print(f"== 처리: ASOS {stn} ({name}) ==")
        wdf = build_weather_for_asos(stn, start_dt, end_dt)
        if wdf.empty:
            print(f"ASOS {stn}에서 데이터를 가져오지 못했습니다.")
            continue
        feat = make_features(wdf)

        aws_matches = [aws for aws, asos in AWS_TO_ASOS.items() if asos == stn]
        # Keep weather generation independent. Water-quality and dam data are
        # joined exactly once in preprocess_data.py, where the final schema is enforced.
        merged = feat

        out_csv = os.path.join(OUT_DIR, f"weather_{stn}_10y.csv")
        merged.to_csv(out_csv, index=False)
        print(f"Saved {out_csv} rows={len(merged)}")
        all_out.append(out_csv)

    print("생성 파일:")
    for p in all_out:
        print(" -", p)

    if not all_out:
        raise RuntimeError(
            "No ASOS weather files were created. Check network access, the KMA endpoint, "
            "and the requested date range before rerunning `python src/pipeline.py --fetch weather`."
        )

    # Concatenate and merge AWS data if key present
    dfs = [pd.read_csv(p, parse_dates=['date']) for p in all_out]
    combined = pd.concat(dfs, ignore_index=True, sort=False)

    aws_groups = {}
    for aws, asos in AWS_TO_ASOS.items():
        aws_groups.setdefault(asos, []).append(aws)

    AUTH_KEY = os.environ.get('KMA_SERVICE_KEY')
    fetch_aws = os.environ.get('KMA_FETCH_AWS', '1').lower() not in {'0', 'false', 'no', 'n'}
    aws_dfs = []
    if AUTH_KEY and fetch_aws:
        for asos, aws_list in aws_groups.items():
            parts = []
            for aws in aws_list:
                print(f"Fetching AWS {aws} for ASOS {asos}")
                part = fetch_aws_range(start_dt, end_dt, aws, AUTH_KEY)
                if not part.empty:
                    part['station'] = str(aws)
                    parts.append(part)
            if parts:
                adf = pd.concat(parts, ignore_index=True, sort=False)
                # aggregate across AWS stations mapped to same ASOS: mean for numeric cols
                num_cols = ['avg_temp','min_temp','max_temp','daily_rain','avg_wind','max_wind_gust','sunshine','solar_rad','cloud_cover']
                agg_map = {c: 'mean' for c in num_cols if c in adf.columns}
                adf['asos'] = asos
                grp = adf.groupby(['asos', 'date']).agg(agg_map).reset_index()
                grp = grp.rename(columns={'asos': 'station'})
                aws_dfs.append(grp)
    elif AUTH_KEY:
        print("KMA_SERVICE_KEY is set, but AWS detail fetch is skipped by default. Set KMA_FETCH_AWS=1 to enable it.")

    if aws_dfs:
        aws_comb = pd.concat(aws_dfs, ignore_index=True, sort=False)
        aws_comb['station'] = aws_comb['station'].astype(str)
        combined['station'] = combined['station'].astype(str)
        merged_all = pd.merge(combined, aws_comb, how='left', on=['station', 'date'], suffixes=('', '_aws'))
        for col in ['sunshine', 'solar_rad', 'cloud_cover']:
            if f"{col}_aws" in merged_all.columns:
                merged_all[col] = merged_all.get(col).fillna(merged_all[f"{col}_aws"]) if col in merged_all.columns else merged_all[f"{col}_aws"]
                merged_all.drop(columns=[f"{col}_aws"], inplace=True, errors='ignore')
    else:
        merged_all = combined

    # If we have AWS numeric fields merged, ensure full date coverage per station and handle missing dates
    if aws_dfs:
        # ensure full date index for each station
        full_rows = []
        missing_report = {}
        # aws_comb may be empty if no monthly AWS data was fetched
        if aws_comb is None or aws_comb.empty:
            full_rows = []
        else:
            for st, g in aws_comb.groupby('station'):
                g = g.copy()
                g['date'] = pd.to_datetime(g['date'])
                idx = pd.date_range(start_dt, end_dt, freq='D')
                g2 = g.set_index('date').reindex(idx).reset_index().rename(columns={'index':'date'})
                g2['station'] = str(st)
                # record missing dates
                missing_dates = g2[g2.isnull().all(axis=1) | g2[['avg_temp','min_temp','max_temp','daily_rain','avg_wind']].isnull().all(axis=1)]['date']
                missing_report[st] = list(missing_dates.dt.strftime('%Y-%m-%d'))
                full_rows.append(g2)
        # only concat if we have rows
        if not full_rows:
            aws_full = pd.DataFrame()
        else:
            aws_full = pd.concat(full_rows, ignore_index=True, sort=False)

        # Log missing date counts and create empty rows already done via reindex
        for st, miss in missing_report.items():
            if miss:
                print(f"[AWS station {st}] 누락된 날짜 수: {len(miss)}")
                print(f"누락된 날짜: {miss[:10]}")

        # merge aws_full into merged_all by station/date (station are ASOS codes in merged_all)
        # aws_full station currently contains ASOS mapping values (we set station=asos earlier)
        if not aws_full.empty:
            merged_all = pd.merge(merged_all, aws_full, how='left', on=['station','date'], suffixes=('','_awsfull'))
        else:
            # nothing to merge; keep merged_all as-is
            pass

        # For numeric columns from aws, prefer aws_full values if present
        for col in ['avg_temp','min_temp','max_temp','daily_rain','avg_wind','max_wind_gust','sunshine','solar_rad','cloud_cover']:
            if col in merged_all.columns and f"{col}_awsfull" in merged_all.columns:
                merged_all[col] = merged_all[col].fillna(merged_all[f"{col}_awsfull"]) if col in merged_all.columns else merged_all[f"{col}_awsfull"]
                merged_all.drop(columns=[f"{col}_awsfull"], inplace=True, errors='ignore')

        # Now handle missing values per rules
        # Create missing-date rows already done; now impute
        # 강수량 -> 전부 0
        if 'daily_rain' in merged_all.columns:
            merged_all['daily_rain'] = merged_all['daily_rain'].fillna(0)

        # 나머지(기온, 풍속) -> 전후날로 선형보간 per station
        interp_cols = [c for c in ['avg_temp','min_temp','max_temp','avg_wind','max_wind_gust'] if c in merged_all.columns]
        merged_all = merged_all.sort_values(['station','date']).reset_index(drop=True)
        merged_all[interp_cols] = merged_all.groupby('station')[interp_cols].apply(lambda g: g.interpolate(method='linear', limit_direction='both'))

        # After interpolation, fill any remaining numeric NaNs with forward/backfill per station
        merged_all[interp_cols] = merged_all.groupby('station')[interp_cols].apply(lambda g: g.fillna(method='ffill').fillna(method='bfill'))

        # Derived features for AWS fields
        gb2 = merged_all.groupby('station')
        if 'avg_temp' in merged_all.columns:
            merged_all['air_temp_7d_mean'] = gb2['avg_temp'].transform(lambda x: x.rolling(7, min_periods=1).mean())
        if 'max_temp' in merged_all.columns:
            merged_all['hot_days_30c_7d'] = gb2['max_temp'].transform(lambda x: (x >= 30).rolling(7, min_periods=1).sum())
        if 'daily_rain' in merged_all.columns:
            merged_all['rain_3d_sum'] = gb2['daily_rain'].transform(lambda x: x.rolling(3, min_periods=1).sum())
            merged_all['rain_7d_sum'] = gb2['daily_rain'].transform(lambda x: x.rolling(7, min_periods=1).sum())
            merged_all['rain_14d_sum'] = gb2['daily_rain'].transform(lambda x: x.rolling(14, min_periods=1).sum())
        if 'avg_wind' in merged_all.columns:
            merged_all['wind_7d_mean'] = gb2['avg_wind'].transform(lambda x: x.rolling(7, min_periods=1).mean())
            merged_all['low_wind_days_2ms_7d'] = gb2['avg_wind'].transform(lambda x: (x <= 2.0).rolling(7, min_periods=1).sum())

    # Final feature pass
    merged_all['date'] = pd.to_datetime(merged_all['date'])
    merged_all = merged_all.sort_values(['station', 'date']).reset_index(drop=True)
    gb = merged_all.groupby('station')
    if 'daily_rain' in merged_all.columns:
        merged_all['rain_7d_sum'] = gb['daily_rain'].transform(lambda x: x.rolling(7, min_periods=1).sum())

    out_final = os.path.join(OUT_DIR, 'combined_weather_water_10y.csv')
    merged_all.to_csv(out_final, index=False)
    save_standard_weather_csv(merged_all)
    print('최종 파일 저장:', out_final)


if __name__ == '__main__':
    run()
