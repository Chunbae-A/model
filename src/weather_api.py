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

# Ensure repo root is on path so we can import fetch_daejeon_10y
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
    from fetch_10y import fetch_kma_range
except Exception:
    def fetch_kma_range(*a, **k):
        raise RuntimeError('fetch_kma_range not available; ensure fetch_10y.py exists')


ASOS_STATIONS = {
    "133": "daejeon",
    "226": "boeun",
    "131": "cheongju",
}

# AWS <-> ASOS mapping for merging water data
AWS_TO_ASOS = {
    "888": "131",  # 청남대 -> 청주
    "648": "133",  # 장동 -> 대전
    "643": "133",  # 세천 -> 대전
    "604": "226",  # 옥천 -> 보은
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
    # common column normalization
    # Some KMA outputs may have column names like 'avg_temp' or 'TAVG' etc. Keep flexible.
    # Ensure numeric columns exist or create placeholders
    if 'avg_temp' not in df.columns and 'TAVG' in df.columns:
        df['avg_temp'] = pd.to_numeric(df['TAVG'], errors='coerce')
    for col in ['avg_temp', 'min_temp', 'max_temp', 'daily_rain', 'avg_wind', 'max_wind_gust']:
        if col not in df.columns:
            df[col] = np.nan
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Ensure date
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['station', 'date']).reset_index(drop=True)
    gb = df.groupby('station')

    df['air_temp_7d_mean'] = gb['avg_temp'].transform(lambda x: x.rolling(7, min_periods=1).mean())
    df['rain_7d_sum'] = gb['daily_rain'].transform(lambda x: x.rolling(7, min_periods=1).sum())
    df['wind_7d_mean'] = gb['avg_wind'].transform(lambda x: x.rolling(7, min_periods=1).mean())

    # seasonal sin/cos
    df['doy'] = df['date'].dt.dayofyear
    df['sin_season'] = np.sin(2 * np.pi * df['doy'] / 365.25)
    df['cos_season'] = np.cos(2 * np.pi * df['doy'] / 365.25)

    return df


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
        water_csv = os.path.join(str(ROOT), 'data', 'processed', 'model_input', 'algae_model_input.csv')
        merged = merge_with_water(feat, water_csv, aws_matches) if aws_matches else feat

        out_csv = os.path.join(OUT_DIR, f"weather_{stn}_10y.csv")
        merged.to_csv(out_csv, index=False)
        print(f"Saved {out_csv} rows={len(merged)}")
        all_out.append(out_csv)

    print("생성 파일:")
    for p in all_out:
        print(" -", p)

    # Concatenate and merge AWS data if key present
    dfs = [pd.read_csv(p, parse_dates=['date']) for p in all_out]
    combined = pd.concat(dfs, ignore_index=True, sort=False)

    aws_groups = {}
    for aws, asos in AWS_TO_ASOS.items():
        aws_groups.setdefault(asos, []).append(aws)

    AUTH_KEY = os.environ.get('KMA_SERVICE_KEY')
    aws_dfs = []
    if AUTH_KEY:
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
    print('최종 파일 저장:', out_final)


if __name__ == '__main__':
    run()
