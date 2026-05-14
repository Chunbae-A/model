import os
from datetime import datetime
import time
import requests
import pandas as pd

STATION_COORDS = {
    '133': {'lat': 36.351, 'lon': 127.385},  # Daejeon approximate
    '226': {'lat': 36.487, 'lon': 127.731},  # Boeun approx
    '131': {'lat': 36.642, 'lon': 127.489},  # Cheongju approx
}


def _load_from_cache(stn, sd, ed):
    fp = os.path.join(os.getcwd(), 'data', f'weather_{stn}_10y.csv')
    if not os.path.exists(fp):
        return None
    try:
        df = pd.read_csv(fp, parse_dates=['date'])
        df = df[(df['date'] >= pd.to_datetime(sd)) & (df['date'] <= pd.to_datetime(ed))]
        if df.empty:
            return None
        # Ensure TM column as YYYYMMDD
        df['TM'] = df['date'].dt.strftime('%Y%m%d')
        return df
    except Exception:
        return None


def _fetch_open_meteo(lat, lon, start_date, end_date):
    base = 'https://archive-api.open-meteo.com/v1/archive'
    params = {
        'latitude': lat,
        'longitude': lon,
        'start_date': start_date,
        'end_date': end_date,
        'daily': 'temperature_2m_mean,temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max',
        'timezone': 'Asia/Seoul'
    }
    r = requests.get(base, params=params, timeout=60)
    r.raise_for_status()
    j = r.json()
    if 'daily' not in j:
        return pd.DataFrame()
    d = j['daily']
    df = pd.DataFrame({
        'date': pd.to_datetime(d['time']),
        'avg_temp': d.get('temperature_2m_mean'),
        'max_temp': d.get('temperature_2m_max'),
        'min_temp': d.get('temperature_2m_min'),
        'daily_rain': d.get('precipitation_sum'),
        'max_wind_gust': d.get('windspeed_10m_max'),
    })
    df['TM'] = df['date'].dt.strftime('%Y%m%d')
    return df


def fetch_kma_range(kma_start: str, kma_end: str, stn: str) -> pd.DataFrame:
    """
    kma_start/kma_end: 'YYYYMMDD' strings
    stn: station code as string
    Returns DataFrame with at least columns ['TM','date'] and weather cols where available.
    """
    sd = datetime.strptime(kma_start, '%Y%m%d').date()
    ed = datetime.strptime(kma_end, '%Y%m%d').date()

    # Try cache first
    cached = _load_from_cache(stn, sd, ed)
    if cached is not None:
        return cached

    # Fallback to Open-Meteo (best-effort)
    coord = STATION_COORDS.get(str(stn))
    if not coord:
        return pd.DataFrame()

    try:
        df = _fetch_open_meteo(coord['lat'], coord['lon'], sd.isoformat(), ed.isoformat())
        # attach station
        if not df.empty:
            df['station'] = str(stn)
        return df
    except Exception:
        # If external fetch fails, return empty DataFrame
        return pd.DataFrame()
