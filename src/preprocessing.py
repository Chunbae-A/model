import pandas as pd

def apply_feature_engineering(df: pd.DataFrame, date_column: str = "date", site_column: str = "station") -> pd.DataFrame:
    """
    대청호 조류 예측을 위한 맞춤형 파생 변수(Feature Engineering)를 생성합니다.
    시계열 데이터이므로 반드시 지점별, 날짜순으로 정렬한 뒤 계산해야 합니다.
    """
    print("🧪 [Feature Engineering] 대청호 맞춤형 파생 변수 생성을 시작합니다...")
    df = df.copy()

    # 1. 안전한 계산을 위해 날짜순 정렬 (지점이 섞이거나 날짜가 뒤집히는 것 방지)
    df[date_column] = pd.to_datetime(df[date_column])
    df = df.sort_values(by=[site_column, date_column]).reset_index(drop=True)

    # =================================================================
    # 💡 1. 델타(Delta) 피처: "어제보다 오늘 얼마나 급격히 변했는가?"
    # =================================================================
    # 수온 급상승 여부
    df["water_temp_diff_1d"] = df.groupby(site_column)["water_temp"].diff()
    # 흙탕물(강수) 급유입 여부 (회남 지점에서 특히 유용)
    if "inflow" in df.columns:
        df["inflow_diff_1d"] = df.groupby(site_column)["inflow"].diff()
    # 수위 급변동 여부
    if "water_level" in df.columns:
        df["water_level_diff_1d"] = df.groupby(site_column)["water_level"].diff()

    # =================================================================
    # 💡 2. 상호작용(Interaction) 피처: "최악의 조건이 만났을 때"
    # =================================================================
    # 녹조 폭발 조건 = 뜨거운 물(water_temp) + 고인 물(residence_proxy 또는 storage_rate)
    if "residence_proxy" in df.columns:
        df["heat_stagnation_index"] = df["water_temp"] * df["residence_proxy"]
    elif "storage_rate" in df.columns:
        df["heat_stagnation_index"] = df["water_temp"] * df["storage_rate"]

    # =================================================================
    # 💡 3. 시계열 지연(Lag) 피처: "어제 비가 많이 왔다면?"
    # =================================================================
    # 당일 강수량보다 1~2일 전 강수량이 영양염류 유입에 더 큰 영향을 줌
    if "daily_rain" in df.columns:
        df["rain_lag_1d"] = df.groupby(site_column)["daily_rain"].shift(1)
        df["rain_lag_2d"] = df.groupby(site_column)["daily_rain"].shift(2)

    # 4. 결측치(NaN) 처리
    # diff()나 shift()를 쓰면 첫째 날이나 둘째 날은 과거 데이터가 없어 NaN이 됩니다.
    # 모델 에러를 막기 위해 0으로 채워줍니다.
    new_features = [
        "water_temp_diff_1d", "inflow_diff_1d", "water_level_diff_1d", 
        "rain_lag_1d", "rain_lag_2d"
    ]
    for col in new_features:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    print("✅ [Feature Engineering] 완료! (추가된 피처: water_temp_diff_1d, heat_stagnation_index 등)")
    
    return df