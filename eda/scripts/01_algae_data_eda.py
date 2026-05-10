from __future__ import annotations

"""ALGAE_DATA 탐색적 데이터 분석(EDA) 산출물을 생성한다.

이 스크립트는 모델 학습을 직접 수행하지 않는다. 대신 병합 데이터의 구조,
target 분포, 위치별 수질 차이, station별 기상 차이, 주요 feature 상관관계를
그림과 표로 저장해 모델 결과를 해석할 배경 자료를 만든다.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "src/data/team-raw/ALGAE_DATA.csv"
FIG_DIR = ROOT / "eda/figures"
TABLE_DIR = ROOT / "eda/tables"

STATION_LABELS = {
    604: "옥천(604)",
    643: "세천(643)",
    648: "장동(648)",
    888: "청남대(888)",
}

LOC_LABELS = {
    0: "문의(0)",
    1: "추동(1)",
    2: "하남(2)",
}


def setup_style() -> None:
    """한글 라벨이 깨지지 않도록 matplotlib/seaborn 기본 스타일을 설정한다."""

    plt.rcParams["font.family"] = "Noto Sans CJK KR"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 120
    sns.set_theme(
        style="whitegrid",
        context="notebook",
        rc={
            "font.family": "Noto Sans CJK KR",
            "axes.titlesize": 15,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.titlesize": 18,
        },
    )


def savefig(path: Path) -> None:
    """그림 저장 규칙을 한 곳에 모은다."""

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def load_data() -> pd.DataFrame:
    """ALGAE_DATA를 읽고 EDA용 파생 컬럼을 추가한다."""

    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["log_cyano_cells"] = np.log10(df["cyano_cells"] + 1)
    df["target_alert_next"] = (df["next_log_cells"] >= np.log10(1000 + 1)).astype(int)
    df["station_label"] = df["station"].map(STATION_LABELS).fillna(df["station"].astype(str))
    df["loc_label"] = df["loc_encoded"].map(LOC_LABELS).fillna(df["loc_encoded"].astype(str))
    return df


def make_summary_tables(df: pd.DataFrame, raw_column_count: int) -> None:
    """데이터 구조 요약표와 수치형 feature 요약 통계를 저장한다."""

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    numeric_summary = df.select_dtypes(include="number").describe().T
    numeric_summary["zero_pct"] = (df.select_dtypes(include="number").eq(0).mean() * 100)
    numeric_summary.to_csv(TABLE_DIR / "01_numeric_feature_summary.csv")

    structure = pd.DataFrame(
        {
            "metric": [
                "rows",
                "columns",
                "unique_dates",
                "date_min",
                "date_max",
                "missing_total",
                "stations",
                "locations",
            ],
            "value": [
                len(df),
                raw_column_count,
                df["date"].nunique(),
                df["date"].min().date().isoformat(),
                df["date"].max().date().isoformat(),
                int(df.isna().sum().sum()),
                ", ".join(STATION_LABELS.get(x, str(x)) for x in sorted(df["station"].unique())),
                ", ".join(LOC_LABELS.get(x, str(x)) for x in sorted(df["loc_encoded"].unique())),
            ],
        }
    )
    structure.to_csv(TABLE_DIR / "00_dataset_structure.csv", index=False)


def plot_dataset_structure(df: pd.DataFrame) -> None:
    """date, loc, station 확장 구조가 실제로 어떻게 분포하는지 확인한다."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("ALGAE_DATA 구조 확인", fontweight="bold")

    by_year = df.drop_duplicates(["date", "loc_encoded"]).groupby("year").size()
    sns.barplot(x=by_year.index, y=by_year.values, ax=axes[0, 0], color="#2f6f9f")
    axes[0, 0].set_title("연도별 조사 지점 행 수")
    axes[0, 0].set_xlabel("연도")
    axes[0, 0].set_ylabel("date x loc 행 수")

    station_counts = df["station_label"].value_counts().reindex([STATION_LABELS[x] for x in sorted(STATION_LABELS)])
    sns.barplot(x=station_counts.index, y=station_counts.values, ax=axes[0, 1], color="#518b58")
    axes[0, 1].set_title("기상 station별 행 수")
    axes[0, 1].set_xlabel("기상 관측소")
    axes[0, 1].set_ylabel("행 수")
    axes[0, 1].tick_params(axis="x", rotation=15)

    loc_counts = df.drop_duplicates(["date", "loc_encoded"])["loc_label"].value_counts().reindex(
        [LOC_LABELS[x] for x in sorted(LOC_LABELS)]
    )
    sns.barplot(x=loc_counts.index, y=loc_counts.values, ax=axes[1, 0], color="#bf6f45")
    axes[1, 0].set_title("조사 위치별 행 수")
    axes[1, 0].set_xlabel("조사 위치")
    axes[1, 0].set_ylabel("date x loc 행 수")

    rows_per_date = df.groupby("date").size()
    sns.histplot(rows_per_date, discrete=True, ax=axes[1, 1], color="#7666a8")
    axes[1, 1].set_title("조사일별 병합 행 수")
    axes[1, 1].set_xlabel("한 조사일의 행 수")
    axes[1, 1].set_ylabel("조사일 수")

    savefig(FIG_DIR / "00_dataset_structure.png")


def plot_target_distribution(df: pd.DataFrame) -> None:
    """현재/다음 세포수와 경보 target의 분포를 확인한다."""

    base = df.drop_duplicates(["date", "loc_encoded"]).copy()
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Target 및 세포수 분포", fontweight="bold")

    sns.histplot(base["cyano_cells"], bins=50, ax=axes[0, 0], color="#5d87a1")
    axes[0, 0].set_title("유해남조류 세포수 원 단위")
    axes[0, 0].set_xlabel("cyano_cells")

    sns.histplot(base["log_target"], bins=40, ax=axes[0, 1], color="#5a9b72")
    axes[0, 1].set_title("현재 세포수 로그값")
    axes[0, 1].set_xlabel("log_target")

    sns.histplot(base["next_log_cells"], bins=40, ax=axes[1, 0], color="#b98054")
    axes[1, 0].set_title("다음 조사 시점 세포수 로그값")
    axes[1, 0].set_xlabel("next_log_cells")

    alert_counts = base["target_alert_next"].value_counts().sort_index()
    alert_plot = pd.DataFrame({"target_alert_next": alert_counts.index.astype(str), "count": alert_counts.values})
    sns.barplot(
        data=alert_plot,
        x="target_alert_next",
        y="count",
        hue="target_alert_next",
        ax=axes[1, 1],
        palette=["#7d9aaa", "#c35d4d"],
        legend=False,
    )
    axes[1, 1].set_title("다음 시점 경보 위험 target")
    axes[1, 1].set_xlabel("target_alert_next")
    axes[1, 1].set_ylabel("행 수")

    savefig(FIG_DIR / "01_target_distribution.png")


def plot_target_time_series(df: pd.DataFrame) -> None:
    """조사 위치별로 현재 세포수와 다음 세포수의 시간 흐름을 그린다."""

    base = df.drop_duplicates(["date", "loc_encoded"]).copy()
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    fig.suptitle("조사일별 조류 변화", fontweight="bold")

    sns.lineplot(data=base, x="date", y="log_target", hue="loc_label", ax=axes[0], linewidth=1.4, palette="Set2")
    axes[0].set_title("현재 유해남조류 세포수 로그값")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("log_target")
    axes[0].legend(title="조사 위치", ncol=3)

    sns.lineplot(data=base, x="date", y="next_log_cells", hue="loc_label", ax=axes[1], linewidth=1.4, palette="Set2")
    axes[1].set_title("다음 조사 시점 세포수 로그값")
    axes[1].set_xlabel("조사일")
    axes[1].set_ylabel("next_log_cells")
    axes[1].legend(title="조사 위치", ncol=3)

    savefig(FIG_DIR / "02_target_time_series_by_location.png")


def plot_water_quality(df: pd.DataFrame) -> None:
    """문의/추동/하남 위치별 수질 feature 분포를 boxplot으로 비교한다."""

    base = df.drop_duplicates(["date", "loc_encoded"]).copy()
    cols = ["water_temp", "pH", "DO", "transparency", "turbidity", "Chl_a"]
    labels = {
        "water_temp": "수온",
        "pH": "pH",
        "DO": "DO",
        "transparency": "투명도",
        "turbidity": "탁도",
        "Chl_a": "Chl-a",
    }
    plot_df = base[cols + ["loc_label"]].melt(id_vars="loc_label", var_name="feature", value_name="value")
    plot_df["feature_ko"] = plot_df["feature"].map(labels)

    plt.figure(figsize=(14, 7))
    ax = sns.boxplot(data=plot_df, x="feature_ko", y="value", hue="loc_label", palette="Set2", fliersize=2)
    ax.set_title("수질 피처 분포: 위치별 비교", fontweight="bold")
    ax.set_xlabel("수질 피처")
    ax.set_ylabel("값")
    ax.legend(title="조사 위치", ncol=3, loc="upper right")
    savefig(FIG_DIR / "03_water_quality_boxplot_by_location.png")


def plot_algae_species(df: pd.DataFrame) -> None:
    """유해남조류 4종의 누적 규모와 0값 비율을 비교한다."""

    base = df.drop_duplicates(["date", "loc_encoded"]).copy()
    species = ["Microcystis", "Anabaena", "Oscillatoria", "Aphanizomenon"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("유해남조류 종별 특성", fontweight="bold")

    totals = base[species].sum().sort_values(ascending=False)
    sns.barplot(x=totals.values, y=totals.index, ax=axes[0], color="#397c8b")
    axes[0].set_title("종별 누적 세포수")
    axes[0].set_xlabel("누적 세포수")
    axes[0].set_ylabel("")

    zero_pct = (base[species].eq(0).mean() * 100).sort_values(ascending=False)
    sns.barplot(x=zero_pct.values, y=zero_pct.index, ax=axes[1], color="#b56d54")
    axes[1].set_title("종별 0값 비율")
    axes[1].set_xlabel("0값 비율(%)")
    axes[1].set_ylabel("")

    savefig(FIG_DIR / "04_algae_species_summary.png")


def plot_hydrology(df: pd.DataFrame) -> None:
    """강우, 유입, 방류, 체류시간 계열 feature의 치우친 분포를 로그 스케일로 본다."""

    base = df.drop_duplicates(["date", "loc_encoded"]).copy()
    cols = ["rainfall", "rain_7d_sum_x", "inflow", "outflow", "inflow_7d_sum", "outflow_7d_sum", "residence_proxy", "nutrient_stagnation_index"]
    plot_df = base[cols].copy()
    for col in cols:
        plot_df[col] = np.log10(plot_df[col].clip(lower=0) + 1)
    melted = plot_df.melt(var_name="feature", value_name="log_value")

    plt.figure(figsize=(15, 7))
    ax = sns.boxplot(data=melted, x="feature", y="log_value", color="#6d8fb3", fliersize=2)
    ax.set_title("수문 피처 분포: log10(x + 1) 기준", fontweight="bold")
    ax.set_xlabel("수문 피처")
    ax.set_ylabel("log10(value + 1)")
    ax.tick_params(axis="x", rotation=25)
    savefig(FIG_DIR / "05_hydrology_log_boxplot.png")


def plot_weather_by_station(df: pd.DataFrame) -> None:
    """청남대/장동/세천/옥천 station별 기상 feature 차이를 비교한다."""

    cols = ["avg_temp", "daily_rain", "avg_wind", "sunshine", "solar_rad", "cloud_cover"]
    labels = {
        "avg_temp": "평균기온",
        "daily_rain": "일강수량",
        "avg_wind": "평균풍속",
        "sunshine": "일조",
        "solar_rad": "일사량",
        "cloud_cover": "운량",
    }
    weather = df[["station_label"] + cols].copy()
    weather["daily_rain"] = np.log10(weather["daily_rain"] + 1)
    melted = weather.melt(id_vars="station_label", var_name="feature", value_name="value")
    melted["feature_ko"] = melted["feature"].map(labels)
    melted.loc[melted["feature"].eq("daily_rain"), "feature_ko"] = "일강수량 log"

    plt.figure(figsize=(15, 8))
    ax = sns.boxplot(data=melted, x="feature_ko", y="value", hue="station_label", palette="Set3", fliersize=1.5)
    ax.set_title("station별 주요 기상 피처 분포", fontweight="bold")
    ax.set_xlabel("기상 피처")
    ax.set_ylabel("값")
    ax.legend(title="기상 관측소", ncol=4, loc="upper right")
    savefig(FIG_DIR / "06_weather_boxplot_by_station.png")


def plot_correlation(df: pd.DataFrame) -> None:
    """주요 feature와 다음 세포수 target 사이의 선형 상관 구조를 저장한다."""

    base = df.drop_duplicates(["date", "loc_encoded"]).copy()
    cols = [
        "next_log_cells",
        "log_target",
        "water_temp",
        "DO",
        "transparency",
        "turbidity",
        "Chl_a",
        "cyano_cells",
        "acc_temp_7d",
        "TSI_Chla",
        "TSI_SD",
        "microcystis_ratio",
        "water_level",
        "inflow_7d_sum",
        "outflow_7d_sum",
        "residence_proxy",
        "nutrient_stagnation_index",
        "alert_encoded",
    ]
    corr = base[cols].corr()
    plt.figure(figsize=(13, 11))
    ax = sns.heatmap(corr, cmap="RdBu_r", center=0, vmin=-1, vmax=1, square=True, linewidths=0.3)
    ax.set_title("주요 피처 상관관계", fontweight="bold", pad=14)
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)
    savefig(FIG_DIR / "07_feature_correlation_heatmap.png")

    target_corr = corr["next_log_cells"].drop("next_log_cells").sort_values(key=lambda s: s.abs(), ascending=False)
    target_corr.to_csv(TABLE_DIR / "02_corr_with_next_log_cells.csv", header=["corr_with_next_log_cells"])


def write_report_index() -> None:
    """생성된 EDA 산출물 목록을 Markdown 색인으로 저장한다."""

    report = ROOT / "eda/EDA_INDEX.md"
    report.write_text(
        "\n".join(
            [
                "# ALGAE_DATA EDA 산출물",
                "",
                "## 생성된 시각화",
                "",
                "| 파일 | 내용 |",
                "| --- | --- |",
                "| `figures/00_dataset_structure.png` | 데이터 구조, station/위치/조사일 행 수 |",
                "| `figures/01_target_distribution.png` | target 및 세포수 분포 |",
                "| `figures/02_target_time_series_by_location.png` | 위치별 조류 로그값 시간 변화 |",
                "| `figures/03_water_quality_boxplot_by_location.png` | 위치별 수질 피처 분포 |",
                "| `figures/04_algae_species_summary.png` | 종별 누적 세포수와 0값 비율 |",
                "| `figures/05_hydrology_log_boxplot.png` | 수문 피처 log 분포 |",
                "| `figures/06_weather_boxplot_by_station.png` | station별 기상 피처 분포 |",
                "| `figures/07_feature_correlation_heatmap.png` | 주요 피처 상관관계 |",
                "",
                "## 생성된 표",
                "",
                "| 파일 | 내용 |",
                "| --- | --- |",
                "| `tables/00_dataset_structure.csv` | 데이터 구조 요약 |",
                "| `tables/01_numeric_feature_summary.csv` | 수치형 피처 요약 통계 |",
                "| `tables/02_corr_with_next_log_cells.csv` | 다음 세포수 로그값과의 상관계수 |",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    """EDA 전체 실행 진입점."""

    setup_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    make_summary_tables(df, raw_column_count=57)
    plot_dataset_structure(df)
    plot_target_distribution(df)
    plot_target_time_series(df)
    plot_water_quality(df)
    plot_algae_species(df)
    plot_hydrology(df)
    plot_weather_by_station(df)
    plot_correlation(df)
    write_report_index()

    for path in sorted(FIG_DIR.glob("*.png")):
        print(path.relative_to(ROOT))
    for path in sorted(TABLE_DIR.glob("*.csv")):
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
