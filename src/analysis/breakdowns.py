"""error breakdowns by route, and rush hour vs midday"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src import config, db

LEAD_MIN_FOR_COMPARISON = 10
TOLERANCE_MIN = 1.0

VALIDATED_ROUTES = ("Red", "Orange")


def load_outcomes() -> pd.DataFrame:
    """validated routes only"""
    conn = db.connect()
    placeholders = ",".join("?" * len(VALIDATED_ROUTES))
    df = pd.read_sql_query(
        "SELECT route_id, lead_time_sec, error_sec, hour_local, is_rush_hour "
        f"FROM prediction_outcomes WHERE route_id IN ({placeholders})",
        conn,
        params=VALIDATED_ROUTES,
    )
    df["abs_error_sec"] = df["error_sec"].abs()
    return df


def _at_comparison_horizon(df: pd.DataFrame) -> pd.DataFrame:
    lo = (LEAD_MIN_FOR_COMPARISON - TOLERANCE_MIN) * 60
    hi = (LEAD_MIN_FOR_COMPARISON + TOLERANCE_MIN) * 60
    return df[(df["lead_time_sec"] >= lo) & (df["lead_time_sec"] < hi)]


def by_route(df: pd.DataFrame) -> pd.DataFrame:
    sub = _at_comparison_horizon(df)
    return (
        sub.groupby("route_id")["abs_error_sec"]
        .agg(median_abs_error_sec="median", n="count")
        .reset_index()
        .sort_values("median_abs_error_sec")
    )


def by_rush_hour(df: pd.DataFrame) -> pd.DataFrame:
    sub = _at_comparison_horizon(df)
    out = (
        sub.groupby("is_rush_hour")["abs_error_sec"]
        .agg(median_abs_error_sec="median", n="count")
        .reset_index()
    )
    out["label"] = out["is_rush_hour"].map({0: "midday/off-peak", 1: "rush hour"})
    return out[["label", "median_abs_error_sec", "n"]]


def by_hour_of_day(df: pd.DataFrame) -> pd.DataFrame:
    sub = _at_comparison_horizon(df)
    return (
        sub.groupby("hour_local")["abs_error_sec"]
        .agg(median_abs_error_sec="median", n="count")
        .reset_index()
        .sort_values("hour_local")
    )


def plot_hourly(hourly: pd.DataFrame, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(hourly["hour_local"], hourly["median_abs_error_sec"] / 60.0, color="#ED8B00")
    for h in config.RUSH_HOURS:
        ax.axvspan(h - 0.5, h + 0.5, color="gray", alpha=0.15)
    ax.set_xlabel("Hour of day (America/New_York)")
    ax.set_ylabel(f"Median absolute error at {LEAD_MIN_FOR_COMPARISON}min lead (minutes)")
    ax.set_title("Prediction error by hour of day (shaded = rush hour)")
    ax.set_xticks(range(0, 24, 2))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    df = load_outcomes()
    print(f"{len(df):,} matched prediction/outcome rows loaded")

    print(f"\nBy route (at ~{LEAD_MIN_FOR_COMPARISON}min lead time):")
    route_table = by_route(df)
    print(route_table.to_string(index=False))

    print(f"\nRush hour vs midday (at ~{LEAD_MIN_FOR_COMPARISON}min lead time):")
    rush_table = by_rush_hour(df)
    print(rush_table.to_string(index=False))

    hourly = by_hour_of_day(df)
    reports_dir = config.ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)
    out_path = reports_dir / "error_by_hour.png"
    plot_hourly(hourly, str(out_path))
    print(f"\nSaved hourly breakdown chart -> {out_path}")


if __name__ == "__main__":
    main()
