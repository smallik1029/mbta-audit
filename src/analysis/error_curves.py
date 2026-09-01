"""how prediction error shrinks as a train gets closer"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src import db
from src.model.train import VALIDATED_ROUTES

REPORTS_DIR_NAME = "reports"
HEADLINE_TARGETS_MIN = [15, 10, 5, 2]
HEADLINE_TOLERANCE_MIN = 1.0
CURVE_MAX_MIN = 30


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
    df["lead_min"] = df["lead_time_sec"] / 60.0
    df["abs_error_sec"] = df["error_sec"].abs()
    return df


def headline_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in HEADLINE_TARGETS_MIN:
        lo = (target - HEADLINE_TOLERANCE_MIN) * 60
        hi = (target + HEADLINE_TOLERANCE_MIN) * 60
        sub = df[(df["lead_time_sec"] >= lo) & (df["lead_time_sec"] < hi)]
        rows.append({
            "lead_time_min": target,
            "n": len(sub),
            "median_abs_error_sec": sub["abs_error_sec"].median() if len(sub) else None,
        })
    return pd.DataFrame(rows)


def full_curve(df: pd.DataFrame) -> pd.DataFrame:
    curve = df[(df["lead_min"] >= 0) & (df["lead_min"] <= CURVE_MAX_MIN)].copy()
    curve["lead_min_bucket"] = curve["lead_min"].round().astype(int)
    grouped = (
        curve.groupby("lead_min_bucket")["abs_error_sec"]
        .agg(median="median", n="count")
        .reset_index()
    )
    return grouped[grouped["n"] >= 20]


def plot_curve(curve: pd.DataFrame, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(curve["lead_min_bucket"], curve["median"] / 60.0, color="#DA291C", linewidth=2)
    ax.set_xlabel("Minutes before arrival (prediction lead time)")
    ax.set_ylabel("Median absolute error (minutes)")
    ax.set_title("MBTA Red/Orange Line prediction error vs. lead time")
    ax.invert_xaxis()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    df = load_outcomes()
    print(f"{len(df):,} matched prediction/outcome rows loaded")

    headline = headline_table(df)
    print("\nHeadline: median absolute error by lead time")
    print(headline.to_string(index=False))

    curve = full_curve(df)
    reports_dir = db.config.ROOT / REPORTS_DIR_NAME
    reports_dir.mkdir(exist_ok=True)
    out_path = reports_dir / "error_curve.png"
    plot_curve(curve, str(out_path))
    print(f"\nSaved error curve chart -> {out_path}")


if __name__ == "__main__":
    main()
