"""Blue Line correction -- a fully ISOLATED addition, not part of the"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src import config, db
from src.analysis.error_curves import HEADLINE_TARGETS_MIN, HEADLINE_TOLERANCE_MIN
from src.model.features import add_lead_bucket, time_split
from src.model.train import export_lookup_csv, fit_lookup
from src.process.match import build_outcomes

ROUTE = "Blue"
ARTIFACTS_DIR = config.ROOT / "model_artifacts_blue"


def load_outcomes() -> pd.DataFrame:
    conn = db.connect()
    arrivals = pd.read_sql_query(
        "SELECT trip_id, stop_id, route_id, direction_id, actual_arrival "
        "FROM actual_arrivals WHERE route_id = ?",
        conn, params=[ROUTE],
    )
    arrivals["actual_arrival"] = pd.to_datetime(arrivals["actual_arrival"], utc=True, format="ISO8601")

    predictions = pd.read_sql_query(
        "SELECT observed_at, trip_id, stop_id, predicted_arrival, schedule_relationship "
        "FROM prediction_snapshots WHERE route_id = ? AND predicted_arrival IS NOT NULL",
        conn, params=[ROUTE],
    )
    predictions = predictions[~predictions["schedule_relationship"].isin({"CANCELLED", "SKIPPED"})]
    predictions["observed_at"] = pd.to_datetime(predictions["observed_at"], utc=True, format="ISO8601")
    predictions["predicted_arrival"] = pd.to_datetime(
        predictions["predicted_arrival"], utc=True, format="ISO8601"
    )

    outcomes = build_outcomes(arrivals, predictions)
    return add_lead_bucket(outcomes)


def correct(test: pd.DataFrame, lookup: dict) -> pd.Series:
    idx_stop = pd.MultiIndex.from_arrays([test["route_id"], test["stop_id"], test["lead_bucket"]])
    bias = lookup["by_stop"].reindex(idx_stop).to_numpy()
    bucket_bias = lookup["by_bucket"].reindex(test["lead_bucket"]).to_numpy()
    bias = pd.Series(bias).where(~pd.isna(bias), bucket_bias).to_numpy()
    return test["error_sec"] - bias


def summarize(test: pd.DataFrame, corrected_error: pd.Series) -> pd.DataFrame:
    rows = []
    for target in HEADLINE_TARGETS_MIN:
        lo = (target - HEADLINE_TOLERANCE_MIN) * 60
        hi = (target + HEADLINE_TOLERANCE_MIN) * 60
        mask = (test["lead_time_sec"] >= lo) & (test["lead_time_sec"] < hi)
        if mask.sum() < 20:
            continue
        baseline = test.loc[mask, "error_sec"].abs().median()
        corrected = corrected_error.loc[mask].abs().median()
        rows.append({
            "lead_time_min": target, "n": int(mask.sum()),
            "baseline_median_abs_error_sec": round(baseline, 1),
            "corrected_median_abs_error_sec": round(corrected, 1),
            "improvement_pct": round(100 * (1 - corrected / baseline), 1) if baseline else None,
        })
    return pd.DataFrame(rows)


def plot_comparison(table: pd.DataFrame, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = range(len(table))
    width = 0.35
    ax.bar([i - width / 2 for i in x], table["baseline_median_abs_error_sec"],
           width, label="MBTA raw prediction", color="#8a8a8a")
    ax.bar([i + width / 2 for i in x], table["corrected_median_abs_error_sec"],
           width, label="Corrected (Blue Line, isolated)", color="#003DA5")
    for i, row in table.iterrows():
        ax.text(i, max(row["baseline_median_abs_error_sec"], row["corrected_median_abs_error_sec"]) + 2,
                f"-{row['improvement_pct']:.0f}%", ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{m} min out" for m in table["lead_time_min"]])
    ax.set_ylabel("Median absolute error (seconds)")
    ax.set_title("[EXPERIMENTAL] Blue Line: corrected vs raw (held-out test set)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    df = load_outcomes()
    print(f"{len(df):,} Blue Line matched outcomes")
    if len(df) < 500:
        print("Too little data for a meaningful split -- stopping.")
        return

    train, test = time_split(df)
    print(f"train: {len(train):,} rows, test: {len(test):,} rows")

    lookup = fit_lookup(train)

    corrected_error = correct(test, lookup)
    overall_baseline = test["error_sec"].abs().median()
    overall_corrected = corrected_error.abs().median()
    print(f"\nOverall: baseline={overall_baseline:.1f}s, corrected={overall_corrected:.1f}s "
          f"({100 * (1 - overall_corrected / overall_baseline):.1f}% improvement)")

    table = summarize(test, corrected_error)
    print("\nBy lead time:")
    print(table.to_string(index=False))

    if len(table):
        out_path = config.ROOT / "reports" / "blue_model_improvement.png"
        plot_comparison(table, str(out_path))
        print(f"\nSaved -> {out_path}")

    final_lookup = fit_lookup(df)
    export_lookup_csv(final_lookup, df, ARTIFACTS_DIR)
    print(f"\nproduction lookup exported to {ARTIFACTS_DIR}/ "
          f"(fit on all {len(df):,} rows, isolated from model_artifacts/)")


if __name__ == "__main__":
    main()
