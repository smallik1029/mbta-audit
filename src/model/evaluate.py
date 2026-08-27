"""evaluate the correction table on the held-out split"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src import config
from src.analysis.error_curves import HEADLINE_TARGETS_MIN, HEADLINE_TOLERANCE_MIN


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    data_dir = config.ROOT / "data"
    train = pd.read_parquet(data_dir / "_train_split.parquet")
    test = pd.read_parquet(data_dir / "_test_split.parquet")
    lookup = {
        key: pd.read_parquet(data_dir / f"_lookup_{key}.parquet")["bias_sec"]
        for key in ("by_stop", "by_route", "by_bucket")
    }
    return train, test, lookup


def correct(test: pd.DataFrame, lookup: dict) -> pd.Series:
    """Apply the lookup, falling back route->global when a (route,stop,bucket)"""
    idx_stop = pd.MultiIndex.from_arrays([test["route_id"], test["stop_id"], test["lead_bucket"]])
    idx_route = pd.MultiIndex.from_arrays([test["route_id"], test["lead_bucket"]])

    bias = lookup["by_stop"].reindex(idx_stop).to_numpy()
    route_bias = lookup["by_route"].reindex(idx_route).to_numpy()
    bucket_bias = lookup["by_bucket"].reindex(test["lead_bucket"]).to_numpy()

    missing_stop = pd.isna(bias)
    bias = pd.Series(bias).where(~missing_stop, route_bias).to_numpy()
    missing_route = pd.isna(bias)
    bias = pd.Series(bias).where(~missing_route, bucket_bias).to_numpy()

    return test["error_sec"] - bias, missing_stop.sum(), missing_route.sum()


MIN_SAMPLES_FOR_HEADLINE = 20


def summarize(test: pd.DataFrame, corrected_error: pd.Series) -> pd.DataFrame:
    rows = []
    for target in HEADLINE_TARGETS_MIN:
        lo = (target - HEADLINE_TOLERANCE_MIN) * 60
        hi = (target + HEADLINE_TOLERANCE_MIN) * 60
        mask = (test["lead_time_sec"] >= lo) & (test["lead_time_sec"] < hi)
        if mask.sum() < MIN_SAMPLES_FOR_HEADLINE:
            continue
        baseline = test.loc[mask, "error_sec"].abs().median()
        corrected = corrected_error.loc[mask].abs().median()
        rows.append({
            "lead_time_min": target,
            "n": int(mask.sum()),
            "baseline_median_abs_error_sec": round(baseline, 1),
            "corrected_median_abs_error_sec": round(corrected, 1),
            "improvement_pct": round(100 * (1 - corrected / baseline), 1) if baseline else None,
        })
    return pd.DataFrame(rows)


def summarize_by_route(test: pd.DataFrame, corrected_error: pd.Series) -> pd.DataFrame:
    """per-route improvement, so one line cannot hide behind another"""
    rows = []
    for route_id, idx in test.groupby("route_id").groups.items():
        baseline = test.loc[idx, "error_sec"].abs().median()
        corrected = corrected_error.loc[idx].abs().median()
        rows.append({
            "route_id": route_id,
            "n": len(idx),
            "baseline_median_abs_error_sec": round(baseline, 1),
            "corrected_median_abs_error_sec": round(corrected, 1),
            "improvement_pct": round(100 * (1 - corrected / baseline), 1) if baseline else None,
        })
    return pd.DataFrame(rows).sort_values("n", ascending=False)


def plot_comparison(table: pd.DataFrame, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = range(len(table))
    width = 0.35
    ax.bar([i - width / 2 for i in x], table["baseline_median_abs_error_sec"],
           width, label="MBTA raw prediction", color="#8a8a8a")
    ax.bar([i + width / 2 for i in x], table["corrected_median_abs_error_sec"],
           width, label="Corrected (this project)", color="#DA291C")
    for i, row in table.iterrows():
        ax.text(i, max(row["baseline_median_abs_error_sec"], row["corrected_median_abs_error_sec"]) + 2,
                f"-{row['improvement_pct']:.0f}%", ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{m} min out" for m in table["lead_time_min"]])
    ax.set_ylabel("Median absolute error (seconds)")
    ax.set_title("Bias-corrected predictions vs. MBTA's raw predictions (held-out test set)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    train, test, lookup = load_splits()
    print(f"evaluating on {len(test):,} held-out test rows "
          f"({test['observed_at'].min()} -> {test['observed_at'].max()})")

    corrected_error, n_missing_stop, n_missing_route = correct(test, lookup)
    print(f"fallback usage: {n_missing_stop:,} rows fell back from stop->route level, "
          f"{n_missing_route:,} rows fell back further to route->global level")

    overall_baseline = test["error_sec"].abs().median()
    overall_corrected = corrected_error.abs().median()
    overall_improvement = 100 * (1 - overall_corrected / overall_baseline)
    print(f"\nOverall: baseline median |error| = {overall_baseline:.1f}s, "
          f"corrected = {overall_corrected:.1f}s "
          f"({overall_improvement:.1f}% improvement)")

    table = summarize(test, corrected_error)
    print("\nBy lead time:")
    print(table.to_string(index=False))

    print("\nBy route:")
    print(summarize_by_route(test, corrected_error).to_string(index=False))

    reports_dir = config.ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)
    out_path = reports_dir / "model_improvement.png"
    plot_comparison(table, str(out_path))
    print(f"\nSaved comparison chart -> {out_path}")


if __name__ == "__main__":
    main()
