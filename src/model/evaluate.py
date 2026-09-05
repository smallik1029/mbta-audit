"""evaluate the correction table on the held-out split"""
import json
from datetime import UTC, datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src import config
from src.analysis.error_curves import HEADLINE_TARGETS_MIN, HEADLINE_TOLERANCE_MIN
from src.model.features import TOP_LEAD_BUCKET
from src.model.features import apply_correction as correct


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    data_dir = config.ROOT / "data"
    train = pd.read_parquet(data_dir / "_train_split.parquet")
    test = pd.read_parquet(data_dir / "_test_split.parquet")
    lookup = {
        key: pd.read_parquet(data_dir / f"_lookup_{key}.parquet")["bias_sec"]
        for key in ("by_stop", "by_route", "by_bucket")
    }
    return train, test, lookup


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


def route_records(by_route) -> list:
    """the per-route rows the status page reads"""
    return [
        {
            "route_id": row["route_id"],
            "n": int(row["n"]),
            "baseline_sec": float(row["baseline_median_abs_error_sec"]),
            "corrected_sec": float(row["corrected_median_abs_error_sec"]),
            "improvement_pct": float(row["improvement_pct"]),
        }
        for _, row in by_route.iterrows()
        if row["improvement_pct"] is not None
    ]


def overall_stats(test, corrected_error) -> dict:
    """baseline and corrected medians for one population"""
    baseline = float(test["error_sec"].abs().median())
    corrected = float(corrected_error.abs().median())
    return {
        "baseline_sec": round(baseline, 1),
        "corrected_sec": round(corrected, 1),
        "improvement_pct": round(100 * (1 - corrected / baseline), 1),
    }


def write_report(test, overall: dict, overall_served: dict, by_route, by_route_served,
                 served_rows: int) -> None:
    """dated json the status page reads, so a stale number announces itself"""
    report = {
        "measured_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "test_rows": int(len(test)),
        "test_start": str(test["observed_at"].min()),
        "test_end": str(test["observed_at"].max()),
        "overall": overall,
        "overall_served": overall_served,
        "served_rows": served_rows,
        "served_max_lead_min": TOP_LEAD_BUCKET,
        "by_route": route_records(by_route),
        "by_route_served": route_records(by_route_served),
    }
    out = config.ROOT / "model_artifacts" / "evaluation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"Saved evaluation report -> {out}")


def main() -> None:
    train, test, lookup = load_splits()
    print(f"evaluating on {len(test):,} held-out test rows "
          f"({test['observed_at'].min()} -> {test['observed_at'].max()})")

    corrected_error, n_missing_stop, n_missing_route = correct(test, lookup)
    print(f"fallback usage: {n_missing_stop:,} rows fell back from stop->route level, "
          f"{n_missing_route:,} rows fell back further to route->global level")

    overall = overall_stats(test, corrected_error)
    print(f"\nOverall, every lead time: {overall['baseline_sec']:.1f}s -> "
          f"{overall['corrected_sec']:.1f}s "
          f"({overall['improvement_pct']:.1f}% improvement)")

    table = summarize(test, corrected_error)
    print("\nBy lead time:")
    print(table.to_string(index=False))

    by_route = summarize_by_route(test, corrected_error)
    print("\nBy route, every lead time:")
    print(by_route.to_string(index=False))

    served = test["lead_bucket"] < TOP_LEAD_BUCKET
    overall_served = overall_stats(test[served], corrected_error[served])
    by_route_served = summarize_by_route(test[served], corrected_error[served])
    print(f"\nBy route, only the {served.sum():,} rows the site serves "
          f"(under {TOP_LEAD_BUCKET} min), overall "
          f"{overall_served['baseline_sec']:.1f}s -> {overall_served['corrected_sec']:.1f}s "
          f"({overall_served['improvement_pct']:.1f}%):")
    print(by_route_served.to_string(index=False))

    write_report(test, overall, overall_served, by_route, by_route_served,
                 int(served.sum()))

    reports_dir = config.ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)
    out_path = reports_dir / "model_improvement.png"
    plot_comparison(table, str(out_path))
    print(f"\nSaved comparison chart -> {out_path}")


if __name__ == "__main__":
    main()
