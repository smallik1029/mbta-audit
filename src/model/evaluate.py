"""evaluate the correction table on the held-out split"""
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


def summarize(test: pd.DataFrame, corrected_error: pd.Series) -> pd.DataFrame:
    rows = []
    for target in HEADLINE_TARGETS_MIN:
        lo = (target - HEADLINE_TOLERANCE_MIN) * 60
        hi = (target + HEADLINE_TOLERANCE_MIN) * 60
        mask = (test["lead_time_sec"] >= lo) & (test["lead_time_sec"] < hi)
        if mask.sum() == 0:
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


if __name__ == "__main__":
    main()
