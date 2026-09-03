"""improvement per route and lead bucket, the cut that moves least between runs"""
import pandas as pd

from src.model.evaluate import load_splits
from src.model.features import apply_correction

MIN_ROWS = 200


def improvement_table(test: pd.DataFrame, corrected_error: pd.Series) -> pd.DataFrame:
    """median absolute error before and after, per route and lead bucket"""
    scored = test.assign(_base=test["error_sec"].abs(), _corr=corrected_error.abs())
    stats = scored.groupby(["route_id", "lead_bucket"]).agg(
        n=("_base", "size"),
        baseline=("_base", "median"),
        corrected=("_corr", "median"),
    )
    stats = stats[(stats["n"] >= MIN_ROWS) & (stats["baseline"] > 0)]
    stats["improvement_pct"] = 100 * (1 - stats["corrected"] / stats["baseline"])
    return stats


def main() -> None:
    train, test, lookup = load_splits()
    corrected_error, missing_stop, missing_route = apply_correction(test, lookup)
    stats = improvement_table(test, corrected_error)

    print(f"test rows: {len(test):,} "
          f"({test['observed_at'].min()} -> {test['observed_at'].max()})")
    print(f"fell back from stop level: {missing_stop:,}, and further to global: {missing_route:,}")
    print()

    grid = stats["improvement_pct"].unstack("lead_bucket").round(1)
    print("IMPROVEMENT PER ROUTE AND LEAD BUCKET, percent. positive means we beat MBTA")
    print(grid.to_string())
    print()

    print("MBTA MEDIAN ABSOLUTE ERROR, seconds")
    print(stats["baseline"].unstack("lead_bucket").round(0).to_string())
    print()

    print("ROWS BEHIND EACH CELL")
    print(stats["n"].unstack("lead_bucket").to_string())
    print()

    print("where each route stops helping:")
    for route, row in grid.iterrows():
        helps = [int(b) for b, v in row.items() if pd.notna(v) and v > 0]
        hurts = [int(b) for b, v in row.items() if pd.notna(v) and v <= 0]
        print(f"  {route:<8} helps at {helps}")
        print(f"  {'':<8} hurts at {hurts}")


if __name__ == "__main__":
    main()
