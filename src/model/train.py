"""fit the bias-correction lookup table"""
import pandas as pd

from src import config, db
from src.model.features import (
    LEAD_BUCKET_MAX_MIN,
    LEAD_BUCKET_WIDTH_MIN,
    add_lead_bucket,
    apply_correction,
    time_split,
)
from src.process import histogram

VALIDATED_ROUTES = ("Red", "Orange", "Blue",
                    "Green-B", "Green-C", "Green-D", "Green-E")

EVAL_WINDOW_DAYS = 3

EVAL_MAX_ROWS = 1_500_000


def fit_production_lookup() -> tuple[dict, dict, int]:
    """the lookup that ships, fit from the histogram"""
    cells = histogram.load_cells(VALIDATED_ROUTES)
    counts = {
        key: histogram.counts_from_cells(cells, cols) for key, cols in LOOKUP_KEYS.items()
    }
    return lookup_from_cells(cells), counts, int(cells["n"].sum())


def lookup_from_cells(cells: pd.DataFrame) -> dict:
    """a lookup in the production shape, from binned counts"""
    return {
        key: histogram.medians_from_cells(cells, cols).rename("bias_sec")
        for key, cols in LOOKUP_KEYS.items()
    }


def eval_stride(total_rows: int, max_rows: int = EVAL_MAX_ROWS) -> int:
    """rows to skip per row kept, so any window fits in max_rows"""
    if total_rows <= max_rows:
        return 1
    return -(-total_rows // max_rows)


def load_recent_outcomes(
    days: int = EVAL_WINDOW_DAYS, max_rows: int = EVAL_MAX_ROWS
) -> pd.DataFrame:
    """raw rows from the last few days, for the split and the guard"""
    conn = db.connect()
    placeholders = ",".join("?" * len(VALIDATED_ROUTES))
    cutoff = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)).isoformat()

    total = conn.execute(
        "SELECT COUNT(*) FROM prediction_outcomes WHERE observed_at > ?", (cutoff,)
    ).fetchone()[0]
    stride = eval_stride(total, max_rows)

    query = (
        "SELECT route_id, stop_id, observed_at, lead_time_sec, error_sec "
        f"FROM prediction_outcomes WHERE observed_at > ? AND route_id IN ({placeholders})"
    )
    if stride > 1:
        query += f" AND id % {stride} = 0"

    df = pd.read_sql_query(query, conn, params=(cutoff, *VALIDATED_ROUTES))
    print(f"eval window: {days}d, {total:,} rows available, stride {stride} "
          f"-> {len(df):,} loaded", flush=True)
    df["observed_at"] = pd.to_datetime(df["observed_at"], utc=True, format="ISO8601")
    return add_lead_bucket(df)


GUARD_GROUP = ["route_id", "lead_bucket"]
GUARD_MIN_SAMPLES = 200

GUARD_CHUNKS = 4

GUARD_EXEMPT_TOP_BUCKET = True


def guard_scores(fit_window: pd.DataFrame) -> pd.DataFrame:
    """per (route, lead_bucket), how the shipped model does out of sample"""
    holdout = _chronological_chunks(fit_window, GUARD_CHUNKS)[-1]
    corrected_error, _, _ = apply_correction(holdout, holdout_lookup(holdout))
    scored = holdout.assign(_baseline=holdout["error_sec"].abs(),
                            _corrected=corrected_error.abs())
    stats = scored.groupby(GUARD_GROUP).agg(
        n=("_baseline", "size"),
        baseline=("_baseline", "median"),
        corrected=("_corrected", "median"),
    )
    stats = stats[stats["baseline"] > 0]
    stats["improvement_pct"] = 100 * (1 - stats["corrected"] / stats["baseline"])
    stats["folds"] = 1
    return stats


def holdout_lookup(holdout: pd.DataFrame, cells: pd.DataFrame | None = None) -> dict:
    """the production lookup with the holdout period taken back out"""
    start = holdout["observed_at"].min().isoformat()
    end = holdout["observed_at"].max().isoformat()
    if cells is None:
        cells = histogram.load_cells(VALIDATED_ROUTES)
    excluded = histogram.cells_between(VALIDATED_ROUTES, start, end)
    return lookup_from_cells(histogram.subtract_cells(cells, excluded))


def select_harmful_groups(fit_window: pd.DataFrame) -> set:
    """groups where correcting did not beat leaving MBTA time alone"""
    stats = guard_scores(fit_window)
    judged = stats[(stats["n"] >= GUARD_MIN_SAMPLES) & (stats["baseline"] > 0)]
    worst = judged.nsmallest(5, "improvement_pct")
    print("guard: weakest groups on the holdout ->", ", ".join(
        f"{route}@{bucket}min {row.improvement_pct:+.1f}% (n={row.n:,})"
        for (route, bucket), row in worst.iterrows()
    ), flush=True)
    harmful = set(judged.index[judged["improvement_pct"] < 0])
    if GUARD_EXEMPT_TOP_BUCKET:
        top = LEAD_BUCKET_MAX_MIN - (LEAD_BUCKET_MAX_MIN % LEAD_BUCKET_WIDTH_MIN)
        harmful = {(route, bucket) for route, bucket in harmful if bucket != top}
    return harmful


def _chronological_chunks(df: pd.DataFrame, n: int) -> list:
    """split into n time-ordered chunks"""
    df = df.sort_values("observed_at")
    bounds = [len(df) * i // n for i in range(n + 1)]
    return [df.iloc[bounds[i]:bounds[i + 1]] for i in range(n)]


def suppress_groups(lookup: dict, harmful: pd.MultiIndex) -> dict:
    """zero the bias at every level, so a suppressed group cannot return via the fallback"""
    if not len(harmful):
        return lookup
    harmful_pairs = set(harmful)
    guarded = dict(lookup)

    by_stop = lookup["by_stop"].copy()
    stop_pairs = list(zip(by_stop.index.get_level_values("route_id"),
                          by_stop.index.get_level_values("lead_bucket"), strict=False))
    by_stop[[p in harmful_pairs for p in stop_pairs]] = 0.0
    guarded["by_stop"] = by_stop

    by_route = lookup["by_route"].copy()
    route_pairs = list(zip(by_route.index.get_level_values("route_id"),
                           by_route.index.get_level_values("lead_bucket"), strict=False))
    by_route[[p in harmful_pairs for p in route_pairs]] = 0.0
    guarded["by_route"] = by_route

    return guarded


def fit_lookup(data: pd.DataFrame) -> dict:
    by_stop = data.groupby(["route_id", "stop_id", "lead_bucket"])["error_sec"].median()
    by_route = data.groupby(["route_id", "lead_bucket"])["error_sec"].median()
    by_bucket = data.groupby(["lead_bucket"])["error_sec"].median()
    return {"by_stop": by_stop, "by_route": by_route, "by_bucket": by_bucket}


LOOKUP_KEYS = {
    "by_stop": ["route_id", "stop_id", "lead_bucket"],
    "by_route": ["route_id", "lead_bucket"],
    "by_bucket": ["lead_bucket"],
}


def export_lookup_csv(lookup: dict, counts: dict, out_dir) -> None:
    """the small csvs the serving code reads. includes n so the ui can show what backs each number"""
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, series in lookup.items():
        merged = series.to_frame("bias_sec").join(counts[key])
        merged.reset_index().to_csv(out_dir / f"bias_{key}.csv", index=False)


def main() -> None:
    db.init_schema()

    folded = histogram.update()
    if folded:
        print(f"histogram: folded {folded:,} new outcome rows")

    df = load_recent_outcomes()

    fit_window, test = time_split(df)
    print(f"fit:   {len(fit_window):,} rows "
          f"({fit_window['observed_at'].min()} -> {fit_window['observed_at'].max()})")
    print(f"test:  {len(test):,} rows ({test['observed_at'].min()} -> {test['observed_at'].max()})")

    harmful = select_harmful_groups(fit_window)
    lookup = suppress_groups(holdout_lookup(test), harmful)
    if harmful:
        pairs = ", ".join(f"{r} @{b}min" for r, b in sorted(harmful))
        print(f"\ndo-no-harm guard: suppressing {len(harmful)} (route, lead_bucket) "
              f"group(s) that did not beat MBTA's raw time out of sample -> {pairs}")
    else:
        print("\ndo-no-harm guard: no groups suppressed (every bias was stable enough to apply)")
    print(f"validation lookup table: {len(lookup['by_stop']):,} (route,stop,lead_bucket) cells")
    train = fit_window
    train.to_parquet(config.ROOT / "data" / "_train_split.parquet")
    test.to_parquet(config.ROOT / "data" / "_test_split.parquet")
    for key, series in lookup.items():
        series.to_frame("bias_sec").to_parquet(config.ROOT / "data" / f"_lookup_{key}.parquet")

    production, counts, true_total = fit_production_lookup()
    final_lookup = suppress_groups(production, harmful)
    artifacts_dir = config.ROOT / "model_artifacts"
    export_lookup_csv(final_lookup, counts, artifacts_dir)
    print(f"production lookup table: {len(final_lookup['by_stop']):,} cells, "
          f"fit on all {true_total:,} observations -> {artifacts_dir}/")


if __name__ == "__main__":
    main()
