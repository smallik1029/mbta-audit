"""fit the bias-correction lookup table"""
import pandas as pd

from src import config, db
from src.model.features import add_lead_bucket, time_split

VALIDATED_ROUTES = ("Red", "Orange", "Blue")

SAMPLE_ROW_CAP = 1_500_000

_SQL_LEAD_BUCKET = "CAST(MIN(lead_time_sec / 60.0, 30.0) / 3.0 AS INT) * 3"


def count_by_cell(routes: tuple[str, ...]) -> pd.DataFrame:
    """The TRUE historical count per (route, stop, lead_bucket) cell, via a"""
    conn = db.connect()
    placeholders = ",".join("?" * len(routes))
    df = pd.read_sql_query(
        f"SELECT route_id, stop_id, {_SQL_LEAD_BUCKET} AS lead_bucket, COUNT(*) AS n "
        f"FROM prediction_outcomes WHERE route_id IN ({placeholders}) "
        "GROUP BY route_id, stop_id, lead_bucket",
        conn,
        params=routes,
    )
    return df


def counts_by_key(cell_counts: pd.DataFrame) -> dict:
    """Roll the finest-grain true counts up to each of the three lookup"""
    return {key: cell_counts.groupby(cols)["n"].sum() for key, cols in LOOKUP_KEYS.items()}


def load_outcomes(sample_cap: int = SAMPLE_ROW_CAP) -> pd.DataFrame:
    """Loads matching outcome rows for the median computation. Below"""
    conn = db.connect()
    placeholders = ",".join("?" * len(VALIDATED_ROUTES))
    total = conn.execute(
        f"SELECT COUNT(*) FROM prediction_outcomes WHERE route_id IN ({placeholders})",
        VALIDATED_ROUTES,
    ).fetchone()[0]

    base_select = (
        "SELECT route_id, stop_id, observed_at, lead_time_sec, error_sec "
        f"FROM prediction_outcomes WHERE route_id IN ({placeholders})"
    )
    if total <= sample_cap:
        df = pd.read_sql_query(base_select, conn, params=VALIDATED_ROUTES)
    else:
        stride = -(-total // sample_cap)
        df = pd.read_sql_query(
            base_select + " AND id % ? = 0", conn, params=(*VALIDATED_ROUTES, stride)
        )
        print(f"sampled 1-in-{stride} rows ({total:,} total, {len(df):,} loaded) "
              "to keep the median computation memory-bounded")

    df["observed_at"] = pd.to_datetime(df["observed_at"], utc=True, format="ISO8601")
    return add_lead_bucket(df)


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
    df = load_outcomes()
    cell_counts = count_by_cell(VALIDATED_ROUTES)
    counts = counts_by_key(cell_counts)
    true_total = int(cell_counts["n"].sum())

    train, test = time_split(df)
    print(f"train: {len(train):,} rows ({train['observed_at'].min()} -> {train['observed_at'].max()})")
    print(f"test:  {len(test):,} rows ({test['observed_at'].min()} -> {test['observed_at'].max()})")

    lookup = fit_lookup(train)
    print(f"\nvalidation lookup table: {len(lookup['by_stop']):,} (route,stop,lead_bucket) cells")
    train.to_parquet(config.ROOT / "data" / "_train_split.parquet")
    test.to_parquet(config.ROOT / "data" / "_test_split.parquet")
    for key, series in lookup.items():
        series.to_frame("bias_sec").to_parquet(config.ROOT / "data" / f"_lookup_{key}.parquet")

    final_lookup = fit_lookup(df)
    artifacts_dir = config.ROOT / "model_artifacts"
    export_lookup_csv(final_lookup, counts, artifacts_dir)
    print(f"production lookup table: {len(final_lookup['by_stop']):,} cells, "
          f"fit on {len(df):,} loaded rows ({true_total:,} true total) -> {artifacts_dir}/")


if __name__ == "__main__":
    main()
