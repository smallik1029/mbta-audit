"""fit the bias-correction lookup table"""
import pandas as pd

from src import config, db
from src.model.features import add_lead_bucket, time_split

VALIDATED_ROUTES = ("Red", "Orange")


def load_outcomes() -> pd.DataFrame:
    conn = db.connect()
    placeholders = ",".join("?" * len(VALIDATED_ROUTES))
    df = pd.read_sql_query(
        "SELECT route_id, stop_id, observed_at, lead_time_sec, error_sec "
        f"FROM prediction_outcomes WHERE route_id IN ({placeholders})",
        conn,
        params=VALIDATED_ROUTES,
    )
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


def export_lookup_csv(lookup: dict, data: pd.DataFrame, out_dir) -> None:
    """the small csvs the serving code reads. includes n so the ui can show what backs each number"""
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, series in lookup.items():
        counts = data.groupby(LOOKUP_KEYS[key]).size().rename("n")
        merged = series.to_frame("bias_sec").join(counts)
        merged.reset_index().to_csv(out_dir / f"bias_{key}.csv", index=False)


def main() -> None:
    df = load_outcomes()
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
    export_lookup_csv(final_lookup, df, artifacts_dir)
    print(f"production lookup table: {len(final_lookup['by_stop']):,} cells, "
          f"fit on all {len(df):,} rows -> {artifacts_dir}/")


if __name__ == "__main__":
    main()
