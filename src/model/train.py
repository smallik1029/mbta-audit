"""fit the bias-correction lookup table"""
import pandas as pd

from src import config, db
from src.model.features import add_lead_bucket, time_split


def load_outcomes() -> pd.DataFrame:
    conn = db.connect()
    placeholders = ",".join("?" * len(config.ROUTES))
    df = pd.read_sql_query(
        "SELECT route_id, stop_id, observed_at, lead_time_sec, error_sec "
        f"FROM prediction_outcomes WHERE route_id IN ({placeholders})",
        conn,
        params=config.ROUTES,
    )
    df["observed_at"] = pd.to_datetime(df["observed_at"], utc=True, format="ISO8601")
    return add_lead_bucket(df)


def fit_lookup(train: pd.DataFrame) -> dict:
    by_stop = train.groupby(["route_id", "stop_id", "lead_bucket"])["error_sec"].median()
    by_route = train.groupby(["route_id", "lead_bucket"])["error_sec"].median()
    by_bucket = train.groupby(["lead_bucket"])["error_sec"].median()
    return {"by_stop": by_stop, "by_route": by_route, "by_bucket": by_bucket}


def main() -> None:
    df = load_outcomes()
    train, test = time_split(df)
    print(f"train: {len(train):,} rows ({train['observed_at'].min()} -> {train['observed_at'].max()})")
    print(f"test:  {len(test):,} rows ({test['observed_at'].min()} -> {test['observed_at'].max()})")

    lookup = fit_lookup(train)
    print(f"\nlookup table: {len(lookup['by_stop']):,} (route,stop,lead_bucket) cells")

    train.to_parquet(config.ROOT / "data" / "_train_split.parquet")
    test.to_parquet(config.ROOT / "data" / "_test_split.parquet")
    for key, series in lookup.items():
        series.to_frame("bias_sec").to_parquet(config.ROOT / "data" / f"_lookup_{key}.parquet")
    print("saved train/test splits and lookup table to data/ (gitignored, intermediate only)")


if __name__ == "__main__":
    main()
