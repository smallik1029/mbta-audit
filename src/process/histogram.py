"""fold outcomes into a pre-aggregated error histogram"""
import pandas as pd

from src import db

STATE_KEY = "histogram_last_outcome_id"

BATCH_ROWS = 500_000

_SQL_LEAD_BUCKET = "CAST(MIN(lead_time_sec / 60.0, 30.0) / 3.0 AS INT) * 3"

CELL_DTYPES = {
    "route_id": "string[pyarrow]",
    "stop_id": "string[pyarrow]",
    "lead_bucket": "int16",
    "error_bin": "int32",
    "n": "int64",
}

CELL_CHUNK_ROWS = 500_000


def compact(cells: pd.DataFrame) -> pd.DataFrame:
    """narrow dtypes. an object string costs sixty bytes and the histogram has millions"""
    return cells.astype({k: v for k, v in CELL_DTYPES.items() if k in cells.columns})


def _read_cells(sql: str, params: tuple) -> pd.DataFrame:
    """read in bounded chunks so the whole histogram is never resident as python objects"""
    chunks = [
        compact(chunk)
        for chunk in pd.read_sql_query(
            sql, db.connect(), params=params, chunksize=CELL_CHUNK_ROWS
        )
    ]
    if not chunks:
        return compact(pd.DataFrame(columns=list(CELL_DTYPES)))
    return pd.concat(chunks, ignore_index=True)


def _plain_keys(series: pd.Series) -> pd.Series:
    """group keys go back to objects and int64 so lookups against the outcome table match"""
    def plain(level):
        if pd.api.types.is_integer_dtype(level):
            return level.astype("int64")
        return level.astype("object")

    idx = series.index
    if isinstance(idx, pd.MultiIndex):
        series.index = idx.set_levels([plain(level) for level in idx.levels])
    else:
        series.index = plain(idx)
    return series


def get_watermark() -> int:
    conn = db.connect()
    row = conn.execute("SELECT value FROM pipeline_state WHERE key = ?", (STATE_KEY,)).fetchone()
    return int(row[0]) if row else 0


def set_watermark(value: int) -> None:
    conn = db.connect()
    conn.execute(
        "INSERT INTO pipeline_state (key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (STATE_KEY, str(value)),
    )
    conn.commit()


def fold_batch(since_id: int, limit: int = BATCH_ROWS) -> tuple[int, int]:
    """fold one bounded batch into the histogram"""
    conn = db.connect()
    batch_max = conn.execute(
        "SELECT MAX(id) FROM (SELECT id FROM prediction_outcomes "
        "WHERE id > ? ORDER BY id LIMIT ?)",
        (since_id, limit),
    ).fetchone()[0]
    if batch_max is None:
        return 0, since_id

    cur = conn.execute(
        f"""
        INSERT INTO outcome_histogram (route_id, stop_id, lead_bucket, month, error_bin, n)
        SELECT route_id, stop_id, {_SQL_LEAD_BUCKET},
               substr(observed_at, 1, 7),
               CAST(ROUND(error_sec) AS INTEGER),
               COUNT(*)
        FROM prediction_outcomes
        WHERE id > ? AND id <= ? AND route_id IS NOT NULL
        GROUP BY 1, 2, 3, 4, 5
        ON CONFLICT (route_id, stop_id, lead_bucket, month, error_bin)
        DO UPDATE SET n = n + excluded.n
        """,
        (since_id, batch_max),
    )
    folded = conn.execute(
        "SELECT COUNT(*) FROM prediction_outcomes WHERE id > ? AND id <= ?",
        (since_id, batch_max),
    ).fetchone()[0]
    conn.commit()
    _ = cur
    return folded, batch_max


def update() -> int:
    """fold everything new since the last run, in batches"""
    since = get_watermark()
    total = 0
    while True:
        folded, new_watermark = fold_batch(since)
        if not folded:
            break
        set_watermark(new_watermark)
        since = new_watermark
        total += folded
    return total


def load_cells(routes: tuple[str, ...]) -> pd.DataFrame:
    """the histogram for these routes, summed across months"""
    placeholders = ",".join("?" * len(routes))
    return _read_cells(
        "SELECT route_id, stop_id, lead_bucket, error_bin, SUM(n) AS n "
        f"FROM outcome_histogram WHERE route_id IN ({placeholders}) "
        "GROUP BY route_id, stop_id, lead_bucket, error_bin",
        tuple(routes),
    )


def cells_between(routes: tuple[str, ...], start: str, end: str) -> pd.DataFrame:
    """bins for one time range, straight from prediction_outcomes"""
    placeholders = ",".join("?" * len(routes))
    return _read_cells(
        f"SELECT route_id, stop_id, {_SQL_LEAD_BUCKET} AS lead_bucket, "
        "CAST(ROUND(error_sec) AS INTEGER) AS error_bin, COUNT(*) AS n "
        "FROM prediction_outcomes "
        "WHERE observed_at >= ? AND observed_at <= ? AND route_id IS NOT NULL "
        f"AND route_id IN ({placeholders}) GROUP BY 1, 2, 3, 4",
        (start, end, *routes),
    )


def subtract_cells(total: pd.DataFrame, part: pd.DataFrame) -> pd.DataFrame:
    """total minus part, bin by bin"""
    keys = ["route_id", "stop_id", "lead_bucket", "error_bin"]
    if part.empty or total.empty:
        return total
    merged = total.merge(part, on=keys, how="left", suffixes=("", "_part"))
    merged["n"] = merged["n"] - merged["n_part"].fillna(0)
    return compact(merged.loc[merged["n"] > 0, [*keys, "n"]].reset_index(drop=True))


def medians_from_cells(cells: pd.DataFrame, group_cols: list) -> pd.Series:
    """median error per group, read off the counts"""
    if cells.empty:
        return pd.Series(dtype=float, name="error_sec")

    ordered = cells.sort_values([*group_cols, "error_bin"])
    grouped = ordered.groupby(group_cols, sort=False)
    cumulative = grouped["n"].cumsum()
    totals = grouped["n"].transform("sum")

    lower_rank = (totals + 1) // 2
    upper_rank = (totals + 2) // 2
    work = ordered.assign(_cum=cumulative, _lo=lower_rank, _hi=upper_rank)

    lo = work[work["_cum"] >= work["_lo"]].groupby(group_cols, sort=False)["error_bin"].first()
    hi = work[work["_cum"] >= work["_hi"]].groupby(group_cols, sort=False)["error_bin"].first()
    return _plain_keys(((lo + hi) / 2.0).rename("error_sec"))


def counts_from_cells(cells: pd.DataFrame, group_cols: list) -> pd.Series:
    """how many observations back each group"""
    if cells.empty:
        return pd.Series(dtype="int64", name="n")
    return _plain_keys(cells.groupby(group_cols)["n"].sum())


def prune_months(cutoff_month: str) -> int:
    """drop whole months older than the cutoff"""
    conn = db.connect()
    cur = conn.execute("DELETE FROM outcome_histogram WHERE month < ?", (cutoff_month,))
    conn.commit()
    return cur.rowcount


def main() -> None:
    db.init_schema()
    before = get_watermark()
    folded = update()
    conn = db.connect()
    cells = conn.execute("SELECT COUNT(*) FROM outcome_histogram").fetchone()[0]
    observations = conn.execute("SELECT COALESCE(SUM(n), 0) FROM outcome_histogram").fetchone()[0]
    if folded:
        print(f"histogram: folded {folded:,} new outcome rows "
              f"(id {before:,} -> {get_watermark():,})")
    else:
        print("histogram: nothing new to fold")
    print(f"histogram: {cells:,} bins summarizing {observations:,} observations")


if __name__ == "__main__":
    main()
