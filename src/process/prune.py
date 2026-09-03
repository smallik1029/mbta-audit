"""delete rows past their retention window and give the space back"""
from datetime import UTC, datetime, timedelta

from src import db
from src.process import histogram

RAW_RETENTION_DAYS = 14
OUTCOMES_RETENTION_DAYS = 35
HISTOGRAM_RETENTION_DAYS = None


def record_collection_start() -> str | None:
    """pin the real start date before retention trims the table it was read from"""
    conn = db.connect()
    row = conn.execute(
        "SELECT value FROM pipeline_state WHERE key = ?", (db.COLLECTION_START_KEY,)
    ).fetchone()
    if row:
        return row["value"]

    earliest = conn.execute("SELECT MIN(observed_at) FROM prediction_snapshots").fetchone()[0]
    if earliest is None:
        earliest = conn.execute("SELECT MIN(observed_at) FROM prediction_outcomes").fetchone()[0]
    if earliest is None:
        return None

    conn.execute(
        "INSERT INTO pipeline_state (key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (db.COLLECTION_START_KEY, earliest),
    )
    conn.commit()
    return earliest


def prune_raw(now: datetime | None = None) -> dict:
    now = now or datetime.now(UTC)
    cutoff = (now - timedelta(days=RAW_RETENTION_DAYS)).isoformat()
    conn = db.connect()
    cur = conn.execute("DELETE FROM prediction_snapshots WHERE observed_at < ?", (cutoff,))
    pred_deleted = cur.rowcount
    cur = conn.execute("DELETE FROM vehicle_snapshots WHERE observed_at < ?", (cutoff,))
    veh_deleted = cur.rowcount
    conn.commit()
    return {"prediction_snapshots": pred_deleted, "vehicle_snapshots": veh_deleted}


def prune_outcomes(now: datetime | None = None) -> dict:
    """outcomes only. the histogram backs the model, so it retires on its own schedule"""
    now = now or datetime.now(UTC)
    cutoff = (now - timedelta(days=OUTCOMES_RETENTION_DAYS)).isoformat()
    conn = db.connect()
    cur = conn.execute("DELETE FROM prediction_outcomes WHERE observed_at < ?", (cutoff,))
    deleted = cur.rowcount
    conn.commit()
    return {"prediction_outcomes": deleted}


def prune_histogram(now: datetime | None = None) -> int:
    """kept in full while HISTOGRAM_RETENTION_DAYS is None"""
    if HISTOGRAM_RETENTION_DAYS is None:
        return 0
    now = now or datetime.now(UTC)
    cutoff_month = (now - timedelta(days=HISTOGRAM_RETENTION_DAYS)).strftime("%Y-%m")
    return histogram.prune_months(cutoff_month)


def reclaim_space(pages: int = 20000) -> None:
    """hand freed pages back to the os. only does anything if auto_vacuum is on"""
    conn = db.connect()
    conn.execute(f"PRAGMA incremental_vacuum({pages})")
    conn.commit()


def main() -> None:
    db.init_schema()
    now = datetime.now(UTC)
    started = record_collection_start()
    raw_deleted = prune_raw(now)
    outcomes_deleted = prune_outcomes(now)
    histogram_deleted = prune_histogram(now)

    print(f"collection start pinned at {started}")
    print(f"pruned {raw_deleted['prediction_snapshots']:,} prediction_snapshots, "
          f"{raw_deleted['vehicle_snapshots']:,} vehicle_snapshots rows "
          f"(older than {RAW_RETENTION_DAYS} days)")
    print(f"pruned {outcomes_deleted['prediction_outcomes']:,} prediction_outcomes rows "
          f"(older than {OUTCOMES_RETENTION_DAYS} days)")
    if HISTOGRAM_RETENTION_DAYS is None:
        print("outcome_histogram kept in full, it backs the model and the observation counts")
    else:
        print(f"pruned {histogram_deleted:,} outcome_histogram bins "
              f"(older than {HISTOGRAM_RETENTION_DAYS} days)")
    reclaim_space()
    print("reclaimed freed space back to the OS (if auto_vacuum is enabled)")


if __name__ == "__main__":
    main()
