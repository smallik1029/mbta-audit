"""delete rows past their retention window and give the space back"""
from datetime import UTC, datetime, timedelta

from src import db

RAW_RETENTION_DAYS = 14
OUTCOMES_RETENTION_DAYS = 182


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
    now = now or datetime.now(UTC)
    cutoff = (now - timedelta(days=OUTCOMES_RETENTION_DAYS)).isoformat()
    conn = db.connect()
    cur = conn.execute("DELETE FROM prediction_outcomes WHERE observed_at < ?", (cutoff,))
    conn.commit()
    return {"prediction_outcomes": cur.rowcount}


def reclaim_space(pages: int = 20000) -> None:
    """hand freed pages back to the os. only does anything if auto_vacuum is on"""
    conn = db.connect()
    conn.execute(f"PRAGMA incremental_vacuum({pages})")
    conn.commit()


def main() -> None:
    db.init_schema()
    raw_deleted = prune_raw()
    outcomes_deleted = prune_outcomes()
    print(f"pruned {raw_deleted['prediction_snapshots']:,} prediction_snapshots, "
          f"{raw_deleted['vehicle_snapshots']:,} vehicle_snapshots rows "
          f"(older than {RAW_RETENTION_DAYS} days)")
    print(f"pruned {outcomes_deleted['prediction_outcomes']:,} prediction_outcomes rows "
          f"(older than {OUTCOMES_RETENTION_DAYS} days / ~6 months)")
    reclaim_space()
    print("reclaimed freed space back to the OS (if auto_vacuum is enabled)")


if __name__ == "__main__":
    main()
