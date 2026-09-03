"""copy expiring outcome rows to s3 before prune deletes them"""
from datetime import date, timedelta

import pandas as pd

from src import config, db

ARCHIVED_THROUGH_KEY = "outcomes_archived_through"
STAGING_DIR = config.ROOT / "data" / "_archive"


def enabled() -> bool:
    return bool(config.S3_ARCHIVE_BUCKET)


def _client():
    try:
        import boto3
    except ImportError:
        raise SystemExit(
            "S3_ARCHIVE_BUCKET is set but boto3 is not installed.\n"
            "  pip install --user boto3   or clear S3_ARCHIVE_BUCKET to turn archiving off"
        ) from None
    return boto3.client("s3")


def archived_through() -> date | None:
    """the last day copied to s3 in full"""
    row = db.connect().execute(
        "SELECT value FROM pipeline_state WHERE key = ?", (ARCHIVED_THROUGH_KEY,)
    ).fetchone()
    return date.fromisoformat(row["value"]) if row else None


def _remember(day: date) -> None:
    conn = db.connect()
    conn.execute(
        "INSERT INTO pipeline_state (key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (ARCHIVED_THROUGH_KEY, day.isoformat()),
    )
    conn.commit()


def _earliest_day() -> date | None:
    stamp = db.connect().execute("SELECT MIN(observed_at) FROM prediction_outcomes").fetchone()[0]
    return pd.Timestamp(stamp).date() if stamp else None


def pending_days(cutoff_day: date) -> list[date]:
    """whole days aged past retention that s3 does not have yet"""
    last = archived_through()
    day = last + timedelta(days=1) if last else _earliest_day()
    if day is None:
        return []
    days = []
    while day < cutoff_day:
        days.append(day)
        day += timedelta(days=1)
    return days


def object_key(day: date) -> str:
    return (f"{config.S3_ARCHIVE_PREFIX}/{day.year:04d}/{day.month:02d}/"
            f"outcomes-{day.isoformat()}.parquet")


def _day_rows(day: date) -> pd.DataFrame:
    start = f"{day.isoformat()}T00:00:00.000+00:00"
    end = f"{(day + timedelta(days=1)).isoformat()}T00:00:00.000+00:00"
    return pd.read_sql_query(
        "SELECT * FROM prediction_outcomes WHERE observed_at >= ? AND observed_at < ? ORDER BY id",
        db.connect(),
        params=(start, end),
    )


def archive_day(day: date, client=None) -> int:
    """one parquet file per day. the watermark only moves after the upload returns"""
    rows = _day_rows(day)
    if len(rows):
        STAGING_DIR.mkdir(parents=True, exist_ok=True)
        path = STAGING_DIR / f"outcomes-{day.isoformat()}.parquet"
        rows.to_parquet(path, index=False, compression="zstd")
        (client or _client()).upload_file(str(path), config.S3_ARCHIVE_BUCKET, object_key(day))
        path.unlink(missing_ok=True)
    _remember(day)
    return len(rows)


def run(cutoff_day: date, client=None) -> dict:
    """archive every whole day that has aged past the outcome retention window"""
    if not enabled():
        return {"enabled": False, "days": 0, "rows": 0}
    days = pending_days(cutoff_day)
    total = 0
    for day in days:
        total += archive_day(day, client)
    return {"enabled": True, "days": len(days), "rows": total}
