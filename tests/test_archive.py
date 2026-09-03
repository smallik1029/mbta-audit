"""s3 archiving, and the interlock that keeps prune from getting ahead of it"""
from datetime import UTC, date, datetime

import pytest

from src import config, db
from src.process import archive, prune


@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    """a real sqlite file. the cutoffs are string compares in sql, so a stub tests nothing"""
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(db, "_conn", None)
    db.init_schema()
    yield db.connect()
    monkeypatch.setattr(db, "_conn", None)


@pytest.fixture
def bucket(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "S3_ARCHIVE_BUCKET", "mbta-audit-archive")
    monkeypatch.setattr(config, "S3_ARCHIVE_PREFIX", "prediction_outcomes")
    monkeypatch.setattr(archive, "STAGING_DIR", tmp_path / "staging")


class FakeS3:
    """records what would have been uploaded"""

    def __init__(self):
        self.uploads = []

    def upload_file(self, path, bucket, key):
        self.uploads.append((bucket, key))


def _insert_day(conn, day, count, start_id):
    conn.executemany(
        "INSERT INTO prediction_outcomes (id, trip_id, stop_id, route_id, observed_at, "
        "predicted_arrival, actual_arrival, lead_time_sec, error_sec) VALUES (?,?,?,?,?,?,?,?,?)",
        [(start_id + i, f"t{start_id + i}", "S1", "Red",
          f"{day}T{8 + i:02d}:00:00.000+00:00", f"{day}T09:00:00.000+00:00",
          f"{day}T09:00:00.000+00:00", 300.0, 10.0) for i in range(count)],
    )
    conn.commit()


def _count(conn):
    return conn.execute("SELECT COUNT(*) FROM prediction_outcomes").fetchone()[0]


def test_archiving_is_off_until_a_bucket_is_configured(fresh_db, monkeypatch):
    monkeypatch.setattr(config, "S3_ARCHIVE_BUCKET", "")
    assert archive.enabled() is False
    assert archive.run(date(2026, 8, 1)) == {"enabled": False, "days": 0, "rows": 0}


def test_pending_days_starts_at_the_earliest_row(fresh_db, bucket):
    _insert_day(fresh_db, "2026-07-01", 3, 1)
    _insert_day(fresh_db, "2026-07-03", 3, 10)
    assert archive.pending_days(date(2026, 7, 4)) == [
        date(2026, 7, 1), date(2026, 7, 2), date(2026, 7, 3)
    ]


def test_archive_day_uploads_then_moves_the_watermark(fresh_db, bucket):
    _insert_day(fresh_db, "2026-07-01", 4, 1)
    client = FakeS3()

    assert archive.archive_day(date(2026, 7, 1), client) == 4
    assert client.uploads == [
        ("mbta-audit-archive", "prediction_outcomes/2026/07/outcomes-2026-07-01.parquet")
    ]
    assert archive.archived_through() == date(2026, 7, 1)


def test_an_empty_day_advances_without_uploading(fresh_db, bucket):
    client = FakeS3()
    assert archive.archive_day(date(2026, 7, 1), client) == 0
    assert client.uploads == []
    assert archive.archived_through() == date(2026, 7, 1)


def test_prune_holds_everything_until_something_reaches_s3(fresh_db, bucket):
    _insert_day(fresh_db, "2026-07-01", 3, 1)
    result = prune.prune_outcomes(datetime(2026, 8, 20, tzinfo=UTC))

    assert result == {"prediction_outcomes": 0, "held_for_archive": True}
    assert _count(fresh_db) == 3


def test_prune_stops_at_the_archived_boundary(fresh_db, bucket):
    _insert_day(fresh_db, "2026-07-01", 3, 1)
    _insert_day(fresh_db, "2026-07-02", 3, 10)
    _insert_day(fresh_db, "2026-07-10", 3, 20)
    client = FakeS3()

    archive.run(date(2026, 7, 3), client)
    assert archive.archived_through() == date(2026, 7, 2)

    result = prune.prune_outcomes(datetime(2026, 8, 20, tzinfo=UTC))

    assert result["held_for_archive"] is False
    assert result["prediction_outcomes"] == 6
    assert _count(fresh_db) == 3


def test_prune_ignores_the_interlock_when_archiving_is_off(fresh_db, monkeypatch):
    monkeypatch.setattr(config, "S3_ARCHIVE_BUCKET", "")
    _insert_day(fresh_db, "2026-07-01", 3, 1)
    _insert_day(fresh_db, "2026-08-19", 3, 10)

    result = prune.prune_outcomes(datetime(2026, 8, 20, tzinfo=UTC))

    assert result["prediction_outcomes"] == 3
    assert _count(fresh_db) == 3


def test_the_histogram_survives_an_outcome_prune(fresh_db, monkeypatch):
    monkeypatch.setattr(config, "S3_ARCHIVE_BUCKET", "")
    _insert_day(fresh_db, "2026-07-01", 3, 1)
    fresh_db.execute(
        "INSERT INTO outcome_histogram (route_id, stop_id, lead_bucket, month, error_bin, n) "
        "VALUES (?,?,?,?,?,?)",
        ("Red", "S1", 12, "2026-01", 3, 999),
    )
    fresh_db.commit()

    prune.prune_outcomes(datetime(2026, 8, 20, tzinfo=UTC))
    prune.prune_histogram(datetime(2026, 8, 20, tzinfo=UTC))

    kept = fresh_db.execute("SELECT SUM(n) FROM outcome_histogram").fetchone()[0]
    assert kept == 999
