"""matching predictions to arrivals"""
import pandas as pd
import pytest

from src import config, db
from src.collect.run_collector import latest_observed_at
from src.process.match import (
    BATCH_ARRIVALS,
    build_outcomes,
    exclude_cancelled,
    get_watermark,
    load_joined_predictions,
    run_batches,
    seed_watermark,
)


def _predictions(rows):
    df = pd.DataFrame(rows, columns=[
        "observed_at", "trip_id", "stop_id", "predicted_arrival", "schedule_relationship",
    ])
    df["observed_at"] = pd.to_datetime(df["observed_at"], utc=True)
    df["predicted_arrival"] = pd.to_datetime(df["predicted_arrival"], utc=True)
    return df


def _arrivals(rows):
    df = pd.DataFrame(rows, columns=["trip_id", "stop_id", "route_id", "direction_id", "actual_arrival"])
    df["actual_arrival"] = pd.to_datetime(df["actual_arrival"], utc=True)
    return df


def _joined(arrivals_rows, prediction_rows):
    """build_outcomes takes an already-joined frame, so fixtures join the same way"""
    arrivals = _arrivals(arrivals_rows)
    preds = _predictions(prediction_rows)
    return preds.merge(arrivals, on=["trip_id", "stop_id"], how="inner")


def test_exclude_cancelled_drops_cancelled_and_skipped_only():
    preds = _predictions([
        ("2026-08-21T16:00:00Z", "T1", "S1", "2026-08-21T16:10:00Z", "CANCELLED"),
        ("2026-08-21T16:00:00Z", "T2", "S1", "2026-08-21T16:10:00Z", "SKIPPED"),
        ("2026-08-21T16:00:00Z", "T3", "S1", "2026-08-21T16:10:00Z", None),
    ])
    result = exclude_cancelled(preds)
    assert list(result["trip_id"]) == ["T3"]


def test_cancelled_trip_produces_no_outcome_rows():
    """predictions exist, arrival never happens"""
    preds = [
        ("2026-08-21T16:00:00Z", "CANCELLED_TRIP", "S1", "2026-08-21T16:10:00Z", "CANCELLED"),
        ("2026-08-21T16:05:00Z", "CANCELLED_TRIP", "S1", "2026-08-21T16:10:00Z", "CANCELLED"),
    ]
    joined = _joined([], preds)

    outcomes = build_outcomes(joined)
    assert len(outcomes) == 0


def test_lead_time_and_error_computed_correctly():
    joined = _joined(
        [("T1", "S1", "Red", 0, "2026-08-21T16:10:00Z")],
        [
            ("2026-08-21T16:05:00Z", "T1", "S1", "2026-08-21T16:11:00Z", None),
        ],
    )

    outcomes = build_outcomes(joined)
    assert len(outcomes) == 1
    row = outcomes.iloc[0]
    assert row["lead_time_sec"] == 300
    assert row["error_sec"] == 60


def test_predictions_observed_after_arrival_are_excluded():
    joined = _joined(
        [("T1", "S1", "Red", 0, "2026-08-21T16:10:00Z")],
        [
            ("2026-08-21T16:12:00Z", "T1", "S1", "2026-08-21T16:11:00Z", None),
        ],
    )

    outcomes = build_outcomes(joined)
    assert len(outcomes) == 0


def test_unmatched_trip_stop_pairs_are_dropped():
    """a prediction for a trip/stop with no recorded arrival contributes nothing"""
    joined = _joined(
        [("T1", "S1", "Red", 0, "2026-08-21T16:10:00Z")],
        [
            ("2026-08-21T16:00:00Z", "T2", "S1", "2026-08-21T16:10:00Z", None),
        ],
    )

    outcomes = build_outcomes(joined)
    assert len(outcomes) == 0


@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(db, "_conn", None)
    db.init_schema()
    yield db.connect()
    monkeypatch.setattr(db, "_conn", None)


def _seed_day(conn, service_date, trip, arrival, observed):
    conn.execute(
        "INSERT INTO actual_arrivals (trip_id, stop_id, route_id, direction_id, "
        "vehicle_id, actual_arrival, service_date) VALUES (?,?,?,?,?,?,?)",
        (trip, "S1", "Red", 0, "V1", arrival, service_date),
    )
    conn.execute(
        "INSERT INTO prediction_snapshots (observed_at, trip_id, stop_id, route_id, "
        "predicted_arrival) VALUES (?,?,?,?,?)",
        (observed, trip, "S1", "Red", arrival),
    )
    conn.commit()


def test_backlog_is_batched_and_is_idempotent(fresh_db, monkeypatch):
    """bounded batches, and a re-run adds nothing, so an interrupted catch-up resumes"""
    monkeypatch.setattr("src.process.match.BATCH_ARRIVALS", 2)
    for i, day in enumerate(["2026-08-27", "2026-08-28", "2026-08-29", "2026-08-30", "2026-08-31"]):
        _seed_day(fresh_db, day, f"T{i}", f"{day}T16:10:00+00:00", f"{day}T16:00:00+00:00")

    total_new, first_id = run_batches()
    assert total_new == 5
    assert first_id == 0
    assert fresh_db.execute("SELECT COUNT(*) FROM prediction_outcomes").fetchone()[0] == 5

    rerun_new, _ = run_batches()
    assert rerun_new == 0
    assert fresh_db.execute("SELECT COUNT(*) FROM prediction_outcomes").fetchone()[0] == 5


def test_watermark_persists_so_an_interrupted_run_resumes(fresh_db, monkeypatch):
    monkeypatch.setattr("src.process.match.BATCH_ARRIVALS", 2)
    for i, day in enumerate(["2026-08-27", "2026-08-28", "2026-08-29", "2026-08-30"]):
        _seed_day(fresh_db, day, f"T{i}", f"{day}T16:10:00+00:00", f"{day}T16:00:00+00:00")

    run_batches()
    assert get_watermark() == 4

    _seed_day(fresh_db, "2026-09-01", "T9", "2026-09-01T16:10:00+00:00", "2026-09-01T16:00:00+00:00")
    total_new, _ = run_batches()
    assert total_new == 1


def test_seed_watermark_is_zero_on_an_empty_outcomes_table(fresh_db):
    _seed_day(fresh_db, "2026-08-27", "T1", "2026-08-27T16:10:00+00:00", "2026-08-27T16:00:00+00:00")
    assert seed_watermark() == 0


def test_batch_size_is_a_fixed_arrival_count(fresh_db):
    assert isinstance(BATCH_ARRIVALS, int) and BATCH_ARRIVALS > 0


def test_latest_observed_at_uses_the_newest_across_both_feeds(fresh_db):
    """stays per-table so each MAX uses its index, and still returns the true newest"""
    fresh_db.execute(
        "INSERT INTO prediction_snapshots (observed_at, trip_id, stop_id, route_id) "
        "VALUES ('2026-08-31T10:00:00+00:00','T1','S1','Red')"
    )
    fresh_db.execute(
        "INSERT INTO vehicle_snapshots (observed_at, vehicle_id) "
        "VALUES ('2026-08-31T12:00:00+00:00','V1')"
    )
    fresh_db.commit()
    assert latest_observed_at() == "2026-08-31T12:00:00+00:00"


def test_latest_observed_at_is_none_when_both_feeds_are_empty(fresh_db):
    assert latest_observed_at() is None


def test_stale_prediction_from_an_earlier_day_is_not_matched(fresh_db):
    """trip_id repeats daily, so an unbounded join invents multi-day lead times"""
    fresh_db.execute(
        "INSERT INTO actual_arrivals (trip_id, stop_id, route_id, direction_id, "
        "vehicle_id, actual_arrival, service_date) VALUES ('T1','S1','Red',0,'V1',?,?)",
        ("2026-08-28T16:10:00+00:00", "2026-08-28"),
    )
    fresh_db.executemany(
        "INSERT INTO prediction_snapshots (observed_at, trip_id, stop_id, route_id, "
        "predicted_arrival) VALUES (?,'T1','S1','Red','2026-08-28T16:10:00+00:00')",
        [("2026-08-28T16:00:00+00:00",), ("2026-08-25T16:00:00+00:00",)],
    )
    fresh_db.commit()

    joined = load_joined_predictions(0, 1)
    assert list(joined["observed_at"].astype(str)) == ["2026-08-28 16:00:00+00:00"]

    run_batches()
    leads = [r[0] for r in fresh_db.execute("SELECT lead_time_sec FROM prediction_outcomes")]
    assert leads == [600.0]
