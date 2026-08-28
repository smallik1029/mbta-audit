"""the pre-aggregated error histogram"""
import numpy as np
import pandas as pd
import pytest

from src import config, db
from src.model.features import add_lead_bucket
from src.process import histogram


@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    """a real sqlite file. the folding is sql, so an in-memory stub would test nothing"""
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(db, "_conn", None)
    db.init_schema()
    yield db.connect()
    monkeypatch.setattr(db, "_conn", None)


def _insert(conn, rows, start_id=1):
    conn.executemany(
        "INSERT INTO prediction_outcomes (id, trip_id, stop_id, route_id, observed_at, "
        "predicted_arrival, actual_arrival, lead_time_sec, error_sec) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        [(start_id + i, f"t{start_id + i}", stop, route, obs, obs, obs, lead, err)
         for i, (route, stop, obs, lead, err) in enumerate(rows)],
    )
    conn.commit()


def _row(route="Red", stop="S1", obs="2026-08-21T12:00:00+00:00", lead=600.0, err=-42.0):
    return (route, stop, obs, lead, err)


def test_sql_bucketing_matches_the_models_bucketing(fresh_db):
    """one rule, two implementations. checked across the boundaries"""
    leads = [0.0, 1.0, 179.9, 180.0, 180.1, 540.0, 1799.0, 1800.0, 1800.1, 7200.0]
    _insert(fresh_db, [_row(lead=x, err=1.0) for x in leads])
    histogram.update()

    from_sql = dict(fresh_db.execute(
        "SELECT lead_bucket, SUM(n) FROM outcome_histogram GROUP BY lead_bucket").fetchall())
    from_python = (add_lead_bucket(pd.DataFrame({"lead_time_sec": leads}))["lead_bucket"]
                   .value_counts().to_dict())
    assert from_sql == from_python


def test_folding_is_incremental_and_never_double_counts(fresh_db):
    _insert(fresh_db, [_row(err=-10.0)] * 3, start_id=1)
    assert histogram.update() == 3

    assert histogram.update() == 0
    assert fresh_db.execute("SELECT SUM(n) FROM outcome_histogram").fetchone()[0] == 3

    _insert(fresh_db, [_row(err=-10.0)] * 2, start_id=100)
    assert histogram.update() == 2
    assert fresh_db.execute("SELECT SUM(n) FROM outcome_histogram").fetchone()[0] == 5


def test_batching_folds_everything_when_input_exceeds_one_batch(fresh_db, monkeypatch):
    """a first run on a full database is the case that must not load everything at once"""
    monkeypatch.setattr(histogram, "BATCH_ROWS", 10)
    _insert(fresh_db, [_row(err=float(i % 7)) for i in range(95)])
    assert histogram.update() == 95
    assert fresh_db.execute("SELECT SUM(n) FROM outcome_histogram").fetchone()[0] == 95


def test_medians_and_counts_match_the_raw_rows(fresh_db):
    """the claim the whole approach rests on"""
    rng = np.random.default_rng(0)
    rows = []
    for route, stop, n in (("Red", "S1", 101), ("Blue", "S2", 100), ("Orange", "S3", 40)):
        for err in rng.normal(-60, 90, n).round(0):
            rows.append(_row(route=route, stop=stop, err=float(err)))
    _insert(fresh_db, rows)
    histogram.update()

    raw = pd.read_sql_query("SELECT route_id, stop_id, error_sec FROM prediction_outcomes",
                            fresh_db)
    raw["lead_bucket"] = 9
    keys = ["route_id", "stop_id", "lead_bucket"]
    truth = raw.groupby(keys)["error_sec"].median()

    cells = histogram.load_cells(("Red", "Orange", "Blue"))
    est = histogram.medians_from_cells(cells, keys)
    counts = histogram.counts_from_cells(cells, keys)

    pd.testing.assert_series_equal(truth.sort_index(), est.sort_index(),
                                   check_names=False, check_dtype=False)
    pd.testing.assert_series_equal(raw.groupby(keys).size().sort_index(),
                                   counts.sort_index(),
                                   check_names=False, check_dtype=False)


def test_pruning_drops_expired_months_only(fresh_db):
    _insert(fresh_db, [_row(obs="2026-01-15T00:00:00+00:00")], start_id=1)
    _insert(fresh_db, [_row(obs="2026-08-15T00:00:00+00:00")], start_id=2)
    histogram.update()
    assert fresh_db.execute("SELECT SUM(n) FROM outcome_histogram").fetchone()[0] == 2

    histogram.prune_months("2026-08")
    remaining = fresh_db.execute("SELECT month, SUM(n) FROM outcome_histogram GROUP BY month")
    assert [tuple(r) for r in remaining.fetchall()] == [("2026-08", 1)]


def test_rows_without_a_route_are_skipped(fresh_db):
    """route_id is nullable, and an unrouted row belongs to no cell"""
    _insert(fresh_db, [_row(route=None), _row(route="Red")])
    histogram.update()
    assert fresh_db.execute("SELECT SUM(n) FROM outcome_histogram").fetchone()[0] == 1
