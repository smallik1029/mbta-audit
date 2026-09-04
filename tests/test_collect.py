"""the dedup cache, and what it does when a write is refused"""
import sqlite3

import pytest

from src import db
from src.collect.predictions import PredictionCollector
from src.collect.vehicles import VehicleCollector

OBSERVED_AT = "2026-09-05T08:00:00.000+00:00"


def _prediction(trip: str, stop: str, arrival: str) -> dict:
    return {
        "id": f"{trip}-{stop}",
        "attributes": {
            "arrival_time": arrival,
            "departure_time": None,
            "schedule_relationship": None,
            "direction_id": 0,
            "stop_sequence": 1,
        },
        "relationships": {
            "trip": {"data": {"id": trip}},
            "stop": {"data": {"id": stop}},
            "route": {"data": {"id": "Red"}},
            "vehicle": {"data": {"id": "v1"}},
        },
    }


def _vehicle(vehicle_id: str, status: str) -> dict:
    return {
        "id": vehicle_id,
        "attributes": {
            "current_status": status,
            "current_stop_sequence": 1,
            "direction_id": 0,
            "latitude": 42.0,
            "longitude": -71.0,
            "updated_at": OBSERVED_AT,
        },
        "relationships": {
            "trip": {"data": {"id": "t1"}},
            "stop": {"data": {"id": "s1"}},
            "route": {"data": {"id": "Red"}},
        },
    }


class _Writes:
    """stands in for insert_many so a locked database can be simulated"""

    def __init__(self):
        self.batches = []
        self.refuse = False

    def __call__(self, table, columns, rows):
        if self.refuse:
            raise sqlite3.OperationalError("database is locked")
        self.batches.append(rows)
        return len(rows)


def _wire(collector, items, monkeypatch):
    writes = _Writes()
    monkeypatch.setattr(db, "insert_many", writes)
    monkeypatch.setattr(collector.feed, "fetch", lambda: (items, OBSERVED_AT))
    return writes


def test_a_refused_write_leaves_the_cache_alone_so_the_next_poll_retries(monkeypatch):
    collector = PredictionCollector()
    writes = _wire(collector, [_prediction("t1", "s1", "2026-09-05T08:05:00-04:00")], monkeypatch)

    writes.refuse = True
    with pytest.raises(sqlite3.OperationalError):
        collector.poll_once()
    assert collector.last_seen == {}

    writes.refuse = False
    assert collector.poll_once() == 1
    assert len(writes.batches) == 1
    assert collector.last_seen != {}


def test_an_unchanged_prediction_is_written_once(monkeypatch):
    collector = PredictionCollector()
    _wire(collector, [_prediction("t1", "s1", "2026-09-05T08:05:00-04:00")], monkeypatch)

    assert collector.poll_once() == 1
    assert collector.poll_once() == 0
    assert collector.total_seen == 2


def test_the_same_key_twice_in_one_poll_is_still_written_once(monkeypatch):
    collector = PredictionCollector()
    same = _prediction("t1", "s1", "2026-09-05T08:05:00-04:00")
    writes = _wire(collector, [same, same], monkeypatch)

    assert collector.poll_once() == 1
    assert len(writes.batches[0]) == 1


def test_a_revised_prediction_is_written_again(monkeypatch):
    collector = PredictionCollector()
    writes = _wire(collector, [_prediction("t1", "s1", "2026-09-05T08:05:00-04:00")], monkeypatch)
    assert collector.poll_once() == 1

    monkeypatch.setattr(
        collector.feed, "fetch",
        lambda: ([_prediction("t1", "s1", "2026-09-05T08:07:00-04:00")], OBSERVED_AT),
    )
    assert collector.poll_once() == 1
    assert len(writes.batches) == 2


def test_the_vehicle_cache_also_survives_a_refused_write(monkeypatch):
    collector = VehicleCollector()
    writes = _wire(collector, [_vehicle("v1", "STOPPED_AT")], monkeypatch)

    writes.refuse = True
    with pytest.raises(sqlite3.OperationalError):
        collector.poll_once()
    assert collector.last_seen == {}

    writes.refuse = False
    assert collector.poll_once() == 1


class _RefusingConn:
    def __init__(self):
        self.rolled_back = False

    def executemany(self, sql, rows):
        return None

    def commit(self):
        raise sqlite3.OperationalError("database is locked")

    def rollback(self):
        self.rolled_back = True


def test_a_refused_commit_is_rolled_back(monkeypatch):
    conn = _RefusingConn()
    monkeypatch.setattr(db, "connect", lambda: conn)

    with pytest.raises(sqlite3.OperationalError):
        db.insert_many("prediction_snapshots", ["observed_at"], [(OBSERVED_AT,)])
    assert conn.rolled_back
