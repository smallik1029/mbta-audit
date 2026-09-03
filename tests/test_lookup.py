"""the minimum sample floor, and the clamp that stops a corrected time going negative"""
import pandas as pd

from src.model import train
from src.serve import correction

CELL_COLS = ["route_id", "stop_id", "lead_bucket", "error_bin", "n"]


def _cells(rows):
    return pd.DataFrame(rows, columns=CELL_COLS)


def test_a_thin_stop_cell_drops_out():
    cells = _cells([
        ("Red", "S1", 12, 0, train.MIN_STOP_SAMPLES - 1),
        ("Red", "S2", 12, 0, train.MIN_STOP_SAMPLES),
    ])

    by_stop = train.lookup_from_cells(cells)["by_stop"]

    assert ("Red", "S2", 12) in by_stop.index
    assert ("Red", "S1", 12) not in by_stop.index


def test_the_coarser_levels_are_never_filtered():
    """a dropped stop cell has to land somewhere, so the fallback chain keeps every group"""
    lookup = train.lookup_from_cells(_cells([("Red", "S1", 12, 0, 5)]))

    assert lookup["by_stop"].empty
    assert ("Red", 12) in lookup["by_route"].index
    assert 12 in lookup["by_bucket"].index


def test_a_cell_exactly_on_the_threshold_survives():
    cells = _cells([("Red", "S1", 12, 0, train.MIN_STOP_SAMPLES)])

    assert len(train.lookup_from_cells(cells)["by_stop"]) == 1


def test_thin_cells_do_not_change_the_observation_total():
    """the count the site shows comes from every cell, filtered or not"""
    cells = _cells([
        ("Red", "S1", 12, 0, 3),
        ("Red", "S2", 12, 0, 900),
    ])

    assert int(cells["n"].sum()) == 903
    assert len(train.lookup_from_cells(cells)["by_stop"]) == 1


def _lookup_with_bias(bias_sec, bucket):
    empty_route = pd.DataFrame(columns=["route_id", "lead_bucket", "bias_sec", "n"])
    empty_bucket = pd.DataFrame(columns=["lead_bucket", "bias_sec", "n"])
    return {
        "by_stop": pd.DataFrame([{
            "route_id": "Red", "stop_id": "S1",
            "lead_bucket": bucket, "bias_sec": bias_sec, "n": 1,
        }]),
        "by_route": empty_route,
        "by_bucket": empty_bucket,
    }


def _fake_feed(minutes_out):
    arrival = pd.Timestamp.now(tz="UTC") + pd.Timedelta(minutes=minutes_out)
    item = {
        "attributes": {"arrival_time": arrival.isoformat()},
        "relationships": {
            "route": {"data": {"id": "Red"}},
            "trip": {"data": {"id": "t1"}},
        },
    }
    return [item], []


def test_a_wild_bias_cannot_produce_a_negative_arrival(monkeypatch):
    """4 minutes out, not 3, so elapsed microseconds cannot tip it into another bucket"""
    monkeypatch.setattr(correction, "fetch_live_predictions", lambda *a: _fake_feed(4))

    results = correction.get_corrected_predictions("S1", "Red", _lookup_with_bias(1444.0, 3))

    assert results[0]["confidence"] == "stop"
    assert results[0]["corrected_min"] == 0.0


def test_a_normal_bias_still_shifts_the_time(monkeypatch):
    monkeypatch.setattr(correction, "fetch_live_predictions", lambda *a: _fake_feed(10))

    results = correction.get_corrected_predictions("S1", "Red", _lookup_with_bias(-60.0, 9))

    assert results[0]["confidence"] == "stop"
    assert results[0]["corrected_min"] > results[0]["raw_min"]
