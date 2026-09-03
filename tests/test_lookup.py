"""the minimum sample floor on stop cells"""
import pandas as pd

from src.model import train

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
