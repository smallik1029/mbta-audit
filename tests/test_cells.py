"""the histogram's group keys have to keep matching the outcome table's"""
import pandas as pd

from src.model.features import apply_correction
from src.model.train import lookup_from_cells
from src.process import histogram

STOP_A = "place-pktrm"
STOP_B = "place-dwnxg"


def _cells() -> pd.DataFrame:
    """two stops on one route with deliberately different medians, keys unique like load_cells"""
    return pd.DataFrame({
        "route_id": ["Red", "Red"],
        "stop_id": [STOP_A, STOP_B],
        "lead_bucket": [6, 6],
        "error_bin": [20, 100],
        "n": [300, 300],
    })


def _outcome(stop_id: str) -> pd.DataFrame:
    return pd.DataFrame({
        "route_id": ["Red"],
        "stop_id": [stop_id],
        "lead_bucket": [6],
        "error_sec": [50.0],
    })


def test_a_stop_level_correction_actually_matches():
    lookup = lookup_from_cells(_cells())
    corrected, missing_stop, missing_route = apply_correction(_outcome(STOP_A), lookup)

    assert missing_stop == 0
    assert missing_route == 0
    assert corrected.iloc[0] == 30.0


def test_a_stop_level_correction_matches_after_the_cells_are_compacted():
    lookup = lookup_from_cells(histogram.compact(_cells()))
    corrected, missing_stop, missing_route = apply_correction(_outcome(STOP_A), lookup)

    assert missing_stop == 0
    assert missing_route == 0
    assert corrected.iloc[0] == 30.0


def test_compacting_does_not_change_any_median():
    plain = lookup_from_cells(_cells())
    narrow = lookup_from_cells(histogram.compact(_cells()))

    for key in ("by_stop", "by_route", "by_bucket"):
        assert plain[key].to_dict() == narrow[key].to_dict()


def test_the_stop_median_is_used_and_not_the_route_median():
    lookup = lookup_from_cells(_cells())
    a, _, _ = apply_correction(_outcome(STOP_A), lookup)
    b, _, _ = apply_correction(_outcome(STOP_B), lookup)

    assert a.iloc[0] == 30.0
    assert b.iloc[0] == -50.0
    assert lookup["by_route"].loc[("Red", 6)] == 60.0


def test_an_unknown_stop_falls_back_to_the_route_median():
    lookup = lookup_from_cells(_cells())
    corrected, missing_stop, missing_route = apply_correction(_outcome("place-nowhere"), lookup)

    assert missing_stop == 1
    assert missing_route == 0
    assert corrected.iloc[0] == -10.0


def test_subtracting_a_window_leaves_the_remaining_counts_intact():
    total = _cells()
    part = pd.DataFrame({
        "route_id": ["Red"],
        "stop_id": [STOP_A],
        "lead_bucket": [6],
        "error_bin": [20],
        "n": [100],
    })
    left = histogram.subtract_cells(total, part)

    kept = left[(left["stop_id"] == STOP_A) & (left["error_bin"] == 20)]["n"].sum()
    assert kept == 200
    assert left[left["stop_id"] == STOP_B]["n"].sum() == 300


def test_a_bin_emptied_by_subtraction_disappears():
    total = pd.DataFrame({
        "route_id": ["Red"],
        "stop_id": [STOP_A],
        "lead_bucket": [6],
        "error_bin": [20],
        "n": [40],
    })
    part = total.copy()

    assert histogram.subtract_cells(total, part).empty
