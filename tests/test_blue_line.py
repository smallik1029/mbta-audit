"""Tests for the Blue Line correction's own logic (correct(), summarize())"""
import pandas as pd

from src.model.blue_line import correct, summarize


def _lookup(by_stop_rows, by_bucket_rows):
    """Matches fit_lookup()'s actual output shape: a Series with a"""
    by_stop_df = pd.DataFrame(by_stop_rows, columns=["route_id", "stop_id", "lead_bucket", "bias_sec"])
    by_stop = by_stop_df.set_index(["route_id", "stop_id", "lead_bucket"])["bias_sec"]
    by_bucket = pd.DataFrame(by_bucket_rows, columns=["lead_bucket", "bias_sec"]).set_index(
        "lead_bucket"
    )["bias_sec"]
    return {"by_stop": by_stop, "by_bucket": by_bucket}


def test_correct_uses_stop_level_when_available():
    test = pd.DataFrame({
        "route_id": ["Blue"], "stop_id": ["70060"], "lead_bucket": [9], "error_sec": [100.0],
    })
    lookup = _lookup(
        by_stop_rows=[("Blue", "70060", 9, -40.0)],
        by_bucket_rows=[(9, -999.0)],
    )
    result = correct(test, lookup)
    assert result.iloc[0] == 100.0 - (-40.0)


def test_correct_falls_back_to_bucket_level():
    test = pd.DataFrame({
        "route_id": ["Blue"], "stop_id": ["UNSEEN_STOP"], "lead_bucket": [9], "error_sec": [50.0],
    })
    lookup = _lookup(
        by_stop_rows=[("Blue", "70060", 9, -40.0)],
        by_bucket_rows=[(9, -20.0)],
    )
    result = correct(test, lookup)
    assert result.iloc[0] == 50.0 - (-20.0)


def test_summarize_drops_sparse_buckets():
    """Spec-consistent with the main pipeline: don't trust a headline number"""
    test = pd.DataFrame({
        "lead_time_sec": [900] * 5,
        "error_sec": [30.0] * 5,
    })
    corrected_error = pd.Series([10.0] * 5)
    table = summarize(test, corrected_error)
    assert len(table) == 0
