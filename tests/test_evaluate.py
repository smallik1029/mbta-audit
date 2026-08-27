"""correction and evaluation logic"""
import pandas as pd

from src.model.evaluate import MIN_SAMPLES_FOR_HEADLINE, correct, summarize


def _lookup(by_stop_rows, by_route_rows, by_bucket_rows):
    """same shape fit_lookup returns: a Series with a MultiIndex"""
    by_stop = pd.DataFrame(
        by_stop_rows, columns=["route_id", "stop_id", "lead_bucket", "bias_sec"]
    ).set_index(["route_id", "stop_id", "lead_bucket"])["bias_sec"]
    by_route = pd.DataFrame(
        by_route_rows, columns=["route_id", "lead_bucket", "bias_sec"]
    ).set_index(["route_id", "lead_bucket"])["bias_sec"]
    by_bucket = pd.DataFrame(
        by_bucket_rows, columns=["lead_bucket", "bias_sec"]
    ).set_index("lead_bucket")["bias_sec"]
    return {"by_stop": by_stop, "by_route": by_route, "by_bucket": by_bucket}


def _one_row(stop_id, error_sec, route_id="Blue"):
    return pd.DataFrame({
        "route_id": [route_id], "stop_id": [stop_id],
        "lead_bucket": [9], "error_sec": [error_sec],
    })


def test_correct_uses_stop_level_when_available():
    lookup = _lookup(
        by_stop_rows=[("Blue", "70060", 9, -40.0)],
        by_route_rows=[("Blue", 9, -888.0)],
        by_bucket_rows=[(9, -999.0)],
    )
    corrected, n_missing_stop, n_missing_route = correct(_one_row("70060", 100.0), lookup)
    assert corrected.iloc[0] == 100.0 - (-40.0)
    assert n_missing_stop == 0
    assert n_missing_route == 0


def test_correct_falls_back_to_route_level():
    lookup = _lookup(
        by_stop_rows=[("Blue", "70060", 9, -40.0)],
        by_route_rows=[("Blue", 9, -30.0)],
        by_bucket_rows=[(9, -999.0)],
    )
    corrected, n_missing_stop, n_missing_route = correct(_one_row("UNSEEN_STOP", 50.0), lookup)
    assert corrected.iloc[0] == 50.0 - (-30.0)
    assert n_missing_stop == 1
    assert n_missing_route == 0


def test_correct_falls_back_to_bucket_level_when_route_also_misses():
    lookup = _lookup(
        by_stop_rows=[("Blue", "70060", 9, -40.0)],
        by_route_rows=[("Orange", 9, -30.0)],
        by_bucket_rows=[(9, -20.0)],
    )
    corrected, n_missing_stop, n_missing_route = correct(_one_row("UNSEEN_STOP", 50.0), lookup)
    assert corrected.iloc[0] == 50.0 - (-20.0)
    assert n_missing_stop == 1
    assert n_missing_route == 1


def test_summarize_drops_sparse_buckets():
    """do not publish a headline number backed by a handful of rows"""
    n = MIN_SAMPLES_FOR_HEADLINE - 1
    test = pd.DataFrame({"lead_time_sec": [900] * n, "error_sec": [30.0] * n})
    assert len(summarize(test, pd.Series([10.0] * n))) == 0


def test_summarize_keeps_buckets_at_the_threshold():
    """the boundary is inclusive. catches an off-by-one"""
    n = MIN_SAMPLES_FOR_HEADLINE
    test = pd.DataFrame({"lead_time_sec": [900] * n, "error_sec": [30.0] * n})
    table = summarize(test, pd.Series([10.0] * n))
    assert len(table) == 1
    assert table.iloc[0]["n"] == n
