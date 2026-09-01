"""the do-no-harm guard"""
import numpy as np
import pandas as pd
import pytest

from src import config, db
from src.model.features import LEAD_BUCKET_MAX_MIN, LEAD_BUCKET_WIDTH_MIN
from src.model.train import (
    GUARD_CHUNKS,
    GUARD_MIN_SAMPLES,
    fit_lookup,
    guard_scores,
    select_harmful_groups,
    suppress_groups,
)
from src.process import histogram

TOP_BUCKET = LEAD_BUCKET_MAX_MIN - (LEAD_BUCKET_MAX_MIN % LEAD_BUCKET_WIDTH_MIN)


def _window(errors, route="Red", bucket=9, stop="S1"):
    """a fit window in time order, so the chunking sees the errors in the order given"""
    ts = pd.date_range("2026-08-21T00:00:00Z", periods=len(errors), freq="1min", tz="UTC")
    return pd.DataFrame({
        "route_id": route,
        "stop_id": stop,
        "lead_bucket": bucket,
        "observed_at": ts,
        "error_sec": np.asarray(errors, dtype=float),
    })


def _consistent(value, n=GUARD_MIN_SAMPLES * GUARD_CHUNKS * 2):
    """a bias that holds steady across every chunk, with a little jitter so the median is well defined"""
    return np.linspace(value - 2.0, value + 2.0, n)


def _shifting_bias(values, n=GUARD_MIN_SAMPLES * GUARD_CHUNKS * 2):
    """a different bias in every chunk, so a fit that excludes any one chunk mispredicts it"""
    per = n // len(values)
    return np.concatenate([np.full(per, v) for v in values])


def _regime_change(before, after, n=GUARD_MIN_SAMPLES * GUARD_CHUNKS * 2):
    """a bias that holds for most of the window then reverses in the final chunk"""
    split = n * (GUARD_CHUNKS - 1) // GUARD_CHUNKS
    return np.r_[np.full(split, before), np.full(n - split, after)]


@pytest.fixture
def guard_db(tmp_path, monkeypatch):
    """the guard judges the histogram lookup, so it needs a real db"""
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "guard.db")
    monkeypatch.setattr(db, "_conn", None)
    db.init_schema()
    yield db.connect()
    monkeypatch.setattr(db, "_conn", None)


def _seed(window):
    """write a window into prediction_outcomes and fold it in, so the guard sees it as production data"""
    conn = db.connect()
    conn.executemany(
        "INSERT INTO prediction_outcomes (trip_id, stop_id, route_id, observed_at, "
        "predicted_arrival, actual_arrival, lead_time_sec, error_sec) VALUES (?,?,?,?,?,?,?,?)",
        [
            (f"{r.route_id}-{r.lead_bucket}-{i}", r.stop_id, r.route_id, r.observed_at.isoformat(),
             r.observed_at.isoformat(), r.observed_at.isoformat(),
             2400.0 if r.lead_bucket >= TOP_BUCKET else (r.lead_bucket + 1) * 60.0,
             r.error_sec)
            for i, r in enumerate(window.itertuples())
        ],
    )
    conn.commit()
    histogram.update()
    return window


def test_helpful_correction_is_not_suppressed(guard_db):
    assert select_harmful_groups(_seed(_window(_consistent(-60.0)))) == set()


def test_correction_harmful_in_every_fold_is_suppressed(guard_db):
    """harm that reproduces across folds is what this catches"""
    window = _seed(_window(_shifting_bias([-200.0, -100.0, 100.0, 200.0]), route="Orange"))
    assert ("Orange", 9) in select_harmful_groups(window)


def test_correction_that_stops_working_is_suppressed(guard_db):
    window = _seed(_window(_regime_change(-100.0, 100.0), route="Orange"))
    assert ("Orange", 9) in select_harmful_groups(window)


def test_top_bucket_is_exempt_even_when_it_scores_badly(guard_db):
    """the catch-all spans ~90 min where the others span 3, so it is not like for like"""
    window = _seed(_window(_shifting_bias([-200.0, -100.0, 100.0, 200.0]),
                           route="Red", bucket=TOP_BUCKET))
    scores = guard_scores(window)
    assert scores.loc[("Red", TOP_BUCKET), "improvement_pct"] < 0
    assert select_harmful_groups(window) == set()


def test_thin_groups_are_left_alone_not_suppressed(guard_db):
    """too little data means cannot judge, not harmful"""
    thin = _shifting_bias([-200.0, -100.0, 100.0, 200.0], n=40)
    assert select_harmful_groups(_seed(_window(thin))) == set()


def test_guard_only_ever_reads_the_window_it_is_given(guard_db):
    """deciding the model on the data used to report it would inflate the claim"""
    window = _seed(_window(_consistent(-60.0)))
    baseline = select_harmful_groups(window)
    unseen = _seed(_window(_shifting_bias([-200.0, -100.0, 100.0, 200.0]),
                           route="Blue", bucket=12))
    assert select_harmful_groups(window) == baseline
    assert not any(route == "Blue" for route, _ in baseline)
    assert len(unseen)


def test_suppressed_group_is_zeroed_at_every_level():
    """zeroing only by_stop would let by_route put the same correction back"""
    data = pd.concat([
        _window(_consistent(-120.0), route="Orange"),
        _window(_consistent(-60.0), route="Red"),
    ])
    guarded = suppress_groups(fit_lookup(data), {("Orange", 9)})
    assert guarded["by_stop"].loc[("Orange", "S1", 9)] == 0.0
    assert guarded["by_route"].loc[("Orange", 9)] == 0.0
    assert guarded["by_route"].loc[("Red", 9)] != 0.0


def test_suppression_never_mutates_the_input_lookup():
    """two lookups get suppressed separately, so in-place mutation would couple them"""
    original = fit_lookup(_window(_consistent(-120.0), route="Orange"))
    before = original["by_route"].loc[("Orange", 9)]
    suppress_groups(original, {("Orange", 9)})
    assert original["by_route"].loc[("Orange", 9)] == before
