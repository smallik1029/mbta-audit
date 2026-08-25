"""matching predictions to arrivals"""
import pandas as pd

from src.process.match import build_outcomes, exclude_cancelled


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
