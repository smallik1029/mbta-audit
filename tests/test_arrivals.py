"""arrival derivation"""
import pandas as pd

from src.process.arrivals import derive_arrivals


def _snapshots(rows):
    df = pd.DataFrame(rows, columns=[
        "observed_at", "vehicle_id", "trip_id", "route_id", "direction_id",
        "current_stop_id", "current_status",
    ])
    df["observed_at"] = pd.to_datetime(df["observed_at"], utc=True)
    return df.sort_values(["vehicle_id", "observed_at"])


def test_terminus_arrival_is_also_filed_under_the_inbound_trip():
    """the vehicle is already on its outbound trip by STOPPED_AT, so record the inbound one too"""
    df = _snapshots([
        ("2026-08-31T16:00:00Z", "V1", "INBOUND", "Red", 1, "70061", "INCOMING_AT"),
        ("2026-08-31T16:00:30Z", "V1", "OUTBOUND", "Red", 0, "70061", "STOPPED_AT"),
    ])
    out = derive_arrivals(df)
    assert sorted(out["trip_id"]) == ["INBOUND", "OUTBOUND"]
    assert set(out["stop_id"]) == {"70061"}
    assert out["actual_arrival"].nunique() == 1


def test_normal_mid_line_arrival_is_recorded_once():
    df = _snapshots([
        ("2026-08-31T16:00:00Z", "V1", "T1", "Red", 0, "70063", "INCOMING_AT"),
        ("2026-08-31T16:00:30Z", "V1", "T1", "Red", 0, "70063", "STOPPED_AT"),
    ])
    out = derive_arrivals(df)
    assert list(out["trip_id"]) == ["T1"]
    assert out.iloc[0]["incoming_at"] is not pd.NaT
    assert out.iloc[0]["detection_bound_sec"] == 30.0


def test_trip_change_at_a_different_stop_is_not_treated_as_a_handover():
    """only a trip change at the same stop counts, otherwise arrivals get duplicated"""
    df = _snapshots([
        ("2026-08-31T16:00:00Z", "V1", "T1", "Red", 0, "70061", "STOPPED_AT"),
        ("2026-08-31T16:05:00Z", "V1", "T2", "Red", 0, "70063", "STOPPED_AT"),
    ])
    out = derive_arrivals(df)
    assert sorted(zip(out["trip_id"], out["stop_id"], strict=False)) == [("T1", "70061"), ("T2", "70063")]
