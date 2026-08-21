"""join predictions to real arrivals"""
import pandas as pd

from src import config, db

CANCELLED_STATES = {"CANCELLED", "SKIPPED"}


def exclude_cancelled(predictions: pd.DataFrame) -> pd.DataFrame:
    """drop predictions MBTA flagged as not happening"""
    return predictions[~predictions["schedule_relationship"].isin(CANCELLED_STATES)]


def load_actual_arrivals() -> pd.DataFrame:
    conn = db.connect()
    df = pd.read_sql_query(
        "SELECT trip_id, stop_id, route_id, direction_id, actual_arrival "
        "FROM actual_arrivals",
        conn,
    )
    df["actual_arrival"] = pd.to_datetime(df["actual_arrival"], utc=True, format="ISO8601")
    return df


def load_predictions() -> pd.DataFrame:
    conn = db.connect()
    df = pd.read_sql_query(
        "SELECT observed_at, trip_id, stop_id, predicted_arrival, "
        "schedule_relationship FROM prediction_snapshots "
        "WHERE predicted_arrival IS NOT NULL",
        conn,
    )
    df = exclude_cancelled(df)
    df["observed_at"] = pd.to_datetime(df["observed_at"], utc=True, format="ISO8601")
    df["predicted_arrival"] = pd.to_datetime(df["predicted_arrival"], utc=True, format="ISO8601")
    return df


def build_outcomes(arrivals: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    merged = predictions.merge(arrivals, on=["trip_id", "stop_id"], how="inner")

    merged = merged[merged["observed_at"] < merged["actual_arrival"]].copy()

    merged["lead_time_sec"] = (merged["actual_arrival"] - merged["observed_at"]).dt.total_seconds()
    merged["error_sec"] = (merged["predicted_arrival"] - merged["actual_arrival"]).dt.total_seconds()

    local = merged["observed_at"].dt.tz_convert(config.LOCAL_TZ)
    merged["hour_local"] = local.dt.hour
    merged["is_rush_hour"] = merged["hour_local"].isin(config.RUSH_HOURS).astype(int)

    return merged[[
        "trip_id", "stop_id", "route_id", "direction_id", "observed_at",
        "predicted_arrival", "actual_arrival", "lead_time_sec", "error_sec",
        "hour_local", "is_rush_hour",
    ]]


def write_outcomes(outcomes: pd.DataFrame) -> int:
    rows = [
        (
            r.trip_id, r.stop_id, r.route_id, r.direction_id,
            r.observed_at.isoformat(), r.predicted_arrival.isoformat(),
            r.actual_arrival.isoformat(), r.lead_time_sec, r.error_sec,
            int(r.hour_local), int(r.is_rush_hour),
        )
        for r in outcomes.itertuples()
    ]
    conn = db.connect()
    conn.execute("DELETE FROM prediction_outcomes")
    conn.executemany(
        "INSERT INTO prediction_outcomes "
        "(trip_id, stop_id, route_id, direction_id, observed_at, predicted_arrival, "
        "actual_arrival, lead_time_sec, error_sec, hour_local, is_rush_hour) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        rows,
    )
    conn.commit()
    return len(rows)


def main() -> None:
    arrivals = load_actual_arrivals()
    predictions = load_predictions()
    print(f"{len(arrivals):,} actual arrivals, {len(predictions):,} live predictions to join")

    outcomes = build_outcomes(arrivals, predictions)
    n = write_outcomes(outcomes)

    matched_trips = outcomes[["trip_id", "stop_id"]].drop_duplicates().shape[0]
    unmatched = len(arrivals) - matched_trips
    print(f"wrote {n:,} prediction_outcomes rows across {matched_trips:,} "
          f"trip/stop pairs ({unmatched:,} arrivals had no usable predictions)")
    if n:
        print(f"median |error|: {outcomes['error_sec'].abs().median():.0f}s "
              f"median lead_time: {outcomes['lead_time_sec'].median():.0f}s")


if __name__ == "__main__":
    main()
