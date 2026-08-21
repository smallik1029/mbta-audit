"""work out real arrival times from vehicle status changes"""
import pandas as pd

from src import config, db


def load_vehicle_snapshots() -> pd.DataFrame:
    conn = db.connect()
    df = pd.read_sql_query(
        "SELECT observed_at, vehicle_id, trip_id, route_id, direction_id, "
        "current_stop_id, current_status FROM vehicle_snapshots "
        "WHERE trip_id IS NOT NULL AND current_stop_id IS NOT NULL "
        "ORDER BY vehicle_id, observed_at",
        conn,
    )
    df["observed_at"] = pd.to_datetime(df["observed_at"], utc=True)
    return df


def derive_arrivals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["prev_status"] = df.groupby("vehicle_id")["current_status"].shift(1)
    df["prev_stop_id"] = df.groupby("vehicle_id")["current_stop_id"].shift(1)
    df["prev_observed_at"] = df.groupby("vehicle_id")["observed_at"].shift(1)

    arrivals = df[df["current_status"] == "STOPPED_AT"].copy()

    arrivals = arrivals.sort_values("observed_at")
    arrivals = arrivals.drop_duplicates(subset=["trip_id", "current_stop_id"], keep="first")

    has_incoming = (
        (arrivals["prev_status"] == "INCOMING_AT")
        & (arrivals["prev_stop_id"] == arrivals["current_stop_id"])
    )
    arrivals["incoming_at"] = arrivals["prev_observed_at"].where(has_incoming)

    bound = (arrivals["observed_at"] - arrivals["prev_observed_at"]).dt.total_seconds()
    arrivals["detection_bound_sec"] = bound.where(has_incoming, config.VEHICLES_POLL_SECONDS)

    arrivals = arrivals.rename(columns={"current_stop_id": "stop_id", "observed_at": "actual_arrival"})
    return arrivals[[
        "trip_id", "stop_id", "route_id", "direction_id", "vehicle_id",
        "actual_arrival", "incoming_at", "detection_bound_sec",
    ]]


def write_arrivals(arrivals: pd.DataFrame) -> int:
    rows = [
        (
            r.trip_id, r.stop_id, r.route_id, r.direction_id, r.vehicle_id,
            r.actual_arrival.isoformat(),
            r.incoming_at.isoformat() if pd.notna(r.incoming_at) else None,
            r.detection_bound_sec,
        )
        for r in arrivals.itertuples()
    ]
    conn = db.connect()
    conn.execute("DELETE FROM actual_arrivals")
    conn.executemany(
        "INSERT INTO actual_arrivals "
        "(trip_id, stop_id, route_id, direction_id, vehicle_id, actual_arrival, "
        "incoming_at, detection_bound_sec) VALUES (?,?,?,?,?,?,?,?)",
        rows,
    )
    conn.commit()
    return len(rows)


def main() -> None:
    db.init_schema()
    df = load_vehicle_snapshots()
    print(f"loaded {len(df):,} vehicle snapshot rows")
    arrivals = derive_arrivals(df)
    n = write_arrivals(arrivals)
    bounded = int(arrivals["incoming_at"].notna().sum())
    print(f"derived {n:,} actual arrivals "
          f"({bounded:,} bias-bounded via INCOMING_AT, "
          f"{n - bounded:,} bounded by poll interval fallback)")


if __name__ == "__main__":
    main()
