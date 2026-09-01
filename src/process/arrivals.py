"""work out real arrival times from vehicle status changes"""
from datetime import timedelta

import pandas as pd

from src import config, db

OVERLAP = timedelta(minutes=15)

WINDOW = timedelta(days=1)


def get_watermark() -> pd.Timestamp | None:
    """latest arrival already recorded, or None if the table is empty"""
    conn = db.connect()
    row = conn.execute("SELECT MAX(actual_arrival) FROM actual_arrivals").fetchone()
    if row is None or row[0] is None:
        return None
    return pd.Timestamp(row[0])


def load_vehicle_snapshots(
    since: pd.Timestamp | None, until: pd.Timestamp | None = None
) -> pd.DataFrame:
    conn = db.connect()
    query = (
        "SELECT observed_at, vehicle_id, trip_id, route_id, direction_id, "
        "current_stop_id, current_status FROM vehicle_snapshots "
        "WHERE trip_id IS NOT NULL AND current_stop_id IS NOT NULL"
    )
    params: list = []
    if since is not None:
        query += " AND observed_at > ?"
        params.append((since - OVERLAP).isoformat())
    if until is not None:
        query += " AND observed_at <= ?"
        params.append(until.isoformat())
    query += " ORDER BY vehicle_id, observed_at"

    df = pd.read_sql_query(query, conn, params=params)
    df["observed_at"] = pd.to_datetime(df["observed_at"], utc=True, format="ISO8601")
    return df


def derive_arrivals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["prev_status"] = df.groupby("vehicle_id")["current_status"].shift(1)
    df["prev_stop_id"] = df.groupby("vehicle_id")["current_stop_id"].shift(1)
    df["prev_observed_at"] = df.groupby("vehicle_id")["observed_at"].shift(1)
    df["prev_trip_id"] = df.groupby("vehicle_id")["trip_id"].shift(1)

    arrivals = df[df["current_status"] == "STOPPED_AT"].copy()

    arrivals["service_date"] = arrivals["observed_at"].dt.date.astype(str)
    arrivals = arrivals.sort_values("observed_at")
    arrivals = arrivals.drop_duplicates(
        subset=["trip_id", "current_stop_id", "service_date"], keep="first"
    )

    has_incoming = (
        (arrivals["prev_status"] == "INCOMING_AT")
        & (arrivals["prev_stop_id"] == arrivals["current_stop_id"])
    )
    arrivals["incoming_at"] = arrivals["prev_observed_at"].where(has_incoming)

    bound = (arrivals["observed_at"] - arrivals["prev_observed_at"]).dt.total_seconds()
    arrivals["detection_bound_sec"] = bound.where(has_incoming, config.VEHICLES_POLL_SECONDS)

    arrivals = arrivals.rename(columns={"current_stop_id": "stop_id", "observed_at": "actual_arrival"})
    arrivals = pd.concat([arrivals, _inbound_trip_arrivals(arrivals)], ignore_index=True)
    arrivals = arrivals.sort_values("actual_arrival").drop_duplicates(
        subset=["trip_id", "stop_id", "service_date"], keep="first"
    )
    return arrivals[[
        "trip_id", "stop_id", "route_id", "direction_id", "vehicle_id",
        "actual_arrival", "incoming_at", "detection_bound_sec", "service_date",
    ]]


def _inbound_trip_arrivals(arrivals: pd.DataFrame) -> pd.DataFrame:
    """terminus arrivals, refiled under the trip that was actually arriving"""
    handover = arrivals[
        arrivals["prev_trip_id"].notna()
        & (arrivals["prev_trip_id"] != arrivals["trip_id"])
        & (arrivals["prev_stop_id"] == arrivals["stop_id"])
    ].copy()
    handover["trip_id"] = handover["prev_trip_id"]
    return handover


def write_arrivals(arrivals: pd.DataFrame) -> int:
    """append with INSERT OR IGNORE. returns how many were new"""
    rows = [
        (
            r.trip_id, r.stop_id, r.route_id, r.direction_id, r.vehicle_id,
            r.actual_arrival.isoformat(),
            r.incoming_at.isoformat() if pd.notna(r.incoming_at) else None,
            r.detection_bound_sec, r.service_date,
        )
        for r in arrivals.itertuples()
    ]
    conn = db.connect()
    before = conn.execute("SELECT COUNT(*) FROM actual_arrivals").fetchone()[0]
    conn.executemany(
        "INSERT OR IGNORE INTO actual_arrivals "
        "(trip_id, stop_id, route_id, direction_id, vehicle_id, actual_arrival, "
        "incoming_at, detection_bound_sec, service_date) VALUES (?,?,?,?,?,?,?,?,?)",
        rows,
    )
    conn.commit()
    after = conn.execute("SELECT COUNT(*) FROM actual_arrivals").fetchone()[0]
    return after - before


def earliest_snapshot() -> pd.Timestamp | None:
    conn = db.connect()
    row = conn.execute("SELECT MIN(observed_at) FROM vehicle_snapshots").fetchone()
    if row is None or row[0] is None:
        return None
    return pd.Timestamp(row[0])


def main() -> None:
    db.init_schema()
    watermark = get_watermark()
    start = watermark if watermark is not None else earliest_snapshot()
    if start is None:
        print("no vehicle snapshots to process")
        return

    now = pd.Timestamp.now(tz="UTC")
    scope = "full history (first run)" if watermark is None else f"since {watermark.isoformat()}"
    print(f"deriving arrivals in {WINDOW.days}-day windows [{scope}]", flush=True)

    total_rows = total_candidates = total_new = total_bounded = 0
    window_start = start
    while window_start < now:
        window_end = min(window_start + WINDOW, now)
        df = load_vehicle_snapshots(window_start, window_end)
        if len(df):
            arrivals = derive_arrivals(df)
            n_new = write_arrivals(arrivals)
            total_rows += len(df)
            total_candidates += len(arrivals)
            total_new += n_new
            total_bounded += int(arrivals["incoming_at"].notna().sum())
            print(f"  {window_start:%Y-%m-%d}: {len(df):,} snapshot rows, "
                  f"{n_new:,} new arrivals", flush=True)
        window_start = window_end

    print(f"loaded {total_rows:,} vehicle snapshot rows")
    print(f"processed {total_candidates:,} candidate arrivals, {total_new:,} were new "
          f"({total_bounded:,} bias-bounded via INCOMING_AT, "
          f"{total_candidates - total_bounded:,} bounded by poll interval fallback)")


if __name__ == "__main__":
    main()
