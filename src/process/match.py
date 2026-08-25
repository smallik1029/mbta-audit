"""join predictions to real arrivals"""
from datetime import timedelta

import pandas as pd

from src import config, db

CANCELLED_STATES = {"CANCELLED", "SKIPPED"}

SERVED_ROUTES = ("Red", "Orange", "Blue")

OVERLAP = timedelta(minutes=90)


def exclude_cancelled(predictions: pd.DataFrame) -> pd.DataFrame:
    """drop predictions MBTA flagged as not happening"""
    return predictions[~predictions["schedule_relationship"].isin(CANCELLED_STATES)]


def get_watermark() -> pd.Timestamp | None:
    """highest arrival id already matched"""
    conn = db.connect()
    row = conn.execute("SELECT MAX(observed_at) FROM prediction_outcomes").fetchone()
    if row is None or row[0] is None:
        return None
    return pd.Timestamp(row[0])


def load_joined_predictions(
    since: pd.Timestamp | None = None, service_date: str | None = None
) -> pd.DataFrame:
    """predictions joined to a slice of arrivals, in sql"""
    conn = db.connect()
    placeholders = ",".join("?" * len(SERVED_ROUTES))
    query = (
        "SELECT p.observed_at, p.trip_id, p.stop_id, p.predicted_arrival, "
        "p.schedule_relationship, a.route_id, a.direction_id, a.actual_arrival "
        "FROM prediction_snapshots p "
        "JOIN actual_arrivals a ON p.trip_id = a.trip_id AND p.stop_id = a.stop_id "
        f"WHERE p.predicted_arrival IS NOT NULL AND a.route_id IN ({placeholders})"
    )
    params: list = list(SERVED_ROUTES)
    if since is not None:
        query += " AND p.observed_at > ?"
        params.append((since - OVERLAP).isoformat())
    if service_date is not None:
        query += " AND a.service_date = ?"
        params.append(service_date)

    df = pd.read_sql_query(query, conn, params=params)
    df = exclude_cancelled(df)
    df["observed_at"] = pd.to_datetime(df["observed_at"], utc=True, format="ISO8601")
    df["predicted_arrival"] = pd.to_datetime(df["predicted_arrival"], utc=True, format="ISO8601")
    df["actual_arrival"] = pd.to_datetime(df["actual_arrival"], utc=True, format="ISO8601")
    return df


def list_service_dates() -> list[str]:
    conn = db.connect()
    rows = conn.execute("SELECT DISTINCT service_date FROM actual_arrivals ORDER BY service_date").fetchall()
    return [r[0] for r in rows]


def build_outcomes(joined: pd.DataFrame) -> pd.DataFrame:
    merged = joined[joined["observed_at"] < joined["actual_arrival"]].copy()

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


def write_outcomes(outcomes: pd.DataFrame) -> tuple[int, int]:
    """append with INSERT OR IGNORE. returns (new rows, the id boundary before this write)"""
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
    before = conn.execute("SELECT COUNT(*) FROM prediction_outcomes").fetchone()[0]
    since_id = conn.execute("SELECT COALESCE(MAX(id), 0) FROM prediction_outcomes").fetchone()[0]
    conn.executemany(
        "INSERT OR IGNORE INTO prediction_outcomes "
        "(trip_id, stop_id, route_id, direction_id, observed_at, predicted_arrival, "
        "actual_arrival, lead_time_sec, error_sec, hour_local, is_rush_hour) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        rows,
    )
    conn.commit()
    after = conn.execute("SELECT COUNT(*) FROM prediction_outcomes").fetchone()[0]
    return after - before, since_id


def print_new_by_stop(since_id: int, top_n: int = 5) -> None:
    """how many new rows landed on each stop"""
    conn = db.connect()
    rows = conn.execute(
        "SELECT o.route_id, o.stop_id, COALESCE(g.stop_name, o.stop_id) AS name, COUNT(*) AS n "
        "FROM prediction_outcomes o LEFT JOIN gtfs_stops g ON g.stop_id = o.stop_id "
        "WHERE o.id > ? GROUP BY o.route_id, o.stop_id ORDER BY n DESC",
        (since_id,),
    ).fetchall()
    if not rows:
        print("  (no new outcomes this run)")
        return

    def _line(row):
        route_id, _stop_id, name, n = row
        return f"    {route_id:6s} {name}: {n:,}"

    print(f"  new outcomes by stop ({len(rows)} stops had new data):")
    if len(rows) <= 2 * top_n:
        for row in rows:
            print(_line(row))
        return

    for row in rows[:top_n]:
        print(_line(row))
    hidden = len(rows) - 2 * top_n
    print(f"    ... {hidden} more stop(s) ...")
    for row in rows[-top_n:]:
        print(_line(row))


def _process_and_write(joined: pd.DataFrame) -> tuple[int, int, pd.DataFrame]:
    outcomes = build_outcomes(joined)
    n_new, since_id = write_outcomes(outcomes)
    return n_new, since_id, outcomes


def run_bootstrap() -> None:
    """First-ever run: no watermark yet, so chunk by calendar day instead of"""
    dates = list_service_dates()
    print(f"no watermark yet -- bootstrapping {len(dates)} calendar day(s) one at a time")

    total_new = 0
    for d in dates:
        joined = load_joined_predictions(service_date=d)
        n_new, _since_id, outcomes = _process_and_write(joined)
        total_new += n_new
        print(f"  {d}: {len(joined):,} predictions joined, {n_new:,} new outcomes written")

    print(f"bootstrap complete: {total_new:,} total new outcomes across {len(dates)} day(s)")
    conn = db.connect()
    row = conn.execute(
        "SELECT COUNT(*), AVG(ABS(error_sec)), AVG(lead_time_sec) FROM prediction_outcomes"
    ).fetchone()
    if row and row[0]:
        print(f"prediction_outcomes now has {row[0]:,} total rows "
              f"(mean |error|: {row[1]:.0f}s, mean lead_time: {row[2]:.0f}s)")


def main() -> None:
    db.init_schema()
    watermark = get_watermark()
    if watermark is None:
        run_bootstrap()
        return

    joined = load_joined_predictions(since=watermark)
    print(f"{len(joined):,} predictions joined to arrivals [since {watermark.isoformat()}]")

    n_new, since_id, outcomes = _process_and_write(joined)
    print(f"processed {len(outcomes):,} candidate outcomes, {n_new:,} were new")
    if len(outcomes):
        print(f"median |error|: {outcomes['error_sec'].abs().median():.0f}s "
              f"median lead_time: {outcomes['lead_time_sec'].median():.0f}s")
    print_new_by_stop(since_id)


if __name__ == "__main__":
    main()
