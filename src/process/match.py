"""join predictions to real arrivals"""
import time

import pandas as pd

from src import config, db

CANCELLED_STATES = {"CANCELLED", "SKIPPED"}

SERVED_ROUTES = ("Red", "Orange", "Blue", "Green-B", "Green-C", "Green-D", "Green-E")


def exclude_cancelled(predictions: pd.DataFrame) -> pd.DataFrame:
    """drop predictions MBTA flagged as not happening"""
    return predictions[~predictions["schedule_relationship"].isin(CANCELLED_STATES)]


STATE_KEY = "match_last_arrival_id"

BATCH_ARRIVALS = 3_000

MAX_LOOKBACK_HOURS = 3


def seed_watermark() -> int:
    """where to start on a db that was matched before batching existed"""
    conn = db.connect()
    row = conn.execute(
        "SELECT observed_at FROM prediction_outcomes ORDER BY id DESC LIMIT 1"
    ).fetchone()
    if row is None or row[0] is None:
        return 0
    cutoff = (pd.Timestamp(row[0]) - pd.Timedelta(days=1)).isoformat()
    return int(conn.execute(
        "SELECT COALESCE(MAX(id), 0) FROM actual_arrivals WHERE actual_arrival <= ?",
        (cutoff,),
    ).fetchone()[0])


def set_watermark(value: int) -> None:
    conn = db.connect()
    conn.execute(
        "INSERT INTO pipeline_state (key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (STATE_KEY, str(value)),
    )
    conn.commit()


def get_watermark() -> int:
    """highest arrival id already matched"""
    conn = db.connect()
    row = conn.execute("SELECT value FROM pipeline_state WHERE key = ?", (STATE_KEY,)).fetchone()
    if row:
        return int(row[0])
    seeded = seed_watermark()
    set_watermark(seeded)
    return seeded


def next_batch_end(since_id: int, limit: int = BATCH_ARRIVALS) -> int | None:
    """highest arrival id in the next batch, or None when there is nothing left"""
    conn = db.connect()
    return conn.execute(
        "SELECT MAX(id) FROM (SELECT id FROM actual_arrivals WHERE id > ? ORDER BY id LIMIT ?)",
        (since_id, limit),
    ).fetchone()[0]


def load_joined_predictions(since_id: int, until_id: int) -> pd.DataFrame:
    """predictions joined to a slice of arrivals, in sql"""
    conn = db.connect()
    placeholders = ",".join("?" * len(SERVED_ROUTES))
    query = (
        "SELECT p.observed_at, p.trip_id, p.stop_id, p.predicted_arrival, "
        "p.schedule_relationship, a.route_id, a.direction_id, a.actual_arrival "
        "FROM actual_arrivals a "
        "JOIN prediction_snapshots p ON p.trip_id = a.trip_id AND p.stop_id = a.stop_id "
        "WHERE a.id > ? AND a.id <= ? AND p.predicted_arrival IS NOT NULL "
        "AND julianday(a.actual_arrival) - julianday(p.observed_at) BETWEEN 0 AND ? "
        f"AND a.route_id IN ({placeholders})"
    )
    params: list = [since_id, until_id, MAX_LOOKBACK_HOURS / 24.0, *SERVED_ROUTES]

    df = pd.read_sql_query(query, conn, params=params)
    df = exclude_cancelled(df)
    df["observed_at"] = pd.to_datetime(df["observed_at"], utc=True, format="ISO8601")
    df["predicted_arrival"] = pd.to_datetime(df["predicted_arrival"], utc=True, format="ISO8601")
    df["actual_arrival"] = pd.to_datetime(df["actual_arrival"], utc=True, format="ISO8601")
    return df


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


def print_new_by_stop(since_id: int) -> None:
    """how many new rows landed on each stop"""
    conn = db.connect()
    rows = conn.execute(
        "SELECT o.route_id, o.stop_id, COALESCE(g.stop_name, o.stop_id) AS name, "
        "       SUM(CASE WHEN o.id > :since_id THEN 1 ELSE 0 END) AS n_new, "
        "       COUNT(*) AS n_total "
        "FROM prediction_outcomes o LEFT JOIN gtfs_stops g ON g.stop_id = o.stop_id "
        "GROUP BY o.route_id, o.stop_id HAVING n_new > 0 ORDER BY n_new DESC",
        {"since_id": since_id},
    ).fetchall()
    if not rows:
        print("  (no new outcomes this run)")
        return

    def _line(row):
        route_id, _stop_id, name, n_new, n_total = row
        return f"    {route_id:6s} {name}: +{n_new:,} -> {n_total:,} total"

    print(f"  new outcomes by stop ({len(rows)} stops had new data):")
    for row in rows:
        print(_line(row))


def _process_and_write(joined: pd.DataFrame) -> tuple[int, int, pd.DataFrame]:
    outcomes = build_outcomes(joined)
    n_new, since_id = write_outcomes(outcomes)
    return n_new, since_id, outcomes


def run_batches() -> tuple[int, int | None]:
    """work through unmatched arrivals a batch at a time"""
    since = get_watermark()
    total_new = 0
    first_id = None
    batches = 0
    while True:
        batch_end = next_batch_end(since)
        if batch_end is None:
            break
        started = time.monotonic()
        joined = load_joined_predictions(since, batch_end)
        n_new, since_id, _outcomes = _process_and_write(joined)
        if first_id is None:
            first_id = since_id
        total_new += n_new
        batches += 1
        set_watermark(batch_end)
        print(f"  arrivals {since:,}-{batch_end:,}: {len(joined):,} predictions joined, "
              f"{n_new:,} new outcomes ({time.monotonic() - started:.0f}s)", flush=True)
        since = batch_end
    if not batches:
        print("  nothing new to match")
    return total_new, first_id


def main() -> None:
    db.init_schema()
    print(f"matching in batches of {BATCH_ARRIVALS:,} arrivals "
          f"[from arrival id {get_watermark():,}]", flush=True)
    total_new, first_id = run_batches()
    print(f"{total_new:,} new outcomes")
    if first_id is not None:
        print_new_by_stop(first_id)


if __name__ == "__main__":
    main()
