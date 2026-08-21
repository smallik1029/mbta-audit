"""one-time load of static gtfs reference data"""
import pandas as pd

from src import config, db

GTFS_DIR = config.ROOT / "data" / "gtfs"


def load_routes() -> int:
    df = pd.read_csv(GTFS_DIR / "routes.txt", dtype=str)
    df = df[df["route_id"].isin(config.ROUTES)]
    rows = list(df[["route_id", "route_long_name", "route_type"]].itertuples(index=False, name=None))

    conn = db.connect()
    conn.execute("DELETE FROM gtfs_routes")
    conn.executemany(
        "INSERT INTO gtfs_routes (route_id, route_name, route_type) VALUES (?,?,?)",
        rows,
    )
    conn.commit()
    return len(rows)


def load_stops() -> int:
    df = pd.read_csv(GTFS_DIR / "stops.txt", dtype=str)
    df = df[["stop_id", "stop_name", "parent_station"]]
    rows = list(df.itertuples(index=False, name=None))

    conn = db.connect()
    conn.execute("DELETE FROM gtfs_stops")
    conn.executemany(
        "INSERT INTO gtfs_stops (stop_id, stop_name, parent_station) VALUES (?,?,?)",
        rows,
    )
    conn.commit()
    return len(rows)


def main() -> None:
    db.init_schema()
    n_routes = load_routes()
    n_stops = load_stops()
    print(f"loaded {n_routes} routes, {n_stops:,} stops (all of GTFS, not just Red/Orange)")


if __name__ == "__main__":
    main()
