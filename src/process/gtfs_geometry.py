"""build the map geometry parquets from the mbta gtfs bundle"""
import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

from src import config

GTFS_URL = "https://cdn.mbta.com/MBTA_GTFS.zip"
GTFS_DIR = config.ROOT / "data" / "gtfs"

MAP_ROUTES = ("Red", "Orange", "Blue", "Green-B", "Green-C", "Green-D", "Green-E")

STOP_TIMES_CHUNK = 500_000


def download_bundle() -> zipfile.ZipFile:
    resp = requests.get(GTFS_URL, timeout=300)
    resp.raise_for_status()
    return zipfile.ZipFile(io.BytesIO(resp.content))


def open_bundle(source: str | Path | None) -> zipfile.ZipFile:
    if source is None:
        print(f"downloading {GTFS_URL}", flush=True)
        return download_bundle()
    print(f"reading {source}", flush=True)
    return zipfile.ZipFile(source)


def build(bundle: zipfile.ZipFile, routes: tuple[str, ...]) -> dict[str, pd.DataFrame]:
    with bundle.open("trips.txt") as f:
        trips = pd.read_csv(f, dtype=str)
    trips = trips[trips["route_id"].isin(routes)]
    trip_route = trips[["trip_id", "route_id", "shape_id", "direction_id", "trip_headsign"]].copy()
    trip_route["direction_id"] = trip_route["direction_id"].astype(int)
    keep_trips = set(trip_route["trip_id"])
    keep_shapes = set(trip_route["shape_id"].dropna())

    with bundle.open("shapes.txt") as f:
        shapes = pd.read_csv(f, dtype={"shape_id": str})
    shapes = shapes[shapes["shape_id"].isin(keep_shapes)]
    shapes = shapes[["shape_id", "shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"]]

    stop_times = _read_stop_times(bundle, keep_trips)
    keep_stops = set(stop_times["stop_id"])

    with bundle.open("stops.txt") as f:
        stops = pd.read_csv(f, dtype={"stop_id": str})
    stops = stops[stops["stop_id"].isin(keep_stops)]
    stop_latlon = stops[["stop_id", "stop_lat", "stop_lon", "stop_name"]]

    return {
        "ro_stop_latlon": stop_latlon,
        "ro_stop_times": stop_times,
        "ro_trip_route": trip_route,
        "ro_shapes": shapes,
    }


def _read_stop_times(bundle: zipfile.ZipFile, keep_trips: set) -> pd.DataFrame:
    """only rows for trips we keep, read in chunks. the file is ~240mb"""
    cols = ["trip_id", "arrival_time", "departure_time", "stop_id", "stop_sequence"]
    parts = []
    with bundle.open("stop_times.txt") as f:
        for chunk in pd.read_csv(f, dtype=str, usecols=cols, chunksize=STOP_TIMES_CHUNK):
            parts.append(chunk[chunk["trip_id"].isin(keep_trips)])
    out = pd.concat(parts, ignore_index=True)
    out["stop_sequence"] = out["stop_sequence"].astype(int)
    return out[cols]


def main(source: str | Path | None = None, routes: tuple[str, ...] = MAP_ROUTES) -> None:
    frames = build(open_bundle(source), routes)
    GTFS_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in frames.items():
        path = GTFS_DIR / f"{name}.parquet"
        df.to_parquet(path, index=False)
        print(f"{name:16} {len(df):>9,} rows -> {path}")

    covered = sorted(frames["ro_trip_route"]["route_id"].unique())
    print(f"routes covered: {', '.join(covered)}")


if __name__ == "__main__":
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else None)
