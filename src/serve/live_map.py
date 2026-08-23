"""builds the two-dot-per-train map data"""
from datetime import UTC, datetime

import pandas as pd
import requests

from src import config
from src.serve.correction import lead_bucket, lookup_bias

GTFS_DIR = config.ROOT / "data" / "gtfs"


def _parse_gtfs_time_to_seconds(hms: str) -> float:
    """gtfs times can go past 24:00:00, so this is a duration parse, not a clock time"""
    h, m, s = hms.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


def load_shapes() -> dict:
    shapes = pd.read_parquet(GTFS_DIR / "ro_shapes.parquet")
    trip_route = pd.read_parquet(GTFS_DIR / "ro_trip_route.parquet")
    shape_route = trip_route[["shape_id", "route_id"]].drop_duplicates("shape_id")
    shapes = shapes.merge(shape_route, on="shape_id", how="left")

    out = {"Red": [], "Orange": []}
    for _shape_id, group in shapes.groupby("shape_id"):
        route_id = group["route_id"].iloc[0]
        if route_id not in out:
            continue
        ordered = group.sort_values("shape_pt_sequence")
        points = ordered[["shape_pt_lat", "shape_pt_lon"]].astype(float).values.tolist()
        out[route_id].append(points)
    return out


class ScheduleIndex:
    """given (trip, next stop), find the previous stop and how long that leg should take"""

    def __init__(self):
        st = pd.read_parquet(GTFS_DIR / "ro_stop_times.parquet")
        st["stop_sequence"] = st["stop_sequence"].astype(int)
        st = st.sort_values(["trip_id", "stop_sequence"])
        st["arrival_sec"] = st["arrival_time"].map(_parse_gtfs_time_to_seconds)
        st["departure_sec"] = st["departure_time"].map(_parse_gtfs_time_to_seconds)
        st["prev_stop_id"] = st.groupby("trip_id")["stop_id"].shift(1)
        st["prev_departure_sec"] = st.groupby("trip_id")["departure_sec"].shift(1)
        self.st = st.set_index(["trip_id", "stop_id"])

        latlon = pd.read_parquet(GTFS_DIR / "ro_stop_latlon.parquet").set_index("stop_id")
        self.latlon = latlon

    def get_leg(self, trip_id: str, next_stop_id: str):
        try:
            row = self.st.loc[(trip_id, next_stop_id)]
        except KeyError:
            return None
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]

        prev_stop_id = row["prev_stop_id"]
        if pd.isna(prev_stop_id):
            return None
        leg_duration = row["arrival_sec"] - row["prev_departure_sec"]
        if leg_duration <= 0:
            return None

        try:
            prev_lat, prev_lon = self.latlon.loc[prev_stop_id][["stop_lat", "stop_lon"]]
            next_lat, next_lon = self.latlon.loc[next_stop_id][["stop_lat", "stop_lon"]]
        except KeyError:
            return None

        return {
            "prev_lat": float(prev_lat), "prev_lon": float(prev_lon),
            "next_lat": float(next_lat), "next_lon": float(next_lon),
            "leg_duration_sec": float(leg_duration),
        }


def _fetch(path: str, routes: list[str]) -> list[dict]:
    resp = requests.get(
        f"{config.API_BASE}{path}",
        params={"filter[route]": ",".join(routes)},
        headers={"x-api-key": config.API_KEY} if config.API_KEY else {},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json().get("data", [])


def _rel_id(item: dict, name: str):
    rel = (item.get("relationships") or {}).get(name) or {}
    return (rel.get("data") or {}).get("id")


def _lerp(leg: dict, frac: float) -> list:
    return [
        leg["prev_lat"] + (leg["next_lat"] - leg["prev_lat"]) * frac,
        leg["prev_lon"] + (leg["next_lon"] - leg["prev_lon"]) * frac,
    ]


def build_live_trains(schedule: ScheduleIndex, lookup: dict, routes: list[str]) -> list[dict]:
    predictions = _fetch("/predictions", routes)
    now = datetime.now(UTC)

    next_pred: dict[str, dict] = {}
    for item in predictions:
        attrs = item.get("attributes") or {}
        arrival = attrs.get("arrival_time")
        trip_id = _rel_id(item, "trip")
        if not arrival or not trip_id:
            continue
        arrival_dt = pd.Timestamp(arrival).tz_convert("UTC")
        lead_sec = (arrival_dt - now).total_seconds()
        if lead_sec < 0:
            continue
        if trip_id not in next_pred or lead_sec < next_pred[trip_id]["lead_sec"]:
            next_pred[trip_id] = {
                "stop_id": _rel_id(item, "stop"),
                "route_id": _rel_id(item, "route"),
                "lead_sec": lead_sec,
            }

    results = []
    for trip_id, pred in next_pred.items():
        leg = schedule.get_leg(trip_id, pred["stop_id"])
        if leg is None:
            continue

        route_id = pred["route_id"]
        bucket = lead_bucket(pred["lead_sec"])
        bias, _level = lookup_bias(lookup, route_id, pred["stop_id"], bucket)
        corrected_lead_sec = pred["lead_sec"] - bias

        raw_frac = 1 - min(max(pred["lead_sec"] / leg["leg_duration_sec"], 0), 1)
        corrected_frac = 1 - min(max(corrected_lead_sec / leg["leg_duration_sec"], 0), 1)

        results.append({
            "trip_id": trip_id,
            "route_id": route_id,
            "raw_pos": _lerp(leg, raw_frac),
            "corrected_pos": _lerp(leg, corrected_frac),
        })

    return results
