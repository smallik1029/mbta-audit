"""builds the two-dot-per-train map data"""
import math
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import requests

from src import config
from src.serve.correction import lead_bucket, lookup_bias

GTFS_DIR = config.ROOT / "data" / "gtfs"


def _parse_gtfs_time_to_seconds(hms: str) -> float:
    """gtfs times can go past 24:00:00, so this is a duration parse, not a clock time"""
    h, m, s = hms.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


def _haversine_m(lat1, lon1, lat2, lon2) -> float:
    r = 6_371_000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def load_stations() -> list[dict]:
    """one marker per station, not per platform"""
    latlon = pd.read_parquet(GTFS_DIR / "ro_stop_latlon.parquet")
    stop_times = pd.read_parquet(GTFS_DIR / "ro_stop_times.parquet")[["trip_id", "stop_id"]]
    trip_route = pd.read_parquet(GTFS_DIR / "ro_trip_route.parquet")[["trip_id", "route_id"]]
    stop_route = stop_times.merge(trip_route, on="trip_id")[["stop_id", "route_id"]].drop_duplicates()

    merged = latlon.merge(stop_route, on="stop_id", how="left")
    grouped = merged.groupby("stop_name").agg(
        lat=("stop_lat", "mean"), lon=("stop_lon", "mean"), route_id=("route_id", "first"),
    ).reset_index()
    return grouped.to_dict(orient="records")


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
        points = ordered[["shape_pt_lat", "shape_pt_lon"]].values.tolist()
        out[route_id].append(points)
    return out


class ShapeIndex:
    """shape points plus cumulative distance, for snapping a train onto real track"""

    def __init__(self):
        shapes = pd.read_parquet(GTFS_DIR / "ro_shapes.parquet")
        self._points: dict[str, np.ndarray] = {}
        self._cumdist: dict[str, np.ndarray] = {}
        for shape_id, group in shapes.groupby("shape_id"):
            ordered = group.sort_values("shape_pt_sequence")
            pts = ordered[["shape_pt_lat", "shape_pt_lon"]].to_numpy()
            cum = np.zeros(len(pts))
            for i in range(1, len(pts)):
                cum[i] = cum[i - 1] + _haversine_m(*pts[i - 1], *pts[i])
            self._points[shape_id] = pts
            self._cumdist[shape_id] = cum

    def _nearest_index(self, shape_id: str, lat: float, lon: float) -> int:
        pts = self._points[shape_id]
        d2 = (pts[:, 0] - lat) ** 2 + (pts[:, 1] - lon) ** 2
        return int(np.argmin(d2))

    def position_along(self, shape_id: str, prev_latlon, next_latlon, frac: float):
        """point frac of the way from prev to next, snapped onto the shape"""
        if shape_id not in self._points:
            return None
        i0 = self._nearest_index(shape_id, *prev_latlon)
        i1 = self._nearest_index(shape_id, *next_latlon)
        lo, hi = min(i0, i1), max(i0, i1)
        if lo == hi:
            return list(self._points[shape_id][lo])

        cum = self._cumdist[shape_id]
        leg_start, leg_end = cum[lo], cum[hi]
        target = leg_start + (leg_end - leg_start) * frac

        idx = np.searchsorted(cum[lo:hi + 1], target) + lo
        idx = min(max(idx, lo + 1), hi)
        seg_frac_denom = cum[idx] - cum[idx - 1]
        seg_frac = 0.0 if seg_frac_denom <= 0 else (target - cum[idx - 1]) / seg_frac_denom

        p0, p1 = self._points[shape_id][idx - 1], self._points[shape_id][idx]
        return [
            float(p0[0] + (p1[0] - p0[0]) * seg_frac),
            float(p0[1] + (p1[1] - p0[1]) * seg_frac),
        ]


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

        self.latlon = pd.read_parquet(GTFS_DIR / "ro_stop_latlon.parquet").set_index("stop_id")
        self.trip_info = pd.read_parquet(GTFS_DIR / "ro_trip_route.parquet").set_index("trip_id")

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
            prev = self.latlon.loc[prev_stop_id]
            nxt = self.latlon.loc[next_stop_id]
            trip = self.trip_info.loc[trip_id]
        except KeyError:
            return None
        if isinstance(trip, pd.DataFrame):
            trip = trip.iloc[0]

        return {
            "prev_latlon": (float(prev["stop_lat"]), float(prev["stop_lon"])),
            "next_latlon": (float(nxt["stop_lat"]), float(nxt["stop_lon"])),
            "next_stop_name": str(nxt["stop_name"]),
            "leg_duration_sec": float(leg_duration),
            "shape_id": str(trip["shape_id"]),
            "headsign": str(trip["trip_headsign"]),
            "direction_id": int(trip["direction_id"]),
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


def build_live_trains(
    schedule: ScheduleIndex, shape_index: ShapeIndex, lookup: dict, routes: list[str]
) -> list[dict]:
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
        bias, confidence = lookup_bias(lookup, route_id, pred["stop_id"], bucket)
        corrected_lead_sec = pred["lead_sec"] - bias

        raw_frac = 1 - min(max(pred["lead_sec"] / leg["leg_duration_sec"], 0), 1)
        corrected_frac = 1 - min(max(corrected_lead_sec / leg["leg_duration_sec"], 0), 1)

        shape_id, prev_ll, next_ll = leg["shape_id"], leg["prev_latlon"], leg["next_latlon"]
        raw_pos = shape_index.position_along(shape_id, prev_ll, next_ll, raw_frac)
        corrected_pos = shape_index.position_along(shape_id, prev_ll, next_ll, corrected_frac)
        if raw_pos is None or corrected_pos is None:
            continue

        results.append({
            "trip_id": trip_id,
            "route_id": route_id,
            "headsign": leg["headsign"],
            "next_stop_name": leg["next_stop_name"],
            "raw_pos": raw_pos,
            "corrected_pos": corrected_pos,
            "raw_min": round(pred["lead_sec"] / 60, 1),
            "corrected_min": round(corrected_lead_sec / 60, 1),
            "adjustment_sec": round(-bias),
            "confidence": confidence,
        })

    return results
