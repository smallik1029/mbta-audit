"""shared logic for correcting a live mbta prediction"""
from datetime import UTC, datetime

import pandas as pd
import requests

from src import config, db

ARTIFACTS_DIR = config.ROOT / "model_artifacts"
LEAD_BUCKET_WIDTH_MIN = 3
LEAD_BUCKET_MAX_MIN = 30
UNCORRECTED_ROUTES = frozenset({"Orange"})


def load_lookup(artifacts_dir=None) -> dict:
    artifacts_dir = artifacts_dir or ARTIFACTS_DIR
    lookup = {}
    for key in ("by_stop", "by_route", "by_bucket"):
        path = artifacts_dir / f"bias_{key}.csv"
        if not path.exists():
            raise SystemExit(f"Missing {path} -- run `python -m src.model.train` first.")
        dtype = {"stop_id": str} if key == "by_stop" else None
        lookup[key] = pd.read_csv(path, dtype=dtype)
    return lookup


def lead_bucket(lead_time_sec: float) -> int:
    lead_min = min(lead_time_sec / 60.0, LEAD_BUCKET_MAX_MIN)
    return int(lead_min // LEAD_BUCKET_WIDTH_MIN * LEAD_BUCKET_WIDTH_MIN)


def lookup_bias(lookup: dict, route_id: str, stop_id: str, bucket: int) -> tuple[float, str, int]:
    """returns (bias_sec, confidence, sample_count)"""
    by_stop = lookup["by_stop"]
    hit = by_stop[
        (by_stop.route_id == route_id) & (by_stop.stop_id == stop_id) & (by_stop.lead_bucket == bucket)
    ]
    if len(hit):
        return float(hit.iloc[0]["bias_sec"]), "stop", int(hit.iloc[0]["n"])

    by_route = lookup["by_route"]
    hit = by_route[(by_route.route_id == route_id) & (by_route.lead_bucket == bucket)]
    if len(hit):
        return float(hit.iloc[0]["bias_sec"]), "route", int(hit.iloc[0]["n"])

    by_bucket = lookup["by_bucket"]
    hit = by_bucket[by_bucket.lead_bucket == bucket]
    if len(hit):
        return float(hit.iloc[0]["bias_sec"]), "global", int(hit.iloc[0]["n"])

    return 0.0, "none", 0


def fetch_live_predictions(stop_id: str, route_id: str | None) -> tuple[list[dict], list[dict]]:
    """predictions plus the trips they belong to, so we can name a destination"""
    params = {"filter[stop]": stop_id, "include": "trip"}
    if route_id:
        params["filter[route]"] = route_id
    resp = requests.get(
        f"{config.API_BASE}/predictions",
        params=params,
        headers={"x-api-key": config.API_KEY} if config.API_KEY else {},
        timeout=15,
    )
    resp.raise_for_status()
    payload = resp.json()
    return payload.get("data", []), payload.get("included", [])


def stop_name(stop_id: str) -> str:
    conn = db.connect()
    row = conn.execute("SELECT stop_name FROM gtfs_stops WHERE stop_id = ?", (stop_id,)).fetchone()
    return row["stop_name"] if row else stop_id


def get_corrected_predictions(stop_id: str, route_id: str | None, lookup: dict) -> list[dict]:
    """what both the cli and the web app call"""
    predictions, included = fetch_live_predictions(stop_id, route_id)
    headsigns = {
        item["id"]: (item.get("attributes") or {}).get("headsign")
        for item in included
        if item.get("type") == "trip"
    }
    now = datetime.now(UTC)
    results = []

    for item in predictions:
        attrs = item.get("attributes") or {}
        arrival = attrs.get("arrival_time")
        if not arrival:
            continue
        arrival_dt = pd.Timestamp(arrival).tz_convert("UTC")
        lead_sec = (arrival_dt - now).total_seconds()
        if lead_sec < 0:
            continue

        rels = item.get("relationships") or {}
        rel = rels.get("route") or {}
        this_route = (rel.get("data") or {}).get("id", route_id or "?")
        trip = (rels.get("trip") or {}).get("data") or {}
        bucket = lead_bucket(lead_sec)
        if this_route in UNCORRECTED_ROUTES:
            bias, level, n = 0.0, "uncorrected", 0
        else:
            bias, level, n = lookup_bias(lookup, this_route, stop_id, bucket)
        corrected_lead_sec = lead_sec - bias

        results.append({
            "route_id": this_route,
            "headsign": headsigns.get(trip.get("id")) or "",
            "arrival_iso": arrival_dt.isoformat(),
            "raw_min": round(lead_sec / 60, 1),
            "corrected_min": round(corrected_lead_sec / 60, 1),
            "adjustment_sec": round(-bias),
            "confidence": level,
            "sample_size": n,
        })

    return results
