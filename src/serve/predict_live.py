"""cli: apply the correction table to a real, current prediction"""
import sys
from datetime import UTC, datetime

import pandas as pd
import requests

from src import config, db

ARTIFACTS_DIR = config.ROOT / "model_artifacts"
LEAD_BUCKET_WIDTH_MIN = 3
LEAD_BUCKET_MAX_MIN = 30


def load_lookup() -> dict:
    lookup = {}
    for key in ("by_stop", "by_route", "by_bucket"):
        path = ARTIFACTS_DIR / f"bias_{key}.csv"
        if not path.exists():
            raise SystemExit(f"Missing {path} -- run `python -m src.model.train` first.")
        dtype = {"stop_id": str} if key == "by_stop" else None
        lookup[key] = pd.read_csv(path, dtype=dtype)
    return lookup


def lead_bucket(lead_time_sec: float) -> int:
    lead_min = min(lead_time_sec / 60.0, LEAD_BUCKET_MAX_MIN)
    return int(lead_min // LEAD_BUCKET_WIDTH_MIN * LEAD_BUCKET_WIDTH_MIN)


def lookup_bias(lookup: dict, route_id: str, stop_id: str, bucket: int) -> tuple[float, str]:
    by_stop = lookup["by_stop"]
    hit = by_stop[
        (by_stop.route_id == route_id) & (by_stop.stop_id == stop_id) & (by_stop.lead_bucket == bucket)
    ]
    if len(hit):
        return float(hit.iloc[0]["bias_sec"]), "stop"

    by_route = lookup["by_route"]
    hit = by_route[(by_route.route_id == route_id) & (by_route.lead_bucket == bucket)]
    if len(hit):
        return float(hit.iloc[0]["bias_sec"]), "route"

    by_bucket = lookup["by_bucket"]
    hit = by_bucket[by_bucket.lead_bucket == bucket]
    if len(hit):
        return float(hit.iloc[0]["bias_sec"]), "global"

    return 0.0, "none"


def fetch_live_predictions(stop_id: str, route_id: str | None) -> list[dict]:
    params = {"filter[stop]": stop_id}
    if route_id:
        params["filter[route]"] = route_id
    resp = requests.get(
        f"{config.API_BASE}/predictions",
        params=params,
        headers={"x-api-key": config.API_KEY} if config.API_KEY else {},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json().get("data", [])


def stop_name(stop_id: str) -> str:
    conn = db.connect()
    row = conn.execute("SELECT stop_name FROM gtfs_stops WHERE stop_id = ?", (stop_id,)).fetchone()
    return row["stop_name"] if row else stop_id


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python -m src.serve.predict_live <stop_id> [route_id]")
    stop_id = sys.argv[1]
    route_id = sys.argv[2] if len(sys.argv) > 2 else None

    lookup = load_lookup()
    predictions = fetch_live_predictions(stop_id, route_id)
    now = datetime.now(UTC)

    print(f"Live predictions for {stop_name(stop_id)} ({stop_id}), fetched {now:%H:%M:%S} UTC:\n")
    if not predictions:
        print("No live predictions right now for this stop (train may not be running or "
              "outside service hours).")
        return

    for item in predictions:
        attrs = item.get("attributes") or {}
        arrival = attrs.get("arrival_time")
        if not arrival:
            continue
        arrival_dt = pd.Timestamp(arrival).tz_convert("UTC")
        lead_sec = (arrival_dt - now).total_seconds()
        if lead_sec < 0:
            continue

        rel = (item.get("relationships") or {}).get("route") or {}
        this_route = (rel.get("data") or {}).get("id", route_id or "?")
        bucket = lead_bucket(lead_sec)
        bias, level = lookup_bias(lookup, this_route, stop_id, bucket)
        corrected_dt = arrival_dt - pd.Timedelta(seconds=bias)
        corrected_lead_sec = (corrected_dt - now).total_seconds()

        print(f"  [{this_route}] MBTA says: {lead_sec / 60:5.1f} min "
              f"-> Corrected: {corrected_lead_sec / 60:5.1f} min "
              f"(adjustment: {-bias:+.0f}s, based on {level}-level history)")


if __name__ == "__main__":
    main()
