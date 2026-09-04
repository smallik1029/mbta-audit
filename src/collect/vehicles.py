"""vehicle position poller. this is the ground truth feed"""
from src import config, db
from src.collect.api import Feed, rel_id

COLUMNS = [
    "observed_at", "vehicle_id", "trip_id", "route_id", "direction_id",
    "current_stop_id", "current_status", "stop_sequence", "latitude",
    "longitude", "updated_at",
]


class VehicleCollector:
    def __init__(self):
        self.feed = Feed(
            "vehicles",
            "/vehicles",
            {"filter[route]": ",".join(config.ROUTES)},
        )
        self.last_seen: dict[str, tuple] = {}
        self.total_written = 0
        self.total_seen = 0

    def poll_once(self) -> int:
        result = self.feed.fetch()
        if result is None:
            return 0
        items, observed_at = result

        rows = []
        pending: dict[str, tuple] = {}
        for item in items:
            attrs = item.get("attributes") or {}
            vehicle_id = item.get("id")
            if not vehicle_id:
                continue

            trip_id = rel_id(item, "trip")
            stop_id = rel_id(item, "stop")
            status = attrs.get("current_status")

            key = vehicle_id
            value = (status, stop_id, trip_id)
            self.total_seen += 1
            if pending.get(key, self.last_seen.get(key)) == value:
                continue
            pending[key] = value

            rows.append((
                observed_at,
                vehicle_id,
                trip_id,
                rel_id(item, "route"),
                attrs.get("direction_id"),
                stop_id,
                status,
                attrs.get("current_stop_sequence"),
                attrs.get("latitude"),
                attrs.get("longitude"),
                attrs.get("updated_at"),
            ))

        written = db.insert_many("vehicle_snapshots", COLUMNS, rows)
        self.last_seen.update(pending)
        self.total_written += written
        return written
