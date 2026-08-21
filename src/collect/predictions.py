"""prediction poller"""
from src import config, db
from src.collect.api import Feed, rel_id

COLUMNS = [
    "observed_at", "prediction_id", "trip_id", "stop_id", "route_id",
    "direction_id", "vehicle_id", "predicted_arrival", "predicted_departure",
    "schedule_relationship", "stop_sequence",
]


class PredictionCollector:
    def __init__(self):
        self.feed = Feed(
            "predictions",
            "/predictions",
            {"filter[route]": ",".join(config.ROUTES)},
        )
        self.last_seen: dict[tuple[str, str], tuple] = {}
        self.total_written = 0
        self.total_seen = 0

    def poll_once(self) -> int:
        result = self.feed.fetch()
        if result is None:
            return 0
        items, observed_at = result

        rows = []
        for item in items:
            attrs = item.get("attributes") or {}
            trip_id = rel_id(item, "trip")
            stop_id = rel_id(item, "stop")
            if not trip_id or not stop_id:
                continue

            arrival = attrs.get("arrival_time")
            departure = attrs.get("departure_time")
            sched_rel = attrs.get("schedule_relationship")

            key = (trip_id, stop_id)
            value = (arrival, departure, sched_rel)
            self.total_seen += 1
            if self.last_seen.get(key) == value:
                continue
            self.last_seen[key] = value

            rows.append((
                observed_at,
                item.get("id"),
                trip_id,
                stop_id,
                rel_id(item, "route"),
                attrs.get("direction_id"),
                rel_id(item, "vehicle"),
                arrival,
                departure,
                sched_rel,
                attrs.get("stop_sequence"),
            ))

        written = db.insert_many("prediction_snapshots", COLUMNS, rows)
        self.total_written += written
        return written
