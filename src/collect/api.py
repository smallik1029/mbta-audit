"""thin MBTA v3 api client with retry and gap logging"""
import time
from datetime import UTC, datetime

import requests

from src import config, db


def utc_now_iso() -> str:
    """when we saw it. the observed_at stamp"""
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def rel_id(item: dict, name: str):
    """pull an id out of a JSON:API relationship, tolerating nulls anywhere"""
    rel = (item.get("relationships") or {}).get(name) or {}
    data = rel.get("data") or {}
    return data.get("id")


class Feed:
    """polls one endpoint, stamps observed_at the moment it arrives"""

    def __init__(self, name: str, path: str, params: dict):
        self.name = name
        self.path = path
        self.params = params
        self.session = requests.Session()
        self.session.headers.update({"x-api-key": config.API_KEY})
        self.backoff = 1.0
        self.gap_started: str | None = None

    def fetch(self) -> tuple[list[dict], str] | None:
        """returns (items, observed_at), or None on failure"""
        try:
            resp = self.session.get(
                f"{config.API_BASE}{self.path}",
                params=self.params,
                timeout=config.REQUEST_TIMEOUT,
            )
            observed_at = utc_now_iso()
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:
            if self.gap_started is None:
                self.gap_started = utc_now_iso()
            time.sleep(self.backoff)
            self.backoff = min(self.backoff * 2, config.MAX_BACKOFF_SECONDS)
            print(f"[{self.name}] fetch failed ({type(exc).__name__}: {exc}); "
                  f"backoff {self.backoff:.0f}s", flush=True)
            return None

        if self.gap_started is not None:
            db.log_gap(self.name, self.gap_started, observed_at, "poll failures")
            print(f"[{self.name}] recovered; gap {self.gap_started} -> {observed_at}", flush=True)
            self.gap_started = None
        self.backoff = 1.0
        return payload.get("data", []), observed_at
