"""health facts for the status page, built only from cheap indexed lookups"""
import json
import shutil
from datetime import timedelta

import pandas as pd

from src import config, db
from src.serve.correction import UNCORRECTED_ROUTES

EVALUATION_PATH = config.ROOT / "model_artifacts" / "evaluation.json"
TRAIN_HOUR_UTC = 8
LOCAL_TZ = "America/New_York"
SERVICE_START_HOUR = 5
SERVICE_END_HOUR = 2
COLLECTOR_STALE_SEC = 900
PIPELINE_LAG_WARN_SEC = 10800
MODEL_STALE_SEC = 172800
HISTORY_DAYS = 30
PROBE_HOURS = (8, 12, 17, 21)
PROBE_SETTLE_SEC = 10800
CACHE_SECONDS = 60

_cache = None
_cache_at = None


def _scalar(sql: str):
    """one aggregate per query, never combined. see the index notes in the schema"""
    row = db.connect().execute(sql).fetchone()
    return row[0] if row else None


def _ts(value):
    if not value:
        return None
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _bound(stamp) -> str:
    """match the millisecond iso the collector writes, so string compares line up"""
    return stamp.tz_convert("UTC").isoformat(timespec="milliseconds")


def _in_service_hours(now_local) -> bool:
    hour = now_local.hour
    return hour >= SERVICE_START_HOUR or hour < SERVICE_END_HOUR


def _date_text(stamp) -> str:
    """portable, strftime has no %-d on windows"""
    return f"{stamp.strftime('%b')} {stamp.day}, {stamp.year}"


def _datetime_text(stamp) -> str:
    hour = stamp.hour % 12 or 12
    ampm = "AM" if stamp.hour < 12 else "PM"
    return f"{_date_text(stamp)} at {hour}:{stamp.minute:02d} {ampm}"


def _age_text(seconds: float) -> str:
    seconds = max(int(seconds), 0)
    if seconds < 90:
        return f"{seconds}s ago"
    if seconds < 5400:
        return f"{seconds // 60} min ago"
    if seconds < 172800:
        return f"{seconds // 3600}h ago"
    return f"{seconds // 86400}d ago"


def _duration_text(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60} min"
    hours, minutes = seconds // 3600, (seconds % 3600) // 60
    return f"{hours}h {minutes}m" if minutes else f"{hours}h"


def _state(key: str):
    row = db.connect().execute(
        "SELECT value FROM pipeline_state WHERE key = ?", (key,)
    ).fetchone()
    return _ts(row["value"]) if row else None


def _schedule(now):
    """when each timer last finished and when it is next due"""
    next_train = now.normalize() + timedelta(hours=TRAIN_HOUR_UTC)
    if next_train <= now:
        next_train += timedelta(days=1)
    jobs = [
        ("Hourly data update", _state("last_hourly_run"), (now + timedelta(hours=1)).floor("h")),
        ("Daily model train", _state("last_train_run"), next_train),
    ]
    return [
        {
            "name": name,
            "last": _age_text((now - last).total_seconds()) if last is not None else "not recorded yet",
            "next": "in " + _duration_text((nxt - now).total_seconds()),
        }
        for name, last, nxt in jobs
    ]


def _accuracy(now):
    """whatever the last evaluate run wrote, carrying its own measurement date"""
    if not EVALUATION_PATH.exists():
        return None
    try:
        report = json.loads(EVALUATION_PATH.read_text())
    except (OSError, ValueError):
        return None

    routes = [
        {
            "route_id": r["route_id"],
            "n": r["n"],
            "baseline_sec": r["baseline_sec"],
            "corrected_sec": r["corrected_sec"],
            "improvement_pct": r["improvement_pct"],
            "applied": r["route_id"] not in UNCORRECTED_ROUTES,
        }
        for r in report.get("by_route_served") or report.get("by_route", [])
    ]
    routes.sort(key=lambda r: r["improvement_pct"], reverse=True)

    measured = _ts(report.get("measured_at"))
    return {
        "measured_on": _date_text(measured.tz_convert(LOCAL_TZ)) if measured else "an unknown date",
        "measured_age": _age_text((now - measured).total_seconds()) if measured else "",
        "test_rows": report.get("served_rows") or report.get("test_rows"),
        "max_lead_min": report.get("served_max_lead_min"),
        "overall": report.get("overall"),
        "routes": routes,
    }


def _collection_history(now, first_seen):
    """downtime per day from the gaps the collector records itself"""
    gaps = db.connect().execute(
        "SELECT started_at, ended_at FROM collector_gaps ORDER BY started_at"
    ).fetchall()
    parsed = [(_ts(g["started_at"]), _ts(g["ended_at"]) or now) for g in gaps]

    days = []
    for offset in range(HISTORY_DAYS - 1, -1, -1):
        day_start = (now - timedelta(days=offset)).normalize()
        day_end = day_start + timedelta(days=1)
        label = day_start.strftime("%Y-%m-%d")
        if first_seen is None or day_end <= first_seen:
            days.append({"date": label, "pct": None, "detail": "before collection started"})
            continue
        covered = max((min(day_end, now) - max(day_start, first_seen)).total_seconds(), 0)
        if covered <= 0:
            days.append({"date": label, "pct": None, "detail": "before collection started"})
            continue
        down = 0.0
        for start, end in parsed:
            overlap = (min(end, day_end) - max(start, day_start)).total_seconds()
            if overlap > 0:
                down += overlap
        pct = max(0.0, min(100.0, (1 - down / covered) * 100))
        detail = "no gaps recorded" if down < 1 else f"{_duration_text(down)} of gaps recorded"
        days.append({"date": label, "pct": round(pct, 2), "detail": detail})
    return days


def _processing_history(now, first_seen):
    """sample four service hours a day and ask whether processed data landed in each"""
    conn = db.connect()
    now_local = now.tz_convert(LOCAL_TZ)
    days = []
    for offset in range(HISTORY_DAYS - 1, -1, -1):
        day_local = (now_local - timedelta(days=offset)).normalize()
        label = day_local.strftime("%Y-%m-%d")
        hits = 0
        probes = 0
        pending = False
        for hour in PROBE_HOURS:
            start_local = day_local + timedelta(hours=hour)
            if start_local + timedelta(seconds=PROBE_SETTLE_SEC) > now_local:
                pending = True
                continue
            start = start_local.tz_convert("UTC")
            end = start + timedelta(hours=1)
            if first_seen is not None and end <= first_seen:
                continue
            probes += 1
            row = conn.execute(
                "SELECT 1 FROM prediction_outcomes WHERE observed_at >= ? AND observed_at < ? LIMIT 1",
                (_bound(start), _bound(end)),
            ).fetchone()
            if row:
                hits += 1
        if probes == 0:
            detail = "no sampled hours settled yet" if pending else "before collection started"
            days.append({"date": label, "pct": None, "detail": detail})
        else:
            days.append({
                "date": label,
                "pct": round(hits / probes * 100, 2),
                "detail": f"{hits} of {probes} sampled hours had processed data",
            })
    return days


def build_status(model_mtime: float | None, observations: int, model_cells: int) -> dict:
    global _cache, _cache_at
    now = pd.Timestamp.now(tz="UTC")
    if _cache is not None and (now - _cache_at).total_seconds() < CACHE_SECONDS:
        return _cache

    last_prediction = _ts(_scalar("SELECT MAX(observed_at) FROM prediction_snapshots"))
    last_outcome = _ts(_scalar("SELECT MAX(observed_at) FROM prediction_outcomes"))
    first_seen = _state(db.COLLECTION_START_KEY) or _ts(
        _scalar("SELECT MIN(observed_at) FROM prediction_snapshots")
    )
    now_local = now.tz_convert(LOCAL_TZ)

    components = []

    if last_prediction is None:
        components.append({"name": "Data collection", "state": "down", "detail": "no data recorded"})
    else:
        age = (now - last_prediction).total_seconds()
        if age <= COLLECTOR_STALE_SEC:
            state, detail = "operational", f"last poll {_age_text(age)}"
        elif not _in_service_hours(now_local):
            state, detail = "idle", "no scheduled service right now"
        else:
            state, detail = "down", f"no data for {_age_text(age)}"
        components.append({"name": "Data collection", "state": state, "detail": detail})

    if last_outcome is None or last_prediction is None:
        components.append({
            "name": "Data processing",
            "state": "down",
            "detail": "nothing processed yet",
        })
    else:
        lag = (last_prediction - last_outcome).total_seconds()
        state = "operational" if lag <= PIPELINE_LAG_WARN_SEC else "degraded"
        components.append({
            "name": "Data processing",
            "state": state,
            "detail": f"caught up to within {_duration_text(lag)}",
        })

    if model_mtime is None:
        components.append({"name": "Correction model", "state": "down", "detail": "no model on disk"})
    else:
        age = now.timestamp() - model_mtime
        state = "operational" if age <= MODEL_STALE_SEC else "degraded"
        components.append({
            "name": "Correction model",
            "state": state,
            "detail": f"retrained {_age_text(age)}",
        })

    components.append({"name": "Website", "state": "operational", "detail": "serving this page"})

    usage = shutil.disk_usage(config.DB_PATH.parent)
    db_bytes = config.DB_PATH.stat().st_size if config.DB_PATH.exists() else 0
    since_text = _date_text(first_seen.tz_convert(LOCAL_TZ)) if first_seen else "n/a"

    stats = [
        {"label": "Observations behind the model", "value": f"{observations:,}"},
        {"label": "Stop and lead-time cells", "value": f"{model_cells:,}"},
        {"label": "Collecting since", "value": since_text},
        {"label": "Database size", "value": f"{db_bytes / 1e9:.1f} GB"},
        {"label": "Disk free", "value": f"{usage.free / 1e9:.0f} GB of {usage.total / 1e9:.0f} GB"},
    ]

    overall = "operational"
    if any(c["state"] == "down" for c in components):
        overall = "down"
    elif any(c["state"] == "degraded" for c in components):
        overall = "degraded"

    payload = {
        "overall": overall,
        "generated_at": _datetime_text(now_local),
        "components": components,
        "schedule": _schedule(now),
        "accuracy": _accuracy(now),
        "stats": stats,
        "series": [
            {
                "title": "Data collection",
                "note": "from the gaps the collector records when a poll fails or it restarts",
                "days": _collection_history(now, first_seen),
            },
            {
                "title": "Data processing",
                "note": "four service hours sampled per day, checking whether processed data landed in each",
                "days": _processing_history(now, first_seen),
            },
        ],
    }
    _cache, _cache_at = payload, now
    return payload
