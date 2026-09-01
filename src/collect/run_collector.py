"""collector entrypoint, runs both feeds at once"""
import signal
import threading
import time
from datetime import UTC, datetime

from src import config, db
from src.collect.api import utc_now_iso
from src.collect.predictions import PredictionCollector
from src.collect.vehicles import VehicleCollector

_stop = threading.Event()


GAP_SOURCE_TABLES = ("prediction_snapshots", "vehicle_snapshots")


def latest_observed_at() -> str | None:
    """newest observed_at across both feeds. one query per table so each uses its index"""
    conn = db.connect()
    stamps = [
        conn.execute(f"SELECT MAX(observed_at) FROM {table}").fetchone()[0]
        for table in GAP_SOURCE_TABLES
    ]
    return max((s for s in stamps if s), default=None)


def _record_restart_gap() -> None:
    """time since the last row is a gap"""
    last = latest_observed_at()
    if last:
        db.log_gap("collector", last, utc_now_iso(), "collector not running")
        print(f"[startup] logged restart gap since {last}", flush=True)


def _run_feed(collector, interval: float, label: str) -> None:
    while not _stop.is_set():
        started = time.monotonic()
        try:
            collector.poll_once()
        except Exception as exc:
            print(f"[{label}] unexpected error: {type(exc).__name__}: {exc}", flush=True)
        elapsed = time.monotonic() - started
        _stop.wait(max(0.0, interval - elapsed))


def _report(preds, vehs, started_at: float) -> None:
    while not _stop.is_set():
        _stop.wait(60)
        if _stop.is_set():
            break
        mins = (time.monotonic() - started_at) / 60
        kept_p = 100 * preds.total_written / max(preds.total_seen, 1)
        print(
            f"[{datetime.now(UTC):%H:%M:%S}Z] "
            f"up {mins:5.1f}m | predictions {preds.total_written:>7,} rows "
            f"({kept_p:4.1f}% of {preds.total_seen:,} seen) | "
            f"vehicles {vehs.total_written:>6,} rows | "
            f"{preds.total_written / max(mins, 0.01):,.0f} pred-rows/min",
            flush=True,
        )


def main() -> None:
    if not config.API_KEY:
        raise SystemExit(
            "MBTA_API_KEY is not set.\n"
            "  1. copy .env.example to .env\n"
            "  2. paste your key after MBTA_API_KEY="
        )

    db.init_schema()
    _record_restart_gap()

    preds = PredictionCollector()
    vehs = VehicleCollector()

    print(f"routes={config.ROUTES} db={config.DB_PATH}", flush=True)
    print(f"poll: predictions {config.PREDICTIONS_POLL_SECONDS}s, "
          f"vehicles {config.VEHICLES_POLL_SECONDS}s", flush=True)
    print("collecting -- Ctrl+C to stop", flush=True)

    started_at = time.monotonic()
    threads = [
        threading.Thread(
            target=_run_feed,
            args=(preds, config.PREDICTIONS_POLL_SECONDS, "predictions"),
            daemon=True,
        ),
        threading.Thread(
            target=_run_feed,
            args=(vehs, config.VEHICLES_POLL_SECONDS, "vehicles"),
            daemon=True,
        ),
        threading.Thread(target=_report, args=(preds, vehs, started_at), daemon=True),
    ]
    for t in threads:
        t.start()

    def _shutdown(*_):
        print("\nstopping...", flush=True)
        _stop.set()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        while not _stop.is_set():
            time.sleep(0.5)
    except KeyboardInterrupt:
        _shutdown()

    for t in threads:
        t.join(timeout=5)
    print(f"final: wrote {preds.total_written:,} prediction rows, "
          f"{vehs.total_written:,} vehicle rows this run", flush=True)


if __name__ == "__main__":
    main()
