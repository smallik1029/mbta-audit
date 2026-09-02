"""stamp a pipeline_state key with the time a scheduled job finished"""
import sys
from datetime import UTC, datetime

from src import db


def mark(key: str) -> str:
    stamp = datetime.now(UTC).isoformat(timespec="milliseconds")
    conn = db.connect()
    conn.execute(
        "INSERT INTO pipeline_state (key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, stamp),
    )
    conn.commit()
    return stamp


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m src.process.mark_run <key>")
    key = sys.argv[1]
    print(f"marked {key} at {mark(key)}")


if __name__ == "__main__":
    main()
