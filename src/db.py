"""sqlite connection and schema setup"""
import sqlite3
import threading

from src import config

_lock = threading.Lock()
_conn: sqlite3.Connection | None = None


def connect() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(config.DB_PATH, check_same_thread=False, timeout=30)
        _conn.row_factory = sqlite3.Row
        _conn.execute("PRAGMA journal_mode = WAL")
        _conn.execute("PRAGMA synchronous = NORMAL")
    return _conn


def init_schema() -> None:
    conn = connect()
    with _lock:
        conn.executescript(config.SCHEMA_PATH.read_text())
        conn.commit()


def insert_many(table: str, columns: list[str], rows: list[tuple]) -> int:
    if not rows:
        return 0
    placeholders = ",".join("?" * len(columns))
    sql = f"INSERT INTO {table} ({','.join(columns)}) VALUES ({placeholders})"
    conn = connect()
    with _lock:
        conn.executemany(sql, rows)
        conn.commit()
    return len(rows)


def log_gap(feed: str, started_at: str, ended_at: str, reason: str) -> None:
    conn = connect()
    with _lock:
        conn.execute(
            "INSERT INTO collector_gaps (feed, started_at, ended_at, reason) VALUES (?,?,?,?)",
            (feed, started_at, ended_at, reason[:500]),
        )
        conn.commit()


def count(table: str) -> int:
    conn = connect()
    with _lock:
        return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
