"""central config, sourced from .env"""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

API_KEY = os.getenv("MBTA_API_KEY", "").strip()
API_BASE = "https://api-v3.mbta.com"

ROUTES = [r.strip() for r in os.getenv("MBTA_ROUTES", "Red,Orange").split(",") if r.strip()]

DB_PATH = ROOT / os.getenv("DB_PATH", "data/mbta.db")
SCHEMA_PATH = ROOT / "sql" / "schema.sql"

PREDICTIONS_POLL_SECONDS = float(os.getenv("PREDICTIONS_POLL_SECONDS", "10"))
VEHICLES_POLL_SECONDS = float(os.getenv("VEHICLES_POLL_SECONDS", "5"))

HISTORY_API_URL = os.getenv("HISTORY_API_URL", "http://3.143.170.113:8000")

REQUEST_TIMEOUT = 15
MAX_BACKOFF_SECONDS = 60

LOCAL_TZ = "America/New_York"
RUSH_HOURS = {7, 8, 9, 16, 17, 18}


def headers() -> dict:
    if not API_KEY:
        raise SystemExit(
            "MBTA_API_KEY is not set.\n"
            "  cp .env.example .env   then paste your key into .env"
        )
    return {"x-api-key": API_KEY, "accept": "text/event-stream"}
