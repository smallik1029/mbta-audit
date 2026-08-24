"""read-only history api. runs on the same box as the collector"""
import sqlite3

import pandas as pd
from flask import Flask, jsonify, request

from src import config

app = Flask(__name__)

CANCELLED_STATES = {"CANCELLED", "SKIPPED"}
DEFAULT_WINDOW_MIN = 40


def _connect_readonly() -> sqlite3.Connection:
    uri = f"file:{config.DB_PATH}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def historical_state(t: pd.Timestamp, routes: list[str], window_min: int) -> pd.DataFrame:
    window_start = t - pd.Timedelta(minutes=window_min)
    placeholders = ",".join("?" * len(routes))
    conn = _connect_readonly()
    df = pd.read_sql_query(
        "SELECT trip_id, stop_id, route_id, predicted_arrival, observed_at, schedule_relationship "
        f"FROM prediction_snapshots WHERE observed_at <= ? AND observed_at > ? "
        f"AND route_id IN ({placeholders})",
        conn,
        params=[t.isoformat(), window_start.isoformat(), *routes],
    )
    conn.close()

    df = df[~df["schedule_relationship"].isin(CANCELLED_STATES)]
    df["observed_at"] = pd.to_datetime(df["observed_at"], utc=True, format="ISO8601")
    df["predicted_arrival"] = pd.to_datetime(df["predicted_arrival"], utc=True, format="ISO8601")

    df = df.sort_values("observed_at").drop_duplicates(subset=["trip_id", "stop_id"], keep="last")

    df["lead_sec"] = (df["predicted_arrival"] - t).dt.total_seconds()
    df = df[df["lead_sec"] >= 0]

    df = df.sort_values("lead_sec").drop_duplicates(subset=["trip_id"], keep="first")
    return df


@app.route("/historical_predictions")
def api_historical_predictions():
    t_param = request.args.get("t")
    routes_param = request.args.get("routes", "Red,Orange,Blue")
    window_min = int(request.args.get("window_min", DEFAULT_WINDOW_MIN))
    if not t_param:
        return jsonify({"error": "t (ISO8601 timestamp) is required"}), 400

    try:
        t = pd.Timestamp(t_param)
        if t.tzinfo is None:
            t = t.tz_localize("UTC")
        else:
            t = t.tz_convert("UTC")
    except ValueError:
        return jsonify({"error": f"could not parse timestamp: {t_param}"}), 400

    routes = [r.strip() for r in routes_param.split(",") if r.strip()]
    df = historical_state(t, routes, window_min)

    predictions = [
        {
            "trip_id": row.trip_id,
            "stop_id": row.stop_id,
            "route_id": row.route_id,
            "lead_sec": row.lead_sec,
        }
        for row in df.itertuples()
    ]
    return jsonify({"t": t.isoformat(), "predictions": predictions})


@app.route("/data_range")
def api_data_range():
    """how far back the current continuous run goes"""
    conn = _connect_readonly()
    gap_row = conn.execute("SELECT MAX(ended_at) FROM collector_gaps").fetchone()
    if gap_row and gap_row[0]:
        earliest = gap_row[0]
    else:
        earliest = conn.execute("SELECT MIN(observed_at) FROM prediction_snapshots").fetchone()[0]
    conn.close()
    return jsonify({"earliest": earliest})


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
