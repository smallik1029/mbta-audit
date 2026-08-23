"""web front end. live map and single-stop lookup"""
import pandas as pd
from flask import Flask, jsonify, render_template, request

from src import config, db
from src.serve.correction import get_corrected_predictions, load_lookup
from src.serve.live_map import ScheduleIndex, build_live_trains, load_shapes

app = Flask(__name__)
_lookup = None
_schedule = None
_shapes = None


def lookup() -> dict:
    global _lookup
    if _lookup is None:
        _lookup = load_lookup()
    return _lookup


def schedule() -> ScheduleIndex:
    global _schedule
    if _schedule is None:
        _schedule = ScheduleIndex()
    return _schedule


def shapes() -> dict:
    global _shapes
    if _shapes is None:
        _shapes = load_shapes()
    return _shapes


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/map")
def map_page():
    return render_template("map.html")


@app.route("/api/shapes")
def api_shapes():
    return jsonify(shapes())


@app.route("/api/live_map")
def api_live_map():
    try:
        trains = build_live_trains(schedule(), lookup(), config.ROUTES)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 502
    return jsonify({"trains": trains})


@app.route("/api/stops")
def api_stops():
    """stops we actually have history for, so every dropdown option works"""
    by_stop = lookup()["by_stop"]
    stop_route = by_stop[["route_id", "stop_id"]].drop_duplicates()

    conn = db.connect()
    names = pd.read_sql_query("SELECT stop_id, stop_name FROM gtfs_stops", conn)
    names["stop_id"] = names["stop_id"].astype(str)

    merged = stop_route.merge(names, on="stop_id", how="left")
    merged["stop_name"] = merged["stop_name"].fillna(merged["stop_id"])
    merged = merged.sort_values(["route_id", "stop_name"])

    return jsonify(merged.to_dict(orient="records"))


@app.route("/api/predict")
def api_predict():
    stop_id = request.args.get("stop_id")
    route_id = request.args.get("route_id")
    if not stop_id:
        return jsonify({"error": "stop_id is required"}), 400

    try:
        results = get_corrected_predictions(stop_id, route_id, lookup())
    except Exception as exc:
        return jsonify({"error": str(exc)}), 502

    return jsonify({"stop_id": stop_id, "route_id": route_id, "predictions": results})


if __name__ == "__main__":
    app.run(debug=True, port=5050)
