
PRAGMA journal_mode = WAL;
PRAGMA synchronous  = NORMAL;
PRAGMA auto_vacuum   = INCREMENTAL;

CREATE TABLE IF NOT EXISTS prediction_snapshots (
    id                    INTEGER PRIMARY KEY,
    observed_at           TEXT    NOT NULL,
    prediction_id         TEXT,
    trip_id               TEXT    NOT NULL,
    stop_id               TEXT    NOT NULL,
    route_id              TEXT    NOT NULL,
    direction_id          INTEGER,
    vehicle_id            TEXT,
    predicted_arrival     TEXT,
    predicted_departure   TEXT,
    schedule_relationship TEXT,
    stop_sequence         INTEGER
);
CREATE INDEX IF NOT EXISTS idx_pred_trip_stop ON prediction_snapshots (trip_id, stop_id);
CREATE INDEX IF NOT EXISTS idx_pred_observed  ON prediction_snapshots (observed_at);

CREATE TABLE IF NOT EXISTS vehicle_snapshots (
    id              INTEGER PRIMARY KEY,
    observed_at     TEXT    NOT NULL,
    vehicle_id      TEXT    NOT NULL,
    trip_id         TEXT,
    route_id        TEXT,
    direction_id    INTEGER,
    current_stop_id TEXT,
    current_status  TEXT,
    stop_sequence   INTEGER,
    latitude        REAL,
    longitude       REAL,
    updated_at      TEXT
);
CREATE INDEX IF NOT EXISTS idx_veh_trip_stop ON vehicle_snapshots (trip_id, current_stop_id);
CREATE INDEX IF NOT EXISTS idx_veh_observed  ON vehicle_snapshots (observed_at);
CREATE INDEX IF NOT EXISTS idx_veh_vehicle   ON vehicle_snapshots (vehicle_id, observed_at);

CREATE TABLE IF NOT EXISTS collector_gaps (
    id         INTEGER PRIMARY KEY,
    feed       TEXT NOT NULL,
    started_at TEXT NOT NULL,
    ended_at   TEXT,
    reason     TEXT
);

CREATE TABLE IF NOT EXISTS actual_arrivals (
    id                  INTEGER PRIMARY KEY,
    trip_id             TEXT NOT NULL,
    stop_id             TEXT NOT NULL,
    route_id            TEXT,
    direction_id        INTEGER,
    vehicle_id          TEXT,
    actual_arrival      TEXT NOT NULL,
    incoming_at         TEXT,
    detection_bound_sec REAL,
    service_date        TEXT NOT NULL,
    UNIQUE (trip_id, stop_id, service_date)
);
CREATE INDEX IF NOT EXISTS idx_arr_trip_stop ON actual_arrivals (trip_id, stop_id);

CREATE TABLE IF NOT EXISTS prediction_outcomes (
    id                INTEGER PRIMARY KEY,
    trip_id           TEXT NOT NULL,
    stop_id           TEXT NOT NULL,
    route_id          TEXT,
    direction_id      INTEGER,
    observed_at       TEXT NOT NULL,
    predicted_arrival TEXT NOT NULL,
    actual_arrival    TEXT NOT NULL,
    lead_time_sec     REAL NOT NULL,
    error_sec         REAL NOT NULL,
    hour_local        INTEGER,
    is_rush_hour      INTEGER,
    UNIQUE (trip_id, stop_id, observed_at)
);
CREATE INDEX IF NOT EXISTS idx_out_route_stop ON prediction_outcomes (route_id, stop_id);
CREATE INDEX IF NOT EXISTS idx_out_observed ON prediction_outcomes (observed_at);

CREATE TABLE IF NOT EXISTS outcome_histogram (
    route_id    TEXT    NOT NULL,
    stop_id     TEXT    NOT NULL,
    lead_bucket INTEGER NOT NULL,
    month       TEXT    NOT NULL,
    error_bin   INTEGER NOT NULL,
    n           INTEGER NOT NULL,
    PRIMARY KEY (route_id, stop_id, lead_bucket, month, error_bin)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS pipeline_state (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS gtfs_stops (
    stop_id   TEXT PRIMARY KEY,
    stop_name TEXT,
    parent_station TEXT
);
CREATE TABLE IF NOT EXISTS gtfs_routes (
    route_id    TEXT PRIMARY KEY,
    route_name  TEXT,
    route_type  INTEGER
);
