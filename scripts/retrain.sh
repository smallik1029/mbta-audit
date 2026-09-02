#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."

if command -v flock >/dev/null 2>&1; then
    exec 200>/tmp/mbta-retrain.lock
    flock -n 200 || { echo "=== retrain already running, skipping this trigger ==="; exit 0; }
else
    echo "!!! flock unavailable -- running without a concurrency guard"
fi

run_step() {
    local label="$1"; shift
    local start; start=$(date +%s)
    "$@"
    echo "--- ${label} finished in $(( $(date +%s) - start ))s"
}

echo "=== retrain run started: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
overall_start=$(date +%s)

run_step arrivals  python3.11 -m src.process.arrivals
run_step match     python3.11 -m src.process.match
run_step histogram python3.11 -m src.process.histogram
run_step prune     python3.11 -m src.process.prune

echo "=== retrain run finished: $(date -u +%Y-%m-%dT%H:%M:%SZ) (total $(( $(date +%s) - overall_start ))s) ==="
