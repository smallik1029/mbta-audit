#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."

ping_health() {
    if [ -n "${HEALTHCHECK_URL:-}" ]; then
        curl -fsS -m 10 --retry 3 "${HEALTHCHECK_URL}$1" > /dev/null 2>&1 || true
    fi
}

finish() {
    local code=$?
    if [ "$code" -eq 0 ]; then ping_health ""; else ping_health "/fail"; fi
}
trap finish EXIT

if command -v flock >/dev/null 2>&1; then
    exec 200>/tmp/mbta-retrain.lock
    flock -n 200 || { echo "=== retrain already running, skipping this trigger ==="; exit 0; }
else
    echo "!!! flock unavailable -- running without a concurrency guard"
fi

collector_ok=1
if ! systemctl is-active --quiet mbta-collector; then
    echo "!!! mbta-collector is not active"
    collector_ok=0
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

python3.11 -m src.process.mark_run last_hourly_run

echo "=== retrain run finished: $(date -u +%Y-%m-%dT%H:%M:%SZ) (total $(( $(date +%s) - overall_start ))s) ==="

[ "$collector_ok" -eq 1 ] || exit 1
