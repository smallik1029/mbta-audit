#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."

ping_health() {
    if [ -n "${TRAIN_HEALTHCHECK_URL:-}" ]; then
        curl -fsS -m 10 --retry 3 "${TRAIN_HEALTHCHECK_URL}$1" > /dev/null 2>&1 || true
    fi
}

finish() {
    local code=$?
    if [ "$code" -eq 0 ]; then ping_health ""; else ping_health "/fail"; fi
}
trap finish EXIT

if command -v flock >/dev/null 2>&1; then
    exec 200>/tmp/mbta-retrain.lock
    flock -w 3600 200 || { echo "=== gave up waiting for the retrain lock, skipping this trigger ==="; exit 0; }
else
    echo "!!! flock unavailable -- running without a concurrency guard"
fi

echo "=== train run started: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
start=$(date +%s)

python3.11 -m src.model.train

python3.11 -m src.process.mark_run last_train_run

python3.11 -m src.model.evaluate || echo "!!! evaluate failed, the model itself is unaffected"

python3.11 -m src.process.backup || echo "!!! histogram snapshot failed, the model itself is unaffected"

echo "=== train run finished: $(date -u +%Y-%m-%dT%H:%M:%SZ) (total $(( $(date +%s) - start ))s) ==="
