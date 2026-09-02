#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."

if command -v flock >/dev/null 2>&1; then
    exec 200>/tmp/mbta-retrain.lock
    flock -w 3600 200 || { echo "=== gave up waiting for the retrain lock, skipping this trigger ==="; exit 0; }
else
    echo "!!! flock unavailable -- running without a concurrency guard"
fi

echo "=== train run started: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
start=$(date +%s)

python3.11 -m src.model.train

echo "=== train run finished: $(date -u +%Y-%m-%dT%H:%M:%SZ) (total $(( $(date +%s) - start ))s) ==="
