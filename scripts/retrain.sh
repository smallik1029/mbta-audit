#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== retrain run started: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

python3.11 -m src.process.arrivals
python3.11 -m src.process.match
python3.11 -m src.model.train
python3.11 -m src.model.blue_line

python3.11 -m src.process.prune

echo "=== retrain run finished: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
