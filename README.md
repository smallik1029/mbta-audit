# mbta-audit

### [mbta-audit.com](https://mbta-audit.com)

[![Accuracy vs MBTA](https://img.shields.io/endpoint?url=https%3A%2F%2Fmbta-audit.com%2Fapi%2Fbadge)](https://mbta-audit.com/accuracy)

## Overview

Have you ever wondered: "Hey, the MBTA says that my train will be there in five minutes. How accurate truly is that?"
If you have, then mbta-audit.com is the tool for you.

<!-- overview:start -->
mbta-audit records every prediction the MBTA publishes, observes when the train actually arrived, and learns the bias at each stop. It then serves a corrected arrival time. Over the past few weeks, it has collected over 90 million observations, with **28.7%** more accurate times.
<!-- overview:end -->

## Features

### Site

- Live map of the Red, Orange and Blue lines plus all four Green Line branches
- Corrected arrival times shown beside the MBTA's own estimate at any stop
- Historical replay of recorded train positions and predictions
- Public [accuracy](https://mbta-audit.com/accuracy) and [status](https://mbta-audit.com/status) pages that refresh after every retrain

### Collection

- Polls the MBTA v3 API every 10 seconds for predictions and every 5 for vehicle positions
- Stores only revised predictions, discarding roughly half of every response
- Infers real arrival times from vehicle status transitions, which the MBTA never publishes

### Model

<!-- model:start -->
- 2,948 separate corrections, one per stop and lead-time bucket, rather than one global adjustment
<!-- model:end -->
- Trains on a pre-aggregated error histogram, roughly a tenfold reduction over raw rows
- Chronological holdout with the test window subtracted from the training data bin by bin
- Stops with too little history fall back to a route-level correction instead of guessing

### Operations

- Two systemd timers behind a shared lock file, hourly and daily
- Dead man's switch alerting that fires if the collector process dies
- Tiered retention with S3 archival gated on an upload watermark
- Tests and linting on every push

## Prerequisites

- Python 3.11
- A free MBTA v3 API key from [api-v3.mbta.com](https://api-v3.mbta.com)

## Installation

```bash
git clone https://github.com/smallik1029/mbta-audit.git
cd mbta-audit
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Usage

Start the collector and leave it running:

```bash
export MBTA_API_KEY=your_key
export MBTA_ROUTES=Red,Orange,Blue,Green-B,Green-C,Green-D,Green-E
python3.11 -m src.collect.run_collector                                # starts the collector
```

Once there is data, build the model and serve it:

```bash
python3.11 -m src.process.arrivals    # infer real arrival times
python3.11 -m src.process.match       # pair predictions with outcomes
python3.11 -m src.process.histogram   # fold outcomes into the histogram
python3.11 -m src.model.train         # fit the correction table
python3.11 -m src.model.evaluate      # measure it on a held-out split
python3.11 -m src.serve.app           # serve the site
```

In production the first four steps run hourly and the last two daily, both as systemd
timers. S3 archival stays off unless `S3_ARCHIVE_BUCKET` is set.

## External services

| Service | Used for | Key required |
|---|---|---|
| [MBTA v3 API](https://api-v3.mbta.com) | live predictions and vehicle positions | Yes, free |
| [MBTA GTFS feed](https://cdn.mbta.com/MBTA_GTFS.zip) | stop names and route shapes for the map | No |
| [healthchecks.io](https://healthchecks.io) | alerting when a scheduled job stops reporting | Optional |
| Amazon S3 | nightly model snapshots and archived outcomes | Optional, IAM role |

Only the MBTA v3 API key is required. Register at
[api-v3.mbta.com](https://api-v3.mbta.com) and pass it as `MBTA_API_KEY`. The GTFS feed is a
public download and needs no credentials. Without healthchecks.io the pipeline runs
normally and simply reports nowhere, and without `S3_ARCHIVE_BUCKET` archival is skipped.

## Results

<!-- results:start -->
Measured on Sep 11, 2026 and remeasured after every nightly retrain. The current figures are always at [mbta-audit.com/accuracy](https://mbta-audit.com/accuracy).

Corrections are applied below 30 minutes out. Past that, the MBTA groups a 30 minute prediction together with a 3 hour one, so the underlying figure is too coarse to correct and those predictions are passed through unchanged. Across 177,163 held-out predictions inside the corrected range, median absolute error falls from 73.9s to 52.6s, a 28.7% improvement.

| Route | MBTA | Corrected | Improvement |
|---|---|---|---|
| Red | 103.9s | 52.6s | +49.4% |
| Green-D | 74.6s | 53.4s | +28.4% |
| Green-B | 79.5s | 57.8s | +27.2% |
| Orange | 32.3s | 23.5s | +27.2% |
| Blue | 36.2s | 27.4s | +24.3% |
| Green-E | 90.2s | 68.9s | +23.6% |
| Green-C | 72.0s | 59.9s | +16.8% |

Red gains most at +49.4% and Green-C least at +16.8%. Every route improves.
<!-- results:end -->

The split is chronological rather than random, and the shipped and evaluated lookups are
built by the same function, so a restriction present in one cannot be missing from the
other.

## Stack

Python 3.11, SQLite, pandas, Flask, gunicorn, pyarrow, boto3, systemd, Cloudflare Tunnel,
AWS EC2 and S3. Tested with pytest and linted with ruff on every push.

## License

MIT