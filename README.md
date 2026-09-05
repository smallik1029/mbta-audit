# mbta-audit

### [mbta-audit.com](https://mbta-audit.com)

[![Accuracy vs MBTA](https://img.shields.io/endpoint?url=https%3A%2F%2Fmbta-audit.com%2Fapi%2Fbadge)](https://mbta-audit.com/accuracy)

## Overview

The MBTA says your train is four minutes away. It is usually not four minutes away, and how
wrong it is depends on which stop you are standing at and how far ahead you asked.

<!-- overview:start -->
mbta-audit records every prediction the MBTA publishes, observes when the train actually arrived, and learns the bias at each stop. It then serves a corrected arrival time. Measured on held-out predictions at every lead time the model was never fit on, corrected times are **26.3% more accurate** than the MBTA's own.

It has run continuously on a single EC2 instance since August 21, 2026, and has collected over 54 million observations.
<!-- overview:end -->

## Features

### Site

- Live map of the Red, Orange and Blue lines plus all four Green Line branches
- Corrected arrival times shown beside the MBTA's own estimate at any stop
- Historical replay of recorded train positions and predictions
- Public [accuracy](https://mbta-audit.com/accuracy) and [status](https://mbta-audit.com/status) pages, refreshed after every retrain

### Collection

- Polls the MBTA v3 API every 10 seconds for predictions and every 5 for vehicle positions
- Stores only revised predictions, discarding roughly half of every response
- Infers real arrival times from vehicle status transitions, which the MBTA never publishes

### Model

<!-- model:start -->
- 2,911 separate corrections, one per stop and lead-time bucket, rather than one global adjustment
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
python3.11 -m src.collect.run_collector
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
Measured on Sep 4, 2026 and remeasured after every nightly retrain. The current figures are always at [mbta-audit.com/accuracy](https://mbta-audit.com/accuracy).

Corrections are applied below 30 minutes out. Past that the MBTA groups a 30 minute prediction together with a three hour one, so the underlying figure is too coarse to correct and those predictions are passed through unchanged. Across every lead time, median absolute error falls from 129.7s to 95.6s, a 26.3% improvement. The table below covers the 178,265 held-out predictions inside the corrected range.

| Route | MBTA | Corrected | Improvement |
|---|---|---|---|
| Red | 76.1s | 46.0s | +39.6% |
| Green-B | 89.1s | 59.1s | +33.6% |
| Green-D | 71.0s | 52.5s | +26.1% |
| Green-E | 98.0s | 76.1s | +22.3% |
| Orange | 36.0s | 29.2s | +18.7% |
| Green-C | 72.0s | 60.5s | +16.0% |
| Blue | 46.5s | 44.7s | +3.7% |

Blue gains least because the MBTA already predicts it within 46 seconds, leaving little to recover.
<!-- results:end -->

The split is chronological rather than random, and the shipped and evaluated lookups are
built by the same function, so a restriction present in one cannot be missing from the
other.

## Known issues

- **Stops with fewer than 200 recorded arrivals fall back to the route-level correction**
  rather than a stop-specific one, so quieter stops are corrected less precisely than busy
  ones until they accumulate enough history.
- **The model measures error per prediction update, not per rider.** Because only changed
  predictions are stored, a prediction the MBTA revises frequently contributes more samples
  than a stable one. This weighting is reasonable for the correction itself but is worth
  knowing when reading the headline figure.
- **Prediction trajectories are not recoverable.** The histogram cannot distinguish a
  prediction that converged on the truth from one that drifted away, since both produce
  identical bins.

## Stack

Python 3.11, SQLite, pandas, Flask, gunicorn, pyarrow, boto3, systemd, Cloudflare Tunnel,
AWS EC2 and S3. Tested with pytest and linted with ruff on every push.

## License

MIT