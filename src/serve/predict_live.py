"""cli: apply the correction table to a real, current prediction"""
import sys
from datetime import UTC, datetime

from src.serve.correction import get_corrected_predictions, load_lookup, stop_name


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python -m src.serve.predict_live <stop_id> [route_id]")
    stop_id = sys.argv[1]
    route_id = sys.argv[2] if len(sys.argv) > 2 else None

    lookup = load_lookup()
    results = get_corrected_predictions(stop_id, route_id, lookup)
    now = datetime.now(UTC)

    print(f"Live predictions for {stop_name(stop_id)} ({stop_id}), fetched {now:%H:%M:%S} UTC:\n")
    if not results:
        print("No live predictions right now for this stop (train may not be running or "
              "outside service hours).")
        return

    for r in results:
        print(f"  [{r['route_id']}] MBTA says: {r['raw_min']:5.1f} min "
              f"-> Corrected: {r['corrected_min']:5.1f} min "
              f"(adjustment: {r['adjustment_sec']:+d}s, based on {r['confidence']}-level history, "
              f"n={r['sample_size']:,})")


if __name__ == "__main__":
    main()
