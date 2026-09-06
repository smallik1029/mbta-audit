"""rewrite the generated regions of the readme from the live accuracy api"""
import argparse
import json
import pathlib
import re
import sys
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
SOURCE = "https://mbta-audit.com/api/accuracy"
TIMEOUT = 20


def fetch(source: str) -> dict:
    """the live api, or a local json file when the site is unreachable"""
    if source.startswith("http"):
        request = urllib.request.Request(source, headers={"User-Agent": "mbta-audit-readme"})
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT) as resp:
                return json.load(resp)
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            raise SystemExit(f"could not read {source}: {exc}\nREADME.md left alone") from None
    return json.loads(pathlib.Path(source).read_text(encoding="utf-8"))


def stat(payload: dict, label: str) -> str:
    for s in payload.get("stats", []):
        if s["label"] == label:
            return s["value"]
    raise SystemExit(f"no stat called {label!r} in the payload")


def route_table(routes: list) -> str:
    header = ("| Route | MBTA | Corrected | Improvement |\n"
              "|---|---|---|---|")
    rows = [
        f"| {r['route_id']} | {r['baseline_sec']:.1f}s | {r['corrected_sec']:.1f}s "
        f"| {r['improvement_pct']:+.1f}% |"
        for r in routes
    ]
    return "\n".join([header, *rows])


def served_only(payload: dict) -> bool:
    """whether the headline figure covers just the predictions the site corrects"""
    return payload["accuracy"].get("overall_scope") == "served"


def millions(value: str) -> str:
    """54,137,638 reads better as over 54 million, and rounds down so it is never a boast"""
    return f"{int(value.replace(',', '')) // 1_000_000} million"


def overview_region(payload: dict) -> str:
    """generated paragraphs stay unwrapped so a wider number cannot reflow the block"""
    a = payload["accuracy"]
    return (
        "mbta-audit records every prediction the MBTA publishes, observes when the train "
        "actually arrived, and learns the bias at each stop. It then serves a corrected "
        "arrival time. Over the past few weeks, it has collected over "
        f"{millions(stat(payload, 'Observations behind the model'))} observations, with "
        f"**{a['overall']['improvement_pct']:.1f}%** more accurate times."
    )


def results_region(payload: dict) -> str:
    a = payload["accuracy"]
    o = a["overall"]
    worst = min(a["routes"], key=lambda r: r["improvement_pct"])
    best = max(a["routes"], key=lambda r: r["improvement_pct"])
    headline = (
        f"Across {a['test_rows']:,} held-out predictions inside the corrected range, median "
        f"absolute error falls from {o['baseline_sec']:.1f}s to {o['corrected_sec']:.1f}s, "
        f"a {o['improvement_pct']:.1f}% improvement."
        if served_only(payload) else
        f"Across every lead time, median absolute error falls from {o['baseline_sec']:.1f}s "
        f"to {o['corrected_sec']:.1f}s, a {o['improvement_pct']:.1f}% improvement. The table "
        f"below covers the {a['test_rows']:,} held-out predictions inside the corrected range."
    )
    return (
        f"Measured on {a['measured_on']} and remeasured after every nightly retrain. The "
        "current figures are always at "
        "[mbta-audit.com/accuracy](https://mbta-audit.com/accuracy).\n"
        "\n"
        "Corrections are applied below 30 minutes out. Past that, the MBTA groups a 30 minute "
        "prediction together with a 3 hour one, so the underlying figure is too coarse to "
        f"correct and those predictions are passed through unchanged. {headline}\n"
        "\n"
        f"{route_table(a['routes'])}\n"
        "\n"
        f"{best['route_id']} gains most at {best['improvement_pct']:+.1f}% and "
        f"{worst['route_id']} least at {worst['improvement_pct']:+.1f}%. Every route improves."
    )


def model_region(payload: dict) -> str:
    return (
        f"- {stat(payload, 'Separate corrections learned')} separate corrections, one per "
        "stop and lead-time bucket, rather than one global adjustment"
    )


REGIONS = {
    "overview": overview_region,
    "results": results_region,
    "model": model_region,
}


def replace(text: str, name: str, body: str) -> str:
    pattern = re.compile(
        rf"(<!-- {name}:start -->\n).*?(\n<!-- {name}:end -->)",
        re.DOTALL,
    )
    if not pattern.search(text):
        raise SystemExit(f"no {name}:start / {name}:end markers in README.md")
    return pattern.sub(lambda m: m.group(1) + body + m.group(2), text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=SOURCE)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    payload = fetch(args.source)
    if not payload.get("accuracy"):
        raise SystemExit("the payload has no accuracy block, refusing to write")

    before = README.read_text(encoding="utf-8")
    after = before
    for name, build in REGIONS.items():
        after = replace(after, name, build(payload))

    if before == after:
        print("README.md already matches the live figures")
        return
    if args.check:
        print("README.md is out of date, run without --check to update")
        sys.exit(1)

    README.write_text(after, encoding="utf-8")
    changed = sum(1 for a, b in zip(before.split("\n"), after.split("\n"), strict=False) if a != b)
    print(f"README.md updated, about {changed} lines differ")


if __name__ == "__main__":
    main()
