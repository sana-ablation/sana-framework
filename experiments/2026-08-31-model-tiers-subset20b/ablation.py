#!/usr/bin/env python3
"""Model-tier ablation table, computed over COMPLETED rows only.

An earlier version divided correct answers by *all* rows, so crashed tasks
counted as wrong answers. That turned two infrastructure failures — an
RLIMIT_DATA cap firing during import, and code shipped into a live checkout so
workers imported half-written .py files — into a plausible-looking table where
gpt-5-mini read 0% across every cell and nano's plan axis read -35pp. None of it
was measurement.

So: accuracy is over completed rows, errored rows are reported separately, and a
cell with too few completions is refused rather than printed. Reads pulled
results locally; nothing is copied to the eval box while a run is alive.
"""

import csv
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results" / "modes"
MIN_COMPLETE = 15  # below this a 20-task cell cannot support a number

CELLS = [
    ("reference",         r"search_ideal.*profile_ideal__compute_ideal"),
    ("search: BM25",      r"search_naive.*profile_ideal__compute_ideal"),
    ("search: PNEUMA",    r"search_standard.*profile_ideal__compute_ideal"),
    ("search: Preloaded", r"search_preloaded.*profile_ideal__compute_ideal"),
    ("plan: No Plan",     r"search_ideal.*profile_naive__compute_ideal"),
    ("plan: Default",     r"search_ideal.*profile_standard__compute_ideal"),
    ("exec: Standard",    r"search_ideal.*profile_ideal__compute_standard"),
]


def _num(v, d=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


def load(model_dir: Path, pattern: str):
    hits = [d for d in model_dir.iterdir() if d.is_dir() and re.search(pattern, d.name)]
    if not hits:
        return None
    csv_path = hits[0] / "eval_results.csv"
    if not csv_path.is_file():
        return None
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    done = [r for r in rows if not (r.get("error") or "").strip()]
    errored = len(rows) - len(done)
    return done, errored


def main() -> int:
    if not RESULTS.is_dir():
        print(f"no results at {RESULTS} — run ./pull_remote.sh first")
        return 1

    for model_dir in sorted(RESULTS.iterdir()):
        if not model_dir.is_dir():
            continue
        print(f"\n=== {model_dir.name} ===")
        print(f"  {'cell':<20}{'done':>6}{'err':>5}{'semantic':>10}{'exact':>8}{'vs ref':>9}")
        ref = None
        for label, pattern in CELLS:
            got = load(model_dir, pattern)
            if got is None:
                print(f"  {label:<20}{'not run':>21}")
                continue
            done, errored = got
            if len(done) < MIN_COMPLETE:
                print(f"  {label:<20}{len(done):>6}{errored:>5}"
                      f"{'  too few completions':>27}")
                continue
            has_sem = "semantic_match" in (done[0] or {})
            sem = sum(1 for r in done if _num(r.get("semantic_match")) >= 1) if has_sem else None
            exact = sum(1 for r in done if _num(r.get("exact_match")) >= 1)
            pct = 100 * (sem if sem is not None else exact) / len(done)
            if label == "reference":
                ref = pct
            delta = "" if (ref is None or label == "reference") else f"{pct - ref:+.0f}pp"
            sem_s = f"{100*sem/len(done):>9.0f}%" if sem is not None else f"{'n/a':>10}"
            print(f"  {label:<20}{len(done):>6}{errored:>5}{sem_s}"
                  f"{100*exact/len(done):>7.0f}%{delta:>9}")

        if ref is not None:
            print(f"  (drops are vs reference; one task = {100/20:.0f}pp at n=20)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
