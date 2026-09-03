#!/usr/bin/env python3
"""Ablation with replicate spread: run 1 vs run 2, same config.

Accuracy is over COMPLETED rows; errored rows are counted separately and never
folded into the denominator. The two runs differ only in sampling, so |run1-run2|
per cell is the empirical noise floor — the number that says which ablation drops
are real at n=20, where one task is 5pp.
"""

import csv
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = {"run1": HERE / "results" / "modes", "run2": HERE / "results-rep2" / "modes"}

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


def cell_score(model_dir: Path, pattern: str):
    """Return (pct, n_done, n_err) or None."""
    if not model_dir.is_dir():
        return None
    hits = [d for d in model_dir.iterdir() if d.is_dir() and re.search(pattern, d.name)]
    if not hits:
        return None
    path = hits[0] / "eval_results.csv"
    if not path.is_file():
        return None
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    done = [r for r in rows if not (r.get("error") or "").strip()]
    if not done:
        return None
    key = "semantic_match" if "semantic_match" in done[0] else "exact_match"
    ok = sum(1 for r in done if _num(r.get(key)) >= 1)
    return 100 * ok / len(done), len(done), len(rows) - len(done)


def main() -> int:
    models = sorted({d.name for root in RUNS.values() if root.is_dir() for d in root.iterdir() if d.is_dir()})
    if not models:
        print("no results found — pull them first")
        return 1

    spreads = []
    for model in models:
        print(f"\n=== {model} ===")
        print(f"  {'cell':<20}{'run1':>7}{'run2':>7}{'|Δ|':>6}{'mean':>7}{'vs ref':>9}")
        refs = {}
        for label, pattern in CELLS:
            got = {k: cell_score(root / model, pattern) for k, root in RUNS.items()}
            r1, r2 = got.get("run1"), got.get("run2")
            if r1 is None and r2 is None:
                continue
            p1 = r1[0] if r1 else None
            p2 = r2[0] if r2 else None
            if p1 is not None and p2 is not None:
                d = abs(p1 - p2)
                spreads.append(d)
                mean = (p1 + p2) / 2
            else:
                d = None
                mean = p1 if p1 is not None else p2
            if label == "reference":
                refs["mean"] = mean
            vs = "" if label == "reference" or "mean" not in refs else f"{mean - refs['mean']:+.0f}pp"
            s1 = f"{p1:>6.0f}%" if p1 is not None else f"{'-':>7}"
            s2 = f"{p2:>6.0f}%" if p2 is not None else f"{'-':>7}"
            ds = f"{d:>5.0f}" if d is not None else f"{'-':>6}"
            print(f"  {label:<20}{s1}{s2}{ds}{mean:>6.0f}%{vs:>9}")

    if spreads:
        print(f"\n=== replicate noise floor (|run1 - run2| across {len(spreads)} paired cells) ===")
        print(f"  mean |Δ|   {statistics.mean(spreads):.1f}pp")
        print(f"  median |Δ| {statistics.median(spreads):.1f}pp")
        print(f"  max |Δ|    {max(spreads):.1f}pp")
        print(f"\n  An ablation drop must clear ~{statistics.mean(spreads):.0f}pp to beat run-to-run noise.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
