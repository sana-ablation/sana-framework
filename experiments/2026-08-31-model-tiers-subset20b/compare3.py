#!/usr/bin/env python3
"""Ablation across three replicate rounds: per-cell mean and spread.

With n=3 the noise floor stops being a mean-of-pairs and becomes a real
per-cell standard deviation. Accuracy is over COMPLETED rows only; errored rows
are reported separately and never enter the denominator.

Prefers semantic_match when the audited CSVs are present, else exact_match.
"""
import csv
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROUNDS = [("r1", "results"), ("r2", "results-rep2"), ("r3", "results-rep3")]
# --exact forces exact_match everywhere. Without it a partially-audited tree
# silently mixes semantic and exact cells, which are not comparable.
FORCE_EXACT = "--exact" in sys.argv
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


def score(tree: str, model: str, pattern: str):
    """Prefer the semantic-audited tree; fall back to raw. Returns pct or None."""
    roots = [(HERE / tree / "modes", False)] if FORCE_EXACT else [
        (HERE / f"{tree}_semantic" / "modes", True), (HERE / tree / "modes", False)]
    for root, is_sem in roots:
        md = root / model
        if not md.is_dir():
            continue
        hits = [d for d in md.iterdir() if d.is_dir() and re.search(pattern, d.name)]
        if not hits:
            continue
        path = hits[0] / "eval_results.csv"
        if not path.is_file():
            continue
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        done = [r for r in rows if not (r.get("error") or "").strip()]
        if len(done) < 15:
            continue
        key = "semantic_match" if is_sem and "semantic_match" in done[0] else "exact_match"
        ok = sum(1 for r in done if _num(r.get(key)) >= 1)
        return 100 * ok / len(done), key
    return None


def main() -> int:
    models = sorted({d.name
                     for _, t in ROUNDS
                     for root in [HERE / t / "modes"] if root.is_dir()
                     for d in root.iterdir() if d.is_dir()})
    if not models:
        print("no results found")
        return 1

    all_sd, metric_seen = [], set()
    for model in models:
        print(f"\n=== {model} ===")
        print(f"  {'cell':<20}{'r1':>7}{'r2':>7}{'r3':>7}{'mean':>8}{'sd':>7}{'vs ref':>9}")
        ref = None
        for label, pattern in CELLS:
            vals = []
            for _, tree in ROUNDS:
                got = score(tree, model, pattern)
                if got:
                    vals.append(got[0]); metric_seen.add(got[1])
            if not vals:
                print(f"  {label:<20}{'not run':>22}")
                continue
            m = statistics.mean(vals)
            sd = statistics.stdev(vals) if len(vals) > 1 else None
            if sd is not None:
                all_sd.append(sd)
            if label == "reference":
                ref = m
            cols = "".join(f"{v:>6.0f}%" for v in vals) + "".join(f"{'-':>7}" for _ in range(3 - len(vals)))
            vs = "" if label == "reference" or ref is None else f"{m - ref:+.1f}pp"
            sds = f"{sd:>6.1f}" if sd is not None else f"{'-':>7}"
            print(f"  {label:<20}{cols}{m:>7.1f}%{sds}{vs:>9}")

    if all_sd:
        pooled = statistics.mean(all_sd)
        print(f"\n=== noise floor across {len(all_sd)} cells with >=2 rounds ===")
        print(f"  mean per-cell SD   {pooled:.1f}pp")
        print(f"  median             {statistics.median(all_sd):.1f}pp")
        print(f"  max                {max(all_sd):.1f}pp")
        print(f"  ~95% interval      +/-{2 * pooled:.1f}pp   (2 SD)")
        print(f"\n  metric: {'/'.join(sorted(metric_seen))}")
        print(f"  An effect must exceed ~{2 * pooled:.0f}pp to be distinguishable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
