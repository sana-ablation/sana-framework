#!/usr/bin/env python3
"""Summarise the model-tier grid: accuracy and cost per (model, search arm)."""
import csv, glob, os, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "tmp/results-tiers"
PRICING = {"gpt-5.4-nano": (0.20, 0.02, 1.25),
           "gpt-5-mini":   (0.25, 0.025, 2.00),
           "gpt-5.2":      (1.75, 0.175, 14.00)}
ARMS = ["ideal", "standard", "naive"]

def price(model):
    for key, val in PRICING.items():
        if key in model:
            return val
    return None

cells = defaultdict(lambda: defaultdict(float))
for path in glob.glob(str(RESULTS / "**/eval_results.csv"), recursive=True):
    variant = os.path.basename(os.path.dirname(path))
    model = os.path.basename(os.path.dirname(os.path.dirname(path)))
    arm = variant.split("__")[0].replace("search_", "")
    c = cells[(model, arm)]
    for r in csv.DictReader(open(path)):
        num = lambda k: float(r.get(k) or 0)
        if num("input_tokens") <= 0:
            continue
        c["n"] += 1
        for k in ("input_tokens", "cached_input_tokens", "output_tokens",
                  "exact_match", "f1_score", "runtime_seconds", "cycle_count"):
            c[k] += num(k)
        if (r.get("error") or "").strip():
            c["err"] += 1

if not cells:
    print(f"no results under {RESULTS}")
    raise SystemExit(1)

print(f"{'model':<20}{'arm':<10}{'n':>3}{'EM':>7}{'F1':>7}{'cyc':>6}{'hit%':>7}{'$':>8}{'s/task':>8}")
print("-" * 76)
total = 0.0
for model in sorted({m for m, _ in cells}):
    for arm in ARMS:
        c = cells.get((model, arm))
        if not c or not c["n"]:
            continue
        n, i, ca, o = c["n"], c["input_tokens"], c["cached_input_tokens"], c["output_tokens"]
        p = price(model)
        cost = (p[0]*(i-ca) + p[1]*ca + p[2]*o) / 1e6 if p else 0.0
        total += cost
        print(f"{model:<20}{arm:<10}{n:>3.0f}{100*c['exact_match']/n:>6.0f}%"
              f"{100*c['f1_score']/n:>6.0f}%{c['cycle_count']/n:>6.1f}"
              f"{100*ca/i:>6.1f}%{cost:>8.2f}{c['runtime_seconds']/n:>8.0f}"
              + (f"   ERR={c['err']:.0f}" if c["err"] else ""))
print("-" * 76)
print(f"{'TOTAL':<64}{total:>8.2f}")
