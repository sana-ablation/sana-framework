#!/usr/bin/env python3
"""Summarise the model-tier x per-axis ablation grid.

Prints, per model, each cell's accuracy and the DROP from the all-oracle
reference. The drop is the ablation signal: it localises the bottleneck.
"""
import csv, glob, os, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "tmp/results-tiers"

# variant dir -> readable cell name; order is the report order
CELLS = [
    ("search_ideal__results_ideal__profile_ideal__compute_ideal", "REFERENCE"),
    ("search_ideal__results_ideal__profile_ideal__compute_standard", "compute=standard"),
    ("search_standard__results_ideal__profile_ideal__compute_ideal", "search=standard"),
    ("search_naive__results_ideal__profile_ideal__compute_ideal", "search=naive"),
    ("search_preloaded__results_ideal__profile_ideal__compute_ideal", "search=preloaded"),
    ("search_ideal__results_ideal__profile_standard__compute_ideal", "profile=standard"),
    ("search_ideal__results_ideal__profile_naive__compute_ideal", "profile=naive"),
]
PRICING = {"gpt-5.4-nano": (0.20, 0.02, 1.25),
           "gpt-5-mini": (0.25, 0.025, 2.00),
           "gpt-5.2": (1.75, 0.175, 14.00)}


def price(model):
    for key, val in PRICING.items():
        if key in model:
            return val
    return None


agg = defaultdict(lambda: defaultdict(float))
for path in glob.glob(str(RESULTS / "**/eval_results.csv"), recursive=True):
    variant = os.path.basename(os.path.dirname(path))
    model = os.path.basename(os.path.dirname(os.path.dirname(path)))
    cell = next((name for prefix, name in CELLS if variant.startswith(prefix)), None)
    if cell is None:
        continue
    a = agg[(model, cell)]
    for r in csv.DictReader(open(path)):
        num = lambda k: float(r.get(k) or 0)
        if num("input_tokens") <= 0:
            continue
        a["n"] += 1
        for k in ("input_tokens", "cached_input_tokens", "output_tokens",
                  "exact_match", "f1_score", "runtime_seconds"):
            a[k] += num(k)
        a["sub"] += num("total_cost_with_all_subagents_usd")
        sm = str(r.get("semantic_match", "")).strip().lower()
        if sm:
            a["has_sem"] = 1
            a["semantic"] += 1.0 if sm in {"1", "1.0", "true", "yes", "t"} else 0.0
        if (r.get("error") or "").strip():
            a["err"] += 1

if not agg:
    print(f"no results under {RESULTS}")
    raise SystemExit(1)

grand = 0.0
for model in sorted({m for m, _ in agg}):
    ref = agg.get((model, "REFERENCE"))
    ref_em = 100 * ref["exact_match"] / ref["n"] if ref and ref["n"] else None
    print(f"\n{model}")
    print(f"  {'cell':<20}{'n':>3}{'EM':>7}{'drop':>8}{'F1':>7}{'hit%':>7}{'$':>8}{'s/task':>8}")
    print("  " + "-" * 68)
    for _, cell in CELLS:
        a = agg.get((model, cell))
        if not a or not a["n"]:
            continue
        n = a["n"]
        em = 100 * (a["semantic"] if a["has_sem"] else a["exact_match"]) / n
        drop = "" if cell == "REFERENCE" or ref_em is None else f"{em - ref_em:+.0f}pp"
        p = price(model)
        cost = a["sub"] or ((p[0]*(a["input_tokens"]-a["cached_input_tokens"])
                             + p[1]*a["cached_input_tokens"] + p[2]*a["output_tokens"]) / 1e6 if p else 0)
        grand += cost
        print(f"  {cell:<20}{n:>3.0f}{em:>6.0f}%{drop:>8}"
              f"{100*a['f1_score']/n:>6.0f}%"
              f"{100*a['cached_input_tokens']/a['input_tokens']:>6.1f}%"
              f"{cost:>8.2f}{a['runtime_seconds']/n:>8.0f}"
              + (f"  ERR={a['err']:.0f}" if a["err"] else ""))
any_sem = any(a["has_sem"] for a in agg.values())
print(f"\nGRAND TOTAL  ${grand:.2f}")
print("metric: semantic_match" if any_sem else
      "metric: exact_match  (run the semantic-eval-auditor for paper-comparable numbers)")
