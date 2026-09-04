#!/usr/bin/env python3
"""Recompute cost_usd for rows written before their model had a price.

A model with no MODEL_PRICING entry logs a warning and records 0.0, so rows
collected before the entry landed carry a zero that is indistinguishable from a
genuinely free run. Token counts are recorded either way, and the pricing
formula is the one in helper/result.py, so the cost is recoverable exactly.

Only rows whose cost is currently zero are touched, and only when the model has
a price; a row that already has a cost is never rewritten.

    python backfill_cost.py [--apply] [--model openai_gpt-5.6-luna]
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

from sana_evaluation.helper.constants import MODEL_PRICING  # noqa: E402

TREES = ["results", "results-rep2", "results-rep3",
         "results_semantic", "results-rep2_semantic", "results-rep3_semantic"]


def _num(v, d=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


def price_for(model_dir: str):
    """`openai_gpt-5.6-luna` -> the MODEL_PRICING entry, or None."""
    bare = model_dir.split("_", 1)[1] if "_" in model_dir else model_dir
    return MODEL_PRICING.get(bare) or MODEL_PRICING.get(model_dir)


def row_cost(row: dict, pricing: dict) -> float:
    total_in = _num(row.get("input_tokens"))
    cached = _num(row.get("cached_input_tokens"))
    uncached = _num(row.get("uncached_input_tokens"), max(0.0, total_in - cached))
    cached_rate = pricing.get("cache_read_input", pricing["input"])
    return (pricing["input"] * uncached / 1_000_000
            + cached_rate * cached / 1_000_000
            + pricing["output"] * _num(row.get("output_tokens")) / 1_000_000)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="write changes (default: dry run)")
    ap.add_argument("--model", default="", help="restrict to one model directory")
    args = ap.parse_args()

    touched = files = 0
    for tree in TREES:
        modes = HERE / tree / "modes"
        if not modes.is_dir():
            continue
        for model_dir in sorted(p for p in modes.iterdir() if p.is_dir()):
            if args.model and model_dir.name != args.model:
                continue
            pricing = price_for(model_dir.name)
            if not pricing:
                print(f"  no pricing for {model_dir.name} -- skipped")
                continue
            for path in sorted(model_dir.rglob("eval_results.csv")):
                with open(path, newline="") as f:
                    reader = csv.DictReader(f)
                    fields, rows = reader.fieldnames, list(reader)
                n = 0
                for row in rows:
                    if _num(row.get("cost_usd")) > 0:
                        continue
                    if _num(row.get("input_tokens")) <= 0:
                        continue          # an errored row never called the model
                    row["cost_usd"] = f"{row_cost(row, pricing):.6f}"
                    n += 1
                if not n:
                    continue
                touched += n
                files += 1
                if args.apply:
                    with open(path, "w", newline="") as f:
                        w = csv.DictWriter(f, fieldnames=fields)
                        w.writeheader()
                        w.writerows(rows)
                print(f"  {'wrote' if args.apply else 'would write'} {n:>3} rows  "
                      f"{path.relative_to(HERE)}")

    print(f"\n{'updated' if args.apply else 'would update'} {touched} rows in {files} files")
    if not args.apply and touched:
        print("re-run with --apply to write")
    return 0


if __name__ == "__main__":
    sys.exit(main())
