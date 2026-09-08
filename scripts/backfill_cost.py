#!/usr/bin/env python3
"""Recompute cost columns for rows written before their model had a price.

A model with no MODEL_PRICING entry logs a warning and records 0.0, so rows
collected before the entry landed carry a zero indistinguishable from a free
run. Token counts are recorded either way, and the pricing formula is the one
in helper/result.py and instrumentation/ideal_subagent_costs.py, so the cost is
recoverable exactly.

Three columns are involved, not one. The hidden ideal-mode helper agents bill
separately into ideal_subagent_cost_usd, and the two totals are defined in
run_eval.py as:

    total_cost_with_ideal_subagents_usd = cost_usd + ideal_subagent_cost_usd
    total_cost_with_all_subagents_usd   = that + delegation_subagent_cost_usd

The totals are always recomputed from their parts, since that is their
definition; the two component columns are only filled when they are currently
zero, so a row that already has a real cost is never rewritten.

Not reconstructable: the per-tool splits (search/query/execute_ideal_subagent_
cost_usd) have call counts but no per-tool token columns, so a row whose
subagent cost was unpriced keeps zeros there. They feed no table; the aggregate
and the totals are what the results use.

    python scripts/backfill_cost.py --results-root experiments/<sweep> \\
        [--apply] [--model openai_gpt-5.6-luna]
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sana_evaluation.models import MODEL_PRICING  # noqa: E402

# Every tree a sweep may have produced, raw and audited. The audited trees are
# separate copies of the rows, so a repair applied only to the raw side never
# reaches the tables, which read the audited one whenever it is complete.
TREES = ["results", "results-rep2", "results-rep3",
         "results_semantic", "results-rep2_semantic", "results-rep3_semantic"]
EPS = 1e-9


def _num(v, d=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


def price_for(model_dir: str):
    """`openai_gpt-5.6-luna` -> the MODEL_PRICING entry, or None."""
    bare = model_dir.split("_", 1)[1] if "_" in model_dir else model_dir
    return MODEL_PRICING.get(bare) or MODEL_PRICING.get(model_dir)


def cost_from(row: dict, pricing: dict, prefix: str = "") -> float:
    """Price one token bundle. prefix '' is the main agent, else a subagent."""
    total_in = _num(row.get(f"{prefix}input_tokens"))
    cached = _num(row.get(f"{prefix}cached_input_tokens"))
    uncached = _num(row.get(f"{prefix}uncached_input_tokens"), max(0.0, total_in - cached))
    # Rows written before cache_write_input_tokens existed still carry it
    # implicitly: uncached already excludes writes, so the remainder is the
    # write count. Falling back to zero would price those tokens twice -- once
    # in the remainder and never at the write rate.
    writes = _num(row.get(f"{prefix}cache_write_input_tokens"),
                  max(0.0, total_in - cached - uncached))
    cached_rate = pricing.get("cache_read_input", pricing["input"])
    write_rate = pricing.get("cache_write_input", pricing["input"])
    return (pricing["input"] * uncached / 1_000_000
            + cached_rate * cached / 1_000_000
            + write_rate * writes / 1_000_000
            + pricing["output"] * _num(row.get(f"{prefix}output_tokens")) / 1_000_000)


def fix_row(row: dict, pricing: dict) -> bool:
    """Fill unpriced components and re-derive the totals. True if changed."""
    changed = False

    # Components: only fill a zero, and only when there were tokens to bill.
    for prefix, col in (("", "cost_usd"), ("ideal_subagent_", "ideal_subagent_cost_usd")):
        if col not in row:
            continue
        if _num(row.get(col)) > 0:
            continue
        if _num(row.get(f"{prefix}input_tokens")) <= 0:
            continue                      # never reached the model
        row[col] = f"{cost_from(row, pricing, prefix):.6f}"
        changed = True

    main = _num(row.get("cost_usd"))
    ideal = _num(row.get("ideal_subagent_cost_usd"))
    dele = _num(row.get("delegation_subagent_cost_usd"))
    for col, want in (("total_cost_with_ideal_subagents_usd", main + ideal),
                      ("total_cost_with_all_subagents_usd", main + ideal + dele)):
        if col in row and abs(_num(row.get(col)) - want) > EPS:
            row[col] = f"{want:.6f}"
            changed = True
    return changed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", type=Path, required=True,
                    help="experiment directory holding the results* trees")
    ap.add_argument("--apply", action="store_true", help="write changes (default: dry run)")
    ap.add_argument("--model", default="", help="restrict to one model directory")
    args = ap.parse_args()

    touched = files = 0
    for tree in TREES:
        modes = args.results_root / tree / "modes"
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
                n = sum(1 for row in rows
                        if not (row.get("error") or "").strip() and fix_row(row, pricing))
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
                      f"{path.relative_to(args.results_root)}")

    print(f"\n{'updated' if args.apply else 'would update'} {touched} rows in {files} files")
    if not args.apply and touched:
        print("re-run with --apply to write")
    return 0


if __name__ == "__main__":
    sys.exit(main())
