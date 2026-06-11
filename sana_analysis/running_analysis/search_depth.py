#!/usr/bin/env python3
"""
Search depth curve: mean EM per search-call bin, per condition.

Bins tasks by total search calls used (derived from tool_counts in agent_results.jsonl)
and computes mean EM per bin to reveal the saturation point.

Usage:
    python -m sana_analysis.running_analysis.search_depth [--results-dir results]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

_BINS = [
    ("1", lambda n: n == 1),
    ("2-3", lambda n: 2 <= n <= 3),
    ("4-6", lambda n: 4 <= n <= 6),
    ("7-10", lambda n: 7 <= n <= 10),
    ("11-30", lambda n: 11 <= n <= 30),
]


def _assign_bin(search_calls: int) -> str:
    for label, pred in _BINS:
        if pred(search_calls):
            return label
    return "11-30"


def _count_search_calls(record: dict) -> int:
    total = 0
    for tool in record.get("tool_counts", []) or []:
        name = str(tool.get("tool", ""))
        if name.startswith("search"):
            try:
                total += int(tool.get("call_count", 0) or 0)
            except (TypeError, ValueError):
                continue
    return total


def compute_search_depth_curve(records: list[dict]) -> dict:
    """Bin tasks by search call count, compute mean EM per bin per condition.

    Returns:
        {condition: {bin_label: {"mean_em": float, "n": int}}}
    """
    # condition -> bin_label -> list of EM values
    by_cond: dict = defaultdict(lambda: defaultdict(list))

    for r in records:
        em = r.get("exact_match")
        if em is None:
            continue
        cond = r.get("condition", "?")
        search_calls = _count_search_calls(r)
        bin_label = _assign_bin(search_calls)
        by_cond[cond][bin_label].append(float(em))

    out = {}
    for cond, bins in sorted(by_cond.items()):
        out[cond] = {}
        for label, _ in _BINS:
            vals = bins.get(label, [])
            out[cond][label] = {
                "mean_em": round(sum(vals) / len(vals), 4) if vals else None,
                "n": len(vals),
            }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    records = []
    for path in Path(args.results_dir).glob("*/*/agent_results.jsonl"):
        condition = path.parent.parent.name
        model = path.parent.name
        with path.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                record.setdefault("condition", condition)
                record.setdefault("model_label", model)
                records.append(record)
    if not records:
        print(f"No agent_results.jsonl files found under {args.results_dir}")
        return

    curve = compute_search_depth_curve(records)
    if not curve:
        print("No data with exact_match + tool_counts fields.")
        return

    bin_labels = [label for label, _ in _BINS]
    for cond, bins in sorted(curve.items()):
        print(f"\nCondition {cond} — EM by search-call bin:")
        print(f"  {'Bin':<8} {'EM%':>7} {'n':>5}")
        print(f"  {'-'*22}")
        for label in bin_labels:
            entry = bins.get(label, {})
            em = entry.get("mean_em")
            n = entry.get("n", 0)
            em_str = f"{em*100:.1f}%" if em is not None else "  N/A"
            print(f"  {label:<8} {em_str:>7} {n:>5}")


if __name__ == "__main__":
    main()
