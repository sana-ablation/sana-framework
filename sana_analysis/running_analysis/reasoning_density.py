#!/usr/bin/env python3
"""
EM stratified by reasoning density (gold document count per task).

Reproduces Figure 4 from the LakeQA paper: average EM binned by the number
of gold documents required per task (len(datasets_used)), per condition.

Bins: <=2, 3-4, 5-7, 8-10, >10

Usage:
    python -m sana_analysis.running_analysis.reasoning_density [--results-dir results] [--tasks-dir tasks_mini]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from sana_analysis.running_analysis.discovery_metrics import make_task_stem_key, resolve_task_value

_BINS = [
    ("<=2",  lambda n: n <= 2),
    ("3-4",  lambda n: 3 <= n <= 4),
    ("5-7",  lambda n: 5 <= n <= 7),
    ("8-10", lambda n: 8 <= n <= 10),
    (">10",  lambda n: n > 10),
]


def _assign_bin(n_docs: int) -> str:
    for label, pred in _BINS:
        if pred(n_docs):
            return label
    return ">10"


def load_task_gold_counts(tasks_dir: str) -> dict[str, int]:
    """Return {task_stem_key: len(datasets_used)} for all task JSON files."""
    import glob as glob_mod
    counts: dict = {}
    for path in glob_mod.glob(str(Path(tasks_dir) / "**" / "*.json"), recursive=True):
        with open(path) as f:
            task = json.load(f)
        value = len(task.get("datasets_used", []))
        counts[path] = value
        counts[make_task_stem_key(path)] = value
    return counts


def compute_reasoning_density_curve(
    records: list[dict],
    task_gold_counts: dict[str, int],
) -> dict:
    """Bin tasks by gold doc count, compute mean EM per bin per condition.

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

        task_id = str(r.get("task_id", ""))
        n_docs = resolve_task_value(task_id, task_gold_counts)
        if n_docs is None:
            continue

        bin_label = _assign_bin(n_docs)
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
    parser.add_argument("--tasks-dir", default="benchmarks/lakeqa/tasks-mini/tasks")
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

    task_gold_counts = load_task_gold_counts(args.tasks_dir)
    if not task_gold_counts:
        print(f"No task JSON files found under {args.tasks_dir}")
        return

    curve = compute_reasoning_density_curve(records, task_gold_counts)
    if not curve:
        print("No data with exact_match + resolvable task_ids.")
        return

    bin_labels = [label for label, _ in _BINS]
    for cond, bins in sorted(curve.items()):
        print(f"\nCondition {cond} — EM by gold-doc count (reasoning density):")
        print(f"  {'Bin':<6} {'EM%':>7} {'n':>5}")
        print(f"  {'-'*20}")
        for label in bin_labels:
            entry = bins.get(label, {})
            em = entry.get("mean_em")
            n = entry.get("n", 0)
            em_str = f"{em*100:.1f}%" if em is not None else "  N/A"
            print(f"  {label:<6} {em_str:>7} {n:>5}")


if __name__ == "__main__":
    main()
