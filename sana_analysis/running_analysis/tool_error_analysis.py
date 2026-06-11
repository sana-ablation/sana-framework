#!/usr/bin/env python3
"""
Per-tool error rate analysis.

Reads tool_counts from agent_results.jsonl (which already tracks call_count and
success_count per tool per task) and computes error rates per condition × model.

Output: analysis_results/tool_errors.json

Usage:
    python -m sana_analysis.running_analysis.tool_error_analysis [--results-dir results] [--tasks-dir tasks_mini]
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# Tools we care about for error rate reporting (data access tools only).
# Keep both "peek_files" (historical, pre-rename) and "peek_multiple" (current)
# so analysis still aggregates older eval runs correctly.
_DATA_TOOLS = {
    "query_file",
    "execute_code",
    "peek_file",
    "peek_files",
    "peek_multiple",
    "grep_file",
    "read_file",
    "download",
    "list_files",
}


def compute_tool_error_rates(records: list[dict]) -> dict:
    """Compute per-tool error rates aggregated by condition/model.

    Returns:
        {
          "condition/model": {
            "tool_name": {
              "total_calls": int,
              "total_errors": int,
              "error_rate": float,
            },
            ...
          },
          ...
        }
    """
    # Accumulate: (condition, model, tool) -> (total_calls, total_errors)
    acc: dict = defaultdict(lambda: defaultdict(lambda: [0, 0]))

    for r in records:
        cond = r.get("condition", "unknown")
        model = r.get("model", "unknown")
        key = f"{cond}/{model}"
        for tc in r.get("tool_counts", []):
            name = tc.get("name", "")
            calls = tc.get("call_count", 0)
            successes = tc.get("success_count", 0)
            if calls == 0:
                continue
            errors = max(calls - successes, 0)
            acc[key][name][0] += calls
            acc[key][name][1] += errors

    out = {}
    for key, tools in sorted(acc.items()):
        out[key] = {}
        for tool_name, (total_calls, total_errors) in sorted(tools.items()):
            out[key][tool_name] = {
                "total_calls": total_calls,
                "total_errors": total_errors,
                "error_rate": round(total_errors / total_calls, 4) if total_calls else 0.0,
            }
    return out


def compute_tool_error_rates_by_task_difficulty(
    records: list[dict], task_gold_counts: dict
) -> dict:
    """Same as compute_tool_error_rates but additionally broken down by reasoning density bin.

    Args:
        records: loaded agent_results records (with condition/model fields)
        task_gold_counts: {task_stem_key: int} from load_task_gold_counts()

    Returns:
        {
          "condition/model": {
            "bin_label": {
              "tool_name": {"total_calls", "total_errors", "error_rate"}
            }
          }
        }
    """
    bins = ["<=2", "3-4", "5-7", "8-10", ">10"]

    def _bin(n: int) -> str:
        if n <= 2:
            return "<=2"
        if n <= 4:
            return "3-4"
        if n <= 7:
            return "5-7"
        if n <= 10:
            return "8-10"
        return ">10"

    # acc[(key, bin_label, tool)] -> [total_calls, total_errors]
    acc: dict = defaultdict(lambda: [0, 0])

    for r in records:
        cond = r.get("condition", "unknown")
        model = r.get("model", "unknown")
        key = f"{cond}/{model}"
        task_id = str(r.get("task_id", ""))
        p = Path(task_id)
        stem_key = f"{p.parent.name}/{p.stem}"
        gold_count = task_gold_counts.get(stem_key, 0)
        b = _bin(gold_count)
        for tc in r.get("tool_counts", []):
            name = tc.get("name", "")
            calls = tc.get("call_count", 0)
            successes = tc.get("success_count", 0)
            if calls == 0:
                continue
            errors = max(calls - successes, 0)
            acc[(key, b, name)][0] += calls
            acc[(key, b, name)][1] += errors

    # Restructure
    out: dict = defaultdict(lambda: defaultdict(dict))
    for (key, b, tool_name), (total_calls, total_errors) in acc.items():
        out[key][b][tool_name] = {
            "total_calls": total_calls,
            "total_errors": total_errors,
            "error_rate": round(total_errors / total_calls, 4) if total_calls else 0.0,
        }

    # Ensure all bins present for each key
    result = {}
    for key in out:
        result[key] = {}
        for b in bins:
            result[key][b] = dict(sorted(out[key].get(b, {}).items()))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--tasks-dir", default="benchmarks/lakeqa/tasks-mini/tasks")
    parser.add_argument("--output-dir", default="analysis_results")
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
        print("No records found. Skipping tool error analysis.")
        return

    error_rates = compute_tool_error_rates(records)

    try:
        from sana_analysis.running_analysis.reasoning_density import load_task_gold_counts
        task_gold_counts = load_task_gold_counts(args.tasks_dir)
        by_difficulty = compute_tool_error_rates_by_task_difficulty(records, task_gold_counts)
    except Exception:
        by_difficulty = {}

    out = {
        "by_condition_model": error_rates,
        "by_difficulty": by_difficulty,
    }

    out_path = Path(args.output_dir) / "tool_errors.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {out_path}")

    # Print summary table
    print(f"\nTool Error Rates (data access tools only):")
    print(f"  {'Condition/Model':<45} {'Tool':<20} {'Calls':>6} {'Errors':>6} {'Err%':>6}")
    print(f"  {'-'*90}")
    for key, tools in sorted(error_rates.items()):
        for tool_name, stats in sorted(tools.items()):
            if tool_name not in _DATA_TOOLS:
                continue
            rate = stats["error_rate"] * 100
            flag = " <--" if rate >= 15 else ""
            print(f"  {key:<45} {tool_name:<20} {stats['total_calls']:>6} {stats['total_errors']:>6} {rate:>5.1f}%{flag}")


if __name__ == "__main__":
    main()
