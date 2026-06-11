#!/usr/bin/env python3
"""
Efficiency analysis: cost/runtime/latency distributions, latency vs query length scatter.

Usage:
    python -m sana_analysis.running_analysis.efficiency [--results-dir results] [--sidecar-dir results/sidecar]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_agent_results(results_dir: str) -> list[dict]:
    records = []
    for jsonl_path in Path(results_dir).rglob("agent_results.jsonl"):
        parts = jsonl_path.parts
        condition = parts[-3] if len(parts) >= 3 else "unknown"
        model = parts[-2] if len(parts) >= 2 else "unknown"
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                r.setdefault("condition", condition)
                r.setdefault("model_label", model)
                records.append(r)
    return records


def load_sidecar_traces(sidecar_dir: str) -> list[dict]:
    traces = []
    for jsonl_path in Path(sidecar_dir).rglob("*.jsonl"):
        parts = jsonl_path.parts
        condition = parts[-3] if len(parts) >= 3 else "unknown"
        model = parts[-2] if len(parts) >= 2 else "unknown"
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    t = json.loads(line)
                    t.setdefault("condition", condition)
                    t.setdefault("model_label", model)
                    traces.append(t)
    return traces


def percentile(vals: list, p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    idx = int(len(s) * p / 100)
    return s[min(idx, len(s) - 1)]


def print_distribution(label: str, vals: list, unit: str = "") -> None:
    if not vals:
        print(f"  {label}: no data")
        return
    n = len(vals)
    mean = sum(vals) / n
    p50 = percentile(vals, 50)
    p90 = percentile(vals, 90)
    p99 = percentile(vals, 99)
    print(f"  {label}: mean={mean:.2f}{unit}  p50={p50:.2f}{unit}  p90={p90:.2f}{unit}  p99={p99:.2f}{unit}  n={n}")


def as_float(value) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def total_cost_with_ideal_subagents(record: dict) -> float:
    explicit = record.get("total_cost_with_ideal_subagents_usd")
    if explicit not in (None, ""):
        return as_float(explicit)
    return as_float(record.get("cost_usd")) + as_float(record.get("ideal_subagent_cost_usd"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--sidecar-dir", default="results/sidecar")
    args = parser.parse_args()

    agent_results = load_agent_results(args.results_dir)
    sidecar_traces = load_sidecar_traces(args.sidecar_dir)

    print(f"Loaded {len(agent_results)} task results, {len(sidecar_traces)} tool traces")

    # Group by condition
    by_condition: dict = defaultdict(list)
    for r in agent_results:
        by_condition[r.get("condition", "?")].append(r)

    print("\n=== Per-Task Efficiency by Condition ===")
    for cond, rows in sorted(by_condition.items()):
        print(f"\nCondition: {cond} (n={len(rows)})")
        costs = [as_float(r.get("cost_usd")) for r in rows]
        ideal_subagent_costs = [as_float(r.get("ideal_subagent_cost_usd")) for r in rows]
        combined_costs = [total_cost_with_ideal_subagents(r) for r in rows]
        times = [r.get("time", 0.0) for r in rows]
        tools = [r.get("tool_calls_total", 0) for r in rows]
        cycles = [r.get("cycle_count", 0) for r in rows if r.get("cycle_count")]
        print_distribution("Cost (USD, main agent)", costs, "$")
        print_distribution("Ideal subagent cost (USD)", ideal_subagent_costs, "$")
        print_distribution("Total cost incl. ideal subagents (USD)", combined_costs, "$")
        print_distribution("Runtime (s)", times, "s")
        print_distribution("Tool calls", tools)
        if cycles:
            print_distribution("Cycles", cycles)

    # Sidecar: latency distribution by tool
    by_tool: dict = defaultdict(list)
    for t in sidecar_traces:
        tool = t.get("tool", "?")
        lat = t.get("latency_ms", 0)
        by_tool[tool].append(lat)

    print("\n=== Tool Call Latency Distribution (ms) ===")
    for tool, lats in sorted(by_tool.items()):
        print_distribution(tool, lats, "ms")

    # Latency vs query length
    print("\n=== Latency vs Query Length (token approx) ===")
    query_lens = [t.get("query_length_tokens", 0) for t in sidecar_traces]
    latencies = [t.get("latency_ms", 0) for t in sidecar_traces]
    if query_lens and latencies:
        # Simple correlation
        n = len(query_lens)
        mean_l = sum(query_lens) / n
        mean_lat = sum(latencies) / n
        cov = sum((query_lens[i] - mean_l) * (latencies[i] - mean_lat) for i in range(n)) / n
        std_l = (sum((x - mean_l)**2 for x in query_lens) / n) ** 0.5
        std_lat = (sum((x - mean_lat)**2 for x in latencies) / n) ** 0.5
        corr = cov / (std_l * std_lat) if std_l * std_lat > 0 else 0.0
        print(f"  Pearson r(query_length, latency_ms) = {corr:.3f}  (n={n})")


if __name__ == "__main__":
    main()
