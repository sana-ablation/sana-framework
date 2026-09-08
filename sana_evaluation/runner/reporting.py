"""CSV/JSONL writers and the comparison-table printer for evaluation runs.

Consumed by `runner.orchestration`, which builds the result dicts these
functions serialize. No dependency the other direction: this module knows
nothing about `BatchRunner` or task discovery.
"""

import csv
import json
import logging
import os

logger = logging.getLogger(__name__)


def _json_list_count(raw) -> int:
    if isinstance(raw, list):
        return len(raw)
    text = str(raw or "").strip()
    if not text:
        return 0
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return 0
    return len(payload) if isinstance(payload, list) else 0


_IDEAL_SUBAGENT_CSV_FIELDS = [
    "ideal_subagent_calls",
    "ideal_subagent_input_tokens",
    "ideal_subagent_cached_input_tokens",
    "ideal_subagent_uncached_input_tokens",
    "ideal_subagent_output_tokens",
    "ideal_subagent_total_tokens",
    "ideal_subagent_cost_usd",
    "search_ideal_subagent_calls",
    "search_ideal_subagent_cost_usd",
    "query_ideal_subagent_calls",
    "query_ideal_subagent_cost_usd",
    "execute_ideal_subagent_calls",
    "execute_ideal_subagent_cost_usd",
    "total_cost_with_ideal_subagents_usd",
]


_DELEGATION_SUBAGENT_CSV_FIELDS = [
    "delegation_subagent_calls",
    "delegation_subagent_input_tokens",
    "delegation_subagent_cached_input_tokens",
    "delegation_subagent_uncached_input_tokens",
    "delegation_subagent_output_tokens",
    "delegation_subagent_total_tokens",
    "delegation_subagent_cost_usd",
    "search_subagent_calls",
    "search_subagent_cost_usd",
    "inspect_subagent_calls",
    "inspect_subagent_cost_usd",
    "total_cost_with_all_subagents_usd",
]


def _float_value(raw) -> float:
    try:
        return float(raw or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _combined_cost(row: dict) -> float:
    if row.get("total_cost_with_ideal_subagents_usd") not in (None, ""):
        return _float_value(row.get("total_cost_with_ideal_subagents_usd"))
    return _float_value(row.get("cost_usd")) + _float_value(row.get("ideal_subagent_cost_usd"))


def _combined_cost_with_delegation(row: dict) -> float:
    if row.get("total_cost_with_all_subagents_usd") not in (None, ""):
        return _float_value(row.get("total_cost_with_all_subagents_usd"))
    return (
        _float_value(row.get("cost_usd"))
        + _float_value(row.get("ideal_subagent_cost_usd"))
        + _float_value(row.get("delegation_subagent_cost_usd"))
    )


def _normalize_main_csv_row(row: dict) -> dict:
    normalized = {
        "task_id": row.get("task_id", ""),
        "model": row.get("model", ""),
        "expected_answer": row.get("expected_answer", ""),
        "predicted_answer": row.get("predicted_answer", ""),
        "exact_match": row.get("exact_match", ""),
        "f1_score": row.get("f1_score", ""),
        "required_dataset_count": row.get(
            "required_dataset_count",
            _json_list_count(row.get("required_datasets", "")),
        ),
        "sources_used_count": row.get(
            "sources_used_count",
            _json_list_count(row.get("sources_used", "")),
        ),
        "runtime_seconds": row.get("runtime_seconds", 0),
        "cycle_count": row.get("cycle_count", ""),
        "input_tokens": row.get("input_tokens", 0),
        "cached_input_tokens": row.get("cached_input_tokens", 0),
        "cache_write_input_tokens": row.get("cache_write_input_tokens", 0),
        "uncached_input_tokens": row.get(
            "uncached_input_tokens",
            max(0, int(row.get("input_tokens", 0) or 0) - int(row.get("cached_input_tokens", 0) or 0)),
        ),
        "output_tokens": row.get("output_tokens", 0),
        "total_tokens": row.get("total_tokens", 0),
        "cost_usd": row.get("cost_usd", 0.0),
        "tool_calls_total": row.get("tool_calls_total", 0),
        "api_tool_calls": row.get("api_tool_calls", 0),
        "execute_ideal_agent_repair_calls": row.get("execute_ideal_agent_repair_calls", 0),
        "query_ideal_agent_repair_calls": row.get("query_ideal_agent_repair_calls", 0),
        "success": row.get("success", False),
        "error": row.get("error", ""),
    }
    for field in _IDEAL_SUBAGENT_CSV_FIELDS:
        if field == "total_cost_with_ideal_subagents_usd":
            normalized[field] = _combined_cost(row)
        else:
            normalized[field] = row.get(field, 0)
    for field in _DELEGATION_SUBAGENT_CSV_FIELDS:
        if field == "total_cost_with_all_subagents_usd":
            normalized[field] = _combined_cost_with_delegation(row)
        else:
            normalized[field] = row.get(field, 0)
    return normalized


def write_main_csv(csv_path: str, results: list, tasks_by_id: dict) -> None:
    fieldnames = [
        "task_id", "model",
        "expected_answer", "predicted_answer", "exact_match", "f1_score",
        "required_dataset_count", "sources_used_count",
        "runtime_seconds", "cycle_count",
        "input_tokens", "cached_input_tokens", "cache_write_input_tokens",
        "uncached_input_tokens",
        "output_tokens", "total_tokens", "cost_usd",
        "tool_calls_total", "api_tool_calls",
        "execute_ideal_agent_repair_calls", "query_ideal_agent_repair_calls",
        *_IDEAL_SUBAGENT_CSV_FIELDS,
        *_DELEGATION_SUBAGENT_CSV_FIELDS,
        "success", "error",
    ]
    existing_rows: dict = {}
    if os.path.exists(csv_path):
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("task_id"):
                    existing_rows[row["task_id"]] = _normalize_main_csv_row(row)

    for r in results:
        task_id = r.get("task_id", "")
        task = tasks_by_id.get(task_id, {})
        required = task.get("datasets_used", [])
        sources_used = r.get("sources_used", []) or []
        row = {
            "task_id": task_id,
            "model": r.get("model", ""),
            "expected_answer": task.get("answer", ""),
            "predicted_answer": r.get("predicted_answer", ""),
            "exact_match": r.get("exact_match", ""),
            "f1_score": r.get("f1_score", ""),
            "required_dataset_count": len({str(item) for item in required}),
            "sources_used_count": len({str(item) for item in sources_used}),
            "runtime_seconds": r.get("time", 0),
            "cycle_count": r.get("cycle_count", ""),
            "input_tokens": r.get("input_tokens", 0),
            "cached_input_tokens": r.get("cached_input_tokens", 0),
            "uncached_input_tokens": r.get(
                "uncached_input_tokens",
                max(0, int(r.get("input_tokens", 0) or 0) - int(r.get("cached_input_tokens", 0) or 0)),
            ),
            "output_tokens": r.get("output_tokens", 0),
            "total_tokens": r.get("total_tokens", 0),
            "cost_usd": r.get("cost_usd", 0.0),
            "tool_calls_total": r.get("tool_calls_total", 0),
            "api_tool_calls": r.get("api_tool_calls", 0),
            "execute_ideal_agent_repair_calls": r.get("execute_ideal_agent_repair_calls", 0),
            "query_ideal_agent_repair_calls": r.get("query_ideal_agent_repair_calls", 0),
            "success": r.get("success", False),
            "error": r.get("error", ""),
        }
        for field in _IDEAL_SUBAGENT_CSV_FIELDS:
            if field == "total_cost_with_ideal_subagents_usd":
                row[field] = _combined_cost(r)
            else:
                row[field] = r.get(field, 0)
        for field in _DELEGATION_SUBAGENT_CSV_FIELDS:
            if field == "total_cost_with_all_subagents_usd":
                row[field] = _combined_cost_with_delegation(r)
            else:
                row[field] = r.get(field, 0)
        existing_rows[str(task_id)] = row

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for task_id in sorted(existing_rows.keys()):
            writer.writerow(existing_rows[task_id])


def write_tools_csv(tools_csv_path: str, results: list) -> None:
    fieldnames = ["task_id", "tool_name", "call_count", "success_count", "avg_time_seconds"]
    existing_rows: dict = {}
    if os.path.exists(tools_csv_path):
        with open(tools_csv_path, newline="") as f:
            for row in csv.DictReader(f):
                key = (row.get("task_id", ""), row.get("tool_name", ""))
                existing_rows[key] = row

    for r in results:
        task_id = str(r.get("task_id", ""))
        for tool in r.get("tool_counts", []):
            key = (task_id, tool["name"])
            existing_rows[key] = {
                "task_id": task_id,
                "tool_name": tool["name"],
                "call_count": tool["call_count"],
                "success_count": tool["success_count"],
                "avg_time_seconds": tool["average_time"],
            }

    with open(tools_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for key in sorted(existing_rows.keys()):
            writer.writerow(existing_rows[key])


def write_agent_results_jsonl(jsonl_path: str, results: list) -> None:
    """Append agent results to a JSONL file (one object per task)."""
    with open(jsonl_path, "a") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def print_comparison_table(results: dict) -> None:
    logger.info("\n" + "=" * 100)
    logger.info("MODEL COMPARISON")
    logger.info("=" * 100)
    logger.info(f"{'Model':<40} {'EM Rate':<10} {'Avg F1':<10} {'Avg Time':<12} {'Avg Cost':<12} {'Avg Tools':<10}")
    logger.info("-" * 100)
    for model_id, data in results.items():
        if "error" in data:
            logger.info(f"{model_id:<40} ERROR: {data['error'][:55]}")
        else:
            s = data["summary"]
            logger.info(
                f"{model_id:<40} {s['exact_match_rate']*100:>5.1f}%    "
                f"{s['avg_f1_score']:>6.3f}    {s['avg_time']:>8.1f}s    "
                f"${s['avg_cost_usd']:>7.4f}    {s['avg_tool_calls']:>6.1f}"
            )
    logger.info("=" * 100)
