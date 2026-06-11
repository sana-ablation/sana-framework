"""Orchestration library for evaluation runs.

Not a CLI — invoke evaluations via `sana_evaluation.run_mode_eval` or
`sana_evaluation.run_sana_eval`. This module exposes shared helpers those
entrypoints use:
  - `find_all_task_dirs`, `run_evaluation`, `_run_task_files` — task discovery
    and per-directory orchestration
  - `_display_name`, `_results_dir`, `_maybe_autoset_openai_cache_key` —
    naming and output-path conventions
  - `print_comparison_table` — summary printer
  - CSV/JSONL writers used by the above

Caller responsibility: monkey-patch `BatchRunner` (set
`run_eval.BatchRunner = <YourRunnerClass>`) before invoking helpers that
construct one. Both run_mode_eval and run_sana_eval do this at module load.
"""

import csv
import glob
import json
import logging
import os
from datetime import datetime
from typing import Optional

from sana_evaluation.config import AgentConfig, ConditionConfig, RunConfig
from sana_evaluation.helper.prompting import normalize_debug_mode

logger = logging.getLogger(__name__)

# Monkey-patched by callers before _run_task_files / run_evaluation are
# invoked. Default None — using helpers without patching raises.
BatchRunner = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sanitize_model_name(model_id: str) -> str:
    return model_id.replace("/", "_").replace(":", "_")


def _display_name(agent_config) -> str:
    """Return a clean, human-readable slug for use in paths.

    Prefers model_name (e.g. 'bedrock/claude-haiku-4.5-arn') over raw model_id
    so ARN-based models don't produce garbage directory names. Appends
    reasoning_effort when set (e.g. 'openai_gpt-5.2-xhigh').
    """
    name = agent_config.model_name or agent_config.model_id
    slug = _sanitize_model_name(name)
    effort = (agent_config.extra_model_kwargs or {}).get("reasoning_effort")
    if effort:
        slug = f"{slug}-{effort}"
    return slug


def _maybe_autoset_openai_cache_key(agent_config, variant_label: str) -> None:
    """Auto-derive a variant-stable OpenAI prompt_cache_key when none was provided.

    The key hints to OpenAI's router which shard likely has the matching prefix cached.
    A stable `{model}:{variant}` key keeps requests in the same sweep on the same shard,
    which raises hit rate vs. the default user-hash routing. Explicit CLI/env values win.
    """
    if agent_config.openai_prompt_cache_key is not None:
        return
    if os.getenv("OPENAI_PROMPT_CACHE_KEY"):
        return
    agent_config.openai_prompt_cache_key = f"{_display_name(agent_config)}:{variant_label}"


def _results_root(run_config: RunConfig) -> str:
    return getattr(run_config, "results_output_dir", "results") or "results"


def _uses_mode_layout(condition_label: str) -> bool:
    normalized = str(condition_label or "").replace("\\", "/").lstrip("./")
    return normalized == "modes" or normalized.startswith("modes/")


def _results_dir(run_config: RunConfig, agent_config: AgentConfig) -> str:
    condition_label = run_config.condition_config.condition
    safe_model = _display_name(agent_config)
    if _uses_mode_layout(condition_label):
        return os.path.join(_results_root(run_config), condition_label)
    return os.path.join(_results_root(run_config), condition_label, safe_model)


def _with_debug_suffix(label: str, debug_mode: Optional[str]) -> str:
    normalized = normalize_debug_mode(debug_mode)
    if normalized is None:
        return label
    return f"{label}__debug_{normalized}"


def find_all_task_dirs(base_dir: str = "tasks") -> list:
    """Return all task directories matching the k-*-d-* pattern."""
    return sorted(glob.glob(os.path.join(base_dir, "k-*-d-*")))


# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(
    task_dir: str,
    agent_config: AgentConfig,
    run_config: RunConfig,
    verbose: bool = False,
    only_new: bool = False,
    parallel: int = 6,
    tasks_per_dir: Optional[int] = None,
) -> dict:
    """Run evaluation on a task directory and return {model_id -> {summary, results}}."""
    cond = run_config.condition_config
    condition_label = cond.condition
    safe_model = _display_name(agent_config)
    output_dir = _results_dir(run_config, agent_config)
    os.makedirs(output_dir, exist_ok=True)

    task_files = sorted(glob.glob(os.path.join(task_dir, "*.json")))
    if not task_files:
        logger.info(f"No task files found in {task_dir}")
        return {}

    if tasks_per_dir is not None:
        task_files = task_files[:tasks_per_dir]

    task_dir_name = os.path.basename(task_dir)
    model_id = agent_config.model_id

    logger.info(f"\nEvaluating {len(task_files)} tasks from {task_dir_name}")
    logger.info(f"Model: {model_id}  Condition: {condition_label}")
    logger.info("=" * 60)

    # Load task metadata for CSV annotation
    tasks_by_id: dict = {}
    for path in task_files:
        with open(path) as f:
            task = json.load(f)
            task["id"] = path
            tasks_by_id[path] = task

    csv_path = os.path.join(output_dir, "eval_results.csv")

    # --only-new: skip tasks already present in the CSV
    task_files_to_run = task_files
    if only_new and os.path.exists(csv_path):
        existing_ids: set = set()
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("task_id"):
                    existing_ids.add(row["task_id"])
        task_files_to_run = [p for p in task_files if p not in existing_ids]
        if not task_files_to_run:
            logger.info("  No new tasks to evaluate.")
            return {model_id: {"summary": _empty_summary(model_id, task_dir_name), "results": []}}

    try:
        batch = BatchRunner(agent_config=agent_config, run_config=run_config, max_workers=parallel)
        results = batch.run_from_files(task_files_to_run, verbose=verbose)
    except Exception as e:
        logger.error(f"  Error: {e}", exc_info=True)
        return {model_id: {"error": str(e)}}

    # Summary stats
    total = len(results)
    exact_matches = sum(r.get("exact_match", 0) for r in results)
    f1_scores = [r["f1_score"] for r in results if "f1_score" in r]
    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    avg_time = sum(r.get("time", 0) for r in results) / total if total else 0.0

    total_cost = sum(r.get("cost_usd", 0.0) for r in results)
    total_tool_calls = sum(r.get("tool_calls_total", 0) for r in results)

    summary = {
        "model": model_id,
        "task_dir": task_dir_name,
        "total_tasks": total,
        "exact_match_count": exact_matches,
        "exact_match_rate": exact_matches / total if total else 0.0,
        "avg_f1_score": avg_f1,
        "avg_time": avg_time,
        "total_cost_usd": total_cost,
        "avg_cost_usd": total_cost / total if total else 0.0,
        "avg_tool_calls": total_tool_calls / total if total else 0.0,
    }

    logger.info(f"  Exact Match: {exact_matches}/{total} ({100 * exact_matches / total:.1f}%)" if total else "  No results")
    logger.info(f"  Avg F1: {avg_f1:.3f}")

    # Write main CSV (upsert pattern — preserve existing rows)
    _write_main_csv(csv_path, results, tasks_by_id)

    # Write per-tool breakdown CSV
    tools_csv_path = os.path.join(output_dir, "tools_breakdown.csv")
    _write_tools_csv(tools_csv_path, results)

    # Write agent_results JSONL (one line per task)
    jsonl_path = os.path.join(output_dir, "agent_results.jsonl")
    _write_agent_results_jsonl(jsonl_path, results)

    return {model_id: {"summary": summary, "results": results}}


def _empty_summary(model_id: str, task_dir_name: str) -> dict:
    return {
        "model": model_id,
        "task_dir": task_dir_name,
        "total_tasks": 0,
        "exact_match_count": 0,
        "exact_match_rate": 0.0,
        "avg_f1_score": 0.0,
        "avg_time": 0.0,
        "total_cost_usd": 0.0,
        "avg_cost_usd": 0.0,
        "avg_tool_calls": 0.0,
    }


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


def _write_main_csv(csv_path: str, results: list, tasks_by_id: dict) -> None:
    fieldnames = [
        "task_id", "model",
        "expected_answer", "predicted_answer", "exact_match", "f1_score",
        "required_dataset_count", "sources_used_count",
        "runtime_seconds", "cycle_count",
        "input_tokens", "cached_input_tokens", "uncached_input_tokens",
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


def _write_tools_csv(tools_csv_path: str, results: list) -> None:
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


def _write_agent_results_jsonl(jsonl_path: str, results: list) -> None:
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




# ---------------------------------------------------------------------------
# Run a fixed list of task files (bypass glob in run_evaluation)
# ---------------------------------------------------------------------------

def _run_task_files(
    task_dir: str,
    task_files: list,
    agent_config,
    run_config,
    verbose: bool,
    parallel: int,
) -> None:
    """Run evaluation on an explicit list of task files (bypass glob in run_evaluation)."""
    if BatchRunner is None:
        raise RuntimeError(
            "run_eval.BatchRunner is not configured. Set it via "
            "`run_eval.BatchRunner = <YourBatchRunner>` before calling."
        )
    cond = run_config.condition_config
    condition_label = cond.condition
    output_dir = _results_dir(run_config, agent_config)
    os.makedirs(output_dir, exist_ok=True)

    task_dir_name = os.path.basename(task_dir)
    model_id = agent_config.model_id

    tasks_by_id: dict = {}
    for path in task_files:
        with open(path) as f:
            task = json.load(f)
            task["id"] = path
            tasks_by_id[path] = task

    logger.info(f"\nEvaluating {len(task_files)} tasks from {task_dir_name}")
    logger.info(f"Model: {model_id}  Condition: {condition_label}")
    logger.info("=" * 60)

    try:
        batch = BatchRunner(agent_config=agent_config, run_config=run_config, max_workers=parallel)
        results = batch.run_from_files(task_files, verbose=verbose)
    except Exception as e:
        logger.error(f"  Error: {e}", exc_info=True)
        return

    csv_path = os.path.join(output_dir, "eval_results.csv")
    _write_main_csv(csv_path, results, tasks_by_id)
    _write_tools_csv(os.path.join(output_dir, "tools_breakdown.csv"), results)
    _write_agent_results_jsonl(os.path.join(output_dir, "agent_results.jsonl"), results)

    total = len(results)
    exact_matches = sum(r.get("exact_match", 0) for r in results)
    logger.info(f"  Exact Match: {exact_matches}/{total} ({100*exact_matches/total:.1f}%)" if total else "  No results")
