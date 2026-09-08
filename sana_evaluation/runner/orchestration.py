"""Orchestration for evaluation runs.

Not a CLI — `sana_evaluation.cli` is the entry point. Callers pass the batch
runner class explicitly; there is no module-level seam to monkey-patch.
"""

import csv
import glob
import hashlib
import json
import logging
import os
from datetime import datetime
from typing import Optional

from sana_evaluation.config import AgentConfig, ConditionConfig, RunConfig
from sana_evaluation.helper.prompting import normalize_debug_mode
from sana_evaluation.runner.reporting import (
    print_comparison_table,
    write_agent_results_jsonl,
    write_main_csv,
    write_tools_csv,
)

logger = logging.getLogger(__name__)


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


# OpenAI caps prompt_cache_key at 64 characters. /v1/chat/completions accepts
# longer values silently; /v1/responses rejects them with HTTP 400.
_MAX_PROMPT_CACHE_KEY = 64


def _bounded_cache_key(raw: str) -> str:
    """Fit a cache key into OpenAI's 64-character limit without losing identity.

    Plain truncation would collide, because these variant labels are a shared
    prefix followed by the axis settings that distinguish them -- the tail is
    exactly the part that differs. Keeping a prefix and appending a digest of the
    whole label stays under the cap, stays stable across rounds (so the router
    still lands repeat requests on the same shard), and keeps distinct variants
    distinct.
    """
    if len(raw) <= _MAX_PROMPT_CACHE_KEY:
        return raw
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
    return f"{raw[: _MAX_PROMPT_CACHE_KEY - len(digest) - 1]}-{digest}"


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
    agent_config.openai_prompt_cache_key = _bounded_cache_key(
        f"{_display_name(agent_config)}:{variant_label}"
    )


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
    *,
    batch_runner_cls,
    verbose: bool = False,
    only_new: bool = False,
    parallel: int = 6,
    tasks_per_dir: Optional[int] = None,
    task_files: Optional[list] = None,
) -> dict:
    """Run evaluation on a task directory and return {model_id -> {summary, results}}.

    ``task_files`` overrides the per-directory glob. Passing an explicit list lets
    a caller pool every task across all ``k-*-d-*`` directories into ONE worker
    pool. Without it, ``--all-tasks`` calls this once per directory and each call
    builds its own pool, so directories run sequentially and concurrency is capped
    by the largest directory rather than by ``parallel``.

    The paths are used verbatim: runtime-profile lookup keys off the path suffix
    after ``benchmarks/<bench>/tasks-mini/tasks``, so the ``k-*-d-*`` segment must
    survive. Pooling the file list rather than flattening the tree preserves it,
    and keeps colliding basenames distinct.
    """
    cond = run_config.condition_config
    condition_label = cond.condition
    safe_model = _display_name(agent_config)
    output_dir = _results_dir(run_config, agent_config)
    os.makedirs(output_dir, exist_ok=True)

    if task_files is None:
        task_files = sorted(glob.glob(os.path.join(task_dir, "*.json")))
    else:
        task_files = list(task_files)
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
        batch = batch_runner_cls(agent_config=agent_config, run_config=run_config, max_workers=parallel)
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
    write_main_csv(csv_path, results, tasks_by_id)

    # Write per-tool breakdown CSV
    tools_csv_path = os.path.join(output_dir, "tools_breakdown.csv")
    write_tools_csv(tools_csv_path, results)

    # Write agent_results JSONL (one line per task)
    jsonl_path = os.path.join(output_dir, "agent_results.jsonl")
    write_agent_results_jsonl(jsonl_path, results)

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
    *,
    batch_runner_cls,
) -> None:
    """Run evaluation on an explicit list of task files (bypass glob in run_evaluation)."""
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
        batch = batch_runner_cls(agent_config=agent_config, run_config=run_config, max_workers=parallel)
        results = batch.run_from_files(task_files, verbose=verbose)
    except Exception as e:
        logger.error(f"  Error: {e}", exc_info=True)
        return

    csv_path = os.path.join(output_dir, "eval_results.csv")
    write_main_csv(csv_path, results, tasks_by_id)
    write_tools_csv(os.path.join(output_dir, "tools_breakdown.csv"), results)
    write_agent_results_jsonl(os.path.join(output_dir, "agent_results.jsonl"), results)

    total = len(results)
    exact_matches = sum(r.get("exact_match", 0) for r in results)
    logger.info(f"  Exact Match: {exact_matches}/{total} ({100*exact_matches/total:.1f}%)" if total else "  No results")


# ---------------------------------------------------------------------------
# Whole-task-set run loops
# ---------------------------------------------------------------------------

def _run_all_tasks_pooled(
    *,
    task_set: str,
    agent_config,
    run_config,
    verbose: bool,
    only_new: bool,
    parallel: int,
    tasks_per_dir: Optional[int],
    batch_runner_cls,
) -> None:
    """Run every task across all directories through ONE worker pool.

    The per-directory path builds a separate pool per `k-*-d-*` directory and
    runs them in sequence, so concurrency is capped by the largest directory. On
    a 20-task subset spread over 11 directories that is ~1.3-way concurrency no
    matter what `--parallel` says.

    Paths are passed through untouched: runtime-profile lookup keys off the
    suffix after `benchmarks/<bench>/tasks-mini/tasks`, so the `k-*-d-*` segment
    has to survive. Pooling the file list rather than flattening the tree keeps
    it, and keeps colliding basenames (subset20b has `task_6` three times)
    distinct.
    """
    task_dirs = find_all_task_dirs(task_set)
    logger.info("Found %d task directories in '%s'", len(task_dirs), task_set)

    pooled: list = []
    for task_dir in task_dirs:
        files = sorted(glob.glob(os.path.join(task_dir, "*.json")))
        if tasks_per_dir is not None:
            files = files[:tasks_per_dir]
        pooled.extend(files)

    logger.info("Pooling %d tasks into one pool of %d workers", len(pooled), parallel)
    results = run_evaluation(
        task_dir=task_set,
        agent_config=agent_config,
        run_config=run_config,
        batch_runner_cls=batch_runner_cls,
        verbose=verbose,
        only_new=only_new,
        parallel=parallel,
        task_files=pooled,
    )
    print_comparison_table(results)
