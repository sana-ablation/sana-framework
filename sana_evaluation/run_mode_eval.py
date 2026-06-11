#!/usr/bin/env python3
"""Run multi-axis evaluation with independent mode controls.

This runner controls four orthogonal axes:
  - search_tool quality
  - search_results richness
  - profile style
  - computation_tool behavior
"""

from __future__ import annotations

import argparse
import csv
import glob
import logging
import os
from datetime import datetime
from typing import Optional

from sana_evaluation import run_eval as base_eval
from sana_evaluation.agent_with_mode import BatchRunner as ModeBatchRunner
from sana_evaluation.config import AgentConfig, ConditionConfig, RunConfig
from sana_evaluation.env import load_repo_dotenv
from sana_evaluation.helper.prompting import normalize_debug_mode
from sana_evaluation.preflight import PreflightError, run_preflight
from sana_evaluation.tools.external.ideal.subagent_models import (
    IDEAL_SUBAGENT_MODEL_ENV,
    MAIN_MODEL_ENV,
    REPAIR_IDEAL_SUBAGENT_MODEL_ENV,
    SEARCH_IDEAL_SUBAGENT_MODEL_ENV,
    SEMANTIC_IDEAL_SUBAGENT_MODEL_ENV,
)

logger = logging.getLogger(__name__)

# Reuse run_eval orchestration while swapping only the runner implementation.
base_eval.BatchRunner = ModeBatchRunner

_AXIS_DEFAULTS = {
    "search_tool": "standard",
    "search_results": "naive",
    "profile": "standard",
    "computation_tool": "standard",
}
_BENCHMARK_CHOICES = ("lakeqa", "kramabench")
_DEFAULT_TASK_SET = "benchmarks/lakeqa/tasks-mini/tasks"
_KRAMABENCH_TASK_SET = "benchmarks/kramabench/tasks-mini/tasks"


def _variant_condition_label(
    *,
    search_tool: str,
    search_results: str,
    profile: str,
    computation_tool: str = "standard",
    k: Optional[int] = None,
    search_calls: Optional[int] = None,
    search_free: bool = False,
    search_lessguide: bool = False,
    profile_skills_enabled: bool = False,
) -> str:
    parts = [
        f"search_{search_tool}",
        f"results_{search_results}",
        f"profile_{profile}",
        f"compute_{computation_tool}",
    ]
    if k is not None:
        parts.append(f"k{k}")
    if search_calls is not None:
        parts.append(f"sc{search_calls}")
    if search_free:
        parts.append("free")
    if search_lessguide:
        parts.append("lessguide")
    parts.append("skills_on" if profile_skills_enabled else "skills_off")
    return "__".join(parts)


def _with_debug_suffix(label: str, debug_mode: Optional[str]) -> str:
    normalized = normalize_debug_mode(debug_mode)
    if normalized is None:
        return label
    return f"{label}__debug_{normalized}"


def _resolve_mode_axes(
    *,
    search_tool: Optional[str],
    search_results: Optional[str],
    profile: Optional[str],
    computation_tool: Optional[str] = None,
) -> tuple[str, str, str, str]:
    defaults = _AXIS_DEFAULTS
    return (
        search_tool or defaults["search_tool"],
        search_results or defaults["search_results"],
        profile or defaults["profile"],
        computation_tool or defaults["computation_tool"],
    )


def _validate_axis_combination(*, profile: str, skills: str) -> None:
    if skills == "on" and profile == "naive":
        raise ValueError("--skills on requires --profile standard or --profile ideal.")


def _set_or_clear_env(name: str, value: Optional[str]) -> None:
    if value:
        os.environ[name] = value
    else:
        os.environ.pop(name, None)


def _configure_ideal_subagent_models(
    *,
    main_model_name: str,
    selector_model: Optional[str],
    repair_model: Optional[str],
) -> None:
    os.environ[MAIN_MODEL_ENV] = main_model_name
    os.environ[IDEAL_SUBAGENT_MODEL_ENV] = main_model_name
    _set_or_clear_env(SEARCH_IDEAL_SUBAGENT_MODEL_ENV, selector_model)
    _set_or_clear_env(SEMANTIC_IDEAL_SUBAGENT_MODEL_ENV, selector_model)
    _set_or_clear_env(REPAIR_IDEAL_SUBAGENT_MODEL_ENV, repair_model)


def _default_task_set_for_benchmark(benchmark: str) -> str:
    if benchmark == "kramabench":
        return _KRAMABENCH_TASK_SET
    return _DEFAULT_TASK_SET


def _collect_task_files(args) -> list[str]:
    """Resolve the full task-file list the run will cover, before preflight.

    Precedence mirrors ``main()``: ``--task-continue`` > ``--all-tasks`` > ``--task-dir``.
    """
    if args.task_continue or args.all_tasks:
        task_dirs = base_eval.find_all_task_dirs(args.task_set)
        out: list[str] = []
        for d in task_dirs:
            files = sorted(glob.glob(os.path.join(d, "*.json")))
            if args.tasks_per_dir is not None:
                files = files[: args.tasks_per_dir]
            out.extend(files)
        return out
    if args.task_dir:
        files = sorted(glob.glob(os.path.join(args.task_dir, "*.json")))
        if args.tasks_per_dir is not None:
            files = files[: args.tasks_per_dir]
        return files
    return []


def _run_continue(args, agent_config: AgentConfig, run_config: RunConfig) -> None:
    """Re-run over task_set, skipping tasks already recorded for this variant."""
    condition_label = run_config.condition_config.condition
    safe_model = base_eval._display_name(agent_config)
    results_dir = base_eval._results_dir(run_config, agent_config)
    csv_path = os.path.join(results_dir, "eval_results.csv")

    completed_ids: set = set()
    if os.path.exists(csv_path):
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                tid = row.get("task_id", "").strip()
                if tid:
                    completed_ids.add(tid)

    all_task_dirs = base_eval.find_all_task_dirs(args.task_set)
    pending: dict[str, list[str]] = {}
    for task_dir in all_task_dirs:
        task_files = sorted(glob.glob(os.path.join(task_dir, "*.json")))
        if args.tasks_per_dir is not None:
            task_files = task_files[:args.tasks_per_dir]
        remaining = [p for p in task_files if p not in completed_ids]
        if remaining:
            pending[task_dir] = remaining

    if not pending:
        print("Nothing to do — all tasks already recorded.")
        return

    total_pending = sum(len(v) for v in pending.values())
    print(f"\nCondition : {condition_label}")
    print(f"Model     : {safe_model}")
    print(f"Task set  : {args.task_set}")
    print(f"Results   : {results_dir}")
    print(f"Completed : {len(completed_ids)} tasks already recorded")
    print(f"\nDirectories to evaluate ({len(pending)} dirs, {total_pending} tasks remaining):")
    for task_dir, files in sorted(pending.items()):
        print(f"  {task_dir:<40}  {len(files)} task(s)")

    print()
    answer = input("Proceed? [y/N] ").strip().lower()
    if answer != "y":
        print("Aborted.")
        return

    print()
    for task_dir, task_files in sorted(pending.items()):
        base_eval._run_task_files(
            task_dir=task_dir,
            task_files=task_files,
            agent_config=agent_config,
            run_config=run_config,
            verbose=args.verbose,
            parallel=args.parallel,
        )


def main() -> None:
    load_repo_dotenv()
    parser = argparse.ArgumentParser(description="Run multi-axis evaluation on benchmark tasks")

    # Task selection
    parser.add_argument("--task-dir", "-d", help="Single task directory to evaluate")
    parser.add_argument("--all-tasks", action="store_true", help="Evaluate all k-*-d-* directories")
    parser.add_argument(
        "--task-set",
        default=None,
        help="Base directory for --all-tasks (default: maintained benchmark task set for --benchmark)",
    )
    parser.add_argument("--tasks-per-dir", type=int, default=None, help="Limit tasks per directory")
    parser.add_argument(
        "--task-continue",
        action="store_true",
        help="Continue evaluation, skipping tasks already recorded in results for this variant.",
    )

    # Model
    parser.add_argument(
        "--model-name",
        default="bedrock/claude-sonnet-4.5",
        help="Short model name from MODEL_REGISTRY e.g. bedrock/claude-sonnet-4.5",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=8096)
    parser.add_argument(
        "--reasoning-effort",
        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
        default=None,
        help="Optional reasoning effort for OpenAI chat models (passed as reasoning_effort).",
    )
    parser.add_argument(
        "--openai-prompt-cache-key",
        default=None,
        help="Optional OpenAI prompt_cache_key for better cache routing on repeated prefixes.",
    )
    parser.add_argument(
        "--openai-prompt-cache-retention",
        default=None,
        help="Optional OpenAI prompt_cache_retention value such as 24h.",
    )
    parser.add_argument(
        "--selector-model",
        default=None,
        help="Model for selector-style ideal helper agents. Defaults to --model-name.",
    )
    parser.add_argument(
        "--repair-model",
        default=None,
        help="Model for ideal query/execute repair helper agents. Defaults to --model-name.",
    )
    parser.add_argument(
        "--debug-mode",
        choices=["none", "decision_notes"],
        default="none",
        help="Optional debug behavior. 'decision_notes' asks the model to emit short structured notes before tool calls.",
    )
    parser.add_argument(
        "--decision-notes",
        action="store_true",
        help="Convenience alias for --debug-mode decision_notes.",
    )

    # RunConfig core
    parser.add_argument("--max-tool-calls", type=int, default=30, help="Max tool calls per task")
    parser.add_argument("--sliding-window", type=int, default=40, help="Conversation last-k window")
    parser.add_argument(
        "--conversation-manager-strategy",
        choices=["summarizing", "sliding_window"],
        default="summarizing",
        help="Conversation manager to use during the run.",
    )
    parser.add_argument(
        "--summary-ratio",
        type=float,
        default=0.4,
        help="Fraction of messages to summarize when Strands reduces context.",
    )
    parser.add_argument(
        "--preserve-recent-messages",
        type=int,
        default=12,
        help="How many recent messages the summarizing manager always keeps verbatim.",
    )
    parser.add_argument("--timeout", type=int, default=600, help="Per-task timeout in seconds")
    parser.add_argument(
        "--submit-grace-seconds",
        type=int,
        default=30,
        help="Extra grace period reserved for submit_answer after the soft timeout.",
    )
    parser.add_argument(
        "--submit-only-max-tokens",
        type=int,
        default=2048,
        help="Token cap to enforce once the run enters submit-only mode.",
    )

    # Base condition + storage
    parser.add_argument(
        "--condition",
        choices=["baseline"],
        default="baseline",
        help="Base condition label used in output path nesting.",
    )
    parser.add_argument(
        "--sparse-backend",
        choices=["bm25", "splade"],
        default="bm25",
        help="Sparse backend metadata field (kept for compatibility).",
    )
    parser.add_argument("--results-output-dir", default="results", help="Base directory for outputs")
    parser.add_argument("--logs-output-dir", default="logs", help="Base directory for per-task log files")
    parser.add_argument(
        "--benchmark",
        choices=_BENCHMARK_CHOICES,
        default="lakeqa",
        help="Data-lake benchmark bucket to use for agent data tools.",
    )
    parser.add_argument(
        "--db-path",
        default=None,
        help="Path to LanceDB search index directory (overrides default ./lance_data).",
    )

    # Search controls (existing)
    parser.add_argument("--k", type=int, default=None, help="Hard-coded search result limit.")
    parser.add_argument(
        "--search-calls",
        type=int,
        default=None,
        help="Hard cap for total search calls. Calls beyond this budget are blocked.",
    )
    parser.add_argument(
        "--search-descriptions",
        choices=["naive", "description"],
        default="naive",
        help="Legacy search wrapper mode (kept for compatibility with existing flows).",
    )
    parser.add_argument(
        "--search-free",
        "--search_free",
        dest="search_free",
        action="store_true",
        help="Make active search tools cost zero against the global max-tool-calls limit.",
    )
    parser.add_argument(
        "--search-lessguide",
        "--search_lessguide",
        dest="search_lessguide",
        action="store_true",
        help="Hide search_ideal plan_exhausted guidance fields from tool payloads.",
    )

    # Mode axes
    parser.add_argument(
        "--search_tool",
        choices=["naive", "preloaded", "standard", "ideal"],
        default=None,
        help="Search tool quality axis.",
    )
    parser.add_argument(
        "--search_results",
        choices=["naive", "ideal"],
        default=None,
        help="Search result richness axis.",
    )
    parser.add_argument(
        "--profile",
        dest="profile",
        choices=["naive", "standard", "ideal"],
        default=None,
        help="Profile axis.",
    )
    parser.add_argument(
        "--computation_tool",
        choices=["standard", "ideal"],
        default=None,
        help="Computation tool axis.",
    )
    parser.add_argument(
        "--skills",
        choices=["on", "off"],
        default="off",
        help="Enable or disable the Strands AgentSkills planning/discovery skills plugin.",
    )

    # Execution
    parser.add_argument("--parallel", type=int, default=6, help="Number of parallel worker processes")
    parser.add_argument("--only-new", action="store_true", help="Skip tasks already present in the results CSV")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()
    if args.decision_notes:
        args.debug_mode = "decision_notes"
    if args.task_set is None:
        args.task_set = _default_task_set_for_benchmark(args.benchmark)

    if args.k is not None and args.k <= 0:
        parser.error("--k must be > 0")
    if args.search_calls is not None and args.search_calls <= 0:
        parser.error("--search-calls must be > 0")

    extra_model_kwargs = {}
    if args.reasoning_effort is not None:
        extra_model_kwargs["reasoning_effort"] = args.reasoning_effort

    agent_config = AgentConfig(
        model_name=args.model_name,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        openai_prompt_cache_key=args.openai_prompt_cache_key,
        openai_prompt_cache_retention=args.openai_prompt_cache_retention,
        extra_model_kwargs=extra_model_kwargs,
    )
    _configure_ideal_subagent_models(
        main_model_name=args.model_name,
        selector_model=args.selector_model,
        repair_model=args.repair_model,
    )
    search_tool_mode, search_results_mode, profile_mode, computation_tool_mode = _resolve_mode_axes(
        search_tool=args.search_tool,
        search_results=args.search_results,
        profile=args.profile,
        computation_tool=args.computation_tool,
    )
    try:
        _validate_axis_combination(profile=profile_mode, skills=args.skills)
    except ValueError as exc:
        parser.error(str(exc))
    safe_model_name = base_eval._display_name(agent_config)
    variant_condition = _variant_condition_label(
        search_tool=search_tool_mode,
        search_results=search_results_mode,
        profile=profile_mode,
        computation_tool=computation_tool_mode,
        k=args.k,
        search_calls=args.search_calls,
        search_free=args.search_free,
        search_lessguide=args.search_lessguide,
        profile_skills_enabled=args.skills == "on",
    )
    variant_condition = _with_debug_suffix(variant_condition, args.debug_mode)
    condition_label = f"modes/{safe_model_name}/{variant_condition}"
    base_eval._maybe_autoset_openai_cache_key(agent_config, condition_label)
    traces_root = os.path.join(args.results_output_dir, "traces")
    trace_dir = os.path.join(traces_root, condition_label)

    run_config = RunConfig(
        results_output_dir=args.results_output_dir,
        logs_output_dir=args.logs_output_dir,
        debug_mode=normalize_debug_mode(args.debug_mode),
        max_tool_calls=args.max_tool_calls,
        conversation_manager_strategy=args.conversation_manager_strategy,
        sliding_window_k=args.sliding_window,
        summary_ratio=args.summary_ratio,
        preserve_recent_messages=args.preserve_recent_messages,
        timeout_seconds=args.timeout,
        submit_grace_seconds=args.submit_grace_seconds,
        submit_only_max_tokens=args.submit_only_max_tokens,
        search_k=args.k,
        search_calls_limit=args.search_calls,
        search_descriptions=args.search_descriptions,
        search_db_path=args.db_path,
        search_tool_mode=search_tool_mode,
        search_results_mode=search_results_mode,
        profile_mode=profile_mode,
        computation_tool_mode=computation_tool_mode,
        profile_skills_enabled=args.skills == "on",
        search_free=args.search_free,
        search_lessguide=args.search_lessguide,
        benchmark=args.benchmark,
        condition_config=ConditionConfig(
            condition=condition_label,
            base_condition=args.condition,
            sparse_backend=args.sparse_backend,
            trace_output_dir=trace_dir,
        ),
    )

    logger.info(
        "Mode variant: %s (base=%s, search=%s, results=%s, profile=%s, compute=%s, profile_skills=%s, k=%s, search_calls=%s, search_free=%s, search_lessguide=%s, db_path=%s)",
        condition_label,
        args.condition,
        search_tool_mode,
        search_results_mode,
        profile_mode,
        computation_tool_mode,
        args.skills,
        args.k,
        args.search_calls,
        args.search_free,
        args.search_lessguide,
        args.db_path or "./lance_data",
    )

    task_files = _collect_task_files(args)
    try:
        run_preflight(run_config, task_files)
    except PreflightError as exc:
        parser.exit(2, f"{exc}\n")

    start_time = datetime.now()

    if args.task_continue:
        _run_continue(args, agent_config, run_config)
    elif args.all_tasks:
        task_dirs = base_eval.find_all_task_dirs(args.task_set)
        logger.info("Found %d task directories in '%s'", len(task_dirs), args.task_set)
        for task_dir in task_dirs:
            results = base_eval.run_evaluation(
                task_dir=task_dir,
                agent_config=agent_config,
                run_config=run_config,
                verbose=args.verbose,
                only_new=args.only_new,
                parallel=args.parallel,
                tasks_per_dir=args.tasks_per_dir,
            )
            base_eval.print_comparison_table(results)
    elif args.task_dir:
        results = base_eval.run_evaluation(
            task_dir=args.task_dir,
            agent_config=agent_config,
            run_config=run_config,
            verbose=args.verbose,
            only_new=args.only_new,
            parallel=args.parallel,
            tasks_per_dir=args.tasks_per_dir,
        )
        base_eval.print_comparison_table(results)
    else:
        parser.print_help()

    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info("\nTotal evaluation time: %.1fs", elapsed)


if __name__ == "__main__":
    main()
