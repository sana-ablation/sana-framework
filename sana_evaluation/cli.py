#!/usr/bin/env python3
"""Evaluation entry point.

    python -m sana_evaluation.cli [smoke|full] [options]

A preset is a default set, nothing more: explicit flags always win, and
omitting the preset reproduces the historical run_mode_eval defaults exactly.

This module controls four orthogonal axes:
  - search_tool quality
  - search_results richness (minimal | rich)
  - profile style
  - computation_tool behavior
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

from sana_evaluation.config import AgentConfig, ConditionConfig, RunConfig
from sana_evaluation.env import load_repo_dotenv
from sana_evaluation.helper.prompting import normalize_debug_mode
from sana_evaluation.preflight import PreflightError, run_preflight
from sana_evaluation.runner import orchestration as base_eval
from sana_evaluation.runner.batch import BatchRunner as ModeBatchRunner
from sana_evaluation.runner.reporting import print_comparison_table
from sana_evaluation.tools.external.ideal.subagent_models import (
    IDEAL_SUBAGENT_MODEL_ENV,
    MAIN_MODEL_ENV,
    REPAIR_IDEAL_SUBAGENT_MODEL_ENV,
    SEARCH_IDEAL_SUBAGENT_MODEL_ENV,
    SEMANTIC_IDEAL_SUBAGENT_MODEL_ENV,
)

logger = logging.getLogger(__name__)

BENCHMARKS = ("lakeqa", "kramabench")

_AXIS_DEFAULTS = {
    "search_tool": "standard",
    "search_results": "rich",
    "profile": "standard",
    "computation_tool": "standard",
}
_DEFAULT_TASK_SET = "benchmarks/lakeqa/tasks-mini/tasks"
_KRAMABENCH_TASK_SET = "benchmarks/kramabench/tasks-mini/tasks"
_DEFAULT_SMOKE_TASK_DIR = "k-5-d-4"
_KRAMABENCH_SMOKE_TASK_DIR = "k-2-d-1-s-1"
_KRAMABENCH_LOGS_OUTPUT_DIR = "log-kramabench"
_KRAMABENCH_RESULTS_OUTPUT_DIR = "results-kramabench"

# Shorthands that predate the canonical provider/model spelling. A name with a
# "/" is passed through, so only these bare words need translating.
_MODEL_ALIASES = {
    "gpt5.2": "openai/gpt-5.2",
    "gpt-5.2": "openai/gpt-5.2",
    "gpt5-nano": "openai/gpt-5-nano",
    "gpt-5-nano": "openai/gpt-5-nano",
    "gpt5.4-nano": "openai/gpt-5.4-nano",
    "gpt-5.4-nano": "openai/gpt-5.4-nano",
}

PRESETS = {
    "smoke": dict(search="ideal", results="rich", profile="ideal", compute="ideal",
                  verbose=True, task_continue=False, tasks_per_dir=2),
    "full":  dict(search="ideal", results="rich", profile="ideal", compute="ideal",
                  verbose=True, task_continue=True),
}


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m sana_evaluation.cli")
    p.add_argument("preset", nargs="?", choices=sorted(PRESETS),
                   help="default set to start from; explicit flags override it")

    ax = p.add_argument_group("experiment axes")
    ax.add_argument("--search", "--search-tool", "--search_tool", dest="search",
                    choices=("naive", "preloaded", "standard", "ideal", "web"),
                    default="standard")
    # naive/ideal are the former names for minimal/rich; they are canonicalised
    # in _resolve_mode_axes so they cannot produce a second variant directory.
    ax.add_argument("--results", "--search-results", "--search_results", dest="results",
                    choices=("minimal", "rich", "naive", "ideal"), default="rich")
    ax.add_argument("--profile", "--plans", dest="profile",
                    choices=("naive", "standard", "ideal"), default="standard")
    ax.add_argument("--compute", "--computation-tool", "--computation_tool",
                    dest="compute", choices=("standard", "ideal"), default="standard")
    ax.add_argument("--skills", choices=("on", "off"), default="off")

    bm = p.add_argument_group("benchmark and tasks")
    bm.add_argument("--benchmark", choices=BENCHMARKS, default="lakeqa")
    bm.add_argument("--task-set")
    bm.add_argument("--task-dir", "-d", help="run this one directory of tasks")
    bm.add_argument("--tasks-per-dir", type=int)
    bm.add_argument("--all-tasks", action="store_true")
    bm.add_argument("--pool-tasks", action="store_true")
    bm.add_argument("--db-path", "--db", dest="db_path")
    cont = bm.add_mutually_exclusive_group()
    cont.add_argument("--task-continue", "--continue", dest="task_continue",
                      action="store_true", default=False)
    cont.add_argument("--no-continue", dest="task_continue", action="store_false")

    md = p.add_argument_group("model")
    md.add_argument("--model", "--model-name", dest="model_name",
                    default="bedrock/claude-sonnet-4.5")
    md.add_argument("--temperature", type=float, default=0.0)
    md.add_argument("--max-tokens", type=int, default=8096)
    md.add_argument("--submit-only-max-tokens", type=int, default=2048)
    md.add_argument("--reasoning-effort",
                    choices=("none", "minimal", "low", "medium", "high", "xhigh"))
    md.add_argument("--openai-prompt-cache-key")
    md.add_argument("--openai-prompt-cache-retention")

    orc = p.add_argument_group("oracle subagents (ideal axes only)")
    orc.add_argument("--selector-model", help="sets the search AND semantic subagent models")
    orc.add_argument("--repair-model")

    rn = p.add_argument_group("run control")
    rn.add_argument("--parallel", type=int, default=6)
    rn.add_argument("--timeout", type=int, default=600)
    rn.add_argument("--max-tool-calls", type=int, default=30)
    rn.add_argument("--submit-grace-seconds", type=int, default=30)
    rn.add_argument("--no-s3", "--no_s3", dest="no_s3", action="store_true")

    sb = p.add_argument_group("search budget")
    sb.add_argument("--k", type=int)
    sb.add_argument("--search-calls", type=int,
                    help="cap on search calls specifically (--max-tool-calls caps all)")
    sb.add_argument("--search-free", "--search_free", dest="search_free",
                    action="store_true",
                    help="search calls cost zero against --max-tool-calls")
    sb.add_argument("--search-descriptions", choices=("naive", "description"),
                    default="naive")

    cm = p.add_argument_group("conversation manager")
    cm.add_argument("--conversation-manager-strategy",
                    choices=("summarizing", "sliding_window"), default="summarizing")
    cm.add_argument("--sliding-window", type=int, default=40)
    cm.add_argument("--preserve-recent-messages", type=int, default=12)
    cm.add_argument("--summary-ratio", type=float, default=0.4)

    out = p.add_argument_group("output and debug")
    out.add_argument("--logs-output-dir", default="logs")
    out.add_argument("--results-output-dir", default="results")
    out.add_argument("--debug-mode", choices=("none", "decision_notes"), default="none")
    out.add_argument("--verbose", "-v", action="store_true", default=False)
    return p


def _default_output_roots(benchmark: str, *, smoke: bool) -> tuple[str, str]:
    if benchmark == "kramabench":
        return _KRAMABENCH_LOGS_OUTPUT_DIR, _KRAMABENCH_RESULTS_OUTPUT_DIR
    if smoke:
        return "test_logs", "test_results"
    return "logs", "results"


def _preset_defaults(preset: str, benchmark: str) -> dict:
    """The preset's defaults, plus the output roots it implies for a benchmark.

    Only a preset moves the output roots: a preset-less run keeps the historical
    ``logs/`` and ``results/`` regardless of ``--benchmark``.
    """
    defaults = dict(PRESETS[preset])
    logs, results = _default_output_roots(benchmark, smoke=preset == "smoke")
    defaults["logs_output_dir"] = logs
    defaults["results_output_dir"] = results
    return defaults


def parse(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Two-pass parse: the preset supplies defaults, explicit flags override."""
    parser = build_parser()
    pre, _ = parser.parse_known_args(argv)
    if pre.preset:
        parser.set_defaults(**_preset_defaults(pre.preset, pre.benchmark))
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Resolution and validation of the parsed namespace
# ---------------------------------------------------------------------------

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
    no_s3: bool = False,
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
    if no_s3:
        parts.append("nos3")
    parts.append("skills_on" if profile_skills_enabled else "skills_off")
    return "__".join(parts)


def _resolve_mode_axes(
    *,
    search_tool: Optional[str],
    search_results: Optional[str],
    profile: Optional[str],
    computation_tool: Optional[str] = None,
) -> tuple[str, str, str, str]:
    from sana_evaluation.runner.modes import _normalize_result_mode

    defaults = _AXIS_DEFAULTS
    return (
        search_tool or defaults["search_tool"],
        # Canonicalised so the deprecated spellings do not produce a second set
        # of variant directories for the same condition.
        _normalize_result_mode(search_results, defaults["search_results"], "search_results"),
        profile or defaults["profile"],
        computation_tool or defaults["computation_tool"],
    )


def _validate_axis_combination(*, profile: str, skills: str) -> None:
    if skills == "on" and profile == "naive":
        raise ValueError("--skills on requires --profile standard or --profile ideal.")


def _normalize_model_name(raw: str) -> str:
    model = (raw or "").strip()
    if not model:
        raise ValueError("--model cannot be empty.")

    alias = _MODEL_ALIASES.get(model.lower())
    if alias:
        return alias
    if "/" in model:
        return model
    raise ValueError(
        f"Unknown model shorthand '{model}'. Use a canonical model name like "
        f"'openai/gpt-5.2' or one of: {', '.join(sorted(_MODEL_ALIASES))}."
    )


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


def _default_smoke_task_dir(benchmark: str) -> str:
    if benchmark == "kramabench":
        return _KRAMABENCH_SMOKE_TASK_DIR
    return _DEFAULT_SMOKE_TASK_DIR


def _named_task_sets(benchmark: str) -> list[str]:
    root = Path("benchmarks") / benchmark
    if not root.is_dir():
        return []
    return sorted(d.name for d in root.iterdir() if (d / "tasks").is_dir())


def resolve_task_set(task_set: Optional[str], benchmark: str) -> str:
    """Accept a named set as well as a path.

    ``--task-set tasks_20_subset`` is shorthand for ``benchmarks/<benchmark>/tasks_20_subset/tasks``.
    A value containing a separator, or naming a directory that exists, is used
    as-is, so paths keep working unchanged.
    """
    if not task_set:
        return _default_task_set_for_benchmark(benchmark)
    raw = str(task_set).strip()
    if os.sep in raw or "/" in raw or Path(raw).is_dir():
        return raw
    candidate = Path("benchmarks") / benchmark / raw / "tasks"
    if candidate.is_dir():
        return str(candidate)
    raise ValueError(
        f"Unknown task set '{task_set}' for benchmark '{benchmark}'. "
        f"Expected a path, or a named set under benchmarks/{benchmark}/<name>/tasks. "
        f"Available: {', '.join(_named_task_sets(benchmark)) or '(none)'}"
    )


def _resolve_task_dir(task_dir: str, task_set: str) -> str:
    """Accept a bare bucket name (``k-5-d-4``) as well as a path."""
    candidate = Path(task_dir).expanduser()
    if candidate.is_dir():
        return str(candidate)
    nested = Path(task_set) / task_dir
    if nested.is_dir():
        return str(nested)
    raise ValueError(f"--task-dir does not exist: {task_dir}")


def resolve(args: argparse.Namespace) -> argparse.Namespace:
    """Normalise and validate a parsed namespace in place.

    Everything the run needs is decided here, against the resolved values --
    never against raw argv. Raises ``ValueError`` with a user-facing message.
    """
    if args.k is not None and args.k <= 0:
        raise ValueError("--k must be > 0")
    if args.search_calls is not None and args.search_calls <= 0:
        raise ValueError("--search-calls must be > 0")
    if args.parallel <= 0:
        raise ValueError("--parallel must be > 0")
    if args.timeout <= 0:
        raise ValueError("--timeout must be > 0")
    if args.submit_grace_seconds < 0:
        raise ValueError("--submit-grace-seconds must be >= 0")

    args.model_name = _normalize_model_name(args.model_name)
    if args.selector_model is not None:
        args.selector_model = _normalize_model_name(args.selector_model)
    if args.repair_model is not None:
        args.repair_model = _normalize_model_name(args.repair_model)

    args.task_set = resolve_task_set(args.task_set, args.benchmark)

    # A preset picks the task scope when nothing else does: smoke takes one
    # small bucket, full takes the whole set.
    selected = bool(args.task_dir or args.all_tasks or args.task_continue)
    if not selected and args.preset == "smoke":
        args.task_dir = _default_smoke_task_dir(args.benchmark)
    elif not selected and args.preset == "full":
        args.all_tasks = True

    if args.task_dir:
        args.task_dir = _resolve_task_dir(args.task_dir, args.task_set)

    (
        args.search,
        args.results,
        args.profile,
        args.compute,
    ) = _resolve_mode_axes(
        search_tool=args.search,
        search_results=args.results,
        profile=args.profile,
        computation_tool=args.compute,
    )
    _validate_axis_combination(profile=args.profile, skills=args.skills)
    return args


def _task_scope(args: argparse.Namespace) -> str:
    if args.task_continue:
        return f"resume pending tasks under {args.task_set}"
    if args.all_tasks:
        return f"all tasks under {args.task_set}"
    if args.task_dir:
        if args.tasks_per_dir is not None:
            return f"{args.task_dir} (first {args.tasks_per_dir} tasks)"
        return args.task_dir
    return "nothing selected — pass --task-dir, --all-tasks or --task-continue"


def _collect_task_files(args: argparse.Namespace) -> list:
    """Resolve the full task-file list the run will cover, before preflight.

    Precedence mirrors ``main()``: ``--task-continue`` > ``--all-tasks`` > ``--task-dir``.
    """
    if args.task_continue or args.all_tasks:
        task_dirs = base_eval.find_all_task_dirs(args.task_set)
        out: list = []
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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    load_repo_dotenv()
    parser = build_parser()
    args = parse(argv)
    try:
        resolve(args)
    except ValueError as exc:
        parser.error(str(exc))

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
    safe_model_name = base_eval._display_name(agent_config)
    variant_condition = _variant_condition_label(
        search_tool=args.search,
        search_results=args.results,
        profile=args.profile,
        computation_tool=args.compute,
        k=args.k,
        search_calls=args.search_calls,
        search_free=args.search_free,
        profile_skills_enabled=args.skills == "on",
        no_s3=args.no_s3,
    )
    variant_condition = base_eval._with_debug_suffix(variant_condition, args.debug_mode)
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
        search_tool_mode=args.search,
        search_results_mode=args.results,
        profile_mode=args.profile,
        computation_tool_mode=args.compute,
        profile_skills_enabled=args.skills == "on",
        search_free=args.search_free,
        no_s3=args.no_s3,
        benchmark=args.benchmark,
        condition_config=ConditionConfig(
            condition=condition_label,
            base_condition="baseline",
            trace_output_dir=trace_dir,
        ),
    )

    print(f"Variant     : {condition_label}")
    print(f"Model       : {args.model_name}")
    print(
        "Axes        : "
        f"search={args.search} results={args.results} profile={args.profile} "
        f"compute={args.compute} skills={args.skills}"
    )
    print(f"Lance DB    : {args.db_path or './lance_data'}")
    print(f"Task scope  : {_task_scope(args)}")
    print(f"Logs root   : {args.logs_output_dir}")
    print(f"Results root: {args.results_output_dir}")

    task_files = _collect_task_files(args)
    try:
        run_preflight(run_config, task_files)
    except PreflightError as exc:
        parser.exit(2, f"{exc}\n")

    start_time = datetime.now()

    if args.task_continue:
        base_eval._run_continue(
            task_set=args.task_set,
            tasks_per_dir=args.tasks_per_dir,
            agent_config=agent_config,
            run_config=run_config,
            verbose=args.verbose,
            parallel=args.parallel,
            batch_runner_cls=ModeBatchRunner,
        )
    elif args.all_tasks and args.pool_tasks:
        base_eval._run_all_tasks_pooled(
            task_set=args.task_set,
            agent_config=agent_config,
            run_config=run_config,
            verbose=args.verbose,
            parallel=args.parallel,
            tasks_per_dir=args.tasks_per_dir,
            batch_runner_cls=ModeBatchRunner,
        )
    elif args.all_tasks:
        task_dirs = base_eval.find_all_task_dirs(args.task_set)
        logger.info("Found %d task directories in '%s'", len(task_dirs), args.task_set)
        for task_dir in task_dirs:
            results = base_eval.run_evaluation(
                task_dir=task_dir,
                agent_config=agent_config,
                run_config=run_config,
                batch_runner_cls=ModeBatchRunner,
                verbose=args.verbose,
                parallel=args.parallel,
                tasks_per_dir=args.tasks_per_dir,
            )
            print_comparison_table(results)
    elif args.task_dir:
        results = base_eval.run_evaluation(
            task_dir=args.task_dir,
            agent_config=agent_config,
            run_config=run_config,
            batch_runner_cls=ModeBatchRunner,
            verbose=args.verbose,
            parallel=args.parallel,
            tasks_per_dir=args.tasks_per_dir,
        )
        print_comparison_table(results)
    else:
        parser.print_help()
        return 1

    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info("\nTotal evaluation time: %.1fs", elapsed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
