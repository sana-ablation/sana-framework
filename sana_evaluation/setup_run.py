#!/usr/bin/env python3
"""Friendly preset wrapper around ``run_mode_eval``."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional, Sequence

from sana_evaluation.env import load_repo_dotenv

_SEARCH_MODE_CHOICES = ("naive", "preloaded", "standard", "ideal")
_MANAGEMENT_MODE_CHOICES = ("naive", "standard", "ideal")
_RESULT_MODE_CHOICES = ("naive", "ideal")
_COMPUTATION_MODE_CHOICES = ("standard", "ideal")
_SKILLS_CHOICES = ("on", "off")
_BENCHMARK_CHOICES = ("lakeqa", "kramabench")
_REASONING_EFFORT_CHOICES = ("none", "minimal", "low", "medium", "high", "xhigh")
_DEFAULT_TASK_SET = "benchmarks/lakeqa/tasks-mini/tasks"
_DEFAULT_SMOKE_TASK_DIR = "k-5-d-4"
_KRAMABENCH_TASK_SET = "benchmarks/kramabench/tasks-mini/tasks"
_KRAMABENCH_SMOKE_TASK_DIR = "k-2-d-1-s-1"
_DEFAULT_SMOKE_TASK_LIMIT = 2
_KRAMABENCH_LOGS_OUTPUT_DIR = "log-kramabench"
_KRAMABENCH_RESULTS_OUTPUT_DIR = "results-kramabench"
_MODEL_ALIASES = {
    "gpt5.2": "openai/gpt-5.2",
    "gpt-5.2": "openai/gpt-5.2",
    "gpt5-nano": "openai/gpt-5-nano",
    "gpt-5-nano": "openai/gpt-5-nano",
    "gpt5.4-nano": "openai/gpt-5.4-nano",
    "gpt-5.4-nano": "openai/gpt-5.4-nano",
}
_DB_HINTS = ("lance_data", "lance_kramabench_base", "lance_kramabench_infused")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Friendly presets for sana_evaluation.run_mode_eval",
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--search",
        choices=_SEARCH_MODE_CHOICES,
        default=None,
        help="Search-tool axis. Default: ideal.",
    )
    common.add_argument(
        "--results",
        choices=_RESULT_MODE_CHOICES,
        default=None,
        help="Search-results axis. Default: ideal.",
    )
    common.add_argument(
        "--profile",
        "--plans",
        dest="profile",
        choices=_MANAGEMENT_MODE_CHOICES,
        default=None,
        help="Planning/profile axis: naive, standard, or ideal. Default: ideal.",
    )
    common.add_argument(
        "--skills",
        choices=_SKILLS_CHOICES,
        default=None,
        help="Enable or disable the Strands AgentSkills planning/discovery skills plugin.",
    )
    common.add_argument(
        "--compute",
        choices=_COMPUTATION_MODE_CHOICES,
        default=None,
        help="Data-analysis/compute axis. Default: ideal.",
    )
    common.add_argument(
        "--benchmark",
        choices=_BENCHMARK_CHOICES,
        default="lakeqa",
        help="Data-lake benchmark bucket to use for agent data tools.",
    )
    common.add_argument("--k", type=int, default=None)
    common.add_argument("--model", default="bedrock/claude-sonnet-4.5")
    common.add_argument(
        "--reasoning-effort",
        choices=_REASONING_EFFORT_CHOICES,
        default=None,
    )
    common.add_argument("--openai-prompt-cache-key", default=None)
    common.add_argument("--openai-prompt-cache-retention", default=None)
    common.add_argument(
        "--selector-model",
        default=None,
        help="Model for selector-style ideal helper agents. Defaults to --model.",
    )
    common.add_argument(
        "--repair-model",
        default=None,
        help="Model for ideal query/execute repair helper agents. Defaults to --model.",
    )
    common.add_argument("--db", default=None, help="Lance DB root, required on every run.")
    common.add_argument(
        "--condition",
        choices=("baseline",),
        default="baseline",
    )
    common.add_argument("--parallel", type=int, default=None)
    common.add_argument("--timeout", type=int, default=None)
    common.add_argument("--submit-grace-seconds", type=int, default=None)
    common.add_argument(
        "--search-free",
        "--search_free",
        dest="search_free",
        action="store_true",
        help="Make active search tools cost zero against the global max-tool-calls limit.",
    )
    common.add_argument(
        "--search-lessguide",
        "--search_lessguide",
        dest="search_lessguide",
        action="store_true",
        help="Hide search_ideal plan_exhausted guidance fields from tool payloads.",
    )
    common.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=True,
        help="Print verbose per-task runtime logs. Enabled by default.",
    )

    smoke = subparsers.add_parser("smoke", parents=[common], help="Run a lightweight smoke eval.")
    smoke.add_argument(
        "--task-dir",
        default=None,
        help="Optional task directory override inside the default task set.",
    )

    full = subparsers.add_parser("full", parents=[common], help="Run the full default task-set eval.")
    full_continue = full.add_mutually_exclusive_group()
    full_continue.add_argument(
        "--task-continue",
        "--continue",
        dest="task_continue",
        action="store_true",
        default=True,
        help="Resume: skip tasks already recorded in this variant's CSV.",
    )
    full_continue.add_argument(
        "--no-continue",
        dest="task_continue",
        action="store_false",
        help="Run every task even if this variant's CSV already contains rows.",
    )
    return parser


def _known_db_choices(cwd: Path) -> list[str]:
    return [name for name in _DB_HINTS if (cwd / name).exists()]


def _missing_db_error(cwd: Path) -> str:
    choices = _known_db_choices(cwd)
    if choices:
        return (
            "--db is required. Choose one of: "
            + ", ".join(choices)
            + ", or provide a custom path."
        )
    return (
        "--db is required. Use a Lance DB root such as lance_data, "
        "or provide a custom path."
    )


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


def _validate_db_arg(db_arg: Optional[str], cwd: Path) -> str:
    if not db_arg:
        raise ValueError(_missing_db_error(cwd))

    db_path = Path(db_arg).expanduser()
    if not db_path.is_absolute():
        db_path = cwd / db_path
    if not db_path.exists():
        raise ValueError(f"--db path not found: {db_arg}")
    return _display_path(db_path, cwd)


def _display_path(path: Path, cwd: Path) -> str:
    try:
        return str(path.relative_to(cwd))
    except ValueError:
        return str(path)


def _default_task_set(benchmark: str) -> str:
    if benchmark == "kramabench":
        return _KRAMABENCH_TASK_SET
    return _DEFAULT_TASK_SET


def _default_smoke_task_dir(benchmark: str) -> str:
    if benchmark == "kramabench":
        return _KRAMABENCH_SMOKE_TASK_DIR
    return _DEFAULT_SMOKE_TASK_DIR


def _resolve_smoke_task_dir(task_dir_arg: Optional[str], cwd: Path, *, benchmark: str) -> Path:
    if task_dir_arg:
        candidate = Path(task_dir_arg).expanduser()
        if not candidate.is_absolute():
            candidate = cwd / candidate
        if not candidate.is_dir():
            raise ValueError(f"--task-dir does not exist: {task_dir_arg}")
        return candidate

    task_set = _default_task_set(benchmark)
    smoke_task_dir = _default_smoke_task_dir(benchmark)
    candidate = cwd / task_set / smoke_task_dir
    if not candidate.is_dir():
        raise ValueError(
            f"Default smoke task dir {task_set}/{smoke_task_dir} "
            "not found. Pass --task-dir explicitly."
        )
    return candidate


def _display_command(command: Sequence[str]) -> str:
    if not command:
        return ""
    printable = list(command)
    if Path(printable[0]) == Path(sys.executable):
        printable[0] = "python"
    return shlex.join(printable)


def _default_output_roots(benchmark: str, *, smoke: bool) -> tuple[str, str]:
    if benchmark == "kramabench":
        return _KRAMABENCH_LOGS_OUTPUT_DIR, _KRAMABENCH_RESULTS_OUTPUT_DIR
    if smoke:
        return "test_logs", "test_results"
    return "logs", "results"


_AXIS_DEFAULTS = {
    "search": "ideal",
    "results": "ideal",
    "profile": "ideal",
    "compute": "ideal",
}


def _resolve_axes(args: argparse.Namespace) -> None:
    """Fill axis defaults. Explicit user args always win."""
    for axis, fallback in _AXIS_DEFAULTS.items():
        if getattr(args, axis) is None:
            setattr(args, axis, fallback)


def _validate_axis_combination(args: argparse.Namespace) -> None:
    if args.skills == "on" and args.profile == "naive":
        raise ValueError("--skills on requires --profile standard or --profile ideal.")


def _build_run_mode_command(args: argparse.Namespace, cwd: Path) -> tuple[list[str], dict[str, str]]:
    _resolve_axes(args)
    _validate_axis_combination(args)
    model_name = _normalize_model_name(args.model)
    db_arg = _validate_db_arg(args.db, cwd)

    command = [
        sys.executable,
        "-m",
        "sana_evaluation.run_mode_eval",
        "--search_tool",
        args.search,
        "--search_results",
        args.results,
        "--profile",
        args.profile,
        "--model-name",
        model_name,
        "--condition",
        args.condition,
        "--db-path",
        db_arg,
    ]

    if args.k is not None:
        command.extend(["--k", str(args.k)])
    if args.compute is not None:
        command.extend(["--computation_tool", args.compute])
    if args.benchmark != "lakeqa":
        command.extend(["--benchmark", args.benchmark])
    if args.skills is not None:
        command.extend(["--skills", args.skills])
    if args.reasoning_effort is not None:
        command.extend(["--reasoning-effort", args.reasoning_effort])
    if args.openai_prompt_cache_key is not None:
        command.extend(["--openai-prompt-cache-key", args.openai_prompt_cache_key])
    if args.openai_prompt_cache_retention is not None:
        command.extend(["--openai-prompt-cache-retention", args.openai_prompt_cache_retention])
    if args.selector_model is not None:
        command.extend(["--selector-model", _normalize_model_name(args.selector_model)])
    if args.repair_model is not None:
        command.extend(["--repair-model", _normalize_model_name(args.repair_model)])
    if args.parallel is not None:
        command.extend(["--parallel", str(args.parallel)])
    if args.timeout is not None:
        command.extend(["--timeout", str(args.timeout)])
    if args.submit_grace_seconds is not None:
        command.extend(["--submit-grace-seconds", str(args.submit_grace_seconds)])
    if args.search_free:
        command.append("--search-free")
    if args.search_lessguide:
        command.append("--search-lessguide")
    if args.verbose:
        command.append("--verbose")

    if args.subcommand == "smoke":
        task_dir = _resolve_smoke_task_dir(args.task_dir, cwd, benchmark=args.benchmark)
        task_dir_display = _display_path(task_dir, cwd)
        logs_output_dir, results_output_dir = _default_output_roots(args.benchmark, smoke=True)
        command.extend(
            [
                "--task-dir",
                task_dir_display,
                "--tasks-per-dir",
                str(_DEFAULT_SMOKE_TASK_LIMIT),
                "--logs-output-dir",
                logs_output_dir,
                "--results-output-dir",
                results_output_dir,
            ]
        )
        metadata = {
            "db_path": db_arg,
            "task_scope": f"{task_dir_display} (first {_DEFAULT_SMOKE_TASK_LIMIT} tasks)",
            "logs_output_dir": logs_output_dir,
            "results_output_dir": results_output_dir,
        }
        return command, metadata

    task_continue = bool(getattr(args, "task_continue", False))
    logs_output_dir, results_output_dir = _default_output_roots(args.benchmark, smoke=False)
    task_set = _default_task_set(args.benchmark)
    if task_continue:
        command.extend(
            [
                "--task-continue",
                "--task-set",
                task_set,
                "--logs-output-dir",
                logs_output_dir,
                "--results-output-dir",
                results_output_dir,
            ]
        )
        scope = f"resume pending tasks under {task_set}"
    else:
        command.extend(
            [
                "--all-tasks",
                "--task-set",
                task_set,
                "--logs-output-dir",
                logs_output_dir,
                "--results-output-dir",
                results_output_dir,
            ]
        )
        scope = f"all tasks under {task_set}"
    metadata = {
        "db_path": db_arg,
        "task_scope": scope,
        "logs_output_dir": logs_output_dir,
        "results_output_dir": results_output_dir,
    }
    return command, metadata


def run(
    argv: Optional[Sequence[str]] = None,
    *,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    cwd: Optional[Path] = None,
) -> list[str]:
    parser = _build_parser()
    args = parser.parse_args(argv)
    repo_root = Path.cwd() if cwd is None else Path(cwd)
    load_repo_dotenv(repo_root)

    if args.k is not None and args.k <= 0:
        parser.error("--k must be > 0")
    if args.parallel is not None and args.parallel <= 0:
        parser.error("--parallel must be > 0")
    if args.timeout is not None and args.timeout <= 0:
        parser.error("--timeout must be > 0")
    if args.submit_grace_seconds is not None and args.submit_grace_seconds < 0:
        parser.error("--submit-grace-seconds must be >= 0")

    try:
        command, metadata = _build_run_mode_command(args, repo_root)
    except ValueError as exc:
        parser.error(str(exc))

    print(f"Resolved command: {_display_command(command)}")
    print(f"Lance DB: {metadata['db_path']}")
    print(f"Task scope: {metadata['task_scope']}")
    print(f"Logs root: {metadata['logs_output_dir']}")
    print(f"Results root: {metadata['results_output_dir']}")

    runner(command, check=True, cwd=str(repo_root))
    return command


def main(argv: Optional[Sequence[str]] = None) -> int:
    run(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
