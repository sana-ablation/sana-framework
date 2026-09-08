#!/usr/bin/env python3
"""One entry point for the offline benchmark-artifact pipeline.

The five stages below each consume the previous one's output, and each used to
be its own script that hardcoded `benchmarks/lakeqa/tasks-mini/artifacts` as a
default. Running the pipeline for another benchmark therefore meant editing five
files. Here the artifact root is derived once from --benchmark and threaded
through, so the same command works for any of them.

    python -m dataindexing.cli.benchmark_artifacts --benchmark lakeqa <stage>

Stages, in dependency order:

    manifest            which files each task touches   -> task_file_manifest.jsonl
    describe            per-file descriptions           -> task_file_manifest_descriptions.jsonl
    merge-descriptions  fold into the table set         -> descriptions.jsonl
    snippets PARQUET    sample rows per table           -> snippets.jsonl
    check               audit coverage of the above     -> coverage_missing_*.jsonl

`all` runs manifest, describe, merge-descriptions and check in order; snippets
is excluded because it needs a parquet path only the caller knows.

Each stage delegates to the module that implements it, so behaviour and tests
are unchanged -- this adds an entry point, it does not reimplement a pipeline.
Arguments after the stage name are passed through, and override the defaults
derived from --benchmark.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Sequence

STAGES = ("manifest", "describe", "merge-descriptions", "snippets", "check")
# Stages `all` runs, in order. snippets is absent by design: its input parquet
# is not derivable from the benchmark name.
PIPELINE = ("manifest", "describe", "merge-descriptions", "check")


def artifact_root(benchmark: str) -> Path:
    return Path("benchmarks") / benchmark / "tasks-mini" / "artifacts"


def task_root(benchmark: str) -> Path:
    return Path("benchmarks") / benchmark / "tasks-mini" / "tasks"


def _stage_entry(stage: str) -> tuple[Callable[..., int], list[str]]:
    """Return the delegate for a stage plus the defaults derived per benchmark."""
    from dataindexing.cli import (
        build_snippet_jsonl,
        build_task_file_manifest,
        build_task_manifest_descriptions,
        check_manifest_coverage,
        merge_table_descriptions,
    )

    return {
        "manifest": (build_task_file_manifest.main, ["--task-root", "{tasks}",
                                                     "--output", "{art}/task_file_manifest.jsonl"]),
        "describe": (build_task_manifest_descriptions.main,
                     ["--manifest", "{art}/task_file_manifest.jsonl"]),
        "merge-descriptions": (merge_table_descriptions.main, []),
        "snippets": (build_snippet_jsonl.main, ["--output", "{art}/snippets.jsonl"]),
        "check": (check_manifest_coverage.main, ["--manifest", "{art}/task_file_manifest.jsonl",
                                                 "--snippets", "{art}/snippets.jsonl",
                                                 "--profiles", "{art}/table_profiles.jsonl"]),
    }[stage]


def run_stage(stage: str, benchmark: str, extra: Sequence[str]) -> int:
    entry, defaults = _stage_entry(stage)
    art, tasks = artifact_root(benchmark), task_root(benchmark)
    argv = [a.format(art=art, tasks=tasks) for a in defaults]

    # Caller-supplied flags win: drop a default whose flag is passed explicitly.
    supplied = {a for a in extra if a.startswith("--")}
    pruned: list[str] = []
    skip = False
    for token in argv:
        if skip:
            skip = False
            continue
        if token in supplied:
            skip = True
            continue
        pruned.append(token)

    full = pruned + list(extra)
    print(f"[{stage}] {' '.join(full) or '(defaults)'}", file=sys.stderr)
    try:
        result = entry(full)
    except TypeError:
        # build_snippet_jsonl.main() takes no argv; fall back to sys.argv.
        saved = sys.argv
        sys.argv = [stage, *full]
        try:
            result = entry()
        finally:
            sys.argv = saved
    return int(result or 0)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", default="lakeqa",
                    help="benchmark whose artifact root the stages default to")
    ap.add_argument("stage", choices=(*STAGES, "all"))
    ap.add_argument("extra", nargs=argparse.REMAINDER,
                    help="arguments passed through to the stage, overriding defaults")
    args = ap.parse_args(argv)

    stages = PIPELINE if args.stage == "all" else (args.stage,)
    for stage in stages:
        rc = run_stage(stage, args.benchmark, args.extra)
        if rc:
            print(f"stage '{stage}' failed (rc={rc}); stopping", file=sys.stderr)
            return rc
    return 0


if __name__ == "__main__":
    sys.exit(main())
