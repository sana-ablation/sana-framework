"""Maintained benchmark artifact registry for framework users.

This module documents the repository-owned LakeQA and Kramabench artifact roots
that can be run and analyzed without converting a new benchmark.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from sana_evaluation.tools.external.ideal.benchmark_paths import (
    artifact_paths,
    benchmark_runtime_profiles_root,
    benchmark_tasks_root,
    normalize_benchmark,
)


@dataclass(frozen=True)
class BenchmarkArtifacts:
    name: str
    display_name: str
    task_root: Path
    runtime_profile_root: Path
    artifact_root: Path
    results_root: Path
    semantic_results_root: Path
    analysis_output_root: Path
    log_root: Path
    default_smoke_task_dir: Path
    default_db_hint: Path
    docs: Path

    @property
    def plan_root(self) -> Path:
        """Compatibility alias for older callers."""
        return self.runtime_profile_root

    @property
    def required_roots(self) -> tuple[Path, ...]:
        paths = artifact_paths(self.name)
        return (
            self.task_root,
            self.runtime_profile_root,
            self.artifact_root,
            Path("sana_evaluation/prompts"),
            paths.descriptions,
            paths.snippets,
            paths.schemas,
        )

    @property
    def optional_roots(self) -> tuple[Path, ...]:
        paths = artifact_paths(self.name)
        return (
            self.default_db_hint,
            self.results_root,
            self.semantic_results_root,
            self.analysis_output_root,
            paths.profiles,
        )


_ARTIFACT_REGISTRY = {
    "lakeqa": BenchmarkArtifacts(
        name="lakeqa",
        display_name="LakeQA",
        task_root=benchmark_tasks_root("lakeqa"),
        runtime_profile_root=benchmark_runtime_profiles_root("lakeqa"),
        artifact_root=Path("benchmarks/lakeqa/tasks-mini/artifacts"),
        results_root=Path("results"),
        semantic_results_root=Path("results_semantic"),
        analysis_output_root=Path("analysis_results_mode_semantic"),
        log_root=Path("logs"),
        default_smoke_task_dir=benchmark_tasks_root("lakeqa") / "k-5-d-4",
        default_db_hint=Path("lance_data"),
        docs=Path("benchmarks/README.md"),
    ),
    "kramabench": BenchmarkArtifacts(
        name="kramabench",
        display_name="Kramabench",
        task_root=benchmark_tasks_root("kramabench"),
        runtime_profile_root=benchmark_runtime_profiles_root("kramabench"),
        artifact_root=Path("benchmarks/kramabench/tasks-mini/artifacts"),
        results_root=Path("results-kramabench"),
        semantic_results_root=Path("results-kramabench_semantic"),
        analysis_output_root=Path("analysis_results_mode_kramabench_semantic"),
        log_root=Path("log-kramabench"),
        default_smoke_task_dir=benchmark_tasks_root("kramabench") / "k-2-d-1-s-1",
        default_db_hint=Path("lance_kramabench_infused"),
        docs=Path("benchmarks/README.md"),
    ),
}


class ArtifactValidationError(RuntimeError):
    """Raised when a maintained benchmark artifact tree is incomplete."""


def benchmark_artifacts(benchmark: str | None = None) -> BenchmarkArtifacts:
    """Return the maintained artifact roots for ``benchmark``."""
    normalized = normalize_benchmark(benchmark)
    return _ARTIFACT_REGISTRY[normalized]


def _missing_paths(root: Path, paths: Iterable[Path]) -> list[Path]:
    return [path for path in paths if not (root / path).exists()]


def validate_benchmark_artifacts(
    benchmark: str | None = None,
    *,
    root: str | Path = ".",
) -> BenchmarkArtifacts:
    """Validate required maintained artifacts and return the benchmark registry row."""
    repo_root = Path(root)
    artifacts = benchmark_artifacts(benchmark)
    missing = _missing_paths(repo_root, artifacts.required_roots)
    if missing:
        formatted = ", ".join(str(path) for path in missing)
        raise ArtifactValidationError(
            f"{artifacts.display_name} maintained artifacts are incomplete; missing: {formatted}"
        )
    return artifacts


def _status_lines(root: Path, paths: Sequence[Path]) -> list[str]:
    lines: list[str] = []
    for path in paths:
        status = "ok" if (root / path).exists() else "missing"
        lines.append(f"- {path}: {status}")
    return lines


def format_artifact_report(
    benchmark: str | None = None,
    *,
    root: str | Path = ".",
) -> str:
    """Build a concise report with paths and commands for a maintained benchmark."""
    repo_root = Path(root)
    artifacts = benchmark_artifacts(benchmark)
    required_missing = _missing_paths(repo_root, artifacts.required_roots)
    status = "ready" if not required_missing else "incomplete"

    smoke_command = (
        f"python -m sana_evaluation.setup_run smoke --benchmark {artifacts.name} "
        f"--k 5 --db {artifacts.default_db_hint} --model openai/gpt-5.4-nano"
    )
    full_command = (
        f"python -m sana_evaluation.setup_run full --benchmark {artifacts.name} "
        f"--k 5 --db {artifacts.default_db_hint} --model openai/gpt-5.4-nano"
    )
    analysis_command = (
        "python -m sana_analysis.run_mode_analysis_semantic "
        f"--results-dir {artifacts.semantic_results_root / 'modes'} "
        f"--base-results-dir {artifacts.results_root / 'modes'} "
        f"--traces-dir {artifacts.results_root / 'traces' / 'modes'} "
        f"--tasks-dir {artifacts.task_root} "
        f"--output-dir {artifacts.analysis_output_root}"
    )

    lines = [
        f"# {artifacts.display_name} Artifacts",
        "",
        f"Status: {status}",
        "",
        "Required roots:",
        *_status_lines(repo_root, artifacts.required_roots),
        "",
        "Optional/generated roots:",
        *_status_lines(repo_root, artifacts.optional_roots),
        "",
        "Smoke run:",
        f"  {smoke_command}",
        "",
        "Full/resume run:",
        f"  {full_command}",
        "",
        "Evaluate existing results:",
        f"  {analysis_command}",
        "",
        f"Docs: {artifacts.docs}",
    ]
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect maintained LakeQA/Kramabench artifacts.",
    )
    parser.add_argument(
        "--benchmark",
        choices=sorted(_ARTIFACT_REGISTRY),
        default="lakeqa",
        help="Maintained benchmark to inspect.",
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Repository root to inspect.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if required artifacts are missing.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    print(format_artifact_report(args.benchmark, root=args.root))
    if args.check:
        try:
            validate_benchmark_artifacts(args.benchmark, root=args.root)
        except ArtifactValidationError as exc:
            print(f"\n{exc}", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
