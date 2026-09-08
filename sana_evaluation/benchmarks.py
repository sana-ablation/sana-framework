"""Maintained benchmark artifact registry for framework users.

This module documents the repository-owned LakeQA and Kramabench artifact roots
that can be run and analyzed without converting a new benchmark.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from sana_evaluation.tools.lake import BENCHMARK_BUCKETS


# ---------------------------------------------------------------------------
# Benchmark-aware paths and URI helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IdealArtifactPaths:
    descriptions: Path
    snippets: Path
    schemas: Path
    profiles: Path


_BENCHMARK_ROOTS = {
    "lakeqa": Path("benchmarks/lakeqa/tasks-mini"),
    "kramabench": Path("benchmarks/kramabench/tasks-mini"),
}


_ARTIFACTS = {
    "lakeqa": IdealArtifactPaths(
        descriptions=_BENCHMARK_ROOTS["lakeqa"] / "artifacts/descriptions.jsonl",
        snippets=_BENCHMARK_ROOTS["lakeqa"] / "artifacts/snippets.jsonl",
        schemas=_BENCHMARK_ROOTS["lakeqa"] / "artifacts/table_schemas_full.jsonl",
        profiles=_BENCHMARK_ROOTS["lakeqa"] / "artifacts/table_profiles.jsonl",
    ),
    "kramabench": IdealArtifactPaths(
        descriptions=_BENCHMARK_ROOTS["kramabench"] / "artifacts/descriptions.jsonl",
        snippets=_BENCHMARK_ROOTS["kramabench"] / "artifacts/snippets.jsonl",
        schemas=_BENCHMARK_ROOTS["kramabench"] / "artifacts/table_schemas_full.jsonl",
        profiles=_BENCHMARK_ROOTS["kramabench"] / "artifacts/table_profiles.jsonl",
    ),
}


def normalize_benchmark(benchmark: str | None = None) -> str:
    value = (benchmark or os.getenv("LAKEQA_BENCHMARK") or "lakeqa").strip().lower()
    if value not in BENCHMARK_BUCKETS:
        expected = ", ".join(sorted(BENCHMARK_BUCKETS))
        raise ValueError(f"Unsupported benchmark '{benchmark}'. Expected one of: {expected}")
    return value


def benchmark_bucket(benchmark: str | None = None) -> str:
    normalized = normalize_benchmark(benchmark)
    if benchmark is None:
        return os.getenv("LAKEQA_BUCKET", BENCHMARK_BUCKETS[normalized])
    return BENCHMARK_BUCKETS[normalized]


def artifact_paths(benchmark: str | None = None) -> IdealArtifactPaths:
    return _ARTIFACTS[normalize_benchmark(benchmark)]


def benchmark_tasks_root(benchmark: str | None = None, task_set: str = "tasks-mini") -> Path:
    normalized = normalize_benchmark(benchmark)
    if task_set != "tasks-mini":
        return Path("benchmarks") / normalized / task_set / "tasks"
    return _BENCHMARK_ROOTS[normalized] / "tasks"


def benchmark_runtime_profiles_root(
    benchmark: str | None = None,
    task_set: str = "tasks-mini",
) -> Path:
    normalized = normalize_benchmark(benchmark)
    if task_set != "tasks-mini":
        return Path("benchmarks") / normalized / task_set / "runtime-profiles"
    return _BENCHMARK_ROOTS[normalized] / "runtime-profiles"


def source_key(source: str) -> str:
    value = str(source or "").strip()
    if value.startswith("s3://"):
        remainder = value[len("s3://") :]
        _bucket, _sep, key = remainder.partition("/")
        return key.lstrip("/")
    return value.lstrip("/")


def canonical_source_uri(source: str, benchmark: str | None = None) -> str:
    value = str(source or "").strip()
    if value.startswith("s3://"):
        return value
    return f"s3://{benchmark_bucket(benchmark)}/{value.lstrip('/')}"


# ---------------------------------------------------------------------------
# Maintained benchmark artifact registry
# ---------------------------------------------------------------------------


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
        f"python -m sana_evaluation.cli smoke --benchmark {artifacts.name} "
        f"--k 5 --db {artifacts.default_db_hint} --model openai/gpt-5.4-nano"
    )
    full_command = (
        f"python -m sana_evaluation.cli full --benchmark {artifacts.name} "
        f"--k 5 --db {artifacts.default_db_hint} --model openai/gpt-5.4-nano"
    )
    analysis_command = (
        "python -m sana_analysis.run_mode_analysis "
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
