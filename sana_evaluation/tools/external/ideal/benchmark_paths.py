"""Benchmark-aware paths and URI helpers for ideal-mode artifacts."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from sana_evaluation.tools.agent_tools import BENCHMARK_BUCKETS


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
