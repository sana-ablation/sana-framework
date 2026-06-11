from pathlib import Path

import pytest

from sana_evaluation.artifacts import (
    ArtifactValidationError,
    benchmark_artifacts,
    format_artifact_report,
    validate_benchmark_artifacts,
)


def test_benchmark_artifacts_exposes_maintained_examples():
    lakeqa = benchmark_artifacts("lakeqa")
    kramabench = benchmark_artifacts("kramabench")

    assert lakeqa.name == "lakeqa"
    assert lakeqa.task_root == Path("benchmarks/lakeqa/tasks-mini/tasks")
    assert lakeqa.runtime_profile_root == Path(
        "benchmarks/lakeqa/tasks-mini/runtime-profiles"
    )
    assert lakeqa.plan_root == lakeqa.runtime_profile_root
    assert lakeqa.artifact_root == Path("benchmarks/lakeqa/tasks-mini/artifacts")
    assert lakeqa.results_root == Path("results")
    assert lakeqa.semantic_results_root == Path("results_semantic")
    assert lakeqa.default_db_hint == Path("lance_data")

    assert kramabench.name == "kramabench"
    assert kramabench.task_root == Path("benchmarks/kramabench/tasks-mini/tasks")
    assert kramabench.runtime_profile_root == Path(
        "benchmarks/kramabench/tasks-mini/runtime-profiles"
    )
    assert kramabench.plan_root == kramabench.runtime_profile_root
    assert kramabench.artifact_root == Path("benchmarks/kramabench/tasks-mini/artifacts")
    assert kramabench.results_root == Path("results-kramabench")
    assert kramabench.semantic_results_root == Path("results-kramabench_semantic")


def test_validate_benchmark_artifacts_reports_missing_required_roots(tmp_path):
    (tmp_path / "benchmarks/lakeqa/tasks-mini/tasks").mkdir(parents=True)

    with pytest.raises(ArtifactValidationError) as excinfo:
        validate_benchmark_artifacts("lakeqa", root=tmp_path)

    message = str(excinfo.value)
    assert "runtime-profiles" in message
    assert "prompts" in message


def test_format_artifact_report_includes_run_and_analysis_commands(tmp_path):
    for path in (
        "benchmarks/kramabench/tasks-mini/tasks",
        "benchmarks/kramabench/tasks-mini/runtime-profiles",
        "benchmarks/kramabench/tasks-mini/artifacts",
        "benchmarks/kramabench/tasks-mini/artifacts/descriptions.jsonl",
        "benchmarks/kramabench/tasks-mini/artifacts/snippets.jsonl",
        "benchmarks/kramabench/tasks-mini/artifacts/table_schemas_full.jsonl",
        "sana_evaluation/prompts",
        "results-kramabench",
        "results-kramabench_semantic",
    ):
        target = tmp_path / path
        if target.suffix:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("")
        else:
            target.mkdir(parents=True, exist_ok=True)

    report = format_artifact_report("kramabench", root=tmp_path)

    assert "Kramabench" in report
    assert "python -m sana_evaluation.setup_run smoke --benchmark kramabench" in report
    assert "python -m sana_analysis.run_mode_analysis_semantic" in report
    assert "benchmarks/kramabench/tasks-mini/tasks" in report
