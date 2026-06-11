from pathlib import Path

from sana_evaluation.artifacts import benchmark_artifacts


ROOT = Path(__file__).resolve().parents[1]


def test_runtime_and_analysis_packages_have_canonical_imports():
    import sana_analysis.run_mode_analysis_semantic as semantic_analysis
    import sana_evaluation.setup_run as setup_run

    assert callable(setup_run.run)
    assert callable(semantic_analysis.run_analysis)


def test_legacy_package_shims_are_removed():
    assert not (ROOT / "analysis").exists()
    assert not (ROOT / "strands_evaluation").exists()


def test_maintained_benchmarks_use_tasks_profiles_artifacts_layout():
    lakeqa = benchmark_artifacts("lakeqa")
    kramabench = benchmark_artifacts("kramabench")

    assert lakeqa.task_root == Path("benchmarks/lakeqa/tasks-mini/tasks")
    assert lakeqa.runtime_profile_root == Path(
        "benchmarks/lakeqa/tasks-mini/runtime-profiles"
    )
    assert lakeqa.artifact_root == Path("benchmarks/lakeqa/tasks-mini/artifacts")

    assert kramabench.task_root == Path("benchmarks/kramabench/tasks-mini/tasks")
    assert kramabench.runtime_profile_root == Path(
        "benchmarks/kramabench/tasks-mini/runtime-profiles"
    )
    assert kramabench.artifact_root == Path("benchmarks/kramabench/tasks-mini/artifacts")
