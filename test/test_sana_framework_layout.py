from pathlib import Path

from sana_evaluation.benchmarks import benchmark_artifacts


ROOT = Path(__file__).resolve().parents[1]


def test_runtime_and_analysis_packages_have_canonical_imports():
    import sana_analysis.run_mode_analysis as semantic_analysis
    import sana_evaluation.cli as cli

    assert callable(cli.main)
    assert callable(semantic_analysis.run_analysis)


def test_legacy_package_shims_are_removed():
    assert not (ROOT / "analysis").exists()
    assert not (ROOT / "strands_evaluation").exists()


def test_the_merged_cli_leaves_no_shim_behind():
    import importlib

    assert not (ROOT / "sana_evaluation" / "run_mode_eval.py").exists()
    assert not (ROOT / "sana_evaluation" / "setup_run.py").exists()

    for dead in ("sana_evaluation.run_mode_eval", "sana_evaluation.setup_run"):
        try:
            importlib.import_module(dead)
        except ModuleNotFoundError:
            continue
        raise AssertionError(f"{dead} still importable")


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


def test_renamed_analysis_layout_has_no_survivors():
    import importlib

    assert not (ROOT / "sana_analysis" / "running_analysis").exists()
    assert not (ROOT / "sana_analysis" / "report_generator").exists()
    assert not (ROOT / "sana_analysis" / "run_sana_mode_analysis.py").exists()
    assert not (ROOT / "sana_analysis" / "run_mode_analysis_semantic.py").exists()

    for dead in (
        "sana_analysis.running_analysis",
        "sana_analysis.report_generator",
        "sana_analysis.run_sana_mode_analysis",
        "sana_analysis.run_mode_analysis_semantic",
    ):
        try:
            importlib.import_module(dead)
        except ModuleNotFoundError:
            continue
        raise AssertionError(f"{dead} still importable")
