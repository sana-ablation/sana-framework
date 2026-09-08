import re
from pathlib import Path

from sana_evaluation.benchmarks import benchmark_artifacts


ROOT = Path(__file__).resolve().parents[1]

# Matches `from sana_evaluation...`, `import sana_evaluation...` and
# `importlib.import_module("sana_evaluation...")` (any quote style, with or
# without a string prefix like f/r/b). A bare substring search for
# "from sana_evaluation" misses the other two forms entirely.
_SANA_EVALUATION_IMPORT_RE = re.compile(
    r"(?:^|\n)\s*(?:from\s+sana_evaluation\b|import\s+sana_evaluation\b)"
    r"|import_module\(\s*[a-zA-Z]{0,2}['\"]sana_evaluation\b"
)


def test_runtime_and_analysis_packages_have_canonical_imports():
    import sana_analysis.run_mode_analysis as semantic_analysis
    import sana_evaluation.cli as cli

    assert callable(cli.main)
    assert callable(semantic_analysis.run_analysis)


def test_legacy_package_shims_are_removed():
    assert not (ROOT / "analysis").exists()
    assert not (ROOT / "strands_evaluation").exists()


def test_sana_evaluation_legacy_paths_are_removed():
    import importlib

    for path in (
        "sana_evaluation/agent_with_mode.py",
        "sana_evaluation/run_eval.py",
        "sana_evaluation/run_mode_eval.py",
        "sana_evaluation/setup_run.py",
        "sana_evaluation/artifacts.py",
        "sana_evaluation/helper",
        "sana_evaluation/prompts",
        "sana_evaluation/tools/agent_tools.py",
        "sana_evaluation/tools/agent_tools_v2.py",
        "sana_evaluation/tools/external",
        "sana_evaluation/tools/helper",
        "sana_evaluation/tools/skills",
    ):
        assert not (ROOT / path).exists(), f"{path} should have been removed"

    for dead in (
        "sana_evaluation.agent_with_mode",
        "sana_evaluation.run_eval",
        "sana_evaluation.run_mode_eval",
        "sana_evaluation.setup_run",
        "sana_evaluation.artifacts",
        "sana_evaluation.helper",
        "sana_evaluation.tools.agent_tools",
        "sana_evaluation.tools.agent_tools_v2",
        "sana_evaluation.tools.external",
    ):
        try:
            importlib.import_module(dead)
        except ModuleNotFoundError:
            continue
        raise AssertionError(f"{dead} still importable")


def test_dataindexing_does_not_import_sana_evaluation():
    """The packages were mutually dependent; the substrate move ended that.

    Catches ``from sana_evaluation import ...``, ``import sana_evaluation``
    and ``importlib.import_module("sana_evaluation...")`` alike -- a plain
    "from sana_evaluation" substring check misses the last two forms.
    """
    offenders = []
    for path in (ROOT / "dataindexing").rglob("*.py"):
        text = path.read_text(errors="ignore")
        if _SANA_EVALUATION_IMPORT_RE.search(text):
            offenders.append(str(path.relative_to(ROOT)))
    assert not offenders, f"dataindexing must not import sana_evaluation: {offenders}"


def test_public_api_survived_the_move():
    import sana_evaluation

    for name in (
        "DataLakeAgent",
        "BatchRunner",
        "AgentResult",
        "AgentConfig",
        "RunConfig",
        "build_model",
        "compute_exact_match",
        "compute_f1_score",
        "normalize_text",
    ):
        assert hasattr(sana_evaluation, name), f"public API lost {name}"


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
