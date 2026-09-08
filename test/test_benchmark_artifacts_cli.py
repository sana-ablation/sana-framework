"""The unified artifact pipeline CLI resolves per-benchmark defaults."""
import pytest

from dataindexing.cli import benchmark_artifacts as ba


def test_artifact_root_follows_the_benchmark():
    assert str(ba.artifact_root("lakeqa")) == "benchmarks/lakeqa/tasks-mini/artifacts"
    assert str(ba.artifact_root("kramabench")) == "benchmarks/kramabench/tasks-mini/artifacts"


def test_all_runs_the_pipeline_in_dependency_order():
    assert ba.PIPELINE == ("manifest", "describe", "merge-descriptions", "check")


def test_snippets_is_excluded_from_all():
    # Its input parquet is not derivable from the benchmark name.
    assert "snippets" in ba.STAGES and "snippets" not in ba.PIPELINE


def test_each_stage_has_a_delegate():
    for stage in ba.STAGES:
        entry, defaults = ba._stage_entry(stage)
        assert callable(entry), stage
        assert isinstance(defaults, list), stage


def test_defaults_are_templated_per_benchmark():
    _, defaults = ba._stage_entry("check")
    rendered = [d.format(art=ba.artifact_root("kramabench"), tasks=ba.task_root("kramabench"))
                for d in defaults]
    assert all("kramabench" in r for r in rendered if r.startswith("benchmarks"))
    assert not any("lakeqa" in r for r in rendered)


def test_caller_flags_replace_the_derived_default(monkeypatch):
    # A --manifest passed by the caller must not end up alongside the derived one.
    seen = {}
    monkeypatch.setattr(ba, "_stage_entry",
                        lambda s: (lambda argv: seen.update(argv=argv) or 0,
                                   ["--manifest", "{art}/task_file_manifest.jsonl"]))
    ba.run_stage("manifest", "lakeqa", ["--manifest", "/tmp/mine.jsonl"])
    assert seen["argv"].count("--manifest") == 1
    assert seen["argv"] == ["--manifest", "/tmp/mine.jsonl"]


def test_a_failing_stage_stops_the_pipeline(monkeypatch):
    calls = []
    def fake(stage, benchmark, extra):
        calls.append(stage)
        return 1 if stage == "describe" else 0
    monkeypatch.setattr(ba, "run_stage", fake)
    rc = ba.main(["--benchmark", "lakeqa", "all"])
    assert rc == 1
    # Stopped at describe rather than running merge-descriptions on stale input.
    assert calls == ["manifest", "describe"]
