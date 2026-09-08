"""Presets resolve to a namespace, not to a subprocess command line.

``setup_run`` resolved defaults, printed a command, then ``subprocess.run``'d
``run_mode_eval``, so these assertions used to inspect a built ``argv`` list
through an injected fake runner. That seam is gone: one parser resolves one
namespace and calls the evaluator directly, so the namespace is what to assert.
"""

import os

import pytest

from sana_evaluation import cli


def _write_lakeqa_fixture(root):
    (root / "lance_data").mkdir(parents=True, exist_ok=True)
    for task_dir in ("k-1-d-1", "k-5-d-4"):
        target = root / "benchmarks" / "lakeqa" / "tasks-mini" / "tasks" / task_dir
        target.mkdir(parents=True, exist_ok=True)
        (target / "task_1.json").write_text("{}")
        (target / "task_2.json").write_text("{}")


def _write_kramabench_fixture(root):
    (root / "lance_data").mkdir(parents=True, exist_ok=True)
    target = root / "benchmarks" / "kramabench" / "tasks-mini" / "tasks" / "k-2-d-1-s-1"
    target.mkdir(parents=True, exist_ok=True)
    (target / "task_1.json").write_text("{}")
    (target / "task_2.json").write_text("{}")


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A minimal repo tree, with the CLI resolving paths against it."""
    _write_lakeqa_fixture(tmp_path)
    _write_kramabench_fixture(tmp_path)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def resolved(argv):
    return cli.resolve(cli.parse(argv))


# ---------------------------------------------------------------------------
# smoke
# ---------------------------------------------------------------------------

def test_smoke_resolves_the_kramabench_run(repo):
    a = resolved([
        "smoke",
        "--search", "ideal",
        "--results", "ideal",
        "--profile", "ideal",
        "--k", "5",
        "--model", "gpt5.2",
        "--reasoning-effort", "xhigh",
        "--openai-prompt-cache-key", "assistant-v3:tools-v1",
        "--openai-prompt-cache-retention", "24h",
        "--benchmark", "kramabench",
        "--verbose",
        "--db", "lance_data",
    ])

    assert a.search == "ideal"
    assert a.model_name == "openai/gpt-5.2"
    assert a.db_path == "lance_data"
    assert a.task_dir == "benchmarks/kramabench/tasks-mini/tasks/k-2-d-1-s-1"
    assert a.tasks_per_dir == 2
    assert a.logs_output_dir == "log-kramabench"
    assert a.results_output_dir == "results-kramabench"
    assert a.reasoning_effort == "xhigh"
    assert a.openai_prompt_cache_key == "assistant-v3:tools-v1"
    assert a.openai_prompt_cache_retention == "24h"
    assert a.benchmark == "kramabench"
    assert a.verbose is True
    assert cli._task_scope(a) == (
        "benchmarks/kramabench/tasks-mini/tasks/k-2-d-1-s-1 (first 2 tasks)"
    )


def test_kramabench_smoke_defaults_to_the_kramabench_task_dir(repo):
    a = resolved([
        "smoke",
        "--benchmark", "kramabench",
        "--search", "ideal",
        "--results", "naive",
        "--profile", "standard",
        "--model", "gpt-5.4-nano",
        "--db", "lance_data",
    ])

    assert a.task_dir == "benchmarks/kramabench/tasks-mini/tasks/k-2-d-1-s-1"
    assert cli._task_scope(a) == (
        "benchmarks/kramabench/tasks-mini/tasks/k-2-d-1-s-1 (first 2 tasks)"
    )


def test_smoke_defaults_to_ideal_axes_and_verbose(repo):
    a = resolved([
        "smoke",
        "--benchmark", "kramabench",
        "--model", "openai/gpt-5-mini",
        "--db", "lance_data",
    ])

    assert (a.search, a.results, a.profile, a.compute) == (
        "ideal", "rich", "ideal", "ideal"
    )
    assert a.verbose is True


def test_preloaded_search_mode_resolves(repo):
    a = resolved([
        "smoke",
        "--search", "preloaded",
        "--results", "naive",
        "--profile", "ideal",
        "--model", "bedrock/claude-haiku-4.5",
        "--db", "lance_data",
    ])

    assert a.search == "preloaded"
    assert a.results == "minimal"       # naive is the former name for minimal
    assert a.profile == "ideal"


def test_smoke_ideal_compute_axis_resolves(repo):
    a = resolved([
        "smoke",
        "--search", "preloaded",
        "--results", "ideal",
        "--profile", "standard",
        "--compute", "ideal",
        "--db", "lance_data",
    ])

    assert a.compute == "ideal"


# ---------------------------------------------------------------------------
# full
# ---------------------------------------------------------------------------

def test_full_no_continue_uses_the_default_task_set_and_output_roots(repo):
    a = resolved([
        "full",
        "--no-continue",
        "--search", "ideal",
        "--results", "ideal",
        "--profile", "ideal",
        "--k", "5",
        "--model", "openai/gpt-5.2",
        "--db", "lance_data",
    ])

    assert a.task_set == "benchmarks/lakeqa/tasks-mini/tasks"
    assert a.logs_output_dir == "logs"
    assert a.results_output_dir == "results"


def test_full_no_continue_selects_all_tasks(repo):
    a = resolved([
        "full",
        "--no-continue",
        "--model", "openai/gpt-5-mini",
        "--db", "lance_data",
    ])

    assert a.all_tasks is True
    assert a.task_continue is False
    assert cli._task_scope(a) == "all tasks under benchmarks/lakeqa/tasks-mini/tasks"


def test_kramabench_full_defaults_to_kramabench_output_roots(repo):
    a = resolved([
        "full",
        "--search", "standard",
        "--results", "naive",
        "--profile", "standard",
        "--benchmark", "kramabench",
        "--model", "openai/gpt-5.2",
        "--db", "lance_kramabench_base",
    ])

    assert a.db_path == "lance_kramabench_base"
    assert a.logs_output_dir == "log-kramabench"
    assert a.results_output_dir == "results-kramabench"


def test_kramabench_full_defaults_to_the_kramabench_task_set(repo):
    a = resolved([
        "full",
        "--no-continue",
        "--benchmark", "kramabench",
        "--search", "ideal",
        "--results", "naive",
        "--profile", "standard",
        "--model", "gpt-5.4-nano",
        "--db", "lance_data",
    ])

    assert a.task_set == "benchmarks/kramabench/tasks-mini/tasks"
    assert cli._task_scope(a) == "all tasks under benchmarks/kramabench/tasks-mini/tasks"


def test_continue_alias_and_timeout_passthrough(repo):
    a = resolved([
        "full",
        "--continue",
        "--search", "ideal",
        "--results", "ideal",
        "--profile", "ideal",
        "--model", "openai/gpt-5.2",
        "--timeout", "600",
        "--submit-grace-seconds", "15",
        "--db", "lance_data",
    ])

    assert a.task_continue is True
    assert a.all_tasks is False
    assert a.timeout == 600
    assert a.submit_grace_seconds == 15


def test_full_defaults_to_ideal_axes_verbose_and_continue_with_plans_alias(repo):
    a = resolved([
        "full",
        "--benchmark", "kramabench",
        "--search", "ideal",
        "--plans", "standard",
        "--compute", "ideal",
        "--k", "5",
        "--parallel", "4",
        "--model", "openai/gpt-5-mini",
        "--db", "lance_kramabench_infused",
        "--timeout", "600",
        "--submit-grace-seconds", "30",
    ])

    assert a.search == "ideal"
    assert a.results == "rich"          # the preset's default, canonicalised
    assert a.profile == "standard"      # --plans is the alias for --profile
    assert a.compute == "ideal"
    assert a.verbose is True
    assert a.task_continue is True
    assert a.all_tasks is False
    assert a.parallel == 4
    assert a.timeout == 600
    assert a.submit_grace_seconds == 30
    assert cli._task_scope(a) == (
        "resume pending tasks under benchmarks/kramabench/tasks-mini/tasks"
    )


def test_full_preset_compute_defaults_to_ideal(repo):
    a = resolved([
        "full",
        "--search", "ideal",
        "--results", "ideal",
        "--profile", "ideal",
        "--model", "gpt-5.4-nano",
        "--db", "lance_data",
    ])

    assert a.compute == "ideal"


def test_explicit_standard_compute_wins_over_the_preset(repo):
    a = resolved([
        "full",
        "--search", "ideal",
        "--results", "ideal",
        "--profile", "ideal",
        "--compute", "standard",
        "--model", "gpt-5.4-nano",
        "--db", "lance_data",
    ])

    assert a.compute == "standard"


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------

def test_smoke_normalizes_new_openai_model_alias(repo):
    a = resolved(["smoke", "--model", "gpt-5.4-nano", "--db", "lance_data"])
    assert a.model_name == "openai/gpt-5.4-nano"


def test_smoke_normalizes_gpt_5_nano_alias(repo):
    a = resolved(["smoke", "--model", "gpt-5-nano", "--db", "lance_data"])
    assert a.model_name == "openai/gpt-5-nano"


def test_selector_and_repair_models_are_normalized(repo):
    a = resolved([
        "smoke",
        "--benchmark", "kramabench",
        "--model", "gemini/gemini-3.1-flash-lite",
        "--selector-model", "openai/gpt-5-mini",
        "--repair-model", "gpt-5.4-nano",
        "--db", "lance_data",
    ])

    assert a.selector_model == "openai/gpt-5-mini"
    assert a.repair_model == "openai/gpt-5.4-nano"


def test_the_per_subagent_model_flags_do_not_exist(repo):
    # --selector-model and --repair-model are the only two knobs; the per-role
    # environment variables behind them are not separately settable.
    for dead in ("--ideal-subagent-model", "--search-ideal-subagent-model",
                 "--semantic-ideal-subagent-model", "--repair-ideal-subagent-model"):
        with pytest.raises(SystemExit):
            cli.parse(["smoke", dead, "openai/gpt-5-mini"])


# ---------------------------------------------------------------------------
# search and skills flags
# ---------------------------------------------------------------------------

def test_search_free_alias_resolves(repo):
    a = resolved([
        "smoke",
        "--search", "ideal",
        "--results", "ideal",
        "--profile", "standard",
        "--search_free",
        "--db", "lance_data",
    ])

    assert a.search_free is True


def test_search_lessguide_is_gone(repo):
    for dead in ("--search-lessguide", "--search_lessguide"):
        with pytest.raises(SystemExit):
            cli.parse(["smoke", dead])


def test_skills_flag_resolves(repo):
    common = ["smoke", "--search", "ideal", "--results", "ideal",
              "--profile", "standard", "--db", "lance_data"]

    assert resolved(common + ["--skills", "on"]).skills == "on"
    assert resolved(common + ["--skills", "off"]).skills == "off"
    assert resolved(common).skills == "off"


def test_skills_on_with_naive_profile_is_rejected(repo, capsys):
    argv = ["smoke", "--search", "ideal", "--results", "ideal",
            "--profile", "naive", "--skills", "on", "--db", "lance_data"]

    with pytest.raises(ValueError, match="--skills on requires --profile standard or --profile ideal"):
        resolved(argv)

    with pytest.raises(SystemExit) as excinfo:
        cli.main(argv)
    assert excinfo.value.code == 2
    assert "--skills on requires --profile standard or --profile ideal" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

def test_main_loads_the_repo_dotenv_before_the_run(repo, monkeypatch):
    (repo / ".env").write_text("GEMINI_API_KEY=temp-gemini-key\n")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    seen = {}

    def stop_at_preflight(run_config, task_files):
        seen["gemini_api_key"] = os.environ.get("GEMINI_API_KEY")
        raise cli.PreflightError("stop")

    monkeypatch.setattr(cli, "run_preflight", stop_at_preflight)

    with pytest.raises(SystemExit):
        cli.main(["smoke", "--model", "gemini/gemini-3.1-flash-lite", "--db", "lance_data"])

    assert seen["gemini_api_key"] == "temp-gemini-key"
