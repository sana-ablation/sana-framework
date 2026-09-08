"""The planning axis is spelled --plan, and --search-lessguide is fully gone.

The axis picks the management/planning treatment (no planning / planning style
plus skills / an injected gold reasoning chain). It used to be spelled
``--profile``, which collides with the *other* meaning of "profile" in this
repo -- the runtime/ideal profile, the gold per-task data. Nothing here should
touch that concept; these tests only pin the axis.
"""

import dataclasses

import pytest

from sana_evaluation import cli
from sana_evaluation.config import RunConfig
from sana_evaluation.runner import modes


# ---------------------------------------------------------------------------
# The flag
# ---------------------------------------------------------------------------

def test_plan_and_plans_both_parse_to_plan():
    assert cli.parse(["--plan", "ideal"]).plan == "ideal"
    assert cli.parse(["--plans", "ideal"]).plan == "ideal"


def test_plan_defaults_to_standard():
    assert cli.parse([]).plan == "standard"


@pytest.mark.parametrize("dead", [
    ["--profile", "ideal"],       # took a value; "ideal" was one of its legal choices
    ["--profile", "standard"],    # ... and so was its own default
])
def test_profile_flag_is_gone(dead):
    """The argv is otherwise valid, so only the missing option can raise.

    Same trap as test_cli.py::test_deleted_flags_are_gone: the obvious spelling
    ``parse(["--profile", "x"])`` exits whether or not the flag exists, because
    an unknown value lands on the ``preset`` positional, which rejects it. So the
    preset is supplied up front and the flag is given a value it used to accept.
    A parser that still defined ``--profile`` would parse both of these cleanly.
    """
    with pytest.raises(SystemExit):
        cli.parse(["smoke"] + dead)


# ---------------------------------------------------------------------------
# The output label
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("plan", ["naive", "standard", "ideal"])
def test_variant_condition_label_says_plan_not_profile(plan):
    label = cli._variant_condition_label(
        search_tool="ideal",
        search_results="rich",
        plan=plan,
        computation_tool="standard",
    )
    assert f"plan_{plan}" in label
    assert "profile" not in label


# ---------------------------------------------------------------------------
# The internals
# ---------------------------------------------------------------------------

def test_build_plan_replaced_build_management():
    assert callable(modes.build_plan)
    assert not hasattr(modes, "build_management")


def test_run_config_carries_plan_fields_not_profile_fields():
    names = {f.name for f in dataclasses.fields(RunConfig)}
    assert "plan_mode" in names
    assert "plan_skills_enabled" in names
    assert "profile_mode" not in names
    assert "profile_skills_enabled" not in names


def test_mode_bundle_reports_the_plan_axis():
    bundle = modes.build_mode_bundle(
        RunConfig(search_tool_mode="naive", search_results_mode="minimal", plan_mode="naive"),
        data_tools=[],
        task_context=None,
    )
    assert bundle.modes["plan"] == "naive"
    assert bundle.modes["plan_skills"] == "off"
    assert "profile" not in bundle.modes
    assert "profile_skills" not in bundle.modes


# ---------------------------------------------------------------------------
# --search-lessguide, deleted rather than merely disconnected
# ---------------------------------------------------------------------------

def test_run_config_has_no_search_lessguide_field():
    assert "search_lessguide" not in {f.name for f in dataclasses.fields(RunConfig)}


def test_search_ideal_has_no_lessguide_plumbing():
    import sana_evaluation.tools.oracle.search as search_ideal

    assert not hasattr(search_ideal, "set_lessguide")
    assert not hasattr(search_ideal, "_apply_lessguide")
    assert not hasattr(search_ideal, "_LESSGUIDE")


@pytest.mark.parametrize("dead", [
    ["--search-lessguide"],
    ["--search_lessguide"],
])
def test_search_lessguide_flag_is_gone(dead):
    """Store_true flags, so a parser that still defined them would accept these."""
    with pytest.raises(SystemExit):
        cli.parse(["smoke"] + dead)


# ---------------------------------------------------------------------------
# Task scope reporting (Task 12 review carry-over)
# ---------------------------------------------------------------------------

def test_task_scope_announces_only_new_on_the_task_dir_path():
    """--only-new is honoured on both paths, so it must be announced on both.

    Under ``full --task-dir X`` the operator otherwise sees no sign that already
    recorded tasks will be skipped.
    """
    args = cli.parse(["--task-dir", "k-5-d-4", "--only-new"])
    assert "(only new)" in cli._task_scope(args)

    args = cli.parse(["--task-dir", "k-5-d-4"])
    assert "(only new)" not in cli._task_scope(args)

    args = cli.parse(["--all-tasks", "--only-new"])
    assert "(only new)" in cli._task_scope(args)
