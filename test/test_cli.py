import pytest

from sana_evaluation import cli


def test_no_preset_reproduces_the_axis_defaults():
    a = cli.parse([])
    assert (a.search, a.results, a.profile, a.compute) == (
        "standard", "rich", "standard", "standard"
    )
    assert a.verbose is False
    assert a.task_continue is False
    assert a.parallel == 6
    assert a.timeout == 600


def test_preset_only_shifts_defaults():
    a = cli.parse(["smoke"])
    assert a.profile == "ideal"
    assert a.verbose is True


def test_explicit_flag_beats_the_preset():
    a = cli.parse(["full", "--search", "web"])
    assert a.search == "web"
    assert a.task_continue is True      # still from the preset


def test_no_continue_overrides_the_full_preset():
    assert cli.parse(["full", "--no-continue"]).task_continue is False


@pytest.mark.parametrize("argv,attr,expected", [
    (["--search_tool", "ideal"], "search", "ideal"),
    (["--search_results", "minimal"], "results", "minimal"),
    (["--computation_tool", "ideal"], "compute", "ideal"),
    (["--no_s3"], "no_s3", True),
    (["--db", "x.lance"], "db_path", "x.lance"),
    (["--model", "openai/gpt-5.2"], "model_name", "openai/gpt-5.2"),
])
def test_legacy_spellings_still_parse(argv, attr, expected):
    assert getattr(cli.parse(argv), attr) == expected


@pytest.mark.parametrize("dead", [
    ["--condition", "baseline"],        # took a value; "baseline" was its one legal choice
    ["--sparse-backend", "bm25"],       # took a value; "bm25" was its default
    ["--decision-notes"],               # store_true alias for --debug-mode decision_notes
    ["--search-lessguide"],             # store_true
    ["--search_lessguide"],             # ... and its underscore spelling
    ["--only-new"],                     # store_true duplicate of --task-continue
])
def test_deleted_flags_are_gone(dead):
    """Each argv is otherwise valid, so only the missing option can raise.

    The obvious spelling of this test asserts nothing: `parse(["--condition", "x"])`
    exits whether or not the flag exists — if it exists, "x" is an illegal choice,
    and if it does not, "x" lands on the `preset` positional, which rejects it. So
    the preset is supplied up front and every flag that took a value is given a
    legal one; a parser that still defined these would parse all six cleanly.
    """
    with pytest.raises(SystemExit):
        cli.parse(["smoke"] + dead)
