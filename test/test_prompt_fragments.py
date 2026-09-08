"""The fragment model's three structural guarantees.

Each test names a bug the base-selection-plus-overlay composer could not rule
out: a prompt that advertises a tool the run does not have (6a), a correction
bullet that exists only to undo a stale tool list, and preflight validating a
different file than the run reads (6b).
"""
import pytest

from sana_evaluation.prompting import compose

LAKE_TOOLS = ("list_files", "peek_file", "peek_multiple", "read_file",
              "grep_file", "parse_xml_records", "query_file")


@pytest.mark.parametrize("plan", ["naive", "standard"])
@pytest.mark.parametrize("benchmark", ["lakeqa", "kramabench"])
def test_web_mode_never_advertises_a_lake_tool(benchmark, plan):
    prompt = compose.build(plan=plan, search="web",
                           benchmark=benchmark, skills=False)
    for tool in LAKE_TOOLS:
        assert f"`{tool}`" not in prompt, (
            f"{plan}+{benchmark}+web still lists {tool}, which the run does not have"
        )


def test_web_mode_never_advertises_a_skill_it_cannot_load():
    """The tool-advertisement bug, one axis over.

    skill_paths_for_modes gives web planning + query-data and no discover-data,
    so a `skills("discover-data")` bullet in the web prompt names a capability
    the run cannot serve.
    """
    loaded = compose.skill_paths_for_modes("web", "standard")
    assert not any("discover-data" in path for path in loaded)

    prompt = compose.build(plan="standard", search="web",
                           benchmark="lakeqa", skills=True)
    assert "discover-data" not in prompt
    assert 'skills("query-data")' in prompt, "web does load query-data"


def test_kramabench_never_mentions_query_file():
    prompt = compose.build(plan="standard", search="standard",
                           benchmark="kramabench", skills=False)
    assert "query_file" not in prompt
    assert "no SQL table tool" not in prompt, (
        "the correction is unnecessary once query_file is simply not listed"
    )


def _paths_the_run_really_composes(case):
    """The fragment list the runtime dispatch actually resolved, not a model of it.

    ``runner.modes`` picks one of three wrappers before any resolver is reached,
    and a wrapper may normalise an axis on the way -- kramabench composes the
    managed plan fragment whatever plan mode the caller asked for. Asserting
    ``_prompt_files_for_modes == fragment_paths`` would compare the two sides of
    one delegation and prove only that preflight still delegates. Recording the
    call the wrapper really made is what makes this a check on 6b.
    """
    from test import test_prompt_corpus  # _compose mirrors runner/modes.py's dispatch

    recorded = []
    real = compose.fragment_paths

    def spy(**kwargs):
        paths = real(**kwargs)
        recorded.append(paths)
        return paths

    compose.fragment_paths = spy
    try:
        test_prompt_corpus._compose(*case)
    finally:
        compose.fragment_paths = real

    assert len(recorded) == 1, f"expected one resolution per prompt, got {len(recorded)}"
    return recorded[0]


def test_preflight_and_runtime_resolve_the_same_paths():
    from sana_evaluation.preflight import _prompt_files_for_modes
    from test.test_prompt_corpus import CASES

    for case in CASES:
        plan, search, benchmark, skills = case
        preflight_paths = _prompt_files_for_modes(
            plan=plan, search=search, benchmark=benchmark, skills=skills
        )
        assert preflight_paths == _paths_the_run_really_composes(case), (
            f"preflight validates different files than {plan}+{search}+{benchmark} reads"
        )
