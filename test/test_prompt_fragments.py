"""The fragment model's structural guarantees.

Each test names a bug the base-selection-plus-overlay composer could not rule
out: a prompt that advertises a tool the run does not have (6a), a correction
bullet that exists only to undo a stale tool list, and preflight validating a
different file than the run reads (6b).

The 6a and 6b guards both assert against the runtime, not against a model of
it: 6a reads ``build_mode_bundle``'s prompt, injections included, and 6b's
fragment list is compared with what ``build_plan``'s own dispatch resolves.
"""
from types import SimpleNamespace

import pytest

from sana_evaluation.config import RunConfig
from sana_evaluation.prompting import compose
from sana_evaluation.runner.modes import build_data_tools, build_mode_bundle

LAKE_TOOLS = ("list_files", "peek_file", "peek_multiple", "read_file",
              "grep_file", "parse_xml_records", "query_file")


def _web_bundle(plan, benchmark):
    """The prompt the agent is really handed, not the composer's output.

    ``build_mode_bundle`` appends run-level sections to the composed prompt
    after every fragment has been concatenated, so the fragment model alone
    cannot rule out 6a. Asserting on ``compose.build`` would leave the whole
    injection layer unguarded -- which is exactly where the lake-tool bullets
    survived.
    """
    config = RunConfig(
        search_tool_mode="web",
        search_results_mode="naive",
        plan_mode=plan,
        computation_tool_mode="standard",
        plan_skills_enabled=False,
        benchmark=benchmark,
    )
    return build_mode_bundle(
        config,
        data_tools=build_data_tools(search_tool_mode="web"),
        task_context={},
    )


@pytest.mark.parametrize("plan", ["naive", "standard"])
@pytest.mark.parametrize("benchmark", ["lakeqa", "kramabench"])
def test_web_mode_never_advertises_a_lake_tool(benchmark, plan):
    bundle = _web_bundle(plan, benchmark)
    bound = {tool.tool_spec["name"] for tool in bundle.tools}
    assert bound.isdisjoint(LAKE_TOOLS), (
        "premise of this test: the web arm binds none of the lake tools"
    )

    prompt = bundle.system_prompt
    for tool in LAKE_TOOLS:
        assert f"`{tool}`" not in prompt, (
            f"{plan}+{benchmark}+web still lists {tool}, which the run does not have"
        )


@pytest.mark.parametrize("search", ["web", "preloaded"])
def test_search_mode_never_advertises_a_skill_it_cannot_load(search):
    """The tool-advertisement bug, one axis over.

    skill_paths_for_modes gives web and preloaded planning + query-data and no
    discover-data -- neither does lake discovery -- so a
    `skills("discover-data")` bullet in either prompt names a capability the
    run cannot serve.
    """
    loaded = compose.skill_paths_for_modes(search, "standard")
    assert not any("discover-data" in path for path in loaded)

    prompt = compose.build(plan="standard", search=search,
                           benchmark="lakeqa", skills=True)
    assert "discover-data" not in prompt
    assert 'skills("query-data")' in prompt, f"{search} does load query-data"


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


def _prompt_the_runtime_dispatch_returns(case, monkeypatch):
    """The prompt ``runner.modes.build_plan`` itself returns for one case.

    ``build_plan`` also assembles a task-specific trailer (the gold reasoning
    chain, the preloaded URI block), and for the preloaded and plan=ideal arms
    that trailer needs a runtime profile keyed by ``task_id``. The trailer is
    not part of the prompt -- it is appended per task, after the cacheable
    prefix -- so the profile lookup is stubbed and only the prompt, which is a
    function of the axes alone, is returned.
    """
    from sana_evaluation.runner import modes

    profile = SimpleNamespace(reasoning_chain_text="", source_sequence=["s3://stub/one.csv"])
    monkeypatch.setattr(modes, "load_ideal_profile_for_context", lambda ctx: profile)
    monkeypatch.setattr(modes, "set_ideal_profile_task_context", lambda ctx: None)

    plan, search, benchmark, skills = case
    prompt, *_ = modes.build_plan(
        plan,
        search_tool_mode=search,
        task_context=None,
        plan_skills_enabled=skills,
        benchmark=benchmark,
    )
    return prompt


def test_the_corpus_mirror_still_mirrors_the_runtime_dispatch(monkeypatch):
    """``test_prompt_corpus._compose`` is a hand-written copy of ``build_plan``'s
    wrapper dispatch, and the golden corpus plus ``_paths_the_run_really_composes``
    both read the runtime through it. Without this equality a change to
    ``build_plan``'s dispatch would move no golden and fail no test: the mirror
    would simply stop describing the runtime, and the 6b guard above would go on
    checking a model of a dispatch that no longer exists.
    """
    from test.test_prompt_corpus import CASES, _compose

    for case in CASES:
        assert _compose(*case) == _prompt_the_runtime_dispatch_returns(case, monkeypatch), (
            "test_prompt_corpus._compose no longer mirrors runner.modes.build_plan "
            f"for {case}; the golden corpus and the 6b guard are both reading it"
        )


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
