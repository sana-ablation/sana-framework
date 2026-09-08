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


def test_kramabench_never_mentions_query_file():
    prompt = compose.build(plan="standard", search="standard",
                           benchmark="kramabench", skills=False)
    assert "query_file" not in prompt
    assert "no SQL table tool" not in prompt, (
        "the correction is unnecessary once query_file is simply not listed"
    )


def test_preflight_and_runtime_resolve_the_same_paths():
    from sana_evaluation.preflight import _prompt_files_for_modes
    for benchmark in ("lakeqa", "kramabench"):
        for search in ("naive", "standard", "ideal", "preloaded", "web"):
            for plan in ("naive", "standard", "ideal"):
                kwargs = dict(plan=plan, search=search,
                              benchmark=benchmark, skills=False)
                assert _prompt_files_for_modes(**kwargs) == compose.fragment_paths(**kwargs)
