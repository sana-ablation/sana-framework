"""search_eval_tools was reachable only when all four axes were unset.

The deleted agent branch ran when every one of ``RunConfig``'s four mode axes
was ``None``. ``DataLakeAgent.__init__`` and ``BatchRunner.__init__`` both do
``run_config or RunConfig()``, so it is the *dataclass* defaults -- not
argparse's -- that decide whether that branch was reachable at all. A library
caller writing ``DataLakeAgent(AgentConfig())``, the usage example in
``sana_evaluation/__init__.py``, used to land there.

So the axes are defaulted once, in ``config.AXIS_DEFAULTS``, and every source
that resolves them is asserted here to agree with it.
"""
import importlib

import pytest

from sana_evaluation.cli import PRESETS  # noqa: F401  (import guard)
from sana_evaluation.config import AXIS_DEFAULTS, RunConfig

_AXIS_FIELDS = (
    "search_tool_mode",
    "search_results_mode",
    "plan_mode",
    "computation_tool_mode",
)


def test_bare_run_config_leaves_no_axis_unset():
    """The real precondition for deleting the legacy arm."""
    run_config = RunConfig()
    for field in _AXIS_FIELDS:
        assert getattr(run_config, field), (
            f"RunConfig().{field} is unset — this would reach the deleted legacy path"
        )


def test_axis_defaults_leave_no_axis_unset():
    from sana_evaluation import cli
    a = cli.parse([])
    for axis in ("search", "results", "plan", "compute"):
        assert getattr(a, axis), f"{axis} unset — this would reach the deleted legacy path"


def test_run_config_and_cli_defaults_describe_the_same_run():
    """AXIS_DEFAULTS, the argparse defaults and RunConfig's fields cannot drift."""
    from sana_evaluation import cli
    a = cli.parse([])
    assert AXIS_DEFAULTS == {
        "search_tool_mode": a.search,
        "search_results_mode": a.results,
        "plan_mode": a.plan,
        "computation_tool_mode": a.compute,
    }
    run_config = RunConfig()
    assert AXIS_DEFAULTS == {f: getattr(run_config, f) for f in _AXIS_FIELDS}
    assert cli._resolve_mode_axes(
        search_tool=None, search_results=None, plan=None, computation_tool=None
    ) == (
        run_config.search_tool_mode,
        run_config.search_results_mode,
        run_config.plan_mode,
        run_config.computation_tool_mode,
    )


def test_runner_resolves_a_bare_run_config_to_the_axis_defaults():
    """The runner's own coalescing agrees, so the dataclass defaults change nothing."""
    from sana_evaluation.runner.modes import (
        _STANDARD_SEARCH_TOOLS_AVAILABLE,
        build_data_tools,
        build_mode_bundle,
    )

    if not _STANDARD_SEARCH_TOOLS_AVAILABLE:
        pytest.skip("standard search backend unavailable in this environment")

    bundle = build_mode_bundle(
        RunConfig(), data_tools=build_data_tools(), task_context=None
    )
    assert {
        "search_tool_mode": bundle.modes["search_tool"],
        "search_results_mode": bundle.modes["search_results"],
        "plan_mode": bundle.modes["plan"],
        "computation_tool_mode": bundle.modes["computation_tool"],
    } == AXIS_DEFAULTS


def test_search_eval_tools_is_gone():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("sana_evaluation.tools.external.search_eval_tools")
