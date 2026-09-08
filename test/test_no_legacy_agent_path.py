"""search_eval_tools was reachable only when all four axes were unset."""
import importlib

import pytest

from sana_evaluation.cli import PRESETS  # noqa: F401  (import guard)


def test_axis_defaults_leave_no_axis_unset():
    from sana_evaluation import cli
    a = cli.parse([])
    for axis in ("search", "results", "plan", "compute"):
        assert getattr(a, axis), f"{axis} unset — this would reach the deleted legacy path"


def test_search_eval_tools_is_gone():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("sana_evaluation.tools.external.search_eval_tools")
