"""Slicing axes inlined from the deleted search_depth / reasoning_density /
tool_error_analysis modules.

These had no direct test before -- the modules' own compute_* entry points were
dead, and the pipeline imported only the constants. Inlining them surfaced a
NameError that the whole suite passed straight over, so they are pinned here.
"""
import json
import unittest
from pathlib import Path

from sana_analysis.run_mode_analysis import (
    _DATA_TOOLS,
    _assign_reasoning_density_bin,
    _assign_search_depth_bin,
    load_task_gold_counts,
)


def test_search_depth_bins_cover_the_range():
    assert [_assign_search_depth_bin(n) for n in (1, 2, 3, 4, 6, 7, 10, 11, 30)] == [
        "1", "2-3", "2-3", "4-6", "4-6", "7-10", "7-10", "11-30", "11-30"
    ]


def test_search_depth_bin_saturates_above_the_top_bin():
    assert _assign_search_depth_bin(999) == "11-30"


def test_reasoning_density_bins_cover_the_range():
    assert [_assign_reasoning_density_bin(n) for n in (0, 2, 3, 4, 5, 7, 8, 10, 11)] == [
        "<=2", "<=2", "3-4", "3-4", "5-7", "5-7", "8-10", "8-10", ">10"
    ]


def test_load_task_gold_counts_keys_by_both_path_and_stem(tmp_path):
    task_dir = tmp_path / "k-1-d-2"
    task_dir.mkdir()
    (task_dir / "task_1.json").write_text(json.dumps({"datasets_used": ["a", "b"]}))

    counts = load_task_gold_counts(str(tmp_path))

    assert set(counts.values()) == {2}
    # One entry keyed by full path, one by stem -- callers hold either.
    assert len(counts) == 2
    assert any(k.endswith("task_1.json") for k in counts)


def test_load_task_gold_counts_on_a_real_task_set():
    counts = load_task_gold_counts("benchmarks/lakeqa/tasks_20_subset/tasks")
    assert counts, "no gold counts loaded from the shipped task set"
    assert all(isinstance(v, int) and v >= 0 for v in counts.values())


def test_data_tools_are_the_data_touching_ones():
    assert "query_file" in _DATA_TOOLS and "read_file" in _DATA_TOOLS
    # Planning and answer submission are not data tools.
    assert "plan" not in _DATA_TOOLS and "submit_answer" not in _DATA_TOOLS


class TestCanonicalVariantDecoding(unittest.TestCase):
    """Gen-4 is what every directory on disk uses. The old parser read it as
    all-None, which is why every canonical lookup missed."""

    CANONICAL = "search_ideal__plan_ideal__compute_ideal__results_rich__k5__skills_off"

    def test_canonical_variant_resolves_every_axis(self):
        from sana_analysis.run_mode_analysis import _parse_variant

        axes = _parse_variant(self.CANONICAL)
        self.assertEqual(axes["search_tool"], "ideal")
        self.assertEqual(axes["search_results"], "rich")
        self.assertEqual(axes["agent_management"], "ideal")
        self.assertEqual(axes["computation_tool"], "ideal")
        self.assertEqual(axes["k"], 5)
        self.assertEqual(axes["plan_skills"], "off")

    def test_canonical_variant_renders_a_real_legend(self):
        from sana_analysis.run_mode_analysis import _compact_variant_label

        label = _compact_variant_label(self.CANONICAL)
        self.assertNotIn("?", label)
        self.assertIn("S:Ideal", label)

    def test_standard_compute_is_not_silently_reported_as_ideal(self):
        from sana_analysis.run_mode_analysis import _parse_variant

        standard = "search_ideal__plan_ideal__compute_standard__results_rich__k5__skills_off"
        self.assertEqual(_parse_variant(standard)["computation_tool"], "standard")
        self.assertEqual(_parse_variant(self.CANONICAL)["computation_tool"], "ideal")

    def test_gen1_literals_still_decode(self):
        from sana_analysis.run_mode_analysis import _parse_variant

        axes = _parse_variant("search_i_results_i_plani_computei_k5_skills_off")
        self.assertEqual(axes["search_tool"], "ideal")
        self.assertEqual(axes["agent_management"], "ideal")
        self.assertEqual(axes["computation_tool"], "ideal")

    def test_condition_figure_order_matches_canonical_directories(self):
        from sana_analysis.run_mode_analysis import TURN_WASTE_CONDITION_FIGURE_ORDER
        from sana_analysis.variants import parse_variant

        names = {
            "No Plan": "search_ideal__plan_naive__compute_ideal__results_rich__k5__skills_off",
            "Standard Plan": "search_ideal__plan_standard__compute_ideal__results_rich__k5__skills_off",
            "BM25": "search_naive__plan_ideal__compute_ideal__results_rich__k5__skills_off",
            "Pneuma Hybrid": "search_standard__plan_ideal__compute_ideal__results_rich__k5__skills_off",
            "Standard Computation": "search_ideal__plan_ideal__compute_standard__results_rich__k5__skills_off",
            "Ideal": "search_ideal__plan_ideal__compute_ideal__results_rich__k5__skills_off",
        }
        for label, axes in TURN_WASTE_CONDITION_FIGURE_ORDER:
            self.assertTrue(
                parse_variant(names[label]).matches(**axes),
                f"{label} does not match {names[label]}",
            )


class _FakeBar:
    """Just enough of a matplotlib bar artist for the value-label loop inside
    _plot_turn_waste_reconciled_groups_by_condition to run unmodified."""

    def __init__(self, x, width=0.72):
        self._x = x
        self._width = width

    def get_x(self):
        return self._x

    def get_width(self):
        return self._width


class _FakeAxis:
    """Records exactly the two calls that reveal model and condition
    ordering (set_xticklabels, and the transform-carrying text() calls);
    everything else the real function calls on an Axes is a harmless no-op,
    so the production code runs to completion without real matplotlib."""

    def __init__(self):
        self.text_calls = []
        self.xticklabels = None

    def bar(self, x_positions, values, **kwargs):
        return [_FakeBar(x) for x in x_positions]

    def text(self, *args, **kwargs):
        self.text_calls.append((args, kwargs))

    def set_xticklabels(self, labels, **kwargs):
        self.xticklabels = list(labels)

    def get_xaxis_transform(self):
        return "xaxis-transform"

    def __getattr__(self, _name):
        def _noop(*args, **kwargs):
            return None
        return _noop


class _FakeFig:
    def __getattr__(self, _name):
        def _noop(*args, **kwargs):
            return None
        return _noop


class _FakePlt:
    def __init__(self):
        self.axis = _FakeAxis()

    def subplots(self, *args, **kwargs):
        return _FakeFig(), self.axis

    def close(self, *args, **kwargs):
        pass


class TestTurnWasteConditionFigureRealFunction(unittest.TestCase):
    """`_plot_turn_waste_reconciled_groups_by_condition` had no committed
    regression test at all -- Task 4's deferred minor. It is never exercised
    end-to-end in the real pipeline (turn_waste_grouped_dir is None
    upstream), so its ordering/wiring was only ever checked ad-hoc in a
    transcript. These drive the real function (with a fake plotting backend,
    since it needs a `plt`-shaped object) and assert on what it actually
    computed -- not a local reimplementation of its ordering logic, which is
    what the previous version of this test did (it would have passed even
    with an empty TURN_WASTE_CONDITION_FIGURE_MODEL_PREFERENCE)."""

    NO_PLAN = "search_ideal__plan_naive__compute_ideal__results_rich__k5__skills_off"
    IDEAL = "search_ideal__plan_ideal__compute_ideal__results_rich__k5__skills_off"

    def _condition_groups(self):
        # Insertion order is deliberately scrambled relative to both
        # CONDITION_ORDER (Ideal's entries come first here, but "No Plan"
        # must still come first in the output) and any preference/sorted
        # ordering of models, so neither assertion below can be satisfied by
        # accidentally preserving input order.
        group = "Data/source access and repair loops"
        return {
            "e1": {"variant": self.IDEAL, "model": "openai_gpt-5.2", "groups": {group: {"n": 3}}},
            "e2": {"variant": self.IDEAL, "model": "openai_gpt-5-mini", "groups": {group: {"n": 4}}},
            "e3": {"variant": self.IDEAL, "model": "openai_gpt-5.4-nano", "groups": {group: {"n": 5}}},
            "e4": {"variant": self.NO_PLAN, "model": "openai_gpt-5.2", "groups": {group: {"n": 6}}},
            "e5": {"variant": self.NO_PLAN, "model": "openai_gpt-5-mini", "groups": {group: {"n": 7}}},
            "e6": {"variant": self.NO_PLAN, "model": "openai_gpt-5.4-nano", "groups": {group: {"n": 8}}},
        }

    def test_models_ordered_by_preference_with_unlisted_model_appended(self):
        from sana_analysis.run_mode_analysis import _plot_turn_waste_reconciled_groups_by_condition

        plt = _FakePlt()
        _plot_turn_waste_reconciled_groups_by_condition(
            plt, self._condition_groups(), Path("unused-turn-waste-by-condition.pdf")
        )

        self.assertIsNotNone(plt.axis.xticklabels, "the function returned before drawing anything")
        # openai_gpt-5.4-nano and openai_gpt-5-mini are in
        # TURN_WASTE_CONDITION_FIGURE_MODEL_PREFERENCE, in that order.
        # openai_gpt-5.2 is not in the preference list at all -- the old
        # constant (TURN_WASTE_CONDITION_FIGURE_MODEL_ORDER) would have
        # dropped it entirely instead of appending it. There are two
        # conditions, so this block repeats twice.
        expected_block = ["5.4\nnano", "5\nmini", "gpt\n5.2"]
        self.assertEqual(plt.axis.xticklabels, expected_block + expected_block)

    def test_conditions_ordered_by_condition_order_not_input_order(self):
        from sana_analysis.run_mode_analysis import _plot_turn_waste_reconciled_groups_by_condition

        plt = _FakePlt()
        _plot_turn_waste_reconciled_groups_by_condition(
            plt, self._condition_groups(), Path("unused-turn-waste-by-condition.pdf")
        )

        # Only the per-condition axis labels pass a `transform` kwarg to
        # ax.text(); the bar-value and per-column-total labels don't, so this
        # filter isolates exactly the condition-ordering calls, in the order
        # the function made them.
        condition_labels = [args[2] for args, kwargs in plt.axis.text_calls if "transform" in kwargs]
        # _condition_groups() inserts Ideal's rows before No Plan's, but
        # CONDITION_ORDER says No Plan comes first -- this is the ordering
        # Task 4 left unpinned.
        self.assertEqual(condition_labels, ["No\nPlan", "Ideal"])


def test_variant_sort_key_orders_rich_before_minimal_on_the_results_axis():
    # Same search/plan/compute; only the results axis differs. _MODE_PRIORITY
    # is the only thing standing between this and a silent fall-through to
    # the shared default (4), which would leave ordering to the k/sc/variant
    # tiebreak instead of an explicit priority.
    from sana_analysis.run_mode_analysis import _variant_sort_key

    rich = "search_ideal__plan_ideal__compute_ideal__results_rich__k5__skills_off"
    minimal = "search_ideal__plan_ideal__compute_ideal__results_minimal__k5__skills_off"

    assert _variant_sort_key(rich) < _variant_sort_key(minimal)
