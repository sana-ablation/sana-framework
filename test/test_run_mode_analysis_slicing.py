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


class TestModelOrderKeepsUnlistedModels(unittest.TestCase):
    def test_preference_list_orders_but_does_not_filter(self):
        from sana_analysis.run_mode_analysis import (
            TURN_WASTE_CONDITION_FIGURE_MODEL_PREFERENCE as PREFERENCE,
        )

        observed = {"openai_gpt-5-mini", "openai_gpt-5.2", "openai_gpt-5.4-nano"}
        order = [m for m in PREFERENCE if m in observed]
        order += [m for m in sorted(observed) if m not in order]
        self.assertEqual(
            order, ["openai_gpt-5.4-nano", "openai_gpt-5-mini", "openai_gpt-5.2"]
        )
        self.assertEqual(set(order), observed)
