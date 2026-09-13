import unittest

from sana_analysis.paper.plan_ablation_figure import (
    PLAN_D_AXES,
    PLAN_I_AXES,
    _normalize_plan_similarity,
    _summarize_rows,
)


class TestPlanDefaultFigureGenerator(unittest.TestCase):
    def test_normalize_skips_missing_both_missing_plan_i_and_maps_missing_plan_d_to_no_plan(self):
        self.assertIsNone(_normalize_plan_similarity({"missing_plan_type": "missing_both", "plan_similarity": "not_comparable"}))
        self.assertIsNone(_normalize_plan_similarity({"missing_plan_type": "missing_plan_i", "plan_similarity": "not_comparable"}))
        self.assertEqual(
            _normalize_plan_similarity({"missing_plan_type": "missing_plan_d", "plan_similarity": "not_comparable"}),
            "no_plan",
        )
        self.assertEqual(
            _normalize_plan_similarity({"missing_plan_type": "", "plan_similarity": "incomplete_plan"}),
            "incomplete_plan",
        )

    def test_summarize_rows_filters_to_canonical_d_vs_i(self):
        canonical_d = "search_i_results_i_pland_computei_k5_skills_off"
        canonical_i = "search_i_results_i_plani_computei_k5_skills_off"
        rows = [
            {
                "benchmark": "lakeqa",
                "model_variant": "openai_gpt-5-mini",
                "plan_d_mode": canonical_d,
                "plan_i_mode": canonical_i,
                "missing_plan_type": "",
                "plan_similarity": "similar",
            },
            {
                "benchmark": "lakeqa",
                "model_variant": "openai_gpt-5-mini",
                "plan_d_mode": canonical_d,
                "plan_i_mode": canonical_i,
                "missing_plan_type": "missing_plan_d",
                "plan_similarity": "not_comparable",
            },
            {
                "benchmark": "lakeqa",
                "model_variant": "openai_gpt-5-mini",
                "plan_d_mode": "search_d_results_i_pland_k5_skills_off",
                "plan_i_mode": canonical_i,
                "missing_plan_type": "",
                "plan_similarity": "operation_mismatch",
            },
            {
                "benchmark": "kramabench",
                "model_variant": "openai_gpt-5-mini",
                "plan_d_mode": canonical_d,
                "plan_i_mode": canonical_i,
                "missing_plan_type": "missing_plan_i",
                "plan_similarity": "not_comparable",
            },
            {
                "benchmark": "kramabench",
                "model_variant": "openai_gpt-5-mini",
                "plan_d_mode": canonical_d,
                "plan_i_mode": canonical_i,
                "missing_plan_type": "missing_both",
                "plan_similarity": "not_comparable",
            },
        ]

        summary = _summarize_rows(rows)

        lakeqa = summary[("lakeqa", "openai_gpt-5-mini")]
        self.assertEqual(lakeqa["n"], 2)
        self.assertEqual(lakeqa["counts"]["similar"], 1)
        self.assertEqual(lakeqa["counts"]["no_plan"], 1)
        self.assertNotIn(("kramabench", "openai_gpt-5-mini"), summary)


class TestCanonicalPlanModes(unittest.TestCase):
    OBSERVED = [
        "search_ideal__plan_standard__compute_ideal__results_rich__k5__skills_off",
        "search_ideal__plan_ideal__compute_ideal__results_rich__k5__skills_off",
        "search_naive__plan_ideal__compute_ideal__results_rich__k5__skills_off",
    ]

    def test_the_plan_pair_resolves_from_canonical_directories(self):
        from sana_analysis.variants import find_variant

        self.assertEqual(find_variant(self.OBSERVED, **PLAN_D_AXES), self.OBSERVED[0])
        self.assertEqual(find_variant(self.OBSERVED, **PLAN_I_AXES), self.OBSERVED[1])

    def test_the_plan_pair_still_resolves_from_gen1_directories(self):
        from sana_analysis.variants import find_variant

        gen1 = [
            "search_i_results_i_pland_computei_k5_skills_off",
            "search_i_results_i_plani_computei_k5_skills_off",
        ]
        self.assertEqual(find_variant(gen1, **PLAN_D_AXES), gen1[0])
        self.assertEqual(find_variant(gen1, **PLAN_I_AXES), gen1[1])

    def test_the_two_predicates_differ_only_on_the_plan_axis(self):
        self.assertEqual(PLAN_D_AXES["plan"], "standard")
        self.assertEqual(PLAN_I_AXES["plan"], "ideal")
        self.assertEqual(
            {k: v for k, v in PLAN_D_AXES.items() if k != "plan"},
            {k: v for k, v in PLAN_I_AXES.items() if k != "plan"},
        )

    def test_the_figure_title_names_the_condition_the_way_every_table_does(self):
        # The plan=standard condition is "Standard Plan" in run_mode_analysis,
        # combine_grouped_models and delta_figures. A figure captioned "Default
        # Plan" beside a CSV saying "Standard Plan" is the same condition under
        # two names, which is how the naming drift this package just retired
        # began.
        from sana_analysis.paper.plan_ablation_figure import FIGURE_TITLE
        from sana_analysis.paper.delta_figures import PLAN_ABLATION

        self.assertIn("Standard Plan", FIGURE_TITLE)
        self.assertNotIn("Default Plan", FIGURE_TITLE)
        # And the label the CSV beside it uses, so the two cannot drift apart.
        self.assertIn("Standard Plan", [label for _code, label in PLAN_ABLATION])


if __name__ == "__main__":
    unittest.main()
