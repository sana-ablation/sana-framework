import unittest

from sana_analysis.report_generator.plan_default_figure_generator import (
    CANONICAL_PLAN_D_MODE,
    CANONICAL_PLAN_I_MODE,
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
        rows = [
            {
                "benchmark": "lakeqa",
                "model_variant": "openai_gpt-5-mini",
                "plan_d_mode": CANONICAL_PLAN_D_MODE,
                "plan_i_mode": CANONICAL_PLAN_I_MODE,
                "missing_plan_type": "",
                "plan_similarity": "similar",
            },
            {
                "benchmark": "lakeqa",
                "model_variant": "openai_gpt-5-mini",
                "plan_d_mode": CANONICAL_PLAN_D_MODE,
                "plan_i_mode": CANONICAL_PLAN_I_MODE,
                "missing_plan_type": "missing_plan_d",
                "plan_similarity": "not_comparable",
            },
            {
                "benchmark": "lakeqa",
                "model_variant": "openai_gpt-5-mini",
                "plan_d_mode": "search_d_results_i_pland_k5_skills_off",
                "plan_i_mode": CANONICAL_PLAN_I_MODE,
                "missing_plan_type": "",
                "plan_similarity": "operation_mismatch",
            },
            {
                "benchmark": "kramabench",
                "model_variant": "openai_gpt-5-mini",
                "plan_d_mode": CANONICAL_PLAN_D_MODE,
                "plan_i_mode": CANONICAL_PLAN_I_MODE,
                "missing_plan_type": "missing_plan_i",
                "plan_similarity": "not_comparable",
            },
            {
                "benchmark": "kramabench",
                "model_variant": "openai_gpt-5-mini",
                "plan_d_mode": CANONICAL_PLAN_D_MODE,
                "plan_i_mode": CANONICAL_PLAN_I_MODE,
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


if __name__ == "__main__":
    unittest.main()
