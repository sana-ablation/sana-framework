import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sana_analysis.report_generator.run_mode_delta_figures import (
    SEMANTIC_DELTA_COMPACT_AXIS_LABEL_FONTSIZE,
    SEMANTIC_DELTA_COMPACT_LABEL_FONTSIZE,
    SEMANTIC_DELTA_COMPACT_MODEL_FONTSIZE,
    SEMANTIC_DELTA_COMPACT_TITLE_FONTSIZE,
    SEMANTIC_DELTA_COMPACT_VALUE_FONTSIZE,
    _delta_value_label,
    _comparison_models,
    _paired_condition_axis_label,
    build_paired_mode_metric_rows,
    build_semantic_delta_rows,
    generate_delta_figures,
    load_summary_rows,
)


class TestRunModeDeltaFigures(unittest.TestCase):
    def test_compact_semantic_delta_figure_uses_paper_readable_type(self):
        self.assertGreaterEqual(SEMANTIC_DELTA_COMPACT_TITLE_FONTSIZE, 14.0)
        self.assertLessEqual(SEMANTIC_DELTA_COMPACT_TITLE_FONTSIZE, 17.0)
        self.assertGreaterEqual(SEMANTIC_DELTA_COMPACT_LABEL_FONTSIZE, 12.5)
        self.assertGreaterEqual(SEMANTIC_DELTA_COMPACT_VALUE_FONTSIZE, 11.5)
        self.assertGreaterEqual(SEMANTIC_DELTA_COMPACT_MODEL_FONTSIZE, 14.0)
        self.assertLessEqual(SEMANTIC_DELTA_COMPACT_AXIS_LABEL_FONTSIZE, 9.0)

    def test_delta_value_label_wraps_delta_in_parentheses(self):
        self.assertEqual(_delta_value_label(34.1, 0.0), "34.1% (+0.0%)")
        self.assertEqual(_delta_value_label(31.1, -3.0), "31.1% (-3.0%)")

    def test_paired_condition_axis_labels_are_compact_for_plotting(self):
        self.assertEqual(_paired_condition_axis_label("nns"), "BM25 / No Plan / Std DA")
        self.assertEqual(_paired_condition_axis_label("sss"), "Pneuma / Plan / Std DA")
        self.assertEqual(_paired_condition_axis_label("iii"), "Ideal / Ideal / Ideal")

    def test_comparison_models_orders_nano_before_mini_for_openai_model_names(self):
        rows_by_model = {
            "openai_gpt-5-mini": [],
            "openai_gpt-5.4-nano": [],
        }

        self.assertEqual(
            _comparison_models(rows_by_model),
            ["openai_gpt-5.4-nano", "openai_gpt-5-mini"],
        )

    def test_build_semantic_delta_rows_compares_one_axis_against_naive_baseline(self):
        rows = [
            {
                "model": "model_a",
                "variant": "search_i_results_i_plann_computei_k5_skills_off",
                "semantic_match": 0.40,
            },
            {
                "model": "model_a",
                "variant": "search_i_results_i_pland_computei_k5_skills_off",
                "semantic_match": 0.50,
            },
            {
                "model": "model_a",
                "variant": "search_i_results_i_plani_computei_k5_skills_off",
                "semantic_match": 0.80,
            },
            {
                "model": "model_a",
                "variant": "search_n_results_i_plani_computei_k5_skills_off",
                "semantic_match": 0.20,
            },
            {
                "model": "model_a",
                "variant": "search_d_results_i_plani_computei_k5_skills_off",
                "semantic_match": 0.50,
            },
            {
                "model": "model_a",
                "variant": "search_p_results_i_plani_computei_k5_skills_off",
                "semantic_match": 0.70,
            },
            {
                "model": "model_a",
                "variant": "search_i_results_i_plani_k5_skills_off",
                "semantic_match": 0.60,
            },
            {
                "model": "model_a",
                "variant": "search_n_results_i_plann_k5_skills_off",
                "semantic_match": 0.10,
            },
            {
                "model": "model_a",
                "variant": "search_d_results_i_pland_k5_skills_off",
                "semantic_match": 0.10,
            },
        ]

        delta_rows = build_semantic_delta_rows(rows)
        by_key = {(row["ablation"], row["label"]): row for row in delta_rows}

        self.assertEqual(by_key[("Plan Ablation", "No Plan")]["delta"], 0.0)
        self.assertEqual(by_key[("Plan Ablation", "No Plan")]["semantic_match"], 0.40)
        self.assertEqual(by_key[("Plan Ablation", "Default Plan")]["delta"], 0.10)
        self.assertEqual(by_key[("Plan Ablation", "Ideal Plan")]["delta"], 0.40)
        self.assertEqual(by_key[("Search Ablation", "BM25 Search")]["delta"], 0.0)
        self.assertEqual(by_key[("Search Ablation", "PNEUMA Hybrid Search")]["delta"], 0.30)
        self.assertEqual(by_key[("Search Ablation", "Preloaded Sources")]["delta"], 0.50)
        self.assertEqual(by_key[("Search Ablation", "PNEUMA Hybrid Search")]["semantic_match"], 0.50)
        self.assertEqual(by_key[("Data Analysis Ablation", "Standard Data Analysis")]["delta"], 0.0)
        self.assertEqual(by_key[("Data Analysis Ablation", "Ideal Data Analysis")]["delta"], 0.20)
        self.assertEqual(by_key[("Data Analysis Ablation", "Standard Data Analysis")]["semantic_match"], 0.60)
        self.assertEqual(by_key[("Plan Ablation", "Ideal Plan")]["semantic_match"], 0.80)

    def test_build_paired_mode_metric_rows_uses_canonical_condition_columns(self):
        rows = [
            {
                "model": "model_a",
                "variant": "search_n_results_i_plann_k5_skills_off",
                "semantic_match": 0.11,
                "em": 0.1,
                "D_ret": 0.2,
                "D_acc": 0.3,
            },
            {
                "model": "model_a",
                "variant": "search_d_results_i_pland_k5_skills_off",
                "semantic_match": 0.44,
                "em": 0.4,
                "D_ret": 0.5,
                "D_acc": 0.6,
            },
            {
                "model": "model_a",
                "variant": "search_i_results_i_plani_computei_k5_skills_off",
                "semantic_match": 0.77,
                "em": 0.7,
                "D_ret": 0.8,
                "D_acc": 0.9,
            },
            {
                "model": "model_b",
                "variant": "search_i_results_i_plani_computei_k5_skills_off",
                "semantic_match": 0.9,
                "em": 0.9,
                "D_ret": 0.8,
                "D_acc": 0.7,
            },
            {
                "model": "model_b",
                "variant": "search_n_results_n_plann_k5_skills_off",
                "em": 0.2,
                "D_ret": 0.3,
                "D_acc": 0.4,
            },
        ]

        paired_rows = build_paired_mode_metric_rows(rows)
        model_a_rows = [row for row in paired_rows if row["model"] == "model_a"]

        self.assertEqual(
            [row["condition_code"] for row in model_a_rows],
            [
                "BM25 Search, No Plan, Standard Data Analysis",
                "Pneuma Search, Plan, Standard Data Analysis",
                "Ideal Search, Ideal Plan, Ideal Data Analysis",
            ],
        )
        self.assertEqual(paired_rows[0]["condition_label"], "BM25 Search, No Plan, Standard Data Analysis")
        self.assertEqual(paired_rows[0]["model"], "model_a")
        self.assertEqual(paired_rows[0]["semantic_match"], 0.11)
        self.assertEqual(paired_rows[0]["em"], 0.1)
        self.assertEqual(paired_rows[2]["D_acc"], 0.9)
        self.assertEqual({row["model"] for row in paired_rows}, {"model_a", "model_b"})
        self.assertEqual(
            [(row["model"], row["condition_id"]) for row in paired_rows],
            [("model_a", "nns"), ("model_a", "sss"), ("model_a", "iii"), ("model_b", "iii")],
        )

    def test_build_semantic_delta_rows_keeps_partial_ablation_when_baseline_exists(self):
        rows = [
            {
                "model": "model_a",
                "variant": "search_n_results_i_plani_computei_k5_skills_off",
                "semantic_match": 0.20,
            },
            {
                "model": "model_a",
                "variant": "search_d_results_i_plani_computei_k5_skills_off",
                "semantic_match": 0.50,
            },
            {
                "model": "model_a",
                "variant": "search_i_results_i_plani_computei_k5_skills_off",
                "semantic_match": 0.80,
            },
        ]

        delta_rows = build_semantic_delta_rows(rows)

        self.assertEqual(
            [(row["ablation"], row["label"], row["delta"]) for row in delta_rows],
            [
                ("Search Ablation", "BM25 Search", 0.0),
                ("Search Ablation", "PNEUMA Hybrid Search", 0.3),
                ("Search Ablation", "Ideal Search", 0.6),
            ],
        )

    def test_load_summary_rows_backfills_em_from_per_task_semantic_csv(self):
        with TemporaryDirectory() as tmpdir:
            analysis_dir = Path(tmpdir)
            (analysis_dir / "summary.json").write_text(
                json.dumps(
                    [
                        {
                            "condition_model": "model_a/search_i_results_i_plani_k5_skills_off",
                            "model": "model_a",
                            "variant": "search_i_results_i_plani_k5_skills_off",
                        }
                    ]
                )
            )
            (analysis_dir / "per_task_semantic.csv").write_text(
                "condition_model,exact_match\n"
                "model_a/search_i_results_i_plani_k5_skills_off,1\n"
                "model_a/search_i_results_i_plani_k5_skills_off,0\n"
            )

            rows = load_summary_rows(analysis_dir)

            self.assertEqual(rows[0]["em"], 0.5)

    def test_generate_delta_figures_removes_stale_per_model_pdfs(self):
        rows = [
            {
                "model": "model_a",
                "variant": "search_n_results_i_plann_k5_skills_off",
                "em": 0.1,
                "D_ret": 0.2,
                "D_acc": 0.3,
            },
            {
                "model": "model_a",
                "variant": "search_d_results_i_pland_k5_skills_off",
                "em": 0.4,
                "D_ret": 0.5,
                "D_acc": 0.6,
            },
            {
                "model": "model_a",
                "variant": "search_i_results_i_plani_computei_k5_skills_off",
                "em": 0.7,
                "D_ret": 0.8,
                "D_acc": 0.9,
            },
        ]
        with TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            stale_delta = output_dir / "semantic_delta_ablation_model_b.pdf"
            stale_paired = output_dir / "paired_mode_metrics_model_b.pdf"
            stale_delta.write_text("stale")
            stale_paired.write_text("stale")

            generate_delta_figures(rows, output_dir)

            self.assertFalse(stale_delta.exists())
            self.assertFalse(stale_paired.exists())

    def test_generate_delta_figures_writes_side_by_side_comparison_pdfs(self):
        rows = []
        variants = [
            ("search_i_results_i_plann_computei_k5_skills_off", 0.20),
            ("search_i_results_i_pland_computei_k5_skills_off", 0.30),
            ("search_i_results_i_plani_computei_k5_skills_off", 0.40),
            ("search_n_results_i_plani_computei_k5_skills_off", 0.25),
            ("search_d_results_i_plani_computei_k5_skills_off", 0.35),
            ("search_p_results_i_plani_computei_k5_skills_off", 0.45),
            ("search_i_results_i_plani_k5_skills_off", 0.30),
            ("search_n_results_i_plann_k5_skills_off", 0.15),
            ("search_d_results_i_pland_k5_skills_off", 0.25),
        ]
        for model, offset in (("openai_gpt-5.4-nano", 0.0), ("openai_gpt-5-mini", 0.1)):
            for variant, value in variants:
                rows.append(
                    {
                        "model": model,
                        "variant": variant,
                        "semantic_match": value + offset,
                        "em": value + offset,
                        "D_ret": value + offset,
                        "D_acc": value + offset,
                    }
                )

        with TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            generate_delta_figures(rows, output_dir)

            self.assertTrue((output_dir / "semantic_delta_ablation_comparison.pdf").exists())
            self.assertTrue((output_dir / "semantic_delta_ablation_compact.pdf").exists())
            self.assertTrue((output_dir / "paired_mode_metrics_comparison.pdf").exists())


if __name__ == "__main__":
    unittest.main()
