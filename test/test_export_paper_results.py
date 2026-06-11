import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sana_analysis.report_generator.export_paper_results import (
    build_canonical_mode_rows,
    build_main_ablation_table_rows,
    build_main_result_rows,
    load_uncached_cost_summary,
    render_canonical_modes_table,
    render_main_ablation_table,
    render_results_table,
)


class ExportPaperResultsTests(unittest.TestCase):
    def test_canonical_mode_rows_include_call_counts_and_model_labels(self):
        summary_rows = [
            {
                "model": "openai_gpt-5.4-nano",
                "variant": "search_n_results_i_plann_k5_skills_off",
                "search_tool": "naive",
                "search_results": "ideal",
                "agent_management": "naive",
                "computation_tool": "standard",
                "n": 135,
                "semantic_match": 0.20,
                "D_ret": 0.40,
                "D_acc": 0.30,
                "avg_search_calls": 5.01,
                "avg_read_calls": 10.94,
            },
            {
                "model": "openai_gpt-5.4-nano",
                "variant": "search_d_results_i_pland_k5_skills_off",
                "search_tool": "standard",
                "search_results": "ideal",
                "agent_management": "standard",
                "computation_tool": "standard",
                "n": 135,
                "semantic_match": 0.25,
                "D_ret": 0.45,
                "D_acc": 0.35,
                "avg_search_calls": 4.49,
                "avg_read_calls": 11.51,
            },
            {
                "model": "openai_gpt-5.4-nano",
                "variant": "search_i_results_i_plani_computei_k5_skills_off",
                "search_tool": "ideal",
                "search_results": "ideal",
                "agent_management": "ideal",
                "computation_tool": "ideal",
                "n": 135,
                "semantic_match": 0.50,
                "D_ret": 0.60,
                "D_acc_recall": 0.55,
                "avg_search_calls": 3.25,
                "avg_read_calls": 12.75,
            },
        ]

        rows = build_canonical_mode_rows(summary_rows)

        self.assertEqual([row["model"] for row in rows[:3]], ["gpt-5.4-nano", "gpt-5.4-nano", "gpt-5.4-nano"])
        self.assertEqual([row["mode"] for row in rows[:3]], ["Naive", "Standard", "Ideal"])
        self.assertEqual(rows[0]["semantic_match"], "20.0")
        self.assertEqual(rows[1]["semantic_match_delta"], "+5.0")
        self.assertEqual(rows[2]["D_acc"], "55.0")
        self.assertEqual(rows[0]["ret_tool_calls"], "5.0")
        self.assertEqual(rows[2]["acc_tool_calls"], "12.8")

    def test_render_canonical_modes_table_has_ret_and_acc_tool_call_columns(self):
        rows = [
            {
                "model": "gpt-5.4-nano",
                "mode": "Naive",
                "n": "135",
                "semantic_match": "20.0",
                "semantic_match_delta": None,
                "D_ret": "40.0",
                "D_ret_delta": None,
                "D_acc": "30.0",
                "D_acc_delta": None,
                "ret_tool_calls": "5.0",
                "acc_tool_calls": "10.9",
            },
            {
                "model": "gpt-5.4-nano",
                "mode": "Standard",
                "n": "135",
                "semantic_match": "25.0",
                "semantic_match_delta": "+5.0",
                "D_ret": "45.0",
                "D_ret_delta": "+5.0",
                "D_acc": "35.0",
                "D_acc_delta": "+5.0",
                "ret_tool_calls": "4.5",
                "acc_tool_calls": "11.5",
            },
            {
                "model": "gpt-5.4-nano",
                "mode": "Ideal",
                "n": "135",
                "semantic_match": "50.0",
                "semantic_match_delta": "+30.0",
                "D_ret": "60.0",
                "D_ret_delta": "+20.0",
                "D_acc": "55.0",
                "D_acc_delta": "+25.0",
                "ret_tool_calls": "3.2",
                "acc_tool_calls": "12.8",
            },
        ]

        latex = render_canonical_modes_table(rows, benchmark="lakeqa", n_tasks=135)

        self.assertIn("\\textbf{Ret}", latex)
        self.assertIn("\\textbf{Acc}", latex)
        self.assertIn("\\shortstack[l]{\\texttt{gpt-5.4-}\\\\\\texttt{nano}}", latex)
        self.assertIn("Ret Tool Call and Acc Tool Call are average calls per task", latex)
        self.assertIn("Acc Tool Call excludes auxiliary ideal repair bookkeeping traces", latex)

    def test_main_ablation_table_rows_include_ret_and_acc_tool_calls(self):
        summary_rows = [
            {
                "model": "openai_gpt-5.4-nano",
                "variant": "search_i_results_i_plann_computei_k5_skills_off",
                "search_tool": "ideal",
                "search_results": "ideal",
                "agent_management": "naive",
                "computation_tool": "ideal",
                "n": 135,
                "semantic_match": 0.34,
                "D_acc": 0.46,
                "D_ret": 0.56,
                "avg_cost_usd": 0.009,
                "avg_total_cost_with_ideal_subagents_usd": 0.084,
                "avg_tool_calls_total": 15.8,
                "avg_search_calls": 3.4,
                "avg_read_calls": 11.2,
                "avg_uncached_base_cost_usd": 0.021,
                "avg_uncached_total_cost_usd": 0.187,
            },
            {
                "model": "openai_gpt-5.4-nano",
                "variant": "search_p_results_i_plani_computei_k5_skills_off",
                "search_tool": "preloaded",
                "search_results": "ideal",
                "agent_management": "ideal",
                "computation_tool": "ideal",
                "n": 135,
                "semantic_match": 0.52,
                "D_acc": 0.59,
                "D_ret": 0.0,
                "avg_cost_usd": 0.015,
                "avg_total_cost_with_ideal_subagents_usd": 0.086,
                "avg_tool_calls_total": 19.1,
                "avg_search_calls": 0.0,
                "avg_read_calls": 34.3,
                "avg_uncached_base_cost_usd": 0.024,
                "avg_uncached_total_cost_usd": 0.164,
            },
        ]

        rows = build_main_ablation_table_rows(summary_rows)

        naive_row = next(row for row in rows if row["model"] == "gpt-5.4-nano" and row["plan"] == "Naive")
        preloaded_row = next(row for row in rows if row["model"] == "gpt-5.4-nano" and row["search"] == "Preloaded")
        self.assertEqual(naive_row["avg_base_cost"], "0.02")
        self.assertEqual(naive_row["avg_base_ideal_cost"], "0.19")
        self.assertEqual(naive_row["ret_tool_calls"], "3.4")
        self.assertEqual(naive_row["acc_tool_calls"], "11.2")
        self.assertEqual(preloaded_row["D_ret"], "--")
        self.assertEqual(preloaded_row["ret_tool_calls"], "0.0")

    def test_render_main_ablation_table_has_ret_and_acc_tool_call_columns(self):
        rows = [
            {
                "model": "gpt-5.4-nano",
                "plan": "Naive",
                "search": "Ideal",
                "compute": "Ideal",
                "semantic_match": "34.0",
                "D_acc": "46.0",
                "D_ret": "56.0",
                "avg_base_cost": "0.02",
                "avg_base_ideal_cost": "0.19",
                "avg_calls": "15.8",
                "ret_tool_calls": "3.4",
                "acc_tool_calls": "11.2",
            }
        ]

        latex = render_main_ablation_table(rows, benchmark="lakeqa", n_tasks=135)

        self.assertIn("Ret Tool Call", latex)
        self.assertIn("Acc Tool Call", latex)
        self.assertIn("Base (\\$)", latex)
        self.assertIn("Base+Ideal", latex)
        self.assertIn("0.02", latex)
        self.assertIn("0.19", latex)
        self.assertIn("3.4", latex)
        self.assertIn("11.2", latex)

    def test_load_uncached_cost_summary_maps_avg_cost_by_model_and_variant(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "variant_uncached_cost_summary.csv"
            path.write_text(
                "model,variant,n,uncached_main_cost_usd,avg_uncached_total_cost_usd,avg_delta_total_cost_usd\n"
                "openai_gpt-5.4-nano,search_i_results_i_plani_computei_k5_skills_off,10,0.3500,0.1486586814,0.06471552\n",
                encoding="utf-8",
            )

            costs = load_uncached_cost_summary(path)

        self.assertEqual(
            costs[("gpt-5.4-nano", "search_i_results_i_plani_computei_k5_skills_off")]["avg_uncached_total_cost_usd"],
            0.1486586814,
        )
        self.assertAlmostEqual(
            costs[("gpt-5.4-nano", "search_i_results_i_plani_computei_k5_skills_off")]["avg_uncached_main_cost_usd"],
            0.035,
        )

    def test_main_result_rows_include_planned_cells_and_current_values(self):
        summary_rows = [
            {
                "model": "openai_gpt-5.4-nano",
                "variant": "search_i_results_i_plani_computei_k5_skills_off",
                "search_tool": "ideal",
                "search_results": "ideal",
                "agent_management": "ideal",
                "computation_tool": "ideal",
                "n": 87,
                "semantic_match": 0.4023,
                "avg_total_cost_with_ideal_subagents_usd": 0.083,
                "avg_uncached_total_cost_usd": 0.149,
                "avg_tool_calls_total": 17.8161,
                "D_acc_recall": None,
                "D_acc": None,
                "D_ret": None,
                "avg_search_calls": 3.5,
                "avg_read_calls": 10.5,
            },
            {
                "model": "openai_gpt-5.4-nano",
                "variant": "search_i_results_i_plani_k5_skills_off",
                "search_tool": "ideal",
                "search_results": "ideal",
                "agent_management": "ideal",
                "computation_tool": "standard",
                "n": 87,
                "semantic_match": 0.3333,
                "avg_total_cost_with_ideal_subagents_usd": 0.0152,
                "avg_tool_calls_total": 19.908,
                "D_acc_recall": None,
                "D_acc": None,
                "D_ret": None,
                "avg_search_calls": 4.5,
                "avg_read_calls": 11.5,
            },
        ]

        rows = build_main_result_rows(summary_rows)

        self.assertEqual(len(rows), 20)
        full_ideal = next(
            row for row in rows
            if row["model"] == "gpt-5.4-nano"
            and row["plan"] == "Ideal"
            and row["search"] == "Ideal"
            and row["compute"] == "Ideal"
            and row["results"] == "Ideal"
        )
        standard_compute = next(
            row for row in rows
            if row["model"] == "gpt-5.4-nano"
            and row["plan"] == "Ideal"
            and row["search"] == "Ideal"
            and row["compute"] == "Standard"
            and row["results"] == "Ideal"
        )
        preloaded_search = next(
            row for row in rows
            if row["model"] == "gpt-5.4-nano"
            and row["plan"] == "Ideal"
            and row["search"] == "Preloaded"
            and row["compute"] == "Ideal"
            and row["results"] == "Ideal"
        )
        pending_mini = next(row for row in rows if row["model"] == "gpt-5-mini")

        self.assertEqual(full_ideal["semantic_match"], "40.2")
        self.assertEqual(full_ideal["avg_cost"], "0.15")
        self.assertEqual(full_ideal["avg_calls"], "17.8")
        self.assertEqual(full_ideal["ret_tool_calls"], "3.5")
        self.assertEqual(full_ideal["acc_tool_calls"], "10.5")
        self.assertEqual(full_ideal["status"], "observed")
        self.assertEqual(standard_compute["semantic_match"], "33.3")
        self.assertEqual(standard_compute["compute"], "Standard")
        self.assertEqual(preloaded_search["condition"], "I-P-I-I")
        self.assertEqual(preloaded_search["status"], "pending")
        self.assertEqual(pending_mini["semantic_match"], "Pending")
        self.assertEqual(pending_mini["status"], "pending")

    def test_render_results_table_uses_standard_compute_and_pending_placeholders(self):
        rows = [
            {
                "model": "gpt-5.4-nano",
                "condition": "I-I-I-I",
                "plan": "Ideal",
                "search": "Ideal",
                "compute": "Standard",
                "results": "Ideal",
                "n": "87",
                "semantic_match": "33.3",
                "avg_cost": "0.015",
                "avg_calls": "19.9",
                "D_acc": "N/A",
                "D_ret": "N/A",
                "ret_tool_calls": "3.5",
                "acc_tool_calls": "10.5",
                "status": "observed",
            },
            {
                "model": "gpt-5-mini",
                "condition": "N-I-I-I",
                "plan": "Naive",
                "search": "Ideal",
                "compute": "Ideal",
                "results": "Ideal",
                "n": "Pending",
                "semantic_match": "Pending",
                "avg_cost": "Pending",
                "avg_calls": "Pending",
                "D_acc": "Pending",
                "D_ret": "Pending",
                "ret_tool_calls": "Pending",
                "acc_tool_calls": "Pending",
                "status": "pending",
            },
        ]

        latex = render_results_table(rows)

        self.assertIn("Standard", latex)
        self.assertIn("Pending", latex)
        self.assertIn("33.3", latex)
        self.assertIn("Ret Tool Call", latex)
        self.assertIn("Acc Tool Call", latex)
        self.assertNotIn("Compute naive", latex)


if __name__ == "__main__":
    unittest.main()
