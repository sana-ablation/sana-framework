import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from sana_analysis.report_generator.combine_answer_failure_grouped_models import (
    COMBINED_CONDITION_FIGURE_NAME,
    COMBINED_CSV_NAME,
    COMBINED_FIGURE_NAME,
    CONDITION_FIGURE_BAR_LABEL_FONTSIZE,
    CONDITION_FIGURE_BAR_WIDTH,
    CONDITION_FIGURE_CONDITION_LABEL_FONTSIZE,
    CONDITION_FIGURE_LEGEND_LABELS,
    CONDITION_FIGURE_LEGEND_Y_ANCHOR,
    CONDITION_FIGURE_LEGEND_FONTSIZE,
    CONDITION_FIGURE_LEGEND_COLUMNS,
    CONDITION_FIGURE_MODEL_LABEL_FONTSIZE,
    CONDITION_FIGURE_SEGMENT_LABEL_MIN_FRACTION,
    CONDITION_FIGURE_TOTAL_FONTSIZE,
    CONDITION_FIGURE_YMAX,
    CONDITION_FIGURE_YLABEL_FONTSIZE,
    CONDITION_FIGURE_YTICK_FONTSIZE,
    MODEL_GROUP_AXIS_FONTSIZE,
    MODEL_GROUP_LABEL_FONTSIZE,
    MODEL_GROUP_LEGEND_FONTSIZE,
    MODEL_GROUP_TITLE_FONTSIZE,
    MODEL_GROUP_VALUE_FONTSIZE,
    combine_answer_failures,
    condition_group_counts,
    discover_source_files,
    figure_group_for_type,
)


class TestCombineAnswerFailureGroupedModels(unittest.TestCase):
    def test_model_group_figure_uses_paper_readable_type(self):
        self.assertGreaterEqual(MODEL_GROUP_TITLE_FONTSIZE, 20.0)
        self.assertGreaterEqual(MODEL_GROUP_AXIS_FONTSIZE, 16.0)
        self.assertGreaterEqual(MODEL_GROUP_LABEL_FONTSIZE, 15.0)
        self.assertGreaterEqual(MODEL_GROUP_VALUE_FONTSIZE, 14.0)
        self.assertGreaterEqual(MODEL_GROUP_LEGEND_FONTSIZE, 14.0)

    def test_condition_group_figure_uses_large_two_column_type(self):
        self.assertGreaterEqual(CONDITION_FIGURE_BAR_LABEL_FONTSIZE, 14.0)
        self.assertGreaterEqual(CONDITION_FIGURE_TOTAL_FONTSIZE, 14.0)
        self.assertGreaterEqual(CONDITION_FIGURE_MODEL_LABEL_FONTSIZE, 14.0)
        self.assertGreaterEqual(CONDITION_FIGURE_CONDITION_LABEL_FONTSIZE, 17.0)
        self.assertGreaterEqual(CONDITION_FIGURE_YLABEL_FONTSIZE, 17.0)
        self.assertGreaterEqual(CONDITION_FIGURE_YTICK_FONTSIZE, 15.0)
        self.assertGreaterEqual(CONDITION_FIGURE_LEGEND_FONTSIZE, 14.0)

    def test_condition_group_figure_uses_wide_bars_and_compact_legend_labels(self):
        self.assertGreaterEqual(CONDITION_FIGURE_BAR_WIDTH, 0.74)
        self.assertLessEqual(CONDITION_FIGURE_SEGMENT_LABEL_MIN_FRACTION, 0.06)
        self.assertNotIn("failures", " ".join(CONDITION_FIGURE_LEGEND_LABELS.values()).lower())

    def test_condition_group_figure_caps_y_axis_and_keeps_legend_in_header_band(self):
        self.assertGreaterEqual(CONDITION_FIGURE_YMAX, 250)
        self.assertEqual(CONDITION_FIGURE_LEGEND_COLUMNS, 2)

    def _write_events(
        self,
        root: Path,
        model: str,
        variant: str,
        *,
        rows: list[dict[str, str]],
        eval_rows: list[dict[str, str]] | None = None,
    ) -> Path:
        folder = root / "modes" / model / variant
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / "answer_failure_events.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "task_id",
                    "model_variant",
                    "mode_variant",
                    "event_index",
                    "answer_failure_type",
                    "answer_failure_subtype",
                    "failure_stage",
                    "failure_summary",
                    "failure_evidence",
                    "confidence",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
        with (folder / "eval_results.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "task_id",
                    "semantic_match",
                    "answer_failure_validation_status",
                    "answer_failure_model_validation_status",
                ],
            )
            writer.writeheader()
            writer.writerows(eval_rows if eval_rows is not None else [])
        return folder / "answer_failure_events.csv"

    def test_figure_group_for_type_maps_expected_buckets(self):
        self.assertEqual(
            figure_group_for_type("wrong_source_or_dataset"),
            "Wrong source target failures",
        )
        self.assertEqual(
            figure_group_for_type("query_execution_error_loop"),
            "Turn-waste failures",
        )
        self.assertEqual(
            figure_group_for_type("schema_or_shape_inspection_loop"),
            "Turn-waste failures",
        )
        self.assertEqual(
            figure_group_for_type("wrong_scope_or_filter"),
            "Execution/computation failures",
        )
        self.assertEqual(
            figure_group_for_type("computation_or_aggregation_error"),
            "Execution/computation failures",
        )
        self.assertEqual(figure_group_for_type("new_label"), "new_label")

    def test_discover_source_files_reads_model_and_mode_from_modes_layout(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_events(
                root,
                "openai_gpt-5-mini",
                "search_i_results_i_plani_computei_k5_skills_off",
                rows=[],
            )

            sources = discover_source_files(root)

            self.assertEqual(len(sources), 1)
            self.assertEqual(sources[0].model_variant, "openai_gpt-5-mini")
            self.assertEqual(sources[0].mode_variant, "search_i_results_i_plani_computei_k5_skills_off")

    def test_combine_answer_failures_writes_csv_report_and_figures(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_root = root / "results_semantic_answer_failures"
            output_root = root / "combined"
            self._write_events(
                source_root,
                "openai_gpt-5-mini",
                "search_i_results_i_plani_computei_k5_skills_off",
                rows=[
                    {
                        "task_id": "tasks_mini/k-2-d-2/task_1.json",
                        "model_variant": "openai_gpt-5-mini",
                        "mode_variant": "search_i_results_i_plani_computei_k5_skills_off",
                        "event_index": "1",
                        "answer_failure_type": "wrong_source_or_dataset",
                        "answer_failure_subtype": "wrong_dataset",
                        "failure_stage": "retrieval",
                        "failure_summary": "wrong source",
                        "failure_evidence": "evidence",
                        "confidence": "0.8",
                    }
                ],
                eval_rows=[
                    {
                        "task_id": "tasks_mini/k-2-d-2/task_1.json",
                        "semantic_match": "0",
                        "answer_failure_validation_status": "valid",
                        "answer_failure_model_validation_status": "pass",
                    }
                ],
            )
            self._write_events(
                source_root,
                "openai_gpt-5.4-nano",
                "search_d_results_i_plani_computei_k5_skills_off",
                rows=[
                    {
                        "task_id": "tasks_mini/k-2-d-2/task_2.json",
                        "model_variant": "openai_gpt-5.4-nano",
                        "mode_variant": "search_d_results_i_plani_computei_k5_skills_off",
                        "event_index": "1",
                        "answer_failure_type": "computation_or_aggregation_error",
                        "answer_failure_subtype": "wrong_denominator",
                        "failure_stage": "execution",
                        "failure_summary": "wrong math",
                        "failure_evidence": "evidence",
                        "confidence": "0.8",
                    }
                ],
                eval_rows=[
                    {
                        "task_id": "tasks_mini/k-2-d-2/task_2.json",
                        "semantic_match": "0",
                        "answer_failure_validation_status": "valid",
                        "answer_failure_model_validation_status": "pass",
                    }
                ],
            )
            self._write_events(
                source_root,
                "openai_gpt-5-mini",
                "search_p_results_i_plani_computei_k5_skills_off",
                rows=[
                    {
                        "task_id": "tasks_mini/k-2-d-2/task_3.json",
                        "model_variant": "openai_gpt-5-mini",
                        "mode_variant": "search_p_results_i_plani_computei_k5_skills_off",
                        "event_index": "1",
                        "answer_failure_type": "evidence_available_answer_error",
                        "answer_failure_subtype": "wrong_final_value",
                        "failure_stage": "finalization",
                        "failure_summary": "wrong final answer",
                        "failure_evidence": "evidence",
                        "confidence": "0.8",
                    }
                ],
                eval_rows=[
                    {
                        "task_id": "tasks_mini/k-2-d-2/task_3.json",
                        "semantic_match": "0",
                        "answer_failure_validation_status": "valid",
                        "answer_failure_model_validation_status": "pass",
                    }
                ],
            )

            outputs = combine_answer_failures(source_root=source_root, output_root=output_root)

            self.assertTrue((output_root / COMBINED_CSV_NAME).exists())
            self.assertTrue((output_root / "combined_answer_failure_report.md").exists())
            self.assertTrue((output_root / COMBINED_FIGURE_NAME).exists())
            self.assertTrue((output_root / COMBINED_CONDITION_FIGURE_NAME).exists())
            with outputs["csv_path"].open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0]["answer_failure_figure_group"], "Wrong source target failures")
            active_conditions, ordered_groups, counts = condition_group_counts(rows)
            self.assertIn(("Pneuma Hybrid", "search_d_results_i_plani_computei_k5_skills_off"), active_conditions)
            self.assertLess(
                [label for label, _ in active_conditions].index("Ideal"),
                [label for label, _ in active_conditions].index("Preloaded"),
            )
            self.assertIn("Execution/computation failures", ordered_groups)
            self.assertEqual(
                counts[("search_d_results_i_plani_computei_k5_skills_off", "openai_gpt-5.4-nano")][
                    "Execution/computation failures"
                ],
                1,
            )

    def test_combine_answer_failures_drops_untrusted_rows_when_validation_is_available(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_root = root / "results_semantic_answer_failures"
            self._write_events(
                source_root,
                "openai_gpt-5-mini",
                "search_i_results_i_plani_computei_k5_skills_off",
                rows=[
                    {
                        "task_id": "tasks_mini/k-2-d-2/task_1.json",
                        "model_variant": "openai_gpt-5-mini",
                        "mode_variant": "search_i_results_i_plani_computei_k5_skills_off",
                        "event_index": "1",
                        "answer_failure_type": "wrong_source_or_dataset",
                        "answer_failure_subtype": "wrong_dataset",
                        "failure_stage": "retrieval",
                        "failure_summary": "wrong source",
                        "failure_evidence": "evidence",
                        "confidence": "0.8",
                    }
                ],
                eval_rows=[
                    {
                        "task_id": "tasks_mini/k-2-d-2/task_1.json",
                        "semantic_match": "0",
                        "answer_failure_validation_status": "invalid",
                        "answer_failure_model_validation_status": "pass",
                    }
                ],
            )

            outputs = combine_answer_failures(source_root=source_root, output_root=root / "combined")

            with outputs["csv_path"].open(newline="") as handle:
                self.assertEqual(list(csv.DictReader(handle)), [])


if __name__ == "__main__":
    unittest.main()
