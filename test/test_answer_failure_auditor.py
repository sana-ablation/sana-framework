import csv
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sana_analysis.running_analysis.answer_failure_validation import (
    VALIDATION_NOTES_FIELD,
    VALIDATION_STATUS_FIELD,
    validate_answer_failure_root,
    validate_answer_failure_row,
)
from sana_analysis.report_generator.build_answer_failure_report import build_answer_failure_report


class TestAnswerFailureAuditor(unittest.TestCase):
    def _write_csv(self, path: Path, fieldnames: list[str], rows: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _write_log(self, path: Path, lines: list[str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n")

    def test_validate_row_rejects_bad_type_and_impossible_turn(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            log_path = root / "task.log"
            self._write_log(
                log_path,
                [
                    "--- Turn 1 (elapsed: 0.0s) ---",
                    'Executing: search_ideal({"query": "schools"})',
                ],
            )
            result = validate_answer_failure_row(
                {
                    "task_id": "tasks_mini/k-1-d-1/task_1.json",
                    "semantic_bucket": "semantic_incorrect",
                    "semantic_match": "0",
                },
                [
                    {
                        "answer_failure_type": "wrong_dataset",
                        "answer_failure_subtype": "",
                        "failure_stage": "source_selection",
                        "failure_summary": "Chose the wrong dataset.",
                        "failure_evidence": 'Turn 3 | Executing: query_file({"dataset_id": "wrong"})',
                        "confidence": "high",
                    }
                ],
                log_path,
            )

            self.assertEqual(result["status"], "invalid")
            self.assertIn("invalid answer_failure_type `wrong_dataset`", result["notes"])
            self.assertIn("references turns beyond log max turn 1", result["notes"])

    def test_validate_root_rewrites_summary_columns_and_preserves_correct_rows(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_root = root / "results_semantic_answer_failures"
            logs_root = root / "logs" / "modes"
            model = "openai_gpt-test"
            mode = "search_i_results_i_plani_k5"
            file_dir = source_root / "modes" / model / mode
            task_id = "tasks_mini/k-1-d-1/task_1.json"

            self._write_csv(
                file_dir / "eval_results.csv",
                [
                    "task_id",
                    "semantic_bucket",
                    "semantic_match",
                    "answer_failure_event_count",
                    "answer_failure_model_validation_status",
                ],
                [
                    {
                        "task_id": task_id,
                        "semantic_bucket": "semantic_incorrect",
                        "semantic_match": "0",
                        "answer_failure_event_count": "1",
                        "answer_failure_model_validation_status": "pass",
                    },
                    {
                        "task_id": "tasks_mini/k-1-d-1/task_2.json",
                        "semantic_bucket": "semantic_correct",
                        "semantic_match": "1",
                        "answer_failure_event_count": "",
                        "answer_failure_model_validation_status": "",
                    },
                ],
            )
            self._write_csv(
                file_dir / "answer_failure_events.csv",
                [
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
                [
                    {
                        "task_id": task_id,
                        "model_variant": model,
                        "mode_variant": mode,
                        "event_index": "1",
                        "answer_failure_type": "wrong_source_or_dataset",
                        "answer_failure_subtype": "wrong_dataset",
                        "failure_stage": "source_selection",
                        "failure_summary": "The run queried a wrong dataset.",
                        "failure_evidence": 'Turn 2 | Executing: query_file({"dataset_id": "wrong", "sql": "SELECT"})',
                        "confidence": "high",
                    },
                    {
                        "task_id": task_id,
                        "model_variant": model,
                        "mode_variant": mode,
                        "event_index": "2",
                        "answer_failure_type": "computation_or_aggregation_error",
                        "answer_failure_subtype": "wrong_aggregation",
                        "failure_stage": "computation",
                        "failure_summary": "The run aggregated the wrong rows.",
                        "failure_evidence": 'Turn 3 | Tool result: {"rows": [[1]]}',
                        "confidence": "medium",
                    },
                ],
            )
            self._write_log(
                logs_root / model / mode / "tasks_mini" / "k-1-d-1" / "task_1.log",
                [
                    "--- Turn 2 (elapsed: 1.0s) ---",
                    'Executing: query_file({"dataset_id": "wrong", "sql": "SELECT"})',
                    "--- Turn 3 (elapsed: 2.0s) ---",
                    'Tool result: {"rows": [[1]]}',
                ],
            )

            outputs = validate_answer_failure_root(source_root=source_root, logs_dir=logs_root, rewrite=True)
            self.assertEqual(len(outputs["invalid_rows"]), 1)

            with (file_dir / "eval_results.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(rows[0]["answer_failure_event_count"], "2")
            self.assertEqual(rows[0]["answer_failure_types"], "wrong_source_or_dataset; computation_or_aggregation_error")
            self.assertEqual(rows[0][VALIDATION_STATUS_FIELD], "invalid")
            self.assertIn("answer_failure_event_count=1", rows[0][VALIDATION_NOTES_FIELD])
            self.assertTrue(
                all(value == "" for key, value in rows[1].items() if key.startswith("answer_failure"))
            )

    def test_missing_log_with_ungroundable_event_is_valid_with_warning(self):
        result = validate_answer_failure_row(
            {
                "task_id": "tasks_mini/k-1-d-1/task_1.json",
                "semantic_bucket": "semantic_incorrect",
                "semantic_match": "0",
            },
            [
                {
                    "answer_failure_type": "ungroundable",
                    "answer_failure_subtype": "missing_or_malformed_log",
                    "failure_stage": "validation",
                    "failure_summary": "The raw log is missing, so no grounded diagnosis is possible.",
                    "failure_evidence": "matching raw log missing from logs root",
                    "confidence": "high",
                }
            ],
            Path("/definitely/missing/task_1.log"),
        )

        self.assertEqual(result["status"], "valid_with_warnings")
        self.assertIn("row is marked ungroundable", result["notes"])

    def test_report_counts_multievent_rows_without_collapsing_to_one_failure(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_root = root / "results_semantic_answer_failures"
            logs_root = root / "logs" / "modes"
            model = "openai_gpt-test"
            mode = "search_i_results_i_plani_k5"
            file_dir = source_root / "modes" / model / mode
            task_id = "tasks_mini/k-2-d-2/task_3.json"

            self._write_csv(
                file_dir / "eval_results.csv",
                [
                    "task_id",
                    "semantic_bucket",
                    "semantic_match",
                    "answer_failure_validation_status",
                    "answer_failure_model_validation_status",
                ],
                [
                    {
                        "task_id": task_id,
                        "semantic_bucket": "semantic_incorrect",
                        "semantic_match": "0",
                        "answer_failure_validation_status": "valid",
                        "answer_failure_model_validation_status": "repaired_pass",
                    }
                ],
            )
            self._write_csv(
                file_dir / "answer_failure_events.csv",
                [
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
                [
                    {
                        "task_id": task_id,
                        "model_variant": model,
                        "mode_variant": mode,
                        "event_index": "1",
                        "answer_failure_type": "wrong_scope_or_filter",
                        "answer_failure_subtype": "wrong_year",
                        "failure_stage": "source_selection",
                        "failure_summary": "Used the wrong year.",
                        "failure_evidence": 'Turn 4 | Executing: search_ideal({"query": "2018"})',
                        "confidence": "high",
                    },
                    {
                        "task_id": task_id,
                        "model_variant": model,
                        "mode_variant": mode,
                        "event_index": "2",
                        "answer_failure_type": "evidence_available_answer_error",
                        "answer_failure_subtype": "submitted_partial_answer",
                        "failure_stage": "finalization",
                        "failure_summary": "Submitted a partial answer.",
                        "failure_evidence": 'Turn 8 | Executing: submit_answer({"answer": "unknown"})',
                        "confidence": "medium",
                    },
                ],
            )

            outputs = build_answer_failure_report(
                source_root=source_root,
                events_path=file_dir / "answer_failure_events.csv",
                logs_dir=logs_root,
            )
            report_text = (file_dir / "answer_failure_report.md").read_text()

            self.assertIn("wrong_scope_or_filter", report_text)
            self.assertIn("evidence_available_answer_error", report_text)
            self.assertIn("evidence_available_answer_error; wrong_scope_or_filter", report_text)
            self.assertEqual(outputs["event_count"], 2)

    def test_report_links_kramabench_plan_paths(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_root = root / "results-kramabench_semantic_answer_failures"
            logs_root = root / "log-kramabench" / "modes"
            model = "openai_gpt-test"
            mode = "search_i_results_i_plani_k5"
            file_dir = source_root / "modes" / model / mode
            task_id = "tasks-mini-kramabench/k-2-d-1-s-1/task_1.json"
            (root / task_id).parent.mkdir(parents=True, exist_ok=True)
            (root / task_id).write_text("{}\n")
            plan_path = root / "plans-mini-kramabench/k-2-d-1-s-1/task_1.json"
            plan_path.parent.mkdir(parents=True, exist_ok=True)
            plan_path.write_text("{}\n")

            self._write_csv(
                file_dir / "eval_results.csv",
                [
                    "task_id",
                    "semantic_bucket",
                    "semantic_match",
                    "answer_failure_validation_status",
                    "answer_failure_model_validation_status",
                ],
                [
                    {
                        "task_id": task_id,
                        "semantic_bucket": "semantic_incorrect",
                        "semantic_match": "0",
                        "answer_failure_validation_status": "valid",
                        "answer_failure_model_validation_status": "pass",
                    }
                ],
            )
            self._write_csv(
                file_dir / "answer_failure_events.csv",
                [
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
                [
                    {
                        "task_id": task_id,
                        "model_variant": model,
                        "mode_variant": mode,
                        "event_index": "1",
                        "answer_failure_type": "wrong_source_or_dataset",
                        "answer_failure_subtype": "wrong_table",
                        "failure_stage": "source_selection",
                        "failure_summary": "Used the wrong table.",
                        "failure_evidence": 'Turn 1 | Executing: search_ideal({"query": "wrong"})',
                        "confidence": "high",
                    }
                ],
            )

            build_answer_failure_report(
                source_root=source_root,
                events_path=file_dir / "answer_failure_events.csv",
                logs_dir=logs_root,
            )
            report_text = (file_dir / "answer_failure_report.md").read_text()

            self.assertIn("plans-mini-kramabench", report_text)

    def test_validate_row_accepts_split_incomplete_evidence_types(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            log_path = root / "task.log"
            self._write_log(
                log_path,
                [
                    "--- Turn 4 (elapsed: 10.0s) ---",
                    'Executing: submit_answer({"answer": "unable to determine"})',
                    "--- Turn 9 (elapsed: 590.0s) ---",
                    "Tool result: Tool call cancelled. Tool limit reached (30/30 calls used).",
                ],
            )

            for failure_type, evidence in [
                (
                    "incomplete_evidence_early_answer",
                    'Turn 4 | Executing: submit_answer({"answer": "unable to determine"})',
                ),
                (
                    "incomplete_evidence_budget_exhausted",
                    "Turn 9 | Tool result: Tool call cancelled. Tool limit reached (30/30 calls used).",
                ),
            ]:
                result = validate_answer_failure_row(
                    {
                        "task_id": "tasks_mini/k-1-d-1/task_1.json",
                        "semantic_bucket": "answer_unknown_blank",
                        "semantic_match": "0",
                    },
                    [
                        {
                            "answer_failure_type": failure_type,
                            "answer_failure_subtype": "",
                            "failure_stage": "submission",
                            "failure_summary": "The run answered without complete evidence.",
                            "failure_evidence": evidence,
                            "confidence": "high",
                        }
                    ],
                    log_path,
                )
                self.assertEqual(result["status"], "valid")


if __name__ == "__main__":
    unittest.main()
