import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from sana_analysis.report_generator.migrate_answer_failure_taxonomy import (
    AMBIGUOUS_OLD_TYPES,
    apply_migration,
    export_review_packet,
    run_one_shot_migration,
)


class TestAnswerFailureTaxonomyMigrator(unittest.TestCase):
    def _write_csv(self, path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _write_answer_failure_folder(self, root: Path) -> Path:
        folder = root / "modes" / "model" / "mode"
        task_id = "tasks_mini/k-1-d-1/task_1.json"
        self._write_csv(
            folder / "answer_failure_events.csv",
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
                    "model_variant": "model",
                    "mode_variant": "mode",
                    "event_index": "1",
                    "answer_failure_type": "wrong_source_or_scope",
                    "answer_failure_subtype": "wrong year",
                    "failure_stage": "source_selection",
                    "failure_summary": "The query used 2022 instead of 2023.",
                    "failure_evidence": "Turn 2 | query used year=2022",
                    "confidence": "high",
                },
                {
                    "task_id": task_id,
                    "model_variant": "model",
                    "mode_variant": "mode",
                    "event_index": "2",
                    "answer_failure_type": "incomplete_evidence_not_enough_turns",
                    "answer_failure_subtype": "tool budget",
                    "failure_stage": "data_access",
                    "failure_summary": "The run hit the tool limit.",
                    "failure_evidence": "Turn 9 | Tool limit reached.",
                    "confidence": "medium",
                },
            ],
        )
        self._write_csv(
            folder / "eval_results.csv",
            [
                "task_id",
                "semantic_bucket",
                "semantic_match",
                "answer_failure_summary",
                "answer_failure_types",
                "answer_failure_subtypes",
                "answer_failure_event_count",
                "answer_failure_evidence",
            ],
            [
                {
                    "task_id": task_id,
                    "semantic_bucket": "semantic_incorrect",
                    "semantic_match": "0",
                    "answer_failure_summary": "old summary",
                    "answer_failure_types": "wrong_source_or_scope; incomplete_evidence_not_enough_turns",
                    "answer_failure_subtypes": "wrong year; tool budget",
                    "answer_failure_event_count": "2",
                    "answer_failure_evidence": "old evidence",
                }
            ],
        )
        return folder

    def test_export_review_packet_includes_ambiguous_old_rows(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "results_semantic_answer_failures"
            self._write_answer_failure_folder(root)
            review_path = Path(tmpdir) / "review.csv"

            summary = export_review_packet(root, review_path)

            self.assertEqual(summary["rows_written"], 1)
            with review_path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["old_type"], "wrong_source_or_scope")
            self.assertIn(rows[0]["old_type"], AMBIGUOUS_OLD_TYPES)
            self.assertEqual(rows[0]["source_events_path"], "modes/model/mode/answer_failure_events.csv")
            self.assertEqual(rows[0]["suggested_new_type"], "")

    def test_apply_migration_uses_decisions_and_rewrites_eval_summary(self):
        with TemporaryDirectory() as tmpdir:
            source_root = Path(tmpdir) / "results_semantic_answer_failures"
            output_root = Path(tmpdir) / "results_semantic_answer_failures_taxonomy_v3"
            self._write_answer_failure_folder(source_root)
            decisions_path = Path(tmpdir) / "decisions.csv"
            self._write_csv(
                decisions_path,
                [
                    "source_events_path",
                    "task_id",
                    "event_index",
                    "old_type",
                    "new_type",
                    "new_subtype",
                    "review_required",
                    "confidence",
                    "rationale",
                ],
                [
                    {
                        "source_events_path": "modes/model/mode/answer_failure_events.csv",
                        "task_id": "tasks_mini/k-1-d-1/task_1.json",
                        "event_index": "1",
                        "old_type": "wrong_source_or_scope",
                        "new_type": "wrong_scope_or_filter",
                        "new_subtype": "wrong_year",
                        "review_required": "false",
                        "confidence": "high",
                        "rationale": "Same source, wrong year predicate.",
                    }
                ],
            )

            summary = apply_migration(source_root, output_root, decisions_path=decisions_path)

            self.assertEqual(summary["events_changed"], 2)
            self.assertEqual(summary["eval_files_rewritten"], 1)
            with (output_root / "modes/model/mode/answer_failure_events.csv").open(newline="") as handle:
                events = list(csv.DictReader(handle))
            self.assertEqual(events[0]["answer_failure_type"], "wrong_scope_or_filter")
            self.assertEqual(events[0]["answer_failure_subtype"], "wrong_year")
            self.assertEqual(events[1]["answer_failure_type"], "incomplete_evidence_budget_exhausted")

            with (output_root / "modes/model/mode/eval_results.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(
                rows[0]["answer_failure_types"],
                "wrong_scope_or_filter; incomplete_evidence_budget_exhausted",
            )
            self.assertEqual(rows[0]["answer_failure_subtypes"], "wrong_year; tool budget")
            self.assertEqual(rows[0]["answer_failure_event_count"], "2")
            self.assertIn("The query used 2022 instead of 2023.", rows[0]["answer_failure_summary"])

    def test_apply_migration_rejects_ambiguous_rows_without_decision(self):
        with TemporaryDirectory() as tmpdir:
            source_root = Path(tmpdir) / "results_semantic_answer_failures"
            output_root = Path(tmpdir) / "results_semantic_answer_failures_taxonomy_v3"
            self._write_answer_failure_folder(source_root)

            with self.assertRaisesRegex(ValueError, "missing migration decision"):
                apply_migration(source_root, output_root)

    def test_run_one_shot_migration_generates_decisions_and_applies_copy(self):
        with TemporaryDirectory() as tmpdir:
            source_root = Path(tmpdir) / "results_semantic_answer_failures"
            output_root = Path(tmpdir) / "results_semantic_answer_failures_taxonomy_v3"
            decisions_path = Path(tmpdir) / "decisions.csv"
            self._write_answer_failure_folder(source_root)

            summary = run_one_shot_migration(
                source_root,
                output_root,
                decisions_path=decisions_path,
            )

            self.assertEqual(summary["auto_decisions_written"], 1)
            self.assertEqual(summary["events_changed"], 2)
            with decisions_path.open(newline="") as handle:
                decisions = list(csv.DictReader(handle))
            self.assertEqual(decisions[0]["new_type"], "wrong_scope_or_filter")
            self.assertEqual(decisions[0]["review_required"], "false")
            with (output_root / "modes/model/mode/answer_failure_events.csv").open(newline="") as handle:
                events = list(csv.DictReader(handle))
            self.assertEqual(events[0]["answer_failure_type"], "wrong_scope_or_filter")


if __name__ == "__main__":
    unittest.main()
