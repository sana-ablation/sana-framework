import csv
import importlib.util
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


_SCRIPT_DIR = (
    Path(__file__).resolve().parent.parent
    / ".agents"
    / "skills"
    / "task-quality-auditor"
    / "scripts"
)
sys.path.insert(0, str(_SCRIPT_DIR))


def _load_module(filename: str, module_name: str):
    module_path = _SCRIPT_DIR / filename
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


summary_builder = _load_module("build_task_quality_summary.py", "task_quality_summary")


class TestTaskQualitySummary(unittest.TestCase):
    def _write_task(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n")

    def _write_audit(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n")

    def test_build_summary_writes_csv_and_markdown(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_root = root / "tasks_mini"
            output_root = root / "tasks_mini_quality"

            self._write_task(source_root / "k-5-d-3" / "task_7.json")
            self._write_task(source_root / "k-5-d-4" / "task_10.json")

            self._write_audit(
                output_root / "k-5-d-3" / "task_7.json",
                {
                    "task_id": "tasks_mini/k-5-d-3/task_7.json",
                    "heuristic_flags": {
                        "question_text_anomaly": False,
                        "question_text_anomaly_notes": [],
                        "has_explicit_answer_placeholders": False,
                    },
                    "quality_bucket": "high",
                    "derivability_bucket": "clean",
                    "difficulty_bucket": "core",
                    "overall_recommendation": "core_keep",
                    "issue_labels": [],
                    "rationale": "Clean and benchmark-worthy.",
                    "suggested_fix": "No change needed.",
                },
            )
            self._write_audit(
                output_root / "k-5-d-4" / "task_10.json",
                {
                    "task_id": "tasks_mini/k-5-d-4/task_10.json",
                    "heuristic_flags": {
                        "question_text_anomaly": True,
                        "question_text_anomaly_notes": ["awkward least increase wording"],
                        "has_explicit_answer_placeholders": False,
                    },
                    "quality_bucket": "low",
                    "derivability_bucket": "broken",
                    "difficulty_bucket": "unclear",
                    "overall_recommendation": "revise_or_drop",
                    "issue_labels": ["internal_inconsistency", "awkward_wording"],
                    "rationale": "Reasoning chain year is inconsistent.",
                    "suggested_fix": "Repair the contradiction or drop the task.",
                },
            )

            outputs = summary_builder.build_summary(
                source_root=source_root,
                output_root=output_root,
            )

            self.assertEqual(len(outputs["rows"]), 2)
            self.assertTrue(outputs["csv_path"].exists())
            self.assertTrue(outputs["markdown_path"].exists())

            with outputs["csv_path"].open(newline="") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(rows[0]["task_id"], "tasks_mini/k-5-d-3/task_7.json")
            self.assertEqual(rows[0]["overall_recommendation"], "core_keep")
            self.assertEqual(rows[1]["heuristic_flags"], "question_text_anomaly")
            self.assertEqual(
                rows[1]["issue_labels"],
                "internal_inconsistency|awkward_wording",
            )

            report_text = outputs["markdown_path"].read_text()
            self.assertIn("Audited tasks: 2", report_text)
            self.assertIn("Source task count: 2", report_text)
            self.assertIn("`tasks_mini/k-5-d-3/task_7.json`", report_text)
            self.assertIn("internal_inconsistency: 1", report_text)


if __name__ == "__main__":
    unittest.main()
