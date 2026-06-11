import json
import unittest
from pathlib import Path


class TestTaskQualityAuditorExamples(unittest.TestCase):
    def test_locked_example_expectations_are_present(self):
        fixtures_path = (
            Path(__file__).resolve().parent.parent
            / ".agents"
            / "skills"
            / "task-quality-auditor"
            / "references"
            / "example_fixtures.json"
        )
        payload = json.loads(fixtures_path.read_text())
        examples = {entry["task_id"]: entry for entry in payload["examples"]}

        self.assertEqual(
            examples["tasks_mini/k-5-d-3/task_7.json"]["overall_recommendation"],
            "core_keep",
        )
        self.assertEqual(
            examples["tasks_mini/k-5-d-3/task_7.json"]["quality_bucket"],
            "high",
        )
        self.assertEqual(
            examples["tasks_mini/k-5-d-4/task_2.json"]["derivability_bucket"],
            "clean",
        )
        self.assertEqual(
            examples["tasks_mini/k-5-d-4/task_2.json"]["difficulty_bucket"],
            "core",
        )
        self.assertEqual(
            examples["tasks_mini/k-5-d-3/task_1.json"]["overall_recommendation"],
            "revise_or_drop",
        )
        self.assertIn(
            "internal_inconsistency",
            examples["tasks_mini/k-5-d-3/task_1.json"]["required_issue_labels"],
        )
        self.assertEqual(
            examples["tasks_mini/k-5-d-4/task_10.json"]["derivability_bucket"],
            "broken",
        )
        self.assertIn(
            "internal_inconsistency",
            examples["tasks_mini/k-5-d-4/task_10.json"]["required_issue_labels"],
        )


if __name__ == "__main__":
    unittest.main()
