import importlib.util
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


_SCRIPT_DIR = (
    Path(__file__).resolve().parent.parent
    / ".agents"
    / "skills"
    / "author-ideal-computations"
    / "scripts"
)


def _load_module(filename: str, module_name: str):
    module_path = _SCRIPT_DIR / filename
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


seed_records = _load_module(
    "seed_ideal_computation_records.py",
    "author_ideal_computation_seed",
)
check_records = _load_module(
    "check_ideal_computation_records.py",
    "author_ideal_computation_check",
)


class TestAuthorIdealComputationScripts(unittest.TestCase):
    def _write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n")

    def _base_task_payload(self) -> dict:
        return {
            "question": "Q?",
            "datasets_used": ["example-dataset", "Plain_Page"],
            "answer": "7",
            "nodes": {
                "1": {
                    "source": "datagov/example-dataset/files/rows.txt",
                    "subquestion": "Sum the values.",
                    "fact": (
                        "import pandas as pd\n"
                        "df = pd.read_csv('datagov/example-dataset/files/rows.txt')\n"
                        "answer = str(int(df['value'].sum()))\n"
                    ),
                    "answer": "7",
                },
                "2": {
                    "source": "wikipedia/Plain_Page/content.txt",
                    "subquestion": "Read the prose answer.",
                    "fact": "According to the article, the answer is Plain.",
                    "answer": "Plain",
                },
            },
        }

    def test_plan_path_maps_task_roots(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "plans_mini").mkdir()
            (root / "plans_core_quality").mkdir()
            (root / "tasks_mini").mkdir()
            (root / "tasks_core").mkdir()
            (root / "tasks_core_quality").mkdir()

            mini = root / "tasks_mini" / "k-1-d-1" / "task_1.json"
            core = root / "tasks_core" / "k-1-d-1" / "task_2.json"
            quality = root / "tasks_core_quality" / "k-1-d-1" / "task_3.json"

            self.assertEqual(
                seed_records.plan_path_for_task(mini),
                root / "plans_mini" / "k-1-d-1" / "task_1.json",
            )
            self.assertEqual(
                seed_records.plan_path_for_task(core),
                root / "plans_mini" / "tasks_core" / "k-1-d-1" / "task_2.json",
            )
            self.assertEqual(
                seed_records.plan_path_for_task(quality),
                root / "plans_core_quality" / "k-1-d-1" / "task_3.json",
            )

    def test_seed_records_preserves_executable_fact_and_skips_prose_fact(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            task_path = root / "tasks_mini" / "k-1-d-1" / "task_1.json"
            plan_path = root / "plans_mini" / "k-1-d-1" / "task_1.json"
            self._write_json(task_path, self._base_task_payload())
            self._write_json(
                plan_path,
                {
                    "dataset_sequence": ["example-dataset", "Plain_Page"],
                    "source_sequence": [
                        "datagov/example-dataset/files/rows.txt",
                        "wikipedia/Plain_Page/content.txt",
                    ],
                    "reasoning_chain_text": ["1. Compute then read."],
                },
            )

            result_plan_path, records = seed_records.build_ideal_code_records(task_path)

            self.assertEqual(result_plan_path, plan_path)
            self.assertEqual(len(records), 1)
            record = records[0]
            node = self._base_task_payload()["nodes"]["1"]
            self.assertEqual(record["node_id"], "1")
            self.assertEqual(record["dataset_id"], "example-dataset")
            self.assertEqual(record["source"], node["source"])
            self.assertEqual(record["intent"], node["subquestion"])
            self.assertEqual(record["code"], node["fact"])
            self.assertEqual(record["answer"], node["answer"])

    def test_check_records_flags_missing_query_for_executable_node(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            task_path = root / "tasks_mini" / "k-1-d-1" / "task_1.json"
            plan_path = root / "plans_mini" / "k-1-d-1" / "task_1.json"
            task_payload = self._base_task_payload()
            self._write_json(task_path, task_payload)
            self._write_json(
                plan_path,
                {
                    "dataset_sequence": ["example-dataset", "Plain_Page"],
                    "source_sequence": [
                        "datagov/example-dataset/files/rows.txt",
                        "wikipedia/Plain_Page/content.txt",
                    ],
                    "reasoning_chain_text": ["1. Compute then read."],
                },
            )
            code_records = seed_records.build_ideal_code_records(task_path)[1]
            self._write_json(
                plan_path,
                {
                    "dataset_sequence": ["example-dataset", "Plain_Page"],
                    "source_sequence": [
                        "datagov/example-dataset/files/rows.txt",
                        "wikipedia/Plain_Page/content.txt",
                    ],
                    "reasoning_chain_text": ["1. Compute then read."],
                    "ideal_code": code_records,
                },
            )

            result = check_records.evaluate_plan(plan_path)

            self.assertEqual(result["status"], "needs_revision")
            self.assertIn("missing_ideal_query_record", {issue["code"] for issue in result["issues"]})

    def test_check_records_flags_exact_code_mismatch(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            task_path = root / "tasks_mini" / "k-1-d-1" / "task_1.json"
            plan_path = root / "plans_mini" / "k-1-d-1" / "task_1.json"
            task_payload = self._base_task_payload()
            self._write_json(task_path, task_payload)
            self._write_json(
                plan_path,
                {
                    "dataset_sequence": ["example-dataset", "Plain_Page"],
                    "source_sequence": [
                        "datagov/example-dataset/files/rows.txt",
                        "wikipedia/Plain_Page/content.txt",
                    ],
                    "reasoning_chain_text": ["1. Compute then read."],
                },
            )
            records = seed_records.build_ideal_code_records(task_path)[1]
            records[0]["code"] = records[0]["code"] + "\nprint(answer)\n"
            self._write_json(
                plan_path,
                {
                    "dataset_sequence": ["example-dataset", "Plain_Page"],
                    "source_sequence": [
                        "datagov/example-dataset/files/rows.txt",
                        "wikipedia/Plain_Page/content.txt",
                    ],
                    "reasoning_chain_text": ["1. Compute then read."],
                    "ideal_code": records,
                    "ideal_query": [
                        {
                            "node_id": "1",
                            "dataset_id": "example-dataset",
                            "source": "datagov/example-dataset/files/rows.txt",
                            "intent": task_payload["nodes"]["1"]["subquestion"],
                            "sql": "SELECT 7 AS answer",
                            "answer": "7",
                        }
                    ],
                },
            )

            result = check_records.evaluate_plan(plan_path)

            self.assertIn("ideal_code_mismatch", {issue["code"] for issue in result["issues"]})

    def test_check_records_flags_source_dataset_mismatch(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            task_path = root / "tasks_mini" / "k-1-d-1" / "task_1.json"
            plan_path = root / "plans_mini" / "k-1-d-1" / "task_1.json"
            task_payload = self._base_task_payload()
            self._write_json(task_path, task_payload)
            self._write_json(
                plan_path,
                {
                    "dataset_sequence": ["example-dataset", "Plain_Page"],
                    "source_sequence": [
                        "datagov/example-dataset/files/rows.txt",
                        "wikipedia/Plain_Page/content.txt",
                    ],
                    "reasoning_chain_text": ["1. Compute then read."],
                },
            )
            code_records = seed_records.build_ideal_code_records(task_path)[1]
            query_record = {
                "node_id": "1",
                "dataset_id": "other-dataset",
                "source": "datagov/example-dataset/files/rows.txt",
                "intent": task_payload["nodes"]["1"]["subquestion"],
                "sql": "SELECT 7 AS answer",
                "answer": "7",
            }
            self._write_json(
                plan_path,
                {
                    "dataset_sequence": ["example-dataset", "Plain_Page"],
                    "source_sequence": [
                        "datagov/example-dataset/files/rows.txt",
                        "wikipedia/Plain_Page/content.txt",
                    ],
                    "reasoning_chain_text": ["1. Compute then read."],
                    "ideal_code": code_records,
                    "ideal_query": [query_record],
                },
            )

            result = check_records.evaluate_plan(plan_path)

            self.assertIn("dataset_id_mismatch", {issue["code"] for issue in result["issues"]})

    def test_check_records_compares_mocked_sql_result_to_answer(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            task_path = root / "tasks_mini" / "k-1-d-1" / "task_1.json"
            plan_path = root / "plans_mini" / "k-1-d-1" / "task_1.json"
            task_payload = self._base_task_payload()
            self._write_json(task_path, task_payload)
            self._write_json(
                plan_path,
                {
                    "dataset_sequence": ["example-dataset", "Plain_Page"],
                    "source_sequence": [
                        "datagov/example-dataset/files/rows.txt",
                        "wikipedia/Plain_Page/content.txt",
                    ],
                    "reasoning_chain_text": ["1. Compute then read."],
                },
            )
            code_records = seed_records.build_ideal_code_records(task_path)[1]
            self._write_json(
                plan_path,
                {
                    "dataset_sequence": ["example-dataset", "Plain_Page"],
                    "source_sequence": [
                        "datagov/example-dataset/files/rows.txt",
                        "wikipedia/Plain_Page/content.txt",
                    ],
                    "reasoning_chain_text": ["1. Compute then read."],
                    "ideal_code": code_records,
                    "ideal_query": [
                        {
                            "node_id": "1",
                            "dataset_id": "example-dataset",
                            "source": "datagov/example-dataset/files/rows.txt",
                            "intent": task_payload["nodes"]["1"]["subquestion"],
                            "sql": "SELECT 7 AS answer",
                            "answer": "7",
                        }
                    ],
                },
            )

            calls = []

            def query_runner(record):
                calls.append(record)
                return {"columns": ["answer"], "rows": [[7]], "row_count": 1}

            result = check_records.evaluate_plan(plan_path, query_runner=query_runner)

            self.assertEqual(result["status"], "clean")
            self.assertFalse(result["issues"])
            self.assertEqual([call["node_id"] for call in calls], ["1"])


if __name__ == "__main__":
    unittest.main()
