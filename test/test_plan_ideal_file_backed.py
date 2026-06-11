import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sana_evaluation.tools.external.ideal.plan_ideal import (
    inject_reasoning_chain_prompt,
    plan_ideal,
)
import sana_evaluation.tools.external.ideal.runtime_profile_store as runtime_profile_store
from sana_evaluation.tools.external.ideal.runtime_profile_store import (
    load_runtime_profile_for_context,
    load_runtime_profile_for_task,
    set_runtime_profiles_root,
    set_task_context,
)


class _FakeAgent:
    def __init__(self, system_prompt: str) -> None:
        self.system_prompt = system_prompt


class _FakeToolContext:
    def __init__(self, agent: _FakeAgent) -> None:
        self.agent = agent


class TestPlanIdealFileBacked(unittest.TestCase):
    def tearDown(self) -> None:
        set_runtime_profiles_root("runtime-profiles")
        runtime_profile_store._KRAMABENCH_RUNTIME_PROFILES_ROOT = Path("runtime-profiles")
        set_task_context({})

    def _write_computation_plan(self, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(
                {
                    "dataset_sequence": ["ds_a"],
                    "source_sequence": ["datagov/ds_a/files/rows.txt"],
                    "reasoning_chain_text": "1. Compute the answer.",
                    "ideal_query": [
                        {
                            "node_id": "1",
                            "dataset_id": "ds_a",
                            "intent": "count all rows",
                            "sql": "SELECT COUNT(*) AS n FROM t",
                            "answer": 7,
                        }
                    ],
                    "ideal_code": [
                        {
                            "node_id": "2",
                            "source": "datagov/ds_a/files/rows.txt",
                            "intent": "sum values",
                            "code": "print(3 + 4)",
                            "answer": "7",
                        }
                    ],
                }
            )
        )

    def test_inject_reasoning_chain_prompt_uses_file_backed_reasoning_chain(self):
        with TemporaryDirectory() as tmpdir:
            runtime_profiles_root = Path(tmpdir) / "runtime-profiles"
            target = runtime_profiles_root / "k-1-d-1"
            target.mkdir(parents=True, exist_ok=True)
            (target / "task_2.json").write_text(
                json.dumps(
                    {
                        "dataset_sequence": ["ds_a", "ds_b"],
                        "source_sequence": [
                            "datagov/ds_a/files/rows.txt",
                            "datagov/ds_b/files/rows.txt",
                        ],
                        "reasoning_chain_text": "File-backed plan text.",
                    }
                )
            )

            set_runtime_profiles_root(runtime_profiles_root)
            set_task_context({"task_id": "benchmarks/lakeqa/tasks-mini/tasks/k-1-d-1/task_2.json"})

            plan = load_runtime_profile_for_context()
            prompt = inject_reasoning_chain_prompt("BASE", plan.reasoning_chain_text)

            self.assertIn("## GOLD REASONING CHAIN", prompt)
            self.assertIn("File-backed plan text.", prompt)
            self.assertIn("brief and action-oriented", prompt)
            self.assertNotIn("Path:", prompt)
            self.assertNotIn("Dataset sequence:", prompt)
            self.assertNotIn("dataset_sequence", prompt)

    def test_inject_reasoning_chain_prompt_joins_list_backed_reasoning_chain(self):
        with TemporaryDirectory() as tmpdir:
            runtime_profiles_root = Path(tmpdir) / "runtime-profiles"
            target = runtime_profiles_root / "k-1-d-1"
            target.mkdir(parents=True, exist_ok=True)
            (target / "task_3.json").write_text(
                json.dumps(
                    {
                        "dataset_sequence": ["ds_a", "ds_b"],
                        "source_sequence": [
                            "datagov/ds_a/files/rows.txt",
                            "datagov/ds_b/files/rows.txt",
                        ],
                        "reasoning_chain_text": [
                            "1. First file-backed step.",
                            "2. Second file-backed step.",
                        ],
                    }
                )
            )

            set_runtime_profiles_root(runtime_profiles_root)
            set_task_context({"task_id": "benchmarks/lakeqa/tasks-mini/tasks/k-1-d-1/task_3.json"})

            plan = load_runtime_profile_for_context()
            prompt = inject_reasoning_chain_prompt("BASE", plan.reasoning_chain_text)

            self.assertIn("1. First file-backed step.\n2. Second file-backed step.", prompt)

    def test_load_profile_accepts_ideal_query_and_code_records(self):
        with TemporaryDirectory() as tmpdir:
            runtime_profiles_root = Path(tmpdir) / "runtime-profiles"
            target = runtime_profiles_root / "k-1-d-1"
            target.mkdir(parents=True, exist_ok=True)
            (target / "task_4.json").write_text(
                json.dumps(
                    {
                        "dataset_sequence": ["ds_a"],
                        "source_sequence": ["datagov/ds_a/files/rows.txt"],
                        "reasoning_chain_text": "1. Compute the answer.",
                        "ideal_query": [
                            {
                                "node_id": "1",
                                "dataset_id": "ds_a",
                                "intent": "count all rows",
                                "sql": "SELECT COUNT(*) AS n FROM t",
                                "answer": 7,
                            }
                        ],
                        "ideal_code": [
                            {
                                "node_id": "2",
                                "source": "datagov/ds_a/files/rows.txt",
                                "intent": "sum values",
                                "code": "print(3 + 4)",
                                "answer": "7",
                            }
                        ],
                    }
                )
            )

            set_runtime_profiles_root(runtime_profiles_root)
            plan = load_runtime_profile_for_task("benchmarks/lakeqa/tasks-mini/tasks/k-1-d-1/task_4.json")

            self.assertEqual(len(plan.ideal_query), 1)
            self.assertEqual(plan.ideal_query[0].tool, "query")
            self.assertEqual(plan.ideal_query[0].dataset_id, "ds_a")
            self.assertEqual(plan.ideal_query[0].source, "datagov/ds_a/files/rows.txt")
            self.assertEqual(plan.ideal_query[0].payload, "SELECT COUNT(*) AS n FROM t")
            self.assertEqual(plan.ideal_query[0].answer, 7)
            self.assertEqual(len(plan.ideal_code), 1)
            self.assertEqual(plan.ideal_code[0].tool, "code")
            self.assertEqual(plan.ideal_code[0].dataset_id, "ds_a")
            self.assertEqual(plan.ideal_code[0].payload, "print(3 + 4)")

    def test_load_profile_maps_relative_task_ids_to_runtime_profiles_root_records(self):
        with TemporaryDirectory() as tmpdir:
            runtime_profiles_root = Path(tmpdir) / "runtime-profiles"
            self._write_computation_plan(
                runtime_profiles_root / "custom" / "k-1-d-1" / "task_6.json"
            )

            set_runtime_profiles_root(runtime_profiles_root)
            profile = load_runtime_profile_for_task("custom/k-1-d-1/task_6.json")

            self.assertEqual(profile.profile_path, runtime_profiles_root / "custom" / "k-1-d-1" / "task_6.json")
            self.assertEqual(profile.ideal_query[0].answer, 7)
            self.assertEqual(profile.ideal_code[0].payload, "print(3 + 4)")

    def test_load_profile_maps_canonical_lakeqa_task_to_runtime_profile_records(self):
        with TemporaryDirectory() as tmpdir:
            runtime_profiles_root = Path(tmpdir) / "runtime-profiles"
            self._write_computation_plan(runtime_profiles_root / "k-1-d-1" / "task_7.json")

            set_runtime_profiles_root(runtime_profiles_root)
            profile = load_runtime_profile_for_task("benchmarks/lakeqa/tasks-mini/tasks/k-1-d-1/task_7.json")

            self.assertEqual(profile.profile_path, runtime_profiles_root / "k-1-d-1" / "task_7.json")
            self.assertEqual(profile.ideal_query[0].dataset_id, "ds_a")
            self.assertEqual(profile.ideal_code[0].source, "datagov/ds_a/files/rows.txt")

    def test_load_profile_maps_kramabench_to_kramabench_profile_records(self):
        with TemporaryDirectory() as tmpdir:
            kramabench_root = Path(tmpdir) / "runtime-profiles"
            self._write_computation_plan(
                kramabench_root / "k-2-d-2-s-1" / "task_1.json"
            )

            runtime_profile_store._KRAMABENCH_RUNTIME_PROFILES_ROOT = kramabench_root
            profile = load_runtime_profile_for_task("benchmarks/kramabench/tasks-mini/tasks/k-2-d-2-s-1/task_1.json")

            self.assertEqual(
                profile.profile_path,
                kramabench_root / "k-2-d-2-s-1" / "task_1.json",
            )
            self.assertEqual(profile.ideal_query[0].answer, 7)

    def test_load_profile_rejects_malformed_ideal_query_record(self):
        with TemporaryDirectory() as tmpdir:
            runtime_profiles_root = Path(tmpdir) / "runtime-profiles"
            target = runtime_profiles_root / "k-1-d-1"
            target.mkdir(parents=True, exist_ok=True)
            (target / "task_5.json").write_text(
                json.dumps(
                    {
                        "dataset_sequence": ["ds_a"],
                        "source_sequence": ["datagov/ds_a/files/rows.txt"],
                        "reasoning_chain_text": "1. Compute the answer.",
                        "ideal_query": [
                            {
                                "node_id": "1",
                                "dataset_id": "ds_a",
                                "intent": "count all rows",
                                "answer": 7,
                            }
                        ],
                    }
                )
            )

            set_runtime_profiles_root(runtime_profiles_root)
            with self.assertRaisesRegex(ValueError, "ideal_query\\[0\\].*sql"):
                load_runtime_profile_for_task("benchmarks/lakeqa/tasks-mini/tasks/k-1-d-1/task_5.json")

    def test_load_profile_accepts_unsupported_query_blocker_record(self):
        with TemporaryDirectory() as tmpdir:
            runtime_profiles_root = Path(tmpdir) / "runtime-profiles"
            target = runtime_profiles_root / "k-1-d-1"
            target.mkdir(parents=True, exist_ok=True)
            (target / "task_8.json").write_text(
                json.dumps(
                    {
                        "dataset_sequence": ["ds_a"],
                        "source_sequence": ["datagov/ds_a/files/huge_rows.txt"],
                        "reasoning_chain_text": "1. Compute the answer.",
                        "ideal_query": [
                            {
                                "node_id": "1",
                                "dataset_id": "ds_a",
                                "intent": "count all rows",
                                "answer": "Cannot execute SQL: file is too big (1377 MB >= 500 MB limit).",
                            }
                        ],
                        "ideal_code": [
                            {
                                "node_id": "1",
                                "dataset_id": "ds_a",
                                "intent": "count all rows",
                                "code": "print(7)",
                                "answer": "7",
                            }
                        ],
                    }
                )
            )

            set_runtime_profiles_root(runtime_profiles_root)
            plan = load_runtime_profile_for_task("benchmarks/lakeqa/tasks-mini/tasks/k-1-d-1/task_8.json")

            self.assertEqual(len(plan.ideal_query), 1)
            self.assertTrue(plan.ideal_query[0].blocked)
            self.assertEqual(plan.ideal_query[0].payload, "")
            self.assertIn("Cannot execute SQL", plan.ideal_query[0].answer)

    def test_load_profile_accepts_legacy_query_file_xml_blocker_record(self):
        with TemporaryDirectory() as tmpdir:
            runtime_profiles_root = Path(tmpdir) / "runtime-profiles"
            target = runtime_profiles_root / "k-1-d-1"
            target.mkdir(parents=True, exist_ok=True)
            (target / "task_9.json").write_text(
                json.dumps(
                    {
                        "dataset_sequence": ["ds_xml"],
                        "source_sequence": ["datagov/ds_xml/files/data.txt"],
                        "reasoning_chain_text": "1. Use code for XML.",
                        "ideal_query": [
                            {
                                "node_id": "1",
                                "dataset_id": "ds_xml",
                                "intent": "count XML placemarks",
                                "answer": "Query_file doesnt run on XML",
                            }
                        ],
                        "ideal_code": [
                            {
                                "node_id": "1",
                                "dataset_id": "ds_xml",
                                "intent": "count XML placemarks",
                                "code": "print(7)",
                                "answer": "7",
                            }
                        ],
                    }
                )
            )

            set_runtime_profiles_root(runtime_profiles_root)
            plan = load_runtime_profile_for_task("benchmarks/lakeqa/tasks-mini/tasks/k-1-d-1/task_9.json")

            self.assertEqual(len(plan.ideal_query), 1)
            self.assertTrue(plan.ideal_query[0].blocked)
            self.assertEqual(plan.ideal_query[0].payload, "")
            self.assertIn("Query_file", plan.ideal_query[0].answer)

    def test_plan_ideal_records_execution_plan_and_preserves_gold_chain(self):
        gold_prompt = inject_reasoning_chain_prompt("BASE", "1. Gold step.")
        fake_agent = _FakeAgent(gold_prompt)
        fake_ctx = _FakeToolContext(fake_agent)

        plan_text = "1. Find the dataset | search_ideal\n2. Run the aggregation | query_file"
        msg = plan_ideal(plan_text, tool_context=fake_ctx)

        self.assertEqual("Ideal execution plan recorded.", msg)
        self.assertIn("## GOLD REASONING CHAIN", fake_agent.system_prompt)
        self.assertIn("1. Gold step.", fake_agent.system_prompt)
        self.assertIn("## IDEAL EXECUTION PLAN", fake_agent.system_prompt)
        self.assertIn(plan_text, fake_agent.system_prompt)

    def test_plan_ideal_overwrites_only_execution_plan_section(self):
        gold_prompt = inject_reasoning_chain_prompt("BASE", "1. Gold step.")
        fake_agent = _FakeAgent(gold_prompt)
        fake_ctx = _FakeToolContext(fake_agent)

        plan_ideal("1. First pass | search_ideal", tool_context=fake_ctx)
        plan_ideal("1. Revised pass | query_file", tool_context=fake_ctx)

        self.assertEqual(1, fake_agent.system_prompt.count("## GOLD REASONING CHAIN"))
        self.assertEqual(1, fake_agent.system_prompt.count("## IDEAL EXECUTION PLAN"))
        self.assertIn("1. Gold step.", fake_agent.system_prompt)
        self.assertIn("1. Revised pass | query_file", fake_agent.system_prompt)
        self.assertNotIn("1. First pass | search_ideal", fake_agent.system_prompt)


if __name__ == "__main__":
    unittest.main()
