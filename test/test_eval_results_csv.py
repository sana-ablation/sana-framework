import csv
import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path


def _load_write_main_csv():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "sana_evaluation" / "run_eval.py"

    fake_pkg = types.ModuleType("sana_evaluation")
    fake_pkg.__path__ = [str(module_path.parent)]

    fake_agent = types.ModuleType("sana_evaluation.agent")
    fake_agent.BatchRunner = object

    fake_config = types.ModuleType("sana_evaluation.config")
    fake_config.AgentConfig = object
    fake_config.ConditionConfig = object
    fake_config.RunConfig = object

    saved = {
        "sana_evaluation": sys.modules.get("sana_evaluation"),
        "sana_evaluation.agent": sys.modules.get("sana_evaluation.agent"),
        "sana_evaluation.config": sys.modules.get("sana_evaluation.config"),
    }
    sys.modules["sana_evaluation"] = fake_pkg
    sys.modules["sana_evaluation.agent"] = fake_agent
    sys.modules["sana_evaluation.config"] = fake_config
    try:
        spec = importlib.util.spec_from_file_location("_test_run_eval_module", module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        return module._write_main_csv
    finally:
        for name, original in saved.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


_write_main_csv = _load_write_main_csv()


class EvalResultsCsvTests(unittest.TestCase):
    def test_write_main_csv_uses_count_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "eval_results.csv"
            results = [
                {
                    "task_id": "tasks_mini/k-1-d-1/task_1.json",
                    "model": "test-model",
                    "predicted_answer": "[42]",
                    "exact_match": 1.0,
                    "f1_score": 1.0,
                    "sources_used": ["src_a", "src_b", "src_a"],
                    "time": 1.5,
                    "cycle_count": 2,
                    "input_tokens": 10,
                    "cached_input_tokens": 6,
                    "uncached_input_tokens": 4,
                    "output_tokens": 5,
                    "total_tokens": 15,
                    "cost_usd": 0.25,
                    "tool_calls_total": 3,
                    "api_tool_calls": 2,
                    "execute_ideal_agent_repair_calls": 1,
                    "query_ideal_agent_repair_calls": 2,
                    "ideal_subagent_calls": 3,
                    "ideal_subagent_input_tokens": 100,
                    "ideal_subagent_cached_input_tokens": 70,
                    "ideal_subagent_uncached_input_tokens": 30,
                    "ideal_subagent_output_tokens": 20,
                    "ideal_subagent_total_tokens": 120,
                    "ideal_subagent_cost_usd": 0.05,
                    "search_ideal_subagent_calls": 1,
                    "search_ideal_subagent_cost_usd": 0.01,
                    "query_ideal_subagent_calls": 1,
                    "query_ideal_subagent_cost_usd": 0.02,
                    "execute_ideal_subagent_calls": 1,
                    "execute_ideal_subagent_cost_usd": 0.02,
                    "success": True,
                    "error": "",
                    "reasoning": "should not appear in csv",
                }
            ]
            tasks_by_id = {
                "tasks_mini/k-1-d-1/task_1.json": {
                    "answer": "42",
                    "datasets_used": ["ds_one", "ds_two", "ds_one"],
                }
            }

            _write_main_csv(str(csv_path), results, tasks_by_id)

            with csv_path.open(newline="") as f:
                rows = list(csv.DictReader(f))

            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(
                row.keys(),
                {
                    "task_id",
                    "model",
                    "expected_answer",
                    "predicted_answer",
                    "exact_match",
                    "f1_score",
                    "required_dataset_count",
                    "sources_used_count",
                    "runtime_seconds",
                    "cycle_count",
                    "input_tokens",
                    "cached_input_tokens",
                    "uncached_input_tokens",
                    "output_tokens",
                    "total_tokens",
                    "cost_usd",
                    "tool_calls_total",
                    "api_tool_calls",
                    "execute_ideal_agent_repair_calls",
                    "query_ideal_agent_repair_calls",
                    "ideal_subagent_calls",
                    "ideal_subagent_input_tokens",
                    "ideal_subagent_cached_input_tokens",
                    "ideal_subagent_uncached_input_tokens",
                    "ideal_subagent_output_tokens",
                    "ideal_subagent_total_tokens",
                    "ideal_subagent_cost_usd",
                    "search_ideal_subagent_calls",
                    "search_ideal_subagent_cost_usd",
                    "query_ideal_subagent_calls",
                    "query_ideal_subagent_cost_usd",
                    "execute_ideal_subagent_calls",
                    "execute_ideal_subagent_cost_usd",
                    "total_cost_with_ideal_subagents_usd",
                    "delegation_subagent_calls",
                    "delegation_subagent_input_tokens",
                    "delegation_subagent_cached_input_tokens",
                    "delegation_subagent_uncached_input_tokens",
                    "delegation_subagent_output_tokens",
                    "delegation_subagent_total_tokens",
                    "delegation_subagent_cost_usd",
                    "search_subagent_calls",
                    "search_subagent_cost_usd",
                    "inspect_subagent_calls",
                    "inspect_subagent_cost_usd",
                    "total_cost_with_all_subagents_usd",
                    "success",
                    "error",
                },
            )
            self.assertEqual(row["execute_ideal_agent_repair_calls"], "1")
            self.assertEqual(row["query_ideal_agent_repair_calls"], "2")
            self.assertEqual(row["ideal_subagent_calls"], "3")
            self.assertEqual(row["ideal_subagent_uncached_input_tokens"], "30")
            self.assertEqual(row["search_ideal_subagent_cost_usd"], "0.01")
            self.assertEqual(row["query_ideal_subagent_cost_usd"], "0.02")
            self.assertEqual(row["execute_ideal_subagent_cost_usd"], "0.02")
            self.assertAlmostEqual(float(row["total_cost_with_ideal_subagents_usd"]), 0.30)
            self.assertEqual(row["required_dataset_count"], "2")
            self.assertEqual(row["sources_used_count"], "2")
            self.assertNotIn("reasoning", row)
            self.assertNotIn("required_datasets", row)
            self.assertNotIn("sources_used", row)

    def test_write_main_csv_rewrites_legacy_rows_with_counts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "eval_results.csv"
            with csv_path.open("w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "task_id",
                        "model",
                        "expected_answer",
                        "predicted_answer",
                        "exact_match",
                        "f1_score",
                        "reasoning",
                        "required_datasets",
                        "sources_used",
                        "runtime_seconds",
                        "cycle_count",
                        "input_tokens",
                        "output_tokens",
                        "total_tokens",
                        "cost_usd",
                        "tool_calls_total",
                        "api_tool_calls",
                        "execute_ideal_agent_repair_calls",
                        "query_ideal_agent_repair_calls",
                        "ideal_subagent_cost_usd",
                        "success",
                        "error",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "task_id": "tasks_mini/k-1-d-1/task_1.json",
                        "model": "test-model",
                        "expected_answer": "42",
                        "predicted_answer": "[42]",
                        "exact_match": "1.0",
                        "f1_score": "1.0",
                        "reasoning": "reason",
                        "required_datasets": '["ds_one", "ds_two"]',
                        "sources_used": '["src_a"]',
                        "runtime_seconds": "1.5",
                        "cycle_count": "2",
                        "input_tokens": "10",
                        "output_tokens": "5",
                        "total_tokens": "15",
                        "cost_usd": "0.25",
                        "tool_calls_total": "3",
                        "api_tool_calls": "2",
                        "execute_ideal_agent_repair_calls": "4",
                        "query_ideal_agent_repair_calls": "5",
                        "ideal_subagent_cost_usd": "0.03",
                        "success": "True",
                        "error": "",
                    }
                )

            _write_main_csv(str(csv_path), results=[], tasks_by_id={})

            with csv_path.open(newline="") as f:
                rows = list(csv.DictReader(f))

            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row["required_dataset_count"], "2")
            self.assertEqual(row["sources_used_count"], "1")
            self.assertEqual(row["execute_ideal_agent_repair_calls"], "4")
            self.assertEqual(row["query_ideal_agent_repair_calls"], "5")
            self.assertEqual(row["ideal_subagent_cost_usd"], "0.03")
            self.assertAlmostEqual(float(row["total_cost_with_ideal_subagents_usd"]), 0.28)
            self.assertNotIn("reasoning", row)
            self.assertNotIn("required_datasets", row)
            self.assertNotIn("sources_used", row)


    def test_write_main_csv_records_delegation_subagent_costs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "eval_results.csv"
            results = [
                {
                    "task_id": "tasks_mini/k-1-d-1/task_1.json",
                    "model": "test-model",
                    "predicted_answer": "[42]",
                    "exact_match": 1.0,
                    "f1_score": 1.0,
                    "sources_used": [],
                    "time": 1.5,
                    "cycle_count": 2,
                    "input_tokens": 10,
                    "cached_input_tokens": 6,
                    "uncached_input_tokens": 4,
                    "output_tokens": 5,
                    "total_tokens": 15,
                    "cost_usd": 0.004,
                    "tool_calls_total": 3,
                    "api_tool_calls": 0,
                    # No ideal subagents in this run.
                    "ideal_subagent_calls": 0,
                    "ideal_subagent_cost_usd": 0.0,
                    # Delegation subagents — the new columns.
                    "delegation_subagent_calls": 5,
                    "delegation_subagent_input_tokens": 30000,
                    "delegation_subagent_cached_input_tokens": 20000,
                    "delegation_subagent_uncached_input_tokens": 10000,
                    "delegation_subagent_output_tokens": 1500,
                    "delegation_subagent_total_tokens": 31500,
                    "delegation_subagent_cost_usd": 0.025,
                    "search_subagent_calls": 1,
                    "search_subagent_cost_usd": 0.005,
                    "inspect_subagent_calls": 4,
                    "inspect_subagent_cost_usd": 0.020,
                    "success": True,
                    "error": "",
                }
            ]
            tasks_by_id = {
                "tasks_mini/k-1-d-1/task_1.json": {
                    "answer": "42",
                    "datasets_used": [],
                }
            }

            _write_main_csv(str(csv_path), results, tasks_by_id)

            with csv_path.open(newline="") as f:
                rows = list(csv.DictReader(f))

            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row["delegation_subagent_calls"], "5")
            self.assertEqual(row["delegation_subagent_input_tokens"], "30000")
            self.assertEqual(row["delegation_subagent_uncached_input_tokens"], "10000")
            self.assertEqual(row["delegation_subagent_cost_usd"], "0.025")
            self.assertEqual(row["search_subagent_calls"], "1")
            self.assertEqual(row["inspect_subagent_calls"], "4")
            # total_cost_with_ideal stays equal to planner cost when no ideal subagent used.
            self.assertAlmostEqual(float(row["total_cost_with_ideal_subagents_usd"]), 0.004)
            # total_cost_with_all_subagents adds delegation cost on top.
            self.assertAlmostEqual(
                float(row["total_cost_with_all_subagents_usd"]),
                0.004 + 0.025,
            )


if __name__ == "__main__":
    unittest.main()
