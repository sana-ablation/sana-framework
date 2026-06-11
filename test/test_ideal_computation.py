import json
import logging
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
from unittest.mock import patch

from strands.hooks.events import (
    AfterModelCallEvent,
    AfterToolCallEvent,
    AgentInitializedEvent,
    BeforeModelCallEvent,
    BeforeToolCallEvent,
)

from sana_evaluation.agent_with_mode import build_mode_bundle
from sana_evaluation.config import RunConfig
from sana_evaluation.instrumentation import ideal_subagent_costs
from sana_evaluation.instrumentation.trace_plugin import set_trace_context
from sana_evaluation.instrumentation.agent_plugins import LoggingPlugin
from sana_evaluation.tools.agent_tools import execute_code
from sana_evaluation.tools.agent_tools_v2 import query_file
from sana_evaluation.tools.external.ideal import computation_ideal
from sana_evaluation.tools.external.ideal.runtime_profile_store import (
    load_runtime_profile_for_task,
    set_runtime_profiles_root,
    set_task_context,
)

_COMPUTATION_LOG_PATH = Path("test_logs/ideal_computation_tools.jsonl")
_COMPUTATION_TRANSCRIPT_LOG_PATH = Path("test_logs/ideal_computation_tool_transcript.log")
_CORE_QUALITY_TASK_ID = "benchmarks/lakeqa/tasks-mini/tasks/k-1-d-1/task_1.json"


class _FakeMetrics:
    def __init__(self, input_tokens: int, output_tokens: int, cached_input_tokens: int = 0) -> None:
        self.accumulated_usage = {
            "inputTokens": input_tokens,
            "outputTokens": output_tokens,
            "totalTokens": input_tokens + output_tokens,
        }
        if cached_input_tokens:
            self.accumulated_usage["cacheReadInputTokens"] = cached_input_tokens


class _FakeAgentResult:
    def __init__(self, *, input_tokens: int, output_tokens: int, cached_input_tokens: int = 0) -> None:
        self.metrics = _FakeMetrics(input_tokens, output_tokens, cached_input_tokens)


class IdealComputationToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self._model_env_patch = patch.dict(
            "os.environ",
            {
                "SANA_SEMANTIC_IDEAL_SUBAGENT_MODEL": "openai/gpt-5.4-nano",
                "SANA_REPAIR_IDEAL_SUBAGENT_MODEL": "openai/gpt-5.4",
            },
            clear=False,
        )
        self._model_env_patch.start()
        self._tmp = TemporaryDirectory()
        self._runtime_profiles_root = Path(self._tmp.name) / "runtime-profiles"
        target = self._runtime_profiles_root / "k-1-d-1"
        target.mkdir(parents=True, exist_ok=True)
        (target / "task_1.json").write_text(
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
                            "dataset_id": "ds_a",
                            "intent": "sum values",
                            "code": "print(3 + 4)",
                            "answer": "7",
                        }
                    ],
                }
            )
        )
        set_runtime_profiles_root(self._runtime_profiles_root)
        computation_ideal.reset_state()
        ideal_subagent_costs.reset_stats()
        computation_ideal.set_task_context({"task_id": "benchmarks/lakeqa/tasks-mini/tasks/k-1-d-1/task_1.json"})

    @staticmethod
    def _first_semantic_candidate(*, candidates, **_kwargs):
        return list(candidates)[0] if candidates else None

    @staticmethod
    def _first_semantic_decision(*, candidates, **_kwargs):
        rows = list(candidates)
        return computation_ideal._SemanticDecision(
            record=rows[0] if rows else None,
            intent_matches_payload=True,
            reason="same computation",
        )

    def tearDown(self) -> None:
        computation_ideal.reset_state()
        ideal_subagent_costs.reset_stats()
        set_runtime_profiles_root("runtime-profiles")
        set_task_context({})
        self._tmp.cleanup()
        self._model_env_patch.stop()

    def test_query_ideal_returns_plan_answer_on_match(self):
        with patch.object(
            computation_ideal,
            "_semantic_decision",
            create=True,
            side_effect=self._first_semantic_decision,
        ) as semantic:
            result = computation_ideal.query_ideal._tool_func(
                dataset_id="ds_a",
                file_path="files/rows.txt",
                sql="SELECT COUNT(*) AS n FROM t",
                intent="count all rows",
            )

        self.assertTrue(result["success"])
        self.assertEqual(result["columns"], ["answer"])
        self.assertEqual(result["rows"], [[7]])
        self.assertEqual(result["row_count"], 1)
        self.assertFalse(result["truncated"])
        self.assertNotIn("ideal_oracle", result)
        self.assertNotIn("intent", result)
        self.assertNotIn("source", result)
        self.assertNotIn("node_id", result)
        semantic.assert_called_once()

    def test_execute_ideal_returns_plan_answer_on_match(self):
        with patch.object(
            computation_ideal,
            "_semantic_decision",
            create=True,
            side_effect=self._first_semantic_decision,
        ) as semantic:
            result = computation_ideal.execute_ideal._tool_func(
                code="print(3 + 4)",
                intent="sum values",
            )

        self.assertTrue(result["success"])
        self.assertEqual(result["output"], "7")
        self.assertEqual(result["dataset_id"], "ds_a")
        self.assertEqual(result["file_path"], "files/rows.txt")
        self.assertEqual(
            result["s3_uri"],
            "s3://lakeqa-yc4103-datalake/datagov/ds_a/files/rows.txt",
        )
        self.assertNotIn("ideal_oracle", result)
        self.assertNotIn("intent", result)
        self.assertNotIn("source", result)
        self.assertNotIn("node_id", result)
        semantic.assert_called_once()

    def test_semantic_record_match_uses_nano_judge_selection(self):
        plan = load_runtime_profile_for_task("benchmarks/lakeqa/tasks-mini/tasks/k-1-d-1/task_1.json")
        captured: dict[str, str] = {}

        class _JudgeAgent:
            def __init__(self, *args, **kwargs):
                self.tools = kwargs["tools"]
                captured["model"] = kwargs["model"]
                captured["system_prompt"] = kwargs["system_prompt"]

            def __call__(self, prompt: str) -> None:
                captured["prompt"] = prompt
                self.tools[0](index=1, reason="same computation")
                return _FakeAgentResult(input_tokens=1000, output_tokens=50, cached_input_tokens=400)

        with patch.object(computation_ideal, "Agent", _JudgeAgent), patch.object(
            computation_ideal,
            "_semantic_model",
            return_value="fake-nano-model",
        ):
            record = computation_ideal._semantic_record_match(
                profile=plan,
                tool_name="execute_ideal",
                payload_label="Python code",
                payload="print(3 + 4)",
                intent="sum values",
                candidates=plan.ideal_code,
            )

        self.assertEqual(record, plan.ideal_code[0])
        self.assertEqual(captured["model"], "fake-nano-model")
        self.assertIn("semantic equivalence", captured["system_prompt"])
        self.assertIn("Submitted Python code", captured["prompt"])
        self.assertIn("Never select a record merely because its answer would be useful", captured["prompt"])
        self.assertNotIn('"answer"', captured["prompt"])

    def test_profile_records_prompt_does_not_include_authored_answers(self):
        plan = load_runtime_profile_for_task("benchmarks/lakeqa/tasks-mini/tasks/k-1-d-1/task_1.json")

        payload = json.loads(
            computation_ideal._profile_records_for_prompt(plan, plan.ideal_code)
        )

        self.assertEqual(payload["records"][0]["payload"], "print(3 + 4)")
        self.assertNotIn("answer", payload["records"][0])

    def test_semantic_record_match_traces_cost(self):
        plan = load_runtime_profile_for_task("benchmarks/lakeqa/tasks-mini/tasks/k-1-d-1/task_1.json")
        trace_root = Path(self._tmp.name) / "semantic_traces"
        set_trace_context("benchmarks/lakeqa/tasks-mini/tasks/k-1-d-1/task_1.json", [], str(trace_root))
        ideal_subagent_costs.reset_stats()

        class _JudgeAgent:
            def __init__(self, *args, **kwargs):
                self.tools = kwargs["tools"]

            def __call__(self, prompt: str):
                _ = prompt
                self.tools[0](index=1, reason="same computation")
                return _FakeAgentResult(input_tokens=1000, output_tokens=50, cached_input_tokens=400)

        with patch.object(computation_ideal, "Agent", _JudgeAgent), patch.object(
            computation_ideal,
            "_semantic_model",
            return_value="fake-nano-model",
        ):
            record = computation_ideal._semantic_record_match(
                profile=plan,
                tool_name="query_ideal",
                payload_label="SQL",
                payload="SELECT COUNT(*) AS n FROM t",
                intent="count all rows",
                candidates=plan.ideal_query,
            )

        self.assertEqual(record, plan.ideal_query[0])
        trace_path = trace_root / "k-1-d-1" / "task_1.jsonl"
        events = [json.loads(line) for line in trace_path.read_text().splitlines()]
        cost_events = [event for event in events if event.get("event") == "ideal_subagent_cost"]
        self.assertEqual(len(cost_events), 1)
        event = cost_events[0]
        self.assertEqual(event["tool"], "query_ideal")
        self.assertEqual(event["subagent_kind"], "semantic_judge")
        self.assertEqual(event["model_name"], "openai/gpt-5.4-nano")
        self.assertEqual(event["input_tokens"], 1000)
        self.assertEqual(event["cached_input_tokens"], 400)
        self.assertEqual(event["uncached_input_tokens"], 600)
        self.assertEqual(event["output_tokens"], 50)
        self.assertEqual(event["candidate_count"], 1)
        self.assertEqual(event["selected_index"], 1)
        expected_cost = ((600 * 0.20) + (400 * 0.02) + (50 * 1.25)) / 1_000_000
        self.assertAlmostEqual(event["cost_usd"], expected_cost, places=12)
        self.assertEqual(ideal_subagent_costs.get_stats()["query_ideal_subagent_calls"], 1)

    def test_execute_ideal_filters_semantic_candidates_by_s3_uri(self):
        target = self._runtime_profiles_root / "k-1-d-1"
        (target / "task_sources.json").write_text(
            json.dumps(
                {
                    "dataset_sequence": ["ds_a", "ds_b"],
                    "source_sequence": [
                        "datagov/ds_a/files/a.csv",
                        "datagov/ds_b/files/b.csv",
                    ],
                    "reasoning_chain_text": "1. Compute the answer from the requested source.",
                    "ideal_query": [],
                    "ideal_code": [
                        {
                            "node_id": "1",
                            "dataset_id": "ds_a",
                            "source": "datagov/ds_a/files/a.csv",
                            "intent": "count ds_a rows",
                            "code": "print('a')",
                            "answer": "7",
                        },
                        {
                            "node_id": "2",
                            "dataset_id": "ds_b",
                            "source": "datagov/ds_b/files/b.csv",
                            "intent": "count ds_b rows",
                            "code": "print('b')",
                            "answer": "11",
                        },
                    ],
                }
            )
        )
        computation_ideal.set_task_context({"task_id": "benchmarks/lakeqa/tasks-mini/tasks/k-1-d-1/task_sources.json"})
        captured: dict[str, list[str]] = {}

        def semantic(*, candidates, **_kwargs):
            rows = list(candidates)
            captured["sources"] = [record.source for record in rows]
            return computation_ideal._SemanticDecision(
                record=rows[0] if rows else None,
                intent_matches_payload=True,
                reason="same computation",
            )

        with patch.object(
            computation_ideal,
            "_semantic_decision",
            create=True,
            side_effect=semantic,
        ):
            result = computation_ideal.execute_ideal._tool_func(
                code="print('semantically equivalent to b')",
                intent="count ds_b rows",
                s3_uri="s3://lakeqa-yc4103-datalake/datagov/ds_b/files/b.csv",
            )

        self.assertEqual(captured["sources"], ["datagov/ds_b/files/b.csv"])
        self.assertEqual(
            result,
            {
                "output": "11",
                "success": True,
                "dataset_id": "ds_b",
                "file_path": "files/b.csv",
                "s3_uri": "s3://lakeqa-yc4103-datalake/datagov/ds_b/files/b.csv",
            },
        )

    def test_execute_ideal_semantic_reject_runs_submitted_code_when_intent_matches(self):
        with patch.object(
            computation_ideal,
            "_semantic_decision",
            create=True,
            return_value=computation_ideal._SemanticDecision(
                record=None,
                intent_matches_payload=True,
                reason="code matches intent",
            ),
        ) as semantic, patch.object(
            computation_ideal,
            "_repair_code",
            return_value={"code": "print(99)", "reason": "should not repair"},
        ) as repair, patch.object(
            computation_ideal,
            "_base_execute_code",
            return_value={"success": True, "output": "inspected"},
        ) as base:
            result = computation_ideal.execute_ideal._tool_func(
                code="print('inspect local file')",
                intent="inspect a local query result JSON",
            )

        self.assertEqual(result, {"success": True, "output": "inspected"})
        semantic.assert_called_once()
        repair.assert_not_called()
        base.assert_called_once_with("print('inspect local file')")

    def test_execute_ideal_semantic_reject_repairs_when_intent_and_code_mismatch(self):
        with patch.object(
            computation_ideal,
            "_semantic_decision",
            create=True,
            return_value=computation_ideal._SemanticDecision(
                record=None,
                intent_matches_payload=False,
                reason="intent asks for a count but code prints schema keys",
            ),
        ) as semantic, patch.object(
            computation_ideal,
            "_repair_code",
            return_value={"code": "print(99)", "reason": "aligned with intent"},
        ) as repair, patch.object(
            computation_ideal,
            "_base_execute_code",
            return_value={"success": True, "output": "99"},
        ) as base:
            result = computation_ideal.execute_ideal._tool_func(
                code="print(obj.keys())",
                intent="Count poor bridge records.",
            )

        self.assertEqual(result, {"success": True, "output": "99"})
        semantic.assert_called_once()
        repair.assert_called_once()
        self.assertIn("Intent/payload mismatch", repair.call_args.kwargs["previous_error"])
        base.assert_called_once_with("print(99)")

    def test_query_ideal_semantic_reject_runs_submitted_sql_when_intent_matches(self):
        with patch.object(
            computation_ideal,
            "_semantic_decision",
            create=True,
            return_value=computation_ideal._SemanticDecision(
                record=None,
                intent_matches_payload=True,
                reason="SQL matches intent",
            ),
        ) as semantic, patch.object(
            computation_ideal,
            "_repair_query",
            return_value={"sql": "SELECT 7 AS n", "reason": "should not repair"},
        ) as repair, patch.object(
            computation_ideal,
            "_base_query_file",
            return_value={"success": True, "rows": [["County"], ["Poor Status"]], "columns": ["column_name"]},
        ) as base:
            result = computation_ideal.query_ideal._tool_func(
                dataset_id="ds_a",
                file_path="files/rows.txt",
                sql="DESCRIBE t",
                intent="Inspect schema to find columns.",
        )

        self.assertTrue(result["success"])
        self.assertNotIn("ideal_oracle", result)
        self.assertNotIn("repair_attempts", result)
        self.assertNotIn("repairs", result)
        semantic.assert_called_once()
        repair.assert_not_called()
        base.assert_called_once()
        self.assertEqual(base.call_args.kwargs["sql"], "DESCRIBE t")

    def test_query_ideal_semantic_reject_repairs_when_intent_and_sql_mismatch(self):
        with patch.object(
            computation_ideal,
            "_semantic_decision",
            create=True,
            return_value=computation_ideal._SemanticDecision(
                record=None,
                intent_matches_payload=False,
                reason="intent asks for an aggregate but SQL only inspects schema",
            ),
        ) as semantic, patch.object(
            computation_ideal,
            "_repair_query",
            return_value={"sql": "SELECT 7 AS n", "reason": "aligned with intent"},
        ) as repair, patch.object(
            computation_ideal,
            "_base_query_file",
            return_value={"success": True, "rows": [[7]], "columns": ["n"]},
        ) as base:
            result = computation_ideal.query_ideal._tool_func(
                dataset_id="ds_a",
                file_path="files/rows.txt",
                sql="DESCRIBE t",
                intent="Count all rows.",
        )

        self.assertTrue(result["success"])
        self.assertNotIn("ideal_oracle", result)
        self.assertNotIn("repair_attempts", result)
        self.assertNotIn("repairs", result)
        semantic.assert_called_once()
        repair.assert_called_once()
        self.assertIn("Intent/payload mismatch", repair.call_args.kwargs["previous_error"])
        base.assert_called_once()
        self.assertEqual(base.call_args.kwargs["sql"], "SELECT 7 AS n")

    def test_query_ideal_blocked_record_returns_tool_error_without_repair(self):
        target = self._runtime_profiles_root / "k-1-d-1"
        (target / "task_blocked.json").write_text(
            json.dumps(
                {
                    "dataset_sequence": ["ds_huge"],
                    "source_sequence": ["datagov/ds_huge/files/huge.csv"],
                    "reasoning_chain_text": "1. Use code against the large file.",
                    "ideal_query": [
                        {
                            "node_id": "1",
                            "dataset_id": "ds_huge",
                            "intent": "count all rows",
                            "answer": "Cannot execute SQL: file is too big (1377 MB >= 500 MB limit).",
                        }
                    ],
                    "ideal_code": [
                        {
                            "node_id": "1",
                            "dataset_id": "ds_huge",
                            "intent": "count all rows",
                            "code": "print(7)",
                            "answer": "7",
                        }
                    ],
                }
            )
        )
        computation_ideal.set_task_context({"task_id": "benchmarks/lakeqa/tasks-mini/tasks/k-1-d-1/task_blocked.json"})

        with patch.object(computation_ideal, "_repair_query") as repair:
            result = computation_ideal.query_ideal._tool_func(
                dataset_id="ds_huge",
                file_path="files/huge.csv",
                sql="SELECT COUNT(*) FROM t",
                intent="count all rows",
            )

        self.assertFalse(result["success"])
        self.assertIn("Cannot execute SQL", result["error"])
        self.assertIn("execute_ideal", result["recommendation"])
        self.assertEqual(result["dataset_id"], "ds_huge")
        self.assertNotIn("ideal_oracle", result)
        self.assertNotIn("intent", result)
        self.assertNotIn("source", result)
        repair.assert_not_called()

    def test_query_ideal_retries_after_repair_failure(self):
        with patch.object(
            computation_ideal,
            "_semantic_decision",
            return_value=computation_ideal._SemanticDecision(
                record=None,
                intent_matches_payload=False,
                reason="force repair retry test",
            ),
        ), patch.object(
            computation_ideal,
            "_repair_query",
            side_effect=[
                RuntimeError("transient"),
                {"sql": "SELECT 7 AS n", "reason": "second attempt"},
            ],
        ) as repair, patch.object(
            computation_ideal,
            "_base_query_file",
            return_value={"success": True, "rows": [[7]], "columns": ["n"]},
        ):
            result = computation_ideal.query_ideal._tool_func(
                dataset_id="ds_a",
                file_path="files/rows.txt",
                sql="SELECT bad FROM t",
                intent="not in the plan",
            )

        self.assertTrue(result["success"])
        self.assertNotIn("ideal_oracle", result)
        self.assertNotIn("repair_attempts", result)
        self.assertNotIn("repairs", result)
        self.assertEqual(repair.call_count, 2)

    def test_query_ideal_falls_back_to_all_records_for_wrong_dataset(self):
        with patch.object(
            computation_ideal,
            "_semantic_decision",
            create=True,
            side_effect=self._first_semantic_decision,
        ) as semantic, patch.object(
            computation_ideal,
            "_repair_query",
            return_value={"sql": "", "reason": "should not repair"},
        ) as repair:
            result = computation_ideal.query_ideal._tool_func(
                dataset_id="other_ds",
                file_path="files/rows.txt",
                sql="SELECT COUNT(*) AS n FROM t",
                intent="count all rows",
            )

        self.assertTrue(result["success"])
        self.assertEqual(result["rows"], [[7]])
        self.assertNotIn("ideal_oracle", result)
        self.assertNotIn("repair_attempts", result)
        self.assertNotIn("repairs", result)
        semantic.assert_called_once()
        self.assertEqual(
            [record.dataset_id for record in semantic.call_args.kwargs["candidates"]],
            ["ds_a"],
        )
        repair.assert_not_called()

    def test_execute_ideal_returns_failure_after_repair_exhaustion(self):
        with patch.object(
            computation_ideal,
            "_semantic_decision",
            return_value=computation_ideal._SemanticDecision(
                record=None,
                intent_matches_payload=False,
                reason="force repair exhaustion test",
            ),
        ), patch.object(
            computation_ideal,
            "_repair_code",
            side_effect=[
                {"code": "raise ValueError('one')", "reason": "first"},
                {"code": "raise ValueError('two')", "reason": "second"},
            ],
        ), patch.object(
            computation_ideal,
            "_base_execute_code",
            return_value={"success": False, "error": "ValueError: still bad"},
        ):
            result = computation_ideal.execute_ideal._tool_func(
                code="raise ValueError('bad')",
                intent="not in the plan",
            )

        self.assertFalse(result["success"])
        self.assertNotIn("ideal_oracle", result)
        self.assertNotIn("repair_attempts", result)
        self.assertNotIn("repairs", result)
        self.assertIn("ValueError", result["error"])

    def test_repair_agent_calls_are_counted_and_traced(self):
        trace_root = Path(self._tmp.name) / "traces"
        set_trace_context(
            "benchmarks/lakeqa/tasks-mini/tasks/k-1-d-1/task_1.json",
            [],
            str(trace_root),
        )

        class _RepairAgent:
            def __init__(self, *args, **kwargs):
                self.tools = kwargs["tools"]
                self.plugins = kwargs.get("plugins", [])
                self.__class__.last_plugins = self.plugins

            def __call__(self, prompt: str) -> None:
                _ = prompt
                self.tools[0](code="print(7)", reason="matched authored computation")
                return _FakeAgentResult(input_tokens=2000, output_tokens=100, cached_input_tokens=500)

        with patch.object(
            computation_ideal,
            "_semantic_decision",
            return_value=computation_ideal._SemanticDecision(
                record=None,
                intent_matches_payload=False,
                reason="force repair for instrumentation test",
            ),
        ), patch.object(computation_ideal, "Agent", _RepairAgent), patch.object(
            computation_ideal,
            "_repair_model",
            return_value="fake-model",
        ), patch.object(
            computation_ideal,
            "_base_execute_code",
            return_value={"success": True, "output": "7"},
        ):
            result = computation_ideal.execute_ideal._tool_func(
                code="print('not a plan match')",
                intent="not in the plan",
            )

        self.assertEqual(result, {"success": True, "output": "7"})
        self.assertTrue(
            any(isinstance(plugin, LoggingPlugin) for plugin in _RepairAgent.last_plugins)
        )
        self.assertEqual(
            computation_ideal.get_stats(),
            {
                "execute_ideal_agent_repair_calls": 1,
                "query_ideal_agent_repair_calls": 0,
            },
        )
        trace_path = trace_root / "k-1-d-1" / "task_1.jsonl"
        events = [json.loads(line) for line in trace_path.read_text().splitlines()]
        self.assertEqual(
            [event["event"] for event in events],
            ["repair_agent_invoked", "ideal_subagent_cost", "repair_agent_completed"],
        )
        self.assertEqual(events[0]["tool"], "execute_ideal")
        self.assertEqual(events[0]["attempt"], 1)
        self.assertEqual(events[1]["tool"], "execute_ideal")
        self.assertEqual(events[1]["subagent_kind"], "repair_agent")
        self.assertEqual(events[1]["model_name"], "openai/gpt-5.4")
        self.assertEqual(events[1]["attempt"], 1)
        expected_cost = ((1500 * 2.50) + (500 * 0.25) + (100 * 15.00)) / 1_000_000
        self.assertAlmostEqual(events[1]["cost_usd"], expected_cost, places=12)
        self.assertEqual(events[2]["repair_reason"], "matched authored computation")
        self.assertIn("submitted_code", events[2])

    def test_query_repair_agent_traces_cost(self):
        trace_root = Path(self._tmp.name) / "query_repair_traces"
        set_trace_context("benchmarks/lakeqa/tasks-mini/tasks/k-1-d-1/task_1.json", [], str(trace_root))
        ideal_subagent_costs.reset_stats()

        class _RepairAgent:
            def __init__(self, *args, **kwargs):
                self.tools = kwargs["tools"]

            def __call__(self, prompt: str):
                _ = prompt
                self.tools[0](sql="SELECT 7 AS answer", reason="matched authored computation")
                return _FakeAgentResult(input_tokens=3000, output_tokens=120, cached_input_tokens=1000)

        with patch.object(
            computation_ideal,
            "_semantic_decision",
            return_value=computation_ideal._SemanticDecision(
                record=None,
                intent_matches_payload=False,
                reason="force repair for instrumentation test",
            ),
        ), patch.object(computation_ideal, "Agent", _RepairAgent), patch.object(
            computation_ideal,
            "_repair_model",
            return_value="fake-model",
        ), patch.object(
            computation_ideal,
            "_base_query_file",
            return_value={"success": True, "rows": [[7]], "columns": ["answer"]},
        ):
            result = computation_ideal.query_ideal._tool_func(
                dataset_id="ds_a",
                file_path="files/rows.txt",
                sql="SELECT bad FROM t",
                intent="not in the plan",
            )

        self.assertTrue(result["success"])
        trace_path = trace_root / "k-1-d-1" / "task_1.jsonl"
        events = [json.loads(line) for line in trace_path.read_text().splitlines()]
        cost_events = [event for event in events if event.get("event") == "ideal_subagent_cost"]
        self.assertEqual(len(cost_events), 1)
        event = cost_events[0]
        self.assertEqual(event["tool"], "query_ideal")
        self.assertEqual(event["subagent_kind"], "repair_agent")
        self.assertEqual(event["attempt"], 1)
        expected_cost = ((2000 * 2.50) + (1000 * 0.25) + (120 * 15.00)) / 1_000_000
        self.assertAlmostEqual(event["cost_usd"], expected_cost, places=12)
        self.assertEqual(ideal_subagent_costs.get_stats()["query_ideal_subagent_calls"], 1)

    def test_repair_stats_reset_with_task_context(self):
        with patch.object(
            computation_ideal,
            "_semantic_decision",
            return_value=computation_ideal._SemanticDecision(
                record=None,
                intent_matches_payload=False,
                reason="force repair stats test",
            ),
        ), patch.object(
            computation_ideal,
            "_repair_code",
            return_value={"code": "print(7)", "reason": "fixed"},
        ), patch.object(
            computation_ideal,
            "_base_execute_code",
            return_value={"success": True, "output": "7"},
        ):
            computation_ideal.execute_ideal._tool_func(
                code="print('not a plan match')",
                intent="not in the plan",
            )

        self.assertEqual(computation_ideal.get_stats()["execute_ideal_agent_repair_calls"], 1)
        computation_ideal.set_task_context({"task_id": "benchmarks/lakeqa/tasks-mini/tasks/k-1-d-1/task_1.json"})
        self.assertEqual(
            computation_ideal.get_stats(),
            {
                "execute_ideal_agent_repair_calls": 0,
                "query_ideal_agent_repair_calls": 0,
            },
        )

    def test_execute_repair_prompt_includes_target_dataset_context(self):
        captured: dict[str, str] = {}

        class _PromptCapturingAgent:
            def __init__(self, *args, **kwargs):
                self.tools = kwargs["tools"]

            def __call__(self, prompt: str) -> None:
                captured["prompt"] = prompt
                self.tools[0](code="print(7)", reason="context was sufficient")

        with patch.object(
            computation_ideal,
            "_semantic_decision",
            return_value=computation_ideal._SemanticDecision(
                record=None,
                intent_matches_payload=False,
                reason="force repair for prompt test",
            ),
        ), patch.object(computation_ideal, "Agent", _PromptCapturingAgent), patch.object(
            computation_ideal,
            "_repair_model",
            return_value="fake-model",
        ), patch.object(
            computation_ideal,
            "_base_execute_code",
            return_value={"success": True, "output": "7"},
        ), patch.object(
            computation_ideal,
            "_enhanced_peek_context",
            return_value='{"dataset_description": "Target dataset description", "enhanced_peek_file": {"header_columns": ["value"]}}',
        ):
            computation_ideal.execute_ideal._tool_func(
                code="print('not a plan match')",
                intent="not in the plan",
            )

        self.assertIn("Target dataset context", captured["prompt"])
        self.assertIn("Target dataset description", captured["prompt"])
        self.assertIn("enhanced_peek_file", captured["prompt"])
        self.assertNotIn('"answer"', captured["prompt"])

    def test_query_repair_prompt_includes_enhanced_peek_context(self):
        captured: dict[str, str] = {}

        class _PromptCapturingAgent:
            def __init__(self, *args, **kwargs):
                self.tools = kwargs["tools"]

            def __call__(self, prompt: str) -> None:
                captured["prompt"] = prompt
                self.tools[0](sql="SELECT 7 AS answer", reason="context was sufficient")

        with patch.object(
            computation_ideal,
            "_semantic_decision",
            return_value=computation_ideal._SemanticDecision(
                record=None,
                intent_matches_payload=False,
                reason="force repair for prompt test",
            ),
        ), patch.object(computation_ideal, "Agent", _PromptCapturingAgent), patch.object(
            computation_ideal,
            "_repair_model",
            return_value="fake-model",
        ), patch.object(
            computation_ideal,
            "_base_query_file",
            return_value={"success": True, "rows": [[7]], "columns": ["answer"]},
        ), patch.object(
            computation_ideal,
            "_enhanced_peek_context",
            return_value='{"dataset_description": "Target query description", "enhanced_peek_file": {"header_columns": ["n"]}}',
        ):
            computation_ideal.query_ideal._tool_func(
                dataset_id="ds_a",
                file_path="files/rows.txt",
                sql="SELECT bad FROM t",
                intent="not in the plan",
            )

        self.assertIn("Target dataset context", captured["prompt"])
        self.assertIn("Target query description", captured["prompt"])
        self.assertIn("enhanced_peek_file", captured["prompt"])
        self.assertNotIn('"answer"', captured["prompt"])

    def test_ideal_computation_tools_log_results(self):
        plan = load_runtime_profile_for_task(_CORE_QUALITY_TASK_ID)
        query_record = plan.ideal_query[0]
        code_record = plan.ideal_code[0]
        query_file_path = query_record.source.split(f"/{query_record.dataset_id}/", 1)[-1]

        cfg = RunConfig(
            search_tool_mode="preloaded",
            search_results_mode="naive",
            profile_mode="naive",
            computation_tool_mode="ideal",
        )
        bundle = build_mode_bundle(
            cfg,
            data_tools=[query_file, execute_code],
            task_context={"task_id": _CORE_QUALITY_TASK_ID},
        )
        tool_by_name = {
            tool_obj.tool_spec["name"]: tool_obj
            for tool_obj in bundle.tools
            if hasattr(tool_obj, "tool_spec")
        }

        plugin = LoggingPlugin()
        agent = SimpleNamespace()
        log_logger = logging.getLogger("sana_evaluation.instrumentation.agent_plugins")
        previous_level = log_logger.level
        _COMPUTATION_TRANSCRIPT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(_COMPUTATION_TRANSCRIPT_LOG_PATH, mode="w", encoding="utf-8")
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter("%(levelname)s | %(name)s | %(message)s"))
        log_logger.setLevel(logging.DEBUG)
        log_logger.addHandler(handler)

        def run_logged_tool(tool_name: str, tool_input: dict) -> dict:
            tool_use = {
                "name": tool_name,
                "toolUseId": f"{tool_name}-1",
                "input": tool_input,
            }
            plugin.on_before_model(BeforeModelCallEvent(agent=agent, invocation_state={}))
            plugin.on_after_model(
                AfterModelCallEvent(
                    agent=agent,
                    invocation_state={},
                    stop_response=AfterModelCallEvent.ModelStopResponse(
                        message={
                            "role": "assistant",
                            "content": [{"toolUse": tool_use}],
                        },
                        stop_reason="tool_use",
                    ),
                )
            )
            plugin.on_before_tool(
                BeforeToolCallEvent(
                    agent=agent,
                    selected_tool=tool_by_name[tool_name],
                    tool_use=tool_use,
                    invocation_state={},
                )
            )
            result = tool_by_name[tool_name](**tool_input)
            plugin.on_after_tool(
                AfterToolCallEvent(
                    agent=agent,
                    selected_tool=tool_by_name[tool_name],
                    tool_use=tool_use,
                    invocation_state={},
                    result={
                        "toolUseId": tool_use["toolUseId"],
                        "status": "success",
                        "content": [{"text": json.dumps(result, sort_keys=True)}],
                    },
                )
            )
            return result

        try:
            with patch.object(
                computation_ideal,
                "_semantic_decision",
                side_effect=self._first_semantic_decision,
            ):
                plugin.on_agent_initialized(AgentInitializedEvent(agent=agent))
                query_result = run_logged_tool(
                    "query_ideal",
                    {
                        "dataset_id": query_record.dataset_id,
                        "file_path": query_file_path,
                        "sql": query_record.payload,
                        "intent": query_record.intent,
                    },
                )
                execute_result = run_logged_tool(
                    "execute_ideal",
                    {
                        "code": code_record.payload,
                        "intent": code_record.intent,
                    },
                )
        finally:
            log_logger.removeHandler(handler)
            log_logger.setLevel(previous_level)
            handler.close()

        self.assertEqual(bundle.modes["computation_tool"], "ideal")
        self.assertIn("query_ideal", tool_by_name)
        self.assertIn("execute_ideal", tool_by_name)
        self.assertNotIn("query_file", tool_by_name)
        self.assertNotIn("execute_code", tool_by_name)
        self.assertTrue(query_result["success"])
        self.assertEqual(query_result["columns"], ["answer"])
        self.assertEqual(query_result["rows"], [[query_record.answer]])
        self.assertEqual(query_result["row_count"], 1)
        self.assertFalse(query_result["truncated"])
        self.assertNotIn("ideal_oracle", query_result)
        self.assertNotIn("intent", query_result)
        self.assertNotIn("source", query_result)
        self.assertTrue(execute_result["success"])
        self.assertEqual(execute_result["output"], str(code_record.answer))
        self.assertNotIn("ideal_oracle", execute_result)
        self.assertNotIn("intent", execute_result)
        self.assertNotIn("source", execute_result)

        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "task_id": _CORE_QUALITY_TASK_ID,
            "profile_path": str(plan.profile_path),
            "modes": dict(bundle.modes),
            "tool_names": sorted(tool_by_name),
            "results": {
                "query_ideal": {
                    "success": query_result.get("success"),
                    "dataset_id": query_result.get("dataset_id"),
                    "s3_uri": query_result.get("s3_uri"),
                    "columns": query_result.get("columns"),
                    "rows": query_result.get("rows"),
                    "row_count": query_result.get("row_count"),
                    "result_keys": sorted(query_result.keys()),
                },
                "execute_ideal": {
                    "success": execute_result.get("success"),
                    "output": execute_result.get("output"),
                    "result_keys": sorted(execute_result.keys()),
                },
            },
        }
        _COMPUTATION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        _COMPUTATION_LOG_PATH.write_text(json.dumps(row, sort_keys=True) + "\n")
        transcript = _COMPUTATION_TRANSCRIPT_LOG_PATH.read_text()
        self.assertIn("[role=assistant block=1 tool_use] query_ideal(", transcript)
        self.assertIn("Executing: query_ideal(", transcript)
        self.assertIn("Tool result:", transcript)
        self.assertIn("[role=assistant block=1 tool_use] execute_ideal(", transcript)
        self.assertIn(query_record.dataset_id, transcript)
        self.assertIn(str(query_record.answer), transcript)


if __name__ == "__main__":
    unittest.main()
