import json
import os
import sys
import unittest
from contextlib import ExitStack
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sana_evaluation.agent_with_mode import build_mode_bundle, _tool_limit_exclusions_for_run
from sana_evaluation.config import RunConfig
from sana_evaluation.instrumentation import ideal_subagent_costs
from sana_evaluation.instrumentation.trace_plugin import set_trace_context
import sana_evaluation.tools.external.ideal.search_ideal as search_ideal
import sana_evaluation.tools.external.ideal.search_wrapper as search_wrapper

_TASK_ROOT = "k-1-d-1"
_TASK_ID_TEMPLATE = f"benchmarks/lakeqa/tasks-mini/tasks/{_TASK_ROOT}/{{task_name}}"
_LIVE_LOG_PATH = Path("test_logs/search_ideal_judge_samples.jsonl")
_MATRIX_LOG_PATH = Path("test_logs/search_ideal_flag_matrix.jsonl")
_LIVE_MATRIX_LOG_PATH = Path("test_logs/search_ideal_gpt54_nano_matrix.jsonl")
_CORE_QUALITY_TASK_ID = "benchmarks/lakeqa/tasks-mini/tasks/k-1-d-1/task_2.json"
_CORE_QUALITY_QUERY = "provisional drug overdose death counts by state and year"
_CORE_QUALITY_SOURCE = "datagov/vsrr-provisional-drug-overdose-death-counts/files/rows.txt"
_LIVE_SOURCE_SEQUENCE = [
    "datagov/chicago-crime-2015/files/rows.txt",
    "datagov/chicago-crime-2016/files/rows.txt",
    "datagov/chicago-crime-2017/files/rows.txt",
    "datagov/chicago-parks/files/rows.txt",
    "datagov/chicago-building-permits/files/rows.txt",
    "datagov/chicago-school-enrollment/files/rows.txt",
    "datagov/chicago-bike-trips/files/rows.txt",
    "datagov/chicago-fire-stations/files/rows.txt",
    "datagov/chicago-library-locations/files/rows.txt",
    "datagov/chicago-311-service-requests/files/rows.txt",
    "datagov/chicago-public-health-indicators/files/rows.txt",
]
_CRIME_DATASET_IDS = {
    "chicago-crime-2015",
    "chicago-crime-2016",
    "chicago-crime-2017",
}


def _live_openai_enabled() -> bool:
    return bool(os.getenv("OPENAI_API_KEY") and os.getenv("RUN_LIVE_OPENAI_TESTS") == "1")


def _reset_wrapper_caches() -> None:
    search_wrapper._DESC_CACHE_LOADED = False
    search_wrapper._DESC_BY_URI = {}
    search_wrapper._DESC_ROW_BY_URI = {}
    search_wrapper._SNIPPET_CACHE_LOADED = False
    search_wrapper._SNIPPET_BY_URI = {}
    search_wrapper._SCHEMAS_CACHE_LOADED = False
    search_wrapper._SCHEMA_BY_SLUG_FILENAME = {}


def _dataset_id(source_path: str) -> str:
    return search_ideal._dataset_id_from_source(source_path)


def _task_id(task_name: str) -> str:
    return _TASK_ID_TEMPLATE.format(task_name=task_name)


def _write_plan(runtime_profiles_root: Path, *, task_name: str, source_sequence: list[str]) -> str:
    target = runtime_profiles_root / _TASK_ROOT
    target.mkdir(parents=True, exist_ok=True)
    (target / task_name).write_text(
        json.dumps(
            {
                "dataset_sequence": [_dataset_id(source) for source in source_sequence],
                "source_sequence": list(source_sequence),
                "reasoning_chain_text": "1. Find the right dataset.\n2. Query it.",
            }
        )
    )
    return _task_id(task_name)


def _set_task_context(runtime_profiles_root: Path, *, task_name: str, source_sequence: list[str]) -> tuple[str, list[str]]:
    task_id = _write_plan(runtime_profiles_root, task_name=task_name, source_sequence=source_sequence)
    search_ideal.set_runtime_profiles_root(runtime_profiles_root)
    search_ideal.reset_state()
    search_ideal.set_task_context({"task_id": task_id})
    return task_id, [search_ideal._canonical_uri(source) for source in source_sequence]


def _log_judge_call(
    *,
    test_name: str,
    query: str,
    candidates: list[tuple[str, str]],
    picked: list[str] | None,
    reason: str | None,
    count: int,
) -> None:
    _LIVE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _LIVE_LOG_PATH.open("a") as handle:
        handle.write(
            json.dumps(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "test": test_name,
                    "query": query,
                    "candidates": [
                        {"s3_uri": s3_uri, "dataset_id": dataset_id}
                        for s3_uri, dataset_id in candidates
                    ],
                    "picked": list(picked or []),
                    "reason": reason,
                    "count": count,
                }
            )
            + "\n"
        )


class _FakeJudge:
    def __init__(self, *, tools, action, kwargs):
        self.tools = list(tools)
        self._action = action
        self.kwargs = dict(kwargs)

    def __call__(self, prompt: str) -> None:
        if self._action is not None:
            self._action(prompt, self.tools)


class _FakeAgentFactory:
    def __init__(self, action=None):
        self._action = action
        self.calls: list[_FakeJudge] = []

    def __call__(self, *args, **kwargs):
        _ = args
        judge = _FakeJudge(
            tools=kwargs.get("tools", []),
            action=self._action,
            kwargs=kwargs,
        )
        self.calls.append(judge)
        return judge


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


class TestSearchIdealJudge(unittest.TestCase):
    def setUp(self) -> None:
        self._model_env_patch = patch.dict(
            "os.environ",
            {"SANA_SEARCH_IDEAL_SUBAGENT_MODEL": "openai/gpt-5.4-nano"},
            clear=False,
        )
        self._model_env_patch.start()

    def tearDown(self) -> None:
        search_ideal.set_runtime_profiles_root("runtime-profiles")
        search_ideal.reset_state()
        ideal_subagent_costs.reset_stats()
        _reset_wrapper_caches()
        self._model_env_patch.stop()

    def _patch_support_files(
        self,
        root: Path,
        *,
        uri: str,
        dataset_slug: str,
        desc: str = "LLM description",
        snippet: str = "dataset snippet words",
        schema_kind: str = "csv",
        columns: list[str] | None = None,
    ) -> ExitStack:
        desc_path = root / "descriptions.jsonl"
        desc_path.write_text(json.dumps({"dataset_uri": uri, "description": desc}) + "\n")

        snippet_path = root / "snippets.jsonl"
        snippet_path.write_text(json.dumps({"dataset_uri": uri, "dataset_snippet": snippet}) + "\n")

        schema_path = root / "table_schemas_full.jsonl"
        schema_path.write_text(
            json.dumps(
                {
                    "dataset_slug": dataset_slug,
                    "tables": [
                        {
                            "relative_path": "files/rows.txt",
                            "table_kind": schema_kind,
                            "columns": columns if columns is not None else ["col_a", "col_b"],
                        }
                    ],
                }
            )
            + "\n"
        )

        stack = ExitStack()
        stack.enter_context(patch.object(search_wrapper, "_TABLE_DESCRIPTIONS_PATH", desc_path))
        stack.enter_context(patch.object(search_wrapper, "_SNIPPETS_PATH", snippet_path))
        stack.enter_context(patch.object(search_wrapper, "_SCHEMAS_PATH", schema_path))
        return stack

    def test_set_task_context_hard_fails_when_source_sequence_missing(self):
        with TemporaryDirectory() as tmpdir:
            runtime_profiles_root = Path(tmpdir) / "runtime-profiles"
            target = runtime_profiles_root / _TASK_ROOT
            target.mkdir(parents=True, exist_ok=True)
            (target / "task_1.json").write_text(
                json.dumps(
                    {
                        "dataset_sequence": ["ds_one"],
                        "reasoning_chain_text": "Step 1",
                    }
                )
            )

            search_ideal.set_runtime_profiles_root(runtime_profiles_root)
            with self.assertRaisesRegex(ValueError, "source_sequence"):
                search_ideal.set_task_context({"task_id": _task_id("task_1.json")})

    def test_invalid_pick_raises_in_pick_tool(self):
        factory = _FakeAgentFactory()
        remaining = [
            ("s3://lakeqa-yc4103-datalake/datagov/ds_one/files/rows.txt", "ds_one"),
        ]

        with patch.object(search_ideal, "Agent", factory), patch.object(
            search_ideal,
            "_judge_model",
            return_value="fake-model",
        ):
            judge, state = search_ideal._build_judge(remaining)

        self.assertIs(judge, factory.calls[0])
        pick = judge.tools[0]
        with self.assertRaisesRegex(ValueError, "not in candidate list"):
            pick(
                s3_uris=["s3://lakeqa-yc4103-datalake/datagov/not-a-match/files/rows.txt"],
                reason="bad pick",
            )
        self.assertIsNone(state["picked"])

    def test_no_pick_returns_dataset_not_found_without_consuming_source(self):
        with TemporaryDirectory() as tmpdir:
            runtime_profiles_root = Path(tmpdir) / "runtime-profiles"
            _, uris = _set_task_context(
                runtime_profiles_root,
                task_name="task_1.json",
                source_sequence=[
                    "datagov/chicago-crime-2017/files/rows.txt",
                    "datagov/chicago-parks/files/rows.txt",
                ],
            )
            factory = _FakeAgentFactory(action=lambda prompt, tools: None)

            with patch.object(search_ideal, "Agent", factory), patch.object(
                search_ideal,
                "_judge_model",
                return_value="fake-model",
            ):
                with self.assertLogs(search_ideal.logger, level="WARNING") as logs:
                    result = search_ideal.search_ideal("crime data in Chicago 2017")

        self.assertEqual(
            result,
            {
                "results": [],
                "count": 0,
                "query": "crime data in Chicago 2017",
                "message": "Dataset not found",
                "plan_exhausted": False,
            },
        )
        self.assertFalse(result["plan_exhausted"])
        self.assertEqual(search_ideal._USED_S3_URIS, set())
        self.assertEqual(len(factory.calls), 1)
        self.assertTrue(any("no pick" in message for message in logs.output))

    def test_empty_pick_returns_dataset_not_found_without_consuming_source(self):
        with TemporaryDirectory() as tmpdir:
            runtime_profiles_root = Path(tmpdir) / "runtime-profiles"
            _task_id_value, uris = _set_task_context(
                runtime_profiles_root,
                task_name="task_1.json",
                source_sequence=[
                    "datagov/chicago-crime-2017/files/rows.txt",
                    "datagov/chicago-parks/files/rows.txt",
                ],
            )
            factory = _FakeAgentFactory(
                action=lambda prompt, tools: tools[0](s3_uris=[], reason="query is too vague")
            )

            with patch.object(search_ideal, "Agent", factory), patch.object(
                search_ideal,
                "_judge_model",
                return_value="fake-model",
            ):
                result = search_ideal.search_ideal("not enough signal")

        self.assertEqual(
            result,
            {
                "results": [],
                "count": 0,
                "query": "not enough signal",
                "message": "Dataset not found",
                "plan_exhausted": False,
            },
        )
        self.assertEqual(search_ideal._USED_S3_URIS, set())
        self.assertEqual(len(uris), 2)

    def test_search_judge_traces_cost(self):
        source_sequence = ["datagov/chicago-crime-2017/files/rows.txt"]

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            runtime_profiles_root = root / "runtime-profiles"
            task_id, uris = _set_task_context(
                runtime_profiles_root,
                task_name="task_1.json",
                source_sequence=source_sequence,
            )
            uri = uris[0]
            trace_root = root / "traces"
            set_trace_context(task_id, [], str(trace_root))
            ideal_subagent_costs.reset_stats()

            class _CostAgent:
                def __init__(self, *args, **kwargs):
                    self.tools = kwargs["tools"]

                def __call__(self, prompt: str):
                    _ = prompt
                    self.tools[0](s3_uris=[uri], reason="same source")
                    return _FakeAgentResult(input_tokens=900, output_tokens=40, cached_input_tokens=300)

            with self._patch_support_files(root, uri=uri, dataset_slug="chicago-crime-2017"):
                with patch.object(search_ideal, "Agent", _CostAgent), patch.object(
                    search_ideal,
                    "_judge_model",
                    return_value="fake-model",
                ), patch.dict(
                    "os.environ",
                    {"SANA_SEARCH_IDEAL_SUBAGENT_MODEL": "openai/gpt-5.4-nano"},
                    clear=False,
                ):
                    result = search_ideal.search_ideal("crime data in Chicago 2017")

            self.assertEqual(result["count"], 1)
            trace_path = trace_root / "k-1-d-1" / "task_1.jsonl"
            events = [json.loads(line) for line in trace_path.read_text().splitlines()]
            cost_events = [event for event in events if event.get("event") == "ideal_subagent_cost"]
            self.assertEqual(len(cost_events), 1)
            event = cost_events[0]
            self.assertEqual(event["tool"], "search_ideal")
            self.assertEqual(event["subagent_kind"], "judge")
            self.assertEqual(event["model_name"], "openai/gpt-5.4-nano")
            self.assertEqual(event["candidate_count"], 1)
            self.assertEqual(event["selected_count"], 1)
            expected_cost = ((600 * 0.20) + (300 * 0.02) + (40 * 1.25)) / 1_000_000
            self.assertAlmostEqual(event["cost_usd"], expected_cost, places=12)
            self.assertEqual(ideal_subagent_costs.get_stats()["search_ideal_subagent_calls"], 1)

    def test_judge_prompt_includes_table_descriptions(self):
        with TemporaryDirectory() as tmpdir:
            uri = "s3://lakeqa-yc4103-datalake/datagov/chicago-crime-2017/files/rows.txt"
            with self._patch_support_files(
                Path(tmpdir),
                uri=uri,
                dataset_slug="chicago-crime-2017",
                desc="Chicago crime incident records for 2017 with location and offense fields.",
            ):
                prompt = search_ideal._format_judge_prompt(
                    "crime data",
                    [(uri, "chicago-crime-2017")],
                )

        self.assertIn(
            "description=Chicago crime incident records for 2017 with location and offense fields.",
            prompt,
        )

    def test_judge_prompt_includes_merged_runtime_profile_descriptions(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            uri = "s3://lakeqa-yc4103-datalake/datagov/chicago-crime-2017/files/rows.txt"
            desc_path = root / "descriptions.jsonl"
            desc_path.write_text(
                json.dumps(
                    {
                        "dataset_uri": uri,
                        "description": "Merged task mini description for Chicago crime 2017.",
                    }
                )
                + "\n"
            )

            with ExitStack() as stack:
                stack.enter_context(patch.object(search_wrapper, "_TABLE_DESCRIPTIONS_PATH", desc_path))
                prompt = search_ideal._format_judge_prompt(
                    "crime data",
                    [(uri, "chicago-crime-2017")],
                )

        self.assertIn(
            "description=Merged task mini description for Chicago crime 2017.",
            prompt,
        )

    def test_judge_system_prompt_guides_subset_retrieval_without_domain_specific_examples(self):
        prompt = search_ideal._JUDGE_SYSTEM_PROMPT

        self.assertIn("filtered subset", prompt)
        self.assertIn("not need to mention the exact filter value", prompt)
        self.assertIn("Pick multiple datasets only", prompt)
        self.assertNotIn("public/private/postsecondary", prompt)
        self.assertLessEqual(len(prompt.split()), 100)

    def test_set_task_context_resets_used(self):
        with TemporaryDirectory() as tmpdir:
            runtime_profiles_root = Path(tmpdir) / "runtime-profiles"
            _, uris = _set_task_context(
                runtime_profiles_root,
                task_name="task_1.json",
                source_sequence=[
                    "datagov/chicago-crime-2017/files/rows.txt",
                    "datagov/chicago-parks/files/rows.txt",
                ],
            )
            search_ideal._USED_S3_URIS.update(uris)

            search_ideal.set_task_context({"task_id": _task_id("task_1.json")})

        self.assertEqual(search_ideal._USED_S3_URIS, set())
        self.assertEqual(len(search_ideal._CANDIDATES), 2)

    def test_plan_exhausted_returns_empty(self):
        with TemporaryDirectory() as tmpdir:
            runtime_profiles_root = Path(tmpdir) / "runtime-profiles"
            _, uris = _set_task_context(
                runtime_profiles_root,
                task_name="task_1.json",
                source_sequence=[
                    "datagov/chicago-crime-2017/files/rows.txt",
                    "datagov/chicago-parks/files/rows.txt",
                ],
            )
            search_ideal._USED_S3_URIS.update(uris)
            factory = _FakeAgentFactory(action=lambda prompt, tools: None)

            with patch.object(search_ideal, "Agent", factory), patch.object(
                search_ideal,
                "_judge_model",
                return_value="fake-model",
            ):
                result = search_ideal.search_ideal("anything")

        self.assertEqual(
            result,
            {
                "results": [],
                "count": 0,
                "query": "anything",
                "message": "Dataset not found",
                "plan_exhausted": True,
            },
        )
        self.assertEqual(len(factory.calls), 0)

    def test_lessguide_omits_plan_exhausted_from_search_ideal_payloads(self):
        with TemporaryDirectory() as tmpdir:
            runtime_profiles_root = Path(tmpdir) / "runtime-profiles"
            _, uris = _set_task_context(
                runtime_profiles_root,
                task_name="task_1.json",
                source_sequence=["datagov/chicago-crime-2017/files/rows.txt"],
            )
            uri = uris[0]
            factory = _FakeAgentFactory(
                action=lambda prompt, tools: tools[0](s3_uris=[uri], reason="match")
            )
            search_ideal.set_lessguide(True)

            with patch.object(search_ideal, "Agent", factory), patch.object(
                search_ideal,
                "_judge_model",
                return_value="fake-model",
            ):
                result = search_ideal.search_ideal("crime 2017")

        self.assertEqual(result["count"], 1)
        self.assertEqual(result["results"][0]["s3_uri"], uri)
        self.assertNotIn("plan_exhausted", result)

    def test_wrapper_hides_top_k_from_agent(self):
        wrapped = search_wrapper.build_search_tools(
            [search_ideal.search_ideal],
            fixed_k=None,
            results_mode="naive",
        )[0]
        properties = wrapped.tool_spec["inputSchema"]["json"]["properties"]
        self.assertEqual(set(properties), {"query"})
        self.assertNotIn("top_k", properties)

    def test_wrapper_applies_naive_vs_ideal_shaping(self):
        with TemporaryDirectory() as tmpdir:
            runtime_profiles_root = Path(tmpdir) / "runtime-profiles"
            _, uris = _set_task_context(
                runtime_profiles_root,
                task_name="task_1.json",
                source_sequence=["datagov/chicago-crime-2017/files/rows.txt"],
            )
            uri = uris[0]
            factory = _FakeAgentFactory(
                action=lambda prompt, tools: tools[0](s3_uris=[uri], reason="exact match")
            )

            with self._patch_support_files(
                Path(tmpdir),
                uri=uri,
                dataset_slug="chicago-crime-2017",
            ):
                with patch.object(search_ideal, "Agent", factory), patch.object(
                    search_ideal,
                    "_judge_model",
                    return_value="fake-model",
                ):
                    naive_tool = search_wrapper.build_search_tools(
                        [search_ideal.search_ideal],
                        fixed_k=None,
                        results_mode="naive",
                    )[0]
                    naive = naive_tool(query="crime 2017")

                    search_ideal.reset_state()
                    search_ideal.set_runtime_profiles_root(runtime_profiles_root)
                    search_ideal.set_task_context({"task_id": _task_id("task_1.json")})

                    ideal_tool = search_wrapper.build_search_tools(
                        [search_ideal.search_ideal],
                        fixed_k=None,
                        results_mode="ideal",
                    )[0]
                    ideal = ideal_tool(query="crime 2017")

        self.assertEqual(
            naive["results"][0],
            {
                "dataset_id": "chicago-crime-2017",
                "s3_uri": uri,
            },
        )
        self.assertNotIn("llm_desc", naive["results"][0])

        self.assertEqual(
            ideal["results"][0],
            {
                "dataset_id": "chicago-crime-2017",
                "s3_uri": uri,
                "llm_desc": "LLM description",
                "columns": ["col_a", "col_b"],
                "dataset_snippet": "dataset snippet words",
            },
        )


class TestSearchIdealFlagMatrix(unittest.TestCase):
    def setUp(self) -> None:
        self._model_env_patch = patch.dict(
            "os.environ",
            {"SANA_SEARCH_IDEAL_SUBAGENT_MODEL": "openai/gpt-5.4-nano"},
            clear=False,
        )
        self._model_env_patch.start()

    def tearDown(self) -> None:
        search_ideal.set_runtime_profiles_root("runtime-profiles")
        search_ideal.reset_state()
        _reset_wrapper_caches()
        self._model_env_patch.stop()
        try:
            from sana_evaluation.tools.external.ideal import computation_ideal

            computation_ideal.reset_state()
        except Exception:
            pass

    def _patch_support_files(
        self,
        root: Path,
        *,
        uri: str,
        dataset_slug: str,
    ) -> ExitStack:
        desc_path = root / "descriptions.jsonl"
        desc_path.write_text(
            json.dumps(
                {
                    "dataset_uri": uri,
                    "description": "Chicago crime incident records for 2017 with location and offense fields.",
                }
            )
            + "\n"
        )

        snippet_path = root / "snippets.jsonl"
        snippet_path.write_text(
            json.dumps(
                {
                    "dataset_uri": uri,
                    "dataset_snippet": "crime incident case number date block ward arrest district",
                }
            )
            + "\n"
        )

        schema_path = root / "table_schemas_full.jsonl"
        schema_path.write_text(
            json.dumps(
                {
                    "dataset_slug": dataset_slug,
                    "tables": [
                        {
                            "relative_path": "files/rows.txt",
                            "table_kind": "csv",
                            "columns": ["case_number", "date", "primary_type", "arrest"],
                        }
                    ],
                }
            )
            + "\n"
        )

        stack = ExitStack()
        stack.enter_context(patch.object(search_wrapper, "_TABLE_DESCRIPTIONS_PATH", desc_path))
        stack.enter_context(patch.object(search_wrapper, "_SNIPPETS_PATH", snippet_path))
        stack.enter_context(patch.object(search_wrapper, "_SCHEMAS_PATH", schema_path))
        return stack

    def test_search_ideal_four_flag_runs_and_log_results(self):
        rows: list[dict] = []
        source_sequence = ["datagov/chicago-crime-2017/files/rows.txt"]

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            runtime_profiles_root = root / "runtime-profiles"
            task_id = _write_plan(
                runtime_profiles_root,
                task_name="task_matrix.json",
                source_sequence=source_sequence,
            )
            uri = search_ideal._canonical_uri(source_sequence[0])

            with self._patch_support_files(root, uri=uri, dataset_slug="chicago-crime-2017"):
                factory = _FakeAgentFactory(
                    action=lambda prompt, tools: tools[0](s3_uris=[uri], reason="matrix match")
                )
                with patch.object(search_ideal, "Agent", factory), patch.object(
                    search_ideal,
                    "_judge_model",
                    return_value="fake-model",
                ):
                    for search_results_mode, search_lessguide in product(
                        ("naive", "ideal"),
                        (False, True),
                    ):
                        search_ideal.reset_state()
                        search_ideal.set_runtime_profiles_root(runtime_profiles_root)
                        _reset_wrapper_caches()
                        cfg = RunConfig(
                            search_tool_mode="ideal",
                            search_results_mode=search_results_mode,
                            profile_mode="naive",
                            computation_tool_mode="standard",
                            search_free=False,
                            search_lessguide=search_lessguide,
                        )
                        bundle = build_mode_bundle(
                            cfg,
                            data_tools=[],
                            task_context={"task_id": task_id},
                        )
                        tool_by_name = {
                            tool_obj.tool_spec["name"]: tool_obj
                            for tool_obj in bundle.tools
                            if hasattr(tool_obj, "tool_spec")
                        }
                        result = tool_by_name["search_ideal"](query="crime incidents 2017")
                        exclusions = _tool_limit_exclusions_for_run(
                            base_excluded=("skills", "plan"),
                            search_free=False,
                            search_tool_names=bundle.search_tool_names,
                        )

                        self.assertEqual(result["count"], 1)
                        self.assertEqual(result["results"][0]["s3_uri"], uri)
                        self.assertEqual(bundle.search_tool_names, ("search_ideal",))
                        self.assertEqual(bundle.modes["search_tool"], "ideal")
                        self.assertEqual(bundle.modes["search_results"], search_results_mode)
                        self.assertEqual(bundle.modes["profile"], "naive")
                        self.assertEqual(bundle.modes["computation_tool"], "standard")
                        if search_lessguide:
                            self.assertNotIn("plan_exhausted", result)
                        else:
                            self.assertTrue(result["plan_exhausted"])
                        self.assertNotIn("search_ideal", exclusions)

                        rows.append(
                            {
                                "search_tool": "ideal",
                                "search_results": search_results_mode,
                                "profile": "naive",
                                "computation_tool": "standard",
                                "search_free": False,
                                "search_lessguide": search_lessguide,
                                "search_tool_names": list(bundle.search_tool_names),
                                "tool_limit_excluded": list(exclusions),
                                "result": {
                                    "count": result.get("count"),
                                    "message": result.get("message"),
                                    "has_plan_exhausted": "plan_exhausted" in result,
                                    "plan_exhausted": result.get("plan_exhausted"),
                                    "uris": [item["s3_uri"] for item in result.get("results", [])],
                                    "result_keys": sorted(result.keys()),
                                },
                            }
                        )

        self.assertEqual(len(rows), 4)
        _MATRIX_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        _MATRIX_LOG_PATH.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
        )

    @unittest.skipUnless(_live_openai_enabled(), "set RUN_LIVE_OPENAI_TESTS=1 and OPENAI_API_KEY")
    def test_live_gpt54_nano_core_quality_four_flag_runs_and_log_results(self):
        try:
            import openai  # noqa: F401
        except ImportError:
            self.skipTest("openai package not installed")

        rows: list[dict] = []
        expected_uri = search_ideal._canonical_uri(_CORE_QUALITY_SOURCE)

        for search_results_mode, search_lessguide in product(
            ("naive", "ideal"),
            (False, True),
        ):
            search_ideal.reset_state()
            search_ideal.set_runtime_profiles_root("runtime-profiles")
            _reset_wrapper_caches()
            cfg = RunConfig(
                search_tool_mode="ideal",
                search_results_mode=search_results_mode,
                profile_mode="naive",
                computation_tool_mode="standard",
                search_free=False,
                search_lessguide=search_lessguide,
            )
            bundle = build_mode_bundle(
                cfg,
                data_tools=[],
                task_context={"task_id": _CORE_QUALITY_TASK_ID},
            )
            tool_by_name = {
                tool_obj.tool_spec["name"]: tool_obj
                for tool_obj in bundle.tools
                if hasattr(tool_obj, "tool_spec")
            }
            result = tool_by_name["search_ideal"](query=_CORE_QUALITY_QUERY)
            picked_uris = [item["s3_uri"] for item in result.get("results", [])]

            self.assertIn(expected_uri, picked_uris)
            if search_lessguide:
                self.assertNotIn("plan_exhausted", result)
            else:
                self.assertTrue(result["plan_exhausted"])
            if search_results_mode == "ideal":
                self.assertIn("llm_desc", result["results"][0])
                self.assertIn("columns", result["results"][0])

            rows.append(
                {
                    "model": "openai/gpt-5.4-nano",
                    "task_id": _CORE_QUALITY_TASK_ID,
                    "query": _CORE_QUALITY_QUERY,
                    "expected_uri": expected_uri,
                    "search_tool": "ideal",
                    "search_results": search_results_mode,
                    "search_lessguide": search_lessguide,
                    "result": {
                        "count": result.get("count"),
                        "message": result.get("message"),
                        "has_plan_exhausted": "plan_exhausted" in result,
                        "plan_exhausted": result.get("plan_exhausted"),
                        "uris": picked_uris,
                        "result_keys": sorted(result.keys()),
                        "first_result_keys": sorted(result["results"][0].keys())
                        if result.get("results")
                        else [],
                    },
                }
            )

        self.assertEqual(len(rows), 4)
        _LIVE_MATRIX_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        _LIVE_MATRIX_LOG_PATH.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
        )


def _capture_builder_calls(captured_calls: list[dict]):
    original = search_ideal._build_judge

    def _wrapped(remaining: list[tuple[str, str]]):
        judge, state = original(remaining)
        captured_calls.append({"remaining": list(remaining), "state": state})
        return judge, state

    return _wrapped

@unittest.skipUnless(_live_openai_enabled(), "set RUN_LIVE_OPENAI_TESTS=1 and OPENAI_API_KEY")
class TestSearchIdealJudgeLive(unittest.TestCase):
    def tearDown(self) -> None:
        search_ideal.set_runtime_profiles_root("runtime-profiles")
        search_ideal.reset_state()

    def test_live_single_match(self):
        with TemporaryDirectory() as tmpdir:
            runtime_profiles_root = Path(tmpdir) / "runtime-profiles"
            _set_task_context(
                runtime_profiles_root,
                task_name="task_live_single.json",
                source_sequence=_LIVE_SOURCE_SEQUENCE,
            )
            captured_calls: list[dict] = []
            query = "crime data in Chicago 2017"
            with patch.object(search_ideal, "_build_judge", side_effect=_capture_builder_calls(captured_calls)):
                result = search_ideal.search_ideal(query)

        self.assertEqual(result["count"], 1)
        self.assertEqual(result["results"][0]["dataset_id"], "chicago-crime-2017")
        self.assertTrue(captured_calls)
        _log_judge_call(
            test_name="test_live_single_match",
            query=query,
            candidates=captured_calls[0]["remaining"],
            picked=captured_calls[0]["state"]["picked"],
            reason=captured_calls[0]["state"]["reason"],
            count=result["count"],
        )

    def test_live_aggregation_multi_pick(self):
        with TemporaryDirectory() as tmpdir:
            runtime_profiles_root = Path(tmpdir) / "runtime-profiles"
            _set_task_context(
                runtime_profiles_root,
                task_name="task_live_aggregate.json",
                source_sequence=_LIVE_SOURCE_SEQUENCE,
            )
            captured_calls: list[dict] = []
            query = "all years of Chicago crime data"
            with patch.object(search_ideal, "_build_judge", side_effect=_capture_builder_calls(captured_calls)):
                result = search_ideal.search_ideal(query)

        picked_ids = {row["dataset_id"] for row in result["results"]}
        self.assertGreaterEqual(len(picked_ids & _CRIME_DATASET_IDS), 2)
        self.assertTrue(captured_calls)
        _log_judge_call(
            test_name="test_live_aggregation_multi_pick",
            query=query,
            candidates=captured_calls[0]["remaining"],
            picked=captured_calls[0]["state"]["picked"],
            reason=captured_calls[0]["state"]["reason"],
            count=result["count"],
        )

    def test_live_dedup_across_calls(self):
        with TemporaryDirectory() as tmpdir:
            runtime_profiles_root = Path(tmpdir) / "runtime-profiles"
            _set_task_context(
                runtime_profiles_root,
                task_name="task_live_dedup.json",
                source_sequence=_LIVE_SOURCE_SEQUENCE,
            )
            captured_calls: list[dict] = []
            query = "all years of Chicago crime data"
            with patch.object(search_ideal, "_build_judge", side_effect=_capture_builder_calls(captured_calls)):
                first = search_ideal.search_ideal(query)
                first_used = set(search_ideal._USED_S3_URIS)
                second = search_ideal.search_ideal(query)
                second_used = set(search_ideal._USED_S3_URIS)

        first_uris = {row["s3_uri"] for row in first["results"]}
        second_uris = {row["s3_uri"] for row in second["results"]}
        self.assertTrue(first_uris)
        if second_uris:
            self.assertTrue(first_uris.isdisjoint(second_uris))
            self.assertGreater(len(second_used), len(first_used))
        else:
            self.assertEqual(second["message"], "Dataset not found")
            self.assertEqual(second_used, first_used)
        self.assertEqual(len(captured_calls), 2)
        _log_judge_call(
            test_name="test_live_dedup_across_calls:first",
            query=query,
            candidates=captured_calls[0]["remaining"],
            picked=captured_calls[0]["state"]["picked"],
            reason=captured_calls[0]["state"]["reason"],
            count=first["count"],
        )
        _log_judge_call(
            test_name="test_live_dedup_across_calls:second",
            query=query,
            candidates=captured_calls[1]["remaining"],
            picked=captured_calls[1]["state"]["picked"],
            reason=captured_calls[1]["state"]["reason"],
            count=second["count"],
        )

    def test_live_specific_query_ignores_aggregate(self):
        with TemporaryDirectory() as tmpdir:
            runtime_profiles_root = Path(tmpdir) / "runtime-profiles"
            _set_task_context(
                runtime_profiles_root,
                task_name="task_live_specific.json",
                source_sequence=_LIVE_SOURCE_SEQUENCE,
            )
            captured_calls: list[dict] = []
            query = "Chicago crime 2017 specifically"
            with patch.object(search_ideal, "_build_judge", side_effect=_capture_builder_calls(captured_calls)):
                result = search_ideal.search_ideal(query)

        self.assertEqual(result["count"], 1)
        self.assertEqual(result["results"][0]["dataset_id"], "chicago-crime-2017")
        self.assertTrue(captured_calls)
        _log_judge_call(
            test_name="test_live_specific_query_ignores_aggregate",
            query=query,
            candidates=captured_calls[0]["remaining"],
            picked=captured_calls[0]["state"]["picked"],
            reason=captured_calls[0]["state"]["reason"],
            count=result["count"],
        )
