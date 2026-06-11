import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from sana_evaluation.config import AgentConfig, RunConfig
from sana_evaluation.agent_with_mode import (
    DataLakeAgent,
    build_mode_bundle,
    _tool_limit_exclusions_for_run,
)
from sana_evaluation.helper.prompting import (
    compose_baseline_prompt,
    compose_managed_prompt,
    compose_preloaded_block,
    inject_debug_prompt,
    normalize_debug_mode,
    skill_paths_for_modes,
)
from sana_evaluation.tools.external.ideal.runtime_profile_store import (
    get_task_context,
    set_runtime_profiles_root,
    set_task_context,
)


class TestModeWrapper(unittest.TestCase):
    def _write_valid_plan(
        self,
        runtime_profiles_root: Path,
        *,
        task_name: str = "task_1.json",
        reasoning_chain_text: str = "1. Gold reasoning step.",
    ) -> None:
        target = runtime_profiles_root / "k-1-d-1"
        target.mkdir(parents=True, exist_ok=True)
        (target / task_name).write_text(
            json.dumps(
                {
                    "dataset_sequence": ["ds_one"],
                    "source_sequence": ["datagov/ds_one/files/rows.txt"],
                    "reasoning_chain_text": reasoning_chain_text,
                }
            )
        )

    def tearDown(self) -> None:
        set_runtime_profiles_root("runtime-profiles")
        set_task_context({})

    def test_naive_modes_disable_skills(self):
        cfg = RunConfig(
            search_tool_mode="naive",
            search_results_mode="naive",
            profile_mode="naive",
        )
        bundle = build_mode_bundle(cfg, data_tools=[])
        self.assertEqual(bundle.modes["search_tool"], "naive")
        self.assertEqual(bundle.modes["search_results"], "naive")
        self.assertEqual(bundle.modes["profile"], "naive")
        self.assertFalse(bundle.enable_skills)
        self.assertFalse(bundle.enable_stagnation)
        self.assertIn("search_value", bundle.search_tool_names)

    def test_standard_search_includes_hybrid_rrf_value_search(self):
        cfg = RunConfig(
            search_tool_mode="standard",
            search_results_mode="naive",
            profile_mode="naive",
        )
        bundle = build_mode_bundle(cfg, data_tools=[])
        self.assertIn("search_value", bundle.search_tool_names)
        self.assertNotIn("search_reranked", bundle.search_tool_names)
        self.assertNotIn("search_ideal", bundle.search_tool_names)

    def test_managed_naive_prompt_mentions_sparse_search_only(self):
        prompt = compose_managed_prompt("naive")
        self.assertIn("search_value", prompt)
        self.assertIn("search_schema", prompt)
        self.assertIn("search_prefix", prompt)
        self.assertNotIn("search_reranked", prompt)
        self.assertNotIn("search_ideal", prompt)

    def test_managed_standard_prompt_mentions_hybrid_rrf_search(self):
        prompt = compose_managed_prompt("standard")
        self.assertIn("search_value", prompt)
        self.assertIn("search_schema", prompt)
        self.assertIn("search_prefix", prompt)
        self.assertNotIn("search_reranked", prompt)
        self.assertNotIn("search_ideal", prompt)

    def test_managed_ideal_prompt_mentions_only_search_ideal(self):
        prompt = compose_managed_prompt("ideal")
        self.assertIn("search_ideal", prompt)
        self.assertNotIn("search_reranked", prompt)
        self.assertNotIn("search_value", prompt)
        self.assertNotIn("search_schema", prompt)
        self.assertNotIn("search_prefix", prompt)
        self.assertNotIn("reasoning chain", prompt.lower())

    def test_managed_preloaded_prompt_mentions_authoritative_preloaded_regime(self):
        prompt = compose_managed_prompt("preloaded")
        self.assertIn("PRELOADED DATASETS", prompt)
        self.assertIn("Do not search", prompt)
        self.assertNotIn("search_ideal", prompt)
        self.assertNotIn("search_reranked", prompt)

    def test_query_file_schema_rule_uses_schema_knowledge_not_peek_requirement(self):
        expected = "NEVER call `query_file` on a file you do not know the schema of."
        old = "NEVER call `query_file` on a file you have not first inspected with `peek_file`."

        for prompt in (compose_managed_prompt("preloaded"), compose_baseline_prompt("preloaded")):
            self.assertIn(expected, prompt)
            self.assertNotIn(old, prompt)

    def test_baseline_prompt_discourages_schema_inspection_loops(self):
        prompt = compose_baseline_prompt("preloaded")

        self.assertIn("Do not spend many turns inspecting schema.", prompt)
        self.assertIn("One peek, describe, or small row sample is enough", prompt)

    def test_baseline_prompt_mentions_hybrid_rrf_search_in_standard_mode(self):
        prompt = compose_baseline_prompt("standard")
        self.assertIn("search_value", prompt)
        self.assertIn("search_schema", prompt)
        self.assertNotIn("search_reranked", prompt)
        self.assertNotIn("search_ideal", prompt)

    def test_baseline_prompt_mentions_only_search_ideal_in_ideal_mode(self):
        prompt = compose_baseline_prompt("ideal")
        self.assertIn("search_ideal", prompt)
        self.assertNotIn("search_reranked", prompt)
        self.assertNotIn("reasoning chain", prompt.lower())

    def test_compose_preloaded_block_lists_dataset_ids_and_uris(self):
        block = compose_preloaded_block(
            [
                "datagov/ds_one/files/rows.txt",
                "wikipedia/Cameron_County,_Texas/content.txt",
            ]
        )
        self.assertIn("dataset_id: ds_one | uri: datagov/ds_one/files/rows.txt", block)
        self.assertIn(
            "dataset_id: Cameron_County,_Texas | uri: wikipedia/Cameron_County,_Texas/content.txt",
            block,
        )

    def test_skill_paths_are_search_and_management_aware(self):
        standard_paths = skill_paths_for_modes("standard", "standard")
        ideal_paths = skill_paths_for_modes("ideal", "ideal")
        preloaded_paths = skill_paths_for_modes("preloaded", "standard")
        self.assertIn("sana_evaluation/tools/skills/plan-agent", standard_paths)
        self.assertIn("sana_evaluation/tools/skills/discover-data-standard", standard_paths)
        self.assertIn("sana_evaluation/tools/skills/plan-ideal", ideal_paths)
        self.assertIn("sana_evaluation/tools/skills/discover-data-ideal", ideal_paths)
        self.assertIn("sana_evaluation/tools/skills/plan-agent", preloaded_paths)
        self.assertIn("sana_evaluation/tools/skills/query-data", preloaded_paths)
        self.assertFalse(any("discover-data" in path for path in preloaded_paths))

    def test_ideal_management_uses_managed_stack_with_plan_swap(self):
        with TemporaryDirectory() as tmpdir:
            runtime_profiles_root = Path(tmpdir) / "runtime-profiles"
            self._write_valid_plan(runtime_profiles_root)
            set_runtime_profiles_root(runtime_profiles_root)

            cfg = RunConfig(
                search_tool_mode="ideal",
                search_results_mode="ideal",
                profile_mode="ideal",
            )
            bundle = build_mode_bundle(
                cfg,
                data_tools=[],
                task_context={"task_id": "benchmarks/lakeqa/tasks-mini/tasks/k-1-d-1/task_1.json"},
            )
            tool_names = [tool_obj.tool_spec["name"] for tool_obj in bundle.tools]
            self.assertIn("plan_ideal", tool_names)
            self.assertNotIn("plan", tool_names)
            self.assertIn("search_ideal", bundle.search_tool_names)
            self.assertFalse(bundle.enable_skills)
            self.assertTrue(bundle.enable_stagnation)

    def test_profile_skills_flag_enables_skills_for_managed_modes(self):
        cfg = RunConfig(
            search_tool_mode="standard",
            search_results_mode="naive",
            profile_mode="standard",
            profile_skills_enabled=True,
        )
        bundle = build_mode_bundle(cfg, data_tools=[])
        tool_names = [tool_obj.tool_spec["name"] for tool_obj in bundle.tools]
        self.assertIn("plan", tool_names)
        self.assertTrue(bundle.enable_skills)
        self.assertTrue(bundle.enable_stagnation)
        self.assertEqual(bundle.modes["profile_skills"], "on")

    def test_base_agent_does_not_expose_sandbox_admin_tools(self):
        import sana_evaluation.agent_with_mode as agent_with_mode

        captured = {}

        class _FakeStrandsAgent:
            def __init__(self, *args, **kwargs):
                captured["tools"] = kwargs["tools"]
                self.model = kwargs["model"]

        cfg = RunConfig(
            search_tool_mode="standard",
            search_results_mode="naive",
            profile_mode="standard",
        )

        with patch.object(agent_with_mode, "build_model", return_value=object()):
            with patch.object(agent_with_mode, "Agent", _FakeStrandsAgent):
                agent = DataLakeAgent(AgentConfig(model_name="openai/gpt-5.4-nano"), cfg)
                agent._build_agent(MagicMock(), task_context={})

        tool_names = [getattr(tool, "tool_name", None) for tool in captured["tools"]]
        self.assertNotIn("get_sandbox_info", tool_names)
        self.assertNotIn("cleanup_sandbox", tool_names)

    def test_managed_prompt_omits_skills_when_plugin_disabled(self):
        cfg = RunConfig(
            search_tool_mode="standard",
            search_results_mode="naive",
            profile_mode="standard",
            profile_skills_enabled=False,
        )
        bundle = build_mode_bundle(cfg, data_tools=[])

        self.assertFalse(bundle.enable_skills)
        self.assertNotIn("## SKILLS", bundle.system_prompt)
        self.assertNotIn("skills(", bundle.system_prompt)
        self.assertNotIn("Skill loading", bundle.system_prompt)

    def test_ideal_management_stores_task_context_for_plan_loading(self):
        with TemporaryDirectory() as tmpdir:
            runtime_profiles_root = Path(tmpdir) / "runtime-profiles"
            self._write_valid_plan(runtime_profiles_root, task_name="task_6.json")
            set_runtime_profiles_root(runtime_profiles_root)

            cfg = RunConfig(
                search_tool_mode="ideal",
                search_results_mode="ideal",
                profile_mode="ideal",
            )
            build_mode_bundle(
                cfg,
                data_tools=[],
                task_context={"task_id": "benchmarks/lakeqa/tasks-mini/tasks/k-1-d-1/task_6.json"},
            )
            self.assertEqual(
                get_task_context().get("task_id"),
                "benchmarks/lakeqa/tasks-mini/tasks/k-1-d-1/task_6.json",
            )

    def test_ideal_management_preloads_gold_reasoning_chain_into_prompt(self):
        with TemporaryDirectory() as tmpdir:
            runtime_profiles_root = Path(tmpdir) / "runtime-profiles"
            self._write_valid_plan(
                runtime_profiles_root,
                reasoning_chain_text="1. Gold reasoning step.\n2. Gold reasoning step two.",
            )
            set_runtime_profiles_root(runtime_profiles_root)

            cfg = RunConfig(
                search_tool_mode="ideal",
                search_results_mode="ideal",
                profile_mode="ideal",
            )
            bundle = build_mode_bundle(
                cfg,
                data_tools=[],
                task_context={"task_id": "benchmarks/lakeqa/tasks-mini/tasks/k-1-d-1/task_1.json"},
            )

            # Task-specific content now lives in task_trailer so the stable system_prompt
            # stays cacheable across tasks. The agent appends the trailer at invocation time.
            self.assertNotIn("## GOLD REASONING CHAIN", bundle.system_prompt)
            self.assertIn("## GOLD REASONING CHAIN", bundle.task_trailer)
            self.assertIn("1. Gold reasoning step.", bundle.task_trailer)
            self.assertIn("2. Gold reasoning step two.", bundle.task_trailer)
            self.assertNotIn("## IDEAL EXECUTION PLAN", bundle.system_prompt)
            self.assertNotIn("## IDEAL EXECUTION PLAN", bundle.task_trailer)

    def test_standard_management_prompt_matches_standard_search_tools(self):
        cfg = RunConfig(
            search_tool_mode="standard",
            search_results_mode="naive",
            profile_mode="standard",
        )
        bundle = build_mode_bundle(cfg, data_tools=[])
        self.assertIn("search_value", bundle.system_prompt)
        self.assertNotIn("search_reranked", bundle.system_prompt)
        self.assertNotIn("search_ideal", bundle.system_prompt)

    def test_standard_management_prompt_matches_ideal_search_tools(self):
        with TemporaryDirectory() as tmpdir:
            runtime_profiles_root = Path(tmpdir) / "runtime-profiles"
            target = runtime_profiles_root / "k-1-d-1"
            target.mkdir(parents=True, exist_ok=True)
            (target / "task_1.json").write_text(
                '{"dataset_sequence":["ds_one"],"source_sequence":["datagov/ds_one/files/rows.txt"],"reasoning_chain_text":"Step 1"}'
            )
            set_runtime_profiles_root(runtime_profiles_root)

            cfg = RunConfig(
                search_tool_mode="ideal",
                search_results_mode="ideal",
                profile_mode="standard",
            )
            bundle = build_mode_bundle(
                cfg,
                data_tools=[],
                task_context={"task_id": "benchmarks/lakeqa/tasks-mini/tasks/k-1-d-1/task_1.json"},
            )
            self.assertIn("search_ideal", bundle.system_prompt)
            self.assertNotIn("search_reranked", bundle.system_prompt)
            self.assertNotIn("search_value", bundle.system_prompt)

    def test_naive_management_prompt_matches_standard_search_tools(self):
        cfg = RunConfig(
            search_tool_mode="standard",
            search_results_mode="naive",
            profile_mode="naive",
        )
        bundle = build_mode_bundle(cfg, data_tools=[])
        self.assertIn("search_value", bundle.system_prompt)
        self.assertNotIn("search_reranked", bundle.system_prompt)
        self.assertNotIn("search_ideal", bundle.system_prompt)

    def test_naive_management_prompt_matches_ideal_search_tools(self):
        with TemporaryDirectory() as tmpdir:
            runtime_profiles_root = Path(tmpdir) / "runtime-profiles"
            target = runtime_profiles_root / "k-1-d-1"
            target.mkdir(parents=True, exist_ok=True)
            (target / "task_1.json").write_text(
                '{"dataset_sequence":["ds_one"],"source_sequence":["datagov/ds_one/files/rows.txt"],"reasoning_chain_text":"Step 1"}'
            )
            set_runtime_profiles_root(runtime_profiles_root)

            cfg = RunConfig(
                search_tool_mode="ideal",
                search_results_mode="ideal",
                profile_mode="naive",
            )
            bundle = build_mode_bundle(
                cfg,
                data_tools=[],
                task_context={"task_id": "benchmarks/lakeqa/tasks-mini/tasks/k-1-d-1/task_1.json"},
            )
            self.assertIn("search_ideal", bundle.system_prompt)
            self.assertNotIn("search_reranked", bundle.system_prompt)
            self.assertNotIn("reasoning chain", bundle.system_prompt.lower())

    def test_debug_decision_notes_are_injected_for_mode_bundle(self):
        cfg = RunConfig(
            search_tool_mode="standard",
            search_results_mode="naive",
            profile_mode="naive",
            debug_mode="decision_notes",
        )
        bundle = build_mode_bundle(cfg, data_tools=[])
        self.assertIn("## DEBUG DECISION NOTES", bundle.system_prompt)
        self.assertIn("goal:", bundle.system_prompt)
        self.assertIn("why_this_tool:", bundle.system_prompt)
        self.assertIn("what_success_looks_like:", bundle.system_prompt)
        self.assertIn("confidence: <low|medium|high>", bundle.system_prompt)

    def test_search_free_adds_active_search_tools_to_tool_limit_exclusions(self):
        self.assertEqual(
            _tool_limit_exclusions_for_run(
                base_excluded=("skills", "plan", "plan_ideal"),
                search_free=True,
                search_tool_names=("search_ideal", "search_schema"),
            ),
            ("skills", "plan", "plan_ideal", "search_ideal", "search_schema"),
        )

    def test_tool_limit_exclusions_leave_search_tools_counted_by_default(self):
        self.assertEqual(
            _tool_limit_exclusions_for_run(
                base_excluded=("skills", "plan", "plan_ideal"),
                search_free=False,
                search_tool_names=("search_ideal",),
            ),
            ("skills", "plan", "plan_ideal"),
        )

    def test_default_tool_limit_exclusions_include_both_plan_tools(self):
        self.assertEqual(
            tuple(
                DataLakeAgent._tool_limit_excluded_tools(
                    None,
                    search_tool_mode="ideal",
                    profile_mode="ideal",
                )
            ),
            ("skills", "plan", "plan_ideal"),
        )

    def test_inject_debug_prompt_noops_when_disabled(self):
        self.assertEqual("BASE_PROMPT", inject_debug_prompt("BASE_PROMPT", None))
        self.assertEqual("BASE_PROMPT", inject_debug_prompt("BASE_PROMPT", "none"))

    def test_normalize_debug_mode_rejects_unknown_value(self):
        self.assertEqual("decision_notes", normalize_debug_mode("decision_notes"))
        self.assertIsNone(normalize_debug_mode(None))
        with self.assertRaises(ValueError):
            normalize_debug_mode("full_cot")

    def test_ideal_management_fails_fast_without_plan_file(self):
        with TemporaryDirectory() as tmpdir:
            runtime_profiles_root = Path(tmpdir) / "runtime-profiles"
            runtime_profiles_root.mkdir(parents=True, exist_ok=True)
            set_runtime_profiles_root(runtime_profiles_root)

            cfg = RunConfig(
                search_tool_mode="ideal",
                search_results_mode="ideal",
                profile_mode="ideal",
            )
            with self.assertRaises(FileNotFoundError):
                build_mode_bundle(
                    cfg,
                    data_tools=[],
                    task_context={"task_id": "benchmarks/lakeqa/tasks-mini/tasks/k-1-d-1/task_missing.json"},
                )

    def test_ideal_management_fails_fast_with_invalid_plan_file(self):
        with TemporaryDirectory() as tmpdir:
            runtime_profiles_root = Path(tmpdir) / "runtime-profiles"
            target = runtime_profiles_root / "k-1-d-1"
            target.mkdir(parents=True, exist_ok=True)
            (target / "task_1.json").write_text(
                json.dumps(
                    {
                        "dataset_sequence": ["ds_one"],
                        "reasoning_chain_text": "Step 1",
                    }
                )
            )
            set_runtime_profiles_root(runtime_profiles_root)

            cfg = RunConfig(
                search_tool_mode="ideal",
                search_results_mode="ideal",
                profile_mode="ideal",
            )
            with self.assertRaises(ValueError):
                build_mode_bundle(
                    cfg,
                    data_tools=[],
                    task_context={"task_id": "benchmarks/lakeqa/tasks-mini/tasks/k-1-d-1/task_1.json"},
                )

    def test_preloaded_mode_disables_search_tools_and_injects_authoritative_sources(self):
        with TemporaryDirectory() as tmpdir:
            runtime_profiles_root = Path(tmpdir) / "runtime-profiles"
            self._write_valid_plan(runtime_profiles_root)
            set_runtime_profiles_root(runtime_profiles_root)

            cfg = RunConfig(
                search_tool_mode="preloaded",
                search_results_mode="ideal",
                profile_mode="standard",
            )
            bundle = build_mode_bundle(
                cfg,
                data_tools=[],
                task_context={"task_id": "benchmarks/lakeqa/tasks-mini/tasks/k-1-d-1/task_1.json"},
            )

            tool_names = [tool_obj.tool_spec["name"] for tool_obj in bundle.tools]
            self.assertEqual(bundle.search_tool_names, ())
            self.assertNotIn("search_ideal", tool_names)
            self.assertIn("plan", tool_names)
            # The search_preloaded overlay references the "## PRELOADED DATASETS" heading as
            # documentation (in backticks), but the task-specific URI list only lives in the trailer.
            self.assertNotIn("dataset_id: ds_one | uri: datagov/ds_one/files/rows.txt", bundle.system_prompt)
            self.assertIn("## PRELOADED DATASETS", bundle.task_trailer)
            self.assertIn("dataset_id: ds_one | uri: datagov/ds_one/files/rows.txt", bundle.task_trailer)

    def test_preloaded_mode_fails_fast_without_plan_file(self):
        with TemporaryDirectory() as tmpdir:
            runtime_profiles_root = Path(tmpdir) / "runtime-profiles"
            runtime_profiles_root.mkdir(parents=True, exist_ok=True)
            set_runtime_profiles_root(runtime_profiles_root)

            cfg = RunConfig(
                search_tool_mode="preloaded",
                search_results_mode="naive",
                profile_mode="naive",
            )
            with self.assertRaises(FileNotFoundError):
                build_mode_bundle(
                    cfg,
                    data_tools=[],
                    task_context={"task_id": "benchmarks/lakeqa/tasks-mini/tasks/k-1-d-1/task_missing.json"},
                )

    def test_preloaded_mode_fails_fast_with_empty_source_sequence(self):
        with TemporaryDirectory() as tmpdir:
            runtime_profiles_root = Path(tmpdir) / "runtime-profiles"
            target = runtime_profiles_root / "k-1-d-1"
            target.mkdir(parents=True, exist_ok=True)
            (target / "task_1.json").write_text(
                json.dumps(
                    {
                        "dataset_sequence": ["ds_one"],
                        "source_sequence": [],
                        "reasoning_chain_text": "Step 1",
                    }
                )
            )
            set_runtime_profiles_root(runtime_profiles_root)

            cfg = RunConfig(
                search_tool_mode="preloaded",
                search_results_mode="naive",
                profile_mode="standard",
            )
            with self.assertRaises(ValueError):
                build_mode_bundle(
                    cfg,
                    data_tools=[],
                    task_context={"task_id": "benchmarks/lakeqa/tasks-mini/tasks/k-1-d-1/task_1.json"},
                )

    def test_ideal_computation_mode_replaces_query_and_execute_tools(self):
        from strands import tool

        @tool
        def read_file(dataset_id: str = "", file_path: str = "") -> dict:
            return {}

        @tool
        def query_file(dataset_id: str = "", file_path: str = "", sql: str = "") -> dict:
            return {}

        @tool
        def execute_code(code: str) -> dict:
            return {}

        @tool
        def peek_file(dataset_id: str = "", file_path: str = "") -> dict:
            return {}

        with TemporaryDirectory() as tmpdir:
            runtime_profiles_root = Path(tmpdir) / "runtime-profiles"
            self._write_valid_plan(runtime_profiles_root)
            set_runtime_profiles_root(runtime_profiles_root)

            cfg = RunConfig(
                search_tool_mode="preloaded",
                search_results_mode="naive",
                profile_mode="naive",
                computation_tool_mode="ideal",
            )
            bundle = build_mode_bundle(
                cfg,
                data_tools=[peek_file, read_file, query_file, execute_code],
                task_context={"task_id": "benchmarks/lakeqa/tasks-mini/tasks/k-1-d-1/task_1.json"},
            )

        tool_names = [tool_obj.tool_spec["name"] for tool_obj in bundle.tools]
        self.assertIn("peek_file", tool_names)
        self.assertIn("read_file", tool_names)
        self.assertIn("query_ideal", tool_names)
        self.assertIn("execute_ideal", tool_names)
        self.assertNotIn("query_file", tool_names)
        self.assertNotIn("execute_code", tool_names)
        self.assertEqual(bundle.modes["computation_tool"], "ideal")

    def test_kramabench_ideal_computation_disables_query_tools(self):
        from strands import tool

        @tool
        def read_file(dataset_id: str = "", file_path: str = "") -> dict:
            return {}

        @tool
        def query_file(dataset_id: str = "", file_path: str = "", sql: str = "") -> dict:
            return {}

        @tool
        def execute_code(code: str) -> dict:
            return {}

        with TemporaryDirectory() as tmpdir:
            runtime_profiles_root = Path(tmpdir) / "runtime-profiles"
            target = runtime_profiles_root / "k-1-d-1"
            target.mkdir(parents=True, exist_ok=True)
            (target / "task_1.json").write_text(
                json.dumps(
                    {
                        "dataset_sequence": ["kramabench-demo"],
                        "source_sequence": ["datagov/kramabench-demo/files/rows.csv"],
                        "reasoning_chain_text": "Step 1",
                        "ideal_code": [
                            {
                                "node_id": "1",
                                "dataset_id": "kramabench-demo",
                                "source": "datagov/kramabench-demo/files/rows.csv",
                                "intent": "count rows",
                                "code": "print(3)",
                                "answer": 3,
                            }
                        ],
                    }
                )
            )
            from sana_evaluation.tools.external.ideal import runtime_profile_store

            old_root = runtime_profile_store._KRAMABENCH_RUNTIME_PROFILES_ROOT
            runtime_profile_store._KRAMABENCH_RUNTIME_PROFILES_ROOT = runtime_profiles_root
            try:
                cfg = RunConfig(
                    search_tool_mode="preloaded",
                    search_results_mode="naive",
                    profile_mode="naive",
                    computation_tool_mode="ideal",
                    benchmark="kramabench",
                )
                bundle = build_mode_bundle(
                    cfg,
                    data_tools=[read_file, query_file, execute_code],
                    task_context={"task_id": "benchmarks/kramabench/tasks-mini/tasks/k-1-d-1/task_1.json"},
                )
            finally:
                runtime_profile_store._KRAMABENCH_RUNTIME_PROFILES_ROOT = old_root

        tool_names = [tool_obj.tool_spec["name"] for tool_obj in bundle.tools]
        self.assertIn("read_file", tool_names)
        self.assertIn("execute_ideal", tool_names)
        self.assertNotIn("query_file", tool_names)
        self.assertNotIn("query_ideal", tool_names)
        self.assertNotIn("execute_code", tool_names)
        self.assertNotIn("query_file", bundle.system_prompt)
        self.assertIn("KRAMABENCH COMPUTATION", bundle.system_prompt)

    def test_kramabench_standard_computation_disables_query_file(self):
        from strands import tool

        @tool
        def query_file(dataset_id: str = "", file_path: str = "", sql: str = "") -> dict:
            return {}

        @tool
        def execute_code(code: str) -> dict:
            return {}

        with TemporaryDirectory() as tmpdir:
            runtime_profiles_root = Path(tmpdir) / "runtime-profiles"
            target = runtime_profiles_root / "k-1-d-1"
            target.mkdir(parents=True, exist_ok=True)
            (target / "task_1.json").write_text(
                json.dumps(
                    {
                        "dataset_sequence": ["kramabench-demo"],
                        "source_sequence": ["datagov/kramabench-demo/files/rows.csv"],
                        "reasoning_chain_text": "Step 1",
                    }
                )
            )
            from sana_evaluation.tools.external.ideal import runtime_profile_store

            old_root = runtime_profile_store._KRAMABENCH_RUNTIME_PROFILES_ROOT
            runtime_profile_store._KRAMABENCH_RUNTIME_PROFILES_ROOT = runtime_profiles_root
            try:
                cfg = RunConfig(
                    search_tool_mode="preloaded",
                    search_results_mode="naive",
                    profile_mode="standard",
                    computation_tool_mode="standard",
                    benchmark="kramabench",
                )
                bundle = build_mode_bundle(
                    cfg,
                    data_tools=[query_file, execute_code],
                    task_context={"task_id": "benchmarks/kramabench/tasks-mini/tasks/k-1-d-1/task_1.json"},
                )
            finally:
                runtime_profile_store._KRAMABENCH_RUNTIME_PROFILES_ROOT = old_root

        tool_names = [tool_obj.tool_spec["name"] for tool_obj in bundle.tools]
        self.assertIn("execute_code", tool_names)
        self.assertNotIn("query_file", tool_names)
        self.assertNotIn("query_file", bundle.system_prompt)
        self.assertIn("KRAMABENCH COMPUTATION", bundle.system_prompt)
        self.assertIn("download the file and use `execute_code`", bundle.system_prompt)

    def test_kramabench_ideal_search_and_plan_prompt_omits_query_file(self):
        from strands import tool

        @tool
        def query_file(dataset_id: str = "", file_path: str = "", sql: str = "") -> dict:
            return {}

        @tool
        def execute_code(code: str) -> dict:
            return {}

        with TemporaryDirectory() as tmpdir:
            runtime_profiles_root = Path(tmpdir) / "runtime-profiles"
            target = runtime_profiles_root / "k-1-d-1"
            target.mkdir(parents=True, exist_ok=True)
            (target / "task_1.json").write_text(
                json.dumps(
                    {
                        "dataset_sequence": ["kramabench-demo"],
                        "source_sequence": ["datagov/kramabench-demo/files/rows.csv"],
                        "reasoning_chain_text": "Step 1",
                    }
                )
            )
            from sana_evaluation.tools.external.ideal import runtime_profile_store

            old_root = runtime_profile_store._KRAMABENCH_RUNTIME_PROFILES_ROOT
            runtime_profile_store._KRAMABENCH_RUNTIME_PROFILES_ROOT = runtime_profiles_root
            try:
                cfg = RunConfig(
                    search_tool_mode="ideal",
                    search_results_mode="ideal",
                    profile_mode="ideal",
                    computation_tool_mode="standard",
                    benchmark="kramabench",
                )
                bundle = build_mode_bundle(
                    cfg,
                    data_tools=[query_file, execute_code],
                    task_context={"task_id": "benchmarks/kramabench/tasks-mini/tasks/k-1-d-1/task_1.json"},
                )
            finally:
                runtime_profile_store._KRAMABENCH_RUNTIME_PROFILES_ROOT = old_root

        tool_names = [tool_obj.tool_spec["name"] for tool_obj in bundle.tools]
        self.assertIn("search_ideal", tool_names)
        self.assertIn("plan_ideal", tool_names)
        self.assertIn("execute_code", tool_names)
        self.assertNotIn("query_file", tool_names)
        self.assertNotIn("query_file", bundle.system_prompt)
        self.assertIn("KRAMABENCH COMPUTATION", bundle.system_prompt)
        self.assertIn("download the needed files", bundle.system_prompt)

    def test_prompt_blocks_query_and_execute_for_non_tabular_non_json_sources(self):
        with TemporaryDirectory() as tmpdir:
            runtime_profiles_root = Path(tmpdir) / "runtime-profiles"
            self._write_valid_plan(runtime_profiles_root)
            set_runtime_profiles_root(runtime_profiles_root)

            cfg = RunConfig(
                search_tool_mode="preloaded",
                search_results_mode="naive",
                profile_mode="naive",
                computation_tool_mode="standard",
            )
            bundle = build_mode_bundle(
                cfg,
                data_tools=[],
                task_context={"task_id": "benchmarks/lakeqa/tasks-mini/tasks/k-1-d-1/task_1.json"},
            )

        self.assertIn("## COMPUTATION FILE FAMILY RULE", bundle.system_prompt)
        self.assertIn("Do not use `query_file` or `execute_code`", bundle.system_prompt)
        self.assertIn("non-tabular or non-JSON", bundle.system_prompt)
        self.assertNotIn("query_ideal", bundle.system_prompt)
        self.assertNotIn("execute_ideal", bundle.system_prompt)

    def test_ideal_prompt_blocks_ideal_compute_tools_for_non_tabular_non_json_sources(self):
        from strands import tool

        @tool
        def query_file(dataset_id: str = "", file_path: str = "", sql: str = "") -> dict:
            return {}

        @tool
        def execute_code(code: str) -> dict:
            return {}

        with TemporaryDirectory() as tmpdir:
            runtime_profiles_root = Path(tmpdir) / "runtime-profiles"
            self._write_valid_plan(runtime_profiles_root)
            set_runtime_profiles_root(runtime_profiles_root)

            cfg = RunConfig(
                search_tool_mode="preloaded",
                search_results_mode="naive",
                profile_mode="naive",
                computation_tool_mode="ideal",
            )
            bundle = build_mode_bundle(
                cfg,
                data_tools=[query_file, execute_code],
                task_context={"task_id": "benchmarks/lakeqa/tasks-mini/tasks/k-1-d-1/task_1.json"},
            )

        self.assertIn("## COMPUTATION FILE FAMILY RULE", bundle.system_prompt)
        self.assertIn("Do not use `query_file`, `execute_code`, `query_ideal`, or `execute_ideal`", bundle.system_prompt)
        self.assertIn("non-tabular or non-JSON", bundle.system_prompt)


if __name__ == "__main__":
    unittest.main()
