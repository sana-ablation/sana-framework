import io
import json
import unittest
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from sana_evaluation.config import RunConfig
import sana_evaluation.preflight as preflight
from sana_evaluation.preflight import run_preflight
from sana_evaluation.tools.external.ideal import search_wrapper
from sana_evaluation.tools.external.ideal.runtime_profile_store import set_runtime_profiles_root


class PreflightModeTests(unittest.TestCase):
    def _reset_preflight_state(self) -> None:
        set_runtime_profiles_root("benchmarks/lakeqa/tasks-mini/runtime-profiles")
        preflight._PROFILES_PATH = Path("benchmarks/lakeqa/tasks-mini/artifacts/table_profiles.jsonl")
        search_wrapper.configure_dependency_paths(
            descriptions="benchmarks/lakeqa/tasks-mini/artifacts/descriptions.jsonl",
            snippets="benchmarks/lakeqa/tasks-mini/artifacts/snippets.jsonl",
            schemas="benchmarks/lakeqa/tasks-mini/artifacts/table_schemas_full.jsonl",
        )
        search_wrapper._DESC_CACHE_LOADED = False
        search_wrapper._DESC_BY_URI = {}
        search_wrapper._DESC_ROW_BY_URI = {}

    def setUp(self) -> None:
        self._reset_preflight_state()

    def tearDown(self) -> None:
        self._reset_preflight_state()

    def test_preloaded_mode_ignores_ideal_search_result_artifacts(self):
        cfg = RunConfig(
            search_tool_mode="preloaded",
            search_results_mode="ideal",
            profile_mode="ideal",
        )
        output = io.StringIO()
        checks = run_preflight(
            cfg,
            ["benchmarks/lakeqa/tasks-mini/tasks/k-1-d-1/task_2.json"],
            stream=output,
        )

        names = [check.name for check in checks]
        self.assertIn("prompt:managed.txt", names)
        self.assertIn("prompt:search_preloaded.txt", names)
        self.assertIn("runtime_profile:benchmarks/lakeqa/tasks-mini/tasks/k-1-d-1/task_2.json", names)
        self.assertNotIn("table_profiles.jsonl", names)
        self.assertNotIn("table_schemas_full.jsonl", names)
        self.assertIn("Preflight OK", output.getvalue())

    def test_search_results_ideal_checks_legacy_fallback_artifacts_not_profiles(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            db_path = root / "lance_data"
            (db_path / "lakeqa.lance").mkdir(parents=True)
            desc_path = root / "descriptions.jsonl"
            snippet_path = root / "snippets.jsonl"
            schemas_path = root / "table_schemas_full.jsonl"
            profiles_path = root / "missing_profiles.jsonl"
            uri = "s3://lakeqa-yc4103-datalake/datagov/foo/files/rows.csv"
            desc_path.write_text(json.dumps({"dataset_uri": uri, "description": "desc"}) + "\n")
            snippet_path.write_text(json.dumps({"dataset_uri": uri, "dataset_snippet": "snippet"}) + "\n")
            schemas_path.write_text(
                json.dumps(
                    {
                        "dataset_slug": "foo",
                        "tables": [
                            {
                                "relative_path": "files/rows.csv",
                                "table_kind": "delimited_text",
                                "columns": ["a"],
                            }
                        ],
                    }
                )
                + "\n"
            )
            cfg = RunConfig(
                search_tool_mode="standard",
                search_results_mode="ideal",
                profile_mode="naive",
                search_db_path=str(db_path),
            )

            with ExitStack() as stack:
                stack.enter_context(patch.object(search_wrapper, "_TABLE_DESCRIPTIONS_PATH", desc_path))
                stack.enter_context(patch.object(search_wrapper, "_SNIPPETS_PATH", snippet_path))
                stack.enter_context(patch.object(search_wrapper, "_SCHEMAS_PATH", schemas_path))
                stack.enter_context(patch.object(preflight, "_PROFILES_PATH", profiles_path))
                checks = run_preflight(cfg, [], stream=io.StringIO())

        names = [check.name for check in checks]
        self.assertNotIn("table_profiles.jsonl", names)
        self.assertIn("snippets.jsonl", names)
        self.assertIn("table_schemas_full.jsonl", names)

    def test_kramabench_preflight_uses_kramabench_stub_artifacts(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            runtime_profiles_root = root / "runtime-profiles"
            plan_dir = runtime_profiles_root / "k-3-d-1-s-1"
            plan_dir.mkdir(parents=True)
            (plan_dir / "task_1.json").write_text(
                json.dumps(
                    {
                        "dataset_sequence": ["kramabench-archeology-easy-10"],
                        "source_sequence": ["datagov/kramabench-archeology-easy-10/files/worldcities.csv"],
                        "reasoning_chain_text": ["1. Read the first source.", "2. Compute the answer."],
                    }
                )
            )
            for name in [
                "kramabench_descriptions.jsonl",
                "kramabench_snippets.jsonl",
                "kramabench_tables_schemas_full.jsonl",
            ]:
                (root / name).write_text("")

            from sana_evaluation.tools.external.ideal import runtime_profile_store

            old_root = runtime_profile_store._KRAMABENCH_RUNTIME_PROFILES_ROOT
            try:
                with ExitStack() as stack:
                    stack.enter_context(patch.object(runtime_profile_store, "_KRAMABENCH_RUNTIME_PROFILES_ROOT", runtime_profiles_root))
                    stack.enter_context(patch.object(search_wrapper, "_TABLE_DESCRIPTIONS_PATH", root / "kramabench_descriptions.jsonl"))
                    stack.enter_context(patch.object(search_wrapper, "_SNIPPETS_PATH", root / "kramabench_snippets.jsonl"))
                    stack.enter_context(patch.object(search_wrapper, "_SCHEMAS_PATH", root / "kramabench_tables_schemas_full.jsonl"))
                    stack.enter_context(
                        patch.object(
                            preflight,
                            "_check_kramabench_source_objects",
                            return_value=preflight.PreflightCheck(
                                "kramabench source object existence",
                                True,
                                "mocked",
                            ),
                        )
                    )
                    cfg = RunConfig(
                        search_tool_mode="ideal",
                        search_results_mode="ideal",
                        profile_mode="ideal",
                        benchmark="kramabench",
                    )
                    checks = run_preflight(
                        cfg,
                        ["benchmarks/kramabench/tasks-mini/tasks/k-3-d-1-s-1/task_1.json"],
                        stream=io.StringIO(),
                    )
            finally:
                runtime_profile_store._KRAMABENCH_RUNTIME_PROFILES_ROOT = old_root

        names = [check.name for check in checks]
        self.assertIn("kramabench_descriptions.jsonl (ideal enrichment load)", names)
        self.assertIn("kramabench_snippets.jsonl", names)
        self.assertIn("kramabench_tables_schemas_full.jsonl", names)
        coverage = next(check for check in checks if check.name == "runtime profile source description coverage")
        self.assertTrue(coverage.ok)
        self.assertTrue("covered" in coverage.detail or "skipped coverage" in coverage.detail)

    def test_ideal_computation_preflight_allows_missing_code_and_query_records(self):
        with TemporaryDirectory() as tmpdir:
            runtime_profiles_root = Path(tmpdir) / "runtime-profiles"
            target = runtime_profiles_root / "k-1-d-1"
            target.mkdir(parents=True, exist_ok=True)
            (target / "task_1.json").write_text(
                json.dumps(
                    {
                        "dataset_sequence": ["ds_one"],
                        "source_sequence": ["datagov/ds_one/files/rows.txt"],
                        "reasoning_chain_text": "Step 1",
                    }
                )
            )
            set_runtime_profiles_root(runtime_profiles_root)
            cfg = RunConfig(
                search_tool_mode="preloaded",
                search_results_mode="naive",
                profile_mode="naive",
                computation_tool_mode="ideal",
            )

            checks = run_preflight(
                cfg,
                ["benchmarks/lakeqa/tasks-mini/tasks/k-1-d-1/task_1.json"],
                stream=io.StringIO(),
            )

        by_name = {check.name: check for check in checks}
        self.assertIn("ideal_query:benchmarks/lakeqa/tasks-mini/tasks/k-1-d-1/task_1.json", by_name)
        self.assertIn("ideal_code:benchmarks/lakeqa/tasks-mini/tasks/k-1-d-1/task_1.json", by_name)
        self.assertIn("no authored ideal_query records", by_name["ideal_query:benchmarks/lakeqa/tasks-mini/tasks/k-1-d-1/task_1.json"].detail)
        self.assertIn("no authored ideal_code records", by_name["ideal_code:benchmarks/lakeqa/tasks-mini/tasks/k-1-d-1/task_1.json"].detail)

    def test_ideal_computation_preflight_accepts_code_and_query_records(self):
        with TemporaryDirectory() as tmpdir:
            runtime_profiles_root = Path(tmpdir) / "runtime-profiles"
            target = runtime_profiles_root / "k-1-d-1"
            target.mkdir(parents=True, exist_ok=True)
            (target / "task_1.json").write_text(
                json.dumps(
                    {
                        "dataset_sequence": ["ds_one"],
                        "source_sequence": ["datagov/ds_one/files/rows.txt"],
                        "reasoning_chain_text": "Step 1",
                        "ideal_query": [
                            {
                                "node_id": "1",
                                "dataset_id": "ds_one",
                                "intent": "count rows",
                                "sql": "SELECT COUNT(*) FROM t",
                                "answer": 3,
                            }
                        ],
                        "ideal_code": [
                            {
                                "node_id": "2",
                                "dataset_id": "ds_one",
                                "intent": "sum rows",
                                "code": "print(3)",
                                "answer": 3,
                            }
                        ],
                    }
                )
            )
            set_runtime_profiles_root(runtime_profiles_root)
            cfg = RunConfig(
                search_tool_mode="preloaded",
                search_results_mode="naive",
                profile_mode="naive",
                computation_tool_mode="ideal",
            )

            checks = run_preflight(
                cfg,
                ["benchmarks/lakeqa/tasks-mini/tasks/k-1-d-1/task_1.json"],
                stream=io.StringIO(),
            )

        names = [check.name for check in checks]
        self.assertIn("ideal_query:benchmarks/lakeqa/tasks-mini/tasks/k-1-d-1/task_1.json", names)
        self.assertIn("ideal_code:benchmarks/lakeqa/tasks-mini/tasks/k-1-d-1/task_1.json", names)

    def test_kramabench_ideal_computation_preflight_skips_query_requirement(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            runtime_profiles_root = root / "runtime-profiles"
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
            try:
                with ExitStack() as stack:
                    stack.enter_context(patch.object(runtime_profile_store, "_KRAMABENCH_RUNTIME_PROFILES_ROOT", runtime_profiles_root))
                    stack.enter_context(
                        patch.object(
                            preflight,
                            "_check_kramabench_source_objects",
                            return_value=preflight.PreflightCheck(
                                "kramabench source object existence",
                                True,
                                "mocked",
                            ),
                        )
                    )
                    cfg = RunConfig(
                        search_tool_mode="preloaded",
                        search_results_mode="naive",
                        profile_mode="naive",
                        computation_tool_mode="ideal",
                        benchmark="kramabench",
                    )
                    checks = run_preflight(
                        cfg,
                        ["benchmarks/kramabench/tasks-mini/tasks/k-1-d-1/task_1.json"],
                        stream=io.StringIO(),
                    )
            finally:
                runtime_profile_store._KRAMABENCH_RUNTIME_PROFILES_ROOT = old_root

        by_name = {check.name: check for check in checks}
        self.assertIn("ideal_query:benchmarks/kramabench/tasks-mini/tasks/k-1-d-1/task_1.json", by_name)
        self.assertIn("skipped", by_name["ideal_query:benchmarks/kramabench/tasks-mini/tasks/k-1-d-1/task_1.json"].detail)
        self.assertIn("disabled for kramabench", by_name["ideal_query:benchmarks/kramabench/tasks-mini/tasks/k-1-d-1/task_1.json"].detail)
        self.assertIn("ideal_code:benchmarks/kramabench/tasks-mini/tasks/k-1-d-1/task_1.json", by_name)

    def test_kramabench_source_object_preflight_heads_task_and_plan_sources(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            task_path = root / "benchmarks/kramabench/tasks-mini/tasks" / "k-1-d-1" / "task_1.json"
            task_path.parent.mkdir(parents=True)
            task_path.write_text(
                json.dumps(
                    {
                        "nodes": {
                            "1": {
                                "source": "datagov/kramabench-demo/files/task.csv",
                            }
                        }
                    }
                )
            )
            runtime_profiles_root = root / "runtime-profiles"
            plan_dir = runtime_profiles_root / "k-1-d-1"
            plan_dir.mkdir(parents=True)
            (plan_dir / "task_1.json").write_text(
                json.dumps(
                    {
                        "dataset_sequence": ["kramabench-demo"],
                        "source_sequence": ["datagov/kramabench-demo/files/plan.csv"],
                        "reasoning_chain_text": "Step 1",
                    }
                )
            )

            class FakeS3:
                def __init__(self):
                    self.keys = []

                def head_object(self, Bucket, Key):
                    self.keys.append((Bucket, Key))
                    return {}

            fake_s3 = FakeS3()

            from sana_evaluation.tools.external.ideal import runtime_profile_store

            with ExitStack() as stack:
                stack.enter_context(patch.object(runtime_profile_store, "_KRAMABENCH_RUNTIME_PROFILES_ROOT", runtime_profiles_root))
                stack.enter_context(patch("sana_evaluation.tools.agent_tools._get_s3_client", return_value=fake_s3))
                check = preflight._check_kramabench_source_objects([str(task_path)])

        self.assertTrue(check.ok, check)
        self.assertEqual(
            sorted(fake_s3.keys),
            [
                ("sana-kramabench", "datagov/kramabench-demo/files/plan.csv"),
                ("sana-kramabench", "datagov/kramabench-demo/files/task.csv"),
            ],
        )

    def test_ideal_search_preflight_checks_runtime_profile_source_description_coverage(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            runtime_profiles_root = root / "runtime-profiles"
            target = runtime_profiles_root / "k-1-d-1"
            target.mkdir(parents=True, exist_ok=True)
            (target / "task_1.json").write_text(
                json.dumps(
                    {
                        "dataset_sequence": ["ds_one"],
                        "source_sequence": ["datagov/ds_one/files/rows.txt"],
                        "reasoning_chain_text": "Step 1",
                    }
                )
            )
            table_desc_path = root / "descriptions.jsonl"
            table_desc_path.write_text(
                json.dumps(
                    {
                        "dataset_uri": "s3://lakeqa-yc4103-datalake/datagov/ds_one/files/rows.txt",
                        "description": "Task mini source description",
                    }
                )
                + "\n"
            )
            set_runtime_profiles_root(runtime_profiles_root)
            cfg = RunConfig(
                search_tool_mode="ideal",
                search_results_mode="naive",
                profile_mode="naive",
            )

            with patch.object(search_wrapper, "_TABLE_DESCRIPTIONS_PATH", table_desc_path):
                checks = run_preflight(
                    cfg,
                    ["benchmarks/lakeqa/tasks-mini/tasks/k-1-d-1/task_1.json"],
                    stream=io.StringIO(),
                )

        by_name = {check.name: check for check in checks}
        self.assertEqual(
            by_name["runtime profile source description coverage"].detail,
            "covered 1 runtime-profile source(s)",
        )


if __name__ == "__main__":
    unittest.main()
