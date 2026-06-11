import io
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from sana_evaluation.config import RunConfig
from sana_evaluation.preflight import run_preflight
from sana_evaluation.tools.external.ideal.runtime_profile_store import set_runtime_profiles_root, set_task_context


class IdealComputationPreflightTests(unittest.TestCase):
    def tearDown(self) -> None:
        set_runtime_profiles_root("runtime-profiles")
        set_task_context({})

    def test_text_only_plan_with_no_computation_records_passes(self):
        with TemporaryDirectory() as tmpdir:
            runtime_profiles_root = Path(tmpdir) / "runtime-profiles"
            target = runtime_profiles_root / "k-1-d-1"
            target.mkdir(parents=True, exist_ok=True)
            (target / "task_text.json").write_text(
                json.dumps(
                    {
                        "dataset_sequence": ["Text_Source"],
                        "source_sequence": ["wikipedia/Text_Source/content.txt"],
                        "reasoning_chain_text": "1. Read the text source and extract the fact.",
                        "ideal_query": [],
                        "ideal_code": [],
                    }
                )
            )
            set_runtime_profiles_root(runtime_profiles_root)

            stream = io.StringIO()
            checks = run_preflight(
                RunConfig(
                    search_tool_mode="preloaded",
                    search_results_mode="naive",
                    profile_mode="naive",
                    computation_tool_mode="ideal",
                ),
                ["benchmarks/lakeqa/tasks-mini/tasks/k-1-d-1/task_text.json"],
                stream=stream,
            )

            self.assertTrue(all(check.ok for check in checks), stream.getvalue())
            self.assertIn("no authored ideal_query records", stream.getvalue())
            self.assertIn("no authored ideal_code records", stream.getvalue())

