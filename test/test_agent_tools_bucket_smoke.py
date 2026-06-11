import importlib
import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from botocore.exceptions import ClientError


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "smoke_agent_tools_bucket.py"


def load_smoke_module():
    spec = importlib.util.spec_from_file_location("smoke_agent_tools_bucket", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def client_error(code):
    return ClientError({"Error": {"Code": code, "Message": code}}, "ListObjectsV2")


class AgentToolsBucketSmokeTests(unittest.TestCase):
    def test_kramabench_defaults_match_uploaded_sample(self):
        smoke = load_smoke_module()

        target = smoke.build_target("kramabench")

        self.assertEqual(target.bucket, "sana-kramabench")
        self.assertEqual(target.dataset_id, "kramabench-archeology-easy-10")
        self.assertEqual(target.file_path, "files/worldcities.csv")
        self.assertEqual(
            target.s3_uri,
            "s3://sana-kramabench/datagov/kramabench-archeology-easy-10/files/worldcities.csv",
        )

    def test_script_adds_repo_root_to_import_path(self):
        original_path = list(sys.path)
        sys.path = [entry for entry in sys.path if entry != str(REPO_ROOT)]
        try:
            load_smoke_module()
            self.assertIn(str(REPO_ROOT), sys.path)
        finally:
            sys.path = original_path

    def test_expected_error_case_counts_as_passed(self):
        smoke = load_smoke_module()
        case = smoke.ToolCase(
            module_name="agent_tools_v2",
            tool_name="parse_xml_records",
            call=lambda: {"error": "parse_xml_records only supports XML/KML files"},
            call_repr='agent_tools_v2.parse_xml_records(s3_uri="s3://example")',
            expect_error_substring="only supports XML/KML",
        )

        record = smoke.run_case(case)

        self.assertEqual(record["status"], "passed")
        self.assertTrue(record["expected_error"])
        self.assertIn("parse_xml_records", record["call"])

    def test_dependency_stubs_install_identity_strands_tool(self):
        smoke = load_smoke_module()
        original_modules = {name: sys.modules.get(name) for name in ("strands",)}
        sys.modules.pop("strands", None)
        try:
            installed = smoke.install_dependency_stubs(["strands"])

            if "strands" in installed:
                from strands import tool

                def sample():
                    return "ok"

                self.assertIs(tool(sample), sample)
                self.assertIs(tool()(sample), sample)
            else:
                self.assertIsNotNone(importlib.util.find_spec("strands"))
        finally:
            sys.modules.pop("strands", None)
            for name, module in original_modules.items():
                if module is not None:
                    sys.modules[name] = module

    def test_lightweight_package_stubs_avoid_package_init_imports(self):
        smoke = load_smoke_module()
        original_modules = {
            name: sys.modules.get(name)
            for name in ("sana_evaluation", "sana_evaluation.tools")
        }
        for name in original_modules:
            sys.modules.pop(name, None)
        try:
            smoke.install_lightweight_package_stubs()

            self.assertIn("sana_evaluation", sys.modules)
            self.assertIn("sana_evaluation.tools", sys.modules)
            self.assertEqual(
                sys.modules["sana_evaluation"].__path__,
                [str(REPO_ROOT / "sana_evaluation")],
            )
        finally:
            for name in original_modules:
                sys.modules.pop(name, None)
            for name, module in original_modules.items():
                if module is not None:
                    sys.modules[name] = module

    def test_agent_tools_search_skips_access_denied_folder(self):
        smoke = load_smoke_module()
        smoke.install_dependency_stubs(["strands"])
        smoke.install_lightweight_package_stubs()
        original_agent_tools = sys.modules.get("sana_evaluation.tools.agent_tools")
        sys.modules.pop("sana_evaluation.tools.agent_tools", None)
        try:
            agent_tools = importlib.import_module("sana_evaluation.tools.agent_tools")
        finally:
            if original_agent_tools is not None:
                sys.modules["sana_evaluation.tools.agent_tools"] = original_agent_tools
            else:
                sys.modules.pop("sana_evaluation.tools.agent_tools", None)
        fake_s3 = Mock()

        def list_objects_v2(**kwargs):
            if kwargs["Prefix"].startswith("wikipedia/"):
                raise client_error("AccessDenied")
            return {"CommonPrefixes": [{"Prefix": "datagov/kramabench-archeology-easy-10/"}]}

        fake_s3.list_objects_v2.side_effect = list_objects_v2

        with patch.object(agent_tools, "_get_s3_client", return_value=fake_s3):
            result = agent_tools.search(["kramabench-archeology"], limit=5)

        self.assertEqual(result["count"], 1)
        self.assertEqual(result["results"][0]["dataset_id"], "kramabench-archeology-easy-10")

    def test_configure_benchmark_sets_environment_for_spawned_workers(self):
        smoke = load_smoke_module()
        smoke.install_dependency_stubs(["strands"])
        smoke.install_lightweight_package_stubs()
        original_agent_tools = sys.modules.get("sana_evaluation.tools.agent_tools")
        sys.modules.pop("sana_evaluation.tools.agent_tools", None)
        try:
            agent_tools = importlib.import_module("sana_evaluation.tools.agent_tools")
        finally:
            if original_agent_tools is not None:
                sys.modules["sana_evaluation.tools.agent_tools"] = original_agent_tools
            else:
                sys.modules.pop("sana_evaluation.tools.agent_tools", None)

        with patch.dict("os.environ", {}, clear=True):
            bucket = agent_tools.configure_benchmark("kramabench")

            self.assertEqual(bucket, "sana-kramabench")
            self.assertEqual(os.environ["LAKEQA_BENCHMARK"], "kramabench")
            self.assertEqual(os.environ["LAKEQA_BUCKET"], "sana-kramabench")

    def test_write_logs_creates_print_style_transcript_under_run_directory(self):
        smoke = load_smoke_module()
        records = [
            {
                "module": "agent_tools",
                "tool": "search",
                "call": "agent_tools.search(['kramabench-archeology'], limit=5)",
                "status": "passed",
                "result": {"count": 1},
            },
            {
                "module": "agent_tools_v2",
                "tool": "query_file",
                "call": "agent_tools_v2.query_file(s3_uri='s3://example', sql='SELECT * FROM t LIMIT 5')",
                "status": "failed",
                "error": "duckdb unavailable",
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            paths = smoke.write_logs(
                records=records,
                log_dir=Path(tmp),
                run_label="unit-run",
                metadata={"benchmark": "kramabench"},
            )

            transcript_path = paths["transcript_path"]
            self.assertEqual(transcript_path.parent.name, "unit-run")
            self.assertTrue(transcript_path.exists())
            transcript = transcript_path.read_text()

        self.assertIn("CALL:", transcript)
        self.assertIn("agent_tools.search", transcript)
        self.assertIn("RETURNED:", transcript)
        self.assertIn("'count': 1", transcript)
        self.assertIn("ERROR:", transcript)
        self.assertIn("duckdb unavailable", transcript)
        self.assertNotIn("tool_results.jsonl", str(transcript_path))


if __name__ == "__main__":
    unittest.main()
