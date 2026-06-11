import importlib.util
import sys
import types
import unittest
from io import BytesIO
from contextlib import ExitStack
from pathlib import Path
from unittest import mock


_MODULE_NAMES = [
    "strands",
    "boto3",
    "botocore",
    "botocore.config",
    "botocore.exceptions",
    "duckdb",
    "dotenv",
    "requests",
    "sana_evaluation",
    "sana_evaluation.tools",
    "sana_evaluation.tools.helper",
    "sana_evaluation.tools.helper.detect",
    "sana_evaluation.tools.agent_tools",
    "sana_evaluation.tools.agent_tools_v2",
]


class _Config:
    def __init__(self, *_args, **_kwargs):
        pass


class _ClientError(Exception):
    pass


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_agent_tools_v2_module():
    repo_root = Path(__file__).resolve().parents[1]
    previous = {name: sys.modules.get(name) for name in _MODULE_NAMES}

    fake_strands = types.ModuleType("strands")
    fake_strands.tool = lambda func: func
    sys.modules["strands"] = fake_strands

    fake_boto3 = types.ModuleType("boto3")
    fake_boto3.client = lambda *_args, **_kwargs: object()
    sys.modules["boto3"] = fake_boto3

    fake_botocore = types.ModuleType("botocore")
    fake_botocore.UNSIGNED = "UNSIGNED"
    sys.modules["botocore"] = fake_botocore

    fake_config = types.ModuleType("botocore.config")
    fake_config.Config = _Config
    sys.modules["botocore.config"] = fake_config

    fake_exceptions = types.ModuleType("botocore.exceptions")
    fake_exceptions.ClientError = _ClientError
    sys.modules["botocore.exceptions"] = fake_exceptions

    fake_duckdb = types.ModuleType("duckdb")
    fake_duckdb.DuckDBPyConnection = object
    fake_duckdb.connect = lambda *_args, **_kwargs: None
    sys.modules["duckdb"] = fake_duckdb

    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda: None
    sys.modules["dotenv"] = fake_dotenv

    fake_requests = types.ModuleType("requests")
    sys.modules["requests"] = fake_requests

    package = types.ModuleType("sana_evaluation")
    package.__path__ = [str(repo_root / "sana_evaluation")]
    sys.modules["sana_evaluation"] = package

    tools_package = types.ModuleType("sana_evaluation.tools")
    tools_package.__path__ = [str(repo_root / "sana_evaluation" / "tools")]
    sys.modules["sana_evaluation.tools"] = tools_package

    helper_package = types.ModuleType("sana_evaluation.tools.helper")
    helper_package.__path__ = [str(repo_root / "sana_evaluation" / "tools" / "helper")]
    sys.modules["sana_evaluation.tools.helper"] = helper_package

    _load_module(
        "sana_evaluation.tools.helper.detect",
        repo_root / "sana_evaluation" / "tools" / "helper" / "detect.py",
    )
    _load_module(
        "sana_evaluation.tools.agent_tools",
        repo_root / "sana_evaluation" / "tools" / "agent_tools.py",
    )
    module = _load_module(
        "sana_evaluation.tools.agent_tools_v2",
        repo_root / "sana_evaluation" / "tools" / "agent_tools_v2.py",
    )

    def restore():
        for name, old_value in previous.items():
            if old_value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_value

    return module, restore


class AgentToolsV2EmptyS3Tests(unittest.TestCase):
    def setUp(self):
        self.mod, self.restore_modules = _load_agent_tools_v2_module()
        self.ref = {
            "dataset_id": "datagov/empty-dataset",
            "file_path": "files/empty.txt",
            "s3_uri": "s3://lakeqa-yc4103-datalake/datagov/empty-dataset/files/empty.txt",
            "key": "datagov/empty-dataset/files/empty.txt",
        }

    def tearDown(self):
        self.restore_modules()

    def _apply_empty_object_patches(self, stack: ExitStack):
        stack.enter_context(
            mock.patch.object(self.mod, "_resolve_file_reference", return_value=self.ref)
        )
        stack.enter_context(mock.patch.object(self.mod, "_get_s3_client", return_value=object()))
        stack.enter_context(mock.patch.object(self.mod, "_s3_head", return_value=0))
        return stack.enter_context(
            mock.patch.object(
                self.mod,
                "_s3_range_get",
                side_effect=AssertionError("range GET should not run for empty objects"),
            )
        )

    def test_peek_file_handles_empty_object_without_range_get(self):
        with ExitStack() as stack:
            range_get = self._apply_empty_object_patches(stack)
            result = self.mod.peek_file(s3_uri=self.ref["s3_uri"])

        self.assertNotIn("error", result)
        self.assertEqual(result["family"], "text")
        self.assertEqual(result["size_bytes"], 0)
        self.assertEqual(result["preview_text"], "")
        self.assertEqual(result["row_count_estimate"], 0)
        range_get.assert_not_called()

    def test_peek_file_attaches_selected_profile_fields_when_available(self):
        raw_profile = {
            "s3_uri": self.ref["s3_uri"],
            "dataset_id": self.ref["dataset_id"],
            "file_path": self.ref["file_path"],
            "family": "csv",
            "schema_status": "strict",
            "schema_error": False,
            "row_count": 2,
            "columns": [
                {
                    "name": "value",
                    "type": "integer",
                    "null_rate": 0.0,
                    "distinct_count": 2,
                    "min": 1,
                    "max": 2,
                    "mean": 1.5,
                }
            ],
            "top_2_rows": [{"value": 1}, {"value": 2}],
            "debug_only": "do not expose",
        }
        with ExitStack() as stack:
            self._apply_empty_object_patches(stack)
            stack.enter_context(
                mock.patch.object(
                    self.mod,
                    "load_dataset_profile",
                    return_value=raw_profile,
                )
            )
            result = self.mod.peek_file(s3_uri=self.ref["s3_uri"])

        self.assertEqual(
            result["profile"],
            {
                "row_count": 2,
                "columns": [{"name": "value", "type": "integer"}],
                "top_2_rows": [{"value": 1}, {"value": 2}],
            },
        )

    def test_peek_file_omits_profile_when_cached_profile_has_schema_error(self):
        with ExitStack() as stack:
            self._apply_empty_object_patches(stack)
            stack.enter_context(
                mock.patch.object(
                    self.mod,
                    "load_dataset_profile",
                    return_value={
                        "schema_status": "unavailable",
                        "schema_error": True,
                        "snippet": "not useful enough to expose as a profile",
                    },
                )
            )
            result = self.mod.peek_file(s3_uri=self.ref["s3_uri"])

        self.assertNotIn("profile", result)

    def test_peek_file_omits_non_queryable_cached_profiles(self):
        for schema_status in ("metadata", "archive", "unavailable"):
            with self.subTest(schema_status=schema_status):
                with ExitStack() as stack:
                    self._apply_empty_object_patches(stack)
                    stack.enter_context(
                        mock.patch.object(
                            self.mod,
                            "load_dataset_profile",
                            return_value={
                                "schema_status": schema_status,
                                "schema_error": False,
                                "columns": [{"name": "debug", "type": "string"}],
                            },
                        )
                    )
                    result = self.mod.peek_file(s3_uri=self.ref["s3_uri"])

                self.assertNotIn("profile", result)

    def test_peek_file_single_column_profile_says_single_column_only(self):
        with ExitStack() as stack:
            self._apply_empty_object_patches(stack)
            stack.enter_context(
                mock.patch.object(
                    self.mod,
                    "load_dataset_profile",
                    return_value={
                        "schema_status": "single_column",
                        "schema_error": False,
                        "column_count": 1,
                        "columns": [{"name": "value", "type": "string"}],
                        "top_2_rows": [{"value": "hello"}],
                        "snippet": "hello",
                    },
                )
            )
            result = self.mod.peek_file(s3_uri=self.ref["s3_uri"])

        self.assertEqual(
            result["profile"],
            {
                "column_count": 1,
                "snippet": "hello",
            },
        )

    def test_peek_file_exposes_sampled_profile_fields_separately(self):
        with ExitStack() as stack:
            self._apply_empty_object_patches(stack)
            stack.enter_context(
                mock.patch.object(
                    self.mod,
                    "load_dataset_profile",
                    return_value={
                        "schema_status": "sampled",
                        "schema_error": False,
                        "columns": [
                            {"name": "value", "type": "integer", "null_rate": 0.0},
                        ],
                        "top_2_rows": [{"value": 1}, {"value": 2}],
                    },
                )
            )
            result = self.mod.peek_file(s3_uri=self.ref["s3_uri"])

        self.assertEqual(
            result["profile"],
            {
                "columns": [{"name": "value", "type": "integer"}],
                "top_2_rows": [{"value": 1}, {"value": 2}],
            },
        )

    def test_peek_file_omits_profile_when_loader_fails(self):
        with ExitStack() as stack:
            self._apply_empty_object_patches(stack)
            stack.enter_context(
                mock.patch.object(
                    self.mod,
                    "load_dataset_profile",
                    side_effect=RuntimeError("profile cache unavailable"),
                )
            )
            result = self.mod.peek_file(s3_uri=self.ref["s3_uri"])

        self.assertNotIn("profile", result)

    def test_peek_file_previews_xlsx_without_binary_zip_text(self):
        from openpyxl import Workbook

        xlsx_ref = {
            "dataset_id": "datagov/kramabench-biomedical-easy-9",
            "file_path": "files/1-s2.0-S0092867420301070-mmc3.xlsx",
            "s3_uri": "s3://sana-kramabench/datagov/kramabench-biomedical-easy-9/files/1-s2.0-S0092867420301070-mmc3.xlsx",
            "key": "datagov/kramabench-biomedical-easy-9/files/1-s2.0-S0092867420301070-mmc3.xlsx",
        }
        workbook = Workbook()
        readme = workbook.active
        readme.title = "README"
        readme.append(["note"])
        readme.append(["Supplementary workbook"])
        sheet = workbook.create_sheet("F-SS-phospho")
        sheet.append(["Gene", "FDR.phos"])
        sheet.append(["CBX3", 0.01])
        sheet.append(["TP53", 0.02])
        buffer = BytesIO()
        workbook.save(buffer)
        payload = buffer.getvalue()

        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(self.mod, "_resolve_file_reference", return_value=xlsx_ref))
            stack.enter_context(mock.patch.object(self.mod, "_get_s3_client", return_value=object()))
            stack.enter_context(mock.patch.object(self.mod, "_s3_head", return_value=len(payload)))
            stack.enter_context(mock.patch.object(self.mod, "_s3_range_get", return_value=payload))
            result = self.mod.peek_file(s3_uri=xlsx_ref["s3_uri"], max_rows=2)

        self.assertNotIn("error", result)
        self.assertEqual(result["family"], "xlsx")
        self.assertEqual(result["sheet_names"], ["README", "F-SS-phospho"])
        self.assertIn("F-SS-phospho", result["preview_text"])
        self.assertIn("Gene,FDR.phos", result["preview_text"])
        self.assertNotIn("PK", result["preview_text"])
        self.assertEqual(result["sheets"][1]["header_columns"], ["Gene", "FDR.phos"])

    def test_parse_xml_records_reports_empty_object_as_non_xml_without_range_get(self):
        with ExitStack() as stack:
            range_get = self._apply_empty_object_patches(stack)
            result = self.mod._parse_xml_records_impl(s3_uri=self.ref["s3_uri"])

        self.assertIn("error", result)
        self.assertIn("Detected 'text'", result["error"])
        self.assertNotIn("Range", result["error"])
        range_get.assert_not_called()

    def test_query_file_reports_empty_object_as_unqueryable_without_range_get(self):
        with ExitStack() as stack:
            range_get = self._apply_empty_object_patches(stack)
            result = self.mod._query_file_impl(
                s3_uri=self.ref["s3_uri"],
                sql="SELECT COUNT(*) FROM t",
            )

        self.assertIn("error", result)
        self.assertIn("plain text", result["error"])
        self.assertNotIn("Could not detect file family", result["error"])
        range_get.assert_not_called()


if __name__ == "__main__":
    unittest.main()
