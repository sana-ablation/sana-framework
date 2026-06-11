import io
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from sana_evaluation.config import RunConfig
import sana_evaluation.preflight as preflight
from sana_evaluation.tools.external.ideal import search_wrapper


class SanaPreflightTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self._root = Path(self._tmp.name)
        self._db_path = self._root / "lance_data"
        (self._db_path / "lakeqa.lance").mkdir(parents=True, exist_ok=True)

        self._desc_path = self._root / "table_descriptions.jsonl"
        self._snippet_path = self._root / "snippet.jsonl"
        self._schemas_path = self._root / "datagov_tables_schemas_full.jsonl"
        self._profiles_path = self._root / "table_profiles.jsonl"

        self._orig_profiles_path = preflight._PROFILES_PATH
        self._orig_artifact_paths = preflight.artifact_paths
        self._orig_search_wrapper_state = {
            "desc_path": search_wrapper._TABLE_DESCRIPTIONS_PATH,
            "snippet_path": search_wrapper._SNIPPETS_PATH,
            "schemas_path": search_wrapper._SCHEMAS_PATH,
            "desc_loaded": search_wrapper._DESC_CACHE_LOADED,
            "desc_by_uri": search_wrapper._DESC_BY_URI,
            "snippet_loaded": search_wrapper._SNIPPET_CACHE_LOADED,
            "snippet_by_uri": search_wrapper._SNIPPET_BY_URI,
            "schemas_loaded": search_wrapper._SCHEMAS_CACHE_LOADED,
            "schema_by_slug_filename": search_wrapper._SCHEMA_BY_SLUG_FILENAME,
        }

        preflight._PROFILES_PATH = self._profiles_path
        preflight.artifact_paths = lambda _benchmark=None: SimpleNamespace(
            descriptions=self._desc_path,
            snippets=self._snippet_path,
            schemas=self._schemas_path,
            profiles=self._profiles_path,
        )
        search_wrapper._TABLE_DESCRIPTIONS_PATH = self._desc_path
        search_wrapper._SNIPPETS_PATH = self._snippet_path
        search_wrapper._SCHEMAS_PATH = self._schemas_path
        self._reset_search_wrapper_caches()

        self._write_jsonl(
            self._desc_path,
            [{"dataset_uri": "s3://lakeqa-yc4103-datalake/datagov/example/v1/files/rows.csv", "description": "desc"}],
        )
        self._write_jsonl(
            self._snippet_path,
            [{"dataset_uri": "s3://lakeqa-yc4103-datalake/datagov/example/v1/files/rows.csv", "dataset_snippet": "snippet"}],
        )
        self._write_jsonl(
            self._schemas_path,
            [
                {
                    "dataset_slug": "example",
                    "tables": [
                        {
                            "relative_path": "rows.csv",
                            "table_kind": "delimited_text",
                            "columns": ["a", "b"],
                            "delimiter": ",",
                        }
                    ],
                }
            ],
        )

    def tearDown(self) -> None:
        preflight._PROFILES_PATH = self._orig_profiles_path
        preflight.artifact_paths = self._orig_artifact_paths
        search_wrapper._TABLE_DESCRIPTIONS_PATH = self._orig_search_wrapper_state["desc_path"]
        search_wrapper._SNIPPETS_PATH = self._orig_search_wrapper_state["snippet_path"]
        search_wrapper._SCHEMAS_PATH = self._orig_search_wrapper_state["schemas_path"]
        search_wrapper._DESC_CACHE_LOADED = self._orig_search_wrapper_state["desc_loaded"]
        search_wrapper._DESC_BY_URI = self._orig_search_wrapper_state["desc_by_uri"]
        search_wrapper._SNIPPET_CACHE_LOADED = self._orig_search_wrapper_state["snippet_loaded"]
        search_wrapper._SNIPPET_BY_URI = self._orig_search_wrapper_state["snippet_by_uri"]
        search_wrapper._SCHEMAS_CACHE_LOADED = self._orig_search_wrapper_state["schemas_loaded"]
        search_wrapper._SCHEMA_BY_SLUG_FILENAME = self._orig_search_wrapper_state["schema_by_slug_filename"]
        self._tmp.cleanup()

    def _reset_search_wrapper_caches(self) -> None:
        search_wrapper._DESC_CACHE_LOADED = False
        search_wrapper._DESC_BY_URI = {}
        search_wrapper._SNIPPET_CACHE_LOADED = False
        search_wrapper._SNIPPET_BY_URI = {}
        search_wrapper._SCHEMAS_CACHE_LOADED = False
        search_wrapper._SCHEMA_BY_SLUG_FILENAME = {}

    def _write_jsonl(self, path: Path, rows: list[dict]) -> None:
        with path.open("w") as f:
            for row in rows:
                f.write(json.dumps(row))
                f.write("\n")

    def _run_config(self) -> RunConfig:
        run_config = RunConfig(
            search_tool_mode="standard",
            search_results_mode="naive",
            profile_mode="standard",
            search_db_path=str(self._db_path),
        )
        run_config.sana_flags = SimpleNamespace(results=True)
        return run_config

    def test_missing_profiles_file_is_warn_only(self):
        stream = io.StringIO()

        checks = preflight.run_preflight(self._run_config(), [], stream=stream)

        profile_check = next(c for c in checks if c.name == "table_profiles.jsonl")
        self.assertTrue(profile_check.ok)
        self.assertIn("peek_file omits profile", profile_check.detail)
        self.assertIn("Preflight OK", stream.getvalue())

    def test_present_profiles_file_reports_entry_count(self):
        self._write_jsonl(
            self._profiles_path,
            [{"slug": "example", "filename": "rows", "family": "csv", "row_count": 2}],
        )
        stream = io.StringIO()

        checks = preflight.run_preflight(self._run_config(), [], stream=stream)

        profile_check = next(c for c in checks if c.name == "table_profiles.jsonl")
        self.assertTrue(profile_check.ok)
        self.assertEqual(profile_check.detail, f"found: 1 entries")
        self.assertIn("[OK  ] table_profiles.jsonl", stream.getvalue())


if __name__ == "__main__":
    unittest.main()
