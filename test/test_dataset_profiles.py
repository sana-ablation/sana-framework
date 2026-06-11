import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import sana_evaluation.helper.peek_profile as peek_profile


class DatasetProfilesTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self._root = Path(self._tmp.name)
        self._profiles_path = self._root / "table_profiles.jsonl"

        self._orig_peek_state = {
            "profiles_path": peek_profile._PROFILES_PATH,
            "profiles_loaded": peek_profile._PROFILES_LOADED,
            "profile_by_uri": peek_profile._PROFILE_BY_URI,
            "profile_by_slug_filename": peek_profile._PROFILE_BY_SLUG_FILENAME,
        }

        peek_profile._PROFILES_PATH = self._profiles_path
        peek_profile._PROFILES_LOADED = False
        peek_profile._PROFILE_BY_URI = {}
        peek_profile._PROFILE_BY_SLUG_FILENAME = {}

        self._tabular_uri = (
            "s3://lakeqa-yc4103-datalake/datagov/example-dataset/v1/files/rows.csv"
        )
        self._text_uri = (
            "s3://lakeqa-yc4103-datalake/datagov/example-dataset/v1/files/notes.txt"
        )

    def tearDown(self) -> None:
        peek_profile._PROFILES_PATH = self._orig_peek_state["profiles_path"]
        peek_profile._PROFILES_LOADED = self._orig_peek_state["profiles_loaded"]
        peek_profile._PROFILE_BY_URI = self._orig_peek_state["profile_by_uri"]
        peek_profile._PROFILE_BY_SLUG_FILENAME = self._orig_peek_state["profile_by_slug_filename"]
        self._tmp.cleanup()

    def _write_jsonl(self, path: Path, rows: list[dict]) -> None:
        with path.open("w") as f:
            for row in rows:
                f.write(json.dumps(row))
                f.write("\n")

    def _reset_profile_cache(self) -> None:
        peek_profile._PROFILES_LOADED = False
        peek_profile._PROFILE_BY_URI = {}
        peek_profile._PROFILE_BY_SLUG_FILENAME = {}

    def test_profile_lookup_prefers_precomputed_profiles_jsonl(self):
        expected = {
            "s3_uri": self._tabular_uri,
            "slug": "example-dataset",
            "filename": "rows",
            "family": "csv",
            "size_bytes": 123,
            "row_count": 2,
            "top_2_rows": [{"name": "Ada", "value": 1}, {"name": "Grace", "value": 2}],
            "columns": [
                {
                    "name": "name",
                    "type": "string",
                    "null_rate": 0.0,
                    "distinct_count": 2,
                    "min": None,
                    "max": None,
                    "mean": None,
                }
            ],
        }
        self._write_jsonl(self._profiles_path, [expected])
        self._reset_profile_cache()

        profile = peek_profile.load_dataset_profile(self._tabular_uri)

        self.assertEqual(profile, expected)

    def test_profile_lookup_returns_none_when_profile_missing(self):
        profile = peek_profile.load_dataset_profile(self._tabular_uri)

        self.assertIsNone(profile)

    def test_profile_lookup_returns_raw_profile_without_derived_fields(self):
        expected = {
            "s3_uri": self._tabular_uri,
            "slug": "example-dataset",
            "filename": "rows",
            "family": "csv",
            "columns": [{"name": "name"}],
        }
        self._write_jsonl(self._profiles_path, [expected])
        self._reset_profile_cache()

        profile = peek_profile.load_dataset_profile(self._tabular_uri)

        self.assertEqual(profile, expected)
        self.assertNotIn("schema_columns", profile)
        self.assertNotIn("table_kind", profile)
        self.assertNotIn("snippet", profile)

    def test_profiles_cache_supports_tabular_and_non_tabular_entries(self):
        tabular = {
            "s3_uri": self._tabular_uri,
            "slug": "example-dataset",
            "filename": "rows",
            "family": "csv",
            "size_bytes": 123,
            "row_count": 2,
            "top_2_rows": [{"name": "Ada", "value": 1}, {"name": "Grace", "value": 2}],
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
        }
        non_tabular = {
            "s3_uri": self._text_uri,
            "slug": "example-dataset",
            "filename": "notes",
            "family": "text",
            "size_bytes": 55,
            "llm_description": "Some plain text notes",
            "snippet": "First line of notes",
        }
        self._write_jsonl(self._profiles_path, [tabular, non_tabular])
        self._reset_profile_cache()

        tabular_profile = peek_profile.load_dataset_profile(self._tabular_uri)
        text_profile = peek_profile.load_dataset_profile(self._text_uri)

        self.assertIn("row_count", tabular_profile)
        self.assertIn("top_2_rows", tabular_profile)
        self.assertIn("columns", tabular_profile)
        self.assertNotIn("snippet", tabular_profile)
        self.assertEqual(text_profile["family"], "text")
        self.assertIn("snippet", text_profile)
        self.assertNotIn("top_2_rows", text_profile)
        self.assertNotIn("columns", text_profile)

    def test_profile_lookup_prefers_exact_uri_over_slug_filename_fallback(self):
        canonical = {
            "s3_uri": self._tabular_uri,
            "slug": "example-dataset",
            "filename": "rows",
            "family": "csv",
            "row_count": 2,
        }
        conflicting = {
            "slug": "example-dataset",
            "filename": "rows",
            "family": "csv",
            "row_count": 999,
        }
        self._write_jsonl(self._profiles_path, [canonical, conflicting])
        self._reset_profile_cache()

        profile = peek_profile.load_dataset_profile(self._tabular_uri)

        self.assertEqual(profile["row_count"], 2)

    def test_profile_lookup_uses_slug_filename_fallback(self):
        fallback = {
            "slug": "example-dataset",
            "filename": "rows",
            "family": "csv",
            "row_count": 7,
        }
        self._write_jsonl(self._profiles_path, [fallback])
        self._reset_profile_cache()

        profile = peek_profile.load_dataset_profile(self._tabular_uri)

        self.assertEqual(profile, fallback)

    def test_profile_loader_skips_malformed_rows(self):
        self._profiles_path.write_text(
            "{bad json\n"
            + json.dumps(
                {
                    "s3_uri": self._tabular_uri,
                    "slug": "example-dataset",
                    "filename": "rows",
                    "family": "csv",
                }
            )
            + "\n"
        )
        self._reset_profile_cache()

        profile = peek_profile.load_dataset_profile(self._tabular_uri)

        self.assertEqual(profile["family"], "csv")


if __name__ == "__main__":
    unittest.main()
