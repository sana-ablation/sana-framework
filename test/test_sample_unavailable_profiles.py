import importlib.util
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


def _load_sample_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "sample_unavailable_profiles.py"
    spec = importlib.util.spec_from_file_location("_test_sample_unavailable_profiles", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


sample_profiles = _load_sample_module()


class SampleUnavailableProfilesTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self._root = Path(self._tmp.name)
        self._csv_path = self._root / "rows.txt"
        self._jsonl_path = self._root / "rows.jsonl"
        self._csv_path.write_text("a,b\n1,2\n3,4\n5,6\n")
        self._jsonl_path.write_text('{"a":1}\n{"a":2,"b":"x"}\n{"a":3}\n')

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _row(self, path: Path, family: str) -> dict:
        return {
            "slug": "example",
            "filename": path.stem,
            "family": family,
            "schema_status": "unavailable",
            "schema_error": True,
            "size_bytes": path.stat().st_size,
            "source_ref": str(path),
            "s3_uri": f"s3://bucket/example/files/{path.name}",
            "llm_description": "Example",
        }

    def test_csv_sampling_trims_partial_tail_and_uses_several_complete_rows(self):
        sample_bytes = len("a,b\n1,2\n3,4\n5")

        updated, error = sample_profiles.retry_sample_row(
            self._row(self._csv_path, "csv"),
            sample_bytes=sample_bytes,
            duckdb_timeout_seconds=30,
        )

        self.assertIsNone(error)
        self.assertEqual(updated["schema_status"], "sampled")
        self.assertIs(updated["schema_error"], False)
        self.assertEqual(updated["top_2_rows"], [{"a": 1, "b": 2}, {"a": 3, "b": 4}])
        self.assertEqual(
            updated["columns"],
            [{"name": "a", "type": "integer"}, {"name": "b", "type": "integer"}],
        )
        self.assertNotIn("sample_columns", updated)

    def test_csv_sampling_rejects_one_complete_data_row(self):
        sample_bytes = len("a,b\n1,2\n3")

        updated, error = sample_profiles.retry_sample_row(
            self._row(self._csv_path, "csv"),
            sample_bytes=sample_bytes,
            duckdb_timeout_seconds=30,
        )

        self.assertEqual(updated["schema_status"], "unavailable")
        self.assertIsNotNone(error)
        self.assertIn("enough complete CSV lines", error)

    def test_jsonl_sampling_uses_multiple_complete_objects(self):
        sample_bytes = len('{"a":1}\n{"a":2,"b":"x"}\n{"a"')

        updated, error = sample_profiles.retry_sample_row(
            self._row(self._jsonl_path, "json"),
            sample_bytes=sample_bytes,
        )

        self.assertIsNone(error)
        self.assertEqual(updated["schema_status"], "sampled")
        self.assertEqual(
            updated["columns"],
            [{"name": "a", "type": "integer"}, {"name": "b", "type": "string"}],
        )
        self.assertEqual(updated["top_2_rows"], [{"a": 1}, {"a": 2, "b": "x"}])

    def test_retry_profiles_preserves_order_and_reports_failures(self):
        rows = [
            {"schema_status": "strict", "schema_error": False, "s3_uri": "s3://bucket/strict"},
            self._row(self._csv_path, "csv"),
            {"schema_status": "unavailable", "schema_error": True, "family": "binary", "s3_uri": "s3://bucket/bin"},
        ]

        updated_rows, failures, summary = sample_profiles.retry_profiles(
            rows,
            sample_bytes=len("a,b\n1,2\n3,4\n5"),
            parallel=1,
            duckdb_timeout_seconds=30,
        )

        self.assertEqual(updated_rows[0], rows[0])
        self.assertEqual(updated_rows[1]["schema_status"], "sampled")
        self.assertEqual(updated_rows[2], rows[2])
        self.assertEqual(summary["candidates"], 2)
        self.assertEqual(summary["sampled"], 1)
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(len(failures), 1)

    def test_schema_status_error_is_retry_candidate(self):
        self.assertTrue(
            sample_profiles.is_retry_candidate(
                {"schema_status": "error", "schema_error": False, "family": "csv"}
            )
        )


if __name__ == "__main__":
    unittest.main()
