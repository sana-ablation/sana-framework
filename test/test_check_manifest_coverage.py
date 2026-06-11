import importlib.util
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


def _load_coverage_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "check_manifest_coverage.py"
    spec = importlib.util.spec_from_file_location("_test_check_manifest_coverage", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


coverage_script = _load_coverage_module()


class CheckManifestCoverageTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self._root = Path(self._tmp.name)
        self._manifest_path = self._root / "manifest.jsonl"
        self._descriptions_path = self._root / "descriptions.jsonl"
        self._snippets_path = self._root / "snippets.jsonl"
        self._profiles_path = self._root / "profiles.jsonl"
        self._output_prefix = self._root / "coverage"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _write_jsonl(self, path: Path, rows: list[dict]) -> None:
        with path.open("w") as f:
            for row in rows:
                f.write(json.dumps(row))
                f.write("\n")

    def test_cli_defaults_target_tasks_mini_manifest(self):
        args = coverage_script._parse_args([])

        self.assertEqual(
            args.manifest,
            "benchmarks/lakeqa/tasks-mini/artifacts/task_file_manifest.jsonl",
        )
        self.assertIsNone(args.descriptions)
        self.assertEqual(
            coverage_script.default_description_paths(args),
            [Path("benchmarks/lakeqa/tasks-mini/artifacts/descriptions.jsonl")],
        )

    def test_audit_manifest_coverage_counts_missing_rows(self):
        uri1 = "s3://lakeqa-yc4103-datalake/datagov/ds/files/rows.csv"
        uri2 = "s3://lakeqa-yc4103-datalake/wikipedia/Austin,_Texas/content.txt"
        self._write_jsonl(
            self._manifest_path,
            [
                {"dataset_id": "ds", "file_path": "files/rows.csv", "s3_uri": uri1},
                {"dataset_id": "Austin,_Texas", "file_path": "content.txt", "s3_uri": uri2},
            ],
        )
        self._write_jsonl(self._descriptions_path, [{"dataset_uri": uri1, "description": "desc"}])
        self._write_jsonl(self._snippets_path, [{"dataset_uri": uri1, "dataset_snippet": "snippet"}])
        self._write_jsonl(self._profiles_path, [{"s3_uri": uri1, "family": "csv"}])

        summary = coverage_script.audit_manifest_coverage(
            manifest_path=self._manifest_path,
            descriptions_path=self._descriptions_path,
            snippets_path=self._snippets_path,
            profiles_path=self._profiles_path,
            output_prefix=self._output_prefix,
        )

        self.assertEqual(summary["manifest_rows"], 2)
        self.assertEqual(summary["missing_descriptions"], 1)
        self.assertEqual(summary["missing_snippets"], 1)
        self.assertEqual(summary["missing_profiles"], 1)
        self.assertTrue((self._root / "coverage_missing_descriptions.jsonl").exists())
        self.assertTrue((self._root / "coverage_missing_snippets.jsonl").exists())
        self.assertTrue((self._root / "coverage_missing_profiles.jsonl").exists())

    def test_audit_manifest_coverage_rejects_manifest_fallback_descriptions(self):
        uri = "s3://lakeqa-yc4103-datalake/datagov/ds/files/rows.csv"
        self._write_jsonl(
            self._manifest_path,
            [{"dataset_id": "ds", "file_path": "files/rows.csv", "s3_uri": uri}],
        )
        self._write_jsonl(
            self._descriptions_path,
            [
                {
                    "dataset_uri": uri,
                    "description": "fake fallback",
                    "description_source": "tasks_mini_manifest_fallback",
                }
            ],
        )

        with self.assertRaisesRegex(ValueError, "tasks_mini_manifest_fallback"):
            coverage_script.audit_manifest_coverage(
                manifest_path=self._manifest_path,
                descriptions_path=self._descriptions_path,
                snippets_path=self._snippets_path,
                profiles_path=self._profiles_path,
            )


if __name__ == "__main__":
    unittest.main()
