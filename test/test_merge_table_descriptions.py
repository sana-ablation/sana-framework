import importlib.util
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


def _load_merge_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "merge_table_descriptions.py"
    spec = importlib.util.spec_from_file_location("_test_merge_table_descriptions", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


merge_script = _load_merge_module()


class MergeTableDescriptionsTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self._root = Path(self._tmp.name)
        self._base_path = self._root / "table_descriptions.jsonl"
        self._core_path = self._root / "tasks_core_quality_file_manifest_descriptions.jsonl"
        self._mini_path = self._root / "tasks_mini_file_manifest_descriptions.jsonl"
        self._output_path = self._root / "merged_table_descriptions.jsonl"
        self._uri_output_path = self._root / "table_profiles_needed.txt"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _write_jsonl(self, path: Path, rows: list[dict]) -> None:
        with path.open("w") as f:
            for row in rows:
                f.write(json.dumps(row))
                f.write("\n")

    def _read_jsonl(self, path: Path) -> list[dict]:
        with path.open() as f:
            return [json.loads(line) for line in f if line.strip()]

    def test_later_task_sources_override_earlier_description_rows(self):
        uri_base = "s3://lakeqa-yc4103-datalake/datagov/base/files/rows.csv"
        uri_shared = "s3://lakeqa-yc4103-datalake/datagov/shared/files/rows.csv"
        uri_core = "s3://lakeqa-yc4103-datalake/datagov/core/files/rows.csv"
        self._write_jsonl(
            self._base_path,
            [
                {
                    "dataset_uri": uri_shared,
                    "metadata": "base metadata",
                    "content": "base content",
                    "original_metadata": "base original",
                    "generated_metadata": "base generated",
                    "description": "base description",
                    "input_tokens": 1,
                    "output_tokens": 2,
                    "input_cost_usd": 0.1,
                    "output_cost_usd": 0.2,
                    "cost_usd": 0.3,
                    "error": None,
                },
                {"dataset_uri": uri_base, "description": "base only"},
            ],
        )
        self._write_jsonl(
            self._core_path,
            [
                {"dataset_uri": uri_core, "description": "core only"},
                {"dataset_uri": uri_shared, "description": "core override"},
            ],
        )
        self._write_jsonl(
            self._mini_path,
            [{"dataset_uri": uri_shared, "description": "tasks mini override"}],
        )

        summary = merge_script.merge_table_descriptions(
            description_paths=[self._base_path, self._core_path, self._mini_path],
            output_path=self._output_path,
            uri_output_path=self._uri_output_path,
        )

        self.assertEqual(summary["written"], 3)
        self.assertEqual(summary["overridden_rows"], 2)
        rows = self._read_jsonl(self._output_path)
        self.assertEqual([row["dataset_uri"] for row in rows], [uri_shared, uri_base, uri_core])
        self.assertEqual(rows[0]["description"], "tasks mini override")
        self.assertEqual(list(rows[0]), list(merge_script.DESCRIPTION_FIELDS))
        self.assertEqual(
            self._uri_output_path.read_text().splitlines(),
            [uri_shared, uri_base, uri_core],
        )

    def test_merge_rejects_manifest_fallback_rows(self):
        uri = "s3://lakeqa-yc4103-datalake/datagov/bad/files/rows.csv"
        self._write_jsonl(
            self._base_path,
            [
                {
                    "dataset_uri": uri,
                    "description": "fake fallback",
                    "description_source": "tasks_mini_manifest_fallback",
                }
            ],
        )

        with self.assertRaisesRegex(ValueError, "tasks_mini_manifest_fallback"):
            merge_script.merge_table_descriptions(
                description_paths=[self._base_path],
                output_path=self._output_path,
                uri_output_path=self._uri_output_path,
            )


if __name__ == "__main__":
    unittest.main()
