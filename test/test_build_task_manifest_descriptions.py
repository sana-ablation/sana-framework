import importlib.util
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


def _load_description_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "build_task_manifest_descriptions.py"
    spec = importlib.util.spec_from_file_location(
        "_test_build_task_manifest_descriptions", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


desc_script = _load_description_module()


class BuildTaskManifestDescriptionsTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self._root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _write_jsonl(self, path: Path, rows: list[dict]) -> None:
        with path.open("w") as f:
            for row in rows:
                f.write(json.dumps(row))
                f.write("\n")

    def test_audit_writes_exact_missing_manifest_rows_from_layered_sources(self):
        manifest = self._root / "manifest.jsonl"
        table_desc = self._root / "descriptions.jsonl"
        core_desc = self._root / "tasks_core_quality_file_manifest_descriptions.jsonl"
        missing = self._root / "missing.jsonl"
        uri1 = "s3://bucket/datagov/ds1/files/rows.csv"
        uri2 = "s3://bucket/datagov/ds2/files/rows.csv"
        uri3 = "s3://bucket/datagov/ds3/files/rows.csv"
        row3 = {"dataset_id": "ds3", "file_path": "files/rows.csv", "s3_uri": uri3}
        self._write_jsonl(
            manifest,
            [
                {"dataset_id": "ds1", "file_path": "files/rows.csv", "s3_uri": uri1},
                {"dataset_id": "ds2", "file_path": "files/rows.csv", "s3_uri": uri2},
                row3,
            ],
        )
        self._write_jsonl(table_desc, [{"dataset_uri": uri1, "description": "table desc"}])
        self._write_jsonl(core_desc, [{"dataset_uri": uri2, "description": "core desc"}])

        summary = desc_script.audit_missing_descriptions(
            manifest_path=manifest,
            seed_description_paths=[table_desc, core_desc],
            missing_output_path=missing,
        )

        self.assertEqual(summary["manifest_rows"], 3)
        self.assertEqual(summary["covered_by_seed_descriptions"], 2)
        self.assertEqual(summary["missing_descriptions"], 1)
        self.assertEqual([json.loads(line) for line in missing.read_text().splitlines()], [row3])

    def test_merge_writes_manifest_order_normal_schema_and_first_seed_wins(self):
        manifest = self._root / "manifest.jsonl"
        table_desc = self._root / "descriptions.jsonl"
        core_desc = self._root / "tasks_core_quality_file_manifest_descriptions.jsonl"
        generated_desc = self._root / "generated.jsonl"
        output = self._root / "tasks_mini_file_manifest_descriptions.jsonl"
        unresolved = self._root / "unresolved.jsonl"
        uri1 = "s3://bucket/datagov/ds1/files/rows.csv"
        uri2 = "s3://bucket/datagov/ds2/files/rows.csv"
        self._write_jsonl(
            manifest,
            [
                {"dataset_id": "ds1", "file_path": "files/rows.csv", "s3_uri": uri1},
                {"dataset_id": "ds2", "file_path": "files/rows.csv", "s3_uri": uri2},
            ],
        )
        self._write_jsonl(
            table_desc,
            [
                {
                    "dataset_uri": uri1,
                    "generated_metadata": "table meta",
                    "description": "table desc",
                    "content": "table meta table desc",
                    "error": None,
                }
            ],
        )
        self._write_jsonl(
            core_desc,
            [
                {
                    "dataset_uri": uri1,
                    "generated_metadata": "core meta",
                    "description": "core desc",
                    "content": "core meta core desc",
                    "error": None,
                }
            ],
        )
        self._write_jsonl(
            generated_desc,
            [
                {
                    "dataset_uri": uri2,
                    "generated_metadata": "generated meta",
                    "description": "generated desc",
                    "content": "generated meta generated desc",
                    "error": None,
                }
            ],
        )

        summary = desc_script.merge_manifest_descriptions(
            manifest_path=manifest,
            seed_description_paths=[table_desc, core_desc],
            generated_description_paths=[generated_desc],
            output_path=output,
            unresolved_output_path=unresolved,
        )

        rows = [json.loads(line) for line in output.read_text().splitlines()]
        self.assertEqual(summary["written"], 2)
        self.assertEqual([row["dataset_uri"] for row in rows], [uri1, uri2])
        self.assertEqual(rows[0]["description"], "table desc")
        self.assertEqual(rows[1]["description"], "generated desc")
        self.assertFalse(unresolved.exists())
        for row in rows:
            self.assertNotIn("description_source", row)
            self.assertEqual(list(row), list(desc_script.DESCRIPTION_FIELDS))

    def test_merge_rejects_manifest_fallback_description_rows(self):
        manifest = self._root / "manifest.jsonl"
        bad_desc = self._root / "tasks_mini_file_manifest_descriptions.jsonl"
        output = self._root / "out.jsonl"
        uri = "s3://bucket/datagov/ds/files/rows.csv"
        self._write_jsonl(
            manifest,
            [{"dataset_id": "ds", "file_path": "files/rows.csv", "s3_uri": uri}],
        )
        self._write_jsonl(
            bad_desc,
            [
                {
                    "dataset_uri": uri,
                    "description": "fake fallback",
                    "description_source": "tasks_mini_manifest_fallback",
                }
            ],
        )

        with self.assertRaisesRegex(ValueError, "tasks_mini_manifest_fallback"):
            desc_script.merge_manifest_descriptions(
                manifest_path=manifest,
                seed_description_paths=[bad_desc],
                generated_description_paths=[],
                output_path=output,
            )

    def test_merge_fails_instead_of_falling_back_when_descriptions_are_missing(self):
        manifest = self._root / "manifest.jsonl"
        output = self._root / "out.jsonl"
        unresolved = self._root / "unresolved.jsonl"
        uri = "s3://bucket/datagov/ds/files/rows.csv"
        manifest_row = {"dataset_id": "ds", "file_path": "files/rows.csv", "s3_uri": uri}
        self._write_jsonl(manifest, [manifest_row])

        with self.assertRaisesRegex(ValueError, "Missing valid generated descriptions"):
            desc_script.merge_manifest_descriptions(
                manifest_path=manifest,
                seed_description_paths=[],
                generated_description_paths=[],
                output_path=output,
                unresolved_output_path=unresolved,
            )

        self.assertFalse(output.exists())
        self.assertEqual(
            [json.loads(line) for line in unresolved.read_text().splitlines()],
            [manifest_row],
        )


if __name__ == "__main__":
    unittest.main()
