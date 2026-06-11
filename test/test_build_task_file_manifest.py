import importlib.util
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


def _load_manifest_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "build_task_file_manifest.py"
    spec = importlib.util.spec_from_file_location("_test_build_task_file_manifest", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


manifest_builder = _load_manifest_module()


class FakeS3Client:
    def __init__(self, contents_by_prefix):
        self.contents_by_prefix = contents_by_prefix

    def list_objects_v2(self, Bucket, Prefix, MaxKeys=1000, ContinuationToken=None):
        items = list(self.contents_by_prefix.get(Prefix, []))
        start = int(ContinuationToken or 0)
        page = items[start : start + MaxKeys]
        response = {}
        if page:
            response["Contents"] = page
        next_start = start + MaxKeys
        if next_start < len(items):
            response["IsTruncated"] = True
            response["NextContinuationToken"] = str(next_start)
        else:
            response["IsTruncated"] = False
        return response


class BuildTaskFileManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self._root = Path(self._tmp.name)
        self._task_root = self._root / "tasks_core_quality"
        (self._task_root / "k-1-d-1").mkdir(parents=True, exist_ok=True)
        (self._task_root / "k-1-d-1" / "task_1.json").write_text(
            json.dumps({"datasets_used": ["alpha-dataset", "Austin,_Texas"]})
        )
        (self._task_root / "k-1-d-1" / "task_2.json").write_text(
            json.dumps({"datasets_used": ["alpha-dataset", "beta-dataset"]})
        )
        self._output_path = self._root / "manifest.jsonl"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _read_rows(self):
        with self._output_path.open() as f:
            return [json.loads(line) for line in f if line.strip()]

    def test_cli_defaults_target_tasks_mini_artifact(self):
        args = manifest_builder._parse_args([])

        self.assertEqual(args.task_root, "benchmarks/lakeqa/tasks-mini/tasks")
        self.assertEqual(
            args.output,
            "benchmarks/lakeqa/tasks-mini/artifacts/task_file_manifest.jsonl",
        )

    def test_build_manifest_expands_dataset_ids_to_files(self):
        fake_s3 = FakeS3Client(
            {
                "datagov/alpha-dataset/": [
                    {"Key": "datagov/alpha-dataset/files/rows.csv", "Size": 10},
                    {"Key": "datagov/alpha-dataset/files/data.json", "Size": 20},
                ],
                "wikipedia/Austin,_Texas/": [
                    {"Key": "wikipedia/Austin,_Texas/content.txt", "Size": 30},
                ],
                "datagov/beta-dataset/": [
                    {"Key": "datagov/beta-dataset/files/rows.parquet", "Size": 40},
                ],
            }
        )

        summary = manifest_builder.build_manifest(
            task_root=self._task_root,
            output_path=self._output_path,
            bucket="lakeqa-yc4103-datalake",
            s3_client=fake_s3,
        )

        self.assertEqual(summary["datasets"], 3)
        self.assertEqual(summary["files"], 4)
        self.assertEqual(summary["unresolved_count"], 0)

        rows = self._read_rows()
        self.assertEqual(
            [(row["dataset_id"], row["file_path"]) for row in rows],
            [
                ("Austin,_Texas", "content.txt"),
                ("alpha-dataset", "files/data.json"),
                ("alpha-dataset", "files/rows.csv"),
                ("beta-dataset", "files/rows.parquet"),
            ],
        )
        alpha_rows = [row for row in rows if row["dataset_id"] == "alpha-dataset"]
        self.assertEqual(alpha_rows[0]["tasks"], ["tasks_core_quality/k-1-d-1/task_1.json", "tasks_core_quality/k-1-d-1/task_2.json"])
        self.assertEqual(alpha_rows[0]["task_count"], 2)
        self.assertEqual(rows[0]["folder"], "wikipedia")
        self.assertTrue(rows[0]["s3_uri"].startswith("s3://lakeqa-yc4103-datalake/"))

    def test_build_manifest_reports_unresolved_dataset_ids(self):
        fake_s3 = FakeS3Client({})

        summary = manifest_builder.build_manifest(
            task_root=self._task_root,
            output_path=self._output_path,
            bucket="lakeqa-yc4103-datalake",
            s3_client=fake_s3,
        )

        self.assertEqual(summary["files"], 0)
        self.assertEqual(summary["unresolved_count"], 3)
        self.assertCountEqual(
            summary["unresolved_dataset_ids"],
            ["Austin,_Texas", "alpha-dataset", "beta-dataset"],
        )


if __name__ == "__main__":
    unittest.main()
