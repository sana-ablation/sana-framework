import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
UPLOAD_PATH = REPO / "scripts" / "upload_kramabench_sample_to_s3.py"
CORPUS_UPLOAD_PATH = REPO / "scripts" / "upload_kramabench_corpus_to_s3.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class KramabenchS3UploadTests(unittest.TestCase):
    def test_parse_bucket_uri_accepts_bucket_and_optional_prefix(self):
        upload = load_module(UPLOAD_PATH, "kramabench_upload_sample_for_test_parse")

        self.assertEqual(("kramabench", ""), upload.parse_bucket_uri("s3://kramabench"))
        self.assertEqual(
            ("kramabench", "pilot"),
            upload.parse_bucket_uri("s3://kramabench/pilot/"),
        )

    def test_materializes_per_task_dr_input_uploads_without_s3(self):
        upload = load_module(UPLOAD_PATH, "kramabench_upload_sample_for_test_materialize")

        with tempfile.TemporaryDirectory() as tmpdir:
            uploads = upload.materialize_uploads(
                bucket_uri="s3://kramabench",
                staging_dir=Path(tmpdir),
                source_ids=("legal-hard-7",),
            )

            keys = [item.key for item in uploads]
            self.assertIn(
                "datagov/kramabench-legal-hard-7/files/2024_CSN_Top_Three_Identity_Theft_Reports_by_Year.csv",
                keys,
            )
            for item in uploads:
                self.assertEqual("kramabench", item.bucket)
                self.assertTrue(item.local_path.exists(), item)
                self.assertIn("dr-input/legal/legal-hard-7", item.source_path.as_posix())

            legal_upload = next(item for item in uploads if item.key.endswith("Top_Three_Identity_Theft_Reports_by_Year.csv"))
            legal_lines = legal_upload.local_path.read_text(encoding="utf-8-sig").splitlines()
            self.assertEqual("Top Three Identity Theft Reports by Year,,", legal_lines[0])
            self.assertIn("Theft Type,Year,# of Reports", legal_lines)

    def test_upload_uses_injected_client(self):
        upload = load_module(UPLOAD_PATH, "kramabench_upload_sample_for_test_upload")

        calls = []

        class FakeClient:
            def upload_file(self, filename, bucket, key):
                calls.append((Path(filename).name, bucket, key))

        with tempfile.TemporaryDirectory() as tmpdir:
            uploads = upload.materialize_uploads(
                bucket_uri="s3://kramabench/pilot",
                staging_dir=Path(tmpdir),
                source_ids=("legal-hard-7",),
            )
            uploaded = upload.upload_materialized_files(uploads, s3_client=FakeClient())

        self.assertGreater(len(uploaded), 1)
        self.assertIn(
            (
                "2024_CSN_Top_Three_Identity_Theft_Reports_by_Year.csv",
                "kramabench",
                "pilot/datagov/kramabench-legal-hard-7/files/2024_CSN_Top_Three_Identity_Theft_Reports_by_Year.csv",
            ),
            calls,
        )

    def test_upload_can_use_aws_cli_runner(self):
        upload = load_module(UPLOAD_PATH, "kramabench_upload_sample_for_test_aws_cli")

        commands = []

        def fake_run(command):
            commands.append(command)

        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = Path(tmpdir) / "clinical.csv"
            local_path.write_text("idx,Age\nS019,60\n", encoding="utf-8")
            items = [
                upload.UploadItem(
                    local_path=local_path,
                    bucket="kramabench",
                    key="datagov/kramabench-legal-hard-7/files/clinical.csv",
                    source_path=Path("dr-input/legal/legal-hard-7/clinical.csv"),
                )
            ]
            uploaded = upload.upload_materialized_files(
                items,
                uploader="aws-cli",
                run_command=fake_run,
            )

        self.assertEqual(items, uploaded)
        self.assertEqual(
            [
                [
                    "aws",
                    "s3",
                    "cp",
                    str(local_path),
                    "s3://kramabench/datagov/kramabench-legal-hard-7/files/clinical.csv",
                ]
            ],
            commands,
        )

    def test_corpus_upload_uses_candidate_raw_files_and_prefix(self):
        upload = load_module(CORPUS_UPLOAD_PATH, "kramabench_upload_corpus_for_test_materialize")

        with tempfile.TemporaryDirectory() as tmpdir:
            uploads = upload.materialize_corpus_uploads(
                bucket_uri="s3://sana-kramabench",
                candidates_dir=REPO / "benchmarks/kramabench/tasks-mini/tasks/candidates",
                staging_dir=Path(tmpdir),
                source_ids=("archeology-easy-10",),
            )
            keys = [item.key for item in uploads]
            self.assertIn(
                "datagov/kramabench-archeology-easy-10/files/worldcities.csv",
                keys,
            )
            worldcities = next(item for item in uploads if item.key.endswith("worldcities.csv"))
            self.assertEqual("sana-kramabench", worldcities.bucket)
            self.assertTrue(worldcities.local_path.exists(), worldcities)
            self.assertIn("data/archeology/input/worldcities.csv", worldcities.source_path.as_posix())


if __name__ == "__main__":
    unittest.main()
