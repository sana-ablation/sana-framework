import importlib.util
import io
import json
import unittest
from contextlib import redirect_stderr
from unittest import mock
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory


def _load_profile_datasets_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "profile_datasets.py"
    spec = importlib.util.spec_from_file_location("_test_profile_datasets_module", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


profile_datasets = _load_profile_datasets_module()


class ProfileDatasetsScriptTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self._root = Path(self._tmp.name)
        self._input_path = self._root / "schemas.jsonl"
        self._output_path = self._root / "profiles.jsonl"
        self._descriptions_path = self._root / "descriptions.jsonl"
        self._snippets_path = self._root / "snippets.jsonl"
        self._csv_path = self._root / "rows.csv"
        self._jsonl_path = self._root / "rows.jsonl"
        self._text_path = self._root / "notes.txt"
        self._txt_csv_path = self._root / "rows.txt"
        self._single_column_txt_path = self._root / "one_column.txt"
        self._weird_csv_path = self._root / "weird_headers.csv"
        self._binary_txt_path = self._root / "archive-ish.txt"
        self._zip_path = self._root / "archive.zip"
        self._time_csv_path = self._root / "times.csv"
        self._ragged_csv_path = self._root / "ragged.txt"
        self._metadata_path = self._root / "dcat-us.txt"
        self._xml_path = self._root / "schools.kml"
        self._csv_path.write_text(
            "name,value,day\n"
            "Ada,1,2020-01-01\n"
            "Grace,2,2020-01-03\n"
            ",2,2020-01-05\n"
        )
        self._jsonl_path.write_text(
            '{"id":1,"payload":{"nested":{"message":"'
            + ("x" * 120)
            + '"}}}\n'
            '{"id":2,"payload":{"nested":{"message":"ok"}}}\n'
        )
        self._text_path.write_text(
            "This is a prose file, not a table.\n"
            "It has multiple lines and no stable delimiter.\n"
        )
        self._txt_csv_path.write_text("name,value\nAda,1\nGrace,2\n")
        self._single_column_txt_path.write_text("value\n1\n2\n3\n")
        self._weird_csv_path.write_text(
            "Service Request Number,Date [MM/DD (Local Time)],Case Status\n"
            "SR-001,01/02 09:30,Open\n"
            "SR-002,01/03 10:45,Closed\n"
        )
        self._binary_txt_path.write_bytes(b"PK\x03\x04\x14\x00\x00\x00\x00\x00fake zip bytes,name,value\n")
        self._time_csv_path.write_text("name,duration\nAda,12:30:00\nGrace,13:00:00\n")
        self._ragged_csv_path.write_text("name,value\nAda,1\nGrace\nLinus,3,extra\n")
        self._metadata_path.write_text("title,url\nDCAT-US Schema,https://resources.data.gov\n")
        with zipfile.ZipFile(self._zip_path, "w") as zf:
            zf.writestr("inner.csv", "name,value\nAda,1\n")
        self._xml_path.write_text(
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<kml xmlns="http://www.opengis.net/kml/2.2">\n'
            '  <Document>\n'
            '    <Schema name="schools" id="schools">\n'
            '      <SimpleField name="NCESSCH" type="string"/>\n'
            '      <SimpleField name="CITY" type="string"/>\n'
            '    </Schema>\n'
            '    <Placemark><name>School A</name><ExtendedData><SchemaData>\n'
            '      <SimpleData name="NCESSCH">0001</SimpleData>\n'
            '      <SimpleData name="CITY">Austin</SimpleData>\n'
            '    </SchemaData></ExtendedData></Placemark>\n'
            '    <Placemark><name>School B</name><ExtendedData><SchemaData>\n'
            '      <SimpleData name="NCESSCH">0002</SimpleData>\n'
            '      <SimpleData name="CITY">Boston</SimpleData>\n'
            '    </SchemaData></ExtendedData></Placemark>\n'
            '  </Document>\n'
            '</kml>\n'
        )
        self._write_jsonl(
            self._input_path,
            [
                {
                    "dataset_slug": "example-dataset",
                    "tables": [
                        {
                            "relative_path": "rows.csv",
                            "local_path": str(self._csv_path),
                            "table_kind": "delimited_text",
                            "columns": ["name", "value", "day"],
                            "delimiter": ",",
                        }
                    ],
                }
            ],
        )
        self._write_jsonl(
            self._descriptions_path,
            [{"dataset_uri": str(self._csv_path), "description": "Local example dataset"}],
        )
        self._snippets_path.write_text("")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _write_jsonl(self, path: Path, rows: list[dict]) -> None:
        with path.open("w") as f:
            for row in rows:
                f.write(json.dumps(row))
                f.write("\n")

    def _read_output_rows(self) -> list[dict]:
        with self._output_path.open() as f:
            return [json.loads(line) for line in f if line.strip()]

    def test_parse_args_defaults_to_five_gib_file_cap(self):
        args = profile_datasets._parse_args([])

        self.assertEqual(args.max_file_bytes, 5 * 1024 * 1024 * 1024)

    def test_build_profiles_computes_expected_local_csv_stats(self):
        summary = profile_datasets.build_profiles(
            input_path=self._input_path,
            output_path=self._output_path,
            descriptions_path=self._descriptions_path,
            snippets_path=self._snippets_path,
            parallel=1,
            resume=False,
            bucket="unused",
        )

        self.assertEqual(summary, {"written": 1, "skipped": 0, "errors": 0})
        rows = self._read_output_rows()
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["slug"], "example-dataset")
        self.assertEqual(row["filename"], "rows")
        self.assertEqual(row["family"], "csv")
        self.assertEqual(row["schema_status"], "strict")
        self.assertIs(row["schema_error"], False)
        self.assertEqual(row["row_count"], 3)
        self.assertGreater(row["size_bytes"], 0)
        self.assertEqual(row["llm_description"], "Local example dataset")
        self.assertEqual(row["top_2_rows"][0], {"name": "Ada", "value": 1, "day": "2020-01-01"})
        self.assertEqual(row["top_2_rows"][1], {"name": "Grace", "value": 2, "day": "2020-01-03"})

        columns = {col["name"]: col for col in row["columns"]}
        self.assertEqual(columns["name"]["type"], "string")
        self.assertEqual(columns["name"]["null_rate"], 0.33)
        self.assertEqual(columns["name"]["distinct_count"], 2)
        self.assertIsNone(columns["name"]["min"])
        self.assertIsNone(columns["name"]["max"])
        self.assertIsNone(columns["name"]["mean"])

        self.assertEqual(columns["value"]["type"], "integer")
        self.assertEqual(columns["value"]["distinct_count"], 2)
        self.assertEqual(columns["value"]["min"], 1)
        self.assertEqual(columns["value"]["max"], 2)
        self.assertAlmostEqual(columns["value"]["mean"], 5 / 3, places=6)

        self.assertEqual(columns["day"]["type"], "date")
        self.assertEqual(columns["day"]["min"], "2020-01-01")
        self.assertEqual(columns["day"]["max"], "2020-01-05")
        self.assertEqual(columns["day"]["mean"], "2020-01-03")

    def test_build_profiles_resume_skips_existing_entries(self):
        first = profile_datasets.build_profiles(
            input_path=self._input_path,
            output_path=self._output_path,
            descriptions_path=self._descriptions_path,
            snippets_path=self._snippets_path,
            parallel=1,
            resume=False,
            bucket="unused",
        )
        second = profile_datasets.build_profiles(
            input_path=self._input_path,
            output_path=self._output_path,
            descriptions_path=self._descriptions_path,
            snippets_path=self._snippets_path,
            parallel=1,
            resume=True,
            bucket="unused",
        )

        self.assertEqual(first["written"], 1)
        self.assertEqual(second, {"written": 0, "skipped": 1, "errors": 0})
        self.assertEqual(len(self._read_output_rows()), 1)

    def test_build_profiles_accepts_manifest_rows(self):
        manifest_path = self._root / "manifest.jsonl"
        local_uri = f"s3://lakeqa-yc4103-datalake/datagov/example-dataset/files/rows.csv"
        self._write_jsonl(
            manifest_path,
            [
                {
                    "dataset_id": "example-dataset",
                    "folder": "datagov",
                    "file_path": "files/rows.csv",
                    "s3_uri": local_uri,
                    "source_ref": str(self._csv_path),
                    "local_path": str(self._csv_path),
                    "size_bytes": self._csv_path.stat().st_size,
                }
            ],
        )
        self._write_jsonl(
            self._descriptions_path,
            [{"dataset_uri": local_uri, "description": "Manifest description"}],
        )

        summary = profile_datasets.build_profiles(
            input_path=manifest_path,
            output_path=self._output_path,
            input_kind="manifest",
            descriptions_path=self._descriptions_path,
            snippets_path=self._snippets_path,
            parallel=1,
            resume=False,
            bucket="unused",
        )

        self.assertEqual(summary, {"written": 1, "skipped": 0, "errors": 0})
        row = self._read_output_rows()[0]
        self.assertEqual(row["s3_uri"], local_uri)
        self.assertEqual(row["dataset_id"], "example-dataset")
        self.assertEqual(row["file_path"], "files/rows.csv")
        self.assertEqual(row["llm_description"], "Manifest description")
        self.assertEqual(row["schema_status"], "strict")

    def test_oversized_manifest_file_is_not_parsed(self):
        manifest_path = self._root / "manifest_large.jsonl"
        local_uri = f"s3://lakeqa-yc4103-datalake/datagov/example-dataset/files/large.csv"
        self._write_jsonl(
            manifest_path,
            [
                {
                    "dataset_id": "example-dataset",
                    "file_path": "files/large.csv",
                    "s3_uri": local_uri,
                    "source_ref": str(self._csv_path),
                    "size_bytes": 10,
                }
            ],
        )

        stderr = io.StringIO()
        with redirect_stderr(stderr), mock.patch.object(
            profile_datasets,
            "_build_tabular_profile",
            side_effect=AssertionError("oversized file should not be parsed"),
        ):
            summary = profile_datasets.build_profiles(
                input_path=manifest_path,
                output_path=self._output_path,
                input_kind="manifest",
                descriptions_path=self._descriptions_path,
                snippets_path=self._snippets_path,
                parallel=1,
                max_file_bytes=5,
                resume=False,
                bucket="unused",
            )

        self.assertEqual(summary, {"written": 1, "skipped": 0, "errors": 0})
        row = self._read_output_rows()[0]
        self.assertEqual(row["family"], "csv")
        self.assertEqual(row["schema_status"], "unavailable")
        self.assertIs(row["schema_error"], True)
        self.assertEqual(row["size_bytes"], 10)
        self.assertNotIn("columns", row)
        self.assertNotIn("top_2_rows", row)
        self.assertIn("PROFILE_SCHEMA_ERROR", stderr.getvalue())
        self.assertIn("exceeds max profile file size 5", stderr.getvalue())

    def test_build_profiles_accepts_uri_list_input(self):
        uri_list_path = self._root / "table_profiles_needed.txt"
        uri_list_path.write_text(str(self._csv_path) + "\n")
        self._write_jsonl(
            self._descriptions_path,
            [
                {
                    "dataset_uri": str(self._csv_path),
                    "description": "Canonical description from merged table_descriptions",
                }
            ],
        )

        summary = profile_datasets.build_profiles(
            input_path=uri_list_path,
            output_path=self._output_path,
            input_kind="uri-list",
            descriptions_path=self._descriptions_path,
            snippets_path=self._snippets_path,
            parallel=1,
            resume=False,
            bucket="unused",
        )

        self.assertEqual(summary, {"written": 1, "skipped": 0, "errors": 0})
        row = self._read_output_rows()[0]
        self.assertEqual(row["llm_description"], "Canonical description from merged table_descriptions")
        self.assertEqual(row["filename"], "rows")
        self.assertEqual(row["schema_status"], "strict")

    def test_uri_list_txt_prose_does_not_expose_fake_columns(self):
        uri_list_path = self._root / "table_profiles_needed.txt"
        uri_list_path.write_text(str(self._text_path) + "\n")
        self._write_jsonl(
            self._descriptions_path,
            [{"dataset_uri": str(self._text_path), "description": "Plain-text notes"}],
        )

        summary = profile_datasets.build_profiles(
            input_path=uri_list_path,
            output_path=self._output_path,
            input_kind="uri-list",
            descriptions_path=self._descriptions_path,
            snippets_path=self._snippets_path,
            parallel=1,
            resume=False,
            bucket="unused",
        )

        self.assertEqual(summary, {"written": 1, "skipped": 0, "errors": 0})
        row = self._read_output_rows()[0]
        self.assertEqual(row["family"], "text")
        self.assertEqual(row["schema_status"], "unavailable")
        self.assertIn("snippet", row)
        self.assertIs(row["schema_error"], True)
        self.assertNotIn("candidate_columns", row)
        self.assertNotIn("columns", row)
        self.assertNotIn("top_2_rows", row)

    def test_single_column_txt_is_profiled_as_text_without_columns(self):
        uri_list_path = self._root / "table_profiles_needed.txt"
        uri_list_path.write_text(str(self._single_column_txt_path) + "\n")

        summary = profile_datasets.build_profiles(
            input_path=uri_list_path,
            output_path=self._output_path,
            input_kind="uri-list",
            descriptions_path=self._descriptions_path,
            snippets_path=self._snippets_path,
            parallel=1,
            resume=False,
            bucket="unused",
        )

        self.assertEqual(summary, {"written": 1, "skipped": 0, "errors": 0})
        row = self._read_output_rows()[0]
        self.assertEqual(row["family"], "text")
        self.assertEqual(row["schema_status"], "single_column")
        self.assertIs(row["schema_error"], False)
        self.assertEqual(row["column_count"], 1)
        self.assertNotIn("columns", row)
        self.assertNotIn("candidate_columns", row)

    def test_uri_list_txt_with_delimited_content_still_profiles_as_csv(self):
        uri_list_path = self._root / "table_profiles_needed.txt"
        uri_list_path.write_text(str(self._txt_csv_path) + "\n")

        summary = profile_datasets.build_profiles(
            input_path=uri_list_path,
            output_path=self._output_path,
            input_kind="uri-list",
            descriptions_path=self._descriptions_path,
            snippets_path=self._snippets_path,
            parallel=1,
            resume=False,
            bucket="unused",
        )

        self.assertEqual(summary, {"written": 1, "skipped": 0, "errors": 0})
        row = self._read_output_rows()[0]
        self.assertEqual(row["family"], "csv")
        self.assertEqual(row["schema_status"], "strict")
        self.assertEqual([col["name"] for col in row["columns"]], ["name", "value"])

    def test_ragged_txt_csv_failure_is_text_without_candidate_columns(self):
        uri_list_path = self._root / "table_profiles_needed.txt"
        uri_list_path.write_text(str(self._ragged_csv_path) + "\n")

        stderr = io.StringIO()
        with redirect_stderr(stderr):
            summary = profile_datasets.build_profiles(
                input_path=uri_list_path,
                output_path=self._output_path,
                input_kind="uri-list",
                descriptions_path=self._descriptions_path,
                snippets_path=self._snippets_path,
                parallel=1,
                resume=False,
                bucket="unused",
            )

        self.assertEqual(summary, {"written": 1, "skipped": 0, "errors": 0})
        row = self._read_output_rows()[0]
        self.assertEqual(row["family"], "text")
        self.assertEqual(row["schema_status"], "unavailable")
        self.assertIs(row["schema_error"], True)
        self.assertIn("PROFILE_SCHEMA_ERROR", stderr.getvalue())
        self.assertIn("ragged", stderr.getvalue())
        self.assertNotIn("columns", row)
        self.assertNotIn("candidate_columns", row)

    def test_weird_but_queryable_csv_headers_stay_strict_columns(self):
        uri_list_path = self._root / "table_profiles_needed.txt"
        uri_list_path.write_text(str(self._weird_csv_path) + "\n")

        summary = profile_datasets.build_profiles(
            input_path=uri_list_path,
            output_path=self._output_path,
            input_kind="uri-list",
            descriptions_path=self._descriptions_path,
            snippets_path=self._snippets_path,
            parallel=1,
            resume=False,
            bucket="unused",
        )

        self.assertEqual(summary, {"written": 1, "skipped": 0, "errors": 0})
        row = self._read_output_rows()[0]
        self.assertEqual(row["family"], "csv")
        self.assertEqual(row["schema_status"], "strict")
        self.assertEqual(
            [col["name"] for col in row["columns"]],
            ["Service Request Number", "Date [MM/DD (Local Time)]", "Case Status"],
        )

    def test_metadata_like_file_sets_metadata_status_without_columns(self):
        uri_list_path = self._root / "table_profiles_needed.txt"
        uri_list_path.write_text(str(self._metadata_path) + "\n")

        summary = profile_datasets.build_profiles(
            input_path=uri_list_path,
            output_path=self._output_path,
            input_kind="uri-list",
            descriptions_path=self._descriptions_path,
            snippets_path=self._snippets_path,
            parallel=1,
            resume=False,
            bucket="unused",
        )

        self.assertEqual(summary, {"written": 1, "skipped": 0, "errors": 0})
        row = self._read_output_rows()[0]
        self.assertEqual(row["schema_status"], "metadata")
        self.assertNotIn("columns", row)
        self.assertNotIn("candidate_columns", row)

    def test_xml_profile_exposes_primary_record_columns(self):
        uri_list_path = self._root / "table_profiles_needed.txt"
        uri_list_path.write_text(str(self._xml_path) + "\n")

        summary = profile_datasets.build_profiles(
            input_path=uri_list_path,
            output_path=self._output_path,
            input_kind="uri-list",
            descriptions_path=self._descriptions_path,
            snippets_path=self._snippets_path,
            parallel=1,
            resume=False,
            bucket="unused",
        )

        self.assertEqual(summary, {"written": 1, "skipped": 0, "errors": 0})
        row = self._read_output_rows()[0]
        self.assertEqual(row["family"], "xml")
        self.assertEqual(row["schema_status"], "strict")
        self.assertEqual(row["record_tag"], "Placemark")
        self.assertEqual(row["xml_root_tag"], "kml")
        self.assertEqual(row["xml_preview_mode"], "parsed")
        self.assertEqual(row["records_scanned_for_schema"], 2)
        self.assertNotIn("records_scanned_for_schema_truncated", row)
        self.assertEqual([col["name"] for col in row["columns"]], ["NCESSCH", "CITY", "name"])
        self.assertEqual(row["top_2_rows"][0]["CITY"], "Austin")

    def test_binary_txt_prefix_does_not_generate_fake_columns(self):
        uri_list_path = self._root / "table_profiles_needed.txt"
        uri_list_path.write_text(str(self._binary_txt_path) + "\n")

        summary = profile_datasets.build_profiles(
            input_path=uri_list_path,
            output_path=self._output_path,
            input_kind="uri-list",
            descriptions_path=self._descriptions_path,
            snippets_path=self._snippets_path,
            parallel=1,
            resume=False,
            bucket="unused",
        )

        self.assertEqual(summary, {"written": 1, "skipped": 0, "errors": 0})
        row = self._read_output_rows()[0]
        self.assertEqual(row["family"], "archive")
        self.assertEqual(row["schema_status"], "archive")
        self.assertNotIn("columns", row)
        self.assertNotIn("top_2_rows", row)
        self.assertNotIn("snippet", row)

    def test_zip_profile_marks_archive_without_member_hints(self):
        uri_list_path = self._root / "table_profiles_needed.txt"
        uri_list_path.write_text(str(self._zip_path) + "\n")

        summary = profile_datasets.build_profiles(
            input_path=uri_list_path,
            output_path=self._output_path,
            input_kind="uri-list",
            descriptions_path=self._descriptions_path,
            snippets_path=self._snippets_path,
            parallel=1,
            resume=False,
            bucket="unused",
        )

        self.assertEqual(summary, {"written": 1, "skipped": 0, "errors": 0})
        row = self._read_output_rows()[0]
        self.assertEqual(row["schema_status"], "archive")
        self.assertNotIn("archive_members", row)
        self.assertNotIn("columns", row)

    def test_time_columns_do_not_fail_profile_generation(self):
        manifest_path = self._root / "manifest_time.jsonl"
        local_uri = f"s3://lakeqa-yc4103-datalake/datagov/example-dataset/files/times.csv"
        self._write_jsonl(
            manifest_path,
            [
                {
                    "dataset_id": "example-dataset",
                    "file_path": "files/times.csv",
                    "s3_uri": local_uri,
                    "source_ref": str(self._time_csv_path),
                    "size_bytes": self._time_csv_path.stat().st_size,
                }
            ],
        )

        summary = profile_datasets.build_profiles(
            input_path=manifest_path,
            output_path=self._output_path,
            input_kind="manifest",
            descriptions_path=self._descriptions_path,
            snippets_path=self._snippets_path,
            parallel=1,
            resume=False,
            bucket="unused",
        )

        self.assertEqual(summary, {"written": 1, "skipped": 0, "errors": 0})
        row = self._read_output_rows()[0]
        self.assertEqual(row["schema_status"], "strict")
        columns = {col["name"]: col for col in row["columns"]}
        self.assertEqual(columns["duration"]["type"], "time")
        self.assertIsNone(columns["duration"]["mean"])

    def test_build_profiles_stringifies_nested_top_rows(self):
        manifest_path = self._root / "manifest_nested.jsonl"
        local_uri = f"s3://lakeqa-yc4103-datalake/datagov/example-dataset/files/rows.jsonl"
        self._write_jsonl(
            manifest_path,
            [
                {
                    "dataset_id": "example-dataset",
                    "folder": "datagov",
                    "file_path": "files/rows.jsonl",
                    "s3_uri": local_uri,
                    "source_ref": str(self._jsonl_path),
                    "local_path": str(self._jsonl_path),
                    "size_bytes": self._jsonl_path.stat().st_size,
                    "table_kind": "json",
                }
            ],
        )

        summary = profile_datasets.build_profiles(
            input_path=manifest_path,
            output_path=self._output_path,
            input_kind="manifest",
            descriptions_path=self._descriptions_path,
            snippets_path=self._snippets_path,
            parallel=1,
            resume=False,
            bucket="unused",
        )

        self.assertEqual(summary, {"written": 1, "skipped": 0, "errors": 0})
        row = self._read_output_rows()[0]
        self.assertEqual(row["family"], "json")
        self.assertEqual(row["schema_status"], "strict")
        self.assertIsInstance(row["top_2_rows"][0]["payload"], str)
        self.assertTrue(row["top_2_rows"][0]["payload"].endswith("…"))


if __name__ == "__main__":
    unittest.main()
