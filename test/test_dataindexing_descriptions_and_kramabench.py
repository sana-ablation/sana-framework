import json
import tempfile
from pathlib import Path

import pyarrow.parquet as pq

from dataindexing.descriptions import generator as desc
from dataindexing.sources import kramabench


def test_description_generator_auto_detects_kramabench_manifest():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        manifest = root / "manifest.jsonl"
        manifest.write_text(
            json.dumps(
                {
                    "dataset_uri": "s3://sana-kramabench/datagov/kramabench-env/tables/water.csv",
                    "local_table_path": "tables/water.csv",
                    "columns": ["Year", "Value"],
                }
            )
            + "\n",
            encoding="utf-8",
        )

        assert desc.resolve_input_format(str(manifest), "auto") == "kramabench-manifest-jsonl"


def test_kramabench_manifest_no_infer_writes_description_parquet():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        table = root / "tables" / "water.csv"
        table.parent.mkdir()
        table.write_text("Year,Value\n2024,3.5\n", encoding="utf-8")
        manifest = root / "manifest.jsonl"
        manifest.write_text(
            json.dumps(
                {
                    "dataset_uri": "s3://sana-kramabench/datagov/kramabench-environment-easy-1/tables/water.csv",
                    "source_uri": "s3://sana-kramabench/datagov/kramabench-environment-easy-1/files/water.csv",
                    "dataset_id": "kramabench-environment-easy-1",
                    "domain": "environment",
                    "table_name": "water",
                    "columns": ["Year", "Value"],
                    "row_count": 1,
                    "local_table_path": str(table),
                }
            )
            + "\n",
            encoding="utf-8",
        )
        output = root / "table_descriptions.jsonl"
        parquet = root / "table_descriptions.parquet"

        result = desc.run(
            path=str(manifest),
            output=str(output),
            parquet_output=str(parquet),
            max_words=50,
            n=None,
            infer=False,
            model="gpt-4o-mini",
            concurrency=1,
            input_format="auto",
            uri_field="s3_uri",
            skip_unimportant=False,
        )

        assert result[0] == 1
        assert result[6] == 1
        row = pq.read_table(parquet).to_pylist()[0]
        assert (
            row["dataset_uri"]
            == "s3://sana-kramabench/datagov/kramabench-environment-easy-1/tables/water.csv"
        )
        assert "Kramabench environment table water" in row["description"]
        assert "Year" in row["generated_metadata"]
        assert row["error"] is None


def test_retry_errors_from_jsonl_filters_manifest_rows():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        ok_uri = "s3://sana-kramabench/datagov/kramabench-legal-hard-16/tables/ok.csv"
        err_uri = "s3://sana-kramabench/datagov/kramabench-legal-hard-16/tables/error.csv"
        manifest = root / "manifest.jsonl"
        manifest.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "dataset_uri": ok_uri,
                            "local_table_path": "ok.csv",
                            "columns": ["A"],
                            "table_name": "ok",
                        }
                    ),
                    json.dumps(
                        {
                            "dataset_uri": err_uri,
                            "local_table_path": "error.csv",
                            "columns": ["B"],
                            "table_name": "error",
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        previous = root / "previous.jsonl"
        previous.write_text(
            json.dumps({"dataset_uri": ok_uri, "error": None})
            + "\n"
            + json.dumps({"dataset_uri": err_uri, "error": "FileNotFoundError"})
            + "\n",
            encoding="utf-8",
        )
        output = root / "retry.jsonl"

        result = desc.run(
            path=str(manifest),
            output=str(output),
            parquet_output=None,
            max_words=50,
            n=None,
            infer=False,
            model="gpt-4o-mini",
            concurrency=1,
            input_format="auto",
            uri_field="s3_uri",
            skip_unimportant=False,
            retry_errors_from=str(previous),
        )

        rows = [json.loads(line) for line in output.read_text().splitlines()]
        assert result[0] == 1
        assert [row["dataset_uri"] for row in rows] == [err_uri]


def test_kramabench_extract_writes_csv_artifacts():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        eval_root = root / "eval"
        local = (
            eval_root
            / "other-benchmarks/Kramabench/data/environment/input/water-body-testing-2013.csv"
        )
        local.parent.mkdir(parents=True)
        local.write_text("Year,Violation,Value\n2013,yes,1.5\n2013,no,2.5\n", encoding="utf-8")

        source = "datagov/kramabench-environment-easy-1/files/water-body-testing-2013.csv"
        outputs = kramabench.OutputPaths(
            parquet=root / "kramabench_tables.parquet",
            schemas=root / "kramabench_table_schemas.jsonl",
            manifest=root / "kramabench_table_manifest.jsonl",
            report=root / "kramabench_extract_report.json",
            tables_dir=root / "tables",
        )

        report = kramabench.extract_sources(
            [source],
            eval_root=eval_root,
            outputs=outputs,
            bucket="sana-kramabench",
            max_sample_words=100,
            max_sample_rows=5,
        )

        assert report["written_tables"] == 1
        assert outputs.parquet.exists()
        assert outputs.schemas.exists()
        assert outputs.manifest.exists()
        assert outputs.report.exists()
        assert (
            outputs.tables_dir
            / "datagov/kramabench-environment-easy-1/tables/water-body-testing-2013.csv"
        ).exists()

        parquet_rows = pq.read_table(outputs.parquet).to_pylist()
        assert len(parquet_rows) == 1
        assert "Year" in parquet_rows[0]["content"]
        assert "Violation" in parquet_rows[0]["metadata"]
        assert (
            parquet_rows[0]["dataset_uri"]
            == "s3://sana-kramabench/datagov/kramabench-environment-easy-1/files/water-body-testing-2013.csv"
        )

        schema_doc = json.loads(outputs.schemas.read_text().splitlines()[0])
        assert schema_doc["source_uri"] == "s3://sana-kramabench/" + source
        assert schema_doc["tables"][0]["s3_key"] == source
        assert schema_doc["tables"][0]["columns"] == ["Year", "Violation", "Value"]
