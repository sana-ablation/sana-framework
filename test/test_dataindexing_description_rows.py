import json
from pathlib import Path

from dataindexing.descriptions.rows import build_row
from dataindexing.sources.parquet_cache import load_descriptions_full


def test_build_row_normalizes_text_and_content():
    row = build_row(
        dataset_uri="s3://bucket/data.csv",
        original_metadata="original   metadata",
        generated_metadata="generated   metadata",
        description="short   description",
    )

    assert row["metadata"] == "generated metadata"
    assert row["content"] == "generated metadata short description"
    assert row["error"] is None


def test_load_descriptions_full_jsonl_skips_error_rows(tmp_path: Path):
    path = tmp_path / "descriptions.jsonl"
    path.write_text(
        json.dumps(
            {
                "dataset_uri": "s3://bucket/ok.csv",
                "generated_metadata": "meta",
                "description": "desc",
                "original_metadata": "orig",
                "error": None,
            }
        )
        + "\n"
        + json.dumps(
            {
                "dataset_uri": "s3://bucket/bad.csv",
                "generated_metadata": "bad",
                "description": "bad",
                "original_metadata": "bad",
                "error": "failed",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert load_descriptions_full(str(path)) == {
        "s3://bucket/ok.csv": {
            "generated_metadata": "meta",
            "description": "desc",
            "original_metadata": "orig",
        }
    }
