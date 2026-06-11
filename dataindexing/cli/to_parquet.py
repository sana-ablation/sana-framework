"""
Build datalake_with_schema.parquet directly from datagov_table_schemas_full.jsonl.

Fetches raw content from S3 for each table entry and writes it alongside schema
metadata. Chunking happens downstream at index-build time (see simple_hybrid_search/process.py).

Usage
-----
    # Full run
    python parquet_writer.py

    # Test with first N tables
    python parquet_writer.py --limit-n 5

    # Custom output path
    python parquet_writer.py --output /tmp/test.parquet
"""
import argparse
import asyncio
import json
import os
import re
from pathlib import Path

import aioboto3
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.asyncio import tqdm

from dataindexing.sources.s3 import S3Config, parse_s3_uri, s3_fetch

BUCKET = "lakeqa-yc4103-datalake"
TOKEN_PATTERN = re.compile(r'(?u)\b[a-zA-Z_]\w+\b')
YEAR_PATTERN = re.compile(r'\b(14|15|16|17|18|19|20)\d{2}\b')


def load_tables(jsonl_path: Path) -> list[dict]:
    """Flatten all table entries from the JSONL into a list of dicts."""
    tables = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            for table in rec.get("tables", []):
                tables.append({
                    "dataset_uri": f"s3://{BUCKET}/{table['s3_key']}",
                    "metadata": table["s3_key"].replace("/", " ").replace(".csv", "").replace(".json", "").replace(".txt", ""),
                    "columns": json.dumps(table["columns"]) if table.get("columns") is not None else None,
                    "table_kind": table.get("table_kind"),
                    "delimiter": table.get("delimiter"),
                })
    return tables


async def fetch_batch(batch: list[dict], cfg: S3Config) -> list[dict | None]:
    sem = asyncio.Semaphore(cfg.max_async)
    session = aioboto3.Session()

    async with session.client("s3", region_name=cfg.region) as s3:

        async def fetch_one(entry: dict) -> dict | None:
            bucket, key = parse_s3_uri(entry["dataset_uri"])
            async with sem:
                try:
                    raw = await s3_fetch(s3, cfg, bucket, key)
                    text = raw.encode("utf-8", errors="ignore").decode("utf-8")
                    tokens = TOKEN_PATTERN.findall(text) + YEAR_PATTERN.findall(text)
                    return {**entry, "content": " ".join(tokens)}
                except Exception as e:
                    print(f"Failed: {entry['dataset_uri']}: {e}")
                    return None

        return await tqdm.gather(*[fetch_one(e) for e in batch])


SCHEMA = pa.schema([
    ("dataset_uri", pa.string()),
    ("metadata",    pa.string()),
    ("content",     pa.string()),
    ("columns",     pa.string()),
    ("table_kind",  pa.string()),
    ("delimiter",   pa.string()),
])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-n", type=int, default=None, help="Only process first N tables (for testing)")
    parser.add_argument("--output",  type=str, default=None, help="Output parquet path (default: datalake_with_schema.parquet)")
    args = parser.parse_args()

    base = Path(__file__).parent
    jsonl_path = base / "datagov_table_schemas_full.jsonl"
    out_path = args.output or str(base / "datalake_with_schema.parquet")

    tables = load_tables(jsonl_path)
    if args.limit_n:
        tables = tables[: args.limit_n]

    cfg = S3Config()

    print(f"Tables to process: {len(tables)}")
    print(f"Output           : {out_path}")
    print()

    BATCH_SIZE = 100
    with pq.ParquetWriter(out_path, SCHEMA, compression="zstd") as writer:
        for i in range(0, len(tables), BATCH_SIZE):
            batch = tables[i : i + BATCH_SIZE]
            print(f"Batch {i // BATCH_SIZE + 1} / {-(-len(tables) // BATCH_SIZE)} ...")
            results = asyncio.run(fetch_batch(batch, cfg))
            valid = [r for r in results if r is not None]
            if not valid:
                continue
            arrow_table = pa.Table.from_pydict(
                {col: [r.get(col) for r in valid] for col in SCHEMA.names},
                schema=SCHEMA,
            )
            writer.write_table(arrow_table)
            del results, valid, arrow_table

    print(f"\nDone. Written: {out_path}")
