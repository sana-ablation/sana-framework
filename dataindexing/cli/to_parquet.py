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
import sys
import asyncio
import json
import os
import re
from pathlib import Path

import aioboto3
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.asyncio import tqdm

from dataindexing.sources.s3 import S3Config, _client_kwargs, parse_s3_uri, s3_fetch

BUCKET = "lakeqa-yc4103-datalake"

# The shipped table schemas address objects as datagov/<dataset>/v1/files/<name>,
# but the bucket has no version segment: every one of its keys is
# datagov/<dataset>/files/<name>. Checked against the live bucket -- 0 of 4000
# keys contain "/v1/", while 4205 of 4205 schema entries do. The eval-side tools
# never hit this because they build keys from (dataset_id, file_path) rather than
# from the schema's s3_key.
_VERSION_SEGMENT = re.compile(r"/v\d+/(?=files/)")


def _bucket_key(s3_key: str) -> str:
    """Map a schema s3_key onto the key layout the bucket actually uses."""
    return _VERSION_SEGMENT.sub("/", s3_key or "")
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
                    "dataset_uri": f"s3://{BUCKET}/{_bucket_key(table['s3_key'])}",
                    "metadata": table["s3_key"].replace("/", " ").replace(".csv", "").replace(".json", "").replace(".txt", ""),
                    "columns": json.dumps(table["columns"]) if table.get("columns") is not None else None,
                    "table_kind": table.get("table_kind"),
                    "delimiter": table.get("delimiter"),
                })
    return tables


async def fetch_batch(batch: list[dict], cfg: S3Config) -> list[dict | None]:
    sem = asyncio.Semaphore(cfg.max_async)
    session = aioboto3.Session()

    # Same unsigned/signed decision as sources/s3.py -- this module builds its own
    # client rather than going through fetch_bytes, so it has to make it too.
    async with session.client("s3", region_name=cfg.region, **_client_kwargs(cfg)) as s3:

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
    parser.add_argument(
        "--benchmark", default="lakeqa",
        help="Benchmark whose shipped table schemas seed the build (default: lakeqa)")
    parser.add_argument(
        "--schemas", type=str, default=None,
        help="Table-schema JSONL to build from. Defaults to the benchmark's shipped "
             "artifacts/table_schemas_full.jsonl.")
    args = parser.parse_args()

    base = Path(__file__).parent
    # The schema seed used to be read from a hardcoded path next to this script,
    # under the name it carries in the tools-for-lakeagent repo. That file is
    # byte-identical to the artifacts copy this repo already ships, so the
    # pipeline could not run from a clean clone despite having its own input.
    default_schemas = Path("benchmarks") / args.benchmark / "tasks-mini" / "artifacts" / "table_schemas_full.jsonl"
    jsonl_path = Path(args.schemas) if args.schemas else default_schemas
    if not jsonl_path.is_file():
        legacy = base / "datagov_table_schemas_full.jsonl"
        if legacy.is_file():
            jsonl_path = legacy
        else:
            parser.error(
                f"table schemas not found at {jsonl_path}. Pass --schemas, or run from "
                f"the repository root where benchmarks/ lives."
            )
    out_path = args.output or str(base / "datalake_with_schema.parquet")

    print(f"Schemas          : {jsonl_path}")
    tables = load_tables(jsonl_path)
    if args.limit_n:
        tables = tables[: args.limit_n]

    cfg = S3Config()

    print(f"Tables to process: {len(tables)}")
    print(f"Output           : {out_path}")
    print()

    BATCH_SIZE = 100
    written = 0
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
            written += len(valid)
            del results, valid, arrow_table

    failed = len(tables) - written
    print(f"\nWrote {written}/{len(tables)} tables to {out_path}"
          + (f" ({failed} failed)" if failed else ""))

    # A build that fetched nothing used to print "Done." and leave a 0-row
    # parquet, which the description and index stages would then consume without
    # complaint -- an empty lake looks exactly like a lake with no matches. Every
    # failure here is an S3 fetch, so a total failure is a credential or bucket
    # problem, not data.
    if written == 0:
        print(
            f"ERROR: every fetch failed; {out_path} has no rows. Check AWS "
            f"credentials and access to the source bucket.",
            file=sys.stderr,
        )
        sys.exit(1)
    if failed > len(tables) // 2:
        print(
            f"WARNING: {failed} of {len(tables)} tables failed to fetch. The "
            f"parquet is usable but incomplete; downstream artifacts built from "
            f"it will be too.",
            file=sys.stderr,
        )
