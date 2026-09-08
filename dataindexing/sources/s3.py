from __future__ import annotations

import io
import json
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import os

import boto3
from botocore import UNSIGNED
from botocore.config import Config as BotoConfig

import aioboto3
import ijson

# One implementation, XML-aware. This module used to carry a second copy whose
# ContentFamily had no "xml" member. is_metadata_filename and is_table_content
# are re-exported rather than used here: they are part of this module's
# detection surface for the ingestion writers that read it.
from dataindexing.formats import (  # noqa: F401
    ContentFamily,
    detect_family,
    is_metadata_filename,
    is_table_content,
    should_skip,
)

# The benchmark lake. `LAKEQA_BUCKET` overrides the active bucket at runtime;
# that read belongs to the agent runtime (sana_evaluation/tools/lake.py) and to
# the CLIs, not here, so this module only owns the immutable defaults.
DEFAULT_BUCKET = "lakeqa-yc4103-datalake"
BENCHMARK_BUCKETS = {
    "lakeqa": DEFAULT_BUCKET,
    "kramabench": "sana-kramabench",
}
FOLDERS = ["wikipedia", "datagov"]
REGION = "us-east-1"


@dataclass
class S3Config:
    # Bucket / region
    bucket: str = "lakeqa-yc4103-datalake"
    region: str = "us-east-1"
    folders: list[str] = field(default_factory=lambda: ["wikipedia", "datagov"])

    # Concurrency
    max_threads: int = 50           # sync ThreadPoolExecutor workers
    max_async: int = 16             # async semaphore slots

    # The source bucket is public. The eval-side tools read it unsigned
    # (lake.py builds a signature_version=UNSIGNED client), and this side
    # must be able to as well: signing with absent or stale credentials turns a
    # plain 404 into a 403, which reads as an access problem and sends you
    # looking for credentials that were never needed.
    unsigned: bool = field(
        default_factory=lambda: os.getenv("SANA_S3_UNSIGNED", "1") not in ("0", "false", "False")
    )

    # Size gate
    max_file_size_gb: float = 100.0

    # Phase-1 range-GET for family detection
    peek_bytes: int = 256 * 1024

    # Per-family fetch budgets
    max_csv_rows: int = 10_000
    csv_bytes_per_row: int = 5_000

    max_json_items: int = 10_000
    json_bytes_per_item: int = 10_000

    max_text_chars: int = 50_000

    # Optional filename-level filter for ingestion writers.
    # When enabled, callers can skip metadata-like and binary files using
    # the heuristics in dataindexing/formats.py before fetching content from S3.
    skip_unimportant_files: bool = False

    # Parquet cache: if set, process.py will load documents from this local
    # parquet file instead of fetching from S3.  Produce the cache on EC2 and
    # copy it here before running the indexers.
    # Schema: dataset_uri (string), content (string)
    parquet_cache_path: str | None = "../datalake_silver.parquet"


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """Return `(bucket, key)` from an `s3://bucket/key` URI."""
    m = re.match(r"s3://([^/]+)/(.+)", uri.strip())
    if not m:
        raise ValueError(f"Invalid S3 URI: {uri!r}")
    return m.group(1), m.group(2)


def extract_slug(uri: str) -> str:
    """Return the filename stem as a dataset slug."""
    return Path(uri).stem


async def s3_head(s3, bucket: str, key: str) -> int:
    """Return file size in bytes via HeadObject."""
    meta = await s3.head_object(Bucket=bucket, Key=key)
    return int(meta.get("ContentLength", 0))


async def s3_range_get(s3, bucket: str, key: str, start: int, end: int) -> bytes:
    """Fetch a byte range from S3."""
    resp = await s3.get_object(Bucket=bucket, Key=key, Range=f"bytes={start}-{end}")
    return await resp["Body"].read()


async def s3_stream_chunks(
    s3, bucket: str, key: str, chunk_size: int = 65_536
) -> AsyncIterator[bytes]:
    """Async-iterate over raw body chunks from a full S3 GET."""
    resp = await s3.get_object(Bucket=bucket, Key=key)
    async for chunk in resp["Body"].iter_chunks(chunk_size):
        yield chunk


async def s3_peek(
    s3, cfg: S3Config, bucket: str, key: str, file_size: int
) -> tuple[str, ContentFamily]:
    """Range-GET the first configured bytes and detect content family."""
    end = min(cfg.peek_bytes - 1, file_size - 1)
    data = await s3_range_get(s3, bucket, key, 0, end)
    text = data.decode("utf-8", errors="replace")
    return text, detect_family(text)


async def s3_fetch_text(s3, cfg: S3Config, bucket: str, key: str, file_size: int) -> str:
    """Fetch a bounded text sample."""
    end = min(cfg.peek_bytes - 1, file_size - 1)
    data = await s3_range_get(s3, bucket, key, 0, end)
    return data.decode("utf-8", errors="replace")[: cfg.max_text_chars]


async def s3_fetch_csv(s3, cfg: S3Config, bucket: str, key: str, file_size: int) -> str:
    """Fetch a bounded CSV-like byte range."""
    budget = cfg.max_csv_rows * cfg.csv_bytes_per_row
    end = min(budget - 1, file_size - 1)
    data = await s3_range_get(s3, bucket, key, 0, end)
    return data.decode("utf-8", errors="replace")


async def s3_fetch_json(s3, cfg: S3Config, bucket: str, key: str, file_size: int) -> str:
    """Fetch a bounded JSON-like byte range and return balanced items when possible."""
    budget = cfg.max_json_items * cfg.json_bytes_per_item
    end = min(budget - 1, file_size - 1)
    raw_bytes = await s3_range_get(s3, bucket, key, 0, end)
    safe_text = raw_bytes.decode("utf-8", errors="ignore")

    items = []
    try:
        f = io.StringIO(safe_text)
        for item in ijson.items(f, "item"):
            items.append(item)
            if len(items) >= cfg.max_json_items:
                break
    except Exception:
        pass

    if items:
        return json.dumps(items, ensure_ascii=False)
    return safe_text


async def s3_fetch(s3, cfg: S3Config, bucket: str, key: str) -> str:
    """HeadObject, detect family, and fetch a bounded text representation."""
    max_bytes = cfg.max_file_size_gb * 1024**3
    file_size = await s3_head(s3, bucket, key)
    if file_size > max_bytes:
        raise ValueError(
            f"File {file_size / 1024**3:.2f} GB exceeds {cfg.max_file_size_gb} GB limit"
        )

    peek_text, family = await s3_peek(s3, cfg, bucket, key, file_size)
    # "xml" rides the text budget: ijson cannot parse markup, so the json
    # branch would range-GET the whole json budget only to hand back the same
    # decoded bytes. The trailing return stays json-only by elimination.
    if family in ("text", "xml"):
        return peek_text[: cfg.max_text_chars]
    if family == "csv":
        return await s3_fetch_csv(s3, cfg, bucket, key, file_size)
    return await s3_fetch_json(s3, cfg, bucket, key, file_size)


def _client_kwargs(cfg: S3Config) -> dict:
    """Unsigned access for the public source bucket; signed when asked."""
    if getattr(cfg, "unsigned", False):
        return {"config": BotoConfig(signature_version=UNSIGNED)}
    return {}


def build_s3_client(unsigned: bool):
    """Build a synchronous S3 client in signed or unsigned mode."""
    s3_config = {"addressing_style": "path"}
    if unsigned:
        return boto3.client(
            "s3",
            region_name=REGION,
            config=BotoConfig(signature_version=UNSIGNED, s3=s3_config),
        )

    kwargs: Dict[str, Any] = {
        "region_name": REGION,
        "config": BotoConfig(s3=s3_config),
    }
    access_key = os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    session_token = os.getenv("AWS_SESSION_TOKEN")
    if access_key and secret_key:
        kwargs["aws_access_key_id"] = access_key
        kwargs["aws_secret_access_key"] = secret_key
        if session_token:
            kwargs["aws_session_token"] = session_token
    return boto3.client("s3", **kwargs)


async def fetch_bytes(
    session: aioboto3.Session,
    cfg: S3Config,
    uri: str,
    start: int = 0,
    end: int | None = None,
) -> bytes:
    """Fetch raw bytes from S3."""
    bucket, key = parse_s3_uri(uri)
    async with session.client("s3", region_name=cfg.region, **_client_kwargs(cfg)) as s3:
        if end is None:
            resp = await s3.get_object(Bucket=bucket, Key=key)
        else:
            resp = await s3.get_object(Bucket=bucket, Key=key, Range=f"bytes={start}-{end}")
        return await resp["Body"].read()


async def fetch_whole(session: aioboto3.Session, cfg: S3Config, uri: str) -> str:
    """Fetch a whole S3 object as UTF-8 text."""
    return (await fetch_bytes(session, cfg, uri)).decode("utf-8", errors="replace")


async def smart_fetch(session: aioboto3.Session, cfg: S3Config, uri: str) -> str:
    """Fetch an S3 object through size/family gates."""
    bucket, key = parse_s3_uri(uri)
    async with session.client("s3", region_name=cfg.region, **_client_kwargs(cfg)) as s3:
        return await s3_fetch(s3, cfg, bucket, key)


async def smart_fetch_whole(session: aioboto3.Session, cfg: S3Config, uri: str) -> str:
    """Fetch a whole object after only a size check."""
    bucket, key = parse_s3_uri(uri)
    async with session.client("s3", region_name=cfg.region, **_client_kwargs(cfg)) as s3:
        file_size = await s3_head(s3, bucket, key)
        max_bytes = cfg.max_file_size_gb * 1024**3
        if file_size > max_bytes:
            raise ValueError(
                f"File {file_size / 1024**3:.2f} GB exceeds {cfg.max_file_size_gb} GB limit"
            )
        resp = await s3.get_object(Bucket=bucket, Key=key)
        data = await resp["Body"].read()
    return data.decode("utf-8", errors="replace")
