from __future__ import annotations

import io
import json
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import aioboto3
import ijson


@dataclass
class S3Config:
    # Bucket / region
    bucket: str = "lakeqa-yc4103-datalake"
    region: str = "us-east-1"
    folders: list[str] = field(default_factory=lambda: ["wikipedia", "datagov"])

    # Concurrency
    max_threads: int = 50           # sync ThreadPoolExecutor workers
    max_async: int = 16             # async semaphore slots

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
    # the heuristics in detect.py before fetching content from S3.
    skip_unimportant_files: bool = False

    # Parquet cache: if set, process.py will load documents from this local
    # parquet file instead of fetching from S3.  Produce the cache on EC2 and
    # copy it here before running the indexers.
    # Schema: dataset_uri (string), content (string)
    parquet_cache_path: str | None = "../datalake_silver.parquet"


ContentFamily = Literal["csv", "json", "text"]
DELIMITERS = (",", "\t", "|", ";")

_SOCRATA_ID_RE = re.compile(r"^[a-z0-9]{4}-[a-z0-9]{4}$")
_PURE_NUMBER_RE = re.compile(r"^\d+(-\d+)?$")
_RANDOM_SUFFIX_RE = re.compile(r"^.+-[A-Za-z0-9]{6}$")
_METADATA_EXACT: frozenset[str] = frozenset(
    {
        "metadata",
        "gmi",
        "open-licenses",
        "legalcode",
        "government-works",
        "index",
        "odc-odbl",
        "wmsserver",
        "resolve",
        "request",
        "edit",
        "search",
        "contact",
        "policyinformation",
        "gmxcodelists",
        "bios",
        "hires",
        "cwhr",
        "license",
        "readme",
        "signed-metadata",
        "headers",
        "dcat-us",
        "catalog",
        "iso",
        "cc-zero",
        "cc-by",
    }
)
_SKIP_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".pdf", ".zip"})


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """Return `(bucket, key)` from an `s3://bucket/key` URI."""
    m = re.match(r"s3://([^/]+)/(.+)", uri.strip())
    if not m:
        raise ValueError(f"Invalid S3 URI: {uri!r}")
    return m.group(1), m.group(2)


def extract_slug(uri: str) -> str:
    """Return the filename stem as a dataset slug."""
    return Path(uri).stem


def is_metadata_filename(filename: str) -> bool:
    """Return true when a filename looks like metadata rather than data."""
    stem = filename.rsplit(".", 1)[0].lower()
    if stem in _METADATA_EXACT:
        return True
    if _SOCRATA_ID_RE.match(stem):
        return True
    if _PURE_NUMBER_RE.match(stem):
        return True
    return bool(_RANDOM_SUFFIX_RE.match(stem))


def should_skip(filename: str) -> bool:
    """Return true for binary/metadata files that should not be ingested."""
    lower = filename.lower()
    if any(lower.endswith(ext) for ext in _SKIP_EXTENSIONS):
        return True
    return is_metadata_filename(lower.rsplit("/", 1)[-1])


def detect_family(content: str) -> ContentFamily:
    """Detect a simple content family from leading text."""
    stripped = content.lstrip()
    if stripped.startswith("{") or stripped.startswith("["):
        return "json"
    first_line = stripped.split("\n", 1)[0]
    if any(d in first_line for d in DELIMITERS):
        return "csv"
    return "text"


def is_table_content(content: str) -> bool:
    """Return true when a text sample looks structurally tabular."""
    lines = [ln.strip() for ln in content.split("\n") if ln.strip()]
    if len(lines) < 3:
        return False

    jsonl = sum(1 for ln in lines[:5] if ln.startswith("{") and ln.endswith("}"))
    if jsonl >= 3:
        return True

    first_char = content.strip()[0]
    if first_char in ("{", "["):
        return False

    for delim in DELIMITERS:
        counts = []
        for ln in lines[:5]:
            clean = re.sub(r'"[^"]*"', "", ln)
            counts.append(clean.count(delim))
        valid = [c for c in counts if c > 0]
        if len(valid) >= 3 and len(set(valid[:3])) == 1 and valid[0] >= 1:
            return True
    return False


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
    if family == "text":
        return peek_text[: cfg.max_text_chars]
    if family == "csv":
        return await s3_fetch_csv(s3, cfg, bucket, key, file_size)
    return await s3_fetch_json(s3, cfg, bucket, key, file_size)


async def fetch_bytes(
    session: aioboto3.Session,
    cfg: S3Config,
    uri: str,
    start: int = 0,
    end: int | None = None,
) -> bytes:
    """Fetch raw bytes from S3."""
    bucket, key = parse_s3_uri(uri)
    async with session.client("s3", region_name=cfg.region) as s3:
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
    async with session.client("s3", region_name=cfg.region) as s3:
        return await s3_fetch(s3, cfg, bucket, key)


async def smart_fetch_whole(session: aioboto3.Session, cfg: S3Config, uri: str) -> str:
    """Fetch a whole object after only a size check."""
    bucket, key = parse_s3_uri(uri)
    async with session.client("s3", region_name=cfg.region) as s3:
        file_size = await s3_head(s3, bucket, key)
        max_bytes = cfg.max_file_size_gb * 1024**3
        if file_size > max_bytes:
            raise ValueError(
                f"File {file_size / 1024**3:.2f} GB exceeds {cfg.max_file_size_gb} GB limit"
            )
        resp = await s3.get_object(Bucket=bucket, Key=key)
        data = await resp["Body"].read()
    return data.decode("utf-8", errors="replace")
