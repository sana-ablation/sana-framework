#!/usr/bin/env python3
"""Derive table schemas from what is actually in the bucket.

The shipped table_schemas_full.jsonl has no generator in this repo or any other,
and it does not describe the bucket. It records data.gov's *advertised*
distributions -- a .csv and a .json per dataset -- while the crawler stored one
payload per file slot as .txt whatever it contained. Hence the two symptoms
every reader has been compensating for: a /v1/ path segment the bucket does not
use, and extensions that name a format rather than an object. `load_table_schemas`
strips both back out; so does the join in the original pipeline.

This walks the bucket instead, so a schema cannot claim an object that is not
there. For each stored file it sniffs the content family (never the extension --
everything is .txt) and derives columns and delimiter from the bytes.

Output matches the shape readers already expect:

    {"dataset_slug": ..., "document_id": ..., "tables": [
        {"relative_path", "s3_key", "table_kind", "columns", "delimiter"}]}

    python -m dataindexing.cli.build_table_schemas \\
        --prefix datagov/ --output benchmarks/lakeqa/tasks-mini/artifacts/table_schemas_bucket.jsonl
"""
from __future__ import annotations

import argparse
import csv as csv_mod
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

import asyncio

import aioboto3
import ijson

import boto3
from botocore import UNSIGNED
from botocore.config import Config as BotoConfig

from sana_evaluation.tools.helper.detect import detect_family, should_skip

DEFAULT_BUCKET = "lakeqa-yc4103-datalake"
# 32 KiB is ample for a header line. At ~1M candidate objects the difference
# between this and 256 KiB is roughly 240 GB of range reads on a public bucket
# whose owner pays the egress.
PEEK_BYTES = 32 * 1024

# detect_family's vocabulary -> the table_kind readers expect.
_FAMILY_TO_KIND = {"csv": "delimited_text", "json": "json", "xml": "xml", "text": "text"}
_DELIMITERS = (",", "\t", "|", ";")


def _client(unsigned: bool = True):
    cfg = BotoConfig(signature_version=UNSIGNED) if unsigned else None
    return boto3.client("s3", region_name="us-east-1", config=cfg)


def _sniff_delimiter(first_line: str) -> str | None:
    best, best_count = None, 0
    cleaned = first_line.replace('""', "")
    for d in _DELIMITERS:
        n = cleaned.count(d)
        if n > best_count:
            best, best_count = d, n
    return best



_MIN_CSV_COLUMNS = 3
# A real header is not thousands of fields; a five-figure count means the
# delimiter matched something that is not a table at all.
_MAX_COLUMNS = 512

# Binary payloads stored under a .txt name. should_skip filters by filename, so
# it cannot see these: the sample turned up ZIP archives whose compressed bytes
# happen to contain commas, which parsed into a 2180-column "header".
_BINARY_MAGIC = (b"PK\x03\x04", b"\x1f\x8b", b"%PDF", b"\x89PNG", b"\xff\xd8\xff", b"BZh")


def _looks_binary(head: str) -> bool:
    raw = head[:2048].encode("utf-8", errors="surrogateescape")
    if any(raw.startswith(sig) for sig in _BINARY_MAGIC):
        return True
    if "\x00" in head[:2048]:
        return True
    # Decoded bytes that were not valid UTF-8 land as replacement characters;
    # a text file has almost none.
    sample = head[:2048]
    if sample and sum(1 for c in sample if c == "\ufffd") / len(sample) > 0.05:
        return True
    return False


def _field_is_identifier_like(name: str) -> bool:
    """A header field names a column; a data field is a value."""
    name = str(name or "").strip()
    if not name or len(name) > 60:
        return False
    if any(mark in name for mark in (".", ";", "?", "!")):
        return False          # "Mark D." is a name, not a column
    return len([w for w in name.replace("_", " ").replace("-", " ").split() if w]) <= 4


def _looks_like_header(columns: list[str]) -> bool:
    """Decide whether a parsed first line is a header rather than a data row.

    Two failure modes on this corpus rule out the obvious tests. detect_family
    calls any line containing a comma "csv", so a list of "Surname, Given"
    author names reads as a two-column table whose "header" is really its first
    row. And detect.is_table_content, which checks delimiter consistency across
    lines, rejects genuine tables here because a quoted WKT geometry spans
    several lines and its commas are not stripped line-by-line.

    What separates them is the shape of the first line itself: a header is
    several short identifier-like fields, and a data row is not. This is a
    heuristic, not a parser -- it will not save a file whose header is genuinely
    absent.
    """
    if not _MIN_CSV_COLUMNS <= len(columns) <= _MAX_COLUMNS:
        return False
    # One clearly prosaic field condemns the whole line: a title split on its
    # commas yields a few short fragments alongside one long one, which a
    # per-field majority vote would otherwise pass.
    for name in columns:
        if len(str(name).split()) > 5:
            return False
    ok = sum(1 for c in columns if _field_is_identifier_like(c))
    return ok >= max(_MIN_CSV_COLUMNS, int(0.8 * len(columns)))


# JSON-LD envelopes (dcat catalogue records) and service descriptors are
# structured, but they describe a service or a catalogue rather than holding
# rows. They are not filtered by should_skip, which only sees the filename.
_JSONLD_KEYS = {"@context", "@id", "@type"}
_DESCRIPTOR_HINTS = {"capabilities", "supportedQueryFormats", "advancedQueryCapabilities"}


def _stream_first_object(head: str) -> dict | None:
    """Pull the first record's keys out of a JSON document too large to parse.

    ijson reads incrementally, so a truncated prefix is enough to reach the
    first object. Prefixes are tried outermost-first: GeoJSON feature
    properties, then a plain array of records.
    """
    raw = head.encode("utf-8", errors="replace")
    for prefix in ("features.item.properties", "item"):
        try:
            first = next(ijson.items(io.BytesIO(raw), prefix), None)
        except Exception:
            continue
        if isinstance(first, dict) and first:
            return first
    return None


def _is_data_schema(columns: list[str]) -> bool:
    """Reject structured payloads that are metadata rather than data."""
    if len(columns) < _MIN_CSV_COLUMNS:
        # A single-key object is an API envelope ({"tags": [...]}), not a table.
        return False
    names = set(columns)
    if len(_JSONLD_KEYS & names) >= 2:
        return False
    return not (_DESCRIPTOR_HINTS & names)


def derive_table(key: str, head: str) -> dict | None:
    """Return a schema record for one stored object, or None if it has no table."""
    if _looks_binary(head):
        return None
    family = detect_family(head)
    kind = _FAMILY_TO_KIND.get(family, "text")
    columns: list[str] | None = None
    delimiter: str | None = None

    if family == "csv":
        first_line = head.lstrip().split("\n", 1)[0]
        delimiter = _sniff_delimiter(first_line)
        if delimiter:
            try:
                columns = next(csv_mod.reader(io.StringIO(first_line), delimiter=delimiter))
                columns = [c.strip() for c in columns if c.strip()]
            except Exception:
                columns = None
    elif family == "json":
        try:
            doc = json.loads(head)
        except json.JSONDecodeError:
            # Truncated at PEEK_BYTES, or JSON Lines. Try the first line alone
            # (JSON Lines), then stream the prefix (pretty-printed documents far
            # larger than the peek -- a GeoJSON FeatureCollection is typically
            # megabytes, so it never parses whole and would otherwise be lost).
            try:
                doc = json.loads(head.lstrip().split("\n", 1)[0])
            except json.JSONDecodeError:
                doc = _stream_first_object(head)
        if isinstance(doc, dict):
            # GeoJSON keeps its fields under features[].properties.
            feats = doc.get("features")
            if isinstance(feats, list) and feats and isinstance(feats[0], dict):
                props = feats[0].get("properties")
                columns = sorted(props) if isinstance(props, dict) else sorted(doc)
                kind = "geojson"
            else:
                columns = sorted(doc)
        elif isinstance(doc, list) and doc and isinstance(doc[0], dict):
            columns = sorted(doc[0])

    if family == "csv" and columns and not _looks_like_header(columns):
        return None
    if columns and not _is_data_schema(columns):
        return None
    if not columns:
        return None
    return {
        "relative_path": key.split("/files/", 1)[-1] if "/files/" in key else key.rsplit("/", 1)[-1],
        "s3_key": key,
        "table_kind": kind,
        "columns": columns,
        "delimiter": delimiter,
    }


class FetchFailed(Exception):
    """A file could not be read, as distinct from holding no table."""


async def _peek(s3, sem, bucket: str, key: str, attempts: int = 3) -> str:
    last: Exception | None = None
    for attempt in range(attempts):
        async with sem:
            try:
                obj = await s3.get_object(Bucket=bucket, Key=key,
                                          Range=f"bytes=0-{PEEK_BYTES - 1}")
                body = await obj["Body"].read()
                return body.decode("utf-8", errors="replace")
            except Exception as exc:  # noqa: BLE001 - re-raised below
                last = exc
        await asyncio.sleep(0.5 * (2 ** attempt))
    raise FetchFailed(f"{key}: {last}")


async def _derive_dataset(s3, sem, bucket: str, slug: str,
                          keys: list[str]) -> tuple[list[dict], list[str]]:
    """Return (schemas, failures) for one dataset.

    The two are kept apart deliberately. Swallowing a fetch error as "no table
    here" and then checkpointing the dataset as done is how a transient S3 blip
    turns into a permanently missing schema -- serial or concurrent alike, since
    nothing about the ordering makes a dropped error visible.
    """
    async def one(key: str):
        try:
            return key, derive_table(key, await _peek(s3, sem, bucket, key)), None
        except FetchFailed as exc:
            return key, None, str(exc)

    results = await asyncio.gather(*(one(k) for k in sorted(keys)))
    return ([r for _, r, _ in results if r],
            [e for _, _, e in results if e])


async def _run(args, by_dataset: dict[str, list[str]], done: set[str]) -> tuple[int, int]:
    session = aioboto3.Session()
    sem = asyncio.Semaphore(args.concurrency)
    slugs = [s for s in sorted(by_dataset) if s not in done]
    written = tables = incomplete = 0

    # Append, never truncate: the checkpoint file is what makes a re-run resume
    # rather than repeat, and a 30-minute job that dies at minute 28 with no
    # resume is a job you run twice.
    with open(args.output, "a", encoding="utf-8") as out, \
         open(args.checkpoint, "a", encoding="utf-8") as ck:
        async with session.client("s3", region_name="us-east-1",
                                  **_async_client_kwargs(args)) as s3:
            for i in range(0, len(slugs), args.batch):
                chunk = slugs[i : i + args.batch]
                found = await asyncio.gather(*(
                    _derive_dataset(s3, sem, args.bucket, slug, by_dataset[slug])
                    for slug in chunk
                ))
                for slug, (recs, failures) in zip(chunk, found):
                    if recs:
                        out.write(json.dumps({"dataset_slug": slug, "document_id": slug,
                                              "tables": recs}) + "\n")
                        written += 1
                        tables += len(recs)
                    if failures:
                        # Not checkpointed: a re-run must revisit this dataset
                        # rather than inherit a gap that looks like a decision.
                        incomplete += len(failures)
                        for f in failures:
                            print(f"  UNREAD {f}", file=sys.stderr)
                        continue
                    ck.write(slug + "\n")
                out.flush(); ck.flush()
                done_n = i + len(chunk)
                print(f"  {done_n}/{len(slugs)} datasets, {written} with schemas, "
                      f"{tables} tables"
                      + (f", {incomplete} files unread" if incomplete else ""),
                      file=sys.stderr)
    return written, tables, incomplete


def _async_client_kwargs(args) -> dict:
    if args.signed:
        return {}
    return {"config": BotoConfig(signature_version=UNSIGNED)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bucket", default=DEFAULT_BUCKET)
    ap.add_argument("--prefix", default="datagov/")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--limit-datasets", type=int, default=0,
                    help="stop after N datasets (for a quick check)")
    ap.add_argument("--signed", action="store_true",
                    help="sign requests; the source bucket is public and does not need it")
    ap.add_argument("--concurrency", type=int, default=1,
                    help="in-flight range GETs. Default 1, i.e. strictly "
                         "sequential; raise it to trade determinism for speed.")
    ap.add_argument("--batch", type=int, default=200,
                    help="datasets per checkpoint flush (default 200)")
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="slugs already processed; defaults to <output>.done")
    args = ap.parse_args(argv)

    args.checkpoint = args.checkpoint or Path(str(args.output) + ".done")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    done: set[str] = set()
    if args.checkpoint.is_file():
        done = {l.strip() for l in args.checkpoint.read_text().splitlines() if l.strip()}
        print(f"Resuming: {len(done)} datasets already processed", file=sys.stderr)

    s3 = _client(unsigned=not args.signed)
    by_dataset: dict[str, list[str]] = defaultdict(list)

    print(f"Listing s3://{args.bucket}/{args.prefix} ...", file=sys.stderr)
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=args.bucket, Prefix=args.prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            # Metadata siblings (catalog, dcat-us, headers, licence text, Socrata
            # ids, bare numbers) sit next to the payload in every dataset and are
            # 58% of them, so skipping here rather than after the GET is most of
            # the saving.
            if should_skip(key.rsplit("/", 1)[-1]):
                continue
            slug = key[len(args.prefix):].split("/", 1)[0]
            if slug:
                by_dataset[slug].append(key)
        if args.limit_datasets and len(by_dataset) >= args.limit_datasets:
            break

    if args.limit_datasets:
        # A fixed window, not one that grows by however many are already done.
        # Widening it per run made a re-run process a fresh slice and append the
        # same slugs to the checkpoint again, so "resume" quietly became "carry
        # on somewhere else".
        keep = sorted(by_dataset)[: args.limit_datasets]
        by_dataset = {k: by_dataset[k] for k in keep}

    todo = len([k for k in by_dataset if k not in done])
    print(f"Datasets listed: {len(by_dataset)}  to process: {todo}", file=sys.stderr)
    if not todo:
        print("nothing to do")
        return 0

    written, tables, incomplete = asyncio.run(_run(args, by_dataset, done))
    print(f"Wrote {written} datasets / {tables} tables to {args.output}")
    if incomplete:
        print(f"{incomplete} files could not be read; their datasets are NOT "
              f"checkpointed -- re-run to retry them.", file=sys.stderr)
    if written == 0:
        print("ERROR: no schemas derived; check the bucket and prefix.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
