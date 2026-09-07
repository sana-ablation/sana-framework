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

import boto3
from botocore import UNSIGNED
from botocore.config import Config as BotoConfig

from sana_evaluation.tools.helper.detect import detect_family, should_skip

DEFAULT_BUCKET = "lakeqa-yc4103-datalake"
PEEK_BYTES = 256 * 1024

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
    if len(columns) < _MIN_CSV_COLUMNS:
        return False
    # One clearly prosaic field condemns the whole line: a title split on its
    # commas yields a few short fragments alongside one long one, which a
    # per-field majority vote would otherwise pass.
    for name in columns:
        if len(str(name).split()) > 5:
            return False
    ok = sum(1 for c in columns if _field_is_identifier_like(c))
    return ok >= max(_MIN_CSV_COLUMNS, int(0.8 * len(columns)))


def derive_table(key: str, head: str) -> dict | None:
    """Return a schema record for one stored object, or None if it has no table."""
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
            # Truncated at PEEK_BYTES, or JSON Lines. Try the first line alone.
            try:
                doc = json.loads(head.lstrip().split("\n", 1)[0])
            except json.JSONDecodeError:
                doc = None
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
    if not columns:
        return None
    return {
        "relative_path": key.split("/files/", 1)[-1] if "/files/" in key else key.rsplit("/", 1)[-1],
        "s3_key": key,
        "table_kind": kind,
        "columns": columns,
        "delimiter": delimiter,
    }


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
    args = ap.parse_args(argv)

    s3 = _client(unsigned=not args.signed)
    by_dataset: dict[str, list[str]] = defaultdict(list)

    print(f"Listing s3://{args.bucket}/{args.prefix} ...", file=sys.stderr)
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=args.bucket, Prefix=args.prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            name = key.rsplit("/", 1)[-1]
            # Metadata siblings (catalog, dcat-us, headers, licence text, Socrata
            # ids, bare numbers) sit next to the payload in every dataset.
            if should_skip(name):
                continue
            slug = key[len(args.prefix):].split("/", 1)[0]
            if not slug:
                continue
            by_dataset[slug].append(key)
        if args.limit_datasets and len(by_dataset) > args.limit_datasets:
            break

    slugs = sorted(by_dataset)[: args.limit_datasets or None]
    print(f"Datasets: {len(slugs)}", file=sys.stderr)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = tables_written = 0
    with open(args.output, "w", encoding="utf-8") as out:
        for i, slug in enumerate(slugs, 1):
            tables = []
            for key in sorted(by_dataset[slug]):
                try:
                    body = s3.get_object(Bucket=args.bucket, Key=key,
                                         Range=f"bytes=0-{PEEK_BYTES - 1}")["Body"].read()
                except Exception as exc:
                    print(f"  skip {key}: {exc}", file=sys.stderr)
                    continue
                head = body.decode("utf-8", errors="replace")
                rec = derive_table(key, head)
                if rec:
                    tables.append(rec)
            if not tables:
                continue
            out.write(json.dumps({"dataset_slug": slug, "document_id": slug,
                                  "tables": tables}) + "\n")
            written += 1
            tables_written += len(tables)
            if i % 50 == 0:
                print(f"  {i}/{len(slugs)} datasets ...", file=sys.stderr)

    print(f"Wrote {written} datasets / {tables_written} tables to {args.output}")
    if written == 0:
        print("ERROR: no schemas derived; check the bucket and prefix.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
