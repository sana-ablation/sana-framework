"""Manifest readers for S3 URI lists and JSONL records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator


def iter_jsonl_records(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield JSON objects from a JSONL file."""
    with Path(path).open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Manifest line {line_no} is not valid JSON") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Manifest line {line_no} must be a JSON object")
            yield obj


def iter_manifest_uris(path: str | Path) -> Iterator[str]:
    """Yield S3 URIs from plain text or JSONL manifests."""
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("{"):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            uri = str(row.get("s3_uri") or row.get("dataset_uri") or row.get("uri") or "").strip()
            if uri:
                yield uri
            continue
        yield line

