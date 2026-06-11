"""Parquet cache and description artifact readers for indexing pipelines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


def iter_parquet_records(parquet_path: str):
    """Yield `(dataset_uri, metadata, content)` rows without loading the full parquet."""
    pf = pq.ParquetFile(parquet_path)
    total = 0
    for i in range(pf.metadata.num_row_groups):
        batch = pf.read_row_group(i, columns=["dataset_uri", "metadata", "content"])
        for uri, meta, content in zip(
            batch.column("dataset_uri").to_pylist(),
            batch.column("metadata").to_pylist(),
            batch.column("content").to_pylist(),
        ):
            total += 1
            yield uri, meta, content
    print(f"Streamed {total} documents from parquet cache: {parquet_path}")


def load_parquet_records(parquet_path: str) -> list[tuple[str, str, str]]:
    """Load all `(dataset_uri, metadata, content)` rows from a parquet cache."""
    records: list[tuple[str, str, str]] = []
    pf = pq.ParquetFile(parquet_path)

    for i in range(pf.metadata.num_row_groups):
        batch = pf.read_row_group(i, columns=["dataset_uri", "metadata", "content"])
        for uri, meta, content in zip(
            batch.column("dataset_uri").to_pylist(),
            batch.column("metadata").to_pylist(),
            batch.column("content").to_pylist(),
        ):
            records.append((uri, meta, content))

    print(f"Loaded {len(records)} documents from parquet cache: {parquet_path}")
    return records


def load_descriptions(descriptions_path: str) -> dict[str, str]:
    """Load generated metadata + description text keyed by dataset URI."""
    if not Path(descriptions_path).exists():
        return {}

    out: dict[str, str] = {}
    pf = pq.ParquetFile(descriptions_path)
    available = set(pf.schema_arrow.names)
    needed = ["dataset_uri", "generated_metadata", "description", "error"]
    missing = [c for c in needed if c not in available]
    if missing:
        print(
            f"WARN: descriptions parquet missing columns {missing}; "
            f"falling back to no descriptions"
        )
        return {}

    for i in range(pf.metadata.num_row_groups):
        batch = pf.read_row_group(i, columns=needed)
        for uri, gen, desc, err in zip(
            batch.column("dataset_uri").to_pylist(),
            batch.column("generated_metadata").to_pylist(),
            batch.column("description").to_pylist(),
            batch.column("error").to_pylist(),
        ):
            if err is not None:
                continue
            parts = [p for p in (gen, desc) if p]
            if parts:
                out[str(uri)] = " ".join(parts)

    print(f"Loaded {len(out)} descriptions from: {descriptions_path}")
    return out


def load_descriptions_full(descriptions_path: str) -> dict[str, dict[str, str]]:
    """Load structured description rows keyed by dataset URI from parquet or JSONL."""
    path = Path(descriptions_path)
    if not path.exists():
        return {}

    if path.suffix.lower() == ".jsonl":
        return _load_descriptions_full_jsonl(path)

    out: dict[str, dict[str, str]] = {}
    pf = pq.ParquetFile(path)
    available = set(pf.schema_arrow.names)
    needed = [
        "dataset_uri",
        "generated_metadata",
        "description",
        "original_metadata",
        "error",
    ]
    missing = [c for c in needed if c not in available]
    if missing:
        print(
            f"WARN: descriptions parquet missing columns {missing}; "
            f"falling back to no descriptions"
        )
        return {}

    for i in range(pf.metadata.num_row_groups):
        batch = pf.read_row_group(i, columns=needed)
        for uri, gen, desc, orig, err in zip(
            batch.column("dataset_uri").to_pylist(),
            batch.column("generated_metadata").to_pylist(),
            batch.column("description").to_pylist(),
            batch.column("original_metadata").to_pylist(),
            batch.column("error").to_pylist(),
        ):
            if err is not None:
                continue
            if not (gen or desc or orig):
                continue
            out[str(uri)] = {
                "generated_metadata": str(gen or ""),
                "description": str(desc or ""),
                "original_metadata": str(orig or ""),
            }

    print(f"Loaded {len(out)} description records from: {descriptions_path}")
    return out


def _load_descriptions_full_jsonl(path: Path) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                row: dict[str, Any] = json.loads(raw)
            except json.JSONDecodeError:
                print(f"WARN: skipping invalid descriptions JSONL line {line_no}: {path}")
                continue

            uri = row.get("dataset_uri")
            if not uri or row.get("error") not in (None, ""):
                continue

            gen = row.get("generated_metadata") or ""
            desc = row.get("description") or ""
            orig = row.get("original_metadata") or ""
            if not (gen or desc or orig):
                continue

            out[str(uri)] = {
                "generated_metadata": str(gen),
                "description": str(desc),
                "original_metadata": str(orig),
            }

    print(f"Loaded {len(out)} description records from: {path}")
    return out

