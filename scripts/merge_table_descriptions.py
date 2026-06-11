#!/usr/bin/env python3
"""Merge table-description JSONL artifacts into one canonical file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sana_evaluation.tools.external.description_rows import (  # noqa: E402
    DESCRIPTION_FIELDS,
    description_uri,
    has_valid_description,
    reject_forbidden_description_row,
)


DEFAULT_DESCRIPTION_PATHS = (
    Path("benchmarks/lakeqa/tasks-mini/artifacts/descriptions.jsonl"),
    Path("benchmarks/lakeqa/tasks-mini/artifacts/task_file_manifest_descriptions.jsonl"),
)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--description",
        action="append",
        default=None,
        help=(
            "Description JSONL to merge. Repeat in precedence order; later files "
            "override earlier rows for the same dataset_uri. Defaults to "
            "the benchmark artifact description files."
        ),
    )
    parser.add_argument(
        "--output",
        default="benchmarks/lakeqa/tasks-mini/artifacts/descriptions.jsonl",
        help="Canonical merged description JSONL to write.",
    )
    parser.add_argument(
        "--uri-output",
        default="benchmarks/lakeqa/tasks-mini/artifacts/table_profiles_needed.txt",
        help="Plain-text URI list derived from the merged descriptions.",
    )
    return parser.parse_args(argv)


def _jsonl_rows(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row


def _normal_description_row(row: dict[str, Any], *, uri: str) -> dict[str, Any]:
    reject_forbidden_description_row(row)
    generated_metadata = str(row.get("generated_metadata") or row.get("metadata") or "").strip()
    description = str(row.get("description") or "").strip()
    content = str(row.get("content") or "").strip()
    if not content:
        content = " ".join(part for part in (generated_metadata, description) if part).strip()
    metadata = row.get("metadata")
    if metadata is None:
        metadata = generated_metadata
    normalized = {
        "dataset_uri": uri,
        "metadata": metadata,
        "content": content,
        "original_metadata": row.get("original_metadata"),
        "generated_metadata": generated_metadata,
        "description": description,
        "input_tokens": row.get("input_tokens"),
        "output_tokens": row.get("output_tokens"),
        "input_cost_usd": row.get("input_cost_usd"),
        "output_cost_usd": row.get("output_cost_usd"),
        "cost_usd": row.get("cost_usd"),
        "error": row.get("error"),
    }
    return {field: normalized.get(field) for field in DESCRIPTION_FIELDS}


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")
            count += 1
    return count


def _write_uri_list(path: Path, uris: Iterable[str]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for uri in uris:
            f.write(uri)
            f.write("\n")
            count += 1
    return count


def merge_table_descriptions(
    *,
    description_paths: Sequence[Path],
    output_path: Path,
    uri_output_path: Path,
) -> dict[str, Any]:
    rows_by_uri: dict[str, dict[str, Any]] = {}
    ordered_uris: list[str] = []
    per_source: list[dict[str, Any]] = []
    overridden_rows = 0

    for path in description_paths:
        source_rows = 0
        valid_rows = 0
        new_rows = 0
        source_overrides = 0
        for row in _jsonl_rows(path):
            source_rows += 1
            reject_forbidden_description_row(row, path=path)
            if not has_valid_description(row):
                continue
            uri = description_uri(row)
            if uri in rows_by_uri:
                overridden_rows += 1
                source_overrides += 1
            else:
                ordered_uris.append(uri)
                new_rows += 1
            rows_by_uri[uri] = _normal_description_row(row, uri=uri)
            valid_rows += 1
        per_source.append(
            {
                "path": str(path),
                "rows": source_rows,
                "valid_rows": valid_rows,
                "new_rows": new_rows,
                "overridden_rows": source_overrides,
            }
        )

    output_rows = [rows_by_uri[uri] for uri in ordered_uris]
    written = _write_jsonl(output_path, output_rows)
    uri_written = _write_uri_list(uri_output_path, ordered_uris)

    return {
        "description_paths": [str(path) for path in description_paths],
        "output": str(output_path),
        "uri_output": str(uri_output_path),
        "written": written,
        "uri_written": uri_written,
        "overridden_rows": overridden_rows,
        "sources": per_source,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    description_paths = (
        [Path(path) for path in args.description]
        if args.description
        else list(DEFAULT_DESCRIPTION_PATHS)
    )
    summary = merge_table_descriptions(
        description_paths=description_paths,
        output_path=Path(args.output),
        uri_output_path=Path(args.uri_output),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
