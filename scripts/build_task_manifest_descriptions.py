#!/usr/bin/env python3
"""Audit and merge task-scoped dataset description artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sana_evaluation.tools.external.description_rows import (  # noqa: E402
    DESCRIPTION_FIELDS,
    description_uri,
    has_valid_description,
    reject_forbidden_description_row,
)

DEFAULT_SEED_DESCRIPTION_PATHS = (
    Path("benchmarks/lakeqa/tasks-mini/artifacts/descriptions.jsonl"),
)
DEFAULT_GENERATED_DESCRIPTION_PATHS = (
    Path("benchmarks/lakeqa/tasks-mini/artifacts/missing_descriptions_generated.jsonl"),
)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    artifact_root = Path("benchmarks/lakeqa/tasks-mini/artifacts")
    parser.add_argument("--manifest", default=str(artifact_root / "task_file_manifest.jsonl"))
    parser.add_argument(
        "--seed-description",
        action="append",
        default=None,
        help=(
            "Existing trusted description JSONL. Repeat to layer sources. "
            "Defaults to the benchmark artifact descriptions JSONL."
        ),
    )
    parser.add_argument(
        "--generated-description",
        action="append",
        default=None,
        help=(
            "LLM-generated JSONL for rows missing from seed descriptions. "
            "Repeat to layer sources."
        ),
    )
    parser.add_argument(
        "--missing-output",
        default=str(artifact_root / "missing_descriptions.jsonl"),
        help="Manifest rows still missing seed descriptions.",
    )
    parser.add_argument(
        "--unresolved-output",
        default=str(artifact_root / "unresolved_descriptions.jsonl"),
        help="Manifest rows still unresolved during merge.",
    )
    parser.add_argument(
        "--output",
        default=str(artifact_root / "task_file_manifest_descriptions.jsonl"),
        help="Canonical merged description JSONL.",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Only write the missing-description audit; do not merge output.",
    )
    return parser.parse_args(argv)


def _jsonl_rows(path: Path) -> Iterable[dict]:
    if not path.exists():
        return
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


def _load_manifest(path: Path) -> list[dict]:
    rows: list[dict] = []
    for row in _jsonl_rows(path):
        uri = str(row.get("s3_uri") or "").strip()
        if uri:
            rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")
            count += 1
    return count


def _description_paths(paths: Optional[Sequence[Path]], defaults: Sequence[Path]) -> list[Path]:
    selected = list(paths) if paths is not None else list(defaults)
    out: list[Path] = []
    for path in selected:
        if path not in out:
            out.append(path)
    return out


def _load_description_map(paths: Sequence[Path]) -> tuple[dict[str, dict], dict[str, int]]:
    descriptions: dict[str, dict] = {}
    valid_counts: dict[str, int] = {}
    for path in paths:
        count = 0
        for row in _jsonl_rows(path):
            reject_forbidden_description_row(row, path=path)
            if not has_valid_description(row):
                continue
            count += 1
            uri = description_uri(row)
            if uri not in descriptions:
                descriptions[uri] = row
        valid_counts[str(path)] = count
    return descriptions, valid_counts


def _normal_description_row(row: Dict[str, Any], *, uri: str) -> dict:
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
        "content": content,
        "cost_usd": row.get("cost_usd"),
        "dataset_uri": uri,
        "description": description,
        "error": row.get("error"),
        "generated_metadata": generated_metadata,
        "input_cost_usd": row.get("input_cost_usd"),
        "input_tokens": row.get("input_tokens"),
        "metadata": metadata,
        "original_metadata": row.get("original_metadata"),
        "output_cost_usd": row.get("output_cost_usd"),
        "output_tokens": row.get("output_tokens"),
    }
    return {field: normalized.get(field) for field in DESCRIPTION_FIELDS}


def audit_missing_descriptions(
    *,
    manifest_path: Path,
    seed_description_paths: Sequence[Path],
    missing_output_path: Path,
) -> dict:
    manifest_rows = _load_manifest(manifest_path)
    seed_descriptions, valid_counts = _load_description_map(seed_description_paths)
    missing_rows = [
        row for row in manifest_rows if str(row.get("s3_uri") or "").strip() not in seed_descriptions
    ]
    _write_jsonl(missing_output_path, missing_rows)
    return {
        "manifest_rows": len(manifest_rows),
        "seed_description_rows": sum(valid_counts.values()),
        "covered_by_seed_descriptions": len(manifest_rows) - len(missing_rows),
        "missing_descriptions": len(missing_rows),
        "missing_output": str(missing_output_path),
        "seed_description_paths": [str(path) for path in seed_description_paths],
    }


def merge_manifest_descriptions(
    *,
    manifest_path: Path,
    seed_description_paths: Sequence[Path],
    generated_description_paths: Sequence[Path],
    output_path: Path,
    unresolved_output_path: Optional[Path] = None,
) -> dict:
    manifest_rows = _load_manifest(manifest_path)
    seed_descriptions, seed_counts = _load_description_map(seed_description_paths)
    generated_descriptions, generated_counts = _load_description_map(generated_description_paths)

    output_rows: list[dict] = []
    unresolved_rows: list[dict] = []
    seed_used = 0
    generated_used = 0
    for manifest_row in manifest_rows:
        uri = str(manifest_row.get("s3_uri") or "").strip()
        source_row = seed_descriptions.get(uri)
        if source_row is not None:
            seed_used += 1
        else:
            source_row = generated_descriptions.get(uri)
            if source_row is not None:
                generated_used += 1
        if source_row is None:
            unresolved_rows.append(manifest_row)
            continue
        output_rows.append(_normal_description_row(source_row, uri=uri))

    if unresolved_rows:
        if unresolved_output_path is not None:
            _write_jsonl(unresolved_output_path, unresolved_rows)
        raise ValueError(
            f"Missing valid generated descriptions for {len(unresolved_rows)} "
            f"of {len(manifest_rows)} manifest row(s)."
        )

    _write_jsonl(output_path, output_rows)
    return {
        "manifest_rows": len(manifest_rows),
        "written": len(output_rows),
        "seed_used": seed_used,
        "generated_used": generated_used,
        "seed_description_rows": sum(seed_counts.values()),
        "generated_description_rows": sum(generated_counts.values()),
        "output": str(output_path),
        "seed_description_paths": [str(path) for path in seed_description_paths],
        "generated_description_paths": [str(path) for path in generated_description_paths],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    seed_paths = _description_paths(
        [Path(path) for path in args.seed_description] if args.seed_description else None,
        DEFAULT_SEED_DESCRIPTION_PATHS,
    )
    if args.audit_only:
        summary = audit_missing_descriptions(
            manifest_path=Path(args.manifest),
            seed_description_paths=seed_paths,
            missing_output_path=Path(args.missing_output),
        )
    else:
        generated_paths = _description_paths(
            [Path(path) for path in args.generated_description]
            if args.generated_description
            else None,
            DEFAULT_GENERATED_DESCRIPTION_PATHS,
        )
        summary = merge_manifest_descriptions(
            manifest_path=Path(args.manifest),
            seed_description_paths=seed_paths,
            generated_description_paths=generated_paths,
            output_path=Path(args.output),
            unresolved_output_path=Path(args.unresolved_output),
        )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
