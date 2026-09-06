#!/usr/bin/env python3
"""Audit description/snippet/profile coverage for a concrete file manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sana_evaluation.tools.external.description_rows import (  # noqa: E402
    description_uri,
    has_valid_description,
    reject_forbidden_description_row,
)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    artifact_root = Path("benchmarks/lakeqa/tasks-mini/artifacts")
    parser.add_argument("--manifest", default=str(artifact_root / "task_file_manifest.jsonl"))
    parser.add_argument(
        "--descriptions",
        action="append",
        default=None,
        help=(
            "Description JSONL to check. Repeat to union multiple files. "
            "Defaults to canonical descriptions.jsonl."
        ),
    )
    parser.add_argument("--snippets", default=str(artifact_root / "snippets.jsonl"))
    parser.add_argument("--profiles", default=str(artifact_root / "table_profiles.jsonl"))
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Optional prefix for writing missing_rows JSONL files.",
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


def _load_manifest(path: Path) -> Dict[str, dict]:
    rows: Dict[str, dict] = {}
    for row in _jsonl_rows(path):
        uri = str(row.get("s3_uri") or "").strip()
        if uri and uri not in rows:
            rows[uri] = row
    return rows


def _load_uri_set(path: Path, *, field_names: Sequence[str]) -> set[str]:
    values: set[str] = set()
    for row in _jsonl_rows(path):
        for field_name in field_names:
            value = str(row.get(field_name) or "").strip()
            if value:
                values.add(value)
                break
    return values


def _as_paths(paths: Path | Sequence[Path]) -> list[Path]:
    if isinstance(paths, Path):
        return [paths]
    return list(paths)


def _load_description_uri_set(paths: Path | Sequence[Path]) -> set[str]:
    values: set[str] = set()
    for path in _as_paths(paths):
        for row in _jsonl_rows(path):
            reject_forbidden_description_row(row, path=path)
            if has_valid_description(row):
                values.add(description_uri(row))
    return values


def _write_rows(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row))
            f.write("\n")


def default_description_paths(args: argparse.Namespace) -> list[Path]:
    if args.descriptions:
        return [Path(path) for path in args.descriptions]
    return [Path("benchmarks/lakeqa/tasks-mini/artifacts/descriptions.jsonl")]


def audit_manifest_coverage(
    *,
    manifest_path: Path,
    descriptions_path: Path | Sequence[Path],
    snippets_path: Path,
    profiles_path: Path,
    output_prefix: Optional[Path] = None,
) -> dict:
    manifest_rows = _load_manifest(manifest_path)
    desc_uris = _load_description_uri_set(descriptions_path)
    snippet_uris = _load_uri_set(snippets_path, field_names=("dataset_uri", "uri"))
    profile_uris = _load_uri_set(profiles_path, field_names=("s3_uri",))

    missing_desc = [row for uri, row in manifest_rows.items() if uri not in desc_uris]
    missing_snippet = [row for uri, row in manifest_rows.items() if uri not in snippet_uris]
    missing_profile = [row for uri, row in manifest_rows.items() if uri not in profile_uris]

    if output_prefix is not None:
        _write_rows(output_prefix.with_name(output_prefix.name + "_missing_descriptions.jsonl"), missing_desc)
        _write_rows(output_prefix.with_name(output_prefix.name + "_missing_snippets.jsonl"), missing_snippet)
        _write_rows(output_prefix.with_name(output_prefix.name + "_missing_profiles.jsonl"), missing_profile)

    return {
        "manifest_rows": len(manifest_rows),
        "descriptions_present": len(manifest_rows) - len(missing_desc),
        "snippets_present": len(manifest_rows) - len(missing_snippet),
        "profiles_present": len(manifest_rows) - len(missing_profile),
        "missing_descriptions": len(missing_desc),
        "missing_snippets": len(missing_snippet),
        "missing_profiles": len(missing_profile),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    output_prefix = Path(args.output_prefix) if args.output_prefix else None
    description_paths = default_description_paths(args)
    summary = audit_manifest_coverage(
        manifest_path=Path(args.manifest),
        descriptions_path=description_paths,
        snippets_path=Path(args.snippets),
        profiles_path=Path(args.profiles),
        output_prefix=output_prefix,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
