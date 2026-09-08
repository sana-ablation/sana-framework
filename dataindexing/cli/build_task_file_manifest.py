#!/usr/bin/env python3
"""Expand task dataset ids into a concrete file manifest without downloading data."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv
try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable=None, **_kwargs):
        return iterable if iterable is not None else _NullTqdm()

from sana_evaluation.tools.agent_tools import BUCKET, FOLDERS, _build_s3_client

load_dotenv()


class _NullTqdm:
    def update(self, _n: int = 1) -> None:
        return None

    def set_postfix(self, *args, **kwargs) -> None:
        return None

    def close(self) -> None:
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-root", default="benchmarks/lakeqa/tasks-mini/tasks")
    parser.add_argument("--output", default="benchmarks/lakeqa/tasks-mini/artifacts/task_file_manifest.jsonl")
    parser.add_argument("--bucket", default=BUCKET)
    return parser.parse_args(argv)


def _iter_task_files(task_root: Path) -> Iterable[Path]:
    yield from sorted(task_root.rglob("*.json"))


def _collect_dataset_usage(task_root: Path) -> Dict[str, set[str]]:
    usage: Dict[str, set[str]] = {}
    for task_path in _iter_task_files(task_root):
        with task_path.open() as f:
            task = json.load(f)
        task_ref = str(task_path.relative_to(task_root.parent))
        for dataset_id in task.get("datasets_used", []):
            if not dataset_id:
                continue
            usage.setdefault(str(dataset_id), set()).add(task_ref)
    return usage


def _build_s3_client_for_runtime():
    requested = (os.getenv("S3_ACCESS_MODE", "auto") or "auto").strip().lower()
    if requested in {"unsigned", "public", "anonymous", "anon", "no-sign-request"}:
        return _build_s3_client(unsigned=True)
    try:
        return _build_s3_client(unsigned=False)
    except Exception:
        return _build_s3_client(unsigned=True)


def _dataset_exists(s3_client, *, bucket: str, folder: str, dataset_id: str) -> bool:
    response = s3_client.list_objects_v2(
        Bucket=bucket,
        Prefix=f"{folder}/{dataset_id}/",
        MaxKeys=1,
    )
    return "Contents" in response or "CommonPrefixes" in response


def _resolve_dataset_folder(s3_client, *, bucket: str, dataset_id: str) -> Optional[str]:
    matches = [folder for folder in FOLDERS if _dataset_exists(s3_client, bucket=bucket, folder=folder, dataset_id=dataset_id)]
    if len(matches) == 1:
        return matches[0]
    return None


def _list_dataset_files(s3_client, *, bucket: str, folder: str, dataset_id: str) -> List[Dict[str, Any]]:
    prefix = f"{folder}/{dataset_id}/"
    continuation: Optional[str] = None
    files: List[Dict[str, Any]] = []

    while True:
        kwargs: Dict[str, Any] = {"Bucket": bucket, "Prefix": prefix, "MaxKeys": 1000}
        if continuation:
            kwargs["ContinuationToken"] = continuation
        response = s3_client.list_objects_v2(**kwargs)

        for obj in response.get("Contents", []):
            key = str(obj.get("Key") or "")
            if not key or key.endswith("/"):
                continue
            relative_path = key.split(prefix, 1)[-1]
            files.append(
                {
                    "dataset_id": dataset_id,
                    "folder": folder,
                    "file_path": relative_path,
                    "size_bytes": int(obj.get("Size") or 0),
                    "s3_uri": f"s3://{bucket}/{key}",
                }
            )

        if not response.get("IsTruncated"):
            break
        continuation = response.get("NextContinuationToken")
        if not continuation:
            break

    files.sort(key=lambda row: row["file_path"])
    return files


def build_manifest(
    *,
    task_root: Path,
    output_path: Path,
    bucket: str = BUCKET,
    s3_client=None,
) -> Dict[str, Any]:
    usage = _collect_dataset_usage(task_root)
    s3_client = s3_client or _build_s3_client_for_runtime()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    file_count = 0
    unresolved: List[str] = []
    with output_path.open("w") as out:
        for dataset_id in tqdm(
            sorted(usage),
            total=len(usage),
            desc="Listing datasets",
            unit="dataset",
        ):
            folder = _resolve_dataset_folder(s3_client, bucket=bucket, dataset_id=dataset_id)
            if folder is None:
                unresolved.append(dataset_id)
                continue

            files = _list_dataset_files(s3_client, bucket=bucket, folder=folder, dataset_id=dataset_id)
            task_refs = sorted(usage[dataset_id])
            for row in files:
                row["tasks"] = task_refs
                row["task_count"] = len(task_refs)
                out.write(json.dumps(row))
                out.write("\n")
                file_count += 1

    return {
        "datasets": len(usage),
        "files": file_count,
        "unresolved_dataset_ids": unresolved,
        "unresolved_count": len(unresolved),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    summary = build_manifest(
        task_root=Path(args.task_root),
        output_path=Path(args.output),
        bucket=args.bucket,
    )
    print(
        f"Wrote {summary['files']} file rows from {summary['datasets']} datasets to {args.output} "
        f"(unresolved={summary['unresolved_count']})"
    )
    if summary["unresolved_dataset_ids"]:
        for dataset_id in summary["unresolved_dataset_ids"]:
            print(f"UNRESOLVED {dataset_id}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
