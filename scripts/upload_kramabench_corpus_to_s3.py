#!/usr/bin/env python3
"""Stage and upload Kramabench raw corpus files into a LakeQA-shaped S3 layout."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse


REPO = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATES_DIR = REPO / "benchmarks/kramabench/tasks-mini/tasks/candidates"
DEFAULT_STAGING_DIR = REPO / "other-benchmarks" / "data-imports" / "kramabench" / "_s3_corpus_staging"


@dataclass(frozen=True)
class UploadItem:
    local_path: Path
    bucket: str
    key: str
    source_path: Path

    @property
    def s3_uri(self) -> str:
        return f"s3://{self.bucket}/{self.key}"


def parse_bucket_uri(bucket_uri: str) -> tuple[str, str]:
    parsed = urlparse(bucket_uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"expected an S3 URI like s3://bucket[/prefix], got {bucket_uri!r}")
    return parsed.netloc, parsed.path.strip("/")


def _join_key(prefix: str, key: str) -> str:
    return f"{prefix}/{key}" if prefix else key


def _candidate_files(candidates_dir: Path, source_ids: tuple[str, ...] | list[str] | None) -> list[Path]:
    candidates = sorted(Path(candidates_dir).glob("*.json"))
    if not source_ids:
        return candidates
    wanted = set(source_ids)
    selected = [path for path in candidates if path.stem in wanted]
    missing = sorted(wanted - {path.stem for path in selected})
    if missing:
        raise FileNotFoundError(f"missing candidate file(s): {', '.join(missing)}")
    return selected


def _repo_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else REPO / path


def materialize_corpus_uploads(
    bucket_uri: str,
    candidates_dir: Path = DEFAULT_CANDIDATES_DIR,
    staging_dir: Path = DEFAULT_STAGING_DIR,
    source_ids: tuple[str, ...] | list[str] | None = None,
) -> list[UploadItem]:
    """Copy candidate raw files into staging and return the S3 upload manifest."""
    bucket, bucket_prefix = parse_bucket_uri(bucket_uri)
    candidates_dir = Path(candidates_dir)
    if not candidates_dir.is_absolute():
        candidates_dir = REPO / candidates_dir
    staging_dir = Path(staging_dir)

    items: list[UploadItem] = []
    seen_keys: set[str] = set()
    for candidate_path in _candidate_files(candidates_dir, source_ids):
        data = json.loads(candidate_path.read_text(encoding="utf-8"))
        source_id = data.get("source_id") or candidate_path.stem
        s3_mirror_prefix = str(data.get("s3_mirror_prefix") or "").strip("/")
        expected_prefix = f"datagov/kramabench-{source_id}/files"
        if s3_mirror_prefix != expected_prefix:
            raise ValueError(
                f"{candidate_path} has s3_mirror_prefix={s3_mirror_prefix!r}; "
                f"expected {expected_prefix!r}"
            )

        raw_root = _repo_path(str(data.get("raw_data_root") or ""))
        for rel in data.get("raw_data_files") or []:
            source_path = raw_root / rel
            if not source_path.is_file():
                raise FileNotFoundError(f"missing raw input for {source_id}: {source_path}")

            relative_key = f"{s3_mirror_prefix}/{rel}"
            key = _join_key(bucket_prefix, relative_key)
            if key in seen_keys:
                continue
            seen_keys.add(key)

            local_path = staging_dir / relative_key
            local_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, local_path)
            items.append(
                UploadItem(
                    local_path=local_path,
                    bucket=bucket,
                    key=key,
                    source_path=source_path,
                )
            )
    return items


def _upload_with_boto3(items: list[UploadItem]) -> list[UploadItem]:
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError("boto3 is required for --uploader boto3") from exc

    s3_client = boto3.client("s3")
    for item in items:
        s3_client.upload_file(str(item.local_path), item.bucket, item.key)
    return items


def _upload_with_aws_cli(items: list[UploadItem], run_command=None) -> list[UploadItem]:
    if run_command is None:
        run_command = lambda command: subprocess.run(command, check=True)

    for item in items:
        run_command(["aws", "s3", "cp", str(item.local_path), item.s3_uri])
    return items


def upload_materialized_files(
    items: list[UploadItem],
    s3_client=None,
    uploader: str = "auto",
    run_command=None,
) -> list[UploadItem]:
    if s3_client is not None:
        for item in items:
            s3_client.upload_file(str(item.local_path), item.bucket, item.key)
        return items

    if uploader == "aws-cli":
        return _upload_with_aws_cli(items, run_command=run_command)
    if uploader == "boto3":
        return _upload_with_boto3(items)
    if uploader != "auto":
        raise ValueError(f"unknown uploader {uploader!r}; expected auto, boto3, or aws-cli")

    try:
        return _upload_with_boto3(items)
    except RuntimeError:
        return _upload_with_aws_cli(items, run_command=run_command)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bucket-uri",
        default="s3://sana-kramabench",
        help="Target S3 bucket URI, optionally with prefix.",
    )
    parser.add_argument(
        "--candidates-dir",
        type=Path,
        default=DEFAULT_CANDIDATES_DIR,
        help="Directory containing benchmarks/kramabench/tasks-mini/tasks/candidates/*.json.",
    )
    parser.add_argument(
        "--source-id",
        action="append",
        dest="source_ids",
        help="Limit to one Kramabench source id. Can be repeated. Defaults to all candidates.",
    )
    parser.add_argument(
        "--staging-dir",
        type=Path,
        default=DEFAULT_STAGING_DIR,
        help="Directory where upload-ready files are staged before S3 upload.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Materialize files and print the upload plan without calling S3.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print every staged object. By default, only dry-runs print the full manifest.",
    )
    parser.add_argument(
        "--uploader",
        choices=("auto", "boto3", "aws-cli"),
        default="auto",
        help="Upload backend. auto uses boto3 when available, otherwise AWS CLI.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    items = materialize_corpus_uploads(
        bucket_uri=args.bucket_uri,
        candidates_dir=args.candidates_dir,
        staging_dir=args.staging_dir,
        source_ids=tuple(args.source_ids or ()),
    )
    total_bytes = sum(item.local_path.stat().st_size for item in items)
    print(f"Prepared {len(items)} object(s), {total_bytes} byte(s).")
    if args.dry_run or args.verbose:
        for item in items:
            print(f"{item.local_path} -> {item.s3_uri}")

    if args.dry_run:
        return 0

    upload_materialized_files(items, uploader=args.uploader)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
