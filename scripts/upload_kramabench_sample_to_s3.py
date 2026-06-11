#!/usr/bin/env python3
"""Stage and upload Kramabench per-task dr-input bundles into an S3 LakeQA layout."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse


REPO = Path(__file__).resolve().parents[1]
DEFAULT_DR_INPUT_ROOT = REPO / "other-benchmarks" / "Kramabench" / "dr-input"
DEFAULT_STAGING_DIR = REPO / "other-benchmarks" / "data-imports" / "kramabench" / "_s3_staging"
DEFAULT_SOURCE_IDS = ("legal-hard-7",)


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
    prefix = parsed.path.strip("/")
    return parsed.netloc, prefix


def _join_key(prefix: str, key: str) -> str:
    return f"{prefix}/{key}" if prefix else key


def _domain_from_source_id(source_id: str) -> str:
    return source_id.split("-", 1)[0]


def _iter_bundle_files(bundle_dir: Path) -> list[Path]:
    return sorted(path for path in bundle_dir.rglob("*") if path.is_file())


def materialize_uploads(
    bucket_uri: str,
    staging_dir: Path = DEFAULT_STAGING_DIR,
    dr_input_root: Path = DEFAULT_DR_INPUT_ROOT,
    source_ids: tuple[str, ...] | list[str] = DEFAULT_SOURCE_IDS,
) -> list[UploadItem]:
    bucket, prefix = parse_bucket_uri(bucket_uri)
    staging_dir = Path(staging_dir)
    dr_input_root = Path(dr_input_root)
    if not dr_input_root.is_absolute():
        dr_input_root = REPO / dr_input_root

    items: list[UploadItem] = []
    for source_id in source_ids:
        bundle_dir = dr_input_root / _domain_from_source_id(source_id) / source_id
        if not bundle_dir.is_dir():
            raise FileNotFoundError(f"missing Kramabench dr-input bundle: {bundle_dir}")

        for source_path in _iter_bundle_files(bundle_dir):
            rel_file = source_path.relative_to(bundle_dir)
            relative_key = Path("datagov") / f"kramabench-{source_id}" / "files" / rel_file
            key = _join_key(prefix, relative_key.as_posix())
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
    if s3_client is None:
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

    for item in items:
        s3_client.upload_file(str(item.local_path), item.bucket, item.key)
    return items


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage and upload Kramabench per-task dr-input bundles.",
    )
    parser.add_argument(
        "--bucket-uri",
        default="s3://kramabench",
        help="Target S3 bucket URI, optionally with prefix, e.g. s3://kramabench/pilot",
    )
    parser.add_argument(
        "--dr-input-root",
        type=Path,
        default=DEFAULT_DR_INPUT_ROOT,
        help="Path to other-benchmarks/Kramabench/dr-input",
    )
    parser.add_argument(
        "--source-id",
        action="append",
        dest="source_ids",
        help="Kramabench task id to upload. Can be repeated. Defaults to the current dr-input promotion sample.",
    )
    parser.add_argument(
        "--staging-dir",
        type=Path,
        default=DEFAULT_STAGING_DIR,
        help="Directory where upload-ready files are staged before S3 upload",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Materialize files and print the upload plan without calling S3",
    )
    parser.add_argument(
        "--uploader",
        choices=("auto", "boto3", "aws-cli"),
        default="auto",
        help="Upload backend. auto uses boto3 when available, otherwise AWS CLI",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    items = materialize_uploads(
        bucket_uri=args.bucket_uri,
        staging_dir=args.staging_dir,
        dr_input_root=args.dr_input_root,
        source_ids=tuple(args.source_ids or DEFAULT_SOURCE_IDS),
    )

    for item in items:
        print(f"{item.local_path} -> {item.s3_uri}")

    if args.dry_run:
        return 0

    upload_materialized_files(items, uploader=args.uploader)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
