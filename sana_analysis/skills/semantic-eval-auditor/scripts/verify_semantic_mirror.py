#!/usr/bin/env python3
"""Verify that a semantic eval mirror was actually written and is well formed."""

from __future__ import annotations

import argparse
import csv
import io
from pathlib import Path


SEMANTIC_COLUMNS = [
    "semantic_match",
    "semantic_reason",
    "semantic_bucket",
    "log_error_bucket",
    "log_error_evidence",
]

SEMANTIC_BUCKETS = {
    "semantic_correct",
    "semantic_incorrect",
    "answer_unknown_blank",
}

LOG_ERROR_BUCKETS = {
    "",
    "error_turns_exhausted",
    "error_tools_limit",
    "error_tokens_reached",
    "error_context_overflow",
    "error_event_loop",
    "error_unknown",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-eval", required=True)
    parser.add_argument("--mirrored-eval", required=True)
    return parser.parse_args()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        raise FileNotFoundError(path)
    text = path.read_bytes().replace(b"\x00", b"").decode("utf-8", errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    return list(reader.fieldnames or []), list(reader)


def main() -> None:
    args = parse_args()
    source_path = Path(args.source_eval)
    mirror_path = Path(args.mirrored_eval)

    issues: list[str] = []
    try:
        source_fields, source_rows = read_csv(source_path)
    except FileNotFoundError:
        source_fields, source_rows = [], []
        issues.append(f"source_eval does not exist: {source_path}")

    try:
        mirror_fields, mirror_rows = read_csv(mirror_path)
    except FileNotFoundError:
        mirror_fields, mirror_rows = [], []
        issues.append(f"mirrored_eval does not exist: {mirror_path}")

    if issues:
        print("Semantic mirror verification FAILED")
        for issue in issues:
            print(f"- {issue}")
        raise SystemExit(1)

    if len(source_rows) != len(mirror_rows):
        issues.append(f"row count mismatch: source={len(source_rows)} mirror={len(mirror_rows)}")

    missing_columns = [column for column in SEMANTIC_COLUMNS if column not in mirror_fields]
    if missing_columns:
        issues.append(f"missing semantic columns: {', '.join(missing_columns)}")

    if "task_id" in source_fields and "task_id" in mirror_fields:
        source_task_ids = [row.get("task_id", "") for row in source_rows]
        mirror_task_ids = [row.get("task_id", "") for row in mirror_rows]
        if source_task_ids != mirror_task_ids:
            issues.append("task_id sequence changed")

    if "exact_match" in source_fields and "exact_match" in mirror_fields:
        changed = [
            idx
            for idx, (source_row, mirror_row) in enumerate(zip(source_rows, mirror_rows), start=1)
            if source_row.get("exact_match", "") != mirror_row.get("exact_match", "")
        ]
        if changed:
            issues.append(f"exact_match changed on rows: {', '.join(map(str, changed[:10]))}")

    for idx, row in enumerate(mirror_rows, start=1):
        semantic_bucket = row.get("semantic_bucket", "")
        semantic_match = row.get("semantic_match", "")
        semantic_reason = row.get("semantic_reason", "")
        log_error_bucket = row.get("log_error_bucket", "")
        log_error_evidence = row.get("log_error_evidence", "")

        if semantic_bucket not in SEMANTIC_BUCKETS:
            issues.append(f"row {idx}: invalid semantic_bucket `{semantic_bucket}`")
        if semantic_bucket == "semantic_correct" and semantic_match != "1":
            issues.append(f"row {idx}: semantic_correct must have semantic_match=1")
        if semantic_bucket in {"semantic_incorrect", "answer_unknown_blank"} and semantic_match != "0":
            issues.append(f"row {idx}: non-correct bucket must have semantic_match=0")
        if not semantic_reason.strip():
            issues.append(f"row {idx}: semantic_reason is blank")
        if log_error_bucket not in LOG_ERROR_BUCKETS:
            issues.append(f"row {idx}: invalid log_error_bucket `{log_error_bucket}`")
        if not log_error_bucket and log_error_evidence.strip():
            issues.append(f"row {idx}: log_error_evidence set without log_error_bucket")

    if issues:
        print("Semantic mirror verification FAILED")
        for issue in issues:
            print(f"- {issue}")
        raise SystemExit(1)

    print("Semantic mirror verification OK")
    print(f"source_eval={source_path}")
    print(f"mirrored_eval={mirror_path}")
    print(f"rows={len(mirror_rows)}")
    print(f"semantic_columns={','.join(SEMANTIC_COLUMNS)}")
    print("exact_match=unchanged")


if __name__ == "__main__":
    main()
