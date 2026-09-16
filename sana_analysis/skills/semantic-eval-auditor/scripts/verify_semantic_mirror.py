#!/usr/bin/env python3
"""Verify that a semantic eval mirror was actually written and is well formed."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sana_analysis.semantic_mirror import (  # noqa: E402
    SEMANTIC_COLUMNS,
    collect_mirror_issues,
    read_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-eval", required=True)
    parser.add_argument("--mirrored-eval", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_path = Path(args.source_eval)
    mirror_path = Path(args.mirrored_eval)

    issues = collect_mirror_issues(source_path, mirror_path)
    if issues:
        print("Semantic mirror verification FAILED")
        for issue in issues:
            print(f"- {issue}")
        raise SystemExit(1)

    _fields, mirror_rows = read_csv(mirror_path)
    print("Semantic mirror verification OK")
    print(f"source_eval={source_path}")
    print(f"mirrored_eval={mirror_path}")
    print(f"rows={len(mirror_rows)}")
    print(f"semantic_columns={','.join(SEMANTIC_COLUMNS)}")
    print("exact_match=unchanged")


if __name__ == "__main__":
    main()
