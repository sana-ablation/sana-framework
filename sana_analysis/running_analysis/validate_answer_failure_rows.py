#!/usr/bin/env python3
"""Validate answer-failure audit rows and optionally rewrite validation columns."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sana_analysis.running_analysis.answer_failure_validation import validate_answer_failure_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default="results_semantic_answer_failures")
    parser.add_argument("--logs-dir", default="logs/modes")
    parser.add_argument("--rewrite", action="store_true")
    args = parser.parse_args()

    outputs = validate_answer_failure_root(
        source_root=args.source_root,
        logs_dir=args.logs_dir,
        rewrite=args.rewrite,
    )
    print(f"Wrote {outputs['failures_path']}")
    print(f"Wrote {outputs['report_path']}")
    print(f"Invalid rows: {len(outputs['invalid_rows'])}")


if __name__ == "__main__":
    main()
