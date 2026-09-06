#!/usr/bin/env python3
"""Build snippets.jsonl from datalake_silver.parquet.

Each output line has:
    {"dataset_uri": "...", "dataset_snippet": "..."}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq


def _truncate_words(text: str, max_words: int) -> str:
    words = (text or "").split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words])


def build_snippets(parquet_path: Path, output_path: Path, *, max_words: int) -> int:
    pf = pq.ParquetFile(parquet_path)
    written = 0

    with output_path.open("w") as out:
        for i in range(pf.metadata.num_row_groups):
            batch = pf.read_row_group(i, columns=["dataset_uri", "content"])
            for uri, content in zip(
                batch.column("dataset_uri").to_pylist(),
                batch.column("content").to_pylist(),
            ):
                if not uri:
                    continue
                snippet = _truncate_words(str(content or ""), max_words=max_words)
                out.write(
                    json.dumps(
                        {
                            "dataset_uri": str(uri),
                            "dataset_snippet": snippet,
                        }
                    )
                )
                out.write("\n")
                written += 1

    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to datalake_silver.parquet")
    parser.add_argument("--output", default="benchmarks/lakeqa/tasks-mini/artifacts/snippets.jsonl")
    parser.add_argument("--max-words", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    parquet_path = Path(args.input)
    output_path = Path(args.output)
    written = build_snippets(parquet_path, output_path, max_words=args.max_words)
    print(f"Wrote {written} snippets to {output_path}")


if __name__ == "__main__":
    main()
