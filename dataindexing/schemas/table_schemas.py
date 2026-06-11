"""Table schema JSONL loading helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path


def load_table_schemas(schemas_jsonl_path: str) -> dict[str, list[str]]:
    """Load table columns keyed by normalized S3-key stem."""
    if not Path(schemas_jsonl_path).exists():
        print(
            f"WARN: schemas JSONL not found: {schemas_jsonl_path}; "
            f"schema search will run without column metadata"
        )
        return {}

    out: dict[str, list[str]] = {}
    with open(schemas_jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            for table in rec.get("tables", []):
                sk = table.get("s3_key")
                cols = table.get("columns")
                if not sk or not cols:
                    continue
                normalized = str(sk).replace("/v1/", "/")
                stem = os.path.splitext(normalized)[0]
                out[stem] = list(cols)

    print(f"Loaded {len(out)} table schemas from: {schemas_jsonl_path}")
    return out
