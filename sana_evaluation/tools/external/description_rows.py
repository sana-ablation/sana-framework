"""Shared validation helpers for dataset-description JSONL rows."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

FORBIDDEN_DESCRIPTION_SOURCES = {
    "task_manifest_fallback",
    "tasks_mini_manifest_fallback",
}

DESCRIPTION_FIELDS = (
    "dataset_uri",
    "metadata",
    "content",
    "original_metadata",
    "generated_metadata",
    "description",
    "input_tokens",
    "output_tokens",
    "input_cost_usd",
    "output_cost_usd",
    "cost_usd",
    "error",
)


def description_uri(row: Dict[str, Any]) -> str:
    return str(row.get("dataset_uri") or row.get("s3_uri") or row.get("uri") or "").strip()


def reject_forbidden_description_row(
    row: Dict[str, Any],
    *,
    path: Optional[Path] = None,
) -> None:
    source = str(row.get("description_source") or "").strip()
    if source in FORBIDDEN_DESCRIPTION_SOURCES:
        location = f"{path}: " if path is not None else ""
        uri = description_uri(row) or "<unknown uri>"
        raise ValueError(
            f"{location}description_source={source!r} is not allowed for {uri}; "
            "regenerate this row with an LLM instead of using manifest fallback text."
        )


def has_valid_description(row: Dict[str, Any]) -> bool:
    uri = description_uri(row)
    desc = str(row.get("description") or "").strip()
    error = row.get("error")
    return bool(uri and desc and not error)
