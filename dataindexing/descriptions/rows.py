"""Description artifact row construction, normalization and validation."""

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


def normalize_field(value: Any) -> str:
    """Return a single-space-normalized string for text fields."""
    return " ".join(str(value or "").split())


def compose_content(generated_metadata: str, description: str) -> str:
    """Compose the searchable content field from generated metadata and description."""
    return normalize_field(f"{generated_metadata or ''} {description or ''}")


def build_row(
    dataset_uri: str,
    original_metadata: str,
    generated_metadata: str,
    description: str,
    *,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    input_cost_usd: float | None = None,
    output_cost_usd: float | None = None,
    cost_usd: float | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """Build a compatibility row for JSONL/parquet description artifacts."""
    generated_clean = normalize_field(generated_metadata)
    description_clean = normalize_field(description)
    content = compose_content(generated_clean, description_clean)
    return {
        "dataset_uri": dataset_uri,
        "metadata": generated_clean,
        "content": content,
        "original_metadata": original_metadata,
        "generated_metadata": generated_clean,
        "description": description_clean,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_cost_usd": input_cost_usd,
        "output_cost_usd": output_cost_usd,
        "cost_usd": cost_usd,
        "error": error,
    }


# ---------------------------------------------------------------------------
# Reading back rows written above
# ---------------------------------------------------------------------------


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
