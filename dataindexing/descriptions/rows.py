"""Description artifact row construction and normalization."""

from __future__ import annotations

from typing import Any


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

