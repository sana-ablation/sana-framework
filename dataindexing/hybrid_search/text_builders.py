"""Pure text-building helpers for hybrid-search indexes."""

from __future__ import annotations

import re

from dataindexing.hybrid_search.uri_matching import strip_s3_bucket

BUILD_MODES = ("infused", "basic")

_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def tokenize_column_name(col: str) -> str:
    """Split a column identifier on snake_case, kebab-case, and camelCase."""
    s = str(col or "").replace("_", " ").replace("-", " ")
    s = _CAMEL_RE.sub(" ", s)
    return " ".join(s.split())


def metadata_from_uri(uri: str) -> str:
    """Convert an S3 URI into simple searchable path metadata."""
    path = strip_s3_bucket(uri)
    words: list[str] = []
    for part in path.strip("/").split("/"):
        stem = re.sub(r"\.[A-Za-z0-9]+$", "", part)
        words.extend(w for w in re.split(r"[-_]", stem) if w)
    return " ".join(words)


def normalize_text(*parts: str) -> str:
    """Collapse whitespace across non-empty text parts."""
    return " ".join(" ".join(str(p).split()) for p in parts if p and str(p).strip()).strip()


def build_schema_text(
    *,
    build_mode: str,
    uri: str,
    doc: str,
    desc_row: dict[str, str] | None,
    schema_cols: list[str] | None,
) -> str:
    """Build the text inserted into the schema search table for one document."""
    cols_text = (
        " ".join(tokenize_column_name(c) for c in schema_cols) if schema_cols else ""
    )

    if build_mode == "infused":
        gen_meta = desc_row["generated_metadata"] if desc_row else ""
        orig_meta = desc_row["original_metadata"] if desc_row else ""
        schema_text = normalize_text(gen_meta, cols_text, orig_meta)
    elif build_mode == "basic":
        schema_text = normalize_text(metadata_from_uri(uri), cols_text)
    else:
        raise ValueError(f"Unknown build mode: {build_mode}")

    if not schema_text:
        schema_text = " ".join(str(doc or "").split()[:50])
    return schema_text


def build_description_chunk(
    build_mode: str,
    desc_row: dict[str, str] | None,
) -> str | None:
    """Build the optional description chunk for infused content indexes."""
    if build_mode == "basic" or desc_row is None:
        return None
    if build_mode != "infused":
        raise ValueError(f"Unknown build mode: {build_mode}")

    desc_chunk = normalize_text(
        desc_row.get("generated_metadata", ""),
        desc_row.get("description", ""),
    )
    return desc_chunk or None
