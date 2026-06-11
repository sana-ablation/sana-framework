"""URI normalization helpers shared by index builders and schema loaders."""

from __future__ import annotations

import os
import re


def strip_s3_bucket(uri: str) -> str:
    """Strip any S3 bucket prefix from a URI-like string."""
    return re.sub(r"^s3://[^/]+/", "", str(uri or ""))


def uri_to_schema_stem(uri: str) -> str:
    """Normalize an S3 URI to the schema-key stem used by table schema JSONL."""
    raw_key = strip_s3_bucket(uri)
    raw_key = raw_key.replace("/v1/", "/")
    return os.path.splitext(raw_key)[0]

