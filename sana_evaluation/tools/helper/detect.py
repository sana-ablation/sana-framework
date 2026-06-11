"""
Family detection and metadata-filename filtering.

Content families
----------------
  csv   — first non-empty line contains a common delimiter (, \\t | ;)
  json  — stripped content starts with { or [
  xml   — stripped content starts with an XML declaration or root tag
  text  — everything else
"""
import re
from typing import Literal

ContentFamily = Literal["csv", "json", "xml", "text"]

DELIMITERS = (",", "\t", "|", ";")
_XML_ROOT_TAG_RE = re.compile(r"^<(?![!?/])([A-Za-z_][\w:.-]*)\b")

# ---------------------------------------------------------------------------
# Metadata-filename heuristics (ported from legacy s3_search.py)
# ---------------------------------------------------------------------------

_SOCRATA_ID_RE = re.compile(r"^[a-z0-9]{4}-[a-z0-9]{4}$")
_PURE_NUMBER_RE = re.compile(r"^\d+(-\d+)?$")
_RANDOM_SUFFIX_RE = re.compile(r"^.+-[A-Za-z0-9]{6}$")

_METADATA_EXACT: frozenset[str] = frozenset({
    "metadata", "gmi", "open-licenses", "legalcode", "government-works",
    "index", "odc-odbl", "wmsserver", "resolve", "request",
    "edit", "search", "contact", "policyinformation", "gmxcodelists",
    "bios", "hires", "cwhr", "license", "readme",
    "signed-metadata", "headers", "dcat-us", "catalog", "iso", "cc-zero", "cc-by",
})

_SKIP_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".pdf", ".zip"})


def is_metadata_filename(filename: str) -> bool:
    """Return True if *filename* (with or without extension) looks like metadata."""
    stem = filename.rsplit(".", 1)[0].lower()
    if stem in _METADATA_EXACT:
        return True
    if _SOCRATA_ID_RE.match(stem):
        return True
    if _PURE_NUMBER_RE.match(stem):
        return True
    return bool(_RANDOM_SUFFIX_RE.match(stem))


def should_skip(filename: str) -> bool:
    """Return True for binary/metadata files that are never worth ingesting."""
    lower = filename.lower()
    if any(lower.endswith(ext) for ext in _SKIP_EXTENSIONS):
        return True
    return is_metadata_filename(lower.rsplit("/", 1)[-1])


# ---------------------------------------------------------------------------
# Content-family detection
# ---------------------------------------------------------------------------

def detect_family(content: str) -> ContentFamily:
    """Detect content family from the first few bytes of *content*."""
    stripped = content.lstrip()
    if stripped.startswith("{") or stripped.startswith("["):
        return "json"
    head = stripped[:512]
    lower_head = head.lower()
    if (
        lower_head.startswith("<?xml")
        or lower_head.startswith("<kml")
        or _XML_ROOT_TAG_RE.match(head)
    ):
        return "xml"
    first_line = stripped.split("\n", 1)[0]
    if any(d in first_line for d in DELIMITERS):
        return "csv"
    return "text"


def is_table_content(content: str) -> bool:
    """
    Structural peek: return True when content looks like a table (csv or json-lines).
    Used for quick filtering without full parsing.
    """
    lines = [ln.strip() for ln in content.split("\n") if ln.strip()]
    if len(lines) < 3:
        return False

    # JSON-lines (objects on each line)
    jsonl = sum(1 for ln in lines[:5] if ln.startswith("{") and ln.endswith("}"))
    if jsonl >= 3:
        return True

    # Whole-file JSON array/object — not a row-level table
    first_char = content.strip()[0]
    if first_char in ("{", "["):
        return False

    # Delimiter consistency across first 5 lines
    for delim in DELIMITERS:
        counts = []
        for ln in lines[:5]:
            clean = re.sub(r'"[^"]*"', "", ln)
            counts.append(clean.count(delim))
        valid = [c for c in counts if c > 0]
        if len(valid) >= 3 and len(set(valid[:3])) == 1 and valid[0] >= 1:
            return True

    return False
