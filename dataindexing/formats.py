"""
Family detection, metadata-filename filtering and XML/KML shape extraction.

The one implementation of the lake's content-family vocabulary. Both the
evaluation agent's tools and the indexing pipelines read it from here; keeping
a second copy next to the S3 readers is what let ``xml`` silently go missing
from the sources layer.

Content families
----------------
  csv   — first non-empty line contains a common delimiter (, \\t | ;)
  json  — stripped content starts with { or [
  xml   — stripped content starts with an XML declaration or root tag
  text  — everything else
"""
import re
import xml.etree.ElementTree as ET
from collections import Counter
from typing import Any, Dict, List, Literal

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


# ---------------------------------------------------------------------------
# XML/KML shape extraction
# ---------------------------------------------------------------------------

_XML_NAMESPACE_RE = re.compile(r'\bxmlns(?::([A-Za-z_][\w.-]*))?=["\']([^"\']+)["\']')
_XML_SIMPLE_FIELD_RE = re.compile(
    r'<(?:[\w.-]+:)?SimpleField\b[^>]*\bname=["\']([^"\']+)["\']',
    re.IGNORECASE,
)
_XML_SIMPLE_DATA_RE = re.compile(
    r'<(?:[\w.-]+:)?SimpleData\b[^>]*\bname=["\']([^"\']+)["\']',
    re.IGNORECASE,
)
_XML_OPEN_TAG_RE = re.compile(r"<(?![!?/])([A-Za-z_][\w:.-]*)\b")
_XML_LEADING_NOISE_RE = re.compile(
    r"^(?:<\?xml.*?\?>\s*)?(?:<!--.*?-->\s*)*(?:<!DOCTYPE.*?>\s*)*",
    re.DOTALL | re.IGNORECASE,
)

# Callers hand build_xml_preview a bounded peek, not the whole object, so a
# full ElementTree parse is only safe when the object fits inside that peek.
# 64 KB matches the range-GET budget the lake tools use; callers with a
# different budget pass their own via ``peek_bytes``.
_DEFAULT_PEEK_BYTES = 65_536


def local_xml_name(tag: str | None) -> str | None:
    """Return an XML tag without namespace or prefix decoration."""
    if not tag:
        return tag
    if tag.startswith("{") and "}" in tag:
        tag = tag.split("}", 1)[1]
    if ":" in tag:
        tag = tag.split(":", 1)[1]
    return tag


def _unique_preserve_order(values: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _extract_xml_namespaces(text: str) -> Dict[str, str]:
    namespaces: Dict[str, str] = {}
    for prefix, uri in _XML_NAMESPACE_RE.findall(text):
        key = prefix or "default"
        namespaces[key] = uri
    return namespaces


def _strip_xml_leading_noise(text: str) -> str:
    return _XML_LEADING_NOISE_RE.sub("", text.lstrip(), count=1)


def _extract_xml_root_tag(text: str) -> str | None:
    stripped = _strip_xml_leading_noise(text)
    m = _XML_OPEN_TAG_RE.match(stripped)
    if not m:
        return None
    return local_xml_name(m.group(1))


def _extract_xml_schema_fields(text: str) -> List[str]:
    names = _XML_SIMPLE_FIELD_RE.findall(text) + _XML_SIMPLE_DATA_RE.findall(text)
    return _unique_preserve_order([name.strip() for name in names if name.strip()])


def _extract_xml_record_tag_candidates(text: str, root_tag: str | None) -> List[str]:
    tags = [local_xml_name(tag) for tag in _XML_OPEN_TAG_RE.findall(_strip_xml_leading_noise(text))]
    filtered = [tag for tag in tags if tag]
    counts = Counter(filtered)

    candidates: List[str] = []
    if counts.get("Placemark"):
        candidates.append("Placemark")

    for tag, count in counts.most_common():
        if tag in {root_tag, "SimpleField", "SimpleData"}:
            continue
        if count >= 2:
            candidates.append(tag)

    if not candidates:
        for tag, _count in counts.most_common():
            if tag in {root_tag, "SimpleField", "SimpleData"}:
                continue
            candidates.append(tag)
            if len(candidates) >= 5:
                break

    return _unique_preserve_order(candidates)[:5]


def _build_xml_preview_from_tree(root: ET.Element, text: str) -> Dict[str, Any]:
    root_tag = local_xml_name(root.tag)
    tags = [local_xml_name(elem.tag) for elem in root.iter() if isinstance(elem.tag, str)]
    counts = Counter(tag for tag in tags if tag)
    schema_fields = _unique_preserve_order(
        [
            elem.attrib["name"].strip()
            for elem in root.iter()
            if local_xml_name(elem.tag) in {"SimpleField", "SimpleData"}
            and elem.attrib.get("name", "").strip()
        ]
    )

    record_candidates: List[str] = []
    if counts.get("Placemark"):
        record_candidates.append("Placemark")
    for tag, count in counts.most_common():
        if tag in {root_tag, "SimpleField", "SimpleData"}:
            continue
        if count >= 2:
            record_candidates.append(tag)
    if not record_candidates:
        for tag, _count in counts.most_common():
            if tag in {root_tag, "SimpleField", "SimpleData"}:
                continue
            record_candidates.append(tag)
            if len(record_candidates) >= 5:
                break

    return {
        "xml_root_tag": root_tag,
        "xml_namespaces": _extract_xml_namespaces(text),
        "xml_schema_fields": schema_fields,
        "xml_record_tag_candidates": _unique_preserve_order(record_candidates)[:5],
        "xml_preview_mode": "parsed",
    }


def build_xml_preview(
    text: str,
    size_bytes: int,
    *,
    peek_bytes: int = _DEFAULT_PEEK_BYTES,
) -> Dict[str, Any]:
    """Summarize an XML/KML document's root tag, namespaces and record shape."""
    if size_bytes <= peek_bytes:
        try:
            root = ET.fromstring(text)
            return _build_xml_preview_from_tree(root, text)
        except ET.ParseError:
            pass

    root_tag = _extract_xml_root_tag(text)
    return {
        "xml_root_tag": root_tag,
        "xml_namespaces": _extract_xml_namespaces(text),
        "xml_schema_fields": _extract_xml_schema_fields(text),
        "xml_record_tag_candidates": _extract_xml_record_tag_candidates(text, root_tag),
        "xml_preview_mode": "heuristic",
    }


def normalize_xml_record_tag(record_tag: str | None) -> str | None:
    """Return a bare local tag name from a user-supplied record tag."""
    if not record_tag:
        return None
    tag = str(record_tag).strip().strip("<>/")
    return local_xml_name(tag)


def _xml_text(elem: ET.Element) -> str:
    return " ".join(part.strip() for part in elem.itertext() if part and part.strip())


def xml_record_to_row(record: ET.Element) -> Dict[str, str]:
    """
    Convert one XML/KML record element into a shallow row.

    KML data.gov exports usually store useful attributes as
    `<SimpleData name="FIELD">value</SimpleData>`. Plain XML often stores
    useful values in leaf child tags. We support both without inventing a full
    XML-to-table model.
    """
    row: Dict[str, str] = {}

    for name, value in record.attrib.items():
        field = local_xml_name(name)
        text = str(value).strip()
        if field and text:
            row[field] = text

    for elem in record.iter():
        if not isinstance(elem.tag, str):
            continue
        local = local_xml_name(elem.tag)
        if local == "SimpleData":
            name = elem.attrib.get("name", "").strip()
            text = _xml_text(elem)
            if name and text:
                row[name] = text

    for elem in record.iter():
        if elem is record or not isinstance(elem.tag, str):
            continue
        local = local_xml_name(elem.tag)
        if not local or local in {"ExtendedData", "SchemaData", "SimpleData", "SimpleField"}:
            continue
        if list(elem):
            continue
        text = _xml_text(elem)
        if text and local not in row:
            row[local] = text

    return row
