"""
Data Lake Access Tools v2 for LLM Agent

Improvements over v1 (agent_tools.py):
  - peek_file: range-GET + family detection + column headers (replaces inspect_file)
  - query_file: query S3 directly via DuckDB httpfs — no download needed
  - parse_xml_records: stream XML/KML records into fields/counts without arbitrary Python
  - download_smart: budget-aware range-GET per content family, skips metadata/binary files
  - execute_code, search, search_keyword, list_files, get_sandbox_info, cleanup_sandbox: unchanged

Bucket: lakeqa-yc4103-datalake
Folders: wikipedia/, datagov/
"""

from __future__ import annotations

import base64
import datetime
import decimal
import io
import json
import os
import re
import traceback
import uuid
import xml.etree.ElementTree as ET
from collections import Counter, deque
from typing import Any, Dict, List, Optional, Tuple

import duckdb
from dotenv import load_dotenv
from strands import tool

from sana_evaluation.helper.peek_profile import load_dataset_profile, select_dataset_profile_fields

from .agent_tools import (  # noqa: F401 — re-exported
    configure_benchmark as _configure_base_benchmark,
    search,
    search_keyword,
    list_files,
    execute_code,
    download,
    get_sandbox_info,
    cleanup_sandbox,
    set_sandbox_dir,
    _get_s3_client,
    _get_sandbox_dir,
    _resolve_dataset_folder,
    s3_access_mode,
    BUCKET,
    REGION,
    SANDBOX_BASE_DIR,
    _resolve_file_reference,
    _run_tool_with_timeout,
    _tool_timeout_seconds,
)
from .helper.detect import detect_family, should_skip

load_dotenv()


def configure_benchmark(benchmark: str | None = None) -> str:
    """Configure both v1 and v2 tool modules for a benchmark bucket."""
    global BUCKET
    BUCKET = _configure_base_benchmark(benchmark)
    return BUCKET


# ---------------------------------------------------------------------------
# Budget constants (mirrored from streams.py S3Config defaults)
# ---------------------------------------------------------------------------
_PEEK_BYTES = 65_536          # 64 KB for initial range-GET / peek
_READ_BYTES = 1_048_576       # 1 MB budget for read_file / search_in_file
_MAX_CSV_ROWS = 500
_CSV_BYTES_PER_ROW = 512
_MAX_JSON_ITEMS = 200
_JSON_BYTES_PER_ITEM = 2_048
_MAX_TEXT_CHARS = 50_000
_QUERY_ROW_CAP = 200
_SEARCH_MAX_MATCHES = 20
_SEARCH_CONTEXT_LINES = 2
_QUERY_MAX_FILE_BYTES = 500 * 1024 * 1024  # 500 MB — above this, download first
_MAX_SPREADSHEET_PEEK_BYTES = 128 * 1024 * 1024
_MAX_SPREADSHEET_PREVIEW_COLUMNS = 30
_TOOL_RESULT_CHAR_CAP = 6_000              # ~1.5k tokens — keeps single tool results from dominating context


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _s3_range_get(s3, key: str, start: int, end: int) -> bytes:
    """Sync range-GET from the configured bucket."""
    resp = s3.get_object(Bucket=BUCKET, Key=key, Range=f"bytes={start}-{end}")
    return resp["Body"].read()


def _s3_head(s3, key: str) -> int:
    """Return file size in bytes via HeadObject."""
    meta = s3.head_object(Bucket=BUCKET, Key=key)
    return meta.get("ContentLength", 0)


def _s3_peek_text(s3, key: str, size_bytes: int) -> Tuple[str, int]:
    """Return UTF-8 preview text plus sampled byte count for a possibly empty object."""
    if size_bytes <= 0:
        return "", 0
    end = min(_PEEK_BYTES - 1, size_bytes - 1)
    raw = _s3_range_get(s3, key, 0, end)
    return raw.decode("utf-8", errors="replace"), len(raw)


def _s3_read_bounded(s3, key: str, size_bytes: int, max_bytes: int) -> bytes:
    """Read a whole small object, failing clearly when it exceeds the cap."""
    if size_bytes > max_bytes:
        raise ValueError(f"file is {size_bytes} bytes, above {max_bytes} byte spreadsheet peek cap")
    if size_bytes <= 0:
        return b""
    return _s3_range_get(s3, key, 0, size_bytes - 1)


def _duckdb_connection() -> duckdb.DuckDBPyConnection:
    """Create a fresh in-memory DuckDB connection with httpfs and AWS credentials."""
    conn = duckdb.connect(":memory:")
    conn.execute("INSTALL httpfs")
    conn.execute("LOAD httpfs")
    region = os.getenv("AWS_DEFAULT_REGION", REGION)
    conn.execute(f"SET s3_region='{region}'")
    if s3_access_mode() == "unsigned":
        # Keep DuckDB S3 access anonymous for public buckets.
        conn.execute("SET s3_access_key_id=''")
        conn.execute("SET s3_secret_access_key=''")
        conn.execute("SET s3_session_token=''")
    else:
        aws_key = os.getenv("AWS_ACCESS_KEY_ID", "")
        aws_secret = os.getenv("AWS_SECRET_ACCESS_KEY", "")
        aws_session = os.getenv("AWS_SESSION_TOKEN", "")
        if aws_key:
            conn.execute(f"SET s3_access_key_id='{aws_key}'")
        if aws_secret:
            conn.execute(f"SET s3_secret_access_key='{aws_secret}'")
        if aws_session:
            conn.execute(f"SET s3_session_token='{aws_session}'")
    return conn


_MAX_OBJECT_SIZE_RE = re.compile(
    r'"maximum_object_size".*?bytes\s*exceeded.*?\(>(\d+)\s*bytes\)',
    re.DOTALL,
)
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


def _normalize_sql_backticks(sql: str) -> str:
    """
    Convert MySQL-style backtick identifiers to DuckDB's double-quoted form.

    Agents trained on MySQL/SQLite repeatedly write `` `column name` `` even
    though DuckDB only accepts `"column name"`. Observed ~17 times in eval
    logs as Parser Errors. Auto-fix at the source eliminates the round-trip.

    Carefully preserves backticks that fall INSIDE single-quoted string
    literals (e.g. `WHERE name = 'O\\`Brien'`) and inside already-double-
    quoted identifiers (e.g. `"weird\\`col"`). Handles escaped single quotes
    (`''`) inside literals.
    """
    out: List[str] = []
    in_single = False
    in_double = False
    i = 0
    n = len(sql)
    while i < n:
        ch = sql[i]
        if in_single:
            if ch == "'":
                # `''` is an escaped single quote inside a literal
                if i + 1 < n and sql[i + 1] == "'":
                    out.append("''")
                    i += 2
                    continue
                in_single = False
            out.append(ch)
        elif in_double:
            if ch == '"':
                # `""` is an escaped double quote inside an identifier
                if i + 1 < n and sql[i + 1] == '"':
                    out.append('""')
                    i += 2
                    continue
                in_double = False
            out.append(ch)
        else:
            if ch == "'":
                in_single = True
                out.append(ch)
            elif ch == '"':
                in_double = True
                out.append(ch)
            elif ch == "`":
                out.append('"')
            else:
                out.append(ch)
        i += 1
    return "".join(out)


def _rewrite_query_error(raw: str) -> str:
    """
    Rewrite low-level DuckDB error strings into actionable remediation hints
    so the agent stops thrashing on platform-side limits.

    The headline case: when DuckDB's read_json_auto rejects a JSON object
    larger than `maximum_object_size`, the stock hint says "Try increasing
    maximum_object_size" — which is misleading because the cap is hard-coded
    in this module and the agent cannot change it. Logs show the agent then
    retries the same query, malforms a download call, or pivots to peek_file
    which doesn't help. The right remediation is the same one used by the
    pre-check at line 483: download the file and process it with execute_code.
    """
    m = _MAX_OBJECT_SIZE_RE.search(raw)
    if m:
        observed = m.group(1)
        limit_mb = _QUERY_MAX_FILE_BYTES // (1024 * 1024)
        return (
            f"File contains a JSON object larger than the {limit_mb} MB query limit "
            f"({observed} bytes observed). query_file cannot stream it. "
            "Use download to fetch the file, then execute_code with a "
            "streaming JSON parser (e.g. ijson) or pandas.read_json with "
            "chunksize."
        )
    if "Timeout was reached" in raw and "HTTP GET" in raw:
        return (
            "S3 read timed out — file is likely too large to query directly "
            "over httpfs. Use download to fetch it locally, then execute_code "
            "to process it. Original: " + raw
        )
    if "Parser Error" in raw and "`" in raw:
        # Agent is using MySQL-style backtick identifiers; DuckDB requires
        # double quotes. Observed ~17 times in production logs.
        return (
            'DuckDB uses double quotes for identifiers, not backticks. '
            'Replace `column name` with "column name" in your SQL. '
            "Original: " + raw
        )
    return raw


def _rewrite_unqueryable_family_error(family: str) -> str:
    """
    Replace the bare "File family '<X>' is not queryable with SQL" message
    with a hint naming the right tool to use instead. The agent has no way
    to know that text files should go through grep/read/peek; spell it out.
    """
    if family == "xml":
        return (
            "XML/KML was detected. query_file does not support XML because "
            "SQL has no stable row model for arbitrary XML documents. "
            "Use peek_file to inspect tags and schema fields, parse_xml_records "
            "to extract/filter/group structured XML/KML records, grep_file to "
            "search for specific values, or read_file to inspect nearby text. "
            "Do not use query_file or execute_code for XML/KML sources."
        )
    if family == "text":
        return (
            "File contents are plain text — query_file only handles CSV and "
            "JSON. Use peek_file to inspect structure, grep_file to search "
            "for specific values, or read_file to load lines directly."
        )
    return (
        f"File family '{family}' is not queryable with SQL via query_file "
        "(only CSV and JSON are supported). Use peek_file to inspect what "
        "the file actually contains, then pick a tool that matches its "
        "format. Do not use query_file or execute_code unless the source is "
        "tabular or JSON."
    )


def _to_json_safe(value: Any) -> Any:
    """
    Convert DuckDB row values into JSON-serializable primitives.

    DuckDB returns native Python types for SQL DATE / TIMESTAMP / TIME / INTERVAL /
    DECIMAL / UUID / BLOB columns, which json.dumps cannot serialize. Without
    this conversion, even `SELECT * FROM t LIMIT 1` against any table with a
    date column raises `Object of type date is not JSON serializable`.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, datetime.datetime):
        return value.isoformat()
    if isinstance(value, datetime.date):
        return value.isoformat()
    if isinstance(value, datetime.time):
        return value.isoformat()
    if isinstance(value, datetime.timedelta):
        return value.total_seconds()
    if isinstance(value, decimal.Decimal):
        return str(value)
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return base64.b64encode(bytes(value)).decode("ascii")
    if isinstance(value, dict):
        return {str(k): _to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_to_json_safe(v) for v in value]
    # Fallback for unexpected types (e.g., numpy scalars). Stringify rather
    # than crash — surfaces the value while keeping the tool call alive.
    return str(value)


def _strip_folder_prefix(dataset_id: str) -> str:
    """
    Silently strip a hallucinated leading `wikipedia/` or `datagov/` prefix
    from a dataset_id.

    The agent frequently constructs dataset ids of the form `wikipedia/<page>`
    after seeing a Wikipedia mention (~38 read_file errors per eval, plus a
    handful in peek_file/grep_file). The actual dataset_id is the bare name
    (e.g. `Logan_Fontenelle`), which the agent already knows one turn later
    via search_prefix. Per tool_error_findings.md the recommended fix is to
    auto-strip the prefix and resolve.

    Strips a single optional leading slash, then a case-insensitive
    `wikipedia/` or `datagov/` segment. Idempotent. Returns the input
    unchanged for empty/None or strings that don't start with one of the
    known folder prefixes.
    """
    if not dataset_id or not isinstance(dataset_id, str):
        return dataset_id
    candidate = dataset_id.lstrip("/")
    lowered = candidate.lower()
    for folder in ("wikipedia/", "datagov/"):
        if lowered.startswith(folder):
            return candidate[len(folder):]
    return dataset_id


def _local_xml_name(tag: str | None) -> str | None:
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
    return _local_xml_name(m.group(1))


def _extract_xml_schema_fields(text: str) -> List[str]:
    names = _XML_SIMPLE_FIELD_RE.findall(text) + _XML_SIMPLE_DATA_RE.findall(text)
    return _unique_preserve_order([name.strip() for name in names if name.strip()])


def _extract_xml_record_tag_candidates(text: str, root_tag: str | None) -> List[str]:
    tags = [_local_xml_name(tag) for tag in _XML_OPEN_TAG_RE.findall(_strip_xml_leading_noise(text))]
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
    root_tag = _local_xml_name(root.tag)
    tags = [_local_xml_name(elem.tag) for elem in root.iter() if isinstance(elem.tag, str)]
    counts = Counter(tag for tag in tags if tag)
    schema_fields = _unique_preserve_order(
        [
            elem.attrib["name"].strip()
            for elem in root.iter()
            if _local_xml_name(elem.tag) in {"SimpleField", "SimpleData"}
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


def _build_xml_preview(text: str, size_bytes: int) -> Dict[str, Any]:
    if size_bytes <= _PEEK_BYTES:
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


def _coerce_string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple, set)):
        values = list(value)
    else:
        values = [value]
    return [str(item).strip() for item in values if str(item).strip()]


def _normalize_xml_record_tag(record_tag: str | None) -> str | None:
    if not record_tag:
        return None
    tag = str(record_tag).strip().strip("<>/")
    return _local_xml_name(tag)


def _xml_text(elem: ET.Element) -> str:
    return " ".join(part.strip() for part in elem.itertext() if part and part.strip())


def _xml_record_to_row(record: ET.Element) -> Dict[str, str]:
    """
    Convert one XML/KML record element into a shallow row.

    KML data.gov exports usually store useful attributes as
    `<SimpleData name="FIELD">value</SimpleData>`. Plain XML often stores
    useful values in leaf child tags. We support both without inventing a full
    XML-to-table model.
    """
    row: Dict[str, str] = {}

    for name, value in record.attrib.items():
        field = _local_xml_name(name)
        text = str(value).strip()
        if field and text:
            row[field] = text

    for elem in record.iter():
        if not isinstance(elem.tag, str):
            continue
        local = _local_xml_name(elem.tag)
        if local == "SimpleData":
            name = elem.attrib.get("name", "").strip()
            text = _xml_text(elem)
            if name and text:
                row[name] = text

    for elem in record.iter():
        if elem is record or not isinstance(elem.tag, str):
            continue
        local = _local_xml_name(elem.tag)
        if not local or local in {"ExtendedData", "SchemaData", "SimpleData", "SimpleField"}:
            continue
        if list(elem):
            continue
        text = _xml_text(elem)
        if text and local not in row:
            row[local] = text

    return row


def _xml_row_matches_filters(row: Dict[str, str], filters: Dict[str, Any]) -> bool:
    for field, expected in filters.items():
        actual = row.get(str(field))
        if isinstance(expected, (list, tuple, set)):
            allowed = {str(value) for value in expected}
            if actual not in allowed:
                return False
        elif actual != str(expected):
            return False
    return True


def _select_xml_row_fields(row: Dict[str, str], fields: List[str]) -> Dict[str, str]:
    if not fields:
        return row
    return {field: row.get(field, "") for field in fields}


def _is_excel_path(file_path: str) -> bool:
    return file_path.lower().rsplit("?", 1)[0].endswith((".xlsx", ".xlsm", ".xltx", ".xltm"))


def _format_spreadsheet_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (datetime.date, datetime.datetime)):
        return value.isoformat()
    return str(value)


def _first_nonempty_rows(worksheet, max_rows: int, max_columns: int) -> tuple[list[list[str]], int]:
    rows: list[list[str]] = []
    max_seen_columns = 0
    for row in worksheet.iter_rows(values_only=True):
        values = [_format_spreadsheet_cell(value) for value in row]
        while values and values[-1] == "":
            values.pop()
        if not any(value != "" for value in values):
            continue
        max_seen_columns = max(max_seen_columns, len(values))
        if len(values) > max_columns:
            values = values[:max_columns] + [f"... {len(values) - max_columns} more columns"]
        rows.append(values)
        if len(rows) >= max_rows:
            break
    return rows, max_seen_columns


def _build_excel_preview(raw: bytes, max_rows: int) -> Dict[str, Any]:
    try:
        from openpyxl import load_workbook
    except Exception as e:
        return {
            "family": "xlsx",
            "preview_text": (
                "Excel workbook detected, but openpyxl is not installed. "
                "Install openpyxl or use download plus an environment with Excel support."
            ),
            "excel_error": f"openpyxl unavailable: {e}",
        }

    try:
        workbook = load_workbook(io.BytesIO(raw), read_only=True, data_only=True)
    except Exception as e:
        return {
            "family": "xlsx",
            "preview_text": f"Excel workbook detected, but preview failed: {e}",
            "excel_error": str(e),
        }

    sheet_names = list(workbook.sheetnames)
    sheet_infos: list[Dict[str, Any]] = []
    preview_blocks = [f"Excel workbook with sheets: {', '.join(sheet_names)}"]
    for sheet_name in sheet_names[:8]:
        worksheet = workbook[sheet_name]
        rows, max_seen_columns = _first_nonempty_rows(
            worksheet,
            max(max_rows, 1) + 1,
            _MAX_SPREADSHEET_PREVIEW_COLUMNS,
        )
        info: Dict[str, Any] = {
            "name": sheet_name,
            "max_row": worksheet.max_row,
            "max_column": worksheet.max_column,
        }
        if max_seen_columns > _MAX_SPREADSHEET_PREVIEW_COLUMNS:
            info["preview_column_limit"] = _MAX_SPREADSHEET_PREVIEW_COLUMNS
            info["preview_columns_truncated"] = max_seen_columns - _MAX_SPREADSHEET_PREVIEW_COLUMNS
        if rows:
            info["header_columns"] = rows[0]
            info["preview_rows"] = rows[1:max_rows + 1]
        sheet_infos.append(info)

        preview_blocks.append(f"\nSheet: {sheet_name} ({worksheet.max_row} rows x {worksheet.max_column} columns)")
        if rows:
            preview_blocks.extend(",".join(row) for row in rows[:max_rows + 1])
        else:
            preview_blocks.append("(no non-empty rows found)")

    return {
        "family": "xlsx",
        "preview_text": "\n".join(preview_blocks),
        "sheet_names": sheet_names,
        "sheets": sheet_infos,
    }


# ---------------------------------------------------------------------------
# Tool 1: peek_file
# ---------------------------------------------------------------------------

@tool
def peek_file(
    dataset_id: str | None = None,
    file_path: str | None = None,
    max_rows: int = 20,
    s3_uri: str | None = None,
) -> Dict[str, Any]:
    """
    Inspect a SINGLE file via a budget range-GET. Returns the content family
    (csv/json/xml/text), column headers or XML tags/schema hints, and a
    preview — no full download. When available, also returns a compact
    `profile` field from the benchmark table_profiles.jsonl artifact.

    USE THIS for one file at a time. For multiple files in one call, use
    `peek_multiple` instead (different signature: takes a `files` list).

    Prefer `s3_uri` when list_files/search/preloaded results gave you one; it
    is less error-prone than reconstructing dataset_id + file_path.

    Args:
        dataset_id: ONE dataset identifier as a bare string, e.g. "Barack_Obama"
        file_path:  ONE relative path within the dataset, e.g. "files/data.txt"
        s3_uri:     Optional full object URI instead of dataset_id/file_path
        max_rows:   Maximum preview rows to include (default 20)

    Example call:
        peek_file(dataset_id="index-crimes-by-county", file_path="files/rows.txt")

    Returns:
        Dict with keys: family, preview_text, header_columns, row_count_estimate,
        size_bytes, dataset_id, file_path. XML previews may also include
        xml_root_tag, xml_namespaces, xml_schema_fields,
        xml_record_tag_candidates, xml_preview_mode.
        May also include `profile` with selected cached metadata. Strict
        queryable profiles include columns as name/type pairs, row_count,
        top_2_rows, and llm_description. Strict XML/KML profiles may also
        include record_tag. Single-column CSV/TXT profiles include
        column_count, snippet, and llm_description. Profiles retried from a
        safe prefix expose the same columns/top_2_rows shape. Profiles with
        parser errors, metadata, archives, or unavailable schemas are omitted.
        On error: {error: ...}
    """
    ref = _resolve_file_reference(dataset_id=dataset_id, file_path=file_path, s3_uri=s3_uri)
    if "error" in ref:
        return {"error": ref["error"]}

    dataset_id = _strip_folder_prefix(ref["dataset_id"])
    file_path = ref["file_path"]
    s3_uri = ref["s3_uri"]
    s3 = _get_s3_client()
    key = ref["key"]

    try:
        size_bytes = _s3_head(s3, key)
    except Exception as e:
        return {"error": f"HeadObject failed: {e}"}

    if _is_excel_path(file_path):
        try:
            raw = _s3_read_bounded(s3, key, size_bytes, _MAX_SPREADSHEET_PEEK_BYTES)
        except Exception as e:
            return {"error": f"Excel preview failed: {e}"}

        result: Dict[str, Any] = {
            "dataset_id": dataset_id,
            "file_path": file_path,
            "s3_uri": s3_uri,
            "size_bytes": size_bytes,
        }
        result.update(_build_excel_preview(raw, max_rows=max_rows))
        try:
            profile = load_dataset_profile(s3_uri)
        except Exception:
            profile = None
        profile = select_dataset_profile_fields(profile)
        if profile is not None:
            result["profile"] = profile
        return result

    try:
        text, sampled_bytes = _s3_peek_text(s3, key, size_bytes)
    except Exception as e:
        return {"error": f"Range-GET failed: {e}"}

    family = detect_family(text)

    lines = [ln for ln in text.splitlines() if ln.strip()]
    preview_lines = lines[: max_rows + 1]  # +1 to include header for csv
    preview_text = "\n".join(preview_lines[:max_rows])

    result: Dict[str, Any] = {
        "dataset_id": dataset_id,
        "file_path": file_path,
        "s3_uri": s3_uri,
        "size_bytes": size_bytes,
        "family": family,
        "preview_text": preview_text,
    }

    # Estimate row count from sampled bytes
    if size_bytes > 0 and sampled_bytes > 0:
        lines_in_sample = len(lines)
        bytes_per_line = sampled_bytes / max(lines_in_sample, 1)
        result["row_count_estimate"] = int(size_bytes / bytes_per_line) if bytes_per_line > 0 else None
    else:
        result["row_count_estimate"] = len(lines)

    # CSV: extract header columns
    if family == "csv" and lines:
        header = lines[0]
        for delim in (",", "\t", "|", ";"):
            if delim in header:
                result["header_columns"] = [c.strip() for c in header.split(delim)]
                break

    # JSON: extract keys from first object
    if family == "json":
        try:
            first_obj = json.loads(lines[0]) if lines else None
            if isinstance(first_obj, dict):
                result["json_keys"] = sorted(first_obj.keys())
            elif isinstance(first_obj, list) and first_obj and isinstance(first_obj[0], dict):
                result["json_keys"] = sorted(first_obj[0].keys())
        except Exception:
            pass

    if family == "xml":
        result.update(_build_xml_preview(text, size_bytes))

    try:
        profile = load_dataset_profile(s3_uri)
    except Exception:
        profile = None
    profile = select_dataset_profile_fields(profile)
    if profile is not None:
        result["profile"] = profile

    return result


# ---------------------------------------------------------------------------
# Tool 1b: peek_multiple (batch wrapper around peek_file)
# ---------------------------------------------------------------------------

@tool
def peek_multiple(
    files: Optional[List[Dict[str, str]]] = None,
    entries: Optional[List[Dict[str, str]]] = None,
    max_rows: int = 20,
) -> Dict[str, Any]:
    """
    Inspect SEVERAL files in ONE call — a batch wrapper around peek_file.
    Results may include compact `profile` fields when cached profiles are available.

    USE THIS when you already know which 2+ files you need
    (e.g. immediately after `list_files` returned several relevant paths).
    For a single file, use `peek_file` instead — its signature is simpler.

    Prefer per-entry `s3_uri` when list_files/search/preloaded results gave
    you one.

    REQUIRED ARGUMENT SHAPE: You MUST pass a `files` list of dicts, NOT
    `dataset_id`/`file_path` directly:

        CORRECT:
            peek_multiple(files=[
                {"dataset_id": "census-2021", "file_path": "files/rows.txt"},
            ])
        WRONG (these will all error):
            peek_multiple(max_rows=5)                             # missing files

    Args:
        files:    NON-EMPTY list of dicts. Each dict needs 'dataset_id' and
                  'file_path'. The key 'path' is also accepted as an alias for
                  'file_path' so raw list_files output can be passed directly.
                  A per-entry `s3_uri` is also accepted.
        entries:  Alias for `files`. Accepted to be forgiving when the agent
                  uses the older/wrong wrapper key.
        max_rows: Maximum preview rows per file (default 20).

    Returns:
        Dict with 'results' list (one entry per file, same shape as peek_file)
        and 'count'. Each result may include `profile` with selected cached
        metadata for queryable files: columns as name/type pairs, row_count,
        top_2_rows, llm_description, and, for XML/KML, record_tag. Sampled
        retry profiles expose the same columns/top_2_rows shape. Profiles with
        parser errors, metadata, archives, or unavailable schemas are omitted.
    """
    if files is None and entries is not None:
        files = entries
    if isinstance(files, dict):
        files = [files]
    if not isinstance(files, list) or not files:
        return {
            "error": (
                "peek_multiple requires a non-empty `files` list of "
                "{dataset_id, file_path} dicts. Use peek_multiple for 2+ files "
                "after list_files, or peek_file(dataset_id, file_path) for one "
                "file. Example: "
                'peek_multiple(files=[{"dataset_id": "census", "file_path": "files/rows.txt"}], max_rows=5)'
            )
        }

    results = []
    for spec in files:
        if isinstance(spec, str):
            results.append(peek_file(s3_uri=spec, max_rows=max_rows))
            continue
        if not isinstance(spec, dict):
            results.append({"error": "each entry must be a dict with dataset_id/file_path or s3_uri"})
            continue
        ds = spec.get("dataset_id", "")
        fp = spec.get("file_path") or spec.get("path") or ""
        uri = spec.get("s3_uri") or spec.get("uri") or ""
        results.append(peek_file(dataset_id=ds, file_path=fp, max_rows=max_rows, s3_uri=uri))

    return {"results": results, "count": len(results)}


# ---------------------------------------------------------------------------
# Tool 2: read_file
# ---------------------------------------------------------------------------


@tool
def read_file(
    dataset_id: str | None = None,
    file_path: str | None = None,
    start_line: int = 0,
    max_lines: int = 10000,
    s3_uri: str | None = None,
) -> Dict[str, Any]:
    """
    Read lines from a file in the data lake without downloading it.
    Use this to read text, CSV, or JSON files directly. Supports pagination
    via start_line for large files.

    Prefer `s3_uri` when list_files/search/preloaded results gave you one; it
    is less error-prone than reconstructing dataset_id + file_path. For datagov
    files, both "rows.txt" and "files/rows.txt" are accepted, but
    "files/rows.txt" is the canonical path.

    Args:
        dataset_id: Dataset identifier (e.g. "census")
        file_path:  Relative path within the dataset (e.g. "files/data.txt")
        start_line: Line index to start reading from, 0-based (default 0)
        max_lines:  Number of lines to return, capped at 10000 (default 10000)
        s3_uri:     Optional full object URI instead of dataset_id/file_path
    """
    if start_line < 0:
        return {"error": "start_line must be >= 0"}

    max_lines = min(max_lines, 10_000)
    ref = _resolve_file_reference(dataset_id=dataset_id, file_path=file_path, s3_uri=s3_uri)
    if "error" in ref:
        return {"error": ref["error"]}

    dataset_id = _strip_folder_prefix(ref["dataset_id"])
    file_path = ref["file_path"]
    s3_uri = ref["s3_uri"]
    s3 = _get_s3_client()
    key = ref["key"]

    try:
        resp = s3.get_object(Bucket=BUCKET, Key=key)
        body = resp["Body"]
        lines = []
        current = 0
        for raw_line in body.iter_lines():
            if current < start_line:
                current += 1
                continue
            lines.append(raw_line.decode("utf-8", errors="replace"))
            current += 1
            if len(lines) >= max_lines:
                break
        body.close()
    except Exception as e:
        return {"error": f"Failed to read file: {e}"}

    result = {
        "dataset_id": dataset_id,
        "file_path": file_path,
        "s3_uri": s3_uri,
        "start_line": start_line,
        "returned_lines": len(lines),
        "lines": lines,
    }
    result_json = json.dumps(result)
    if len(result_json) > _TOOL_RESULT_CHAR_CAP:
        # Write full content to sandbox, return truncated lines + pointer
        sandbox = _get_sandbox_dir()
        safe_name = file_path.lstrip("/").replace("/", "_")
        dump_path = sandbox / dataset_id / f"{safe_name}.read_result.json"
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        dump_path.write_text(result_json)
        budget = _TOOL_RESULT_CHAR_CAP - 600
        char_count, truncated_lines = 0, []
        for line in lines:
            if char_count + len(line) + 4 > budget:
                break
            truncated_lines.append(line)
            char_count += len(line) + 4
        return {
            "dataset_id": dataset_id,
            "file_path": file_path,
            "s3_uri": s3_uri,
            "start_line": start_line,
            "returned_lines": len(truncated_lines),
            "truncated": True,
            "local_result_path": str(dump_path),
            "truncation_note": (
                f"Output truncated to {len(truncated_lines)} of {len(lines)} fetched lines. "
                f"Full content written to: {dump_path} (use execute_code to read it). "
                f"Or use start_line={start_line + len(truncated_lines)} to page forward, "
                "or grep_file for specific values."
            ),
            "lines": truncated_lines,
        }
    return result


# ---------------------------------------------------------------------------
# Tool 3: grep_file
# ---------------------------------------------------------------------------

@tool
def grep_file(
    dataset_id: str | None = None,
    file_path: str | None = None,
    regex_pattern: str = "",
    context_lines: int = 2,
    s3_uri: str | None = None,
) -> Dict[str, Any]:
    """
    Search for a regex pattern inside a file without downloading it.
    Streams the file from S3 and returns matching lines with surrounding context.
    Use this to locate specific values, IDs, or keywords in large files.

    Prefer `s3_uri` when list_files/search/preloaded results gave you one; it
    is less error-prone than reconstructing dataset_id + file_path. For datagov
    files, both "rows.txt" and "files/rows.txt" are accepted, but
    "files/rows.txt" is the canonical path.

    Args:
        dataset_id:    Dataset identifier (e.g. "public-school-locations-current-23297")
        file_path:     Relative path within the dataset (e.g. "files/data.txt")
        regex_pattern: Case-insensitive regex to search for (e.g. "King County")
        context_lines: Lines of context before/after each match (default 2, max 5)
        s3_uri:        Optional full object URI instead of dataset_id/file_path

    Returns:
        Dict with keys: match_count, matches (list of {line_number, line,
        context_before, context_after}), truncated. On error: {error: ...}
    """
    if not regex_pattern:
        return {"error": "regex_pattern is required"}

    context_lines = min(context_lines, 5)
    # filename = file_path.rsplit("/", 1)[-1]
    # if should_skip(filename):
    #     return {"error": f"File skipped (metadata/binary): {file_path}"}

    try:
        pattern = re.compile(regex_pattern, re.IGNORECASE)
    except re.error as e:
        return {"error": f"Invalid regex pattern: {e}"}

    ref = _resolve_file_reference(dataset_id=dataset_id, file_path=file_path, s3_uri=s3_uri)
    if "error" in ref:
        return {"error": ref["error"]}

    dataset_id = _strip_folder_prefix(ref["dataset_id"])
    file_path = ref["file_path"]
    s3_uri = ref["s3_uri"]
    s3 = _get_s3_client()
    key = ref["key"]

    try:
        resp = s3.get_object(Bucket=BUCKET, Key=key)
        line_iter = resp["Body"].iter_lines()
        
        matches = []
        pending_matches = []
        # deque automatically pushes old lines out when it hits maxlen!
        history = deque(maxlen=context_lines) 

        for i, raw_line in enumerate(line_iter):
            line = raw_line.decode("utf-8", errors="replace")

            # 1. Fill the "after" context for any matches we found previously
            for m in pending_matches:
                if len(m["context_after"]) < context_lines:
                    m["context_after"].append(line)

            # 2. Move fully-populated matches into our final list
            completed = [m for m in pending_matches if len(m["context_after"]) == context_lines]
            for m in completed:
                matches.append(m)
                pending_matches.remove(m)
                
            if len(matches) >= _SEARCH_MAX_MATCHES:
                break

            # 3. Evaluate the CURRENT line for a new match
            if pattern.search(line):
                new_match = {
                    "line_number": i,
                    "line": line,
                    "context_before": list(history),  # Snapshot the current history
                    "context_after": []
                }
                if context_lines == 0:
                    matches.append(new_match)
                    if len(matches) >= _SEARCH_MAX_MATCHES:
                        break
                else:
                    pending_matches.append(new_match)

            # 4. Add current line to history for future matches
            history.append(line)

        # Catch any pending matches if we hit the end of the file early
        for m in pending_matches:
            if m not in matches:
                matches.append(m)

        resp["Body"].close()

    except Exception as e:
        return {"error": f"Stream search failed: {e}", "traceback": traceback.format_exc()}

    result = {
        "dataset_id": dataset_id,
        "file_path": file_path,
        "s3_uri": s3_uri,
        "match_count": len(matches),
        "truncated_matches": len(matches) >= _SEARCH_MAX_MATCHES,
        "matches": matches,
    }
    result_json = json.dumps(result)
    if len(result_json) > _TOOL_RESULT_CHAR_CAP:
        sandbox = _get_sandbox_dir()
        safe_name = file_path.lstrip("/").replace("/", "_")
        dump_path = sandbox / dataset_id / f"{safe_name}.grep_result.json"
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        dump_path.write_text(result_json)
        budget = _TOOL_RESULT_CHAR_CAP - 600
        char_count, capped_matches = 0, []
        for match in matches:
            match_json = json.dumps(match)
            if char_count + len(match_json) + 2 > budget:
                break
            capped_matches.append(match)
            char_count += len(match_json) + 2
        return {
            "dataset_id": dataset_id,
            "file_path": file_path,
            "s3_uri": s3_uri,
            "match_count": len(matches),
            "returned_matches": len(capped_matches),
            "truncated_matches": True,
            "local_result_path": str(dump_path),
            "truncation_note": (
                f"Match output exceeded the {_TOOL_RESULT_CHAR_CAP} char limit. "
                f"Showing {len(capped_matches)} of {len(matches)} matches. "
                f"Full result written to: {dump_path}. "
                "Tighten the regex or reduce context_lines if you need a smaller in-context result."
            ),
            "matches": capped_matches,
        }
    return result

# ---------------------------------------------------------------------------
# Tool 4: parse_xml_records
# ---------------------------------------------------------------------------

@tool
def parse_xml_records(
    dataset_id: str | None = None,
    file_path: str | None = None,
    record_tag: str | None = None,
    fields: Optional[List[str]] = None,
    filters: Optional[Dict[str, Any]] = None,
    group_by: Optional[List[str]] = None,
    limit: int = 50,
    s3_uri: str | None = None,
) -> Dict[str, Any]:
    """
    Parse XML/KML records directly from S3 without downloading or executing
    arbitrary Python.

    Use this for XML/KML sources after peek_file has shown record tags/schema
    fields. It supports KML SimpleData fields, plain XML leaf-tag fields,
    exact-match filters, and COUNT-style grouping.

    Do not use this for CSV/JSON; use query_file for those.

    Args:
        dataset_id: Dataset identifier (e.g. "public-school-locations-current-23297")
        file_path: Relative path within the dataset (e.g. "files/schools.kml")
        record_tag: XML tag for one logical record (e.g. "Placemark"). If omitted,
                    parse_xml_records uses the first record candidate from peek_file.
        fields: Optional field names to return for each matched record.
        filters: Optional exact-match filters, e.g. {"STFIP": "06"}.
        group_by: Optional field names to group by. When provided, rows contain
                  the group fields plus "count".
        limit: Maximum rows/groups to return, capped at 200.
        s3_uri: Optional full object URI instead of dataset_id/file_path.

    Returns:
        Dict with keys: rows, row_count, scanned_records, matched_records,
        truncated. On error: {error: ...}
    """
    timeout_seconds = _tool_timeout_seconds()
    completed, result = _run_tool_with_timeout(
        _parse_xml_records_impl,
        dataset_id,
        file_path,
        record_tag,
        fields,
        filters,
        group_by,
        limit,
        s3_uri,
        timeout_seconds=timeout_seconds,
    )
    if completed:
        return result or {"error": "parse_xml_records failed without returning a result"}

    return {
        "error": (
            f"XML/KML parsing timed out after {timeout_seconds}s. "
            "Use peek_file to confirm the record_tag and filters, then retry "
            "with a narrower filter or lower limit."
        )
    }


def _parse_xml_records_impl(
    dataset_id: str | None = None,
    file_path: str | None = None,
    record_tag: str | None = None,
    fields: Optional[List[str]] = None,
    filters: Optional[Dict[str, Any]] = None,
    group_by: Optional[List[str]] = None,
    limit: int = 50,
    s3_uri: str | None = None,
) -> Dict[str, Any]:
    fields_list = _coerce_string_list(fields)
    group_fields = _coerce_string_list(group_by)
    filters = filters if isinstance(filters, dict) else {}
    try:
        limit = max(1, min(int(limit), _QUERY_ROW_CAP))
    except (TypeError, ValueError):
        limit = 50

    ref = _resolve_file_reference(dataset_id=dataset_id, file_path=file_path, s3_uri=s3_uri)
    if "error" in ref:
        return {"error": ref["error"]}

    dataset_id = _strip_folder_prefix(ref["dataset_id"])
    file_path = ref["file_path"]
    s3_uri = ref["s3_uri"]
    key = ref["key"]
    s3 = _get_s3_client()

    try:
        size = _s3_head(s3, key)
        text, _sampled_bytes = _s3_peek_text(s3, key, size)
        family = detect_family(text)
    except Exception as e:
        return {"error": f"Could not inspect XML/KML file: {e}"}

    if family != "xml":
        return {
            "error": (
                f"parse_xml_records only supports XML/KML files. Detected {family!r}. "
                "Use query_file for CSV/JSON or read_file/grep_file for text."
            )
        }

    preview = _build_xml_preview(text, size)
    chosen_record_tag = _normalize_xml_record_tag(record_tag)
    if not chosen_record_tag:
        candidates = preview.get("xml_record_tag_candidates") or []
        chosen_record_tag = _normalize_xml_record_tag(candidates[0]) if candidates else None
    if not chosen_record_tag:
        return {
            "error": (
                "Could not infer an XML record_tag. Call peek_file first and pass one "
                "of xml_record_tag_candidates to parse_xml_records."
            ),
            **preview,
        }

    rows: List[Dict[str, Any]] = []
    groups: Counter = Counter()
    scanned_records = 0
    matched_records = 0
    body = None

    try:
        resp = s3.get_object(Bucket=BUCKET, Key=key)
        body = resp["Body"]
        for _event, elem in ET.iterparse(body, events=("end",)):
            if _local_xml_name(elem.tag) != chosen_record_tag:
                continue

            scanned_records += 1
            row = _xml_record_to_row(elem)
            if filters and not _xml_row_matches_filters(row, filters):
                elem.clear()
                continue

            matched_records += 1
            if group_fields:
                group_key = tuple(row.get(field, "") for field in group_fields)
                groups[group_key] += 1
            elif len(rows) < limit:
                rows.append(_select_xml_row_fields(row, fields_list))

            elem.clear()
    except ET.ParseError as e:
        return {"error": f"XML/KML parse failed: {e}", "traceback": traceback.format_exc()}
    except Exception as e:
        return {"error": f"XML/KML stream parse failed: {e}", "traceback": traceback.format_exc()}
    finally:
        if body is not None and hasattr(body, "close"):
            body.close()

    if group_fields:
        rows = []
        for group_key, count in sorted(groups.items(), key=lambda item: (-item[1], item[0]))[:limit]:
            group_row = {field: group_key[i] for i, field in enumerate(group_fields)}
            group_row["count"] = count
            rows.append(group_row)
        truncated = len(groups) > limit
    else:
        truncated = matched_records > len(rows)

    return {
        "dataset_id": dataset_id,
        "file_path": file_path,
        "s3_uri": s3_uri,
        "family": family,
        "record_tag": chosen_record_tag,
        "fields": fields_list,
        "filters": filters,
        "group_by": group_fields,
        "scanned_records": scanned_records,
        "matched_records": matched_records,
        "rows": rows,
        "row_count": len(rows),
        "truncated": truncated,
    }


# ---------------------------------------------------------------------------
# Tool 5: query_file
# ---------------------------------------------------------------------------

@tool
def query_file(
    dataset_id: str | None = None,
    file_path: str | None = None,
    sql: str = "",
    s3_uri: str | None = None,
) -> Dict[str, Any]:
    """
    Run a SQL query directly against an S3 file using DuckDB httpfs.
    No download required. The file is referenced as table alias 't'.

    Supported file types: CSV (.csv), JSON (.json, .jsonl, .ndjson).
    Do not use query_file for non-tabular/non-JSON sources such as
    Wikipedia/content.txt, prose/plain text, XML/KML, HTML, PDFs, or binary files.
    XML/KML is detected but not queryable here; query_file returns a hint
    to use parse_xml_records, peek_file, grep_file, or read_file instead.
    Results are capped at 200 rows.

    Prefer `s3_uri` when list_files/search/preloaded results gave you one; it
    is less error-prone than reconstructing dataset_id + file_path. For datagov
    files, both "rows.txt" and "files/rows.txt" are accepted, but
    "files/rows.txt" is the canonical path.

    Args:
        dataset_id: Dataset identifier (e.g. "Barack_Obama")
        file_path:  Relative path within the dataset (e.g. "table_0.csv")
        sql:        SQL query. Use 't' as the table alias.
                    Example: "SELECT * FROM t LIMIT 10"
                    Example: "SELECT col1, COUNT(*) FROM t GROUP BY col1"
        s3_uri:     Optional full object URI instead of dataset_id/file_path

    Returns:
        Dict with keys: columns, rows, row_count, truncated. On error: {error: ...}
    """
    timeout_seconds = _tool_timeout_seconds()
    completed, result = _run_tool_with_timeout(
        _query_file_impl,
        dataset_id,
        file_path,
        sql,
        s3_uri,
        timeout_seconds=timeout_seconds,
    )
    if completed:
        return result or {"error": "query_file failed without returning a result"}

    return {
        "error": (
            f"Query timed out after {timeout_seconds}s. "
            "Narrow the SQL (SELECT fewer columns, add WHERE/GROUP BY/LIMIT), "
            "or use download + execute_code for long-running tabular/JSON work."
        )
    }


def _query_file_impl(
    dataset_id: str | None = None,
    file_path: str | None = None,
    sql: str = "",
    s3_uri: str | None = None,
) -> Dict[str, Any]:
    if not sql or not sql.strip():
        return {"error": "sql is required"}

    # Auto-fix MySQL-style backtick identifiers — DuckDB only accepts double
    # quotes. Preserves backticks inside string literals. Eliminates ~17
    # Parser Errors per eval at the source.
    sql = _normalize_sql_backticks(sql)

    ref = _resolve_file_reference(dataset_id=dataset_id, file_path=file_path, s3_uri=s3_uri)
    if "error" in ref:
        return {"error": ref["error"]}

    dataset_id = _strip_folder_prefix(ref["dataset_id"])
    file_path = ref["file_path"]
    s3_uri = ref["s3_uri"]

    # Determine reader function — peek first for all extensions
    try:
        s3 = _get_s3_client()
        key = ref["key"]
        size = _s3_head(s3, key)
        text, _sampled_bytes = _s3_peek_text(s3, key, size)
        family = detect_family(text)
    except Exception as e:
        return {"error": f"Could not detect file family: {e}"}

    if size > _QUERY_MAX_FILE_BYTES:
        if family not in {"csv", "json"}:
            return {"error": _rewrite_unqueryable_family_error(family)}
        return {
            "error": (
                f"File too large to query directly ({size // (1024*1024)} MB). "
                "Use download + execute_code only if the source is tabular or JSON."
            )
        }

    if family == "csv":
        reader = f"read_csv_auto('{s3_uri}', quote='\"')"
    elif family == "json":
        reader = f"read_json_auto('{s3_uri}', maximum_object_size={_QUERY_MAX_FILE_BYTES})"
    else:
        return {"error": _rewrite_unqueryable_family_error(family)}

    try:
        conn = _duckdb_connection()
        # Create view aliased as 't'
        conn.execute(f"CREATE VIEW t AS SELECT * FROM {reader}")
        # Execute user SQL, cap rows
        rel = conn.execute(sql)
        rows = rel.fetchmany(_QUERY_ROW_CAP + 1)
        columns = [desc[0] for desc in rel.description]
        truncated = len(rows) > _QUERY_ROW_CAP
        if truncated:
            rows = rows[:_QUERY_ROW_CAP]
        # Convert rows to JSON-safe primitives up front so DATE / TIMESTAMP /
        # DECIMAL / UUID / BLOB columns don't crash json.dumps below.
        safe_rows = [[_to_json_safe(v) for v in r] for r in rows]
        result = {
            "dataset_id": dataset_id,
            "file_path": file_path,
            "s3_uri": s3_uri,
            "columns": columns,
            "rows": safe_rows,
            "row_count": len(safe_rows),
            "truncated": truncated,
        }
        result_json = json.dumps(result)
        if len(result_json) > _TOOL_RESULT_CHAR_CAP:
            # Write full result to sandbox so agent can read it via execute_code
            sandbox = _get_sandbox_dir()
            safe_name = file_path.lstrip("/").replace("/", "_")
            dump_path = sandbox / dataset_id / f"{safe_name}.query_result.json"
            dump_path.parent.mkdir(parents=True, exist_ok=True)
            dump_path.write_text(result_json)
            # Pack as many rows as fit within the budget
            budget = _TOOL_RESULT_CHAR_CAP - 600  # headroom for metadata fields
            char_count, capped_rows = 0, []
            for row in safe_rows:
                row_json = json.dumps(row)
                if char_count + len(row_json) + 2 > budget:
                    break
                capped_rows.append(row)
                char_count += len(row_json) + 2
            avg_row_bytes = len(result_json) // max(len(safe_rows), 1)
            return {
                "truncated": True,
                "truncation_note": (
                    f"Result too large for context ({size // 1024}KB source file, "
                    f"~{int(size / max(1, avg_row_bytes))} rows estimated). "
                    f"Showing {len(capped_rows)} of {len(safe_rows)} rows within the {_TOOL_RESULT_CHAR_CAP} char limit. "
                    f"Full result written to: {dump_path} (use execute_code to read it). "
                    "Prefer: query_file with SELECT specific columns + WHERE/GROUP BY/LIMIT "
                    "for aggregates, or grep_file for searching specific values."
                ),
                "local_result_path": str(dump_path),
                "dataset_id": dataset_id,
                "file_path": file_path,
                "s3_uri": s3_uri,
                "size_bytes": size,
                "columns": columns,
                "rows": capped_rows,
                "row_count_shown": len(capped_rows),
                "row_count_estimate": int(size / max(1, avg_row_bytes)),
            }
        return result
    except Exception as e:
        return {
            "error": f"Query failed: {_rewrite_query_error(str(e))}",
            "traceback": traceback.format_exc(),
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # v2-new
    "peek_file",
    "peek_multiple",
    "read_file",
    "grep_file",
    "parse_xml_records",
    "query_file",
    "configure_benchmark",
    # re-exported from agent_tools
    "search",
    "search_keyword",
    "list_files",
    "execute_code",
    "get_sandbox_info",
    "cleanup_sandbox",
    "set_sandbox_dir",
    "download",
]
