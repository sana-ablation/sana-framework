#!/usr/bin/env python3
"""Build table_profiles.jsonl from canonical description/profile inputs."""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import decimal
import json
import math
import os
import re
import sys
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, TypeVar
import xml.etree.ElementTree as ET

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import duckdb
from dotenv import load_dotenv
try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable=None, **_kwargs):
        return iterable if iterable is not None else _NullTqdm()

from sana_evaluation.tools.agent_tools import BUCKET, REGION, _build_s3_client
from sana_evaluation.tools.agent_tools_v2 import (
    _build_xml_preview,
    _local_xml_name,
    _normalize_xml_record_tag,
    _xml_record_to_row,
)
from sana_evaluation.tools.external.description_rows import reject_forbidden_description_row
from sana_evaluation.tools.helper.detect import detect_family

load_dotenv()


class _NullTqdm:
    def update(self, _n: int = 1) -> None:
        return None

    def set_postfix(self, *args, **kwargs) -> None:
        return None

    def close(self) -> None:
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class _UnusableSchemaError(ValueError):
    """Raised when DuckDB can parse bytes but the inferred schema is not useful."""


_ARTIFACT_ROOT = Path("benchmarks/lakeqa/tasks-mini/artifacts")
_DESC_PATH = _ARTIFACT_ROOT / "descriptions.jsonl"
_URI_LIST_PATH = Path("table_profiles_needed.txt")
_SNIPPET_PATH = _ARTIFACT_ROOT / "snippets.jsonl"
_SNIFF_BYTES = 8 * 1024
_SNIPPET_FALLBACK_BYTES = 2 * 1024
_MAX_CELL_CHARS = 80
_QUERY_MAX_FILE_BYTES = 500 * 1024 * 1024
_DEFAULT_MAX_PROFILE_FILE_BYTES = 5 * 1024 * 1024 * 1024
_DEFAULT_DUCKDB_TIMEOUT_SECONDS = 300
_MAX_COLUMNS = 200
_MAX_XML_SCHEMA_RECORDS = 200
_T = TypeVar("_T")

_SOCRATA_ID_RE = re.compile(r"^[a-z0-9]{4}-[a-z0-9]{4}$")
_GENERIC_COLUMN_RE = re.compile(r"^column\d+$", re.IGNORECASE)
_PROFILE_METADATA_STEMS = {
    "metadata",
    "dcat-us",
    "catalog",
    "signed-metadata",
    "headers",
    "gmi",
    "open-licenses",
    "legalcode",
    "government-works",
    "index",
    "odc-odbl",
    "wmsserver",
    "resolve",
    "request",
    "edit",
    "search",
    "contact",
    "policyinformation",
    "gmxcodelists",
    "bios",
    "hires",
    "cwhr",
    "license",
    "readme",
    "cc-zero",
    "cc-by",
}

_TABULAR_KIND_TO_FAMILY = {
    "delimited_text": "csv",
    "csv": "csv",
    "tsv": "csv",
    "parquet": "parquet",
    "json": "json",
    "geojson": "json",
}


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(_URI_LIST_PATH))
    parser.add_argument(
        "--input-kind",
        choices=("auto", "schemas", "manifest", "uri-list", "descriptions"),
        default="auto",
    )
    parser.add_argument("--output", default=str(_ARTIFACT_ROOT / "table_profiles.jsonl"))
    parser.add_argument("--descriptions", default=str(_DESC_PATH))
    parser.add_argument("--snippets", default=str(_SNIPPET_PATH))
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument(
        "--max-file-bytes",
        type=int,
        default=_DEFAULT_MAX_PROFILE_FILE_BYTES,
        help="Maximum object size to profile; default is 5 GiB. Use 0 to disable.",
    )
    parser.add_argument(
        "--duckdb-timeout-seconds",
        type=int,
        default=_DEFAULT_DUCKDB_TIMEOUT_SECONDS,
        help="Seconds before interrupting one DuckDB tabular profiling query group. Use 0 to disable.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--bucket", default=BUCKET)
    return parser.parse_args(argv)


def _jsonl_rows(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


def _peek_first_row(path: Path) -> Optional[Dict[str, Any]]:
    for obj in _jsonl_rows(path):
        return obj
    return None


def _load_description_cache(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not path.exists():
        return out
    for obj in _jsonl_rows(path):
        reject_forbidden_description_row(obj, path=path)
        uri = str(obj.get("dataset_uri") or obj.get("uri") or "").strip()
        desc = str(obj.get("description") or "").strip()
        if uri and desc and uri not in out:
            out[uri] = desc
    return out


def _load_snippet_cache(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not path.exists():
        return out
    for obj in _jsonl_rows(path):
        uri = str(obj.get("dataset_uri") or obj.get("uri") or "").strip()
        snippet = str(obj.get("dataset_snippet") or obj.get("snippet") or obj.get("content") or "").strip()
        if uri and snippet and uri not in out:
            out[uri] = snippet
    return out


def _profile_identity(row: Dict[str, Any]) -> Optional[str]:
    s3_uri = str(row.get("s3_uri") or "").strip()
    if s3_uri:
        return f"uri:{s3_uri}"
    slug = str(row.get("slug") or "").strip()
    filename = str(row.get("filename") or "").strip()
    if slug and filename:
        return f"slug:{slug}::{filename}"
    return None


def _load_existing_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    keys: set[str] = set()
    for obj in _jsonl_rows(path):
        identity = _profile_identity(obj)
        if identity:
            keys.add(identity)
    return keys


def _sql_quote_string(value: str) -> str:
    return "'" + str(value).replace("\\", "\\\\").replace("'", "''") + "'"


def _sql_quote_ident(value: str) -> str:
    return '"' + str(value).replace('"', '""') + '"'


def _is_s3_ref(source_ref: str) -> bool:
    return str(source_ref).startswith("s3://")


def _parse_s3_uri(uri: str) -> Tuple[str, str]:
    raw = str(uri)
    if not raw.startswith("s3://"):
        raise ValueError(f"Expected s3:// URI, got: {uri}")
    bucket_and_key = raw[5:]
    if "/" not in bucket_and_key:
        raise ValueError(f"Malformed s3:// URI, missing key: {uri}")
    bucket, key = bucket_and_key.split("/", 1)
    return bucket, key


def _build_s3_client_for_runtime():
    requested = (os.getenv("S3_ACCESS_MODE", "auto") or "auto").strip().lower()
    if requested in {"unsigned", "public", "anonymous", "anon", "no-sign-request"}:
        return _build_s3_client(unsigned=True)
    try:
        return _build_s3_client(unsigned=False)
    except Exception:
        return _build_s3_client(unsigned=True)


def _head_size_bytes(source_ref: str) -> int:
    if _is_s3_ref(source_ref):
        bucket, key = _parse_s3_uri(source_ref)
        s3 = _build_s3_client_for_runtime()
        meta = s3.head_object(Bucket=bucket, Key=key)
        return int(meta.get("ContentLength") or 0)
    return int(Path(source_ref).stat().st_size)


def _read_prefix_bytes(source_ref: str, num_bytes: int) -> bytes:
    if _is_s3_ref(source_ref):
        bucket, key = _parse_s3_uri(source_ref)
        s3 = _build_s3_client_for_runtime()
        end = max(num_bytes - 1, 0)
        resp = s3.get_object(Bucket=bucket, Key=key, Range=f"bytes=0-{end}")
        return resp["Body"].read()
    with Path(source_ref).open("rb") as f:
        return f.read(num_bytes)


def _looks_binary_prefix(raw: bytes) -> bool:
    if raw.startswith((b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")):
        return True
    if b"\x00" in raw[:512]:
        return True
    if not raw:
        return False
    controls = sum(1 for b in raw[:512] if b < 32 and b not in {9, 10, 13})
    return controls / min(len(raw), 512) > 0.20


def _looks_archive_prefix(raw: bytes) -> bool:
    return raw.startswith((b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08"))


def _metadata_stem(path_like: str) -> str:
    name = str(path_like or "").split("?", 1)[0].rsplit("/", 1)[-1].strip().lower()
    return name.rsplit(".", 1)[0] if "." in name else name


def _is_metadata_profile_path(path_like: str) -> bool:
    stem = _metadata_stem(path_like)
    return stem in _PROFILE_METADATA_STEMS or bool(_SOCRATA_ID_RE.match(stem))


def _expected_family(table_kind: str) -> Optional[str]:
    return _TABULAR_KIND_TO_FAMILY.get((table_kind or "").strip().lower())


def _resolve_family(table: Dict[str, Any], source_ref: str) -> str:
    expected = _expected_family(str(table.get("table_kind") or ""))
    if expected == "parquet":
        return "parquet"

    try:
        raw = _read_prefix_bytes(source_ref, _SNIFF_BYTES)
    except Exception:
        return expected or "text"

    if _looks_archive_prefix(raw):
        return "archive"
    if _looks_binary_prefix(raw):
        return "binary"
    sniff_text = raw.decode("utf-8", errors="replace")
    sniffed = detect_family(sniff_text)
    if expected and sniffed == "text":
        return expected
    return expected or sniffed


def _source_ref_for_table(table: Dict[str, Any], bucket: str) -> str:
    direct_uri = str(table.get("source_uri") or table.get("s3_uri") or "").strip()
    if direct_uri:
        return direct_uri

    local_path = str(table.get("local_path") or "").strip()
    if local_path:
        return local_path

    s3_key = str(table.get("s3_key") or "").strip()
    if not s3_key:
        raise ValueError("Schema table is missing both source_uri/local_path and s3_key")
    return f"s3://{bucket}/{s3_key}"


def _filename_stem(table: Dict[str, Any], source_ref: str) -> str:
    rel = str(table.get("relative_path") or "").strip()
    name = rel.rsplit("/", 1)[-1] if rel else Path(source_ref).name
    return name.rsplit(".", 1)[0] if "." in name else name


def _stem_from_path(path_like: str) -> str:
    name = str(path_like).rsplit("/", 1)[-1]
    return name.rsplit(".", 1)[0] if "." in name else name


def _table_kind_from_path(path_like: str) -> Optional[str]:
    suffix = Path(str(path_like).split("?", 1)[0]).suffix.lower().lstrip(".")
    if suffix in {"csv", "tsv", "json", "geojson", "parquet"}:
        return suffix
    if suffix in {"txt", "text"}:
        return "delimited_text"
    return None


def _dataset_and_file_from_source(source_ref: str) -> Tuple[Optional[str], Optional[str]]:
    raw = str(source_ref or "").strip()
    if raw.startswith("s3://"):
        _, key = _parse_s3_uri(raw)
        parts = key.split("/")
        if len(parts) >= 2:
            dataset_id = parts[1] if parts[0] == "datagov" else parts[0]
            if parts[0] == "datagov":
                file_path = "/".join(parts[2:]) if len(parts) > 2 else None
            else:
                file_path = "/".join(parts[1:]) if len(parts) > 1 else None
            return dataset_id or None, file_path or None
        return None, key or None

    path = Path(raw)
    return path.parent.name or None, path.name or None


def _iter_schema_entries(input_path: Path, bucket: str) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for obj in _jsonl_rows(input_path):
        slug = str(obj.get("dataset_slug") or "").strip()
        if not slug:
            continue
        tables = obj.get("tables") or []
        if not isinstance(tables, list):
            continue
        for table in tables:
            if not isinstance(table, dict):
                continue
            source_ref = _source_ref_for_table(table, bucket)
            entries.append(
                {
                    "slug": slug,
                    "filename": _filename_stem(table, source_ref),
                    "source_ref": source_ref,
                    "s3_uri": source_ref if _is_s3_ref(source_ref) else None,
                    "table": table,
                }
            )
    return entries


def _iter_manifest_entries(input_path: Path) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for obj in _jsonl_rows(input_path):
        source_ref = str(
            obj.get("source_ref")
            or obj.get("local_path")
            or obj.get("s3_uri")
            or obj.get("dataset_uri")
            or obj.get("uri")
            or ""
        ).strip()
        if not source_ref:
            continue
        inferred_dataset_id, inferred_file_path = _dataset_and_file_from_source(source_ref)
        dataset_id = str(obj.get("dataset_id") or inferred_dataset_id or "").strip()
        file_path = str(obj.get("file_path") or obj.get("path") or inferred_file_path or "").strip()
        slug = dataset_id or _stem_from_path(source_ref)
        filename = _stem_from_path(file_path or source_ref)
        table = {
            "delimiter": obj.get("delimiter"),
            "table_kind": obj.get("table_kind") or _table_kind_from_path(file_path or source_ref),
        }
        entry = {
            "slug": slug,
            "filename": filename,
            "source_ref": source_ref,
            "s3_uri": str(obj.get("s3_uri") or "").strip() or None,
            "dataset_id": dataset_id or None,
            "file_path": file_path or None,
            "size_bytes": obj.get("size_bytes") or obj.get("size"),
            "table": table,
        }
        entries.append(entry)
    return entries


def _iter_uri_list_entries(input_path: Path) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    if not input_path.exists():
        return entries
    with input_path.open() as f:
        for line in f:
            source_ref = line.strip()
            if not source_ref:
                continue
            dataset_id, file_path = _dataset_and_file_from_source(source_ref)
            slug = dataset_id or _stem_from_path(source_ref)
            filename = _stem_from_path(file_path or source_ref)
            entries.append(
                {
                    "slug": slug,
                    "filename": filename,
                    "source_ref": source_ref,
                    "s3_uri": source_ref if _is_s3_ref(source_ref) else None,
                    "dataset_id": dataset_id,
                    "file_path": file_path,
                    "size_bytes": None,
                    "table": {
                        "delimiter": None,
                        "table_kind": _table_kind_from_path(file_path or source_ref),
                    },
                }
            )
    return entries


def _detect_input_kind(input_path: Path) -> str:
    if input_path.suffix.lower() == ".txt":
        return "uri-list"
    first_row = _peek_first_row(input_path)
    if first_row is None:
        return "manifest"
    if isinstance(first_row.get("tables"), list):
        return "schemas"
    if first_row.get("dataset_uri") and first_row.get("description") is not None:
        return "descriptions"
    return "manifest"


def _iter_input_entries(input_path: Path, *, input_kind: str, bucket: str) -> List[Dict[str, Any]]:
    resolved = _detect_input_kind(input_path) if input_kind == "auto" else input_kind
    if resolved == "schemas":
        return _iter_schema_entries(input_path, bucket)
    if resolved == "uri-list":
        return _iter_uri_list_entries(input_path)
    if resolved == "descriptions":
        return _iter_manifest_entries(input_path)
    return _iter_manifest_entries(input_path)


def _duckdb_connection(*, needs_httpfs: bool) -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(":memory:")
    if needs_httpfs:
        conn.execute("INSTALL httpfs")
        conn.execute("LOAD httpfs")
        conn.execute(f"SET s3_region='{REGION}'")

        requested = (os.getenv("S3_ACCESS_MODE", "auto") or "auto").strip().lower()
        if requested in {"unsigned", "public", "anonymous", "anon", "no-sign-request"}:
            conn.execute("SET s3_access_key_id=''")
            conn.execute("SET s3_secret_access_key=''")
            conn.execute("SET s3_session_token=''")
        else:
            access_key = os.getenv("AWS_ACCESS_KEY_ID", "")
            secret_key = os.getenv("AWS_SECRET_ACCESS_KEY", "")
            session_token = os.getenv("AWS_SESSION_TOKEN", "")
            if access_key:
                conn.execute(f"SET s3_access_key_id='{access_key}'")
            if secret_key:
                conn.execute(f"SET s3_secret_access_key='{secret_key}'")
            if session_token:
                conn.execute(f"SET s3_session_token='{session_token}'")
    return conn


def _run_with_duckdb_timeout(
    conn: duckdb.DuckDBPyConnection,
    timeout_seconds: int,
    action: Callable[[], _T],
) -> _T:
    if timeout_seconds <= 0:
        return action()

    timed_out = threading.Event()

    def interrupt() -> None:
        timed_out.set()
        try:
            conn.interrupt()
        except Exception:
            pass

    timer = threading.Timer(timeout_seconds, interrupt)
    timer.daemon = True
    timer.start()
    try:
        result = action()
        if timed_out.is_set():
            raise TimeoutError(f"DuckDB profiling exceeded {timeout_seconds} seconds")
        return result
    except Exception as exc:
        if timed_out.is_set():
            raise TimeoutError(f"DuckDB profiling exceeded {timeout_seconds} seconds") from exc
        raise
    finally:
        timer.cancel()


def _scan_sql(source_ref: str, family: str, table: Dict[str, Any]) -> str:
    ref = _sql_quote_string(source_ref)
    if family == "parquet":
        return f"read_parquet({ref})"
    if family == "json":
        return f"read_json_auto({ref}, maximum_object_size={_QUERY_MAX_FILE_BYTES})"

    delimiter = table.get("delimiter")
    options = ["quote='\"'"]
    if delimiter:
        options.append(f"delim={_sql_quote_string(str(delimiter))}")
    return f"read_csv_auto({ref}, {', '.join(options)})"


def _type_category(raw_type: str) -> Tuple[str, Optional[str]]:
    upper = (raw_type or "").upper()
    if upper.startswith(("TINYINT", "SMALLINT", "INTEGER", "BIGINT", "UTINYINT", "USMALLINT", "UINTEGER", "UBIGINT")):
        return "integer", "numeric"
    if upper.startswith(("DECIMAL", "DOUBLE", "FLOAT", "REAL", "HUGEINT")):
        return "number", "numeric"
    if upper.startswith("DATE"):
        return "date", "temporal"
    if upper.startswith("TIMESTAMP"):
        return "datetime", "temporal"
    if upper.startswith("TIME"):
        return "time", "time"
    if upper.startswith("BOOLEAN"):
        return "boolean", None
    return "string", None


def _json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if isinstance(value, decimal.Decimal):
        return float(value)
    if isinstance(value, dt.datetime):
        if value.tzinfo is not None:
            return value.isoformat()
        return value.replace(tzinfo=dt.timezone.utc).isoformat()
    if isinstance(value, dt.date):
        return value.isoformat()
    if isinstance(value, dt.time):
        return value.isoformat()
    return value


def _json_safe(value: Any, _seen: Optional[set[int]] = None) -> Any:
    primitive = _json_value(value)
    if primitive is None or isinstance(primitive, (bool, int, float, str)):
        return primitive

    if _seen is None:
        _seen = set()

    if isinstance(value, dict):
        ident = id(value)
        if ident in _seen:
            return "[circular]"
        _seen.add(ident)
        try:
            out: Dict[str, Any] = {}
            for key, child in value.items():
                safe_key = _json_safe(key, _seen)
                out[str(safe_key)] = _json_safe(child, _seen)
            return out
        finally:
            _seen.remove(ident)

    if isinstance(value, (list, tuple, set)):
        ident = id(value)
        if ident in _seen:
            return "[circular]"
        _seen.add(ident)
        try:
            return [_json_safe(child, _seen) for child in value]
        finally:
            _seen.remove(ident)

    return str(value)


def _truncate_cell(value: Any) -> Any:
    value = _json_safe(value)
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, str) and len(value) > _MAX_CELL_CHARS:
        return value[:_MAX_CELL_CHARS] + "…"
    return value


def _top_rows(conn: duckdb.DuckDBPyConnection, scan_sql: str) -> List[Dict[str, Any]]:
    cursor = conn.execute(f"SELECT * FROM {scan_sql} LIMIT 2")
    colnames = [desc[0] for desc in cursor.description]
    rows = cursor.fetchall()
    out: List[Dict[str, Any]] = []
    for row in rows:
        out.append({name: _truncate_cell(value) for name, value in zip(colnames, row)})
    return out


def _mean_from_epoch(value: Any, kind: str) -> Optional[str]:
    value = _json_value(value)
    if value is None:
        return None
    stamp = dt.datetime.fromtimestamp(float(value), tz=dt.timezone.utc)
    if kind == "date":
        return stamp.date().isoformat()
    return stamp.isoformat()


def _schema_looks_like_prose(columns: List[Tuple[str, str, Optional[str]]]) -> bool:
    if len(columns) > 2:
        return False
    return any(_column_name_looks_like_prose(name) for name, _output_type, _aggregate_kind in columns)


def _column_name_looks_like_prose(name: str) -> bool:
    name = str(name or "").strip()
    words = [part for part in name.replace("/", " ").split() if part]
    if len(name) > 80:
        return True
    if len(words) >= 8:
        return True
    if len(words) >= 5:
        return True
    return len(words) >= 3 and any(mark in name for mark in (".", ":", ";", "?", "!"))


def _is_generic_column_name(name: str) -> bool:
    return bool(_GENERIC_COLUMN_RE.match(str(name or "").strip()))


def _all_generic_columns(columns: List[Dict[str, Any]]) -> bool:
    return bool(columns) and all(_is_generic_column_name(str(col.get("name") or "")) for col in columns)


def _single_column_csv(columns: List[Dict[str, Any]]) -> bool:
    return len(columns or []) == 1


def _validate_profile_schema(columns: List[Tuple[str, str, Optional[str]]], family: str) -> None:
    if family != "csv":
        return
    if _schema_looks_like_prose(columns):
        raise _UnusableSchemaError(
            "DuckDB inferred a single prose-like CSV column; suppressing columns to avoid misleading agents"
        )


def _describe_column_tuples(conn: duckdb.DuckDBPyConnection, scan_sql: str) -> List[Tuple[str, str, Optional[str]]]:
    describe_rows = conn.execute(f"DESCRIBE SELECT * FROM {scan_sql}").fetchall()
    columns: List[Tuple[str, str, Optional[str]]] = []
    for row in describe_rows:
        name = str(row[0])
        raw_type = str(row[1])
        output_type, aggregate_kind = _type_category(raw_type)
        columns.append((name, output_type, aggregate_kind))
    return columns


def _column_summaries(columns: List[Tuple[str, str, Optional[str]]]) -> List[Dict[str, Any]]:
    return [{"name": name, "type": output_type} for name, output_type, _aggregate_kind in columns]


def _cap_list(values: List[Any], limit: int) -> Tuple[List[Any], bool, int]:
    return values[:limit], len(values) > limit, len(values)


def _add_capped_list(profile: Dict[str, Any], key: str, values: List[Any], limit: int) -> None:
    capped, truncated, total = _cap_list(values, limit)
    profile[key] = capped
    if truncated:
        profile[f"{key}_truncated"] = True
        profile[f"{key}_total"] = total


def _column_profiles(conn: duckdb.DuckDBPyConnection, scan_sql: str, family: str) -> Tuple[int, List[Dict[str, Any]]]:
    columns = _describe_column_tuples(conn, scan_sql)
    _validate_profile_schema(columns, family)

    select_exprs = ["COUNT(*) AS row_count"]
    for idx, (name, output_type, aggregate_kind) in enumerate(columns):
        quoted = _sql_quote_ident(name)
        select_exprs.append(f"SUM(CASE WHEN {quoted} IS NULL THEN 1 ELSE 0 END) AS null_count_{idx}")
        select_exprs.append(f"COUNT(DISTINCT CAST({quoted} AS VARCHAR)) AS distinct_count_{idx}")
        if aggregate_kind in {"numeric", "temporal", "time"}:
            select_exprs.append(f"MIN({quoted}) AS min_{idx}")
            select_exprs.append(f"MAX({quoted}) AS max_{idx}")
        else:
            select_exprs.append(f"NULL AS min_{idx}")
            select_exprs.append(f"NULL AS max_{idx}")
        if aggregate_kind == "numeric":
            select_exprs.append(f"AVG({quoted}) AS mean_{idx}")
        elif aggregate_kind == "temporal":
            select_exprs.append(f"AVG(epoch(CAST({quoted} AS TIMESTAMP))) AS mean_{idx}")
        else:
            select_exprs.append(f"NULL AS mean_{idx}")

    aggregate_row = conn.execute(f"SELECT {', '.join(select_exprs)} FROM {scan_sql}").fetchone()
    row_count = int(aggregate_row[0] or 0)
    profiles: List[Dict[str, Any]] = []
    offset = 1
    for idx, (name, output_type, aggregate_kind) in enumerate(columns):
        null_count = int(aggregate_row[offset] or 0)
        distinct_count = int(aggregate_row[offset + 1] or 0)
        min_value = aggregate_row[offset + 2]
        max_value = aggregate_row[offset + 3]
        mean_value = aggregate_row[offset + 4]
        if aggregate_kind == "temporal":
            mean_value = _mean_from_epoch(mean_value, output_type)
        else:
            mean_value = _json_value(mean_value)
        profiles.append(
            {
                "name": name,
                "type": output_type,
                "null_rate": round((null_count / row_count), 2) if row_count else 0.0,
                "distinct_count": distinct_count,
                "min": _json_value(min_value) if aggregate_kind in {"numeric", "temporal", "time"} else None,
                "max": _json_value(max_value) if aggregate_kind in {"numeric", "temporal", "time"} else None,
                "mean": mean_value if aggregate_kind in {"numeric", "temporal"} else None,
            }
        )
        offset += 5
    return row_count, profiles


def _snippet_fallback(source_ref: str) -> str:
    raw = _read_prefix_bytes(source_ref, _SNIPPET_FALLBACK_BYTES)
    return raw.decode("utf-8", errors="replace").strip()


def _base_profile(
    *,
    slug: str,
    filename: str,
    family: str,
    size_bytes: int,
    schema_status: str,
    s3_uri: Optional[str],
    dataset_id: Optional[str],
    file_path: Optional[str],
    description: Optional[str],
) -> Dict[str, Any]:
    profile: Dict[str, Any] = {
        "slug": slug,
        "filename": filename,
        "family": family,
        "schema_status": schema_status,
        "schema_error": False,
        "size_bytes": size_bytes,
    }
    if s3_uri:
        profile["s3_uri"] = s3_uri
    if dataset_id:
        profile["dataset_id"] = dataset_id
    if file_path:
        profile["file_path"] = file_path
    if description:
        profile["llm_description"] = description
    return profile


def _log_schema_error(*, slug: str, filename: str, source_ref: str, detail: str) -> None:
    print(
        f"PROFILE_SCHEMA_ERROR {slug}/{filename} ({source_ref}): {detail}",
        file=sys.stderr,
    )


def _attach_snippet(profile: Dict[str, Any], source_ref: str, snippet_cache: Dict[str, str]) -> None:
    snippet = snippet_cache.get(source_ref) or _snippet_fallback(source_ref)
    if snippet:
        profile["snippet"] = snippet


def _build_tabular_profile(
    *,
    slug: str,
    filename: str,
    source_ref: str,
    s3_uri: Optional[str],
    dataset_id: Optional[str],
    file_path: Optional[str],
    family: str,
    table: Dict[str, Any],
    size_bytes: int,
    description: Optional[str],
    duckdb_timeout_seconds: int,
) -> Dict[str, Any]:
    conn = _duckdb_connection(needs_httpfs=_is_s3_ref(source_ref))
    try:
        def profile_query() -> Tuple[int, List[Dict[str, Any]], List[Dict[str, Any]]]:
            scan_sql = _scan_sql(source_ref, family, table)
            row_count, columns = _column_profiles(conn, scan_sql, family)
            top_2_rows = _top_rows(conn, scan_sql)
            return row_count, columns, top_2_rows

        row_count, columns, top_2_rows = _run_with_duckdb_timeout(
            conn,
            duckdb_timeout_seconds,
            profile_query,
        )
    finally:
        conn.close()

    profile = _base_profile(
        slug=slug,
        filename=filename,
        family=family,
        schema_status="strict",
        size_bytes=size_bytes,
        s3_uri=s3_uri,
        dataset_id=dataset_id,
        file_path=file_path,
        description=description,
    )
    profile["row_count"] = row_count
    profile["top_2_rows"] = top_2_rows
    _add_capped_list(profile, "columns", columns, _MAX_COLUMNS)
    return profile


def _build_non_tabular_profile(
    *,
    slug: str,
    filename: str,
    source_ref: str,
    s3_uri: Optional[str],
    dataset_id: Optional[str],
    file_path: Optional[str],
    family: str,
    schema_status: str,
    size_bytes: int,
    description: Optional[str],
    snippet_cache: Dict[str, str],
    schema_error: bool = False,
    schema_error_detail: Optional[str] = None,
) -> Dict[str, Any]:
    profile = _base_profile(
        slug=slug,
        filename=filename,
        family=family,
        schema_status=schema_status,
        size_bytes=size_bytes,
        s3_uri=s3_uri,
        dataset_id=dataset_id,
        file_path=file_path,
        description=description,
    )
    if schema_status not in {"archive"}:
        _attach_snippet(profile, source_ref, snippet_cache)
    if schema_error:
        profile["schema_error"] = True
        _log_schema_error(
            slug=slug,
            filename=filename,
            source_ref=source_ref,
            detail=schema_error_detail or "schema unavailable",
        )
    return profile


def _build_text_unavailable_profile(
    *,
    slug: str,
    filename: str,
    source_ref: str,
    s3_uri: Optional[str],
    dataset_id: Optional[str],
    file_path: Optional[str],
    size_bytes: int,
    description: Optional[str],
    snippet_cache: Dict[str, str],
    schema_error_detail: str,
) -> Dict[str, Any]:
    return _build_non_tabular_profile(
        slug=slug,
        filename=filename,
        source_ref=source_ref,
        s3_uri=s3_uri,
        dataset_id=dataset_id,
        file_path=file_path,
        family="text",
        schema_status="unavailable",
        size_bytes=size_bytes,
        description=description,
        snippet_cache=snippet_cache,
        schema_error=True,
        schema_error_detail=schema_error_detail,
    )


def _build_oversized_profile(
    *,
    slug: str,
    filename: str,
    source_ref: str,
    s3_uri: Optional[str],
    dataset_id: Optional[str],
    file_path: Optional[str],
    family: str,
    size_bytes: int,
    max_file_bytes: int,
    description: Optional[str],
) -> Dict[str, Any]:
    profile = _base_profile(
        slug=slug,
        filename=filename,
        family=family,
        schema_status="unavailable",
        size_bytes=size_bytes,
        s3_uri=s3_uri,
        dataset_id=dataset_id,
        file_path=file_path,
        description=description,
    )
    profile["schema_error"] = True
    _log_schema_error(
        slug=slug,
        filename=filename,
        source_ref=source_ref,
        detail=f"File size {size_bytes} exceeds max profile file size {max_file_bytes}",
    )
    return profile


def _build_single_column_profile(
    *,
    slug: str,
    filename: str,
    source_ref: str,
    s3_uri: Optional[str],
    dataset_id: Optional[str],
    file_path: Optional[str],
    size_bytes: int,
    description: Optional[str],
    snippet_cache: Dict[str, str],
) -> Dict[str, Any]:
    profile = _build_non_tabular_profile(
        slug=slug,
        filename=filename,
        source_ref=source_ref,
        s3_uri=s3_uri,
        dataset_id=dataset_id,
        file_path=file_path,
        family="text",
        schema_status="single_column",
        size_bytes=size_bytes,
        description=description,
        snippet_cache=snippet_cache,
    )
    profile["column_count"] = 1
    return profile


def _build_archive_profile(
    *,
    slug: str,
    filename: str,
    source_ref: str,
    s3_uri: Optional[str],
    dataset_id: Optional[str],
    file_path: Optional[str],
    family: str,
    size_bytes: int,
    description: Optional[str],
) -> Dict[str, Any]:
    profile = _base_profile(
        slug=slug,
        filename=filename,
        family=family,
        schema_status="archive",
        size_bytes=size_bytes,
        s3_uri=s3_uri,
        dataset_id=dataset_id,
        file_path=file_path,
        description=description,
    )
    return profile


def _iter_xml_rows(source_ref: str, record_tag: str) -> Tuple[List[Dict[str, str]], int, bool]:
    rows: List[Dict[str, str]] = []
    scanned = 0
    truncated = False
    body = None
    try:
        if _is_s3_ref(source_ref):
            bucket, key = _parse_s3_uri(source_ref)
            body = _build_s3_client_for_runtime().get_object(Bucket=bucket, Key=key)["Body"]
        else:
            body = Path(source_ref).open("rb")
        for _event, elem in ET.iterparse(body, events=("end",)):
            if _local_xml_name(elem.tag) != record_tag:
                continue
            scanned += 1
            row = _xml_record_to_row(elem)
            if row:
                rows.append(row)
            elem.clear()
            if scanned >= _MAX_XML_SCHEMA_RECORDS:
                truncated = True
                break
    finally:
        if body is not None and hasattr(body, "close"):
            body.close()
    return rows, scanned, truncated


def _xml_columns_from_rows(rows: List[Dict[str, str]], fallback_fields: List[str]) -> List[Dict[str, Any]]:
    names: List[str] = []
    seen = set()
    for row in rows:
        for name in row:
            if name not in seen:
                seen.add(name)
                names.append(name)
    for name in fallback_fields:
        if name not in seen:
            seen.add(name)
            names.append(name)
    return [{"name": name, "type": "string"} for name in names]


def _build_xml_profile(
    *,
    slug: str,
    filename: str,
    source_ref: str,
    s3_uri: Optional[str],
    dataset_id: Optional[str],
    file_path: Optional[str],
    family: str,
    size_bytes: int,
    description: Optional[str],
    snippet_cache: Dict[str, str],
) -> Dict[str, Any]:
    text = _read_prefix_bytes(source_ref, _SNIFF_BYTES).decode("utf-8", errors="replace")
    preview = _build_xml_preview(text, size_bytes)
    candidates = preview.get("xml_record_tag_candidates") or []
    record_tag = _normalize_xml_record_tag(candidates[0]) if candidates else None
    if not record_tag:
        profile = _build_non_tabular_profile(
            slug=slug,
            filename=filename,
            source_ref=source_ref,
            s3_uri=s3_uri,
            dataset_id=dataset_id,
            file_path=file_path,
            family=family,
            schema_status="unavailable",
            size_bytes=size_bytes,
            description=description,
            snippet_cache=snippet_cache,
            schema_error=True,
            schema_error_detail="Could not infer an XML record_tag",
        )
        profile.update(preview)
        return profile

    try:
        rows, scanned, truncated = _iter_xml_rows(source_ref, record_tag)
    except Exception as exc:
        profile = _build_non_tabular_profile(
            slug=slug,
            filename=filename,
            source_ref=source_ref,
            s3_uri=s3_uri,
            dataset_id=dataset_id,
            file_path=file_path,
            family=family,
            schema_status="unavailable",
            size_bytes=size_bytes,
            description=description,
            snippet_cache=snippet_cache,
            schema_error=True,
            schema_error_detail=f"{type(exc).__name__}: {exc}",
        )
        profile.update(preview)
        profile["record_tag"] = record_tag
        return profile

    columns = _xml_columns_from_rows(rows, [str(field) for field in preview.get("xml_schema_fields") or []])
    if not columns:
        profile = _build_non_tabular_profile(
            slug=slug,
            filename=filename,
            source_ref=source_ref,
            s3_uri=s3_uri,
            dataset_id=dataset_id,
            file_path=file_path,
            family=family,
            schema_status="unavailable",
            size_bytes=size_bytes,
            description=description,
            snippet_cache=snippet_cache,
            schema_error=True,
            schema_error_detail="No queryable XML record fields found",
        )
    else:
        profile = _base_profile(
            slug=slug,
            filename=filename,
            family=family,
            schema_status="strict",
            size_bytes=size_bytes,
            s3_uri=s3_uri,
            dataset_id=dataset_id,
            file_path=file_path,
            description=description,
        )
        _add_capped_list(profile, "columns", columns, _MAX_COLUMNS)
        profile["top_2_rows"] = [{key: _truncate_cell(value) for key, value in row.items()} for row in rows[:2]]
        profile["records_scanned_for_schema"] = scanned
        if truncated:
            profile["records_scanned_for_schema_truncated"] = True
    profile.update(preview)
    profile["record_tag"] = record_tag
    return profile


def _process_entry(
    entry: Dict[str, Any],
    description_cache: Dict[str, str],
    snippet_cache: Dict[str, str],
    max_file_bytes: int,
    duckdb_timeout_seconds: int,
) -> Tuple[Tuple[str, str], Optional[Dict[str, Any]], Optional[str]]:
    slug = str(entry["slug"])
    filename = str(entry["filename"])
    source_ref = str(entry["source_ref"])
    s3_uri = str(entry.get("s3_uri") or "").strip() or None
    dataset_id = entry.get("dataset_id")
    file_path = entry.get("file_path")
    table = entry["table"]
    key = (slug, filename)
    try:
        size_bytes = int(entry.get("size_bytes") or 0) or _head_size_bytes(source_ref)
        cache_key = s3_uri or source_ref
        description = description_cache.get(cache_key)
        family = _resolve_family(table, source_ref)
        metadata_path = file_path or source_ref
        if _is_metadata_profile_path(metadata_path):
            profile = _build_non_tabular_profile(
                slug=slug,
                filename=filename,
                source_ref=source_ref,
                s3_uri=s3_uri,
                dataset_id=dataset_id,
                file_path=file_path,
                family=family,
                schema_status="metadata",
                size_bytes=size_bytes,
                description=description,
                snippet_cache=snippet_cache,
            )
        elif max_file_bytes > 0 and size_bytes > max_file_bytes and family not in {"archive", "binary"}:
            profile = _build_oversized_profile(
                slug=slug,
                filename=filename,
                source_ref=source_ref,
                s3_uri=s3_uri,
                dataset_id=dataset_id,
                file_path=file_path,
                family=family,
                size_bytes=size_bytes,
                max_file_bytes=max_file_bytes,
                description=description,
            )
        elif family in {"archive", "binary"}:
            profile = _build_archive_profile(
                slug=slug,
                filename=filename,
                source_ref=source_ref,
                s3_uri=s3_uri,
                dataset_id=dataset_id,
                file_path=file_path,
                family="archive",
                size_bytes=size_bytes,
                description=description,
            )
        elif family == "xml":
            profile = _build_xml_profile(
                slug=slug,
                filename=filename,
                source_ref=source_ref,
                s3_uri=s3_uri,
                dataset_id=dataset_id,
                file_path=file_path,
                family=family,
                size_bytes=size_bytes,
                description=description,
                snippet_cache=snippet_cache,
            )
        elif family in {"csv", "json", "parquet"}:
            try:
                profile = _build_tabular_profile(
                    slug=slug,
                    filename=filename,
                    source_ref=source_ref,
                    s3_uri=s3_uri,
                    dataset_id=dataset_id,
                    file_path=file_path,
                    family=family,
                    table=table,
                    size_bytes=size_bytes,
                    description=description,
                    duckdb_timeout_seconds=duckdb_timeout_seconds,
                )
                if family == "csv" and _single_column_csv(profile.get("columns") or []):
                    profile = _build_single_column_profile(
                        slug=slug,
                        filename=filename,
                        source_ref=source_ref,
                        s3_uri=s3_uri,
                        dataset_id=dataset_id,
                        file_path=file_path,
                        size_bytes=size_bytes,
                        description=description,
                        snippet_cache=snippet_cache,
                    )
                elif family == "csv" and _all_generic_columns(profile.get("columns") or []):
                    profile = _build_text_unavailable_profile(
                        slug=slug,
                        filename=filename,
                        source_ref=source_ref,
                        s3_uri=s3_uri,
                        dataset_id=dataset_id,
                        file_path=file_path,
                        size_bytes=size_bytes,
                        description=description,
                        snippet_cache=snippet_cache,
                        schema_error_detail="CSV parser inferred generic columns; treating as text",
                    )
            except Exception as exc:
                schema_error = f"{type(exc).__name__}: {exc}"
                if family == "csv":
                    profile = _build_text_unavailable_profile(
                        slug=slug,
                        filename=filename,
                        source_ref=source_ref,
                        s3_uri=s3_uri,
                        dataset_id=dataset_id,
                        file_path=file_path,
                        size_bytes=size_bytes,
                        description=description,
                        snippet_cache=snippet_cache,
                        schema_error_detail=schema_error,
                    )
                else:
                    profile = _build_non_tabular_profile(
                        slug=slug,
                        filename=filename,
                        source_ref=source_ref,
                        s3_uri=s3_uri,
                        dataset_id=dataset_id,
                        file_path=file_path,
                        family=family,
                        schema_status="unavailable",
                        size_bytes=size_bytes,
                        description=description,
                        snippet_cache=snippet_cache,
                        schema_error=True,
                        schema_error_detail=schema_error,
                    )
        else:
            profile = _build_non_tabular_profile(
                slug=slug,
                filename=filename,
                source_ref=source_ref,
                s3_uri=s3_uri,
                dataset_id=dataset_id,
                file_path=file_path,
                family=family,
                schema_status="unavailable",
                size_bytes=size_bytes,
                description=description,
                snippet_cache=snippet_cache,
            )
        return key, profile, None
    except Exception as exc:
        return key, None, f"{type(exc).__name__}: {exc}"


def build_profiles(
    *,
    input_path: Path,
    output_path: Path,
    input_kind: str = "auto",
    descriptions_path: Path = _DESC_PATH,
    snippets_path: Path = _SNIPPET_PATH,
    parallel: int = 4,
    max_file_bytes: int = _DEFAULT_MAX_PROFILE_FILE_BYTES,
    duckdb_timeout_seconds: int = _DEFAULT_DUCKDB_TIMEOUT_SECONDS,
    resume: bool = False,
    bucket: str = BUCKET,
) -> Dict[str, int]:
    description_cache = _load_description_cache(descriptions_path)
    snippet_cache = _load_snippet_cache(snippets_path)
    entries = _iter_input_entries(input_path, input_kind=input_kind, bucket=bucket)
    existing_keys = _load_existing_keys(output_path) if resume else set()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if resume else "w"
    written = 0
    skipped = 0
    errors = 0

    with output_path.open(mode) as out:
        if parallel <= 1:
            for entry in tqdm(
                entries,
                total=len(entries),
                desc="Profiling files",
                unit="file",
            ):
                identity = _profile_identity(entry)
                key = (entry["slug"], entry["filename"])
                if identity is not None and identity in existing_keys:
                    skipped += 1
                    continue
                _, profile, error = _process_entry(
                    entry,
                    description_cache,
                    snippet_cache,
                    max_file_bytes,
                    duckdb_timeout_seconds,
                )
                if error is not None or profile is None:
                    errors += 1
                    print(f"SKIP {key[0]}/{key[1]}: {error}", file=sys.stderr)
                    continue
                out.write(json.dumps(_json_safe(profile)))
                out.write("\n")
                out.flush()
                written += 1
            return {"written": written, "skipped": skipped, "errors": errors}

        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as executor:
            futures: Dict[concurrent.futures.Future, Tuple[str, str]] = {}
            for entry in entries:
                identity = _profile_identity(entry)
                key = (entry["slug"], entry["filename"])
                if identity is not None and identity in existing_keys:
                    skipped += 1
                    continue
                future = executor.submit(
                    _process_entry,
                    entry,
                    description_cache,
                    snippet_cache,
                    max_file_bytes,
                    duckdb_timeout_seconds,
                )
                futures[future] = key

            with tqdm(total=len(futures), desc="Profiling files", unit="file") as pbar:
                for future in concurrent.futures.as_completed(futures):
                    _, profile, error = future.result()
                    if error is not None or profile is None:
                        errors += 1
                        key = futures[future]
                        print(f"SKIP {key[0]}/{key[1]}: {error}", file=sys.stderr)
                        pbar.update(1)
                        continue
                    out.write(json.dumps(_json_safe(profile)))
                    out.write("\n")
                    out.flush()
                    written += 1
                    pbar.update(1)

    return {"written": written, "skipped": skipped, "errors": errors}


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    summary = build_profiles(
        input_path=Path(args.input),
        output_path=Path(args.output),
        input_kind=args.input_kind,
        descriptions_path=Path(args.descriptions),
        snippets_path=Path(args.snippets),
        parallel=args.parallel,
        max_file_bytes=args.max_file_bytes,
        duckdb_timeout_seconds=args.duckdb_timeout_seconds,
        resume=args.resume,
        bucket=args.bucket,
    )
    print(
        f"Wrote {summary['written']} profiles to {args.output} "
        f"(skipped={summary['skipped']}, errors={summary['errors']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
