#!/usr/bin/env python3
"""Retry unavailable dataset profiles with safe prefix samples.

This script is intentionally separate from the full profiler. It reads an
existing profile JSONL, writes the rows that are retry candidates, and emits a
new profile JSONL where successful retries are marked as sampled rather than
strict.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import io
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import duckdb
import ijson
try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable=None, **_kwargs):
        return iterable if iterable is not None else _NullTqdm()

from scripts import profile_datasets


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


_DEFAULT_SAMPLE_BYTES = 10 * 1024 * 1024
_DEFAULT_DUCKDB_TIMEOUT_SECONDS = 120
_MIN_SAMPLE_DATA_ROWS = 2
_MIN_SAMPLE_JSON_ITEMS = 2


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Existing profile JSONL")
    parser.add_argument("--output", required=True, help="Updated profile JSONL")
    parser.add_argument(
        "--candidates-output",
        default=None,
        help="Optional JSONL containing exact rows selected for retry",
    )
    parser.add_argument(
        "--failures-output",
        default=None,
        help="Optional JSONL containing rows that still could not be sampled",
    )
    parser.add_argument("--sample-bytes", type=int, default=_DEFAULT_SAMPLE_BYTES)
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--duckdb-timeout-seconds", type=int, default=_DEFAULT_DUCKDB_TIMEOUT_SECONDS)
    return parser.parse_args(argv)


def _jsonl_rows(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _write_jsonl(path: Optional[Path], rows: Iterable[Dict[str, Any]]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(profile_datasets._json_safe(row)))
            f.write("\n")


def is_retry_candidate(row: Dict[str, Any]) -> bool:
    status = str(row.get("schema_status") or "").strip()
    schema_error = row.get("schema_error")
    if status in {"metadata", "archive"} or row.get("family") == "archive":
        return False
    if status == "sampled":
        return False
    if status in {"unavailable", "error"}:
        return True
    if schema_error is True:
        return True
    return isinstance(schema_error, str) and bool(schema_error.strip())


def _source_ref(row: Dict[str, Any]) -> str:
    return str(row.get("source_ref") or row.get("local_path") or row.get("s3_uri") or "").strip()


def _trim_to_complete_records(raw: bytes, *, source_size: int, requested_bytes: int) -> Tuple[bytes, bool]:
    if not raw:
        return b"", False
    truncated = source_size <= 0 or len(raw) < source_size
    if not truncated or len(raw) < requested_bytes:
        return raw, truncated
    last_newline = raw.rfind(b"\n")
    if last_newline < 0:
        return b"", True
    return raw[: last_newline + 1], True


def _sample_prefix(source_ref: str, sample_bytes: int, source_size: int) -> Tuple[bytes, bool]:
    raw = profile_datasets._read_prefix_bytes(source_ref, sample_bytes)
    return _trim_to_complete_records(raw, source_size=source_size, requested_bytes=sample_bytes)


def _nonempty_lines(raw: bytes) -> List[bytes]:
    return [line for line in raw.splitlines() if line.strip()]


def _sample_csv_profile(
    row: Dict[str, Any],
    sample_bytes: int,
    duckdb_timeout_seconds: int,
) -> Dict[str, Any]:
    source_ref = _source_ref(row)
    if not source_ref:
        raise ValueError("missing source_ref/s3_uri")

    raw, _truncated = _sample_prefix(source_ref, sample_bytes, int(row.get("size_bytes") or 0))
    if len(_nonempty_lines(raw)) < _MIN_SAMPLE_DATA_ROWS + 1:
        raise ValueError(
            "sample did not contain enough complete CSV lines "
            f"(need header plus {_MIN_SAMPLE_DATA_ROWS} data rows)"
        )

    suffix = Path(str(row.get("file_path") or source_ref)).suffix or ".txt"
    tmp_path: Optional[Path] = None
    conn: Optional[duckdb.DuckDBPyConnection] = None
    try:
        with tempfile.NamedTemporaryFile("wb", suffix=suffix, delete=False) as tmp:
            tmp.write(raw)
            tmp_path = Path(tmp.name)

        conn = profile_datasets._duckdb_connection(needs_httpfs=False)

        def query() -> Tuple[List[Tuple[str, str, Optional[str]]], int, List[Dict[str, Any]]]:
            scan_sql = f"read_csv_auto({profile_datasets._sql_quote_string(str(tmp_path))}, quote='\"')"
            columns = profile_datasets._describe_column_tuples(conn, scan_sql)
            profile_datasets._validate_profile_schema(columns, "csv")
            sample_count = int(conn.execute(f"SELECT COUNT(*) FROM {scan_sql}").fetchone()[0] or 0)
            top_rows = profile_datasets._top_rows(conn, scan_sql)
            return columns, sample_count, top_rows

        columns, sample_count, top_rows = profile_datasets._run_with_duckdb_timeout(
            conn,
            duckdb_timeout_seconds,
            query,
        )
    finally:
        if conn is not None:
            conn.close()
        if tmp_path is not None:
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass

    if sample_count < _MIN_SAMPLE_DATA_ROWS:
        raise ValueError(
            f"sample produced only {sample_count} complete CSV data row(s); "
            f"need {_MIN_SAMPLE_DATA_ROWS}"
        )

    if len(columns) == 1:
        updated = dict(row)
        updated["schema_status"] = "single_column"
        updated["schema_error"] = False
        updated["column_count"] = 1
        for key in ("columns", "top_2_rows", "row_count"):
            updated.pop(key, None)
        return updated

    column_summaries = profile_datasets._column_summaries(columns)
    if profile_datasets._all_generic_columns(column_summaries):
        raise ValueError("sample inferred only generic columns")

    updated = dict(row)
    updated["schema_status"] = "sampled"
    updated["schema_error"] = False
    updated["top_2_rows"] = top_rows
    updated["columns"] = column_summaries
    for key in ("row_count", "column_count"):
        updated.pop(key, None)
    return updated


def _json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    return "string"


def _json_items_from_sample(raw: bytes) -> List[Dict[str, Any]]:
    items: List[Any] = []
    for prefix in ("item", "features.item"):
        try:
            for item in ijson.items(io.BytesIO(raw), prefix):
                items.append(item)
                if len(items) >= 200:
                    break
        except Exception:
            pass
        if items:
            break

    if not items:
        text = raw.decode("utf-8", errors="ignore")
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if len(items) >= 200:
                break

    return [item for item in items if isinstance(item, dict)]


def _sample_json_profile(row: Dict[str, Any], sample_bytes: int) -> Dict[str, Any]:
    source_ref = _source_ref(row)
    if not source_ref:
        raise ValueError("missing source_ref/s3_uri")
    raw = profile_datasets._read_prefix_bytes(source_ref, sample_bytes)
    items = _json_items_from_sample(raw)
    if len(items) < _MIN_SAMPLE_JSON_ITEMS:
        raise ValueError(
            f"sample contained only {len(items)} complete JSON object(s); "
            f"need {_MIN_SAMPLE_JSON_ITEMS}"
        )

    names: List[str] = []
    seen: set[str] = set()
    types: Dict[str, str] = {}
    for item in items:
        for name, value in item.items():
            if name not in seen:
                seen.add(name)
                names.append(name)
                types[name] = _json_type(value)

    if not names:
        raise ValueError("sample did not contain JSON object keys")

    updated = dict(row)
    updated["schema_status"] = "sampled"
    updated["schema_error"] = False
    updated["columns"] = [{"name": name, "type": types[name]} for name in names[:200]]
    updated["top_2_rows"] = [
        {key: profile_datasets._truncate_cell(value) for key, value in item.items()}
        for item in items[:2]
    ]
    for key in ("row_count", "column_count"):
        updated.pop(key, None)
    return updated


def retry_sample_row(
    row: Dict[str, Any],
    *,
    sample_bytes: int = _DEFAULT_SAMPLE_BYTES,
    duckdb_timeout_seconds: int = _DEFAULT_DUCKDB_TIMEOUT_SECONDS,
) -> Tuple[Dict[str, Any], Optional[str]]:
    source_ref = _source_ref(row)
    if not source_ref:
        return row, "missing source_ref/s3_uri"

    family = str(row.get("family") or "").lower()
    path = str(row.get("file_path") or row.get("s3_uri") or source_ref).lower()
    try:
        if family == "json" or path.endswith((".json", ".jsonl", ".geojson")):
            return _sample_json_profile(row, sample_bytes), None
        if family in {"csv", "text"} or path.endswith((".csv", ".tsv", ".txt")):
            return _sample_csv_profile(row, sample_bytes, duckdb_timeout_seconds), None
        return row, f"unsupported family for sampling: {family}"
    except Exception as exc:
        return row, f"{type(exc).__name__}: {exc}"


def retry_profiles(
    rows: List[Dict[str, Any]],
    *,
    sample_bytes: int = _DEFAULT_SAMPLE_BYTES,
    parallel: int = 4,
    duckdb_timeout_seconds: int = _DEFAULT_DUCKDB_TIMEOUT_SECONDS,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, int]]:
    candidates = [row for row in rows if is_retry_candidate(row)]
    updated_by_index: Dict[int, Dict[str, Any]] = {}
    failures: List[Dict[str, Any]] = []
    changed = 0

    def run_one(index: int, row: Dict[str, Any]) -> Tuple[int, Dict[str, Any], Optional[str]]:
        updated, error = retry_sample_row(
            row,
            sample_bytes=sample_bytes,
            duckdb_timeout_seconds=duckdb_timeout_seconds,
        )
        return index, updated, error

    indexed_candidates = [(idx, row) for idx, row in enumerate(rows) if is_retry_candidate(row)]
    if parallel <= 1:
        iterator = (run_one(idx, row) for idx, row in indexed_candidates)
    else:
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=parallel)
        futures = [executor.submit(run_one, idx, row) for idx, row in indexed_candidates]
        iterator = (future.result() for future in concurrent.futures.as_completed(futures))

    try:
        with tqdm(total=len(indexed_candidates), desc="Sampling unavailable profiles", unit="file") as pbar:
            for idx, updated, error in iterator:
                original = rows[idx]
                if error is None and updated != original:
                    changed += 1
                    updated_by_index[idx] = updated
                elif error is not None:
                    failure = dict(original)
                    failure["sample_error"] = error
                    failures.append(failure)
                pbar.update(1)
    finally:
        if parallel > 1:
            executor.shutdown(wait=True)

    out_rows = [updated_by_index.get(idx, row) for idx, row in enumerate(rows)]
    summary = {
        "rows": len(rows),
        "candidates": len(candidates),
        "sampled": changed,
        "failed": len(failures),
    }
    return out_rows, failures, summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    input_path = Path(args.input)
    output_path = Path(args.output)
    rows = list(_jsonl_rows(input_path))
    candidates = [row for row in rows if is_retry_candidate(row)]
    _write_jsonl(Path(args.candidates_output) if args.candidates_output else None, candidates)

    out_rows, failures, summary = retry_profiles(
        rows,
        sample_bytes=args.sample_bytes,
        parallel=args.parallel,
        duckdb_timeout_seconds=args.duckdb_timeout_seconds,
    )
    _write_jsonl(output_path, out_rows)
    _write_jsonl(Path(args.failures_output) if args.failures_output else None, failures)

    print(
        "Sample retry complete: "
        f"rows={summary['rows']} candidates={summary['candidates']} "
        f"sampled={summary['sampled']} failed={summary['failed']} output={output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
