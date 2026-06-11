#!/usr/bin/env python3
"""Extract current Kramabench task/plan sources into a table-search corpus.

The output parquet is compatible with ``hybrid_search/process.py``: it contains
at least ``dataset_uri``, ``metadata``, and ``content`` columns. Extracted CSV
tables are written to a local directory that mirrors the intended S3 layout.
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_BUCKET = "sana-kramabench"
DEFAULT_EVAL_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_PARQUET = Path("kramabench_tables.parquet")
DEFAULT_OUTPUT_SCHEMAS = Path("kramabench_table_schemas.jsonl")
DEFAULT_OUTPUT_MANIFEST = Path("kramabench_table_manifest.jsonl")
DEFAULT_OUTPUT_REPORT = Path("kramabench_extract_report.json")
DEFAULT_TABLES_DIR = Path("kramabench_tables")
DEFAULT_MAX_SAMPLE_WORDS = 10_000
SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".html", ".json", ".npz", ".tle", ".sp3", ".dat", ".txt"}


@dataclass(frozen=True)
class SourceRef:
    raw: str
    dataset_id: str
    domain: str
    relative_file_path: str
    local_path: Path
    source_uri: str


@dataclass(frozen=True)
class OutputPaths:
    parquet: Path
    schemas: Path
    manifest: Path
    report: Path
    tables_dir: Path


@dataclass(frozen=True)
class ExtractedTable:
    table_name: str
    columns: list[str]
    dataframe: pd.DataFrame
    description: str = ""


def collect_source_refs(roots: Iterable[Path]) -> list[str]:
    refs: set[str] = set()
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("task_*.json"):
            try:
                payload = json.loads(path.read_text())
            except json.JSONDecodeError:
                continue
            _walk_for_sources(payload, refs)
    return sorted(refs)


def collect_source_refs_from_file(path: Path, *, bucket: str = DEFAULT_BUCKET) -> list[str]:
    refs: set[str] = set()
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            raw = line.strip()
            if not raw:
                continue
            if raw.startswith("{"):
                try:
                    record = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if not isinstance(record, dict):
                    continue
                raw = str(
                    record.get("source")
                    or record.get("source_s3_key")
                    or record.get("source_uri")
                    or record.get("s3_uri")
                    or ""
                ).strip()
            if raw.startswith(f"s3://{bucket}/"):
                raw = raw[len(f"s3://{bucket}/") :]
            if raw.startswith("datagov/kramabench-"):
                refs.add(raw)
    return sorted(refs)


def _walk_for_sources(value: Any, refs: set[str]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"source", "source_sequence"}:
                _add_source_value(item, refs)
            _walk_for_sources(item, refs)
    elif isinstance(value, list):
        for item in value:
            _walk_for_sources(item, refs)


def _add_source_value(value: Any, refs: set[str]) -> None:
    if isinstance(value, str):
        if value.startswith("datagov/kramabench-"):
            refs.add(value)
    elif isinstance(value, list):
        for item in value:
            _add_source_value(item, refs)


def resolve_source_ref(source: str, *, eval_root: Path, bucket: str = DEFAULT_BUCKET) -> SourceRef:
    parts = Path(source).parts
    if len(parts) < 4 or parts[0] != "datagov" or not parts[1].startswith("kramabench-"):
        raise ValueError(f"Unsupported Kramabench source reference: {source}")
    if parts[2] != "files":
        raise ValueError(f"Expected Kramabench source under files/: {source}")

    dataset_id = parts[1]
    source_id = dataset_id[len("kramabench-") :]
    match = re.match(r"(.+?)-(?:easy|hard)-\d+$", source_id)
    if not match:
        raise ValueError(f"Could not infer Kramabench domain from dataset id: {dataset_id}")
    domain = match.group(1)
    relative_file_path = "/".join(parts[3:])
    local_path = (
        eval_root
        / "other-benchmarks"
        / "Kramabench"
        / "data"
        / domain
        / "input"
        / relative_file_path
    )
    return SourceRef(
        raw=source,
        dataset_id=dataset_id,
        domain=domain,
        relative_file_path=relative_file_path,
        local_path=local_path,
        source_uri=f"s3://{bucket}/{source}",
    )


def extract_source(source_ref: SourceRef, *, max_sample_rows: int) -> list[ExtractedTable]:
    suffix = source_ref.local_path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file extension: {suffix or '<none>'}")
    if suffix == ".csv":
        return [_extract_csv(source_ref.local_path, source_ref.local_path.stem)]
    if suffix == ".xlsx":
        return _extract_xlsx(source_ref.local_path)
    if suffix == ".html":
        return _extract_html(source_ref.local_path)
    if suffix == ".json":
        return _extract_json(source_ref.local_path)
    if suffix == ".npz":
        return _extract_npz(source_ref.local_path)
    if suffix == ".tle":
        return [_extract_tle(source_ref.local_path)]
    if suffix == ".sp3":
        return [_extract_sp3(source_ref.local_path)]
    if suffix == ".dat":
        return [_extract_dat(source_ref.local_path)]
    if suffix == ".txt":
        return [_extract_text_lines(source_ref.local_path)]
    raise AssertionError(f"Unhandled extension: {suffix}")


def _clean_columns(columns: Iterable[Any]) -> list[str]:
    out: list[str] = []
    seen: Counter[str] = Counter()
    for idx, col in enumerate(columns):
        name = str(col if col is not None else "").strip()
        if not name or name.lower().startswith("unnamed:"):
            name = f"column_{idx}"
        name = re.sub(r"\s+", " ", name)
        seen[name] += 1
        out.append(name if seen[name] == 1 else f"{name}_{seen[name]}")
    return out


def _finalize_df(df: pd.DataFrame, *, max_rows: int | None = None) -> pd.DataFrame:
    if max_rows is not None:
        df = df.head(max_rows)
    df = df.copy()
    df.columns = _clean_columns(df.columns)
    return df


def _extract_csv(path: Path, table_name: str) -> ExtractedTable:
    df = _read_csv_flexible(path)
    df = _finalize_df(df)
    return ExtractedTable(table_name=_safe_name(table_name), columns=list(df.columns), dataframe=df)


def _read_csv_flexible(path: Path) -> pd.DataFrame:
    attempts: list[tuple[str | None, str | None, int, bool]] = [(None, None, 0, False)]
    header_start_by_encoding: dict[str, int] = {}
    for encoding in ("utf-8-sig", "cp1252", "latin1"):
        try:
            lines = path.read_text(encoding=encoding).splitlines()
        except UnicodeDecodeError:
            continue
        header_start_by_encoding[encoding] = _detect_table_header_start(lines)
        for sep in (None, "\t", ",", ";", "|"):
            attempts.append((encoding, sep, header_start_by_encoding[encoding], True))
            attempts.append((encoding, sep, 0, True))

    errors: list[str] = []
    seen: set[tuple[str | None, str | None, int, bool]] = set()
    for encoding, sep, skiprows, python_engine in attempts:
        key = (encoding, sep, skiprows, python_engine)
        if key in seen:
            continue
        seen.add(key)
        try:
            kwargs: dict[str, Any] = {
                "skiprows": skiprows,
                "on_bad_lines": "skip",
            }
            if encoding is not None:
                kwargs["encoding"] = encoding
            if sep is not None:
                kwargs["sep"] = sep
            if python_engine or sep is None:
                kwargs["engine"] = "python"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=pd.errors.ParserWarning)
                df = pd.read_csv(path, **kwargs)
            if not df.empty and len(df.columns) > 1:
                return df
            if not df.empty and not errors:
                fallback = df
        except Exception as exc:  # noqa: BLE001 - try the next dialect/encoding
            errors.append(f"{type(exc).__name__}: {exc}")

    if "fallback" in locals():
        return fallback
    raise ValueError(f"Could not parse CSV {path}: {'; '.join(errors[:5])}")


def _detect_table_header_start(lines: list[str]) -> int:
    non_empty = [(idx, line.strip()) for idx, line in enumerate(lines[:100]) if line.strip()]
    delimiters = [",", "\t", ";", "|"]
    for position, (idx, line) in enumerate(non_empty):
        counts = {delimiter: line.count(delimiter) for delimiter in delimiters}
        delimiter, count = max(counts.items(), key=lambda item: item[1])
        if count == 0:
            continue
        next_count = 0
        for _, next_line in non_empty[position + 1 : position + 4]:
            next_count = max(next_count, next_line.count(delimiter))
        if next_count == 0 or abs(next_count - count) <= max(2, count // 2):
            return idx
    return 0


def _extract_xlsx(path: Path) -> list[ExtractedTable]:
    tables: list[ExtractedTable] = []
    workbook = pd.ExcelFile(path)
    for sheet_name in workbook.sheet_names:
        df = workbook.parse(sheet_name=sheet_name)
        df = df.dropna(how="all")
        if df.empty:
            continue
        df = _finalize_df(df)
        tables.append(
            ExtractedTable(
                table_name=f"{_safe_name(path.stem)}__sheet_{_safe_name(sheet_name)}",
                columns=list(df.columns),
                dataframe=df,
                description=f"Excel sheet {sheet_name}",
            )
        )
    return tables


def _extract_html(path: Path) -> list[ExtractedTable]:
    tables: list[ExtractedTable] = []
    for idx, df in enumerate(pd.read_html(path)):
        df = df.dropna(how="all")
        if df.empty:
            continue
        df = _finalize_df(df)
        tables.append(
            ExtractedTable(
                table_name=f"{_safe_name(path.stem)}__table_{idx + 1}",
                columns=list(df.columns),
                dataframe=df,
                description=f"HTML table {idx + 1}",
            )
        )
    return tables


def _extract_json(path: Path) -> list[ExtractedTable]:
    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        df = pd.json_normalize(payload)
    elif isinstance(payload, dict):
        list_value = next((v for v in payload.values() if isinstance(v, list)), None)
        df = pd.json_normalize(list_value if list_value is not None else [payload])
    else:
        df = pd.DataFrame({"value": [payload]})
    df = _finalize_df(df)
    return [ExtractedTable(table_name=_safe_name(path.stem), columns=list(df.columns), dataframe=df)]


def _extract_npz(path: Path) -> list[ExtractedTable]:
    tables: list[ExtractedTable] = []
    with np.load(path, allow_pickle=False) as archive:
        for name in archive.files:
            arr = archive[name]
            if arr.ndim == 0:
                df = pd.DataFrame({"value": [arr.item()]})
            elif arr.ndim == 1:
                df = pd.DataFrame({"index": range(len(arr)), "value": arr})
            else:
                flat = arr.reshape((-1, arr.shape[-1]))
                df = pd.DataFrame(flat, columns=[f"dim_{i}" for i in range(flat.shape[1])])
            df = _finalize_df(df)
            tables.append(
                ExtractedTable(
                    table_name=f"{_safe_name(path.stem)}__array_{_safe_name(name)}",
                    columns=list(df.columns),
                    dataframe=df,
                    description=f"NPZ array {name} shape {tuple(arr.shape)}",
                )
            )
    return tables


def _extract_tle(path: Path) -> ExtractedTable:
    lines = [line.strip() for line in path.read_text(errors="replace").splitlines() if line.strip()]
    records: list[dict[str, str]] = []
    i = 0
    while i < len(lines):
        name = ""
        if i + 2 < len(lines) and lines[i + 1].startswith("1 ") and lines[i + 2].startswith("2 "):
            name, line1, line2 = lines[i], lines[i + 1], lines[i + 2]
            i += 3
        elif i + 1 < len(lines) and lines[i].startswith("1 ") and lines[i + 1].startswith("2 "):
            line1, line2 = lines[i], lines[i + 1]
            i += 2
        else:
            i += 1
            continue
        records.append(
            {
                "name": name,
                "norad_id": line1[2:7].strip(),
                "line1": line1,
                "line2": line2,
            }
        )
    df = _finalize_df(pd.DataFrame(records or {"line": lines}))
    return ExtractedTable(table_name=_safe_name(path.stem), columns=list(df.columns), dataframe=df)


def _extract_sp3(path: Path) -> ExtractedTable:
    rows: list[dict[str, Any]] = []
    epoch = ""
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("*"):
            epoch = " ".join(line[1:].split())
            continue
        if not line.startswith("P"):
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        rows.append(
            {
                "epoch": epoch,
                "satellite": parts[0][1:],
                "x_km": _to_float(parts[1]),
                "y_km": _to_float(parts[2]),
                "z_km": _to_float(parts[3]),
                "clock": _to_float(parts[4]),
            }
        )
    df = _finalize_df(pd.DataFrame(rows))
    return ExtractedTable(table_name=_safe_name(path.stem), columns=list(df.columns), dataframe=df)


def _extract_dat(path: Path) -> ExtractedTable:
    df = pd.read_csv(path, sep=r"\s+", header=None, comment="#")
    df.columns = [f"col_{i}" for i in range(len(df.columns))]
    df = _finalize_df(df)
    return ExtractedTable(table_name=_safe_name(path.stem), columns=list(df.columns), dataframe=df)


def _extract_text_lines(path: Path) -> ExtractedTable:
    lines = path.read_text(errors="replace").splitlines()
    df = pd.DataFrame(
        [{"line_number": idx + 1, "text": line.strip()} for idx, line in enumerate(lines) if line.strip()]
    )
    df = _finalize_df(df)
    return ExtractedTable(table_name=_safe_name(path.stem), columns=list(df.columns), dataframe=df)


def _to_float(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def _safe_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    safe = safe.strip("._")
    return safe or "table"


def _table_key(source_ref: SourceRef, table_name: str) -> str:
    relative_stem = _safe_name(Path(source_ref.relative_file_path).with_suffix("").as_posix())
    source_stem = _safe_name(Path(source_ref.relative_file_path).stem)
    if table_name == source_stem:
        table_file = relative_stem
    else:
        table_file = f"{relative_stem}__{table_name}"
    return f"datagov/{source_ref.dataset_id}/tables/{table_file}.csv"


def _content_sample(
    df: pd.DataFrame,
    *,
    max_sample_words: int = DEFAULT_MAX_SAMPLE_WORDS,
    max_sample_rows: int | None = None,
) -> str:
    sample = df.head(max_sample_rows) if max_sample_rows is not None else df
    words = sample.to_csv(index=False).split()
    if max_sample_words > 0:
        words = words[:max_sample_words]
    return " ".join(words)


def _metadata_text(source_ref: SourceRef, table: ExtractedTable) -> str:
    parts = [
        source_ref.domain,
        source_ref.dataset_id,
        source_ref.relative_file_path,
        table.table_name,
        " ".join(table.columns),
        table.description,
    ]
    return " ".join(" ".join(str(part).replace("_", " ").replace("-", " ").split()) for part in parts if part)


def extract_sources(
    sources: list[str],
    *,
    eval_root: Path,
    outputs: OutputPaths,
    bucket: str = DEFAULT_BUCKET,
    max_sample_words: int = DEFAULT_MAX_SAMPLE_WORDS,
    max_sample_rows: int | None = None,
) -> dict[str, Any]:
    for path in (outputs.parquet, outputs.schemas, outputs.manifest, outputs.report):
        path.parent.mkdir(parents=True, exist_ok=True)
    outputs.tables_dir.mkdir(parents=True, exist_ok=True)

    parquet_rows: list[dict[str, Any]] = []
    schema_docs: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    extension_counts: Counter[str] = Counter()

    for source in sources:
        try:
            source_ref = resolve_source_ref(source, eval_root=eval_root, bucket=bucket)
            extension_counts[source_ref.local_path.suffix.lower() or "<none>"] += 1
            if not source_ref.local_path.exists():
                raise FileNotFoundError(str(source_ref.local_path))
            tables = extract_source(source_ref, max_sample_rows=max_sample_rows)
            schema_tables: list[dict[str, Any]] = []
            for table in tables:
                if table.dataframe.empty:
                    continue
                table_key = _table_key(source_ref, table.table_name)
                local_table_path = outputs.tables_dir / table_key
                local_table_path.parent.mkdir(parents=True, exist_ok=True)
                table.dataframe.to_csv(local_table_path, index=False)

                metadata = _metadata_text(source_ref, table)
                content = (
                    f"{metadata}\n"
                    f"{_content_sample(table.dataframe, max_sample_words=max_sample_words, max_sample_rows=max_sample_rows)}"
                )
                parquet_rows.append(
                    {
                        "dataset_uri": source_ref.source_uri,
                        "metadata": metadata,
                        "content": content,
                        "source_uri": source_ref.source_uri,
                        "dataset_id": source_ref.dataset_id,
                        "domain": source_ref.domain,
                        "table_name": table.table_name,
                        "local_table_path": str(local_table_path),
                    }
                )
                table_manifest = {
                    "dataset_uri": source_ref.source_uri,
                    "source_uri": source_ref.source_uri,
                    "dataset_id": source_ref.dataset_id,
                    "domain": source_ref.domain,
                    "table_name": table.table_name,
                    "columns": table.columns,
                    "row_count": int(len(table.dataframe)),
                    "local_table_path": str(local_table_path),
                }
                manifest_rows.append(table_manifest)
                schema_tables.append(
                    {
                        "s3_key": source,
                        "source_s3_key": source,
                        "columns": table.columns,
                        "table_kind": source_ref.local_path.suffix.lower().lstrip(".") or "unknown",
                        "delimiter": ",",
                    }
                )
            if schema_tables:
                schema_docs.append(
                    {
                        "source_uri": source_ref.source_uri,
                        "dataset_id": source_ref.dataset_id,
                        "domain": source_ref.domain,
                        "tables": schema_tables,
                    }
                )
        except Exception as exc:  # noqa: BLE001 - batch extraction should continue
            failures.append({"source": source, "error": f"{type(exc).__name__}: {exc}"})

    _write_parquet(outputs.parquet, parquet_rows)
    _write_jsonl(outputs.schemas, schema_docs)
    _write_jsonl(outputs.manifest, manifest_rows)

    report = {
        "source_count": len(sources),
        "written_tables": len(parquet_rows),
        "schema_documents": len(schema_docs),
        "failures": failures,
        "failure_count": len(failures),
        "extensions": dict(sorted(extension_counts.items())),
        "outputs": {
            "parquet": str(outputs.parquet),
            "schemas": str(outputs.schemas),
            "manifest": str(outputs.manifest),
            "report": str(outputs.report),
            "tables_dir": str(outputs.tables_dir),
        },
    }
    outputs.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    schema = pa.schema(
        [
            ("dataset_uri", pa.string()),
            ("metadata", pa.string()),
            ("content", pa.string()),
            ("source_uri", pa.string()),
            ("dataset_id", pa.string()),
            ("domain", pa.string()),
            ("table_name", pa.string()),
            ("local_table_path", pa.string()),
        ]
    )
    columns = {name: [row.get(name) for row in rows] for name in schema.names}
    table = pa.Table.from_pydict(columns, schema=schema)
    pq.write_table(table, path, compression="zstd")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=DEFAULT_EVAL_ROOT,
        help="Path to the exploratory-qa-eval repo containing tasks/plans and raw Kramabench data.",
    )
    parser.add_argument("--tasks-root", type=Path, default=None)
    parser.add_argument("--plans-root", type=Path, default=None)
    parser.add_argument(
        "--source-list",
        type=Path,
        default=None,
        help="Optional txt/jsonl source list. When set, tasks/plans are not scanned.",
    )
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--output-parquet", type=Path, default=DEFAULT_OUTPUT_PARQUET)
    parser.add_argument("--output-schemas", type=Path, default=DEFAULT_OUTPUT_SCHEMAS)
    parser.add_argument("--output-manifest", type=Path, default=DEFAULT_OUTPUT_MANIFEST)
    parser.add_argument("--output-report", type=Path, default=DEFAULT_OUTPUT_REPORT)
    parser.add_argument("--tables-dir", type=Path, default=DEFAULT_TABLES_DIR)
    parser.add_argument(
        "--max-sample-words",
        type=int,
        default=DEFAULT_MAX_SAMPLE_WORDS,
        help="Maximum CSV sample words to store in each parquet content field.",
    )
    parser.add_argument(
        "--max-sample-rows",
        type=int,
        default=None,
        help="Optional row cap before applying --max-sample-words.",
    )
    parser.add_argument("--limit-sources", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    eval_root = args.eval_root.resolve()
    if args.source_list is not None:
        sources = collect_source_refs_from_file(args.source_list, bucket=args.bucket)
    else:
        tasks_root = args.tasks_root or eval_root / "tasks-mini-kramabench"
        plans_root = args.plans_root or eval_root / "plans-mini-kramabench"
        sources = collect_source_refs([tasks_root, plans_root])
    if args.limit_sources is not None:
        sources = sources[: args.limit_sources]
    outputs = OutputPaths(
        parquet=args.output_parquet,
        schemas=args.output_schemas,
        manifest=args.output_manifest,
        report=args.output_report,
        tables_dir=args.tables_dir,
    )
    report = extract_sources(
        sources,
        eval_root=eval_root,
        outputs=outputs,
        bucket=args.bucket,
        max_sample_words=args.max_sample_words,
        max_sample_rows=args.max_sample_rows,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["written_tables"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
