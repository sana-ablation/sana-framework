"""Standard computation tools — the non-oracle arm of the computation axis.

``query_file`` and ``execute_code`` are the two tools the computation axis
swaps: ``computation=standard`` binds the pair defined here, and
``computation=ideal`` binds ``query_ideal``/``execute_ideal`` from
``tools.computation.oracle``. Keeping both arms in this package means the axis
reads as a set of interchangeable implementations rather than one arm living in
the lake substrate and the other in an oracle directory.

This module owns the *tool surface*: the ``@tool`` decorations, the signatures,
the docstrings the model reads, and the timeout/error envelope each tool wraps
its implementation in. The retrieval, DuckDB, and sandbox machinery underneath
stays in ``tools.lake``, which the eleven other lake tools also depend on, and
is imported back here. ``tools.computation.oracle`` imports from ``lake`` the
same way.
"""

from __future__ import annotations

from typing import Any, Dict

from strands import tool

from sana_evaluation.tools.lake import (
    _collect_sandbox_snapshot,
    _execute_code_impl,
    _get_sandbox_dir,
    _query_file_impl,
    _run_tool_with_timeout,
    _tool_timeout_seconds,
)


@tool
def execute_code(code: str) -> Dict[str, Any]:
    """
    Execute Python code in a sandbox environment with access to downloaded files.
    Use only for tabular or JSON-like sources. Do not use execute_code to
    parse Wikipedia/content.txt, prose/plain text, XML/KML, HTML, PDFs, binary
    files, or other non-tabular/non-JSON sources. Use parse_xml_records for
    XML/KML structured records.

    The code runs with:
    - Working directory set to the sandbox directory
    - Pre-imported: pandas, json, csv, os, glob, re, pathlib, ijson
    - Variable `SANDBOX_DIR` pointing to the sandbox directory (also available
      as os.environ['SANDBOX_DIR'])
    - Variable `FILES` containing list of downloaded file paths
    - Variable `DOWNLOAD_MANIFEST_PATH` and env var of the same name pointing
      to `.download_manifest.json`
    - Variable `DOWNLOAD_PATHS` containing the manifest `path_map`

    Write your analysis code and print() results. The printed output will be returned.
    Do not use this tool to view or extract facts from non-tabular text files;
    use read_file or grep_file for those sources.
    Note: execution has a timeout; avoid inefficient code.

    For large JSON files (100+ MB), use the pre-imported `ijson` module to
    stream-parse without loading everything into memory:
        for feat in ijson.items(open(path, 'rb'), 'features.item'): ...

    Args:
        code: Python code to execute

    Returns:
        Dict with 'output' (stdout), 'error' (if any), 'success' (bool)

    Example:
        >>> execute_code('''
        ... import pandas as pd
        ... df = pd.read_csv(SANDBOX_DIR + "/wikipedia/Barack_Obama/table_0.csv")
        ... print(df.head())
        ... print(f"Total rows: {len(df)}")
        ... ''')
        {'output': '   col1  col2\\n...\\nTotal rows: 50', 'success': True}
    """
    timeout_seconds = _tool_timeout_seconds()
    completed, result = _run_tool_with_timeout(_execute_code_impl, code, timeout_seconds=timeout_seconds)
    if completed:
        return result or {'error': "execute_code failed without returning a result", 'success': False}

    sandbox = _get_sandbox_dir()
    sandbox_files, datasets_in_sandbox = _collect_sandbox_snapshot(sandbox)
    return {
        'output': '',
        'error': (
            f"execute_code timed out after {timeout_seconds}s. "
            "Use a narrower script, stream large files, or print less intermediate data."
        ),
        'success': False,
        'sandbox_dir': str(sandbox),
        'sandbox_files': sandbox_files,
        'datasets_in_sandbox': datasets_in_sandbox,
    }


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


__all__ = ["execute_code", "query_file"]
