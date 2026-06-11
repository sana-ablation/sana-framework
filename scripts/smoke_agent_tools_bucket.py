#!/usr/bin/env python3
"""Smoke-test agent data tools against a LakeQA-shaped S3 bucket.

The script intentionally imports tool modules lazily so dependency/import
failures can be written to test_logs instead of failing before logs exist.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import sys
import time
import traceback
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from pprint import pformat
from types import ModuleType
from typing import Any, Callable, Dict, Iterable, List, Optional


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOG_DIR = REPO_ROOT / "test_logs" / "agent_tools_bucket_smoke"
RESULT_CHAR_CAP = 20_000

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BUCKETS = {
    "lakeqa": "lakeqa-yc4103-datalake",
    "kramabench": "sana-kramabench",
}

DEFAULT_TARGETS = {
    "kramabench": {
        "dataset_id": "kramabench-archeology-easy-10",
        "file_path": "files/worldcities.csv",
        "folder": "datagov",
        "search_prefix": "kramabench-archeology",
        "keyword": "kramabench",
        "regex_pattern": "Singapore",
        "query_sql": (
            "SELECT country, COUNT(*) AS cities "
            "FROM t "
            "WHERE country IS NOT NULL "
            "GROUP BY country "
            "ORDER BY cities DESC "
            "LIMIT 5"
        ),
    },
    "lakeqa": {
        "dataset_id": "index-crimes-by-county",
        "file_path": "files/rows.txt",
        "folder": "datagov",
        "search_prefix": "index-crimes",
        "keyword": "crime",
        "regex_pattern": "county",
        "query_sql": "SELECT * FROM t LIMIT 5",
    },
}


@dataclass(frozen=True)
class Target:
    benchmark: str
    bucket: str
    folder: str
    dataset_id: str
    file_path: str
    s3_uri: str
    search_prefix: str
    keyword: str
    regex_pattern: str
    query_sql: str


@dataclass(frozen=True)
class ToolCase:
    module_name: str
    tool_name: str
    call: Callable[[], Any]
    call_repr: Optional[str] = None
    expect_error_substring: Optional[str] = None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def install_dependency_stubs(names: Iterable[str]) -> List[str]:
    """Install tiny runtime stubs for decorator-only/heavy deps when requested.

    This is meant for local bucket smoke checks in lean environments. It should
    not be used to validate full agent integration.
    """
    installed: List[str] = []
    requested = set(names)

    if "strands" in requested and importlib.util.find_spec("strands") is None:
        strands = ModuleType("strands")

        def tool(fn: Optional[Callable[..., Any]] = None, *args: Any, **kwargs: Any) -> Any:
            if fn is None:
                return lambda real_fn: real_fn
            return fn

        strands.tool = tool  # type: ignore[attr-defined]
        sys.modules["strands"] = strands
        installed.append("strands")

    if "duckdb" in requested and importlib.util.find_spec("duckdb") is None:
        duckdb = ModuleType("duckdb")

        class DuckDBPyConnection:
            pass

        def connect(*args: Any, **kwargs: Any) -> Any:
            raise ImportError("duckdb is not installed; install duckdb to fully test query_file")

        duckdb.DuckDBPyConnection = DuckDBPyConnection  # type: ignore[attr-defined]
        duckdb.connect = connect  # type: ignore[attr-defined]
        sys.modules["duckdb"] = duckdb
        installed.append("duckdb")

    return installed


def install_lightweight_package_stubs() -> None:
    """Let this script import tool modules without executing package __init__."""
    package_specs = {
        "sana_evaluation": REPO_ROOT / "sana_evaluation",
        "sana_evaluation.tools": REPO_ROOT / "sana_evaluation" / "tools",
    }
    for name, path in package_specs.items():
        if name in sys.modules:
            continue
        module = ModuleType(name)
        module.__path__ = [str(path)]  # type: ignore[attr-defined]
        sys.modules[name] = module


def build_target(
    benchmark: str,
    *,
    bucket: Optional[str] = None,
    dataset_id: Optional[str] = None,
    file_path: Optional[str] = None,
    folder: Optional[str] = None,
    s3_uri: Optional[str] = None,
    search_prefix: Optional[str] = None,
    keyword: Optional[str] = None,
    regex_pattern: Optional[str] = None,
    query_sql: Optional[str] = None,
) -> Target:
    normalized = benchmark.strip().lower()
    if normalized not in DEFAULT_TARGETS:
        expected = ", ".join(sorted(DEFAULT_TARGETS))
        raise ValueError(f"Unsupported benchmark '{benchmark}'. Expected one of: {expected}")

    defaults = DEFAULT_TARGETS[normalized]
    resolved_bucket = bucket or BUCKETS[normalized]
    resolved_folder = folder or defaults["folder"]
    resolved_dataset_id = dataset_id or defaults["dataset_id"]
    resolved_file_path = (file_path or defaults["file_path"]).lstrip("/")
    resolved_s3_uri = (
        s3_uri
        or f"s3://{resolved_bucket}/{resolved_folder}/{resolved_dataset_id}/{resolved_file_path}"
    )

    return Target(
        benchmark=normalized,
        bucket=resolved_bucket,
        folder=resolved_folder,
        dataset_id=resolved_dataset_id,
        file_path=resolved_file_path,
        s3_uri=resolved_s3_uri,
        search_prefix=search_prefix or defaults["search_prefix"],
        keyword=keyword or defaults["keyword"],
        regex_pattern=regex_pattern or defaults["regex_pattern"],
        query_sql=query_sql or defaults["query_sql"],
    )


def _truncate_text(value: str, cap: int = RESULT_CHAR_CAP) -> str:
    if len(value) <= cap:
        return value
    return value[:cap] + f"... <truncated {len(value) - cap} chars>"


def sanitize_for_json(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return _truncate_text(value.decode("utf-8", errors="replace"))
    if isinstance(value, str):
        return _truncate_text(value)
    if isinstance(value, dict):
        return {str(k): sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        items = [sanitize_for_json(v) for v in value[:100]]
        if len(value) > 100:
            items.append(f"<truncated {len(value) - 100} list items>")
        return items
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)


def run_case(case: ToolCase) -> Dict[str, Any]:
    started = time.monotonic()
    record: Dict[str, Any] = {
        "module": case.module_name,
        "tool": case.tool_name,
        "call": case.call_repr or f"{case.module_name}.{case.tool_name}()",
        "started_at": utc_now_iso(),
    }

    try:
        result = case.call()
        error = result.get("error") if isinstance(result, dict) else None
        if error:
            if case.expect_error_substring and case.expect_error_substring in str(error):
                record["status"] = "passed"
                record["expected_error"] = True
            else:
                record["status"] = "failed"
                record["error"] = str(error)
        else:
            record["status"] = "passed"
            record["expected_error"] = False
        record["result"] = sanitize_for_json(result)
    except Exception as exc:  # noqa: BLE001 - this is a smoke harness
        record["status"] = "failed"
        record["expected_error"] = False
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc()

    record["duration_seconds"] = round(time.monotonic() - started, 3)
    record.setdefault("expected_error", False)
    return record


def _file_spec(target: Target) -> List[Dict[str, str]]:
    return [{"dataset_id": target.dataset_id, "file_path": target.file_path, "s3_uri": target.s3_uri}]


def _execute_code_snippet() -> str:
    return (
        "import os\n"
        "print('sandbox_dir=' + SANDBOX_DIR)\n"
        "print('downloaded_files=' + str(len(FILES)))\n"
        "print('file_names=' + ','.join(os.path.basename(p) for p in FILES[:5]))\n"
    )


def build_agent_tools_cases(agent_tools: Any, target: Target) -> List[ToolCase]:
    return [
        ToolCase(
            "agent_tools",
            "configure_benchmark",
            lambda: agent_tools.configure_benchmark(target.benchmark),
            f"agent_tools.configure_benchmark({target.benchmark!r})",
        ),
        ToolCase(
            "agent_tools",
            "search",
            lambda: agent_tools.search([target.search_prefix], limit=5),
            f"agent_tools.search({[target.search_prefix]!r}, limit=5)",
        ),
        ToolCase(
            "agent_tools",
            "search_prefix",
            lambda: agent_tools.search_prefix([target.search_prefix], limit=5),
            f"agent_tools.search_prefix({[target.search_prefix]!r}, limit=5)",
        ),
        ToolCase(
            "agent_tools",
            "search_keyword",
            lambda: agent_tools.search_keyword([target.keyword], limit=5),
            f"agent_tools.search_keyword({[target.keyword]!r}, limit=5)",
        ),
        ToolCase(
            "agent_tools",
            "list_files",
            lambda: agent_tools.list_files([target.dataset_id], limit=10),
            f"agent_tools.list_files({[target.dataset_id]!r}, limit=10)",
        ),
        ToolCase(
            "agent_tools",
            "inspect_file",
            lambda: agent_tools.inspect_file(target.dataset_id, target.file_path, max_lines=5),
            f"agent_tools.inspect_file({target.dataset_id!r}, {target.file_path!r}, max_lines=5)",
        ),
        ToolCase(
            "agent_tools",
            "download",
            lambda: agent_tools.download(_file_spec(target)),
            f"agent_tools.download({_file_spec(target)!r})",
        ),
        ToolCase("agent_tools", "get_sandbox_info", agent_tools.get_sandbox_info, "agent_tools.get_sandbox_info()"),
        ToolCase(
            "agent_tools",
            "execute_code",
            lambda: agent_tools.execute_code(_execute_code_snippet()),
            f"agent_tools.execute_code({_execute_code_snippet()!r})",
        ),
        ToolCase(
            "agent_tools",
            "submit_answer",
            lambda: agent_tools.submit_answer("[smoke-test]", "tool smoke test"),
            "agent_tools.submit_answer('[smoke-test]', 'tool smoke test')",
        ),
        ToolCase("agent_tools", "cleanup_sandbox", agent_tools.cleanup_sandbox, "agent_tools.cleanup_sandbox()"),
    ]


def build_agent_tools_v2_cases(agent_tools_v2: Any, target: Target) -> List[ToolCase]:
    return [
        ToolCase(
            "agent_tools_v2",
            "configure_benchmark",
            lambda: agent_tools_v2.configure_benchmark(target.benchmark),
            f"agent_tools_v2.configure_benchmark({target.benchmark!r})",
        ),
        ToolCase(
            "agent_tools_v2",
            "search",
            lambda: agent_tools_v2.search([target.search_prefix], limit=5),
            f"agent_tools_v2.search({[target.search_prefix]!r}, limit=5)",
        ),
        ToolCase(
            "agent_tools_v2",
            "search_keyword",
            lambda: agent_tools_v2.search_keyword([target.keyword], limit=5),
            f"agent_tools_v2.search_keyword({[target.keyword]!r}, limit=5)",
        ),
        ToolCase(
            "agent_tools_v2",
            "list_files",
            lambda: agent_tools_v2.list_files([target.dataset_id], limit=10),
            f"agent_tools_v2.list_files({[target.dataset_id]!r}, limit=10)",
        ),
        ToolCase(
            "agent_tools_v2",
            "peek_file",
            lambda: agent_tools_v2.peek_file(s3_uri=target.s3_uri, max_rows=5),
            f"agent_tools_v2.peek_file(s3_uri={target.s3_uri!r}, max_rows=5)",
        ),
        ToolCase(
            "agent_tools_v2",
            "peek_multiple",
            lambda: agent_tools_v2.peek_multiple(files=_file_spec(target), max_rows=5),
            f"agent_tools_v2.peek_multiple(files={_file_spec(target)!r}, max_rows=5)",
        ),
        ToolCase(
            "agent_tools_v2",
            "read_file",
            lambda: agent_tools_v2.read_file(s3_uri=target.s3_uri, start_line=0, max_lines=5),
            f"agent_tools_v2.read_file(s3_uri={target.s3_uri!r}, start_line=0, max_lines=5)",
        ),
        ToolCase(
            "agent_tools_v2",
            "grep_file",
            lambda: agent_tools_v2.grep_file(s3_uri=target.s3_uri, regex_pattern=target.regex_pattern, context_lines=1),
            (
                f"agent_tools_v2.grep_file(s3_uri={target.s3_uri!r}, "
                f"regex_pattern={target.regex_pattern!r}, context_lines=1)"
            ),
        ),
        ToolCase(
            "agent_tools_v2",
            "parse_xml_records",
            lambda: agent_tools_v2.parse_xml_records(s3_uri=target.s3_uri, limit=5),
            f"agent_tools_v2.parse_xml_records(s3_uri={target.s3_uri!r}, limit=5)",
            expect_error_substring="XML/KML",
        ),
        ToolCase(
            "agent_tools_v2",
            "query_file",
            lambda: agent_tools_v2.query_file(s3_uri=target.s3_uri, sql=target.query_sql),
            f"agent_tools_v2.query_file(s3_uri={target.s3_uri!r}, sql={target.query_sql!r})",
        ),
        ToolCase(
            "agent_tools_v2",
            "download",
            lambda: agent_tools_v2.download(_file_spec(target)),
            f"agent_tools_v2.download({_file_spec(target)!r})",
        ),
        ToolCase("agent_tools_v2", "get_sandbox_info", agent_tools_v2.get_sandbox_info, "agent_tools_v2.get_sandbox_info()"),
        ToolCase(
            "agent_tools_v2",
            "execute_code",
            lambda: agent_tools_v2.execute_code(_execute_code_snippet()),
            f"agent_tools_v2.execute_code({_execute_code_snippet()!r})",
        ),
        ToolCase("agent_tools_v2", "cleanup_sandbox", agent_tools_v2.cleanup_sandbox, "agent_tools_v2.cleanup_sandbox()"),
    ]


def import_tool_modules(module_names: Iterable[str]) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    modules: Dict[str, Any] = {}
    records: List[Dict[str, Any]] = []
    install_lightweight_package_stubs()
    import_paths = {
        "agent_tools": "sana_evaluation.tools.agent_tools",
        "agent_tools_v2": "sana_evaluation.tools.agent_tools_v2",
    }
    for module_name in module_names:
        started = time.monotonic()
        record: Dict[str, Any] = {
            "module": module_name,
            "tool": "__import__",
            "call": f"importlib.import_module({import_paths[module_name]!r})",
            "started_at": utc_now_iso(),
        }
        try:
            modules[module_name] = importlib.import_module(import_paths[module_name])
            record["status"] = "passed"
            record["expected_error"] = False
            record["result"] = {"module_path": import_paths[module_name]}
        except Exception as exc:  # noqa: BLE001 - dependency failures are logged
            record["status"] = "failed"
            record["expected_error"] = False
            record["error"] = f"{type(exc).__name__}: {exc}"
            record["traceback"] = traceback.format_exc()
        record["duration_seconds"] = round(time.monotonic() - started, 3)
        records.append(record)
    return modules, records


def format_transcript(*, records: List[Dict[str, Any]], metadata: Dict[str, Any]) -> str:
    counts = Counter(record.get("status", "unknown") for record in records)
    lines = [
        "Agent Tools Bucket Smoke Test",
        "=" * 80,
        "",
        "TARGET:",
    ]
    for key, value in metadata.items():
        lines.append(f"  {key}: {value}")
    lines.extend([
        "",
        "SUMMARY:",
        f"  passed: {counts.get('passed', 0)}",
        f"  failed: {counts.get('failed', 0)}",
        f"  total: {sum(counts.values())}",
        "",
    ])

    for index, record in enumerate(records, start=1):
        status = str(record.get("status", "unknown")).upper()
        expected = " (expected error)" if record.get("expected_error") else ""
        lines.extend([
            "-" * 80,
            f"{index}. {record.get('module')}.{record.get('tool')}",
            f"STATUS: {status}{expected}",
            f"DURATION: {record.get('duration_seconds', 0)}s",
            "CALL:",
            f"  {record.get('call', record.get('tool'))}",
        ])
        if record.get("error"):
            lines.extend(["ERROR:", pformat(record["error"], width=100)])
        if "result" in record:
            lines.extend(["RETURNED:", pformat(record["result"], width=100, sort_dicts=False)])
        if record.get("traceback"):
            lines.extend(["TRACEBACK:", str(record["traceback"])])
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_logs(
    *,
    records: List[Dict[str, Any]],
    log_dir: Path,
    run_label: str,
    metadata: Dict[str, Any],
) -> Dict[str, Path]:
    run_dir = log_dir / run_label
    run_dir.mkdir(parents=True, exist_ok=True)
    transcript_path = run_dir / "tool_calls.log"
    transcript_path.write_text(
        format_transcript(records=records, metadata=metadata),
        encoding="utf-8",
    )
    return {"run_dir": run_dir, "transcript_path": transcript_path}


def run_smoke_tests(
    *,
    target: Target,
    log_dir: Path,
    run_label: str,
    modules_to_test: List[str],
    dependency_stubs: Optional[List[str]] = None,
) -> Dict[str, Any]:
    records: List[Dict[str, Any]] = []
    installed_stubs = install_dependency_stubs(dependency_stubs or [])
    modules, import_records = import_tool_modules(modules_to_test)
    records.extend(import_records)

    sandbox_dir = log_dir / run_label / "sandbox"
    for module in modules.values():
        if hasattr(module, "set_sandbox_dir"):
            module.set_sandbox_dir(sandbox_dir)

    cases: List[ToolCase] = []
    if "agent_tools" in modules:
        cases.extend(build_agent_tools_cases(modules["agent_tools"], target))
    if "agent_tools_v2" in modules:
        cases.extend(build_agent_tools_v2_cases(modules["agent_tools_v2"], target))

    for case in cases:
        record = run_case(case)
        records.append(record)
        status = record["status"].upper()
        marker = " (expected error)" if record.get("expected_error") else ""
        print(f"[{status}] {record['module']}.{record['tool']}{marker}")

    metadata = {
        "benchmark": target.benchmark,
        "bucket": target.bucket,
        "folder": target.folder,
        "dataset_id": target.dataset_id,
        "file_path": target.file_path,
        "s3_uri": target.s3_uri,
        "modules": modules_to_test,
        "dependency_stubs": installed_stubs,
        "started_at": records[0]["started_at"] if records else utc_now_iso(),
        "finished_at": utc_now_iso(),
        "sandbox_dir": str(sandbox_dir),
    }
    paths = write_logs(records=records, log_dir=log_dir, run_label=run_label, metadata=metadata)
    counts = Counter(record.get("status", "unknown") for record in records)
    return {"records": records, "paths": paths, "counts": dict(counts)}


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=sorted(DEFAULT_TARGETS), default="kramabench")
    parser.add_argument("--bucket", help="Override the benchmark bucket name")
    parser.add_argument("--folder", choices=["datagov", "wikipedia"], help="Override S3 top-level folder")
    parser.add_argument("--dataset-id", help="Dataset id to test")
    parser.add_argument("--file-path", help="Relative file path inside the dataset")
    parser.add_argument("--s3-uri", help="Full S3 URI to test; overrides bucket/folder/dataset/file composition")
    parser.add_argument("--search-prefix", help="Prefix for search/search_prefix")
    parser.add_argument("--keyword", help="Keyword for search_keyword")
    parser.add_argument("--regex-pattern", help="Pattern for grep_file")
    parser.add_argument("--query-sql", help="SQL to run through query_file using table alias t")
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--run-label", help="Directory name under --log-dir")
    parser.add_argument(
        "--module",
        choices=["agent_tools", "agent_tools_v2"],
        action="append",
        dest="modules",
        help="Module to test. Repeat to test both. Defaults to both.",
    )
    parser.add_argument(
        "--s3-access-mode",
        choices=["auto", "signed", "unsigned", "public", "anonymous", "no-sign-request"],
        help="Set S3_ACCESS_MODE for the tool run",
    )
    parser.add_argument(
        "--allow-failures",
        action="store_true",
        help="Exit 0 even if one or more tool checks fail. Logs still record failures.",
    )
    parser.add_argument(
        "--allow-dependency-stubs",
        action="store_true",
        help=(
            "Install lightweight local stubs for missing strands.tool and duckdb imports. "
            "This lets non-DuckDB tool functions run in lean local environments; query_file "
            "will still fail if duckdb is only stubbed."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args.s3_access_mode:
        os.environ["S3_ACCESS_MODE"] = args.s3_access_mode

    target = build_target(
        args.benchmark,
        bucket=args.bucket,
        dataset_id=args.dataset_id,
        file_path=args.file_path,
        folder=args.folder,
        s3_uri=args.s3_uri,
        search_prefix=args.search_prefix,
        keyword=args.keyword,
        regex_pattern=args.regex_pattern,
        query_sql=args.query_sql,
    )
    run_label = args.run_label or f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{target.benchmark}"
    modules_to_test = args.modules or ["agent_tools", "agent_tools_v2"]

    result = run_smoke_tests(
        target=target,
        log_dir=args.log_dir,
        run_label=run_label,
        modules_to_test=modules_to_test,
        dependency_stubs=["strands", "duckdb"] if args.allow_dependency_stubs else [],
    )
    counts = result["counts"]
    paths = result["paths"]
    failed = counts.get("failed", 0)

    print()
    print(f"Summary: {counts}")
    print(f"Tool-call transcript: {paths['transcript_path']}")

    if failed and not args.allow_failures:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
