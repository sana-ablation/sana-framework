"""Diagnostic: exercise the 3x2 (search_tool, search_results) matrix end-to-end.

No agent, no LLM — this script constructs the base search tools for each
search_tool mode, wraps them with ``build_search_tools`` for each search_results
mode, invokes each tool with a fixed query, and writes the raw wrapped payloads
to ``test_logs/search_matrix_<timestamp>.log`` for inspection.

Usage (from the repo root):

    PYTHONPATH=. python test/test_search_matrix.py
    PYTHONPATH=. python test/test_search_matrix.py --query "biological materials" --k 3
    PYTHONPATH=. python test/test_search_matrix.py --task-file benchmarks/lakeqa/tasks-mini/tasks/k-2-d-3/task_1.json

A combination that raises (e.g. because a heavy dependency like
``sentence_transformers`` is not installed for ``search_tool=standard``) is
captured as ``[FAIL]`` in the log and does not abort the rest of the matrix.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sana_evaluation.tools.agent_tools import search_prefix
from sana_evaluation.tools.external.ideal import search_ideal as ideal_search
from sana_evaluation.tools.external.ideal.search_wrapper import build_search_tools

SEARCH_TOOL_MODES = ("naive", "standard", "ideal")
SEARCH_RESULT_MODES = ("naive", "ideal")


def _build_base_tools(search_tool_mode: str, db_path: str, task_file: Optional[str]) -> List:
    """Return the un-wrapped base search tools for a given search_tool mode."""
    if search_tool_mode == "naive":
        from sana_evaluation.tools.external import search_naive_tools
        search_naive_tools.set_db_path(db_path)
        search_naive_tools.setup()
        return [search_naive_tools.search_value, search_naive_tools.search_schema, search_prefix]

    if search_tool_mode == "standard":
        from sana_evaluation.tools.external import search_standard_tools
        search_standard_tools.set_db_path(db_path)
        search_standard_tools.setup()
        return [search_standard_tools.search_value, search_standard_tools.search_schema, search_prefix]

    if search_tool_mode == "ideal":
        if not task_file:
            raise ValueError("search_tool=ideal requires --task-file")
        ideal_search.set_runtime_profiles_root("benchmarks/lakeqa/tasks-mini/runtime-profiles")
        ideal_search.reset_state()
        ideal_search.set_task_context({"task_id": task_file})
        return [ideal_search.search_ideal]

    raise ValueError(f"Unknown search_tool mode: {search_tool_mode}")


def _invoke_tool(tool, query: str) -> Dict[str, Any]:
    """Call a wrapped search tool with the right keyword args for its kind."""
    name = tool.tool_spec.get("name", "")
    if name == "search_prefix":
        return tool(prefixes=[query.strip()])
    return tool(query=query)


def _emit(log, line: str = "") -> None:
    log.write(line + "\n")
    print(line)


def _reset_ideal_state(task_file: str) -> None:
    """Reset search_ideal so every search_results combo sees the full candidate pool."""
    ideal_search.set_task_context({"task_id": task_file})


def run_matrix(
    *,
    db_path: str,
    query: str,
    task_file: str,
    k: int,
    log_path: Path,
    search_tools: Optional[Sequence[str]] = None,
) -> None:
    outer_modes = tuple(search_tools) if search_tools else SEARCH_TOOL_MODES
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        _emit(log, f"=== search 3x2 matrix — {datetime.now().isoformat()} ===")
        _emit(log, f"query     : {query!r}")
        _emit(log, f"db_path   : {db_path}")
        _emit(log, f"task_file : {task_file}  (only used when search_tool=ideal)")
        _emit(log, f"fixed k   : {k}")
        _emit(log, f"search_tools: {outer_modes}")

        for search_tool_mode in outer_modes:
            _emit(log)
            _emit(log, f"===================== search_tool = {search_tool_mode} =====================")
            try:
                base_tools = _build_base_tools(search_tool_mode, db_path, task_file)
            except Exception as exc:
                _emit(log, f"[FAIL] base tool setup for search_tool={search_tool_mode}: {exc}")
                _emit(log, traceback.format_exc())
                continue

            for search_results_mode in SEARCH_RESULT_MODES:
                _emit(log)
                _emit(log, f"----- search_tool={search_tool_mode}, search_results={search_results_mode} -----")
                try:
                    wrapped = build_search_tools(
                        base_tools,
                        fixed_k=k,
                        results_mode=search_results_mode,
                    )
                except Exception as exc:
                    _emit(log, f"[FAIL] build_search_tools: {exc}")
                    _emit(log, traceback.format_exc())
                    continue

                for tool in wrapped:
                    tool_name = tool.tool_spec.get("name", "?")
                    if search_tool_mode == "ideal":
                        _reset_ideal_state(task_file)
                    try:
                        payload = _invoke_tool(tool, query=query)
                    except Exception as exc:
                        _emit(log, f"[FAIL] {tool_name}: {exc}")
                        _emit(log, traceback.format_exc())
                        continue

                    count = payload.get("count") if isinstance(payload, dict) else None
                    _emit(log, f"[OK]   {tool_name} → count={count}")
                    _emit(log, json.dumps(payload, indent=2, default=str))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", default="lance_data", help="Lance DB root for non-ideal search tools.")
    parser.add_argument("--query", default="traffic events", help="Query fed to every search_value/search_schema call.")
    parser.add_argument("--task-file", default="benchmarks/lakeqa/tasks-mini/tasks/k-5-d-4/task_1.json", help="Task id used to set search_ideal context.")
    parser.add_argument("--k", type=int, default=3, help="Fixed result limit (passed as fixed_k to build_search_tools).")
    parser.add_argument("--out", default=None, help="Optional explicit output log path; defaults to test_logs/search_matrix_<ts>.log")
    parser.add_argument("--search-tool", action="append", choices=list(SEARCH_TOOL_MODES), help="Restrict outer search_tool modes; repeat to pass multiple. Defaults to all three.")
    args = parser.parse_args(argv)

    log_path = Path(args.out) if args.out else (
        Path("test_logs") / f"search_matrix_{datetime.now().strftime('%Y%m%dT%H%M%S')}.log"
    )

    run_matrix(
        db_path=args.db,
        query=args.query,
        task_file=args.task_file,
        k=args.k,
        log_path=log_path,
        search_tools=args.search_tool,
    )
    print(f"\nWrote matrix log to {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
