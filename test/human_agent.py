#!/usr/bin/env python3
"""
Interactive human-agent REPL.

Lets you manually run a benchmark task using the same tools the LLM agent has:
  search, search_keyword, list_files, peek_file, peek_multiple, read_file,
  grep_file, query_file, download, execute_code, submit_answer
  sparse, hybrid, graph  (search backends)

Usage:
    python human_agent.py                        # pick a random task
    python human_agent.py --task tasks/k-3-d-3/task_3.json
    python human_agent.py --question "What is ..."
"""

import argparse
import csv
import json
import random
import re
import sys
import textwrap
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap path so we can import from repo-local sana_evaluation/
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from sana_evaluation.tools.agent_tools_v2 import (  # noqa: E402
    search,
    search_keyword,
    list_files,
    peek_file, peek_multiple, read_file, grep_file, query_file,
    download,
    execute_code,
    get_sandbox_info,
    set_sandbox_dir,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pp(obj, indent=2):
    """Pretty-print a dict/list."""
    print(json.dumps(obj, indent=indent, default=str))


def _banner(text, width=70, char="="):
    print(char * width)
    for line in textwrap.wrap(text, width - 4):
        print(f"  {line}")
    print(char * width)


def _short(obj, max_chars=2000):
    """Truncate long output for display."""
    s = json.dumps(obj, indent=2, default=str)
    if len(s) > max_chars:
        return s[:max_chars] + f"\n... [{len(s) - max_chars} chars truncated]"
    return s


def _parse_csv_input(raw: str) -> list[str]:
    """
    Parse a comma-separated input line with CSV semantics.

    This allows entries containing commas when quoted, e.g.
    "Austin,_Texas",vsrr-provisional-drug-overdose-death-counts
    """
    raw = (raw or "").strip()
    if not raw:
        return []
    try:
        row = next(csv.reader([raw], skipinitialspace=True))
    except Exception:
        return [raw] if raw else []
    return [item.strip() for item in row if item and item.strip()]


# ---------------------------------------------------------------------------
# Tool wrappers with friendly I/O
# ---------------------------------------------------------------------------

def cmd_search():
    raw = input("  prefixes (comma-separated; quote items containing commas): ").strip()
    prefixes = _parse_csv_input(raw)
    if not prefixes:
        print("  [!] No prefixes given.")
        return
    t0 = time.time()
    result = search(prefixes)
    print(f"  [{time.time()-t0:.1f}s]")
    print(_short(result))


def cmd_search_keyword():
    raw = input("  keywords (comma-separated; quote items containing commas): ").strip()
    keywords = _parse_csv_input(raw)
    if not keywords:
        print("  [!] No keywords given.")
        return
    limit_raw = input("  limit [20]: ").strip()
    limit = int(limit_raw) if limit_raw.isdigit() else 20
    t0 = time.time()
    result = search_keyword(keywords, limit=limit)
    print(f"  [{time.time()-t0:.1f}s]")
    print(_short(result))


def cmd_list_files():
    raw = input("  dataset_ids (comma-separated; quote IDs containing commas): ").strip()
    ids = _parse_csv_input(raw)
    if not ids:
        print("  [!] No dataset ids given.")
        return
    t0 = time.time()
    result = list_files(ids)
    print(f"  [{time.time()-t0:.1f}s]")
    print(_short(result))


def cmd_peek_file():
    dataset_id = input("  dataset_id: ").strip()
    file_path = input("  file_path: ").strip()
    if not dataset_id or not file_path:
        print("  [!] Both dataset_id and file_path required.")
        return
    t0 = time.time()
    result = peek_file(dataset_id, file_path)
    print(f"  [{time.time()-t0:.1f}s]")
    print(_short(result))


def cmd_peek_multiple():
    print("  Enter files to peek. Empty dataset_id to stop.")
    files = []
    while True:
        d = input("    dataset_id: ").strip()
        if not d:
            break
        f = input("    file_path: ").strip()
        if f:
            files.append({"dataset_id": d, "file_path": f})
    if not files:
        print("  [!] Nothing to peek.")
        return
    t0 = time.time()
    result = peek_multiple(files)
    print(f"  [{time.time()-t0:.1f}s]")
    print(_short(result))


def cmd_read_file():
    dataset_id = input("  dataset_id: ").strip()
    file_path = input("  file_path: ").strip()
    if not dataset_id or not file_path:
        print("  [!] Both dataset_id and file_path required.")
        return
    start_line_raw = input("  start_line [0]: ").strip()
    max_lines_raw = input("  max_lines [200]: ").strip()
    start_line = int(start_line_raw) if start_line_raw.isdigit() else 0
    max_lines = int(max_lines_raw) if max_lines_raw.isdigit() else 200
    t0 = time.time()
    result = read_file(dataset_id, file_path, start_line=start_line, max_lines=max_lines)
    print(f"  [{time.time()-t0:.1f}s]")
    print(_short(result))


def cmd_grep_file():
    dataset_id = input("  dataset_id: ").strip()
    file_path = input("  file_path: ").strip()
    pattern = input("  regex_pattern: ").strip()
    if not dataset_id or not file_path or not pattern:
        print("  [!] dataset_id, file_path, and regex_pattern required.")
        return
    t0 = time.time()
    result = grep_file(dataset_id, file_path, pattern)
    print(f"  [{time.time()-t0:.1f}s]")
    print(_short(result))


def cmd_query_file():
    dataset_id = input("  dataset_id: ").strip()
    file_path = input("  file_path: ").strip()
    sql = input("  sql (use 't' as table): ").strip()
    if not dataset_id or not file_path or not sql:
        print("  [!] dataset_id, file_path, and sql required.")
        return
    t0 = time.time()
    result = query_file(dataset_id, file_path, sql)
    print(f"  [{time.time()-t0:.1f}s]")
    print(_short(result))


def cmd_download():
    print("  Enter files to download. Empty dataset_id to stop.")
    files = []
    while True:
        d = input("    dataset_id: ").strip()
        if not d:
            break
        f = input("    file_path: ").strip()
        if f:
            files.append({"dataset_id": d, "file_path": f})
    if not files:
        print("  [!] Nothing to download.")
        return
    t0 = time.time()
    result = download(files)
    print(f"  [{time.time()-t0:.1f}s]")
    print(_short(result))


def cmd_execute_code():
    print("  Enter Python code. Type END on its own line to run.")
    lines = []
    while True:
        line = input("  >>> ")
        if line.strip() == "END":
            break
        lines.append(line)
    code = "\n".join(lines)
    if not code.strip():
        print("  [!] No code entered.")
        return
    t0 = time.time()
    result = execute_code(code)
    print(f"  [{time.time()-t0:.1f}s]")
    if "output" in result:
        print(result["output"])
    if "error" in result:
        print(f"  [ERROR] {result['error']}")
    other = {k: v for k, v in result.items() if k not in ("output", "error")}
    if other:
        _pp(other)


def cmd_sandbox_info():
    _pp(get_sandbox_info())


def cmd_search_sparse():
    query = input("  query: ").strip()
    if not query:
        print("  [!] Query required.")
        return
    top_k_raw = input("  top_k [10]: ").strip()
    top_k = int(top_k_raw) if top_k_raw.isdigit() else 10
    try:
        # Naive sparse backend.
        from sana_evaluation.tools.external.search_naive_tools import search_value as search_sparse
    except Exception as e:
        print(
            "  [ERROR] sparse backend unavailable "
            "(missing external-tools/hybrid_search/api.py or its deps): "
            f"{e}"
        )
        return
    t0 = time.time()
    result = search_sparse(query=query, top_k=top_k)
    print(f"  [{time.time()-t0:.1f}s]")
    print(_short(result))


def cmd_search_hybrid():
    query = input("  query: ").strip()
    if not query:
        print("  [!] Query required.")
        return
    top_k_raw = input("  top_k [10]: ").strip()
    top_k = int(top_k_raw) if top_k_raw.isdigit() else 10
    try:
        # Standard hybrid backend.
        from sana_evaluation.tools.external.search_standard_tools import search_value as search_hybrid
    except Exception as e:
        print(
            "  [ERROR] hybrid backend unavailable "
            "(missing external-tools/hybrid_search/api.py or its deps): "
            f"{e}"
        )
        return
    t0 = time.time()
    result = search_hybrid(query=query, top_k=top_k)
    print(f"  [{time.time()-t0:.1f}s]")
    print(_short(result))


def cmd_search_graph():
    query = input("  query: ").strip()
    if not query:
        print("  [!] Query required.")
        return
    try:
        # Legacy optional backend; not present in all checkouts.
        from sana_evaluation.tools.external.search_tools import search_graph
    except Exception as e:
        print(
            "  [ERROR] graph backend unavailable "
            "(legacy module sana_evaluation.tools.external.search_tools is missing): "
            f"{e}"
        )
        return
    t0 = time.time()
    result = search_graph(query=query)
    print(f"  [{time.time()-t0:.1f}s]")
    print(_short(result))


# ---------------------------------------------------------------------------
# Command dispatch
# ---------------------------------------------------------------------------

COMMANDS = {
    "search":   ("search datasets by prefix",                      cmd_search),
    "kw":       ("search_keyword — keyword/FTS search",            cmd_search_keyword),
    "ls":       ("list_files in dataset(s)",                       cmd_list_files),
    "peek":     ("peek_file — structure/sample preview",           cmd_peek_file),
    "peeks":    ("peek_multiple — batch structure/sample preview", cmd_peek_multiple),
    "read":     ("read_file — paginated line reader",              cmd_read_file),
    "grep":     ("grep_file — regex search over S3 stream",         cmd_grep_file),
    "query":    ("query_file — SQL query via DuckDB httpfs",        cmd_query_file),
    "dl":       ("download file(s) to sandbox",                    cmd_download),
    "run":      ("execute_code in sandbox",                        cmd_execute_code),
    "sandbox":  ("show sandbox info / downloaded files",           cmd_sandbox_info),
    "sparse":   ("search_sparse — BM25/SPLADE sparse search",      cmd_search_sparse),
    "hybrid":   ("search_hybrid — hybrid dense+sparse + rerank",   cmd_search_hybrid),
    "graph":    ("search_graph — knowledge-graph semantic search",  cmd_search_graph),
}


def print_help():
    print("\nAvailable commands:")
    for cmd, (desc, _) in COMMANDS.items():
        print(f"  {cmd:<12}  {desc}")
    print(f"  {'submit':<12}  submit your final answer")
    print(f"  {'skip':<12}  skip this task (reveal answer)")
    print(f"  {'help':<12}  show this help")
    print(f"  {'quit':<12}  exit\n")


# ---------------------------------------------------------------------------
# Task loading
# ---------------------------------------------------------------------------

def load_task(task_file: str) -> dict:
    with open(task_file) as f:
        return json.load(f)


def pick_random_task() -> str:
    tasks = list(Path("tasks_mini").rglob("*.json"))
    if not tasks:
        sys.exit("No tasks found in tasks/")
    return str(random.choice(tasks))


# ---------------------------------------------------------------------------
# Main REPL
# ---------------------------------------------------------------------------

def run_repl(question: str, answer: str | None):
    sandbox = Path(_ROOT) / ".sandbox" / f"human_{int(time.time())}"
    set_sandbox_dir(sandbox)

    _banner(f"QUESTION:\n{question}")
    print_help()

    start = time.time()

    while True:
        try:
            raw = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[interrupted]")
            break

        if not raw:
            continue

        cmd = raw.lower()

        if cmd in ("quit", "exit", "q"):
            break

        elif cmd in ("help", "h", "?"):
            print_help()

        elif cmd == "skip":
            if answer:
                print(f"\n  Correct answer: {answer}")
            else:
                print("  No answer available for this task.")
            break

        elif cmd == "submit":
            ans = input("  Your answer: ").strip()
            elapsed = time.time() - start
            print(f"\n  Submitted: {ans}")
            print(f"  Time: {elapsed:.1f}s")
            if answer:
                def norm(s): return re.sub(r"[^a-z0-9]", "", s.lower())
                correct = norm(ans) == norm(answer)
                print(f"  Correct answer: {answer}")
                print(f"  Result: {'CORRECT' if correct else 'WRONG'}")
            break

        elif cmd in COMMANDS:
            _, fn = COMMANDS[cmd]
            try:
                fn()
            except Exception as e:
                print(f"  [ERROR] {e}")

        else:
            print(f"  Unknown command '{cmd}'. Type 'help' for options.")

    try:
        import shutil
        shutil.rmtree(sandbox, ignore_errors=True)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run a benchmark task as a human agent")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--task", help="Path to a task JSON file")
    group.add_argument("--question", help="Ad-hoc question to answer")
    args = parser.parse_args()

    if args.question:
        question = args.question
        answer = None
    else:
        task_file = args.task or pick_random_task()
        print(f"Task file: {task_file}")
        task = load_task(task_file)
        question = task["question"]
        answer = task.get("answer")

    run_repl(question, answer)


if __name__ == "__main__":
    main()
