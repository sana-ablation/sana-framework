"""
TracePlugin — Strands Plugin that writes per-search-call JSONL traces.

One JSONL file per task: {output_dir}/{task_id_safe}.jsonl
Call set_trace_context(...) once per task before agent.run().
"""

import json
import os
import time
from pathlib import Path
from typing import Any, List

from strands import Plugin
from strands.hooks import AfterToolCallEvent, BeforeToolCallEvent
from strands.plugins import hook

# ---------------------------------------------------------------------------
# Module-level state (same pattern as set_sandbox_dir)
# ---------------------------------------------------------------------------

_current_task_id: str = ""
_current_gold_ids: List[str] = []
_current_output_dir: str = "results/traces"
_turn_counter: int = 0

_SEARCH_TOOLS = {
    "search_value",
    "search_schema",
    "search_reranked",
    "search_prefix",
    "search_ideal",
    "search",
    "search_keyword",
}
_S3_PREFIX = "s3://lakeqa-yc4103-datalake/"


def set_trace_context(task_id: str, gold_ids: List[str], output_dir: str) -> None:
    """Call once per task before agent.run() to wire up trace output."""
    global _current_task_id, _current_gold_ids, _current_output_dir, _turn_counter
    _current_task_id = task_id
    _current_gold_ids = list(gold_ids)
    _current_output_dir = output_dir
    _turn_counter = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_dataset_id(raw: str) -> str:
    """Strip S3 prefix and folder (datagov/wikipedia) to return the dataset ID."""
    s = str(raw)
    if _S3_PREFIX in s:
        s = s.split(_S3_PREFIX)[-1]
    elif "lakeqa-yc4103-datalake/" in s:
        s = s.split("lakeqa-yc4103-datalake/")[-1]
    # s is now e.g. "datagov/index-violent-.../files/rows.txt" or "wikipedia/Foo/content.txt"
    # skip the folder prefix (datagov, wikipedia, etc.) to get the dataset ID
    parts = s.split("/")
    return parts[1] if len(parts) > 1 else parts[0]


def _extract_dataset_ids(result_text: str) -> List[str]:
    """Parse tool result JSON and return normalized dataset IDs."""
    try:
        data = json.loads(result_text)
    except (json.JSONDecodeError, TypeError):
        return []

    results_list = data.get("results", [])
    ids: List[str] = []
    for item in results_list:
        # Search tools: dataset_uri or uri field.
        raw = item.get("dataset_uri") or item.get("uri")
        if raw:
            ids.append(_normalize_dataset_id(raw))
            continue
        # Baseline tools: dataset_id field
        raw = item.get("dataset_id")
        if raw:
            ids.append(_normalize_dataset_id(raw))
    return ids


def _task_id_safe(task_id: str) -> str:
    """Convert a task file path to a safe filename stem."""
    return Path(task_id).stem if task_id else "unknown"


def _write_record(record: dict) -> None:
    """Append a JSONL record to the current task's trace file."""
    if not _current_output_dir or not _current_task_id:
        return
    p = Path(_current_task_id)
    out_dir = os.path.join(_current_output_dir, p.parent.name)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, p.stem + ".jsonl")
    with open(path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _tool_use_key(tool_use: dict[str, Any]) -> str:
    """Return a stable key for pairing before/after tool hook state."""
    return str(tool_use.get("toolUseId") or "__legacy_single_tool__")


def write_trace_record(record: dict) -> None:
    """Append an auxiliary trace record using the active task trace context."""
    payload = {
        "task_id": _current_task_id,
        "timestamp_ms": int(time.time() * 1000),
    }
    payload.update(record)
    _write_record(payload)


# ---------------------------------------------------------------------------
# TracePlugin
# ---------------------------------------------------------------------------

class TracePlugin(Plugin):
    """Writes per-call JSONL traces for search tools and submit_answer."""

    name = "trace"

    def __init__(self, output_dir: str = "results/traces") -> None:
        super().__init__()
        self._output_dir = output_dir
        self._t0: float = 0.0
        self._pending_query: str = ""
        self._pending_tool: str = ""
        self._pending_searches: dict[str, tuple[float, str, str]] = {}

    @hook
    def on_before_tool(self, event: BeforeToolCallEvent) -> None:
        tool_name = event.tool_use.get("name", "")
        if tool_name not in _SEARCH_TOOLS:
            return
        started_at = time.time()
        pending_query = event.tool_use.get("input", {}).get("query", "")
        self._t0 = started_at
        self._pending_query = pending_query
        self._pending_tool = tool_name
        self._pending_searches[_tool_use_key(event.tool_use)] = (
            started_at,
            pending_query,
            tool_name,
        )

    @hook
    def on_after_tool(self, event: AfterToolCallEvent) -> None:
        global _turn_counter
        tool_name = event.tool_use.get("name", "")

        if tool_name in _SEARCH_TOOLS:
            _turn_counter += 1
            started_at, pending_query, _ = self._pending_searches.pop(
                _tool_use_key(event.tool_use),
                (self._t0, self._pending_query, self._pending_tool),
            )
            latency_ms = int((time.time() - started_at) * 1000)

            # Extract result text
            content = event.result.get("content", [])
            result_text = ""
            for block in content:
                if isinstance(block, dict) and "text" in block:
                    result_text = block["text"]
                    break

            result_ids = _extract_dataset_ids(result_text)
            gold_set = set(_current_gold_ids)
            gold_in_results = [rid for rid in result_ids if rid in gold_set]

            # gold_rank: 1-based rank of first gold hit, -1 if none
            gold_rank = -1
            for i, rid in enumerate(result_ids):
                if rid in gold_set:
                    gold_rank = i + 1
                    break

            _write_record({
                "task_id": _current_task_id,
                "turn": _turn_counter,
                "tool": tool_name,
                "query": pending_query,
                "latency_ms": latency_ms,
                "result_dataset_ids": result_ids,
                "gold_dataset_ids_in_results": gold_in_results,
                "gold_rank": gold_rank,
                "timestamp_ms": int(time.time() * 1000),
            })
            return

        if tool_name == "submit_answer":
            tool_input = event.tool_use.get("input", {})
            _write_record({
                "task_id": _current_task_id,
                "tool": "submit_answer",
                "answer_text": tool_input.get("answer", ""),
                "reasoning": tool_input.get("reasoning", ""),
                "status": event.result.get("status", "success"),
                "timestamp_ms": int(time.time() * 1000),
            })
