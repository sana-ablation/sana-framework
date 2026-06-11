#!/usr/bin/env python3
"""
Semantic-first mode analysis runner.

This is the current mode-analysis entrypoint. Compatibility code in this file
is limited to loading adjacent raw result trees and preserving a few output
shapes for downstream paper-generation scripts.

Reads semantic-audited eval_results.csv files from:
  results-ec2_semantic/modes/{model}/{variant}/eval_results.csv

Joins the richer base run metadata from:
  results-ec2/modes/{model}/{variant}/agent_results.jsonl

And traces from:
  results-ec2/traces/modes/{model}/{variant}/{task_dir}/{task}.jsonl

Outputs (default: analysis_results_mode_semantic/):
  semantic_match.json
  discovery.json
  runtime.json
  efficiency.json
  failure.json
  tools_discovery.json
  tool_errors.json
  semantic_buckets.json
  log_error_buckets.json
  semantic_error_crosstab.json
  search_depth.json
  reasoning_density.json
  search_first_hit_condition.json
  search_first_hit_tool.json
  search_topk_miss_tool.json
  search_tool_efficiency.json
  search_depth_buckets.json
  reasoning_density_buckets.json
  summary.json
  variant_summary.json
  per_task_semantic.csv
  per_task_retrieval.csv
  per_task_search_bottleneck.csv
  per_task_search_tool_bottleneck.csv
  figures/ (unless --no-figures)
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from sana_analysis.running_analysis.discovery_metrics import (
    compute_discovery_metrics,
    compute_tools_discovery,
    load_task_gold_ids_for_traces,
    load_task_gold_ids,
    load_task_gold_source_ids_for_traces,
    load_task_gold_source_ids,
    load_task_gold_unit_ids_for_traces,
    load_task_gold_unit_ids,
    make_task_stem_key,
    _normalize_source_id,
)
from sana_analysis.running_analysis.reasoning_density import _BINS as _REASONING_DENSITY_BINS
from sana_analysis.running_analysis.reasoning_density import _assign_bin as _assign_reasoning_density_bin
from sana_analysis.running_analysis.reasoning_density import load_task_gold_counts
from sana_analysis.running_analysis.search_bottleneck import (
    SEARCH_BOTTLENECK_CUTOFFS,
    compute_search_bottleneck,
    generate_search_bottleneck_figures,
    write_search_bottleneck_csv,
)
from sana_analysis.running_analysis.search_depth import _BINS as _SEARCH_DEPTH_BINS
from sana_analysis.running_analysis.search_depth import _assign_bin as _assign_search_depth_bin
from sana_analysis.running_analysis.tool_error_analysis import _DATA_TOOLS
from sana_analysis.report_generator.run_mode_delta_figures import generate_delta_figures


SEMANTIC_BUCKETS = [
    "semantic_correct",
    "semantic_incorrect",
    "answer_unknown_blank",
]

SOURCE_LOG_ERROR_BUCKETS = [
    "",
    "error_turns_exhausted",
    "error_tools_limit",
    "error_tokens_reached",
    "error_context_overflow",
    "error_event_loop",
    "error_unknown",
]

DISPLAY_LOG_ERROR_BUCKETS = [
    "no_error",
    "error_turns_exhausted",
    "error_tools_limit",
    "error_tokens_reached",
    "error_context_overflow",
    "error_event_loop",
    "error_unknown",
]

FAILURE_PRIMARY_BUCKETS = [
    "semantic_correct",
    "semantic_incorrect",
    "answer_unknown_blank",
    "error_turns_exhausted",
    "error_tools_limit",
    "error_tokens_reached",
    "error_context_overflow",
    "error_event_loop",
    "error_unknown",
]

REQUIRED_FIELDS = [
    "exact_match",
    "semantic_match",
    "semantic_reason",
    "semantic_bucket",
    "log_error_bucket",
    "log_error_evidence",
]

REQUIRED_TURN_WASTE_GROUP_FIELDS = [
    "task_id",
    "turn_waste_global_group",
    "turn_waste_global_group_reason",
]

SEMANTIC_BUCKET_COLORS = {
    "semantic_correct": "#2E8B57",
    "semantic_incorrect": "#C07A2C",
    "answer_unknown_blank": "#C44E52",
}

SEMANTIC_BUCKET_DISPLAY = {
    "semantic_correct": "Correct",
    "semantic_incorrect": "Incorrect",
    "answer_unknown_blank": "Blank",
}

LOG_ERROR_BUCKET_COLORS = {
    "no_error": "#2E8B57",
    "error_turns_exhausted": "#C44E52",
    "error_tools_limit": "#E17C05",
    "error_tokens_reached": "#7A3E9D",
    "error_context_overflow": "#9467BD",
    "error_event_loop": "#7F7F7F",
    "error_unknown": "#BCBD22",
}

LOG_ERROR_BUCKET_DISPLAY = {
    "no_error": "None",
    "error_turns_exhausted": "Turns",
    "error_tools_limit": "Tools",
    "error_tokens_reached": "Tokens",
    "error_context_overflow": "Context",
    "error_event_loop": "Event Loop",
    "error_unknown": "Unknown",
}

TURN_WASTE_FAMILY_DISPLAY = {
    "data_query_loops": "Data/query loops",
    "redundant_rework": "Redundant rework",
    "search_path_drift": "Search/path drift",
    "progress_submission_failures": "Progress/submission failures",
    "other_mixed": "Other/mixed",
}

TURN_WASTE_FAMILY_COLORS = {
    "data_query_loops": "#4C78A8",
    "redundant_rework": "#72B7B2",
    "search_path_drift": "#F58518",
    "progress_submission_failures": "#E45756",
    "other_mixed": "#9D755D",
}

TURN_WASTE_CANONICAL_GROUP_TO_FAMILY = {
    "Data/source access and repair loops": "data_query_loops",
    "Redundant same-hop work after evidence": "redundant_rework",
    "Wrong-source and lookup fixation": "search_path_drift",
    "Final-hop retrieval/finalization failure": "progress_submission_failures",
    "Runtime/blocker fallback": "progress_submission_failures",
    "Structured Data Extraction Loop": "data_query_loops",
    "Source and query recovery churn": "data_query_loops",
    "Source and Query-Shape Churn": "data_query_loops",
    "Intermediate result rework": "redundant_rework",
    "Redundant Revalidation After Progress": "redundant_rework",
    "Redundant final verification": "redundant_rework",
    "Redundant Intermediate Re-querying": "redundant_rework",
    "Candidate Overconfirmation": "redundant_rework",
    "Post-Answer Verification Churn": "redundant_rework",
    "Missing Source Search Loop": "search_path_drift",
    "Wrong-Branch Or Off-Path Drift": "search_path_drift",
    "Late low-yield detour": "search_path_drift",
    "Post-Answer Or Source Non-Submission": "progress_submission_failures",
    "Partial Component Stall": "progress_submission_failures",
    "Downstream Deferral and Chain Crowd-Out": "progress_submission_failures",
}

TURN_WASTE_UNCLASSIFIED_DISPLAY = "Unclassified"
TURN_WASTE_UNCLASSIFIED_COLOR = "#B8B8B8"
TURN_WASTE_MODEL_COLORS = {
    "gpt-5-mini": "#4C78A8",
    "openai_gpt-5-mini": "#4C78A8",
    "gpt-5.4-nano": "#F58518",
    "openai_gpt-5.4-nano": "#F58518",
}
TURN_WASTE_GROUP_COLORS = {
    "Data/source access and repair loops": "#4C78A8",
    "Redundant same-hop work after evidence": "#72B7B2",
    "Wrong-source and lookup fixation": "#F58518",
    "Final-hop retrieval/finalization failure": "#E45756",
}
TURN_WASTE_FIGURE_EXCLUDED_GROUPS = {"Runtime/blocker fallback"}
TURN_WASTE_CONDITION_FIGURE_ORDER = [
    ("No Plan", "search_i_results_i_plann_computei_k5_skills_off"),
    ("Standard Plan", "search_i_results_i_pland_computei_k5_skills_off"),
    ("BM25", "search_n_results_i_plani_computei_k5_skills_off"),
    ("Pneuma Hybrid", "search_d_results_i_plani_computei_k5_skills_off"),
    ("Standard Computation", "search_i_results_i_plani_k5_skills_off"),
    ("Ideal", "search_i_results_i_plani_computei_k5_skills_off"),
]
TURN_WASTE_CONDITION_FIGURE_MODEL_ORDER = ["openai_gpt-5.4-nano", "openai_gpt-5-mini", "gpt-5.4-nano", "gpt-5-mini"]
TURN_WASTE_CONDITION_FIGURE_MODEL_LABELS = {
    "openai_gpt-5.4-nano": "5.4\nnano",
    "gpt-5.4-nano": "5.4\nnano",
    "openai_gpt-5-mini": "5\nmini",
    "gpt-5-mini": "5\nmini",
}
TURN_WASTE_CONDITION_FIGURE_LABELS = {
    "No Plan": "No\nPlan",
    "Standard Plan": "Standard\nPlan",
    "BM25": "BM25",
    "Pneuma Hybrid": "Pneuma\nHybrid",
    "Standard Computation": "Standard\nComputation",
    "Ideal": "Ideal",
}
TURN_WASTE_PREFERRED_GROUP_ORDER = [
    "Data/source access and repair loops",
    "Redundant same-hop work after evidence",
    "Wrong-source and lookup fixation",
    "Final-hop retrieval/finalization failure",
]

_LETTER_TO_MODE = {"n": "naive", "d": "standard", "i": "ideal", "p": "preloaded"}
_MODE_PRIORITY = {"ideal": 0, "standard": 1, "naive": 2, None: 3}
_UNIMPORTANT_TOOLS = {"get_sandbox_info", "submit_answer", "plan", "think"}
_MODE_DISPLAY = {"ideal": "Ideal", "standard": "Std", "naive": "Naive", None: "?"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results-ec2_semantic/modes")
    parser.add_argument("--base-results-dir", default=None)
    parser.add_argument("--turn-waste-grouped-dir", default=None)
    parser.add_argument(
        "--no-turn-waste",
        action="store_true",
        help="Skip the grouped turn-waste join entirely (no turn_waste_* outputs).",
    )
    parser.add_argument("--traces-dir", default="results-ec2/traces/modes")
    parser.add_argument("--tasks-dir", default="benchmarks/lakeqa/tasks-mini/tasks")
    parser.add_argument("--output-dir", default="analysis_results_mode_semantic")
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip graph generation.",
    )
    parser.add_argument(
        "--model-filter",
        default=None,
        help="Substring filter on model directory name (comma-separated OR).",
    )
    return parser.parse_args()


def as_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _cm_key(model: str, variant: str) -> str:
    return f"{model}/{variant}"


def _split_cm_key(key: str) -> Tuple[str, str]:
    parts = key.split("/", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return "unknown", key


def _parse_model_filters(model_filter: Optional[str]) -> Optional[List[str]]:
    if not model_filter:
        return None
    filters = [token.strip().lower() for token in model_filter.split(",") if token.strip()]
    return filters or None


def _parse_variant(variant: str) -> Dict[str, Optional[object]]:
    out: Dict[str, Optional[object]] = {
        "variant": variant,
        "search_tool": None,
        "search_results": None,
        "agent_management": None,
        "computation_tool": "standard",
        "plan_skills": None,
        "k": None,
        "sc": None,
    }
    parts = variant.split("_")
    for idx, token in enumerate(parts):
        if token == "search" and idx + 1 < len(parts):
            out["search_tool"] = _LETTER_TO_MODE.get(parts[idx + 1])
        elif token == "results" and idx + 1 < len(parts):
            out["search_results"] = _LETTER_TO_MODE.get(parts[idx + 1])
        elif token.startswith("plan") and len(token) > 4:
            out["agent_management"] = _LETTER_TO_MODE.get(token[4:])
        elif token.startswith("compute") and len(token) > 7:
            out["computation_tool"] = _LETTER_TO_MODE.get(token[7:])
        elif token == "skills" and idx + 1 < len(parts):
            out["plan_skills"] = parts[idx + 1]
        elif token.startswith("k") and token[1:].isdigit():
            out["k"] = int(token[1:])
        elif token.startswith("sc") and token[2:].isdigit():
            out["sc"] = int(token[2:])
    return out


def _parse_variant_mode_codes(variant: str) -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {
        "search": None,
        "results": None,
        "plan": None,
        "compute": None,
        "skills": None,
        "k": None,
        "sc": None,
    }
    parts = variant.split("_")
    for idx, token in enumerate(parts):
        if token == "search" and idx + 1 < len(parts):
            out["search"] = parts[idx + 1]
        elif token == "results" and idx + 1 < len(parts):
            out["results"] = parts[idx + 1]
        elif token.startswith("plan") and len(token) > 4:
            out["plan"] = token[4:]
        elif token.startswith("compute") and len(token) > 7:
            out["compute"] = token[7:]
        elif token == "skills" and idx + 1 < len(parts):
            out["skills"] = parts[idx + 1]
        elif token.startswith("k") and token[1:].isdigit():
            out["k"] = token[1:]
        elif token.startswith("sc") and token[2:].isdigit():
            out["sc"] = token[2:]
    return out


def _mode_display(mode: Optional[str]) -> str:
    return _MODE_DISPLAY.get(mode, str(mode) if mode is not None else "?")


def _compact_variant_label(variant: str, *, multiline: bool = False) -> str:
    codes = _parse_variant_mode_codes(variant)
    parts = []
    if codes.get("search"):
        parts.append(f"S:{_mode_display(_LETTER_TO_MODE.get(codes['search']))}")
    if codes.get("results") and codes.get("results") != "i":
        parts.append(f"R:{_mode_display(_LETTER_TO_MODE.get(codes['results']))}")
    if codes.get("plan"):
        parts.append(f"P:{_mode_display(_LETTER_TO_MODE.get(codes['plan']))}")
    if codes.get("compute") and _LETTER_TO_MODE.get(codes["compute"]) != "standard":
        parts.append(f"C:{_mode_display(_LETTER_TO_MODE.get(codes['compute']))}")
    if codes.get("k") and codes.get("k") != "5":
        parts.append(f"k={codes['k']}")
    if codes.get("sc"):
        parts.append(f"sc={codes['sc']}")
    if not parts:
        return variant
    return ("\n" if multiline else " | ").join(parts)


def _standard_condition_label(variant: str) -> str:
    axes = _parse_variant(variant)
    search_display = {
        "naive": "BM25",
        "standard": "Hybrid",
        "ideal": "Ideal",
        "preloaded": "Preloaded",
    }
    plan_display = {
        "naive": "None",
        "standard": "Standard",
        "ideal": "Ideal",
    }
    execution_display = {
        "standard": "Standard",
        "ideal": "Ideal",
    }
    search = search_display.get(str(axes.get("search_tool") or "unknown"), "Unknown")
    plan = plan_display.get(str(axes.get("agent_management") or "unknown"), "Unknown")
    execution = execution_display.get(str(axes.get("computation_tool") or "standard"), "Standard")
    return f"{search} / {plan} / {execution}"


def _variant_sort_key(variant: str) -> tuple:
    axes = _parse_variant(variant)
    search_tool = axes.get("search_tool")
    agent_management = axes.get("agent_management")

    if search_tool == "ideal" and agent_management == "ideal":
        ideal_group = 0
    elif search_tool == "ideal":
        ideal_group = 1
    elif agent_management == "ideal":
        ideal_group = 2
    else:
        ideal_group = 3

    return (
        ideal_group,
        _MODE_PRIORITY.get(search_tool, 4),
        _MODE_PRIORITY.get(agent_management, 4),
        _MODE_PRIORITY.get(axes.get("computation_tool"), 4),
        _MODE_PRIORITY.get(axes.get("search_results"), 4),
        axes.get("k") if axes.get("k") is not None else 10**9,
        axes.get("sc") if axes.get("sc") is not None else 10**9,
        variant,
    )


def _condition_model_sort_key(key: str) -> tuple:
    model, variant = _split_cm_key(key)
    return (*_variant_sort_key(variant), model)


def _sort_summary_rows(rows: List[dict]) -> List[dict]:
    return sorted(
        rows,
        key=lambda row: (
            *_variant_sort_key(str(row.get("variant", ""))),
            str(row.get("model", "")),
            str(row.get("condition_model", "")),
        ),
    )


def _sort_variant_rows(rows: List[dict]) -> List[dict]:
    return sorted(rows, key=lambda row: _variant_sort_key(str(row.get("variant", ""))))


def _safe_slug(value: str) -> str:
    out = []
    for char in value.lower():
        if char.isalnum():
            out.append(char)
        else:
            out.append("_")
    slug = "".join(out)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_") or "unknown"


def _default_base_results_dir(results_dir: str) -> Optional[str]:
    """Infer the adjacent raw result tree for audited semantic results."""
    candidate = str(results_dir).replace("_semantic", "", 1)
    if candidate != results_dir:
        return candidate
    return None


def _default_turn_waste_grouped_dir(results_dir: str) -> Optional[str]:
    """Infer the optional grouped turn-waste tree next to semantic results."""
    results_path = Path(results_dir)
    if results_path.name == "modes":
        results_path = results_path.parent

    results_str = str(results_path)
    candidate = results_str.replace("_semantic", "_semantic_turn_waste_grouped", 1)
    if candidate != results_str:
        return candidate
    return None


def _parse_results_csv_path(path: Path, results_root: Path) -> Tuple[str, str]:
    rel = path.relative_to(results_root)
    parts = rel.parts
    if len(parts) >= 3 and parts[-1] == "eval_results.csv":
        return parts[-3], parts[-2]
    raise ValueError(f"Expected <results>/<model>/<variant>/eval_results.csv layout, got {path}")


def _parse_agent_results_path(path: Path, base_root: Path) -> Tuple[str, str]:
    rel = path.relative_to(base_root)
    parts = rel.parts
    if len(parts) >= 3 and parts[-1] == "agent_results.jsonl":
        return parts[-3], parts[-2]
    raise ValueError(f"Expected <results>/<model>/<variant>/agent_results.jsonl layout, got {path}")


def _parse_trace_jsonl_path(path: Path, traces_root: Path) -> Tuple[str, str, str]:
    rel = path.relative_to(traces_root)
    parts = rel.parts
    if len(parts) >= 4:
        task_id = f"{parts[-2]}/{Path(parts[-1]).stem}"
        return parts[-4], parts[-3], task_id
    raise ValueError(f"Expected <traces>/<model>/<variant>/<task_dir>/<task>.jsonl layout, got {path}")


_READ_SOURCE_TOOLS = {
    "read_file",
    "peek_file",
    "peek_multiple",
    "grep_file",
    "parse_xml_records",
    "query_file",
    "download",
    "query_ideal",
    "execute_ideal",
}


def _infer_log_path_for_trace(jsonl_path: Path, traces_root: Path, records: List[dict]) -> Optional[Path]:
    root_parts = list(traces_root.parts)
    if "results-kramabench" in root_parts:
        index = root_parts.index("results-kramabench")
        log_root = Path(*root_parts[:index], "log-kramabench", "modes")
        task_root = "tasks-mini-kramabench"
    elif "results" in root_parts:
        index = root_parts.index("results")
        log_root = Path(*root_parts[:index], "logs", "modes")
        task_root = "tasks_mini"
    else:
        return None
    model, variant, task_id = _parse_trace_jsonl_path(jsonl_path, traces_root)
    for record in records:
        record_task_id = str(record.get("task_id", "") or "")
        parts = Path(record_task_id).parts
        if len(parts) >= 3 and parts[-3].startswith("tasks"):
            task_root = parts[-3]
            break
    task_dir, task_name = task_id.split("/", 1)
    return log_root / model / variant / task_root / task_dir / f"{task_name}.log"


def _extract_json_object(text: str) -> Optional[dict]:
    start = text.find("{")
    if start < 0:
        return None
    try:
        value, _end = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _source_ids_from_payload(payload: dict) -> list[str]:
    source_ids: list[str] = []

    def add(value: str) -> None:
        source_id = _normalize_source_id(value)
        if source_id and source_id not in source_ids:
            source_ids.append(source_id)

    if isinstance(payload.get("s3_uri"), str):
        add(str(payload["s3_uri"]))
    for value in payload.get("s3_uris") or []:
        if isinstance(value, str):
            add(value)
    dataset_id = payload.get("dataset_id")
    file_path = payload.get("file_path")
    if isinstance(dataset_id, str) and isinstance(file_path, str):
        add(_normalize_source_id("", dataset_id=dataset_id, file_path=file_path))
    def collect_from_items(items) -> None:
        for item in items or []:
            if isinstance(item, str):
                add(item)
                continue
            if not isinstance(item, dict):
                continue
            if isinstance(item.get("s3_uri"), str):
                add(str(item["s3_uri"]))
            dataset_id = item.get("dataset_id")
            file_path = item.get("file_path") or item.get("path")
            if isinstance(dataset_id, str) and isinstance(file_path, str):
                add(_normalize_source_id("", dataset_id=dataset_id, file_path=file_path))

    collect_from_items(payload.get("results"))
    collect_from_items(payload.get("files"))
    collect_from_items(payload.get("downloaded"))
    return source_ids


def _dataset_ids_from_payload(payload: dict) -> list[str]:
    dataset_ids: list[str] = []

    def add(value: str) -> None:
        dataset_id = str(value or "").strip()
        if dataset_id and dataset_id not in dataset_ids:
            dataset_ids.append(dataset_id)

    for value in payload.get("dataset_ids") or []:
        if isinstance(value, str):
            add(value)
    dataset_id = payload.get("dataset_id")
    if isinstance(dataset_id, str):
        add(dataset_id)

    def collect_from_items(items) -> None:
        for item in items or []:
            if not isinstance(item, dict):
                continue
            dataset_id = item.get("dataset_id")
            if isinstance(dataset_id, str):
                add(dataset_id)

    collect_from_items(payload.get("results"))
    collect_from_items(payload.get("files"))
    collect_from_items(payload.get("downloaded"))
    return dataset_ids


def _source_ids_from_text(text: str) -> list[str]:
    source_ids: list[str] = []
    for match in re.finditer(r's3://[^"\\\s,}]+', text):
        source_id = _normalize_source_id(match.group(0))
        if source_id and source_id not in source_ids:
            source_ids.append(source_id)
    return source_ids


def _dataset_ids_from_text(text: str) -> list[str]:
    dataset_ids: list[str] = []
    for match in re.finditer(r'"dataset_id"\s*:\s*"([^"]+)"', text):
        dataset_id = match.group(1).strip()
        if dataset_id and dataset_id not in dataset_ids:
            dataset_ids.append(dataset_id)
    return dataset_ids


def _parse_log_timestamp_ms(line: str) -> int | None:
    match = re.match(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
    if not match:
        return None
    try:
        timestamp = dt.datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None
    return int(timestamp.replace(tzinfo=dt.timezone.utc).timestamp() * 1000)


def _trace_time_window(records: List[dict], *, pad_ms: int = 60_000) -> tuple[int, int] | None:
    timestamps = [
        int(record["timestamp_ms"])
        for record in records
        if isinstance(record.get("timestamp_ms"), (int, float))
    ]
    if not timestamps:
        return None
    return min(timestamps) - pad_ms, max(timestamps) + pad_ms


def _filter_events_to_window(events: list, window: tuple[int, int] | None) -> list:
    if window is None:
        return events
    start_ms, end_ms = window
    filtered = [
        event
        for event in events
        if event.get("timestamp_ms") is None
        or start_ms <= int(event["timestamp_ms"]) <= end_ms
    ]
    return filtered or events


def _parse_log_source_events(log_path: Path) -> tuple[list[dict], list[dict], list[dict]]:
    if not log_path.exists():
        return [], [], []
    search_events: list[dict] = []
    read_events: list[dict] = []
    list_events: list[dict] = []
    pending_tool = ""
    execute_re = re.compile(r"Executing:\s*([A-Za-z_][A-Za-z0-9_]*)\(")
    with log_path.open(errors="replace") as handle:
        for line in handle:
            timestamp_ms = _parse_log_timestamp_ms(line)
            if "Executing:" in line:
                match = execute_re.search(line)
                tool = match.group(1) if match else ""
                pending_tool = tool
                if tool in _READ_SOURCE_TOOLS:
                    payload = _extract_json_object(line)
                    if payload:
                        source_ids = _source_ids_from_payload(payload)
                        if source_ids:
                            read_events.append({"tool": tool, "source_ids": source_ids, "timestamp_ms": timestamp_ms})
                continue
            if "Tool result:" not in line:
                continue
            payload = _extract_json_object(line.split("Tool result:", 1)[1])
            result_text = line.split("Tool result:", 1)[1]
            if not payload and pending_tool == "list_files":
                source_ids = _source_ids_from_text(result_text)
                dataset_ids = _dataset_ids_from_text(result_text)
                if source_ids or dataset_ids:
                    list_events.append({
                        "tool": "list_files",
                        "source_ids": source_ids,
                        "dataset_ids": dataset_ids,
                        "timestamp_ms": timestamp_ms,
                    })
                pending_tool = ""
                continue
            if not payload:
                pending_tool = ""
                continue
            if pending_tool == "list_files":
                source_ids = _source_ids_from_payload(payload)
                dataset_ids = _dataset_ids_from_payload(payload)
                if source_ids or dataset_ids:
                    list_events.append({
                        "tool": "list_files",
                        "source_ids": source_ids,
                        "dataset_ids": dataset_ids,
                        "timestamp_ms": timestamp_ms,
                    })
            elif isinstance(payload.get("results"), list):
                source_ids = _source_ids_from_payload(payload)
                if source_ids:
                    search_events.append({"source_ids": source_ids, "timestamp_ms": timestamp_ms})
            pending_tool = ""
    return search_events, read_events, list_events


def _enrich_trace_records_with_log_sources(records: List[dict], log_path: Optional[Path]) -> None:
    if log_path is None:
        return
    search_events, read_events, list_events = _parse_log_source_events(log_path)
    time_window = _trace_time_window(records)
    search_events = _filter_events_to_window(search_events, time_window)
    read_events = _filter_events_to_window(read_events, time_window)
    list_events = _filter_events_to_window(list_events, time_window)
    search_index = 0
    used_read_events: set[int] = set()
    for record in records:
        tool = str(record.get("tool", "") or "")
        if tool.startswith("search") and record.get("result_dataset_ids") is not None:
            if search_index < len(search_events):
                record["result_source_ids"] = search_events[search_index]["source_ids"]
            search_index += 1
        elif tool in _READ_SOURCE_TOOLS and record.get("read_dataset_ids") is not None:
            match_index = next(
                (
                    index
                    for index, event in enumerate(read_events)
                    if index not in used_read_events and event.get("tool") == tool
                ),
                None,
            )
            if match_index is None:
                match_index = next(
                    (index for index, _event in enumerate(read_events) if index not in used_read_events),
                    None,
                )
            if match_index is not None:
                record["read_source_ids"] = read_events[match_index]["source_ids"]
                used_read_events.add(match_index)

    task_id = next((str(record.get("task_id")) for record in records if record.get("task_id")), "")
    for index, event in enumerate(read_events):
        if index in used_read_events or event.get("tool") != "download":
            continue
        records.append({
            "task_id": task_id,
            "tool": "download",
            "status": "success",
            "attempted_read_dataset_ids": [],
            "read_dataset_ids": [],
            "read_source_ids": event["source_ids"],
            "gold_dataset_ids_read": [],
            "log_enriched": True,
        })
    for event in list_events:
        records.append({
            "task_id": task_id,
            "tool": "list_files",
            "status": "success",
            "result_dataset_ids": event.get("dataset_ids", []),
            "result_source_ids": event.get("source_ids", []),
            "gold_dataset_ids_in_results": [],
            "log_enriched": True,
        })


def _validate_required_fields(fieldnames: List[str] | None, path: Path) -> None:
    available = fieldnames or []
    missing = [field for field in REQUIRED_FIELDS if field not in available]
    if missing:
        raise ValueError(f"Missing required semantic columns in {path}: {', '.join(missing)}")


def _validate_required_turn_waste_group_fields(fieldnames: List[str] | None, path: Path) -> None:
    available = fieldnames or []
    missing = [field for field in REQUIRED_TURN_WASTE_GROUP_FIELDS if field not in available]
    if missing:
        raise ValueError(f"Missing required grouped turn-waste columns in {path}: {', '.join(missing)}")


def _normalize_log_error_bucket(value: str) -> str:
    normalized = str(value or "").strip()
    if normalized not in SOURCE_LOG_ERROR_BUCKETS:
        raise ValueError(f"Unexpected log_error_bucket value: {normalized!r}")
    return normalized if normalized else "no_error"


def _extend_field_order(field_order: List[str], new_fields: Iterable[str]) -> None:
    for field in new_fields:
        if field not in field_order:
            field_order.append(field)


def _mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def _round_or_none(value: Optional[float], digits: int = 4) -> Optional[float]:
    if value is None:
        return None
    return round(value, digits)


def _weighted_avg(rows: List[dict], field: str, weight_field: str = "n") -> Optional[float]:
    numerator = 0.0
    denominator = 0.0
    for row in rows:
        value = row.get(field)
        weight = row.get(weight_field)
        if value is None or weight in (None, 0):
            continue
        numerator += float(value) * float(weight)
        denominator += float(weight)
    if denominator == 0:
        return None
    return numerator / denominator


def _tool_count_value(row: dict, tool_name: str) -> float:
    for tc in row.get("tool_counts", []):
        if tc.get("name") == tool_name:
            return float(tc.get("call_count", 0) or 0)
    return 0.0


def _parse_semantic_match(value, csv_path: Path) -> float:
    try:
        semantic_match = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"semantic_match must be numeric in {csv_path}, got {value!r}") from None
    if not math.isfinite(semantic_match) or not 0.0 <= semantic_match <= 1.0:
        raise ValueError(f"semantic_match must be between 0 and 1 in {csv_path}, got {value!r}")
    return semantic_match


def _normalize_eval_row(row: dict, model: str, variant: str, csv_path: Path) -> dict:
    semantic_bucket = str(row.get("semantic_bucket", "") or "").strip()
    if semantic_bucket not in SEMANTIC_BUCKETS:
        raise ValueError(f"Unexpected semantic_bucket in {csv_path}: {semantic_bucket!r}")

    semantic_match = _parse_semantic_match(row.get("semantic_match"), csv_path)

    key = _cm_key(model, variant)
    axes = _parse_variant(variant)
    task_id = str(row.get("task_id", "") or "")

    normalized = dict(row)
    normalized["condition_model"] = key
    normalized["variant"] = variant
    normalized["model_dir"] = model
    normalized["search_tool"] = axes["search_tool"]
    normalized["search_results"] = axes["search_results"]
    normalized["agent_management"] = axes["agent_management"]
    normalized["computation_tool"] = axes["computation_tool"]
    normalized["plan_skills"] = axes["plan_skills"]
    normalized["k"] = axes["k"]
    normalized["sc"] = axes["sc"]
    normalized["task_stem"] = make_task_stem_key(task_id) if task_id else ""
    normalized["log_error_bucket_display"] = _normalize_log_error_bucket(row.get("log_error_bucket", ""))

    normalized["_cm_key"] = key
    normalized["_exact_match"] = as_float(row.get("exact_match"))
    normalized["_semantic_match"] = semantic_match
    normalized["_runtime_seconds"] = as_float(row.get("runtime_seconds"))
    normalized["_input_tokens"] = as_float(row.get("input_tokens"))
    normalized["_output_tokens"] = as_float(row.get("output_tokens"))
    normalized["_total_tokens"] = as_float(row.get("total_tokens"))
    normalized["_cost_usd"] = as_float(row.get("cost_usd"))
    normalized["_ideal_subagent_cost_usd"] = as_float(row.get("ideal_subagent_cost_usd"))
    total_with_ideal = row.get("total_cost_with_ideal_subagents_usd")
    normalized["_total_cost_with_ideal_subagents_usd"] = (
        as_float(total_with_ideal)
        if total_with_ideal not in (None, "")
        else normalized["_cost_usd"] + normalized["_ideal_subagent_cost_usd"]
    )
    normalized["_tool_calls_total"] = as_float(row.get("tool_calls_total"))
    normalized["_api_tool_calls"] = as_float(row.get("api_tool_calls"))
    return normalized


def load_semantic_results_grouped(
    results_dir: str,
    model_filters: Optional[List[str]] = None,
) -> Tuple[Dict[str, List[dict]], List[str]]:
    root = Path(results_dir)
    if not root.exists():
        raise FileNotFoundError(f"Results directory does not exist: {root}")

    def keep_model(model: str) -> bool:
        if not model_filters:
            return True
        lowered = model.lower()
        return any(token in lowered for token in model_filters)

    by_key: Dict[str, List[dict]] = defaultdict(list)
    field_order: List[str] = []
    found = False

    for csv_path in sorted(root.rglob("eval_results.csv")):
        model, variant = _parse_results_csv_path(csv_path, root)
        if not keep_model(model):
            continue
        found = True
        with csv_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            _validate_required_fields(reader.fieldnames, csv_path)
            _extend_field_order(field_order, reader.fieldnames or [])
            for row in reader:
                by_key[_cm_key(model, variant)].append(_normalize_eval_row(row, model, variant, csv_path))

    if not found:
        raise FileNotFoundError(f"No eval_results.csv files found under {root}")

    return dict(by_key), field_order


def load_turn_waste_grouped_results(
    grouped_results_dir: str,
    model_filters: Optional[List[str]] = None,
) -> Dict[str, List[dict]]:
    root = Path(grouped_results_dir)
    if not root.exists():
        raise FileNotFoundError(
            f"Grouped turn-waste directory does not exist: {root}. "
            "Run $turn-waste-grouper first to produce canonical turn-waste groups."
        )

    def keep_model(model: str) -> bool:
        if not model_filters:
            return True
        lowered = model.lower()
        return any(token in lowered for token in model_filters)

    by_key: Dict[str, List[dict]] = defaultdict(list)
    found = False
    for csv_path in sorted(root.rglob("eval_results.csv")):
        model, variant = _parse_results_csv_path(csv_path, root)
        if not keep_model(model):
            continue
        found = True
        with csv_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            _validate_required_turn_waste_group_fields(reader.fieldnames, csv_path)
            for row in reader:
                normalized = dict(row)
                normalized["condition_model"] = _cm_key(model, variant)
                normalized["variant"] = variant
                normalized["model_dir"] = model
                normalized["task_stem"] = make_task_stem_key(str(row.get("task_id", "")))
                by_key[_cm_key(model, variant)].append(normalized)

    if not found:
        raise FileNotFoundError(
            f"No grouped turn-waste eval_results.csv files found under {root}. "
            "Run $turn-waste-grouper first to produce canonical turn-waste groups."
        )

    return dict(by_key)


def load_base_agent_results_grouped(
    base_results_dir: Optional[str],
    model_filters: Optional[List[str]] = None,
) -> Dict[str, List[dict]]:
    if not base_results_dir:
        return {}
    root = Path(base_results_dir)
    if not root.exists():
        return {}

    def keep_model(model: str) -> bool:
        if not model_filters:
            return True
        lowered = model.lower()
        return any(token in lowered for token in model_filters)

    by_key: Dict[str, List[dict]] = defaultdict(list)
    for jsonl_path in sorted(root.rglob("agent_results.jsonl")):
        model, variant = _parse_agent_results_path(jsonl_path, root)
        if not keep_model(model):
            continue
        key = _cm_key(model, variant)
        with jsonl_path.open() as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                row["condition_model"] = key
                row["variant"] = variant
                row["model_dir"] = model
                row["task_stem"] = make_task_stem_key(str(row.get("task_id", "")))
                by_key[key].append(row)
    return dict(by_key)


def load_traces_grouped(
    traces_dir: str,
    model_filters: Optional[List[str]] = None,
) -> Dict[str, dict]:
    root = Path(traces_dir)
    if not root.exists():
        return {}

    def keep_model(model: str) -> bool:
        if not model_filters:
            return True
        lowered = model.lower()
        return any(token in lowered for token in model_filters)

    grouped: Dict[str, dict] = defaultdict(lambda: defaultdict(list))
    for jsonl_path in sorted(root.rglob("*.jsonl")):
        model, variant, task_id = _parse_trace_jsonl_path(jsonl_path, root)
        if not keep_model(model):
            continue
        key = _cm_key(model, variant)
        records: list[dict] = []
        with jsonl_path.open() as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        _enrich_trace_records_with_log_sources(records, _infer_log_path_for_trace(jsonl_path, root, records))
        grouped[key][task_id].extend(records)
    return {key: dict(value) for key, value in grouped.items()}


def build_search_bottleneck_meta(grouped_traces: Dict[str, dict]) -> Dict[str, dict]:
    meta_by_key: Dict[str, dict] = {}
    for key in sorted(grouped_traces.keys(), key=_condition_model_sort_key):
        model, variant = _split_cm_key(key)
        meta_by_key[key] = {
            "variant": variant,
            "base_condition": variant,
            "condition_label": variant,
            "model": model,
        }
    return meta_by_key


def run_discovery(grouped_traces: Dict[str, dict], tasks_dir: str) -> Tuple[dict, Dict[str, Dict[str, dict]]]:
    tasks_root = Path(tasks_dir)
    if not grouped_traces:
        return {}, {}

    task_gold = (
        load_task_gold_source_ids_for_traces(tasks_dir, grouped_traces)
        if "kramabench" in tasks_root.name
        else load_task_gold_ids_for_traces(tasks_dir, grouped_traces)
    )
    if not task_gold and not tasks_root.exists():
        return {}, {}
    aggregates: dict = {}
    task_metrics_by_key: Dict[str, Dict[str, dict]] = {}

    for key, traces in sorted(grouped_traces.items(), key=lambda item: _condition_model_sort_key(item[0])):
        metrics = compute_discovery_metrics(traces, task_gold)
        aggregate = metrics.get("aggregate", {})
        if aggregate:
            aggregates[key] = aggregate
        task_metrics_by_key[key] = {
            str(task_metric.get("task_id", "")): task_metric for task_metric in metrics.get("task_metrics", [])
        }

    return aggregates, task_metrics_by_key


def run_efficiency(by_key_records: Dict[str, List[dict]]) -> dict:
    def percentile(vals: list[float], p: float) -> float:
        if not vals:
            return 0.0
        s = sorted(vals)
        return s[min(int(len(s) * p / 100), len(s) - 1)]

    def dist(vals: list[float]) -> dict:
        if not vals:
            return {}
        n = len(vals)
        return {
            "mean": round(sum(vals) / n, 4),
            "p50": round(percentile(vals, 50), 4),
            "p90": round(percentile(vals, 90), 4),
            "n": n,
        }

    out = {}
    for key, rows in sorted(by_key_records.items(), key=lambda item: _condition_model_sort_key(item[0])):
        out[key] = {
            "cost_usd": dist([float(row.get("_cost_usd", 0.0) or 0.0) for row in rows]),
            "ideal_subagent_cost_usd": dist(
                [float(row.get("_ideal_subagent_cost_usd", 0.0) or 0.0) for row in rows]
            ),
            "total_cost_with_ideal_subagents_usd": dist(
                [float(row.get("_total_cost_with_ideal_subagents_usd", 0.0) or 0.0) for row in rows]
            ),
            "time_s": dist([float(row.get("_runtime_seconds", 0.0) or 0.0) for row in rows]),
            "tool_calls": dist([float(row.get("_tool_calls_total", 0.0) or 0.0) for row in rows]),
            "total_cost_usd": round(sum(float(row.get("_cost_usd", 0.0) or 0.0) for row in rows), 4),
            "total_ideal_subagent_cost_usd": round(
                sum(float(row.get("_ideal_subagent_cost_usd", 0.0) or 0.0) for row in rows), 4
            ),
            "total_cost_with_ideal_subagents_sum_usd": round(
                sum(float(row.get("_total_cost_with_ideal_subagents_usd", 0.0) or 0.0) for row in rows), 4
            ),
            "n": len(rows),
        }
    return out


def _strip_tool_recall_fields(tool_data: dict) -> dict:
    """Trim per-tool discovery details while preserving precision/recall/F1."""
    cleaned = {}
    for tool_name, stats in tool_data.items():
        cleaned_stats = {
            "calls": stats.get("calls"),
            "n_tasks": stats.get("n_tasks"),
            "tasks_with_hit": stats.get("tasks_with_hit"),
            "avg_precision": stats.get("avg_precision"),
            "avg_recall": stats.get("avg_recall"),
            "avg_f1": stats.get("avg_f1"),
        }
        per_folder = {}
        for folder, folder_stats in (stats.get("per_folder") or {}).items():
            cleaned_folder = {
                "n_tasks": folder_stats.get("n_tasks"),
                "calls": folder_stats.get("calls"),
                "avg_precision": folder_stats.get("avg_precision"),
                "avg_recall": folder_stats.get("avg_recall"),
                "avg_f1": folder_stats.get("avg_f1"),
                "tasks_with_hit": folder_stats.get("tasks_with_hit"),
                "per_task": {},
            }
            for task_id, task_stats in (folder_stats.get("per_task") or {}).items():
                cleaned_folder["per_task"][task_id] = {
                    "calls": task_stats.get("calls"),
                    "precision": task_stats.get("precision"),
                    "recall": task_stats.get("recall"),
                    "f1": task_stats.get("f1"),
                    "has_hit": task_stats.get("has_hit"),
                }
            per_folder[folder] = cleaned_folder
        cleaned_stats["per_folder"] = per_folder
        cleaned[tool_name] = cleaned_stats
    return cleaned


def run_tools_discovery(grouped_traces: Dict[str, dict], tasks_dir: str) -> dict:
    if not grouped_traces or not Path(tasks_dir).exists():
        return {}
    task_gold = load_task_gold_ids(tasks_dir)
    out = {}
    for key, traces in sorted(grouped_traces.items(), key=lambda item: _condition_model_sort_key(item[0])):
        raw = compute_tools_discovery(traces, task_gold)
        out[key] = _strip_tool_recall_fields(raw)
    return out


def run_tool_errors(base_by_key_records: Dict[str, List[dict]]) -> dict:
    acc: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for key, rows in base_by_key_records.items():
        for row in rows:
            for tc in row.get("tool_counts", []):
                name = tc.get("name", "")
                calls = int(tc.get("call_count", 0) or 0)
                successes = int(tc.get("success_count", 0) or 0)
                if calls == 0:
                    continue
                errors = max(calls - successes, 0)
                acc[key][name][0] += calls
                acc[key][name][1] += errors

    out = {}
    for key, tools in sorted(acc.items(), key=lambda item: _condition_model_sort_key(item[0])):
        out[key] = {}
        for tool_name, (total_calls, total_errors) in sorted(tools.items()):
            out[key][tool_name] = {
                "total_calls": total_calls,
                "total_errors": total_errors,
                "error_rate": round(total_errors / total_calls, 4) if total_calls else 0.0,
            }
    return out


def _primary_failure_bucket(record: dict) -> str:
    log_error_bucket = str(record.get("log_error_bucket_display", "no_error"))
    if log_error_bucket != "no_error":
        return log_error_bucket
    semantic_bucket = str(record.get("semantic_bucket", "semantic_incorrect"))
    if semantic_bucket in SEMANTIC_BUCKETS:
        return semantic_bucket
    return "error_unknown"


def build_failure(by_key_records: Dict[str, List[dict]]) -> dict:
    out = {}
    for key, records in sorted(by_key_records.items(), key=lambda item: _condition_model_sort_key(item[0])):
        total = len(records)
        semantic_counts = Counter(str(record.get("semantic_bucket", "")) for record in records)
        error_counts = Counter(str(record.get("log_error_bucket_display", "")) for record in records)
        primary_counts = Counter(_primary_failure_bucket(record) for record in records)
        joint_counts = Counter(
            (str(record.get("semantic_bucket", "")), str(record.get("log_error_bucket_display", "")))
            for record in records
        )
        out[key] = {
            "total": total,
            "semantic_buckets": {
                bucket: {
                    "n": semantic_counts.get(bucket, 0),
                    "pct": round(100 * semantic_counts.get(bucket, 0) / total, 1) if total else 0.0,
                }
                for bucket in SEMANTIC_BUCKETS
            },
            "log_error_buckets": {
                bucket: {
                    "n": error_counts.get(bucket, 0),
                    "pct": round(100 * error_counts.get(bucket, 0) / total, 1) if total else 0.0,
                }
                for bucket in DISPLAY_LOG_ERROR_BUCKETS
            },
            "primary": {
                bucket: {
                    "n": primary_counts.get(bucket, 0),
                    "pct": round(100 * primary_counts.get(bucket, 0) / total, 1) if total else 0.0,
                }
                for bucket in FAILURE_PRIMARY_BUCKETS
            },
            "joint": {
                f"{semantic_bucket}|{log_error_bucket}": {
                    "n": joint_counts.get((semantic_bucket, log_error_bucket), 0),
                    "pct": round(100 * joint_counts.get((semantic_bucket, log_error_bucket), 0) / total, 1)
                    if total
                    else 0.0,
                }
                for semantic_bucket in SEMANTIC_BUCKETS
                for log_error_bucket in DISPLAY_LOG_ERROR_BUCKETS
            },
        }
    return out


def _build_semantic_curve(
    by_key_records: Dict[str, List[dict]],
    task_metrics_by_key: Dict[str, Dict[str, dict]],
    *,
    bin_labels: List[str],
    assign_bin_fn,
    selector_fn,
) -> Tuple[dict, dict]:
    variant_acc: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    cm_acc: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    for key in sorted(by_key_records.keys(), key=_condition_model_sort_key):
        model, variant = _split_cm_key(key)
        task_metrics = task_metrics_by_key.get(key, {})
        for record in by_key_records[key]:
            selector_value = selector_fn(record, task_metrics)
            if selector_value is None:
                continue
            bin_label = assign_bin_fn(selector_value)
            semantic_match = float(record.get("_semantic_match", 0) or 0)
            variant_acc[variant][bin_label].append(semantic_match)
            cm_acc[key][bin_label].append(semantic_match)

    def finalize(acc: Dict[str, Dict[str, List[float]]]) -> dict:
        out = {}
        for outer_key, bins in sorted(acc.items(), key=lambda item: _variant_sort_key(item[0]) if "/" not in item[0] else _condition_model_sort_key(item[0])):
            out[outer_key] = {}
            for label in bin_labels:
                vals = bins.get(label, [])
                out[outer_key][label] = {
                    "mean_semantic_match": round(sum(vals) / len(vals), 4) if vals else None,
                    "n": len(vals),
                }
            if "/" in outer_key:
                model, variant = _split_cm_key(outer_key)
                out[outer_key]["model"] = model
                out[outer_key]["variant"] = variant
        return out

    return finalize(variant_acc), finalize(cm_acc)


def build_search_depth_curves(
    by_key_records: Dict[str, List[dict]],
    task_metrics_by_key: Dict[str, Dict[str, dict]],
) -> Tuple[dict, dict]:
    return _build_semantic_curve(
        by_key_records,
        task_metrics_by_key,
        bin_labels=[label for label, _ in _SEARCH_DEPTH_BINS],
        assign_bin_fn=_assign_search_depth_bin,
        selector_fn=lambda record, task_metrics: (
            int(task_metrics.get(str(record.get("task_stem", "")), {}).get("num_search_calls", 0) or 0)
            or None
        ),
    )


def build_reasoning_density_curves(
    by_key_records: Dict[str, List[dict]],
    tasks_dir: str,
) -> Tuple[dict, dict]:
    task_gold_counts = load_task_gold_counts(tasks_dir)

    def selector(record: dict, _task_metrics: Dict[str, dict]) -> Optional[int]:
        n_docs = task_gold_counts.get(str(record.get("task_stem", "")))
        return int(n_docs) if n_docs is not None else None

    dummy_task_metrics: Dict[str, Dict[str, dict]] = {key: {} for key in by_key_records}
    return _build_semantic_curve(
        by_key_records,
        dummy_task_metrics,
        bin_labels=[label for label, _ in _REASONING_DENSITY_BINS],
        assign_bin_fn=_assign_reasoning_density_bin,
        selector_fn=selector,
    )


def _empty_bucket_entry() -> dict:
    entry = {"n": 0}
    for bucket in SEMANTIC_BUCKETS:
        entry[bucket] = 0
        entry[f"{bucket}_rate"] = 0.0
    for bucket in DISPLAY_LOG_ERROR_BUCKETS:
        entry[bucket] = 0
        entry[f"{bucket}_rate"] = 0.0
    return entry


def _finalize_bucket_entry(entry: dict) -> dict:
    total = int(entry.get("n", 0) or 0)
    for bucket in SEMANTIC_BUCKETS:
        count = int(entry.get(bucket, 0) or 0)
        entry[f"{bucket}_rate"] = round(count / total, 4) if total else 0.0
    for bucket in DISPLAY_LOG_ERROR_BUCKETS:
        count = int(entry.get(bucket, 0) or 0)
        entry[f"{bucket}_rate"] = round(count / total, 4) if total else 0.0
    return entry


def _update_bucket_entry(entry: dict, record: dict) -> None:
    entry["n"] = int(entry.get("n", 0) or 0) + 1
    semantic_bucket = str(record.get("semantic_bucket", ""))
    log_error_bucket = str(record.get("log_error_bucket_display", ""))
    if semantic_bucket in SEMANTIC_BUCKETS:
        entry[semantic_bucket] = int(entry.get(semantic_bucket, 0) or 0) + 1
    if log_error_bucket in DISPLAY_LOG_ERROR_BUCKETS:
        entry[log_error_bucket] = int(entry.get(log_error_bucket, 0) or 0) + 1


def _init_binned_outcome_row(model: str, variant: str, bin_labels: List[str]) -> dict:
    axes = _parse_variant(variant)
    return {
        "model": model,
        "variant": variant,
        "search_tool": axes["search_tool"],
        "search_results": axes["search_results"],
        "agent_management": axes["agent_management"],
        "computation_tool": axes["computation_tool"],
        "plan_skills": axes["plan_skills"],
        "k": axes["k"],
        "sc": axes["sc"],
        "bins": {label: _empty_bucket_entry() for label in bin_labels},
    }


def build_search_depth_buckets(
    by_key_records: Dict[str, List[dict]],
    task_metrics_by_key: Dict[str, Dict[str, dict]],
) -> dict:
    bin_labels = [label for label, _ in _SEARCH_DEPTH_BINS]
    out: dict = {}
    for key in sorted(by_key_records.keys(), key=_condition_model_sort_key):
        model, variant = _split_cm_key(key)
        row = _init_binned_outcome_row(model, variant, bin_labels)
        task_metrics = task_metrics_by_key.get(key, {})
        for record in by_key_records[key]:
            task_metric = task_metrics.get(str(record.get("task_stem", "")))
            if not task_metric:
                continue
            search_calls = int(task_metric.get("num_search_calls", 0) or 0)
            if search_calls <= 0:
                continue
            bin_label = _assign_search_depth_bin(search_calls)
            _update_bucket_entry(row["bins"][bin_label], record)
        for label in bin_labels:
            row["bins"][label] = _finalize_bucket_entry(row["bins"][label])
        out[key] = row
    return out


def build_reasoning_density_buckets(
    by_key_records: Dict[str, List[dict]],
    tasks_dir: str,
) -> dict:
    bin_labels = [label for label, _ in _REASONING_DENSITY_BINS]
    task_gold_counts = load_task_gold_counts(tasks_dir)
    out: dict = {}
    for key in sorted(by_key_records.keys(), key=_condition_model_sort_key):
        model, variant = _split_cm_key(key)
        row = _init_binned_outcome_row(model, variant, bin_labels)
        for record in by_key_records[key]:
            task_stem = str(record.get("task_stem", ""))
            n_docs = task_gold_counts.get(task_stem)
            if n_docs is None:
                continue
            bin_label = _assign_reasoning_density_bin(n_docs)
            _update_bucket_entry(row["bins"][bin_label], record)
        for label in bin_labels:
            row["bins"][label] = _finalize_bucket_entry(row["bins"][label])
        out[key] = row
    return out


def build_summary(
    by_key_records: Dict[str, List[dict]],
    discovery: dict,
    efficiency: dict,
    tool_errors: dict,
    base_by_key_records: Dict[str, List[dict]],
    search_bottleneck: Optional[dict] = None,
) -> List[dict]:
    rows: List[dict] = []
    for key in sorted(by_key_records.keys(), key=_condition_model_sort_key):
        records = by_key_records[key]
        model, variant = _split_cm_key(key)
        axes = _parse_variant(variant)
        n = len(records)
        semantic_counts = Counter(str(record.get("semantic_bucket", "")) for record in records)
        error_counts = Counter(str(record.get("log_error_bucket_display", "")) for record in records)
        base_rows = base_by_key_records.get(key, [])

        row = {
            "condition_model": key,
            "model": model,
            "variant": variant,
            "search_tool": axes["search_tool"],
            "search_results": axes["search_results"],
            "agent_management": axes["agent_management"],
            "computation_tool": axes["computation_tool"],
            "plan_skills": axes["plan_skills"],
            "k": axes["k"],
            "sc": axes["sc"],
            "n": n,
            "em": _round_or_none(_mean([record.get("_exact_match", 0.0) for record in records])),
            "exact_match": _round_or_none(_mean([record.get("_exact_match", 0.0) for record in records])),
            "semantic_match": _round_or_none(_mean([record["_semantic_match"] for record in records])),
            "avg_runtime_seconds": _round_or_none(_mean([record["_runtime_seconds"] for record in records])),
            "avg_input_tokens": _round_or_none(_mean([record["_input_tokens"] for record in records])),
            "avg_output_tokens": _round_or_none(_mean([record["_output_tokens"] for record in records])),
            "avg_total_tokens": _round_or_none(_mean([record["_total_tokens"] for record in records])),
            "avg_cost_usd": _round_or_none(_mean([record["_cost_usd"] for record in records])),
            "total_cost_usd": round(sum(record["_cost_usd"] for record in records), 4),
            "avg_ideal_subagent_cost_usd": _round_or_none(
                _mean([record["_ideal_subagent_cost_usd"] for record in records])
            ),
            "total_ideal_subagent_cost_usd": round(
                sum(record["_ideal_subagent_cost_usd"] for record in records), 4
            ),
            "avg_total_cost_with_ideal_subagents_usd": _round_or_none(
                _mean([record["_total_cost_with_ideal_subagents_usd"] for record in records])
            ),
            "total_cost_with_ideal_subagents_usd": round(
                sum(record["_total_cost_with_ideal_subagents_usd"] for record in records), 4
            ),
            "avg_tool_calls_total": _round_or_none(_mean([record["_tool_calls_total"] for record in records])),
            "avg_tool_calls": _round_or_none(_mean([record["_tool_calls_total"] for record in records])),
            "avg_api_tool_calls": _round_or_none(_mean([record["_api_tool_calls"] for record in records])),
        }

        discovery_row = discovery.get(key, {})
        if discovery_row:
            row["D_ret"] = _round_or_none(discovery_row.get("D_ret"))
            row["D_ret_precision"] = _round_or_none(discovery_row.get("D_ret_precision"))
            row["D_ret_f1"] = _round_or_none(discovery_row.get("D_ret_f1"))
            row["D_acc"] = _round_or_none(discovery_row.get("D_acc"))
            row["D_acc_precision"] = _round_or_none(discovery_row.get("D_acc_precision"))
            row["D_acc_recall"] = _round_or_none(discovery_row.get("D_acc_recall"))
            row["D_acc_f1"] = _round_or_none(discovery_row.get("D_acc_f1"))
            row["search_avg_precision"] = _round_or_none(discovery_row.get("avg_precision"))
            row["search_avg_f1"] = _round_or_none(discovery_row.get("avg_f1"))
            row["avg_search_calls"] = _round_or_none(discovery_row.get("avg_search_calls"))
            row["avg_read_calls"] = _round_or_none(discovery_row.get("avg_read_calls"))
            avg_search_calls = discovery_row.get("avg_search_calls")
            semantic_match = row.get("semantic_match")
            row["search_efficiency"] = (
                round(float(semantic_match) / float(avg_search_calls), 4)
                if semantic_match is not None and avg_search_calls
                else None
            )
        else:
            row["D_ret"] = None
            row["D_ret_precision"] = None
            row["D_ret_f1"] = None
            row["D_acc"] = None
            row["D_acc_precision"] = None
            row["D_acc_recall"] = None
            row["D_acc_f1"] = None
            row["search_avg_precision"] = None
            row["search_avg_f1"] = None
            row["avg_search_calls"] = None
            row["avg_read_calls"] = None
            row["search_efficiency"] = None

        if base_rows:
            row["avg_query_file_calls"] = _round_or_none(_mean([_tool_count_value(r, "query_file") for r in base_rows]))
            row["avg_execute_code_calls"] = _round_or_none(
                _mean([_tool_count_value(r, "execute_code") for r in base_rows])
            )
        else:
            row["avg_query_file_calls"] = None
            row["avg_execute_code_calls"] = None

        if key in tool_errors:
            row["query_file_error_rate"] = tool_errors[key].get("query_file", {}).get("error_rate")
            row["execute_code_error_rate"] = tool_errors[key].get("execute_code", {}).get("error_rate")
            row["peek_file_error_rate"] = None
            for tool_name in ("peek_file", "peek_multiple", "peek_files"):
                error_rate = tool_errors[key].get(tool_name, {}).get("error_rate")
                if error_rate is not None:
                    row["peek_file_error_rate"] = error_rate
                    break
        else:
            row["query_file_error_rate"] = None
            row["execute_code_error_rate"] = None
            row["peek_file_error_rate"] = None

        if search_bottleneck is not None:
            sb = search_bottleneck.get(key, {})
            row["n_tasks_with_search"] = sb.get("n_tasks_with_search")
            row["n_tasks_without_search"] = sb.get("n_tasks_without_search")
            for cutoff in SEARCH_BOTTLENECK_CUTOFFS:
                found_tasks = sb.get(f"found_tasks_top_{cutoff}")
                not_found_tasks = sb.get(f"not_found_tasks_top_{cutoff}")
                avg_first_hit = sb.get(f"avg_first_hit_round_top_{cutoff}")
                avg_wasted = sb.get(f"avg_wasted_rounds_top_{cutoff}")
                row[f"found_tasks_top_{cutoff}"] = found_tasks
                row[f"found_tasks_top{cutoff}"] = found_tasks
                row[f"not_found_tasks_top_{cutoff}"] = not_found_tasks
                row[f"not_found_tasks_top{cutoff}"] = not_found_tasks
                row[f"top{cutoff}_not_found_rate"] = sb.get(f"top{cutoff}_not_found_rate")
                row[f"avg_first_hit_round_top_{cutoff}"] = avg_first_hit
                row[f"avg_first_hit_round_top{cutoff}"] = avg_first_hit
                row[f"avg_wasted_rounds_top_{cutoff}"] = avg_wasted
                row[f"avg_wasted_rounds_top{cutoff}"] = avg_wasted

        for bucket in SEMANTIC_BUCKETS:
            count = semantic_counts.get(bucket, 0)
            row[bucket] = count
            row[f"{bucket}_rate"] = round(count / n, 4) if n else 0.0

        for bucket in DISPLAY_LOG_ERROR_BUCKETS:
            count = error_counts.get(bucket, 0)
            row[bucket] = count
            row[f"{bucket}_rate"] = round(count / n, 4) if n else 0.0

        for bucket in FAILURE_PRIMARY_BUCKETS:
            count = sum(1 for record in records if _primary_failure_bucket(record) == bucket)
            row[f"primary_{bucket}"] = count

        rows.append(row)

    return _sort_summary_rows(rows)


def build_variant_summary(summary_rows: List[dict]) -> List[dict]:
    groups: Dict[str, List[dict]] = defaultdict(list)
    for row in summary_rows:
        groups[str(row.get("variant", "unknown"))].append(row)

    out: List[dict] = []
    for variant, rows in sorted(groups.items(), key=lambda item: _variant_sort_key(item[0])):
        axes = _parse_variant(variant)
        total_n = sum(int(row.get("n", 0) or 0) for row in rows)
        variant_row = {
            "variant": variant,
            "search_tool": axes["search_tool"],
            "search_results": axes["search_results"],
            "agent_management": axes["agent_management"],
            "computation_tool": axes["computation_tool"],
            "plan_skills": axes["plan_skills"],
            "k": axes["k"],
            "sc": axes["sc"],
            "n_total": total_n,
            "num_model_rows": len(rows),
            "models": sorted({str(row.get("model", "")) for row in rows if row.get("model")}),
            "em": _round_or_none(_weighted_avg(rows, "em")),
            "exact_match": _round_or_none(_weighted_avg(rows, "exact_match")),
            "semantic_match": _round_or_none(_weighted_avg(rows, "semantic_match")),
            "D_ret": _round_or_none(_weighted_avg(rows, "D_ret")),
            "D_ret_precision": _round_or_none(_weighted_avg(rows, "D_ret_precision")),
            "D_ret_f1": _round_or_none(_weighted_avg(rows, "D_ret_f1")),
            "D_acc": _round_or_none(_weighted_avg(rows, "D_acc")),
            "D_acc_precision": _round_or_none(_weighted_avg(rows, "D_acc_precision")),
            "D_acc_recall": _round_or_none(_weighted_avg(rows, "D_acc_recall")),
            "D_acc_f1": _round_or_none(_weighted_avg(rows, "D_acc_f1")),
            "search_avg_precision": _round_or_none(_weighted_avg(rows, "search_avg_precision")),
            "search_avg_f1": _round_or_none(_weighted_avg(rows, "search_avg_f1")),
            "avg_search_calls": _round_or_none(_weighted_avg(rows, "avg_search_calls")),
            "avg_read_calls": _round_or_none(_weighted_avg(rows, "avg_read_calls")),
            "avg_runtime_seconds": _round_or_none(_weighted_avg(rows, "avg_runtime_seconds")),
            "avg_input_tokens": _round_or_none(_weighted_avg(rows, "avg_input_tokens")),
            "avg_output_tokens": _round_or_none(_weighted_avg(rows, "avg_output_tokens")),
            "avg_total_tokens": _round_or_none(_weighted_avg(rows, "avg_total_tokens")),
            "avg_cost_usd": _round_or_none(_weighted_avg(rows, "avg_cost_usd")),
            "total_cost_usd": round(sum(float(row.get("total_cost_usd", 0) or 0) for row in rows), 4),
            "avg_ideal_subagent_cost_usd": _round_or_none(
                _weighted_avg(rows, "avg_ideal_subagent_cost_usd")
            ),
            "total_ideal_subagent_cost_usd": round(
                sum(float(row.get("total_ideal_subagent_cost_usd", 0) or 0) for row in rows), 4
            ),
            "avg_total_cost_with_ideal_subagents_usd": _round_or_none(
                _weighted_avg(rows, "avg_total_cost_with_ideal_subagents_usd")
            ),
            "total_cost_with_ideal_subagents_usd": round(
                sum(float(row.get("total_cost_with_ideal_subagents_usd", 0) or 0) for row in rows), 4
            ),
            "avg_tool_calls_total": _round_or_none(_weighted_avg(rows, "avg_tool_calls_total")),
            "avg_tool_calls": _round_or_none(_weighted_avg(rows, "avg_tool_calls")),
            "avg_api_tool_calls": _round_or_none(_weighted_avg(rows, "avg_api_tool_calls")),
            "avg_query_file_calls": _round_or_none(_weighted_avg(rows, "avg_query_file_calls")),
            "avg_execute_code_calls": _round_or_none(_weighted_avg(rows, "avg_execute_code_calls")),
            "query_file_error_rate": _round_or_none(_weighted_avg(rows, "query_file_error_rate")),
            "execute_code_error_rate": _round_or_none(_weighted_avg(rows, "execute_code_error_rate")),
            "peek_file_error_rate": _round_or_none(_weighted_avg(rows, "peek_file_error_rate")),
        }
        avg_search_calls = variant_row.get("avg_search_calls")
        semantic_match = variant_row.get("semantic_match")
        variant_row["search_efficiency"] = (
            round(float(semantic_match) / float(avg_search_calls), 4)
            if semantic_match is not None and avg_search_calls
            else None
        )

        variant_row["n_tasks_with_search"] = sum(int(row.get("n_tasks_with_search", 0) or 0) for row in rows)
        variant_row["n_tasks_without_search"] = sum(int(row.get("n_tasks_without_search", 0) or 0) for row in rows)
        for cutoff in SEARCH_BOTTLENECK_CUTOFFS:
            found_tasks = sum(int(row.get(f"found_tasks_top_{cutoff}", 0) or 0) for row in rows)
            not_found_tasks = sum(int(row.get(f"not_found_tasks_top_{cutoff}", 0) or 0) for row in rows)
            avg_first_hit = _round_or_none(
                _weighted_avg(rows, f"avg_first_hit_round_top_{cutoff}", weight_field=f"found_tasks_top_{cutoff}")
            )
            avg_wasted = _round_or_none(
                _weighted_avg(rows, f"avg_wasted_rounds_top_{cutoff}", weight_field="n_tasks_with_search")
            )
            variant_row[f"found_tasks_top_{cutoff}"] = found_tasks
            variant_row[f"found_tasks_top{cutoff}"] = found_tasks
            variant_row[f"not_found_tasks_top_{cutoff}"] = not_found_tasks
            variant_row[f"not_found_tasks_top{cutoff}"] = not_found_tasks
            variant_row[f"top{cutoff}_not_found_rate"] = _round_or_none(
                _weighted_avg(rows, f"top{cutoff}_not_found_rate", weight_field="n_tasks_with_search")
            )
            variant_row[f"avg_first_hit_round_top_{cutoff}"] = avg_first_hit
            variant_row[f"avg_first_hit_round_top{cutoff}"] = avg_first_hit
            variant_row[f"avg_wasted_rounds_top_{cutoff}"] = avg_wasted
            variant_row[f"avg_wasted_rounds_top{cutoff}"] = avg_wasted

        for bucket in SEMANTIC_BUCKETS:
            count = sum(int(row.get(bucket, 0) or 0) for row in rows)
            variant_row[bucket] = count
            variant_row[f"{bucket}_rate"] = round(count / total_n, 4) if total_n else 0.0

        for bucket in DISPLAY_LOG_ERROR_BUCKETS:
            count = sum(int(row.get(bucket, 0) or 0) for row in rows)
            variant_row[bucket] = count
            variant_row[f"{bucket}_rate"] = round(count / total_n, 4) if total_n else 0.0

        out.append(variant_row)

    return _sort_variant_rows(out)


def build_metric_mappings(summary_rows: List[dict]) -> Tuple[dict, dict, dict, dict, dict]:
    semantic_match = {}
    discovery = {}
    runtime = {}
    semantic_buckets = {}
    log_error_buckets = {}

    for row in summary_rows:
        key = str(row["condition_model"])
        semantic_match[key] = {
            "n": row.get("n"),
            "semantic_match": row.get("semantic_match"),
        }
        discovery[key] = {
            "n": row.get("n"),
            "D_ret": row.get("D_ret"),
            "D_ret_precision": row.get("D_ret_precision"),
            "D_ret_f1": row.get("D_ret_f1"),
            "D_acc": row.get("D_acc"),
            "D_acc_precision": row.get("D_acc_precision"),
            "D_acc_recall": row.get("D_acc_recall"),
            "D_acc_f1": row.get("D_acc_f1"),
            "search_avg_precision": row.get("search_avg_precision"),
            "search_avg_f1": row.get("search_avg_f1"),
            "avg_search_calls": row.get("avg_search_calls"),
            "avg_read_calls": row.get("avg_read_calls"),
            "search_efficiency": row.get("search_efficiency"),
        }
        runtime[key] = {
            "n": row.get("n"),
            "avg_runtime_seconds": row.get("avg_runtime_seconds"),
            "avg_input_tokens": row.get("avg_input_tokens"),
            "avg_output_tokens": row.get("avg_output_tokens"),
            "avg_total_tokens": row.get("avg_total_tokens"),
            "avg_cost_usd": row.get("avg_cost_usd"),
            "total_cost_usd": row.get("total_cost_usd"),
            "avg_ideal_subagent_cost_usd": row.get("avg_ideal_subagent_cost_usd"),
            "total_ideal_subagent_cost_usd": row.get("total_ideal_subagent_cost_usd"),
            "avg_total_cost_with_ideal_subagents_usd": row.get("avg_total_cost_with_ideal_subagents_usd"),
            "total_cost_with_ideal_subagents_usd": row.get("total_cost_with_ideal_subagents_usd"),
            "avg_tool_calls_total": row.get("avg_tool_calls_total"),
            "avg_api_tool_calls": row.get("avg_api_tool_calls"),
        }
        semantic_buckets[key] = {"total": row.get("n")}
        for bucket in SEMANTIC_BUCKETS:
            semantic_buckets[key][bucket] = row.get(bucket, 0)
            semantic_buckets[key][f"{bucket}_rate"] = row.get(f"{bucket}_rate")
        log_error_buckets[key] = {"total": row.get("n")}
        for bucket in DISPLAY_LOG_ERROR_BUCKETS:
            log_error_buckets[key][bucket] = row.get(bucket, 0)
            log_error_buckets[key][f"{bucket}_rate"] = row.get(f"{bucket}_rate")

    return semantic_match, discovery, runtime, semantic_buckets, log_error_buckets


def build_semantic_error_crosstab(by_key_records: Dict[str, List[dict]], variant_summary: List[dict]) -> List[dict]:
    rows: List[dict] = []

    def add_rows(level: str, model: Optional[str], variant: str, records: List[dict]) -> None:
        total = len(records)
        counts: Counter = Counter(
            (str(record.get("semantic_bucket", "")), str(record.get("log_error_bucket_display", "")))
            for record in records
        )
        for semantic_bucket in SEMANTIC_BUCKETS:
            for log_error_bucket in DISPLAY_LOG_ERROR_BUCKETS:
                count = counts.get((semantic_bucket, log_error_bucket), 0)
                rows.append(
                    {
                        "level": level,
                        "model": model,
                        "variant": variant,
                        "semantic_bucket": semantic_bucket,
                        "log_error_bucket": log_error_bucket,
                        "n": count,
                        "total": total,
                        "pct_within_group": round(count / total, 4) if total else 0.0,
                    }
                )

    for key in sorted(by_key_records.keys(), key=_condition_model_sort_key):
        model, variant = _split_cm_key(key)
        add_rows("model_variant", model, variant, by_key_records[key])

    grouped_by_variant: Dict[str, List[dict]] = defaultdict(list)
    for records in by_key_records.values():
        for record in records:
            grouped_by_variant[str(record.get("variant", "unknown"))].append(record)

    for row in variant_summary:
        variant = str(row.get("variant", "unknown"))
        add_rows("variant", None, variant, grouped_by_variant.get(variant, []))

    return rows


def build_turn_waste_global_groups(
    by_key_records: Dict[str, List[dict]],
    grouped_turn_waste_by_key: Dict[str, List[dict]],
    variant_rows: List[dict],
) -> Tuple[dict, List[dict], List[dict]]:
    grouped_index_by_key: Dict[str, Dict[str, dict]] = {}

    for key in sorted(by_key_records.keys(), key=_condition_model_sort_key):
        semantic_records = by_key_records[key]
        grouped_records = grouped_turn_waste_by_key.get(key)
        if grouped_records is None:
            raise ValueError(
                f"Missing grouped turn-waste eval_results.csv for {key}. "
                "Run $turn-waste-grouper for the same scope as the semantic results."
            )

        semantic_ids = [str(record.get("task_id", "")) for record in semantic_records]
        grouped_ids = [str(record.get("task_id", "")) for record in grouped_records]
        if len(semantic_ids) != len(grouped_ids):
            raise ValueError(
                f"Grouped turn-waste row count mismatch for {key}: "
                f"semantic={len(semantic_ids)} grouped={len(grouped_ids)}"
            )
        if set(semantic_ids) != set(grouped_ids):
            missing = sorted(set(semantic_ids) - set(grouped_ids))
            extra = sorted(set(grouped_ids) - set(semantic_ids))
            raise ValueError(
                f"Grouped turn-waste task mismatch for {key}: "
                f"missing={missing[:5]} extra={extra[:5]}"
            )
        grouped_index_by_key[key] = {str(record.get("task_id", "")): record for record in grouped_records}

    variant_totals = {
        str(row.get("variant", "")): {
            "n_total_rows": int(row.get("n_total", 0) or 0),
            "models": row.get("models", []),
            "num_model_rows": int(row.get("num_model_rows", 0) or 0),
        }
        for row in variant_rows
    }

    failed_join_rows: List[dict] = []
    group_counts_by_variant: Dict[str, Counter] = defaultdict(Counter)
    failed_counts_by_variant: Counter = Counter()
    grouped_failed_counts_by_variant: Counter = Counter()
    unassigned_failed_counts_by_variant: Counter = Counter()

    for key in sorted(by_key_records.keys(), key=_condition_model_sort_key):
        for semantic_record in by_key_records[key]:
            task_id = str(semantic_record.get("task_id", ""))
            grouped_record = grouped_index_by_key[key][task_id]
            log_error_bucket = str(semantic_record.get("log_error_bucket_display", "no_error"))
            if log_error_bucket == "no_error":
                continue

            group_name = str(grouped_record.get("turn_waste_global_group", "") or "").strip()
            variant = str(semantic_record.get("variant", "unknown"))
            failed_counts_by_variant[variant] += 1
            if group_name:
                grouped_failed_counts_by_variant[variant] += 1
                group_counts_by_variant[variant][group_name] += 1
            else:
                unassigned_failed_counts_by_variant[variant] += 1
            failed_join_rows.append(
                {
                    "condition_model": key,
                    "variant": variant,
                    "task_id": task_id,
                    "semantic_bucket": semantic_record.get("semantic_bucket", ""),
                    "log_error_bucket": log_error_bucket,
                    "turn_waste_global_group": group_name,
                    "turn_waste_global_group_reason": grouped_record.get("turn_waste_global_group_reason", ""),
                    "estimated_wasted_turns": grouped_record.get("estimated_wasted_turns", ""),
                }
            )

    variant_summary: dict = {}
    csv_rows: List[dict] = []
    for variant in [str(row.get("variant", "")) for row in variant_rows]:
        total_info = variant_totals.get(variant, {})
        n_total_rows = int(total_info.get("n_total_rows", 0) or 0)
        n_failed_rows = int(failed_counts_by_variant.get(variant, 0))
        n_grouped_failed_rows = int(grouped_failed_counts_by_variant.get(variant, 0))
        n_unassigned_failed_rows = int(unassigned_failed_counts_by_variant.get(variant, 0))
        groups_counter = group_counts_by_variant.get(variant, Counter())
        groups_payload = {}
        for group_name, count in sorted(groups_counter.items(), key=lambda item: (-item[1], item[0])):
            pct = round(count / n_grouped_failed_rows, 4) if n_grouped_failed_rows else 0.0
            groups_payload[group_name] = {
                "n": count,
                "pct_within_grouped_failed_rows": pct,
            }
            csv_rows.append(
                {
                    "variant": variant,
                    "turn_waste_global_group": group_name,
                    "n": count,
                    "pct_within_grouped_failed_rows": pct,
                    "n_failed_rows": n_failed_rows,
                    "n_grouped_failed_rows": n_grouped_failed_rows,
                    "n_unassigned_failed_rows": n_unassigned_failed_rows,
                    "n_total_rows": n_total_rows,
                }
            )
        variant_summary[variant] = {
            "variant": variant,
            "n_total_rows": n_total_rows,
            "n_failed_rows": n_failed_rows,
            "n_grouped_failed_rows": n_grouped_failed_rows,
            "n_unassigned_failed_rows": n_unassigned_failed_rows,
            "models": total_info.get("models", []),
            "num_model_rows": int(total_info.get("num_model_rows", 0) or 0),
            "groups": groups_payload,
        }

    failed_join_rows = sorted(
        failed_join_rows,
        key=lambda row: (
            _variant_sort_key(str(row.get("variant", ""))),
            str(row.get("turn_waste_global_group", "")),
            str(row.get("task_id", "")),
        ),
    )
    csv_rows = sorted(
        csv_rows,
        key=lambda row: (
            _variant_sort_key(str(row.get("variant", ""))),
            -int(row.get("n", 0) or 0),
            str(row.get("turn_waste_global_group", "")),
        ),
    )
    return variant_summary, csv_rows, failed_join_rows


def build_per_task_rows(
    by_key_records: Dict[str, List[dict]],
    task_metrics_by_key: Dict[str, Dict[str, dict]],
    source_field_order: List[str],
) -> Tuple[List[dict], List[str]]:
    discovery_fields = [
        "D_ret",
        "D_ret_precision",
        "D_ret_f1",
        "D_acc",
        "D_acc_precision",
        "D_acc_recall",
        "D_acc_f1",
        "search_precision",
        "search_f1",
        "num_search_calls",
        "num_read_calls",
        "retrieved_gold_dataset_ids",
    ]
    output_rows: List[dict] = []

    for key in sorted(by_key_records.keys(), key=_condition_model_sort_key):
        model, variant = _split_cm_key(key)
        axes = _parse_variant(variant)
        task_metrics = task_metrics_by_key.get(key, {})
        for record in sorted(by_key_records[key], key=lambda row: str(row.get("task_id", ""))):
            task_metric = task_metrics.get(str(record.get("task_stem", "")), {})
            output_row = {
                "condition_model": key,
                "model_dir": model,
                "variant": variant,
                "search_tool": axes["search_tool"],
                "search_results": axes["search_results"],
                "agent_management": axes["agent_management"],
                "computation_tool": axes["computation_tool"],
                "plan_skills": axes["plan_skills"],
                "k": axes["k"],
                "sc": axes["sc"],
                "task_stem": record.get("task_stem", ""),
                "log_error_bucket_display": record.get("log_error_bucket_display", ""),
                "D_ret": _round_or_none(task_metric.get("d_ret"), 6),
                "D_ret_precision": _round_or_none(task_metric.get("d_ret_precision"), 6),
                "D_ret_f1": _round_or_none(task_metric.get("d_ret_f1"), 6),
                "D_acc": _round_or_none(task_metric.get("d_acc"), 6),
                "D_acc_precision": _round_or_none(task_metric.get("d_acc_precision"), 6),
                "D_acc_recall": _round_or_none(task_metric.get("d_acc_recall"), 6),
                "D_acc_f1": _round_or_none(task_metric.get("d_acc_f1"), 6),
                "search_precision": _round_or_none(task_metric.get("precision"), 6),
                "search_f1": _round_or_none(task_metric.get("f1"), 6),
                "num_search_calls": task_metric.get("num_search_calls"),
                "num_read_calls": task_metric.get("num_read_calls"),
                "retrieved_gold_dataset_ids": ",".join(task_metric.get("retrieved_gold_dataset_ids", [])),
            }
            for field in source_field_order:
                output_row[field] = record.get(field, "")
            output_rows.append(output_row)

    fieldnames = [
        "condition_model",
        "model_dir",
        "variant",
        "search_tool",
        "search_results",
        "agent_management",
        "computation_tool",
        "plan_skills",
        "k",
        "sc",
        "task_stem",
    ] + source_field_order + ["log_error_bucket_display"] + discovery_fields

    deduped_fieldnames: List[str] = []
    for field in fieldnames:
        if field not in deduped_fieldnames:
            deduped_fieldnames.append(field)

    return output_rows, deduped_fieldnames


def build_per_task_retrieval_rows(task_metrics_by_key: Dict[str, Dict[str, dict]]) -> Tuple[List[dict], List[str]]:
    fieldnames = [
        "condition_model",
        "model",
        "variant",
        "search_tool",
        "search_results",
        "agent_management",
        "computation_tool",
        "plan_skills",
        "k",
        "sc",
        "task_id",
        "search_calls_count",
        "gold_datasets_needed_count",
        "datasets_retrieved_count",
        "datasets_retrieved_unique_count",
        "gold_datasets_retrieved_count",
        "retrieval_recall",
        "retrieval_precision",
        "retrieval_f1",
    ]
    rows: List[dict] = []
    for key in sorted(task_metrics_by_key.keys(), key=_condition_model_sort_key):
        model, variant = _split_cm_key(key)
        axes = _parse_variant(variant)
        for task_metric in sorted(task_metrics_by_key[key].values(), key=lambda item: str(item.get("task_id", ""))):
            gold_ids_count = len(task_metric.get("gold_ids", []))
            retrieved_unique_count = len(task_metric.get("retrieved_dataset_ids", []))
            retrieved_ids_count = int(task_metric.get("num_results_total_non_unique", retrieved_unique_count) or 0)
            retrieved_gold_ids_count = len(task_metric.get("retrieved_gold_dataset_ids", []))
            rows.append(
                {
                    "condition_model": key,
                    "model": model,
                    "variant": variant,
                    "search_tool": axes["search_tool"],
                    "search_results": axes["search_results"],
                    "agent_management": axes["agent_management"],
                    "computation_tool": axes["computation_tool"],
                    "plan_skills": axes["plan_skills"],
                    "k": axes["k"],
                    "sc": axes["sc"],
                    "task_id": task_metric.get("task_id", ""),
                    "search_calls_count": int(task_metric.get("num_search_calls", 0) or 0),
                    "gold_datasets_needed_count": gold_ids_count,
                    "datasets_retrieved_count": retrieved_ids_count,
                    "datasets_retrieved_unique_count": retrieved_unique_count,
                    "gold_datasets_retrieved_count": retrieved_gold_ids_count,
                    "retrieval_recall": round(float(task_metric.get("d_ret", 0) or 0), 6),
                    "retrieval_precision": round(float(task_metric.get("d_ret_precision", 0) or 0), 6),
                    "retrieval_f1": round(float(task_metric.get("d_ret_f1", 0) or 0), 6),
                }
            )
    return rows, fieldnames


def write_json(path: Path, data: object) -> None:
    with path.open("w") as handle:
        json.dump(data, handle, indent=2)


def write_csv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _model_output_dirname(model: str) -> str:
    return str(model).replace("/", "__") or "unknown"


def _row_models(row: dict) -> List[str]:
    model = row.get("model") or row.get("model_dir")
    if model:
        return [str(model)]
    condition_model = row.get("condition_model")
    if condition_model:
        return [_split_cm_key(str(condition_model))[0]]
    models = row.get("models")
    if isinstance(models, list):
        return [str(value) for value in models]
    if isinstance(models, str) and models:
        return [part.strip() for part in models.split(",") if part.strip()]
    return []


def _row_belongs_to_model(row: dict, model: str) -> bool:
    models = _row_models(row)
    return not models or model in models


def _filter_rows_for_model(rows: List[dict], model: str) -> List[dict]:
    out: List[dict] = []
    for row in rows:
        if not _row_belongs_to_model(row, model):
            continue
        filtered = dict(row)
        if isinstance(filtered.get("models"), list):
            filtered["models"] = [model]
        elif isinstance(filtered.get("models"), str) and filtered.get("models"):
            filtered["models"] = model
        if "num_condition_model_rows" in filtered:
            filtered["num_condition_model_rows"] = 1
        out.append(filtered)
    return out


def _filter_json_for_model(filename: str, data: object, model: str, summary_rows: List[dict]) -> object:
    if filename == "summary.json":
        return summary_rows
    if filename == "variant_summary.json":
        return build_variant_summary(summary_rows)
    if isinstance(data, dict):
        filtered = {
            key: value
            for key, value in data.items()
            if _split_cm_key(str(key))[0] == model
        }
        if filtered:
            return filtered
        return data
    if isinstance(data, list):
        return _filter_rows_for_model([row for row in data if isinstance(row, dict)], model)
    return data


def _models_from_outputs(files: Dict[str, object], csv_outputs: Dict[str, Tuple[List[dict], List[str]]]) -> List[str]:
    models = set()
    summary = files.get("summary.json")
    if isinstance(summary, list):
        for row in summary:
            if isinstance(row, dict):
                models.update(_row_models(row))
    for rows, _fieldnames in csv_outputs.values():
        for row in rows:
            models.update(_row_models(row))
    for data in files.values():
        if isinstance(data, dict):
            for key in data:
                model, _variant = _split_cm_key(str(key))
                if model != "unknown":
                    models.add(model)
        elif isinstance(data, list):
            for row in data:
                if isinstance(row, dict):
                    models.update(_row_models(row))
    return sorted(models)


def write_per_model_outputs(
    out_dir: Path,
    files: Dict[str, object],
    csv_outputs: Dict[str, Tuple[List[dict], List[str]]],
) -> None:
    models = _models_from_outputs(files, csv_outputs)
    summary = files.get("summary.json")
    all_summary_rows = summary if isinstance(summary, list) else []

    for model in models:
        model_dir = out_dir / "by_model" / _model_output_dirname(model)
        model_dir.mkdir(parents=True, exist_ok=True)
        summary_rows = _filter_rows_for_model(
            [row for row in all_summary_rows if isinstance(row, dict)],
            model,
        )
        for filename, data in files.items():
            write_json(model_dir / filename, _filter_json_for_model(filename, data, model, summary_rows))
        for filename, (rows, fieldnames) in csv_outputs.items():
            write_csv(model_dir / filename, _filter_rows_for_model(rows, model), fieldnames)


def _import_plot_libs():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        print("Skipping figures: matplotlib is not installed.")
        return None


def _cleanup_figure_dir(fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    for path in fig_dir.glob("*.pdf"):
        path.unlink()


def _search_call_cumulative_fieldnames() -> List[str]:
    return [
        "condition_model",
        "condition",
        "variant",
        "base_condition",
        "model",
        "task_id",
        "search_tool",
        "turn",
        "search_call_index",
        "results_returned",
        "n_gold_datasets",
        "cumulative_search_gold_count",
        "cumulative_search_gold_recall",
        "cumulative_read_gold_count",
        "cumulative_read_gold_recall",
    ] + [
        field
        for cutoff in SEARCH_BOTTLENECK_CUTOFFS
        for field in (
            f"gold_hits_top_{cutoff}",
            f"gold_in_top_{cutoff}",
            f"gold_recall_top_{cutoff}",
        )
    ]


def _pretty_model(model_name: str) -> str:
    label = model_name.split("/")[-1]
    label = label.replace("bedrock_", "").replace("openai_", "")
    if label.endswith("-arn"):
        label = label[:-4]
    return label


def _pretty_cm_from_row(row: dict) -> str:
    return f"{_pretty_model(str(row.get('model', '')))} / {_compact_variant_label(str(row.get('variant', '')))}"


def _pretty_cm_from_key(key: str) -> str:
    model, variant = _split_cm_key(key)
    return f"{_pretty_model(model)} / {_compact_variant_label(variant)}"


def _pretty_variant_tick_label(variant: str, n_total: int) -> str:
    label = _compact_variant_label(variant, multiline=True)
    return f"{label}\n(n={n_total})"


def _hex_to_rgb(color: str) -> Tuple[float, float, float]:
    value = color.lstrip("#")
    if len(value) != 6:
        return (0.0, 0.0, 0.0)
    return tuple(int(value[idx : idx + 2], 16) / 255.0 for idx in (0, 2, 4))


def _label_text_color(fill_color: str) -> str:
    red, green, blue = _hex_to_rgb(fill_color)
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "black" if luminance > 0.62 else "white"


def _display_semantic_bucket(bucket: str) -> str:
    return SEMANTIC_BUCKET_DISPLAY.get(bucket, bucket)


def _display_log_error_bucket(bucket: str) -> str:
    return LOG_ERROR_BUCKET_DISPLAY.get(bucket, bucket)


def _display_tool_name(tool_name: str) -> str:
    return {
        "execute_code": "execute",
        "query_file": "query",
        "peek_file": "peek",
        "peek_multiple": "peek_multi",
        "grep_file": "grep",
        "list_files": "list",
        "read_file": "read",
        "search_ideal": "search_ideal",
        "search_reranked": "search_reranked",
        "search_prefix": "search_prefix",
        "search_schema": "search_schema",
        "search_value": "search_value",
    }.get(tool_name, tool_name)


def _annotate_pct_bars(ax, bars, values: List[float], *, min_value: float = 4.0, digits: int = 0) -> None:
    for bar, value in zip(bars, values):
        if value < min_value:
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.0,
            f"{value:.{digits}f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def _plot_grouped_metrics_bars(
    plt,
    rows: List[dict],
    *,
    fields: List[Tuple[str, str, str]],
    label_fn,
    title: str,
    y_label: str,
    output_path: Path,
) -> None:
    if not rows:
        return
    fig_width = max(10, len(rows) * 1.2)
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    x_positions = list(range(len(rows)))
    width = 0.8 / len(fields)
    for idx, (field, label, color) in enumerate(fields):
        values = [float(row.get(field) or 0.0) * 100.0 for row in rows]
        offset = (idx - len(fields) / 2 + 0.5) * width
        bars = ax.bar([x + offset for x in x_positions], values, width=width * 0.9, label=label, color=color)
        _annotate_pct_bars(ax, bars, values)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([label_fn(row) for row in rows], rotation=0, ha="center", fontsize=9)
    ax.set_ylabel(y_label)
    ax.set_ylim(0, 100)
    ax.set_title(title)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.tight_layout(rect=(0, 0, 0.84, 1))
    fig.savefig(output_path)
    plt.close(fig)


def _plot_discovery_semantic_combined(plt, summary_rows: List[dict], output_path: Path) -> None:
    if not summary_rows:
        return
    ordered_rows = sorted(
        summary_rows,
        key=lambda row: _condition_model_sort_key(_cm_key(str(row.get("model", "")), str(row.get("variant", "")))),
    )
    _plot_grouped_metrics_bars(
        plt,
        ordered_rows,
        fields=[
            ("D_ret", "D_ret", "#4C78A8"),
            ("D_acc", "D_acc", "#F58518"),
            ("semantic_match", "Semantic Match", "#54A24B"),
        ],
        label_fn=_pretty_cm_from_row,
        title="Discovery Coverage & Semantic Match by Variant × Model",
        y_label="Rate (%)",
        output_path=output_path,
    )


def _plot_cost_vs_semantic(plt, summary_rows: List[dict], output_path: Path) -> None:
    if not summary_rows:
        return
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for row in summary_rows:
        semantic_match = row.get("semantic_match")
        cost = row.get("avg_total_cost_with_ideal_subagents_usd")
        if cost is None:
            cost = row.get("avg_cost_usd")
        if semantic_match is None or cost is None:
            continue
        ax.scatter(cost, float(semantic_match) * 100.0, s=80, zorder=3)
        ax.annotate(
            _compact_variant_label(str(row.get("variant", ""))),
            (cost, float(semantic_match) * 100.0),
            textcoords="offset points",
            xytext=(5, 3),
            fontsize=7,
        )
    ax.set_xlabel("Avg Cost per Task incl. Ideal Subagents (USD)")
    ax.set_ylabel("Semantic Match (%)")
    ax.set_title("Cost–Semantic Match Frontier")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_tool_precision_recall_f1(plt, tools_discovery: dict, output_path: Path) -> None:
    tool_acc: Dict[str, dict] = defaultdict(lambda: {"weight": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0})
    for tool_data in tools_discovery.values():
        search_tool_names = [tool for tool in tool_data if tool.startswith("search")]
        for tool_name in search_tool_names:
            stats = tool_data.get(tool_name, {})
            weight = float(stats.get("n_tasks", 0) or stats.get("calls", 0) or 1)
            bucket = tool_acc[tool_name]
            bucket["weight"] += weight
            bucket["precision"] += float(stats.get("avg_precision", 0.0) or 0.0) * weight
            bucket["recall"] += float(stats.get("avg_recall", 0.0) or 0.0) * weight
            bucket["f1"] += float(stats.get("avg_f1", 0.0) or 0.0) * weight

    rows = []
    for tool_name, totals in sorted(tool_acc.items()):
        weight = float(totals.get("weight", 0.0) or 0.0)
        if weight <= 0:
            continue
        rows.append(
            {
                "tool": tool_name,
                "precision": totals["precision"] / weight,
                "recall": totals["recall"] / weight,
                "f1": totals["f1"] / weight,
            }
        )
    if not rows:
        return
    ordered_rows = sorted(rows, key=lambda row: str(row["tool"]))
    x_positions = list(range(len(ordered_rows)))
    fields = [
        ("precision", "Precision", "#4C78A8"),
        ("recall", "Recall", "#F58518"),
        ("f1", "F1", "#54A24B"),
    ]
    width = 0.8 / len(fields)
    fig, ax = plt.subplots(figsize=(max(10, len(ordered_rows) * 1.2), 5.5))
    for idx, (field, label, color) in enumerate(fields):
        values = [float(row.get(field, 0.0) or 0.0) * 100.0 for row in ordered_rows]
        offset = (idx - len(fields) / 2 + 0.5) * width
        bars = ax.bar([x + offset for x in x_positions], values, width=width * 0.9, label=label, color=color)
        _annotate_pct_bars(ax, bars, values)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([_display_tool_name(str(row["tool"])) for row in ordered_rows], rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Average Score (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Search Tool Precision, Recall, and F1")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
    fig.tight_layout(rect=(0, 0, 0.84, 1))
    fig.savefig(output_path)
    plt.close(fig)


def _plot_search_calls(plt, summary_rows: List[dict], output_path: Path) -> None:
    if not summary_rows:
        return
    fig, ax = plt.subplots(figsize=(max(10, len(summary_rows) * 0.9), 5.5))
    vals = [float(row.get("avg_search_calls") or 0.0) for row in summary_rows]
    bars = ax.bar(range(len(summary_rows)), vals, color="#72B7B2")
    for bar, val in zip(bars, vals):
        if val <= 0:
            continue
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.08, f"{val:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(summary_rows)))
    ax.set_xticklabels([_compact_variant_label(str(row.get("variant", "")), multiline=True) for row in summary_rows], rotation=0, ha="center", fontsize=9)
    ax.set_ylabel("Avg Search Calls / Task")
    ax.set_title("Search Calls by Variant × Model")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_curve_by_variant(plt, curve: dict, *, title: str, y_label: str, x_label: str, output_path: Path) -> None:
    if not curve:
        return
    ordered_variants = sorted(curve.keys(), key=_variant_sort_key)
    bin_labels = [key for key in next(iter(curve.values())).keys() if not key.startswith("_")]
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for variant in ordered_variants:
        xs, ys = [], []
        for idx, label in enumerate(bin_labels):
            entry = curve[variant].get(label, {})
            mean_value = entry.get("mean_semantic_match")
            n = entry.get("n", 0)
            if mean_value is not None and n > 0:
                xs.append(idx)
                ys.append(float(mean_value) * 100.0)
        if xs:
            ax.plot(xs, ys, marker="o", label=_compact_variant_label(variant))
            for xi, yi in zip(xs, ys):
                ax.annotate(f"{yi:.1f}%", (xi, yi), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
    ax.set_xticks(range(len(bin_labels)))
    ax.set_xticklabels(bin_labels)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_curve_by_model(
    plt,
    curve_by_cm: dict,
    *,
    x_label: str,
    title_prefix: str,
    output_prefix: str,
    output_suffix: str,
    output_dir: Path,
) -> None:
    by_model: Dict[str, Dict[str, dict]] = defaultdict(dict)
    for key, bins in curve_by_cm.items():
        model, variant = _split_cm_key(key)
        by_model[model][variant] = bins
    for model, variant_map in sorted(by_model.items()):
        if not variant_map:
            continue
        bin_labels = [key for key in next(iter(variant_map.values())).keys() if key not in ("model", "variant")]
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        for variant, bins in sorted(variant_map.items(), key=lambda item: _variant_sort_key(item[0])):
            xs, ys = [], []
            for idx, label in enumerate(bin_labels):
                entry = bins.get(label, {})
                mean_value = entry.get("mean_semantic_match")
                n = entry.get("n", 0)
                if mean_value is not None and n > 0:
                    xs.append(idx)
                    ys.append(float(mean_value) * 100.0)
            if xs:
                ax.plot(xs, ys, marker="o", label=_compact_variant_label(variant))
        ax.set_xticks(range(len(bin_labels)))
        ax.set_xticklabels(bin_labels)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Mean Semantic Match (%)")
        ax.set_title(f"{title_prefix} — {_pretty_model(model)}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fname = f"{output_prefix}_{_safe_slug(_pretty_model(model))}_{output_suffix}.pdf"
        fig.savefig(output_dir / fname)
        plt.close(fig)


_TOOL_ERROR_MIN_CALLS = 10


def _plot_tool_error_heatmap(plt, tool_errors: dict, output_path: Path) -> None:
    filtered_tools = sorted({tool for cm_data in tool_errors.values() for tool in cm_data if tool in _DATA_TOOLS})
    cm_keys = sorted(tool_errors.keys(), key=_condition_model_sort_key)
    if not filtered_tools or not cm_keys:
        return
    import numpy as np

    matrix = np.full((len(filtered_tools), len(cm_keys)), np.nan)
    calls = np.zeros((len(filtered_tools), len(cm_keys)), dtype=int)
    for j, cm in enumerate(cm_keys):
        for i, tool in enumerate(filtered_tools):
            cell = tool_errors[cm].get(tool, {})
            total = int(cell.get("total_calls", 0) or 0)
            calls[i, j] = total
            if total >= _TOOL_ERROR_MIN_CALLS:
                matrix[i, j] = float(cell.get("error_rate", 0.0) or 0.0) * 100.0

    masked = np.ma.masked_invalid(matrix)
    cmap = plt.get_cmap("Reds").copy()
    cmap.set_bad(color="#e6e6e6")
    finite_vals = masked.compressed()
    vmax = max(float(finite_vals.max()), 1.0) if finite_vals.size else 1.0

    fig, ax = plt.subplots(figsize=(max(6, len(cm_keys) * 2.0), max(4, len(filtered_tools) * 0.7)))
    im = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=0, vmax=vmax)
    plt.colorbar(im, ax=ax, label="Error Rate (%)")
    ax.set_xticks(range(len(cm_keys)))
    ax.set_xticklabels([_compact_variant_label(_split_cm_key(cm)[1], multiline=True) for cm in cm_keys], rotation=0, ha="center", fontsize=9)
    ax.set_yticks(range(len(filtered_tools)))
    ax.set_yticklabels([_display_tool_name(tool) for tool in filtered_tools], fontsize=9)
    for i in range(len(filtered_tools)):
        for j in range(len(cm_keys)):
            total = calls[i, j]
            if total <= 0:
                continue
            if total < _TOOL_ERROR_MIN_CALLS:
                ax.text(j, i, f"n={total}", ha="center", va="center", fontsize=7, color="#555555")
                continue
            val = matrix[i, j]
            if val <= 0:
                continue
            color = "white" if val > 25 else "black"
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center", fontsize=8, color=color)
    ax.set_title(
        f"Tool Error Rates by Variant × Model (%)  —  cells with <{_TOOL_ERROR_MIN_CALLS} calls shown as n=X"
    )
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_tool_call_heatmap(plt, base_by_key_records: Dict[str, List[dict]], output_path: Path) -> None:
    import numpy as np

    tc_acc: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    n_tasks_by_cm: Dict[str, int] = defaultdict(int)
    for key, rows in base_by_key_records.items():
        n_tasks_by_cm[key] += len(rows)
        for row in rows:
            for tc in row.get("tool_counts", []):
                name = tc.get("name", "")
                if not name or name in _UNIMPORTANT_TOOLS:
                    continue
                tc_acc[key][name] += int(tc.get("call_count", 0) or 0)
    cm_keys = sorted(tc_acc.keys(), key=_condition_model_sort_key)
    all_tools = sorted({tool for cm_data in tc_acc.values() for tool in cm_data})
    if not cm_keys or not all_tools:
        return
    matrix = np.zeros((len(all_tools), len(cm_keys)))
    for j, cm in enumerate(cm_keys):
        n = n_tasks_by_cm.get(cm, 0)
        for i, tool in enumerate(all_tools):
            matrix[i, j] = (tc_acc[cm].get(tool, 0) / n) if n else 0.0

    from matplotlib.colors import LinearSegmentedColormap

    fig, ax = plt.subplots(figsize=(max(6, len(cm_keys) * 2.0), max(4, len(all_tools) * 0.7)))
    vmax = max(1.0, float(matrix.max()))
    white_green = LinearSegmentedColormap.from_list("white_green", ["#ffffff", "#1b7837"])
    im = ax.imshow(matrix, aspect="auto", cmap=white_green, vmin=0, vmax=vmax)
    plt.colorbar(im, ax=ax, label="Avg Calls / Task")
    ax.set_xticks(range(len(cm_keys)))
    ax.set_xticklabels([_compact_variant_label(_split_cm_key(cm)[1], multiline=True) for cm in cm_keys], rotation=0, ha="center", fontsize=9)
    ax.set_yticks(range(len(all_tools)))
    ax.set_yticklabels([_display_tool_name(tool) for tool in all_tools], fontsize=9)
    for i in range(len(all_tools)):
        for j in range(len(cm_keys)):
            val = matrix[i, j]
            if val <= 0:
                continue
            color = "white" if val > vmax * 0.6 else "black"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=8, color=color)
    ax.set_title("Avg Tool Calls per Task by Variant × Model")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_stacked_buckets_on_ax(
    ax,
    rows: List[dict],
    *,
    buckets: List[str],
    colors: Dict[str, str],
    label_fn,
    total_field: str,
    title: str,
    show_legend: bool = True,
    label_min_pct: float = 6.0,
    rate_field_suffix: Optional[str] = None,
    annotate_counts: bool = True,
) -> None:
    if not rows:
        return

    labels = [label_fn(row) for row in rows]
    x_positions = list(range(len(rows)))
    bottoms = [0.0 for _ in rows]

    for bucket in buckets:
        heights = []
        counts = []
        for row in rows:
            total = float(row.get(total_field) or 0.0)
            count = float(row.get(bucket) or 0.0)
            counts.append(int(count))
            if rate_field_suffix:
                rate = float(row.get(f"{bucket}{rate_field_suffix}") or 0.0)
                heights.append(rate * 100.0)
            else:
                heights.append((count / total) * 100.0 if total else 0.0)
        display_bucket = _display_semantic_bucket(bucket) if bucket in SEMANTIC_BUCKETS else _display_log_error_bucket(bucket)
        bars = ax.bar(x_positions, heights, bottom=bottoms, color=colors[bucket], label=display_bucket)
        label_color = _label_text_color(colors[bucket])
        for bar, height, count, bottom in zip(bars, heights, counts, bottoms):
            if count <= 0 or height < label_min_pct:
                continue
            label_text = f"{height:.0f}%"
            if annotate_counts:
                label_text += f"\n({count})"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bottom + height / 2,
                label_text,
                ha="center",
                va="center",
                fontsize=8,
                color=label_color,
            )
        bottoms = [bottom + height for bottom, height in zip(bottoms, heights)]

    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=9)
    ax.set_ylabel("Share of Tasks (%)")
    ax.set_ylim(0, 100)
    ax.set_title(title)
    if show_legend:
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)


def _plot_variant_crosstab_heatmap(plt, crosstab_rows: List[dict], variant_rows: List[dict], output_path: Path) -> None:
    if not variant_rows:
        return
    ordered_variants = [str(row.get("variant", "")) for row in variant_rows]
    grouped = defaultdict(dict)
    max_value = 0.0
    for row in crosstab_rows:
        if row.get("level") != "variant":
            continue
        variant = str(row.get("variant", ""))
        grouped[variant][(str(row.get("semantic_bucket", "")), str(row.get("log_error_bucket", "")))] = (
            float(row.get("pct_within_group", 0.0) or 0.0) * 100.0
        )
        max_value = max(max_value, grouped[variant][(str(row.get("semantic_bucket", "")), str(row.get("log_error_bucket", "")))])

    ncols = 2 if len(ordered_variants) > 1 else 1
    nrows = math.ceil(len(ordered_variants) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.5 * ncols, 4.8 * nrows))
    axes_list = axes.flatten().tolist() if hasattr(axes, "flatten") else [axes]
    image = None
    for ax, variant in zip(axes_list, ordered_variants):
        matrix = []
        for semantic_bucket in SEMANTIC_BUCKETS:
            matrix.append(
                [
                    grouped.get(variant, {}).get((semantic_bucket, log_error_bucket), 0.0)
                    for log_error_bucket in DISPLAY_LOG_ERROR_BUCKETS
                ]
            )
        image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(max_value, 1.0))
        total_n = int(next((row.get("n_total", 0) for row in variant_rows if str(row.get("variant", "")) == variant), 0) or 0)
        ax.set_title(_pretty_variant_tick_label(variant, total_n), fontsize=10)
        ax.set_xticks(range(len(DISPLAY_LOG_ERROR_BUCKETS)))
        ax.set_xticklabels([_display_log_error_bucket(bucket) for bucket in DISPLAY_LOG_ERROR_BUCKETS], rotation=25, ha="right", fontsize=8)
        ax.set_yticks(range(len(SEMANTIC_BUCKETS)))
        ax.set_yticklabels([_display_semantic_bucket(bucket) for bucket in SEMANTIC_BUCKETS], fontsize=8)
        for row_idx, matrix_row in enumerate(matrix):
            for col_idx, value in enumerate(matrix_row):
                if value <= 0:
                    continue
                ax.text(col_idx, row_idx, f"{value:.0f}%", ha="center", va="center", fontsize=8)

    for ax in axes_list[len(ordered_variants):]:
        ax.axis("off")
    if image is not None:
        fig.colorbar(image, ax=axes_list, fraction=0.02, pad=0.02, label="Share of Tasks (%)")
    fig.suptitle("Semantic Bucket × Log Error Bucket by Variant", fontsize=14)
    fig.subplots_adjust(top=0.88, wspace=0.35, hspace=0.45)
    fig.savefig(output_path)
    plt.close(fig)


def _plot_error_vs_semantic_variant(plt, variant_rows: List[dict], output_path: Path) -> None:
    if not variant_rows:
        return

    fig, ax = plt.subplots(figsize=(max(11, len(variant_rows) * 1.45), 6))
    x_positions = list(range(len(variant_rows)))
    width = 0.34
    semantic_vals = [float(row.get("semantic_match", 0.0) or 0.0) * 100.0 for row in variant_rows]
    error_vals = [100.0 - (float(row.get("no_error_rate", 0.0) or 0.0) * 100.0) for row in variant_rows]

    semantic_bars = ax.bar([x - width / 2 for x in x_positions], semantic_vals, width=width, color=SEMANTIC_BUCKET_COLORS["semantic_correct"], label="Semantic Match")
    error_bars = ax.bar([x + width / 2 for x in x_positions], error_vals, width=width, color="#C44E52", label="Any Error")
    _annotate_pct_bars(ax, semantic_bars, semantic_vals, min_value=1.0)
    _annotate_pct_bars(ax, error_bars, error_vals, min_value=1.0)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [_pretty_variant_tick_label(str(row.get("variant", "")), int(row.get("n_total", 0) or 0)) for row in variant_rows],
        rotation=0,
        ha="center",
        fontsize=9,
    )
    ax.set_ylabel("Tasks (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Semantic Match vs. Any Error by Variant")
    ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.7)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=9)
    fig.tight_layout(rect=(0, 0, 0.84, 1))
    fig.savefig(output_path)
    plt.close(fig)


def _turn_waste_group_sort_key(group_name: str, total_counts: Counter) -> tuple:
    normalized = group_name.lower()
    is_mixed = "mixed" in normalized or "unclear" in normalized
    return (1 if is_mixed else 0, -int(total_counts.get(group_name, 0)), group_name)


def _turn_waste_family(group_name: str) -> str:
    if group_name in TURN_WASTE_CANONICAL_GROUP_TO_FAMILY:
        return TURN_WASTE_CANONICAL_GROUP_TO_FAMILY[group_name]
    normalized = group_name.lower()
    if not normalized or "mixed" in normalized or "unclear" in normalized:
        return "other_mixed"
    if (
        "structured data" in normalized
        or "source and query" in normalized
        or "query recovery" in normalized
        or "extraction loop" in normalized
    ):
        return "data_query_loops"
    if (
        "intermediate result" in normalized
        or "redundant revalidation" in normalized
        or "final verification" in normalized
        or "low-yield detour" in normalized
        or "rework" in normalized
    ):
        return "redundant_rework"
    if "missing source" in normalized or "wrong-branch" in normalized or "off-path" in normalized:
        return "search_path_drift"
    if "post-answer" in normalized or "non-submission" in normalized or "partial component stall" in normalized:
        return "progress_submission_failures"
    return "other_mixed"


def _build_turn_waste_condition_model_summary(summary_rows: List[dict], failed_join_rows: List[dict]) -> dict:
    summary_by_key = {str(row.get("condition_model", "")): row for row in summary_rows}
    out = {
        key: {
            "condition_model": key,
            "model": str(row.get("model", "")),
            "variant": str(row.get("variant", "")),
            "n_total_rows": int(row.get("n", 0) or 0),
            "n_failed_rows": 0,
            "n_grouped_failed_rows": 0,
            "n_unassigned_failed_rows": 0,
            "families": {family: {"n": 0} for family in TURN_WASTE_FAMILY_DISPLAY},
            "groups": {},
        }
        for key, row in summary_by_key.items()
    }

    for row in failed_join_rows:
        key = str(row.get("condition_model", ""))
        entry = out.setdefault(
            key,
            {
                "condition_model": key,
                "model": _split_cm_key(key)[0],
                "variant": _split_cm_key(key)[1],
                "n_total_rows": 0,
                "n_failed_rows": 0,
                "n_grouped_failed_rows": 0,
                "n_unassigned_failed_rows": 0,
                "families": {family: {"n": 0} for family in TURN_WASTE_FAMILY_DISPLAY},
                "groups": {},
            },
        )
        entry["n_failed_rows"] += 1
        group_name = str(row.get("turn_waste_global_group", "") or "").strip()
        if group_name:
            reconciled_group_name = reconcile_global_group(group_name)
            entry["n_grouped_failed_rows"] += 1
            entry.setdefault("groups", {})
            entry["groups"].setdefault(reconciled_group_name, {"n": 0})
            entry["groups"][reconciled_group_name]["n"] += 1
            family = _turn_waste_family(reconciled_group_name)
            entry["families"].setdefault(family, {"n": 0})
            entry["families"][family]["n"] += 1
        else:
            entry["n_unassigned_failed_rows"] += 1

    for entry in out.values():
        grouped_total = int(entry.get("n_grouped_failed_rows", 0) or 0)
        for payload in entry.get("families", {}).values():
            count = int(payload.get("n", 0) or 0)
            payload["pct_within_grouped_failed_rows"] = round(count / grouped_total, 4) if grouped_total else 0.0
    return out


def _turn_waste_plot_rows(condition_groups: dict, summary_rows: List[dict]) -> List[dict]:
    rows: List[dict] = []
    for row in sorted(summary_rows, key=lambda item: _condition_model_sort_key(str(item.get("condition_model", "")))):
        key = str(row.get("condition_model", ""))
        entry = condition_groups.get(key)
        if not entry:
            continue
        failed = int(entry.get("n_failed_rows", 0) or 0)
        if failed <= 0:
            continue
        variant = str(row.get("variant", entry.get("variant", "")))
        families = {
            display: int(entry.get("families", {}).get(family, {}).get("n", 0) or 0)
            for family, display in TURN_WASTE_FAMILY_DISPLAY.items()
        }
        assigned = int(entry.get("n_grouped_failed_rows", 0) or 0)
        unclassified = int(entry.get("n_unassigned_failed_rows", 0) or 0)
        rows.append(
            {
                "condition_model": key,
                "model": str(row.get("model", entry.get("model", ""))),
                "variant": variant,
                "label": _standard_condition_label(variant),
                "failed": failed,
                "assigned": assigned,
                "unclassified": unclassified,
                "families": families,
            }
        )
    return rows


def _is_turn_waste_figure_group(group_name: str) -> bool:
    return str(group_name or "").strip() not in TURN_WASTE_FIGURE_EXCLUDED_GROUPS


def _ordered_turn_waste_groups(group_totals: Counter) -> List[str]:
    preferred = [
        group
        for group in TURN_WASTE_PREFERRED_GROUP_ORDER
        if int(group_totals.get(group, 0) or 0) > 0 and _is_turn_waste_figure_group(group)
    ]
    extras = sorted(
        group
        for group, count in group_totals.items()
        if int(count or 0) > 0 and group not in preferred and _is_turn_waste_figure_group(group)
    )
    return preferred + extras


def _plot_turn_waste_groups_variant(plt, turn_waste_groups: dict, variant_rows: List[dict], output_path: Path) -> None:
    if not variant_rows or not turn_waste_groups:
        return

    ordered_variants = [str(row.get("variant", "")) for row in variant_rows]
    overall_counts: Counter = Counter()
    for variant in ordered_variants:
        groups = turn_waste_groups.get(variant, {}).get("groups", {})
        for group_name, payload in groups.items():
            overall_counts[group_name] += int(payload.get("n", 0) or 0)

    ordered_groups = sorted(overall_counts.keys(), key=lambda group_name: _turn_waste_group_sort_key(group_name, overall_counts))
    fig, ax = plt.subplots(figsize=(max(10, len(ordered_variants) * 1.25), 6))
    x_positions = list(range(len(ordered_variants)))
    bottoms = [0 for _ in ordered_variants]
    cmap = plt.get_cmap("tab20")
    colors = {group_name: cmap(idx % 20) for idx, group_name in enumerate(ordered_groups)}

    for group_name in ordered_groups:
        heights = []
        counts = []
        for variant in ordered_variants:
            entry = turn_waste_groups.get(variant, {})
            n_grouped_failed_rows = int(entry.get("n_grouped_failed_rows", 0) or 0)
            group_payload = entry.get("groups", {}).get(group_name, {})
            count = int(group_payload.get("n", 0) or 0)
            counts.append(count)
            heights.append(count if n_grouped_failed_rows else 0)
        bars = ax.bar(x_positions, heights, bottom=bottoms, color=colors[group_name], label=group_name)
        label_color = _label_text_color(
            "#{:02x}{:02x}{:02x}".format(
                int(colors[group_name][0] * 255),
                int(colors[group_name][1] * 255),
                int(colors[group_name][2] * 255),
            )
        )
        for bar, count, height, bottom in zip(bars, counts, heights, bottoms):
            if count <= 0:
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bottom + height / 2,
                str(count),
                ha="center",
                va="center",
                fontsize=8,
                color=label_color,
            )
        bottoms = [bottom + height for bottom, height in zip(bottoms, heights)]

    for x_pos, variant, grouped_total in zip(x_positions, ordered_variants, bottoms):
        entry = turn_waste_groups.get(variant, {})
        unassigned = int(entry.get("n_unassigned_failed_rows", 0) or 0)
        failed = int(entry.get("n_failed_rows", 0) or 0)
        top_label = f"{grouped_total}"
        if unassigned > 0:
            top_label += f" (+{unassigned} unassigned)"
        elif failed > grouped_total:
            top_label += f" / {failed}"
        if grouped_total > 0 or unassigned > 0:
            ax.text(
                x_pos,
                grouped_total + max(0.3, grouped_total * 0.03),
                top_label,
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [
            (
                f"{_compact_variant_label(variant, multiline=True)}\n"
                f"g={turn_waste_groups.get(variant, {}).get('n_grouped_failed_rows', 0)} / "
                f"f={turn_waste_groups.get(variant, {}).get('n_failed_rows', 0)}"
            )
            for variant in ordered_variants
        ],
        rotation=0,
        ha="center",
        fontsize=9,
    )
    ymax = max(max(bottoms, default=0), max(int(turn_waste_groups.get(variant, {}).get("n_failed_rows", 0) or 0) for variant in ordered_variants))
    ax.set_ylabel("Grouped Failed Rows")
    ax.set_ylim(0, max(1, ymax * 1.18))
    ax.set_title("Canonical Turn-Waste Groups by Variant (Counts)")
    ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.7)
    if ordered_groups:
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
    fig.tight_layout(rect=(0, 0, 0.8, 1))
    fig.savefig(output_path)
    plt.close(fig)


def _plot_turn_waste_families_by_condition_model(plt, condition_groups: dict, summary_rows: List[dict], output_path: Path) -> None:
    rows = _turn_waste_plot_rows(condition_groups, summary_rows)
    if not rows:
        fig, ax = plt.subplots(figsize=(8, 2.8))
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "No failed rows available for turn-waste grouping.",
            ha="center",
            va="center",
            fontsize=10,
        )
        fig.suptitle("Turn-Waste Audit Coverage and Assigned Failure Patterns", fontsize=13)
        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)
        return

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.5, max(4.8, 0.43 * len(rows) + 1.7)),
        sharey=True,
        squeeze=False,
    )
    coverage_ax, family_ax = axes[0]
    y_positions = list(range(len(rows)))
    y_labels = [
        f"{_pretty_model(str(row.get('model', '')))}\n{row.get('label', '')}"
        for row in rows
    ]

    classified_color = "#5C6670"
    coverage_specs = [
        ("Classified", classified_color, [int(row.get("assigned", 0) or 0) for row in rows]),
        (TURN_WASTE_UNCLASSIFIED_DISPLAY, TURN_WASTE_UNCLASSIFIED_COLOR, [int(row.get("unclassified", 0) or 0) for row in rows]),
    ]
    lefts = [0.0 for _ in rows]
    coverage_handles = []
    coverage_labels = []
    for display, color, counts in coverage_specs:
        widths = [
            (count / int(row.get("failed", 0) or 0)) * 100.0 if int(row.get("failed", 0) or 0) else 0.0
            for row, count in zip(rows, counts)
        ]
        bars = coverage_ax.barh(y_positions, widths, left=lefts, height=0.72, color=color, label=display)
        coverage_handles.append(bars[0])
        coverage_labels.append(display)
        label_color = _label_text_color(color)
        for bar, width, count, left in zip(bars, widths, counts, lefts):
            if count <= 0 or width < 7.0:
                continue
            coverage_ax.text(
                left + width / 2,
                bar.get_y() + bar.get_height() / 2,
                str(count),
                ha="center",
                va="center",
                fontsize=8,
                color=label_color,
            )
        lefts = [left + width for left, width in zip(lefts, widths)]

    coverage_ax.set_yticks(y_positions)
    coverage_ax.set_yticklabels(y_labels, fontsize=8)
    coverage_ax.invert_yaxis()
    coverage_ax.set_xlim(0, 100)
    coverage_ax.set_xlabel("Share of failed rows (%)")
    coverage_ax.set_title("Audit coverage")
    coverage_ax.grid(axis="x", alpha=0.2, linestyle="--", linewidth=0.7)
    for y_pos, row in zip(y_positions, rows):
        coverage_ax.text(101.0, y_pos, f"n={int(row.get('failed', 0) or 0)}", ha="left", va="center", fontsize=8)

    family_display_to_color = {
        display: TURN_WASTE_FAMILY_COLORS[family]
        for family, display in TURN_WASTE_FAMILY_DISPLAY.items()
    }
    family_lefts = [0.0 for _ in rows]
    family_handles = []
    family_labels = []
    active_families = [
        display
        for display in TURN_WASTE_FAMILY_DISPLAY.values()
        if any(int(row.get("families", {}).get(display, 0) or 0) > 0 for row in rows)
    ]
    for display in active_families:
        counts = [int(row.get("families", {}).get(display, 0) or 0) for row in rows]
        widths = [
            (count / int(row.get("assigned", 0) or 0)) * 100.0 if int(row.get("assigned", 0) or 0) else 0.0
            for row, count in zip(rows, counts)
        ]
        color = family_display_to_color[display]
        bars = family_ax.barh(y_positions, widths, left=family_lefts, height=0.72, color=color, label=display)
        family_handles.append(bars[0])
        family_labels.append(display)
        label_color = _label_text_color(color)
        for bar, width, count, left in zip(bars, widths, counts, family_lefts):
            if count <= 0 or width < 8.0:
                continue
            family_ax.text(
                left + width / 2,
                bar.get_y() + bar.get_height() / 2,
                str(count),
                ha="center",
                va="center",
                fontsize=8,
                color=label_color,
            )
        family_lefts = [left + width for left, width in zip(family_lefts, widths)]

    family_ax.set_xlim(0, 100)
    family_ax.set_xlabel("Share of classified rows (%)")
    family_ax.set_title("Assigned failure family")
    family_ax.grid(axis="x", alpha=0.2, linestyle="--", linewidth=0.7)

    fig.legend(
        handles=coverage_handles + family_handles,
        labels=coverage_labels + family_labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=8,
    )

    fig.suptitle("Turn-Waste Audit Coverage and Assigned Failure Patterns", fontsize=13)
    fig.tight_layout(rect=(0, 0.07, 1, 0.94), w_pad=3.0)
    fig.savefig(output_path)
    plt.close(fig)


def _plot_turn_waste_families_for_model(plt, rows: List[dict], model: str, output_path: Path) -> None:
    model_rows = [row for row in rows if str(row.get("model", "")) == model and int(row.get("assigned", 0) or 0) > 0]
    if not model_rows:
        return

    y_positions = list(range(len(model_rows)))
    height = max(2.8, 0.42 * len(model_rows) + 1.25)
    fig, ax = plt.subplots(figsize=(9.8, height))
    lefts = [0 for _ in model_rows]
    handles = []
    labels = []
    active_families = [
        (family, display)
        for family, display in TURN_WASTE_FAMILY_DISPLAY.items()
        if any(int(row.get("families", {}).get(display, 0) or 0) > 0 for row in model_rows)
    ]

    for family, display in active_families:
        values = [int(row.get("families", {}).get(display, 0) or 0) for row in model_rows]
        color = TURN_WASTE_FAMILY_COLORS[family]
        bars = ax.barh(y_positions, values, left=lefts, height=0.72, color=color, label=display)
        handles.append(bars[0])
        labels.append(display)
        label_color = _label_text_color(color)
        for bar, value, left in zip(bars, values, lefts):
            if value <= 0:
                continue
            current_total = max(sum(int(row.get("families", {}).get(label, 0) or 0) for label in TURN_WASTE_FAMILY_DISPLAY.values()) for row in model_rows)
            if value < max(1.5, current_total * 0.08):
                continue
            ax.text(
                left + value / 2,
                bar.get_y() + bar.get_height() / 2,
                str(value),
                ha="center",
                va="center",
                fontsize=8,
                color=label_color,
            )
        lefts = [left + value for left, value in zip(lefts, values)]

    ax.set_yticks(y_positions)
    ax.set_yticklabels([str(row.get("label", "")) for row in model_rows], fontsize=8)
    ax.set_ylabel("Search / Plan / Execution")
    ax.invert_yaxis()
    ax.set_xlabel("Classified failed rows")
    ax.set_title(f"Turn-Waste Families: {_pretty_model(model)}")
    ax.grid(axis="x", alpha=0.2, linestyle="--", linewidth=0.7)
    xmax = max(max(lefts, default=0), 1)
    ax.set_xlim(0, xmax + max(7, xmax * 0.7))
    for y_pos, row, assigned_total in zip(y_positions, model_rows, lefts):
        ax.text(
            assigned_total + max(0.12, xmax * 0.02),
            y_pos,
            f"n={assigned_total}",
            ha="left",
            va="center",
            fontsize=8,
        )

    if handles:
        ax.legend(handles, labels, loc="lower right", frameon=True, framealpha=0.95, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_turn_waste_families_by_model(plt, condition_groups: dict, summary_rows: List[dict], output_dir: Path) -> None:
    rows = _turn_waste_plot_rows(condition_groups, summary_rows)
    models = []
    for row in rows:
        model = str(row.get("model", ""))
        if model and model not in models:
            models.append(model)
    for model in models:
        _plot_turn_waste_families_for_model(
            plt,
            rows,
            model,
            output_dir / f"turn_waste_groups_{_safe_slug(_pretty_model(model))}.pdf",
        )


def _plot_turn_waste_reconciled_groups_by_model(
    plt,
    condition_groups: dict,
    output_path: Path,
) -> None:
    counts_by_group_model: Dict[str, Counter] = defaultdict(Counter)
    for entry in condition_groups.values():
        model = str(entry.get("model", ""))
        for group_name, payload in entry.get("groups", {}).items():
            if not _is_turn_waste_figure_group(str(group_name)):
                continue
            count = int(payload.get("n", 0) or 0)
            if count > 0:
                counts_by_group_model[str(group_name)][model] += count

    if not counts_by_group_model:
        return

    ordered_groups = sorted(
        counts_by_group_model.keys(),
        key=lambda group: (-sum(counts_by_group_model[group].values()), group),
    )
    preferred_models = ["openai_gpt-5.4-nano", "openai_gpt-5-mini", "gpt-5.4-nano", "gpt-5-mini"]
    observed_models = []
    for model in preferred_models:
        if any(counts_by_group_model[group].get(model, 0) for group in ordered_groups) and model not in observed_models:
            observed_models.append(model)
    for group_counts in counts_by_group_model.values():
        for model in sorted(group_counts):
            if model not in observed_models:
                observed_models.append(model)

    y_positions = list(range(len(ordered_groups)))
    bar_height = 0.34 if len(observed_models) <= 2 else max(0.18, 0.72 / max(1, len(observed_models)))
    fig, ax = plt.subplots(figsize=(10.5, max(3.3, 0.62 * len(ordered_groups) + 1.3)))

    max_count = max(max(counts.values(), default=0) for counts in counts_by_group_model.values())
    for model_idx, model in enumerate(observed_models):
        offset = (model_idx - (len(observed_models) - 1) / 2) * bar_height
        values = [int(counts_by_group_model[group].get(model, 0) or 0) for group in ordered_groups]
        color = TURN_WASTE_MODEL_COLORS.get(model, "#6F6F6F")
        bars = ax.barh(
            [y + offset for y in y_positions],
            values,
            height=bar_height * 0.88,
            color=color,
            label=_pretty_model(model),
        )
        for bar, value in zip(bars, values):
            if value <= 0:
                continue
            ax.text(
                value + max(0.15, max_count * 0.012),
                bar.get_y() + bar.get_height() / 2,
                str(value),
                ha="left",
                va="center",
                fontsize=8,
            )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(ordered_groups, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Grouped failed rows")
    ax.set_title("Reconciled Turn-Waste Error Groups by Model")
    ax.grid(axis="x", alpha=0.22, linestyle="--", linewidth=0.7)
    ax.set_xlim(0, max(1, max_count + max(4, max_count * 0.16)))
    ax.legend(loc="lower right", frameon=True, framealpha=0.95, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_turn_waste_reconciled_groups_by_condition(
    plt,
    condition_groups: dict,
    output_path: Path,
) -> None:
    counts_by_variant_model_group: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
    group_totals: Counter = Counter()
    condition_variants = {variant for _, variant in TURN_WASTE_CONDITION_FIGURE_ORDER}

    for entry in condition_groups.values():
        variant = str(entry.get("variant", ""))
        model = str(entry.get("model", ""))
        if variant not in condition_variants:
            continue
        for group_name, payload in entry.get("groups", {}).items():
            group_name = str(group_name)
            if not _is_turn_waste_figure_group(group_name):
                continue
            count = int(payload.get("n", 0) or 0)
            if count <= 0:
                continue
            counts_by_variant_model_group[(variant, model)][group_name] += count
            group_totals[group_name] += count

    ordered_groups = _ordered_turn_waste_groups(group_totals)
    active_conditions = [
        (label, variant)
        for label, variant in TURN_WASTE_CONDITION_FIGURE_ORDER
        if any(sum(counts.values()) > 0 for (observed_variant, _), counts in counts_by_variant_model_group.items() if observed_variant == variant)
    ]
    if not active_conditions or not ordered_groups:
        return

    observed_by_variant: Dict[str, set] = defaultdict(set)
    observed_models = set()
    for variant, model in counts_by_variant_model_group:
        observed_models.add(model)
        if sum(counts_by_variant_model_group[(variant, model)].values()) > 0:
            observed_by_variant[variant].add(model)
    model_order: List[str] = []
    for model in TURN_WASTE_CONDITION_FIGURE_MODEL_ORDER:
        if model in observed_models and model not in model_order:
            model_order.append(model)
    for model in sorted(observed_models):
        if model not in model_order:
            model_order.append(model)
    condition_models: List[Tuple[str, str, str]] = []
    for label, variant in active_conditions:
        for model in model_order:
            condition_models.append((label, variant, model))
    if not condition_models:
        return

    gap = 1.55
    x_positions: List[float] = []
    current_x = 0.0
    previous_variant = None
    for _, variant, _ in condition_models:
        if previous_variant is not None and variant != previous_variant:
            current_x += gap
        x_positions.append(current_x)
        current_x += 1.0
        previous_variant = variant
    totals = [
        sum(counts_by_variant_model_group.get((variant, model), Counter()).values())
        for _, variant, model in condition_models
    ]
    ymax = max(totals, default=0)
    fig, ax = plt.subplots(figsize=(15.6, 7.2))
    bottoms = [0 for _ in condition_models]

    for group_name in ordered_groups:
        values = [
            int(counts_by_variant_model_group.get((variant, model), Counter()).get(group_name, 0) or 0)
            for _, variant, model in condition_models
        ]
        color = TURN_WASTE_GROUP_COLORS.get(group_name, "#7F7F7F")
        bars = ax.bar(x_positions, values, bottom=bottoms, color=color, label=group_name, width=0.72)
        label_color = _label_text_color(color)
        for bar, value, bottom in zip(bars, values, bottoms):
            if value <= 0:
                continue
            if value >= max(2, ymax * 0.08):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bottom + value / 2,
                    str(value),
                    ha="center",
                    va="center",
                    fontsize=12,
                    color=label_color,
                )
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]

    for x_pos, total in zip(x_positions, totals):
        ax.text(
            x_pos,
            total + max(0.25, ymax * 0.025),
            str(total),
            ha="center",
            va="bottom",
            fontsize=13,
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [
            TURN_WASTE_CONDITION_FIGURE_MODEL_LABELS.get(model, _pretty_model(model).replace("gpt-", "gpt\n"))
            for _, _, model in condition_models
        ],
        fontsize=12,
    )
    for label, variant in active_conditions:
        variant_positions = [x for x, (_, observed_variant, _) in zip(x_positions, condition_models) if observed_variant == variant]
        if not variant_positions:
            continue
        center = sum(variant_positions) / len(variant_positions)
        ax.text(
            center,
            -0.135,
            TURN_WASTE_CONDITION_FIGURE_LABELS.get(label, label),
            ha="center",
            va="top",
            fontsize=15,
            transform=ax.get_xaxis_transform(),
        )
    ax.set_ylabel("Grouped failed rows", fontsize=15)
    ax.set_title("Turn-Waste Error Groups by SANA Condition and Model", fontsize=20, pad=12)
    ax.grid(axis="y", alpha=0.22, linestyle="--", linewidth=0.7)
    ax.set_ylim(0, max(1, ymax + max(3, ymax * 0.18)))
    ax.tick_params(axis="y", labelsize=13)
    ax.legend(loc="upper right", frameon=True, framealpha=0.94, fontsize=13)
    fig.tight_layout(rect=(0, 0.085, 1, 1))
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def generate_figures(
    summary_rows: List[dict],
    variant_rows: List[dict],
    failure: dict,
    tools_discovery: dict,
    tool_errors: dict,
    base_by_key_records: Dict[str, List[dict]],
    crosstab_rows: List[dict],
    turn_waste_global_groups: dict,
    turn_waste_condition_model_groups: dict,
    search_bottleneck: dict,
    search_depth_curve: dict,
    search_depth_curve_by_cm: dict,
    reasoning_density_curve: dict,
    reasoning_density_curve_by_cm: dict,
    output_dir: Path,
) -> None:
    fig_dir = output_dir / "figures"
    _cleanup_figure_dir(fig_dir)

    plt = _import_plot_libs()
    if plt is None:
        return

    _plot_stacked_buckets_on_ax  # keep referenced for lint happiness

    _plot_stacked_buckets = lambda rows, buckets, colors, label_fn, total_field, title, path, label_min_pct=6.0, rate_field_suffix=None, annotate_counts=True: (
        (lambda fig, ax: (
            _plot_stacked_buckets_on_ax(
                ax,
                rows,
                buckets=buckets,
                colors=colors,
                label_fn=label_fn,
                total_field=total_field,
                title=title,
                show_legend=True,
                label_min_pct=label_min_pct,
                rate_field_suffix=rate_field_suffix,
                annotate_counts=annotate_counts,
            ),
            fig.tight_layout(rect=(0, 0, 0.84, 1)),
            fig.savefig(path),
            plt.close(fig),
        ))(*plt.subplots(figsize=(max(10, len(rows) * 0.9), 6)))
        if rows
        else None
    )

    _plot_stacked_buckets(
        variant_rows,
        SEMANTIC_BUCKETS,
        SEMANTIC_BUCKET_COLORS,
        lambda row: _pretty_variant_tick_label(str(row.get("variant", "")), int(row.get("n_total", 0) or 0)),
        "n_total",
        "Semantic Bucket Distribution by Variant",
        fig_dir / "semantic_buckets_by_variant.pdf",
        label_min_pct=0.0,
    )
    _plot_stacked_buckets(
        variant_rows,
        DISPLAY_LOG_ERROR_BUCKETS,
        LOG_ERROR_BUCKET_COLORS,
        lambda row: _compact_variant_label(str(row.get("variant", "")), multiline=True),
        "n_total",
        "Log Error Bucket Distribution by Variant",
        fig_dir / "log_error_buckets_by_variant.pdf",
        rate_field_suffix="_rate",
        annotate_counts=False,
    )
    _plot_variant_crosstab_heatmap(
        plt,
        crosstab_rows,
        variant_rows,
        fig_dir / "semantic_error_crosstab_by_variant.pdf",
    )
    _plot_error_vs_semantic_variant(
        plt,
        variant_rows,
        fig_dir / "error_vs_semantic_by_variant.pdf",
    )
    _plot_turn_waste_reconciled_groups_by_model(
        plt,
        turn_waste_condition_model_groups,
        fig_dir / "turn_waste_groups_by_model.pdf",
    )
    _plot_turn_waste_reconciled_groups_by_condition(
        plt,
        turn_waste_condition_model_groups,
        fig_dir / "turn_waste_groups_by_condition.pdf",
    )

    _plot_discovery_semantic_combined(
        plt,
        summary_rows,
        fig_dir / "retrieval_vs_semantic_combined.pdf",
    )

    _plot_cost_vs_semantic(
        plt,
        summary_rows,
        fig_dir / "cost_vs_semantic.pdf",
    )

    _plot_tool_precision_recall_f1(
        plt,
        tools_discovery,
        fig_dir / "search_tool_precision_recall_f1.pdf",
    )

    _plot_search_calls(
        plt,
        summary_rows,
        fig_dir / "search_calls_by_variant.pdf",
    )

    _plot_curve_by_variant(
        plt,
        search_depth_curve,
        title="Search Depth Curve — Semantic Match vs. Search Call Budget Used",
        y_label="Mean Semantic Match (%)",
        x_label="Search Calls per Task",
        output_path=fig_dir / "search_depth_curve.pdf",
    )
    _plot_curve_by_model(
        plt,
        search_depth_curve_by_cm,
        x_label="Search Calls per Task",
        title_prefix="Search Depth Curve",
        output_prefix="search_depth",
        output_suffix="curve",
        output_dir=fig_dir,
    )

    _plot_curve_by_variant(
        plt,
        reasoning_density_curve,
        title="Semantic Match vs. Reasoning Density",
        y_label="Mean Semantic Match (%)",
        x_label="Number of Gold Documents per Task",
        output_path=fig_dir / "reasoning_density_curve.pdf",
    )
    _plot_curve_by_model(
        plt,
        reasoning_density_curve_by_cm,
        x_label="Number of Gold Documents per Task",
        title_prefix="Semantic Match vs. Reasoning Density",
        output_prefix="reasoning_density",
        output_suffix="curve",
        output_dir=fig_dir,
    )

    _plot_tool_error_heatmap(
        plt,
        tool_errors,
        fig_dir / "tool_error_rates.pdf",
    )

    _plot_tool_call_heatmap(
        plt,
        base_by_key_records,
        fig_dir / "tool_call_counts.pdf",
    )

    generate_search_bottleneck_figures(
        search_bottleneck,
        output_dir,
        include_condition_breakouts=True,
        condition_label_formatter=lambda row: _compact_variant_label(str(row.get("variant", ""))),
    )

    generate_delta_figures(summary_rows, fig_dir)


def run_analysis(
    *,
    results_dir: str,
    base_results_dir: Optional[str],
    turn_waste_grouped_dir: Optional[str],
    traces_dir: str,
    tasks_dir: str,
    output_dir: str,
    model_filter: Optional[str] = None,
    no_figures: bool = False,
) -> dict:
    model_filters = _parse_model_filters(model_filter)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading semantic eval results...")
    by_key_records, source_field_order = load_semantic_results_grouped(results_dir, model_filters=model_filters)

    print("Skipping grouped turn-waste load (legacy analysis removed).")

    print("Loading base agent results...")
    base_by_key_records = load_base_agent_results_grouped(base_results_dir, model_filters=model_filters)

    print("Loading traces for discovery metrics...")
    grouped_traces = load_traces_grouped(traces_dir, model_filters=model_filters)

    print("Computing discovery metrics...")
    discovery_aggregates, task_metrics_by_key = run_discovery(grouped_traces, tasks_dir)

    print("Computing search-style side analyses...")
    efficiency = run_efficiency(by_key_records)
    failure = build_failure(by_key_records)
    tools_discovery = run_tools_discovery(grouped_traces, tasks_dir)
    tool_errors = run_tool_errors(base_by_key_records)
    search_depth_curve, search_depth_curve_by_cm = build_search_depth_curves(by_key_records, task_metrics_by_key)
    reasoning_density_curve, reasoning_density_curve_by_cm = build_reasoning_density_curves(by_key_records, tasks_dir)
    search_depth_buckets = build_search_depth_buckets(by_key_records, task_metrics_by_key)
    reasoning_density_buckets = build_reasoning_density_buckets(by_key_records, tasks_dir)
    search_bottleneck_meta = build_search_bottleneck_meta(grouped_traces)
    search_task_gold = (
        load_task_gold_source_ids(tasks_dir)
        if "kramabench" in Path(tasks_dir).name
        else load_task_gold_ids(tasks_dir)
    )
    search_bottleneck = compute_search_bottleneck(grouped_traces, search_bottleneck_meta, search_task_gold)

    print("Building summary tables...")
    summary_rows = build_summary(
        by_key_records,
        discovery_aggregates,
        efficiency,
        tool_errors,
        base_by_key_records,
        search_bottleneck.get("condition_model_summary"),
    )
    variant_rows = build_variant_summary(summary_rows)
    semantic_match, discovery, runtime, semantic_buckets, log_error_buckets = build_metric_mappings(summary_rows)
    crosstab_rows = build_semantic_error_crosstab(by_key_records, variant_rows)
    turn_waste_global_groups, turn_waste_global_group_rows, turn_waste_joined_failed_rows = {}, [], []
    turn_waste_condition_model_groups = {}
    per_task_rows, per_task_fieldnames = build_per_task_rows(by_key_records, task_metrics_by_key, source_field_order)
    per_task_retrieval_rows, per_task_retrieval_fieldnames = build_per_task_retrieval_rows(task_metrics_by_key)

    files = {
        "semantic_match.json": semantic_match,
        "discovery.json": discovery,
        "runtime.json": runtime,
        "efficiency.json": efficiency,
        "failure.json": failure,
        "tools_discovery.json": tools_discovery,
        "tool_errors.json": tool_errors,
        "semantic_buckets.json": semantic_buckets,
        "log_error_buckets.json": log_error_buckets,
        "semantic_error_crosstab.json": crosstab_rows,
        "search_depth.json": search_depth_curve,
        "reasoning_density.json": reasoning_density_curve,
        "search_first_hit_condition.json": search_bottleneck.get("condition_summary_rows", []),
        "search_first_hit_tool.json": search_bottleneck.get("tool_first_hit_rows", []),
        "search_topk_miss_tool.json": search_bottleneck.get("tool_miss_rows", []),
        "search_tool_efficiency.json": search_bottleneck.get("tool_efficiency_rows", []),
        "search_depth_buckets.json": search_depth_buckets,
        "reasoning_density_buckets.json": reasoning_density_buckets,
        "summary.json": summary_rows,
        "variant_summary.json": variant_rows,
    }

    for filename, data in files.items():
        path = out_dir / filename
        write_json(path, data)
        print(f"  Wrote {path}")

    per_task_path = out_dir / "per_task_semantic.csv"
    write_csv(per_task_path, per_task_rows, per_task_fieldnames)
    print(f"  Wrote {per_task_path} ({len(per_task_rows)} rows)")
    csv_outputs: Dict[str, Tuple[List[dict], List[str]]] = {
        "per_task_semantic.csv": (per_task_rows, per_task_fieldnames),
    }

    per_task_retrieval_path = out_dir / "per_task_retrieval.csv"
    write_csv(per_task_retrieval_path, per_task_retrieval_rows, per_task_retrieval_fieldnames)
    print(f"  Wrote {per_task_retrieval_path} ({len(per_task_retrieval_rows)} rows)")
    csv_outputs["per_task_retrieval.csv"] = (per_task_retrieval_rows, per_task_retrieval_fieldnames)

    per_task_search_path = out_dir / "per_task_search_bottleneck.csv"
    per_task_search_rows = search_bottleneck.get("per_task_rows", [])
    per_task_search_fieldnames = [
        "condition_model",
        "condition",
        "variant",
        "base_condition",
        "model",
        "task_id",
        "num_search_calls",
    ] + [
        field
        for cutoff in SEARCH_BOTTLENECK_CUTOFFS
        for field in (
            f"found_top_{cutoff}",
            f"first_hit_round_top_{cutoff}",
            f"wasted_rounds_top_{cutoff}",
        )
    ]
    num_per_task_search_rows = write_search_bottleneck_csv(
        per_task_search_rows,
        per_task_search_path,
        per_task_search_fieldnames,
    )
    print(f"  Wrote {per_task_search_path} ({num_per_task_search_rows} rows)")
    csv_outputs["per_task_search_bottleneck.csv"] = (per_task_search_rows, per_task_search_fieldnames)

    per_task_search_tool_path = out_dir / "per_task_search_tool_bottleneck.csv"
    per_task_search_tool_rows = search_bottleneck.get("per_task_tool_rows", [])
    per_task_search_tool_fieldnames = [
        "condition_model",
        "condition",
        "variant",
        "base_condition",
        "model",
        "task_id",
        "search_tool",
        "num_tool_calls",
    ] + [
        field
        for cutoff in SEARCH_BOTTLENECK_CUTOFFS
        for field in (
            f"found_top_{cutoff}",
            f"first_hit_round_top_{cutoff}",
            f"wasted_rounds_top_{cutoff}",
        )
    ]
    num_per_task_search_tool_rows = write_search_bottleneck_csv(
        per_task_search_tool_rows,
        per_task_search_tool_path,
        per_task_search_tool_fieldnames,
    )
    print(f"  Wrote {per_task_search_tool_path} ({num_per_task_search_tool_rows} rows)")
    csv_outputs["per_task_search_tool_bottleneck.csv"] = (
        per_task_search_tool_rows,
        per_task_search_tool_fieldnames,
    )

    search_first_hit_condition_csv = out_dir / "search_first_hit_condition.csv"
    search_first_hit_condition_fieldnames = [
        "condition",
        "variant",
        "base_condition",
        "cutoff",
        "n_tasks_with_search",
        "n_tasks_without_search",
        "found_tasks",
        "not_found_tasks",
        "not_found_rate",
        "ever_found_rate",
        "avg_first_hit_round_found_only",
        "median_first_hit_round_found_only",
        "avg_wasted_rounds",
        "max_round",
        "models",
        "num_condition_model_rows",
        "round_counts",
        "cumulative_found_counts",
        "cumulative_found_rates",
    ]
    write_search_bottleneck_csv(
        search_bottleneck.get("condition_summary_rows", []),
        search_first_hit_condition_csv,
        search_first_hit_condition_fieldnames,
    )
    print(f"  Wrote {search_first_hit_condition_csv}")
    csv_outputs["search_first_hit_condition.csv"] = (
        search_bottleneck.get("condition_summary_rows", []),
        search_first_hit_condition_fieldnames,
    )

    search_first_hit_tool_csv = out_dir / "search_first_hit_tool.csv"
    search_first_hit_tool_fieldnames = [
        "search_tool",
        "cutoff",
        "tasks_with_tool",
        "found_tasks",
        "not_found_tasks",
        "not_found_rate",
        "ever_found_rate",
        "avg_first_hit_round_found_only",
        "median_first_hit_round_found_only",
        "avg_wasted_rounds",
        "max_round",
        "conditions",
        "models",
        "num_condition_model_rows",
        "round_counts",
        "cumulative_found_counts",
        "cumulative_found_rates",
    ]
    write_search_bottleneck_csv(
        search_bottleneck.get("tool_first_hit_rows", []),
        search_first_hit_tool_csv,
        search_first_hit_tool_fieldnames,
    )
    print(f"  Wrote {search_first_hit_tool_csv}")
    csv_outputs["search_first_hit_tool.csv"] = (
        search_bottleneck.get("tool_first_hit_rows", []),
        search_first_hit_tool_fieldnames,
    )

    search_topk_miss_tool_csv = out_dir / "search_topk_miss_tool.csv"
    search_topk_miss_tool_fieldnames = [
        "search_tool",
        "cutoff",
        "n_calls",
        "hit_calls",
        "miss_calls",
        "miss_rate",
        "conditions",
        "models",
        "num_condition_model_rows",
    ]
    write_search_bottleneck_csv(
        search_bottleneck.get("tool_miss_rows", []),
        search_topk_miss_tool_csv,
        search_topk_miss_tool_fieldnames,
    )
    print(f"  Wrote {search_topk_miss_tool_csv}")
    csv_outputs["search_topk_miss_tool.csv"] = (
        search_bottleneck.get("tool_miss_rows", []),
        search_topk_miss_tool_fieldnames,
    )

    search_tool_efficiency_csv = out_dir / "search_tool_efficiency.csv"
    search_tool_efficiency_fieldnames = [
        "search_tool",
        "cutoff",
        "tasks_with_tool",
        "ever_found_rate",
        "avg_first_hit_round_found_only",
        "avg_wasted_rounds",
        "conditions",
        "models",
        "num_condition_model_rows",
    ]
    write_search_bottleneck_csv(
        search_bottleneck.get("tool_efficiency_rows", []),
        search_tool_efficiency_csv,
        search_tool_efficiency_fieldnames,
    )
    print(f"  Wrote {search_tool_efficiency_csv}")
    csv_outputs["search_tool_efficiency.csv"] = (
        search_bottleneck.get("tool_efficiency_rows", []),
        search_tool_efficiency_fieldnames,
    )

    search_call_cumulative_csv = out_dir / "search_call_cumulative_retrieval.csv"
    search_call_cumulative_rows = search_bottleneck.get("per_call_rows", [])
    search_call_cumulative_fieldnames = _search_call_cumulative_fieldnames()
    write_search_bottleneck_csv(
        search_call_cumulative_rows,
        search_call_cumulative_csv,
        search_call_cumulative_fieldnames,
    )
    print(f"  Wrote {search_call_cumulative_csv} ({len(search_call_cumulative_rows)} rows)")
    csv_outputs["search_call_cumulative_retrieval.csv"] = (
        search_call_cumulative_rows,
        search_call_cumulative_fieldnames,
    )

    print("Skipping turn-waste CSV outputs (legacy analysis removed).")

    print("Writing per-model outputs...")
    write_per_model_outputs(out_dir, files, csv_outputs)
    print(f"  Wrote per-model outputs under {out_dir / 'by_model'}")

    if no_figures:
        print("Skipping figures (--no-figures).")
    else:
        print("Generating figures...")
        generate_figures(
            summary_rows,
            variant_rows,
            failure,
            tools_discovery,
            tool_errors,
            base_by_key_records,
            crosstab_rows,
            turn_waste_global_groups,
            turn_waste_condition_model_groups,
            search_bottleneck,
            search_depth_curve,
            search_depth_curve_by_cm,
            reasoning_density_curve,
            reasoning_density_curve_by_cm,
            out_dir,
        )
        print(f"  Wrote figures to {out_dir / 'figures'}")

    print(f"\nSummary ({len(summary_rows)} model x variant rows):")
    for row in summary_rows:
        semantic_pct = f"{float(row['semantic_match']) * 100:.1f}%" if row.get("semantic_match") is not None else "N/A"
        d_ret = f"{row.get('D_ret'):.2f}" if row.get("D_ret") is not None else "N/A"
        d_acc = f"{row.get('D_acc'):.2f}" if row.get("D_acc") is not None else "N/A"
        print(
            f"  {row['condition_model']:<70} "
            f"semantic={semantic_pct:<7} D_ret={d_ret:<5} D_acc={d_acc:<5} n={row.get('n')}"
        )

    return {
        "summary": summary_rows,
        "variant_summary": variant_rows,
        "semantic_match": semantic_match,
        "discovery": discovery,
        "runtime": runtime,
        "efficiency": efficiency,
        "failure": failure,
        "tools_discovery": tools_discovery,
        "tool_errors": tool_errors,
        "semantic_buckets": semantic_buckets,
        "log_error_buckets": log_error_buckets,
        "search_bottleneck": search_bottleneck,
        "search_depth": search_depth_curve,
        "reasoning_density": reasoning_density_curve,
        "search_depth_buckets": search_depth_buckets,
        "reasoning_density_buckets": reasoning_density_buckets,
        "semantic_error_crosstab": crosstab_rows,
        "per_task_rows": per_task_rows,
        "per_task_retrieval_rows": per_task_retrieval_rows,
    }


def main() -> None:
    args = parse_args()
    base_results_dir = args.base_results_dir or _default_base_results_dir(args.results_dir)
    if args.turn_waste_grouped_dir and not args.no_turn_waste:
        print("Ignoring --turn-waste-grouped-dir because turn-waste analysis was removed.")
    turn_waste_grouped_dir: Optional[str] = None
    run_analysis(
        results_dir=args.results_dir,
        base_results_dir=base_results_dir,
        turn_waste_grouped_dir=turn_waste_grouped_dir,
        traces_dir=args.traces_dir,
        tasks_dir=args.tasks_dir,
        output_dir=args.output_dir,
        model_filter=args.model_filter,
        no_figures=args.no_figures,
    )


if __name__ == "__main__":
    main()
