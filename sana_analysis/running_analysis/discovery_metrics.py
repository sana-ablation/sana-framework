#!/usr/bin/env python3
"""
Compute discovery metrics D_ret, D_acc, precision/recall/F1 from trace JSONL files.

D_ret: retrieval-evidence gold coverage per task = |retrieved_gold ∩ gold| / |gold|
D_acc: access gold recall per task = |read_gold ∩ gold| / |gold|

Usage:
    python -m sana_analysis.running_analysis.discovery_metrics [--traces-dir results/traces] [--tasks-dir tasks_mini]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

_READ_TOOLS = {
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
_RETRIEVAL_ACCESS_TOOLS = {
    "peek_file",
    "peek_multiple",
    "peek_files",
}
_AUXILIARY_TRACE_EVENTS = {
    "ideal_subagent_cost",
    "repair_agent_invoked",
    "repair_agent_completed",
    "repair_agent_failed",
}


def _is_auxiliary_trace(record: dict) -> bool:
    return record.get("event") in _AUXILIARY_TRACE_EVENTS


def make_task_stem_key(task_id: str) -> str:
    """Return the normalized "{task_dir}/{task_name}" key used across analyses."""
    p = Path(str(task_id))
    return f"{p.parent.name}/{p.stem}"


def make_condition_model_task_key(condition: str, model: str, task_id: str) -> str:
    """Return a stable composite key for multi-run joins."""
    return f"{condition}/{model}/{make_task_stem_key(task_id)}"


def _task_path_identity(task_id: str) -> tuple[str, str, str] | None:
    """Return (task_root, task_dir, task_name) when task_id carries a task root."""
    parts = Path(str(task_id)).parts
    if len(parts) < 3:
        return None
    task_root = parts[-3]
    if not (task_root.startswith("tasks") or task_root == "other-benchmarks"):
        return None
    return (task_root, parts[-2], Path(parts[-1]).stem)


def resolve_task_value(task_id: str, task_values: dict, default=None):
    """Resolve a task-keyed value without crossing explicit task roots.

    If task_id is explicit, e.g. tasks_mini/k-1-d-1/task_1.json, only an entry
    with that same task root may match. Stem-only ids keep the legacy fallback.
    """
    explicit_identity = _task_path_identity(task_id)
    if explicit_identity is not None:
        for path_key, value in task_values.items():
            if _task_path_identity(path_key) == explicit_identity:
                return value
        return default

    if task_id in task_values:
        return task_values[task_id]

    task_stem = make_task_stem_key(task_id) if task_id.count("/") >= 3 else task_id
    for path_key, value in task_values.items():
        if make_task_stem_key(path_key) == task_stem:
            return value
    return default


def resolve_trace_task_id(task_id: str, task_traces: list[dict]) -> str:
    """Prefer an explicit task_id recorded inside trace rows when available."""
    for record in task_traces:
        record_task_id = str(record.get("task_id", "") or "")
        if _task_path_identity(record_task_id) is not None:
            return record_task_id
    return task_id


def load_traces(traces_dir: str) -> dict[str, list]:
    """Load all trace JSONL files. Returns {task_id: [trace_records]}.

    task_id key is "{parent_dir}/{stem}", e.g. "k-2-d-1/task_1".
    """
    traces: dict = defaultdict(list)
    for jsonl_path in Path(traces_dir).rglob("*.jsonl"):
        task_id = f"{jsonl_path.parent.name}/{jsonl_path.stem}"
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    traces[task_id].append(json.loads(line))
    return dict(traces)


def load_traces_with_context(traces_dir: str) -> dict[str, list]:
    """Load trace JSONLs keyed by "{condition}/{model}/{task_dir}/{task_name}"."""
    traces: dict = defaultdict(list)
    for jsonl_path in Path(traces_dir).rglob("*.jsonl"):
        parts = jsonl_path.parts
        if len(parts) < 4:
            continue
        condition = parts[-4]
        model = parts[-3]
        task_key = f"{jsonl_path.parent.name}/{jsonl_path.stem}"
        composite_key = f"{condition}/{model}/{task_key}"
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    traces[composite_key].append(json.loads(line))
    return dict(traces)


def compute_discovery_metrics(
    traces: dict[str, list],
    task_gold: dict[str, list],
) -> dict:
    """Compute per-task and aggregate discovery metrics."""
    task_metrics: list = []

    for task_id, task_traces in traces.items():
        # Resolve task_id stem back to a full path key for gold lookup
        gold_values = list(_resolve_gold(resolve_trace_task_id(task_id, task_traces), task_gold))
        gold_ids = set(gold_values)
        if not gold_ids:
            continue
        use_source_gold = _is_source_gold_values(gold_values)
        use_gold_units = not use_source_gold and len(gold_values) != len(gold_ids)
        n_gold = len(gold_values) if use_gold_units else len(gold_ids)
        dataset_to_unique_source = _dataset_to_unique_source(gold_ids) if use_source_gold else {}

        # Separate search traces from read traces and submit_answer record.
        # D_ret uses retrieval evidence: search/list results plus lightweight
        # file peeks, which can reveal the correct source before full analysis.
        search_traces = [
            t
            for t in task_traces
            if not _is_auxiliary_trace(t)
            and t.get("tool") not in _READ_TOOLS
            and t.get("tool") != "list_files"
            and t.get("tool") != "submit_answer"
        ]
        retrieval_traces = [
            t
            for t in task_traces
            if not _is_auxiliary_trace(t)
            and t.get("tool") != "submit_answer"
            and (
                t.get("tool") == "list_files"
                or t.get("tool") not in _READ_TOOLS
                or t.get("tool") in _RETRIEVAL_ACCESS_TOOLS
            )
        ]
        submit_record = next((t for t in task_traces if t.get("tool") == "submit_answer"), None)

        # D_ret coverage: how much of the gold set was discovered by retrieval evidence?
        # Also keep hit/precision-style companions:
        # - d_ret_hit: was any gold retrieved at all? (legacy binary)
        # - d_ret_precision: how much of retrieved set is gold? (mentor's "7/70 vs 7/10")
        all_result_ids: set = set()
        all_result_ids_total_count = 0
        retrieved_gold_unit_ids: set[str] = set()
        for trace in retrieval_traces:
            result_ids = _retrieval_dataset_ids(trace)
            all_result_ids.update(result_ids)
            all_result_ids_total_count += len(result_ids)
            retrieved_gold_unit_ids.update(_retrieval_gold_ids(trace, gold_ids, dataset_to_unique_source))

        retrieved_gold = all_result_ids & gold_ids
        use_source_or_unit_gold = use_source_gold or use_gold_units
        retrieved_gold_count = min(len(retrieved_gold_unit_ids), n_gold) if use_source_or_unit_gold else len(retrieved_gold)
        retrieved_gold_for_output = sorted(retrieved_gold_unit_ids) if use_source_or_unit_gold else sorted(retrieved_gold)
        d_ret = retrieved_gold_count / n_gold if n_gold else 0.0
        d_ret_hit = int(retrieved_gold_count > 0)
        d_ret_precision_denominator = all_result_ids_total_count if use_gold_units else len(all_result_ids)
        d_ret_precision = (
            retrieved_gold_count / d_ret_precision_denominator
            if d_ret_precision_denominator
            else 0.0
        )
        d_ret_f1 = (
            2 * d_ret_precision * d_ret / (d_ret_precision + d_ret)
            if (d_ret_precision + d_ret) > 0
            else 0.0
        )

        # Precision / Recall: per-call average (standard IR)
        # Compute P/R for each individual search call, then average across calls.
        call_precisions = []
        call_recalls = []
        for trace in retrieval_traces:
            result_ids = _retrieval_dataset_ids(trace)
            if not result_ids:
                continue
            hits = (
                len(_retrieval_gold_ids(trace, gold_ids, dataset_to_unique_source))
                if use_source_or_unit_gold
                else len(set(result_ids) & gold_ids)
            )
            call_precisions.append(hits / len(result_ids))
            call_recalls.append(hits / n_gold)
        precision = sum(call_precisions) / len(call_precisions) if call_precisions else 0.0
        recall = sum(call_recalls) / len(call_recalls) if call_recalls else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        # D_acc read metrics:
        # - d_acc_recall (legacy D_acc): how much of the gold set was accessed
        # - d_acc_precision: how much of accessed set is gold
        # - d_acc_f1: harmonic mean of read precision/recall
        read_records = [
            t for t in task_traces if not _is_auxiliary_trace(t) and t.get("tool") in _READ_TOOLS
        ]
        all_read_ids: set = set()
        all_read_ids_total_count = 0
        read_gold_unit_ids: set[str] = set()
        for rec in read_records:
            read_ids = rec.get("read_dataset_ids", [])
            all_read_ids.update(read_ids)
            all_read_ids_total_count += len(read_ids)
            read_gold_unit_ids.update(
                _trace_gold_ids(
                    rec,
                    gold_ids,
                    fallback_field="read_dataset_ids",
                    explicit_gold_field="gold_dataset_ids_read",
                    source_field="read_source_ids",
                    dataset_to_unique_source=dataset_to_unique_source,
                )
            )
        read_gold = all_read_ids & gold_ids
        read_gold_count = min(len(read_gold_unit_ids), n_gold) if use_source_or_unit_gold else len(read_gold)
        d_acc = read_gold_count / n_gold if n_gold else 0.0
        d_acc_precision_denominator = all_read_ids_total_count if use_gold_units else len(all_read_ids)
        d_acc_precision = (
            read_gold_count / d_acc_precision_denominator
            if d_acc_precision_denominator
            else 0.0
        )
        d_acc_recall = d_acc
        d_acc_f1 = (
            2 * d_acc_precision * d_acc_recall / (d_acc_precision + d_acc_recall)
            if (d_acc_precision + d_acc_recall) > 0
            else 0.0
        )

        task_metrics.append({
            "task_id": task_id,
            "gold_ids": sorted(gold_values) if use_gold_units else sorted(gold_ids),
            "retrieved_dataset_ids": sorted(all_result_ids),
            "retrieved_gold_dataset_ids": retrieved_gold_for_output,
            "d_ret": d_ret,
            "d_ret_hit": d_ret_hit,
            "d_ret_precision": d_ret_precision,
            "d_ret_f1": d_ret_f1,
            "d_acc": d_acc,
            "d_acc_precision": d_acc_precision,
            "d_acc_recall": d_acc_recall,
            "d_acc_f1": d_acc_f1,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "num_search_calls": len(search_traces),
            "num_results_total": len(all_result_ids),
            "num_results_total_non_unique": all_result_ids_total_count,
            "num_read_calls": len(read_records),
        })

    n = len(task_metrics)
    if n == 0:
        return {"task_metrics": [], "aggregate": {}}

    agg = {
        "n": n,
        "D_ret": sum(m["d_ret"] for m in task_metrics) / n,
        "D_ret_hit_rate": sum(m["d_ret_hit"] for m in task_metrics) / n,
        "D_ret_precision": sum(m["d_ret_precision"] for m in task_metrics) / n,
        "D_ret_f1": sum(m["d_ret_f1"] for m in task_metrics) / n,
        "D_acc": sum(m["d_acc"] for m in task_metrics) / n,
        "D_acc_precision": sum(m["d_acc_precision"] for m in task_metrics) / n,
        "D_acc_recall": sum(m["d_acc_recall"] for m in task_metrics) / n,
        "D_acc_f1": sum(m["d_acc_f1"] for m in task_metrics) / n,
        "avg_precision": sum(m["precision"] for m in task_metrics) / n,
        "avg_recall": sum(m["recall"] for m in task_metrics) / n,
        "avg_f1": sum(m["f1"] for m in task_metrics) / n,
        "avg_search_calls": sum(m["num_search_calls"] for m in task_metrics) / n,
        "avg_read_calls": sum(m["num_read_calls"] for m in task_metrics) / n,
    }

    return {"task_metrics": task_metrics, "aggregate": agg}


def _resolve_gold(task_id: str, task_gold: dict[str, list]) -> list:
    """Match a trace task_id ("k-2-d-1/task_1") to a gold entry."""
    return resolve_task_value(task_id, task_gold, [])


def _trace_gold_ids(
    record: dict,
    gold_ids: set[str],
    *,
    fallback_field: str,
    explicit_gold_field: str,
    source_field: str | None = None,
    dataset_to_unique_source: dict[str, str] | None = None,
) -> set[str]:
    if source_field is not None:
        source_ids = {
            source_id
            for value in record.get(source_field, []) or []
            if (source_id := _normalize_source_id(value))
        }
        source_hits = source_ids & gold_ids
        if source_hits:
            return source_hits

    values = record.get(explicit_gold_field)
    if not isinstance(values, list) or not values:
        values = record.get(fallback_field, [])
    if dataset_to_unique_source:
        return {
            dataset_to_unique_source[value]
            for value in values
            if value in dataset_to_unique_source and dataset_to_unique_source[value] in gold_ids
        }
    return {value for value in values if value in gold_ids}


def _retrieval_dataset_ids(record: dict) -> list:
    if record.get("tool") in _RETRIEVAL_ACCESS_TOOLS:
        return record.get("read_dataset_ids", []) or []
    return record.get("result_dataset_ids", []) or []


def _retrieval_gold_ids(
    record: dict,
    gold_ids: set[str],
    dataset_to_unique_source: dict[str, str],
) -> set[str]:
    if record.get("tool") in _RETRIEVAL_ACCESS_TOOLS:
        return _trace_gold_ids(
            record,
            gold_ids,
            fallback_field="read_dataset_ids",
            explicit_gold_field="gold_dataset_ids_read",
            source_field="read_source_ids",
            dataset_to_unique_source=dataset_to_unique_source,
        )
    return _trace_gold_ids(
        record,
        gold_ids,
        fallback_field="result_dataset_ids",
        explicit_gold_field="gold_dataset_ids_in_results",
        source_field="result_source_ids",
        dataset_to_unique_source=dataset_to_unique_source,
    )


def _is_source_gold_values(values: list[str]) -> bool:
    return any("/files/" in str(value) for value in values)


def _dataset_to_unique_source(gold_ids: set[str]) -> dict[str, str]:
    sources_by_dataset: dict[str, set[str]] = defaultdict(set)
    for source_id in gold_ids:
        dataset_id = str(source_id).split("/files/", 1)[0]
        if dataset_id:
            sources_by_dataset[dataset_id].add(source_id)
    return {
        dataset_id: next(iter(source_ids))
        for dataset_id, source_ids in sources_by_dataset.items()
        if len(source_ids) == 1
    }


def compute_per_folder_discovery(
    traces: dict[str, list],
    task_gold: dict[str, list],
) -> dict:
    """Group traces by task-dir prefix and compute discovery metrics per folder.

    Returns {folder_name: aggregate_dict} where aggregate_dict is the same shape
    as compute_discovery_metrics()['aggregate'] (no task_metrics key).
    """
    from collections import defaultdict

    folder_traces: dict = defaultdict(dict)
    for task_id, recs in traces.items():
        folder = task_id.split("/")[0]
        folder_traces[folder][task_id] = recs

    out = {}
    for folder, subset in sorted(folder_traces.items()):
        result = compute_discovery_metrics(subset, task_gold)
        out[folder] = result["aggregate"]
    return out


def compute_tools_discovery(
    traces: dict[str, list],
    task_gold: dict[str, list],
) -> dict:
    """Per-tool micro-averaged precision/recall/F1 with per_folder > per_task nesting.

    For each task × tool:
    For each task × tool, per-call average (standard IR):
      - precision per call = |gold in call results| / |call results|
      - recall per call    = |gold in call results| / |gold for task|
      - task precision/recall = mean over all calls for that tool in that task
      - f1        = harmonic mean

    per_folder averages the per_task values within that folder.
    Top-level averages all per_task values across all folders.

    Returns {tool_name: {avg_precision, avg_recall, avg_f1, calls, tasks_with_hit,
                         n_tasks, per_folder: {folder: {... per_task: {task_id: {...}}}}}}
    """
    from collections import defaultdict

    # tool -> task_id -> list of per-call (precision, recall) + call count
    tool_task: dict = defaultdict(lambda: defaultdict(lambda: {
        "call_precisions": [],
        "call_recalls": [],
        "calls": 0,
    }))

    for task_id, task_traces in traces.items():
        gold_ids = set(_resolve_gold(resolve_trace_task_id(task_id, task_traces), task_gold))
        if not gold_ids:
            continue
        for rec in task_traces:
            if _is_auxiliary_trace(rec):
                continue
            tool = rec.get("tool")
            if not tool or tool == "submit_answer" or tool in _READ_TOOLS:
                continue
            result_ids = rec.get("result_dataset_ids", [])
            if not result_ids:
                continue
            hits = len(set(result_ids) & gold_ids)
            s = tool_task[tool][task_id]
            s["call_precisions"].append(hits / len(result_ids))
            s["call_recalls"].append(hits / len(gold_ids))
            s["calls"] += 1

    out = {}
    for tool, task_map in tool_task.items():
        # Build per_task records grouped by folder
        folder_tasks: dict = defaultdict(dict)
        for task_id, s in task_map.items():
            folder = task_id.split("/")[0]
            cp, cr = s["call_precisions"], s["call_recalls"]
            precision = sum(cp) / len(cp) if cp else 0.0
            recall = sum(cr) / len(cr) if cr else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            folder_tasks[folder][task_id] = {
                "calls": s["calls"],
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "has_hit": int(recall > 0),
            }

        # Build per_folder aggregates
        per_folder = {}
        all_task_records = []
        for folder, tasks in sorted(folder_tasks.items()):
            task_vals = list(tasks.values())
            all_task_records.extend(task_vals)
            n = len(task_vals)
            folder_calls = sum(t["calls"] for t in task_vals)
            folder_hits = sum(t["has_hit"] for t in task_vals)
            per_folder[folder] = {
                "n_tasks": n,
                "calls": folder_calls,
                "avg_precision": round(sum(t["precision"] for t in task_vals) / n, 4),
                "avg_recall": round(sum(t["recall"] for t in task_vals) / n, 4),
                "avg_f1": round(sum(t["f1"] for t in task_vals) / n, 4),
                "tasks_with_hit": folder_hits,
                "per_task": tasks,
            }

        # Top-level aggregate over all tasks
        n_all = len(all_task_records)
        total_calls = sum(t["calls"] for t in all_task_records)
        tasks_with_hit = sum(t["has_hit"] for t in all_task_records)
        out[tool] = {
            "calls": total_calls,
            "n_tasks": n_all,
            "tasks_with_hit": tasks_with_hit,
            "avg_precision": round(sum(t["precision"] for t in all_task_records) / n_all, 4) if n_all else 0.0,
            "avg_recall": round(sum(t["recall"] for t in all_task_records) / n_all, 4) if n_all else 0.0,
            "avg_f1": round(sum(t["f1"] for t in all_task_records) / n_all, 4) if n_all else 0.0,
            "per_folder": per_folder,
        }

    return out


def _normalize_gold_dataset_id(value: str) -> str:
    """Normalize a dataset-relative source path to the dataset ID used in traces."""
    text = str(value).strip()
    if not text:
        return ""

    datagov_marker = "/datagov/"
    if datagov_marker in text:
        text = "datagov/" + text.split(datagov_marker, 1)[1]

    if text.startswith("s3://"):
        without_scheme = text[len("s3://"):]
        text = without_scheme.split("/", 1)[1] if "/" in without_scheme else without_scheme

    parts = [part for part in text.split("/") if part]
    if len(parts) >= 2 and parts[0] == "datagov":
        return parts[1]
    if len(parts) >= 3 and parts[1] == "files":
        return parts[0]
    return text


def _normalize_source_id(value: str, dataset_id: str | None = None, file_path: str | None = None) -> str:
    if dataset_id and file_path:
        path = str(file_path).strip().lstrip("/")
        if path.startswith("files/"):
            return f"{dataset_id}/{path}"
        if "/files/" in path:
            return _normalize_source_id(path)
        return f"{dataset_id}/files/{path}"

    text = str(value).strip()
    if not text:
        return ""

    if text.startswith("s3://"):
        without_scheme = text[len("s3://"):]
        text = without_scheme.split("/", 1)[1] if "/" in without_scheme else without_scheme

    datagov_marker = "/datagov/"
    if datagov_marker in text:
        text = "datagov/" + text.split(datagov_marker, 1)[1]

    parts = [part for part in text.split("/") if part]
    if len(parts) >= 4 and parts[0] == "datagov" and parts[2] == "files":
        return f"{parts[1]}/files/{'/'.join(parts[3:])}"
    if len(parts) >= 3 and parts[1] == "files":
        return f"{parts[0]}/files/{'/'.join(parts[2:])}"
    return text


def _unique_normalized_dataset_ids(values: list) -> list[str]:
    """Normalize IDs/sources while preserving first appearance order."""
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = _normalize_gold_dataset_id(value)
        if normalized and normalized not in seen:
            out.append(normalized)
            seen.add(normalized)
    return out


def _unique_normalized_source_ids(values: list) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = _normalize_source_id(value)
        if normalized and normalized not in seen:
            out.append(normalized)
            seen.add(normalized)
    return out


def _node_sources(task: dict) -> list[str]:
    nodes = task.get("nodes") or {}
    if isinstance(nodes, dict):
        iterable = nodes.values()
    elif isinstance(nodes, list):
        iterable = nodes
    else:
        return []
    return [
        node["source"]
        for node in iterable
        if isinstance(node, dict) and isinstance(node.get("source"), str)
    ]


def _gold_ids_from_task_path(path: Path) -> list[str]:
    with path.open() as f:
        task = json.load(f)
    source_values = task.get("sources_used") or task.get("source_sequence") or _node_sources(task)
    return _unique_normalized_dataset_ids(source_values or task.get("datasets_used", []))


def _gold_unit_ids_from_task_path(path: Path) -> list[str]:
    with path.open() as f:
        task = json.load(f)
    source_values = task.get("sources_used") or task.get("source_sequence") or _node_sources(task)
    values = source_values or task.get("datasets_used", [])
    normalized = [_normalize_gold_dataset_id(value) for value in values]
    normalized = [value for value in normalized if value]
    if any(value.startswith("kramabench-") for value in normalized):
        return normalized
    return _unique_normalized_dataset_ids(values)


def _plan_path_for_task_path(task_path: Path, tasks_root: Path) -> Path | None:
    try:
        rel_path = task_path.relative_to(tasks_root)
    except ValueError:
        return None
    candidates = [
        tasks_root.parent / "plans-mini-kramabench" / rel_path,
        tasks_root.parent / "plans_mini_kramabench" / rel_path,
        tasks_root.parent / "plans_mini" / rel_path,
        tasks_root.parent / "plans-mini" / rel_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _gold_source_ids_from_plan_or_task_path(path: Path, tasks_root: Path | None = None) -> list[str]:
    plan_path = _plan_path_for_task_path(path, tasks_root) if tasks_root is not None else None
    source_path = plan_path or path
    with source_path.open() as f:
        task = json.load(f)
    source_values = task.get("source_sequence") or task.get("sources_used") or _node_sources(task)
    return _unique_normalized_source_ids(source_values or task.get("datasets_used", []))


def load_task_gold_ids(tasks_dir: str) -> dict[str, list]:
    """Load normalized gold dataset IDs from task source paths.

    Trace records store dataset IDs, not exact source paths. Prefer explicit
    source lists when present, then node sources, and only fall back to
    datasets_used for older tasks without source metadata.
    """
    import glob as glob_mod
    gold: dict = {}
    for path in glob_mod.glob(str(Path(tasks_dir) / "**" / "*.json"), recursive=True):
        gold[path] = _gold_ids_from_task_path(Path(path))
    return gold


def load_task_gold_unit_ids(tasks_dir: str) -> dict[str, list]:
    """Load gold IDs, preserving Kramabench source-file units as repeated IDs."""
    import glob as glob_mod
    gold: dict = {}
    for path in glob_mod.glob(str(Path(tasks_dir) / "**" / "*.json"), recursive=True):
        gold[path] = _gold_unit_ids_from_task_path(Path(path))
    return gold


def load_task_gold_source_ids(tasks_dir: str) -> dict[str, list]:
    """Load unique normalized source-file IDs, preferring ideal-plan source_sequence."""
    import glob as glob_mod
    root = Path(tasks_dir)
    gold: dict = {}
    for path in glob_mod.glob(str(root / "**" / "*.json"), recursive=True):
        gold[path] = _gold_source_ids_from_plan_or_task_path(Path(path), root)
    return gold


def _iter_trace_records(traces: dict) -> list[dict]:
    records: list[dict] = []
    for value in traces.values():
        if isinstance(value, list):
            records.extend(record for record in value if isinstance(record, dict))
        elif isinstance(value, dict):
            records.extend(_iter_trace_records(value))
    return records


def load_task_gold_ids_for_traces(tasks_dir: str, traces: dict) -> dict[str, list]:
    """Load gold IDs from tasks_dir plus explicit task paths recorded in traces.

    This preserves explicit task-root matching while allowing analysis reruns to
    recover when --tasks-dir is stale but trace rows carry usable task_id paths.
    """
    gold = load_task_gold_ids(tasks_dir) if Path(tasks_dir).exists() else {}
    for record in _iter_trace_records(traces):
        task_id = str(record.get("task_id", "") or "")
        if _task_path_identity(task_id) is None:
            continue
        path = Path(task_id)
        if not path.exists() or str(path) in gold:
            continue
        gold[str(path)] = _gold_ids_from_task_path(path)
    return gold


def load_task_gold_unit_ids_for_traces(tasks_dir: str, traces: dict) -> dict[str, list]:
    """Load gold IDs for traces, preserving Kramabench source-file units."""
    gold = load_task_gold_unit_ids(tasks_dir) if Path(tasks_dir).exists() else {}
    for record in _iter_trace_records(traces):
        task_id = str(record.get("task_id", "") or "")
        if _task_path_identity(task_id) is None:
            continue
        path = Path(task_id)
        if not path.exists() or str(path) in gold:
            continue
        gold[str(path)] = _gold_unit_ids_from_task_path(path)
    return gold


def load_task_gold_source_ids_for_traces(tasks_dir: str, traces: dict) -> dict[str, list]:
    """Load unique source-file IDs for traces, preferring ideal-plan source_sequence."""
    root = Path(tasks_dir)
    gold = load_task_gold_source_ids(tasks_dir) if root.exists() else {}
    for record in _iter_trace_records(traces):
        task_id = str(record.get("task_id", "") or "")
        if _task_path_identity(task_id) is None:
            continue
        path = Path(task_id)
        if not path.exists() or str(path) in gold:
            continue
        gold[str(path)] = _gold_source_ids_from_plan_or_task_path(path, root if root.exists() else None)
    return gold


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces-dir", default="results/traces")
    parser.add_argument("--tasks-dir", default="benchmarks/lakeqa/tasks-mini/tasks")
    args = parser.parse_args()

    traces = load_traces(args.traces_dir)
    task_gold = (
        load_task_gold_source_ids(args.tasks_dir)
        if "kramabench" in Path(args.tasks_dir).name
        else load_task_gold_ids(args.tasks_dir)
    )

    print(f"Loaded traces for {len(traces)} tasks from {args.traces_dir}")

    metrics = compute_discovery_metrics(traces, task_gold)
    agg = metrics["aggregate"]

    if not agg:
        print("No metrics computed (check task gold IDs and trace files).")
        return

    print(f"\nAggregate Discovery Metrics (n={agg['n']}):")
    print(f"  D_ret (retrieval coverage): {agg['D_ret']:.3f}")
    print(f"  D_ret F1:                   {agg['D_ret_f1']:.3f}")
    print(f"  D_acc (read recall):         {agg['D_acc']:.3f}")
    print(f"  D_acc precision:             {agg['D_acc_precision']:.3f}")
    print(f"  D_acc recall:                {agg['D_acc_recall']:.3f}")
    print(f"  D_acc F1:                    {agg['D_acc_f1']:.3f}")
    print(f"  Search avg precision:        {agg['avg_precision']:.3f}")
    print(f"  Search avg recall:           {agg['avg_recall']:.3f}")
    print(f"  Search avg F1:               {agg['avg_f1']:.3f}")
    print(f"  Avg Search Calls/Task:       {agg['avg_search_calls']:.1f}")
    print(f"  Avg Read Calls/Task:         {agg['avg_read_calls']:.1f}")


if __name__ == "__main__":
    main()
