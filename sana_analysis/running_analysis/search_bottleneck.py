#!/usr/bin/env python3
"""
Search bottleneck analysis helpers.

This module computes:
- task-level first-hit rounds over all search calls
- task-level first-hit rounds per search tool (tool-local rounds)
- per-call gold-hit/recall rows over search-call index
- per-tool top-k miss rates
- compact condition-model summaries that can be merged into higher-level tables
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from sana_analysis.running_analysis.discovery_metrics import (
    _dataset_to_unique_source,
    _is_source_gold_values,
    _normalize_source_id,
    make_task_stem_key,
    resolve_task_value,
    resolve_trace_task_id,
)


SEARCH_BOTTLENECK_CUTOFFS = (1, 3, 5)
SEARCH_TOOL_ORDER = [
    "search_ideal",
    "search_reranked",
    "search_value",
    "search_prefix",
    "search_schema",
]
SEARCH_TOOL_COLORS = {
    "search_ideal": "#2E8B57",
    "search_reranked": "#1F77B4",
    "search_value": "#C07A2C",
    "search_prefix": "#C44E52",
    "search_schema": "#7F7F7F",
}
READ_TOOLS = {
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


def _split_cm_key(key: str) -> Tuple[str, str, str]:
    parts = key.split("/", 2)
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    if len(parts) == 2:
        return parts[0], parts[0], parts[1]
    if len(parts) == 1:
        return parts[0], parts[0], ""
    return "unknown", "unknown", key


def _condition_label(variant: str, base_condition: str) -> str:
    return f"{variant}/{base_condition}"


def _parse_turn(value) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _order_events(events: List[dict]) -> List[dict]:
    if events and all(event.get("parsed_turn") is not None for event in events):
        return sorted(events, key=lambda event: (int(event["parsed_turn"]), int(event["line_index"])))
    return list(events)


def _tool_sort_key(tool_name: str) -> tuple:
    if tool_name in SEARCH_TOOL_ORDER:
        return (SEARCH_TOOL_ORDER.index(tool_name), tool_name)
    return (len(SEARCH_TOOL_ORDER), tool_name)


def _condition_sort_key(condition_label: str) -> tuple:
    if "/" in condition_label:
        variant, base_condition = condition_label.split("/", 1)
        return (variant, base_condition)
    return (condition_label, "")


def _condition_display_label(row: dict, label_formatter: Optional[Callable[[dict], str]] = None) -> str:
    if label_formatter is not None:
        label = label_formatter(row)
        if label:
            return str(label)
    return str(row.get("condition", ""))


def _task_id_for_row(task_id: str) -> str:
    if task_id.count("/") >= 3:
        return make_task_stem_key(task_id)
    p = Path(str(task_id))
    return f"{p.parent.name}/{p.stem}"


def _resolve_gold_values(task_id: str, task_gold: Dict[str, List[str]]) -> List[str]:
    return list(resolve_task_value(task_id, task_gold, []))


def _gold_ids_in_values(values: List[str], gold_ids: set[str]) -> set[str]:
    return {value for value in values if value in gold_ids}


def _event_gold_ids(
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
    return _gold_ids_in_values(list(values or []), gold_ids)


def _is_search_event(record: dict) -> bool:
    tool = str(record.get("tool", "") or "")
    return tool.startswith("search") and record.get("result_dataset_ids") is not None


def _is_read_event(record: dict) -> bool:
    tool = str(record.get("tool", "") or "")
    return tool in READ_TOOLS and record.get("read_dataset_ids") is not None


def _summarize_first_hit_rows(
    rows: List[dict],
    *,
    cutoff: int,
    rounds_field: str,
) -> dict:
    n_tasks = len(rows)
    found_rounds: List[int] = []
    wasted_rounds: List[int] = []
    round_counts: Dict[int, int] = defaultdict(int)
    max_round = 0

    for row in rows:
        round_total = int(row.get(rounds_field, 0) or 0)
        max_round = max(max_round, round_total)
        wasted_value = row.get(f"wasted_rounds_top_{cutoff}")
        if wasted_value not in (None, ""):
            wasted_rounds.append(int(wasted_value))
        first_hit_round = row.get(f"first_hit_round_top_{cutoff}")
        if first_hit_round in (None, ""):
            continue
        found_round = int(first_hit_round)
        found_rounds.append(found_round)
        round_counts[found_round] += 1

    found_tasks = len(found_rounds)
    not_found_tasks = n_tasks - found_tasks
    cumulative_found_counts = {}
    cumulative_found_rates = {}
    running_found = 0
    for round_number in range(1, max_round + 1):
        running_found += round_counts.get(round_number, 0)
        cumulative_found_counts[str(round_number)] = running_found
        cumulative_found_rates[str(round_number)] = round(running_found / n_tasks, 4) if n_tasks else 0.0

    return {
        "found_tasks": found_tasks,
        "not_found_tasks": not_found_tasks,
        "not_found_rate": round(not_found_tasks / n_tasks, 4) if n_tasks else 0.0,
        "ever_found_rate": round(found_tasks / n_tasks, 4) if n_tasks else 0.0,
        "avg_first_hit_round_found_only": round(sum(found_rounds) / len(found_rounds), 4) if found_rounds else None,
        "median_first_hit_round_found_only": median(found_rounds) if found_rounds else None,
        "avg_wasted_rounds": round(sum(wasted_rounds) / len(wasted_rounds), 4) if wasted_rounds else None,
        "max_round": max_round,
        "round_counts": {str(round_number): count for round_number, count in sorted(round_counts.items())},
        "cumulative_found_counts": cumulative_found_counts,
        "cumulative_found_rates": cumulative_found_rates,
    }


def compute_search_bottleneck(
    grouped_traces: Dict[str, dict],
    meta_by_key: Dict[str, dict],
    task_gold: Dict[str, List[str]],
) -> dict:
    per_task_rows: List[dict] = []
    per_task_tool_rows: List[dict] = []
    per_call_rows: List[dict] = []

    cm_task_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {"tasks_seen": 0, "tasks_without_search": 0})
    condition_task_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {"tasks_seen": 0, "tasks_without_search": 0})
    tool_miss_acc: Dict[str, dict] = defaultdict(
        lambda: {
            "calls": {cutoff: 0 for cutoff in SEARCH_BOTTLENECK_CUTOFFS},
            "hits": {cutoff: 0 for cutoff in SEARCH_BOTTLENECK_CUTOFFS},
            "conditions": set(),
            "models": set(),
            "condition_models": set(),
        }
    )

    for key, traces in sorted(grouped_traces.items()):
        meta = meta_by_key.get(key)
        if meta is None:
            variant, base_condition, model = _split_cm_key(key)
            meta = {
                "variant": variant,
                "base_condition": base_condition,
                "condition_label": _condition_label(variant, base_condition),
                "model": model,
            }

        variant = str(meta.get("variant", "unknown"))
        base_condition = str(meta.get("base_condition", variant))
        condition_label = str(meta.get("condition_label", _condition_label(variant, base_condition)))
        model = str(meta.get("model", "unknown"))

        for task_id, task_traces in sorted(traces.items()):
            gold_values = _resolve_gold_values(resolve_trace_task_id(task_id, task_traces), task_gold)
            gold_ids = set(gold_values)
            if not gold_ids:
                continue
            use_source_gold = _is_source_gold_values(gold_values)
            use_gold_units = not use_source_gold and len(gold_values) != len(gold_ids)
            n_gold_ids = len(gold_values) if use_gold_units else len(gold_ids)
            dataset_to_unique_source = _dataset_to_unique_source(gold_ids) if use_source_gold else {}

            cm_task_counts[key]["tasks_seen"] += 1
            condition_task_counts[condition_label]["tasks_seen"] += 1

            search_events: List[dict] = []
            trajectory_events: List[dict] = []
            for line_index, record in enumerate(task_traces):
                parsed_turn = _parse_turn(record.get("turn"))
                if not _is_search_event(record):
                    if _is_read_event(record):
                        trajectory_events.append(
                            {
                                "event_type": "read",
                                "tool": str(record.get("tool", "") or ""),
                                "read_ids": list(record.get("read_dataset_ids") or []),
                                "read_gold_ids": (
                                    _event_gold_ids(
                                        record,
                                        gold_ids,
                                        fallback_field="read_dataset_ids",
                                        explicit_gold_field="gold_dataset_ids_read",
                                        source_field="read_source_ids",
                                        dataset_to_unique_source=dataset_to_unique_source,
                                    )
                                    if use_source_gold or use_gold_units
                                    else len(set(record.get("read_dataset_ids") or []) & gold_ids)
                                ),
                                "parsed_turn": parsed_turn,
                                "line_index": line_index,
                                "task_id": str(record.get("task_id", "") or task_id),
                            }
                        )
                    continue
                tool_name = str(record.get("tool", "") or "")
                result_ids = list(record.get("result_dataset_ids") or [])
                result_gold_ids = sorted(set(result_ids) & gold_ids)
                cutoff_hit_counts = {
                    cutoff: (
                        len(
                            _event_gold_ids(
                                {**record, "result_dataset_ids": result_ids[:cutoff]},
                                gold_ids,
                                fallback_field="result_dataset_ids",
                                explicit_gold_field="gold_dataset_ids_in_results",
                                source_field="result_source_ids",
                                dataset_to_unique_source=dataset_to_unique_source,
                            )
                        )
                        if use_source_gold or use_gold_units
                        else len(set(result_ids[:cutoff]) & gold_ids)
                    )
                    for cutoff in SEARCH_BOTTLENECK_CUTOFFS
                }
                cutoff_hits = {
                    cutoff: bool(cutoff_hit_counts[cutoff])
                    for cutoff in SEARCH_BOTTLENECK_CUTOFFS
                }

                event = {
                    "tool": tool_name,
                    "result_ids": result_ids,
                    "result_gold_ids": result_gold_ids,
                    "result_gold_unit_ids": (
                        _event_gold_ids(
                            record,
                            gold_ids,
                            fallback_field="result_dataset_ids",
                            explicit_gold_field="gold_dataset_ids_in_results",
                            source_field="result_source_ids",
                            dataset_to_unique_source=dataset_to_unique_source,
                        )
                        if use_source_gold or use_gold_units
                        else len(result_gold_ids)
                    ),
                    "cutoff_hit_counts": cutoff_hit_counts,
                    "cutoff_hits": cutoff_hits,
                    "parsed_turn": parsed_turn,
                    "line_index": line_index,
                    "task_id": str(record.get("task_id", "") or task_id),
                }
                search_events.append(event)
                trajectory_events.append({**event, "event_type": "search"})

                tool_bucket = tool_miss_acc[tool_name]
                tool_bucket["conditions"].add(condition_label)
                tool_bucket["models"].add(model)
                tool_bucket["condition_models"].add(key)
                for cutoff in SEARCH_BOTTLENECK_CUTOFFS:
                    tool_bucket["calls"][cutoff] += 1
                    if cutoff_hits[cutoff]:
                        tool_bucket["hits"][cutoff] += 1

            if not search_events:
                cm_task_counts[key]["tasks_without_search"] += 1
                condition_task_counts[condition_label]["tasks_without_search"] += 1
                continue

            ordered_events = _order_events(search_events)
            ordered_trajectory = _order_events(trajectory_events)
            cumulative_retrieved_gold: set[str] = set()
            cumulative_read_gold: set[str] = set()
            current_search_call_index = 0
            current_search_call_row: Optional[dict] = None
            for event in ordered_trajectory:
                if str(event.get("event_type", "")) == "read":
                    if use_source_gold or use_gold_units:
                        cumulative_read_gold.update(set(event.get("read_gold_ids") or []))
                    else:
                        cumulative_read_gold.update(set(event.get("read_ids") or []) & gold_ids)
                    cumulative_read_gold_units = min(len(cumulative_read_gold), n_gold_ids)
                    if current_search_call_row is not None:
                        current_search_call_row["cumulative_read_gold_count"] = cumulative_read_gold_units
                        current_search_call_row["cumulative_read_gold_recall"] = (
                            round(cumulative_read_gold_units / n_gold_ids, 4) if n_gold_ids else 0.0
                        )
                    continue

                current_search_call_index += 1
                if use_source_gold or use_gold_units:
                    cumulative_retrieved_gold.update(set(event.get("result_gold_unit_ids") or []))
                else:
                    cumulative_retrieved_gold.update(set(event.get("result_gold_ids") or []))
                cumulative_retrieved_gold_units = min(len(cumulative_retrieved_gold), n_gold_ids)
                cumulative_read_gold_units = min(len(cumulative_read_gold), n_gold_ids)
                per_call_row = {
                    "condition_model": key,
                    "condition": condition_label,
                    "variant": variant,
                    "base_condition": base_condition,
                    "model": model,
                    "task_id": _task_id_for_row(str(event.get("task_id", "") or task_id)),
                    "search_tool": str(event.get("tool", "") or ""),
                    "turn": event.get("parsed_turn"),
                    "search_call_index": current_search_call_index,
                    "results_returned": len(list(event.get("result_ids") or [])),
                    "n_gold_datasets": n_gold_ids,
                    "cumulative_search_gold_count": cumulative_retrieved_gold_units,
                    "cumulative_search_gold_recall": round(cumulative_retrieved_gold_units / n_gold_ids, 4)
                    if n_gold_ids
                    else 0.0,
                    "cumulative_read_gold_count": cumulative_read_gold_units,
                    "cumulative_read_gold_recall": round(cumulative_read_gold_units / n_gold_ids, 4)
                    if n_gold_ids
                    else 0.0,
                }
                for cutoff in SEARCH_BOTTLENECK_CUTOFFS:
                    hit_count = int(event.get("cutoff_hit_counts", {}).get(cutoff, 0) or 0)
                    per_call_row[f"gold_hits_top_{cutoff}"] = hit_count
                    per_call_row[f"gold_in_top_{cutoff}"] = 1 if hit_count else 0
                    per_call_row[f"gold_recall_top_{cutoff}"] = round(hit_count / n_gold_ids, 4) if n_gold_ids else 0.0
                per_call_rows.append(per_call_row)
                current_search_call_row = per_call_row

            task_row = {
                "condition_model": key,
                "condition": condition_label,
                "variant": variant,
                "base_condition": base_condition,
                "model": model,
                "task_id": _task_id_for_row(str(ordered_events[0].get("task_id", task_id))),
                "num_search_calls": len(ordered_events),
            }
            for cutoff in SEARCH_BOTTLENECK_CUTOFFS:
                first_hit_round = next(
                    (
                        round_number
                        for round_number, event in enumerate(ordered_events, start=1)
                        if bool(event.get("cutoff_hits", {}).get(cutoff))
                    ),
                    None,
                )
                task_row[f"found_top_{cutoff}"] = 1 if first_hit_round is not None else 0
                task_row[f"first_hit_round_top_{cutoff}"] = first_hit_round
                task_row[f"wasted_rounds_top_{cutoff}"] = (
                    first_hit_round - 1 if first_hit_round is not None else len(ordered_events)
                )
            per_task_rows.append(task_row)

            events_by_tool: Dict[str, List[dict]] = defaultdict(list)
            for event in search_events:
                events_by_tool[str(event["tool"])].append(event)

            for tool_name, tool_events in sorted(events_by_tool.items(), key=lambda item: _tool_sort_key(item[0])):
                ordered_tool_events = _order_events(tool_events)
                task_tool_row = {
                    "condition_model": key,
                    "condition": condition_label,
                    "variant": variant,
                    "base_condition": base_condition,
                    "model": model,
                    "task_id": _task_id_for_row(str(ordered_tool_events[0].get("task_id", task_id))),
                    "search_tool": tool_name,
                    "num_tool_calls": len(ordered_tool_events),
                }
                for cutoff in SEARCH_BOTTLENECK_CUTOFFS:
                    first_hit_round = next(
                        (
                            round_number
                            for round_number, event in enumerate(ordered_tool_events, start=1)
                            if bool(event.get("cutoff_hits", {}).get(cutoff))
                        ),
                        None,
                    )
                    task_tool_row[f"found_top_{cutoff}"] = 1 if first_hit_round is not None else 0
                    task_tool_row[f"first_hit_round_top_{cutoff}"] = first_hit_round
                    task_tool_row[f"wasted_rounds_top_{cutoff}"] = (
                        first_hit_round - 1 if first_hit_round is not None else len(ordered_tool_events)
                    )
                per_task_tool_rows.append(task_tool_row)

    cm_summary: Dict[str, dict] = {}
    cm_groups: Dict[str, List[dict]] = defaultdict(list)
    for row in per_task_rows:
        cm_groups[str(row["condition_model"])].append(row)

    for key, rows in sorted(cm_groups.items()):
        meta = meta_by_key.get(key)
        if meta is None:
            variant, base_condition, model = _split_cm_key(key)
            condition_label = _condition_label(variant, base_condition)
        else:
            variant = str(meta.get("variant", "unknown"))
            base_condition = str(meta.get("base_condition", variant))
            model = str(meta.get("model", "unknown"))
            condition_label = str(meta.get("condition_label", variant))
        cm_row = {
            "condition_model": key,
            "condition": condition_label,
            "variant": variant,
            "base_condition": base_condition,
            "model": model,
            "n_tasks_with_search": len(rows),
            "n_tasks_without_search": int(cm_task_counts[key]["tasks_without_search"]),
        }
        for cutoff in SEARCH_BOTTLENECK_CUTOFFS:
            stats = _summarize_first_hit_rows(rows, cutoff=cutoff, rounds_field="num_search_calls")
            cm_row[f"found_tasks_top_{cutoff}"] = stats["found_tasks"]
            cm_row[f"not_found_tasks_top_{cutoff}"] = stats["not_found_tasks"]
            cm_row[f"top{cutoff}_not_found_rate"] = stats["not_found_rate"]
            cm_row[f"avg_first_hit_round_top_{cutoff}"] = stats["avg_first_hit_round_found_only"]
            cm_row[f"avg_wasted_rounds_top_{cutoff}"] = stats["avg_wasted_rounds"]
        cm_summary[key] = cm_row

    condition_rows: List[dict] = []
    condition_groups: Dict[str, List[dict]] = defaultdict(list)
    for row in per_task_rows:
        condition_groups[str(row["condition"])].append(row)

    for condition_label, rows in sorted(condition_groups.items(), key=lambda item: _condition_sort_key(item[0])):
        variant = str(rows[0]["variant"])
        base_condition = str(rows[0]["base_condition"])
        models = sorted({str(row["model"]) for row in rows})
        num_condition_model_rows = len({str(row["condition_model"]) for row in rows})
        tasks_without_search = int(condition_task_counts[condition_label]["tasks_without_search"])
        for cutoff in SEARCH_BOTTLENECK_CUTOFFS:
            stats = _summarize_first_hit_rows(rows, cutoff=cutoff, rounds_field="num_search_calls")
            condition_rows.append(
                {
                    "condition": condition_label,
                    "variant": variant,
                    "base_condition": base_condition,
                    "cutoff": cutoff,
                    "n_tasks_with_search": len(rows),
                    "n_tasks_without_search": tasks_without_search,
                    "found_tasks": stats["found_tasks"],
                    "not_found_tasks": stats["not_found_tasks"],
                    "not_found_rate": stats["not_found_rate"],
                    "ever_found_rate": stats["ever_found_rate"],
                    "avg_first_hit_round_found_only": stats["avg_first_hit_round_found_only"],
                    "median_first_hit_round_found_only": stats["median_first_hit_round_found_only"],
                    "avg_wasted_rounds": stats["avg_wasted_rounds"],
                    "max_round": stats["max_round"],
                    "models": models,
                    "num_condition_model_rows": num_condition_model_rows,
                    "round_counts": stats["round_counts"],
                    "cumulative_found_counts": stats["cumulative_found_counts"],
                    "cumulative_found_rates": stats["cumulative_found_rates"],
                }
            )

    tool_first_hit_rows: List[dict] = []
    tool_groups: Dict[str, List[dict]] = defaultdict(list)
    for row in per_task_tool_rows:
        tool_groups[str(row["search_tool"])].append(row)

    for search_tool, rows in sorted(tool_groups.items(), key=lambda item: _tool_sort_key(item[0])):
        conditions = sorted({str(row["condition"]) for row in rows}, key=_condition_sort_key)
        models = sorted({str(row["model"]) for row in rows})
        for cutoff in SEARCH_BOTTLENECK_CUTOFFS:
            stats = _summarize_first_hit_rows(rows, cutoff=cutoff, rounds_field="num_tool_calls")
            tool_first_hit_rows.append(
                {
                    "search_tool": search_tool,
                    "cutoff": cutoff,
                    "tasks_with_tool": len(rows),
                    "found_tasks": stats["found_tasks"],
                    "not_found_tasks": stats["not_found_tasks"],
                    "not_found_rate": stats["not_found_rate"],
                    "ever_found_rate": stats["ever_found_rate"],
                    "avg_first_hit_round_found_only": stats["avg_first_hit_round_found_only"],
                    "median_first_hit_round_found_only": stats["median_first_hit_round_found_only"],
                    "avg_wasted_rounds": stats["avg_wasted_rounds"],
                    "max_round": stats["max_round"],
                    "conditions": conditions,
                    "models": models,
                    "num_condition_model_rows": len({str(row["condition_model"]) for row in rows}),
                    "round_counts": stats["round_counts"],
                    "cumulative_found_counts": stats["cumulative_found_counts"],
                    "cumulative_found_rates": stats["cumulative_found_rates"],
                }
            )

    tool_miss_rows: List[dict] = []
    for search_tool, stats in sorted(tool_miss_acc.items(), key=lambda item: _tool_sort_key(item[0])):
        conditions = sorted(stats["conditions"], key=_condition_sort_key)
        models = sorted(stats["models"])
        num_condition_model_rows = len(stats["condition_models"])
        for cutoff in SEARCH_BOTTLENECK_CUTOFFS:
            n_calls = int(stats["calls"][cutoff])
            hit_calls = int(stats["hits"][cutoff])
            miss_calls = n_calls - hit_calls
            tool_miss_rows.append(
                {
                    "search_tool": search_tool,
                    "cutoff": cutoff,
                    "n_calls": n_calls,
                    "hit_calls": hit_calls,
                    "miss_calls": miss_calls,
                    "miss_rate": round(miss_calls / n_calls, 4) if n_calls else 0.0,
                    "conditions": conditions,
                    "models": models,
                    "num_condition_model_rows": num_condition_model_rows,
                }
            )

    tool_efficiency_rows: List[dict] = []
    for row in tool_first_hit_rows:
        tool_efficiency_rows.append(
            {
                "search_tool": row["search_tool"],
                "cutoff": row["cutoff"],
                "tasks_with_tool": row["tasks_with_tool"],
                "ever_found_rate": row["ever_found_rate"],
                "avg_first_hit_round_found_only": row["avg_first_hit_round_found_only"],
                "avg_wasted_rounds": row["avg_wasted_rounds"],
                "conditions": row["conditions"],
                "models": row["models"],
                "num_condition_model_rows": row["num_condition_model_rows"],
            }
        )

    return {
        "condition_summary_rows": condition_rows,
        "tool_first_hit_rows": tool_first_hit_rows,
        "tool_miss_rows": tool_miss_rows,
        "tool_efficiency_rows": tool_efficiency_rows,
        "per_task_rows": per_task_rows,
        "per_task_tool_rows": per_task_tool_rows,
        "per_call_rows": per_call_rows,
        "condition_model_summary": cm_summary,
    }


def iter_csv_rows(rows: Iterable[dict], json_fields: Iterable[str]) -> Iterable[dict]:
    json_field_set = set(json_fields)
    for row in rows:
        normalized = dict(row)
        for field in json_field_set:
            if field in normalized:
                normalized[field] = "" if normalized[field] is None else str(normalized[field])
        yield normalized


def write_search_bottleneck_csv(rows: List[dict], output_csv: Path, fieldnames: List[str]) -> int:
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            normalized = {}
            for field in fieldnames:
                value = row.get(field)
                if isinstance(value, (list, dict)):
                    normalized[field] = json.dumps(value, sort_keys=isinstance(value, dict))
                elif value is None:
                    normalized[field] = ""
                else:
                    normalized[field] = value
            writer.writerow(normalized)
    return len(rows)


def _import_plot_libs():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception:
        print("Skipping bottleneck figures: matplotlib is not installed.")
        return None


def _cumulative_series(cumulative: dict, max_round: int) -> List[float]:
    """Expand a sparse per-round cumulative dict to a dense list of percentages.

    Forward-fills past the last recorded round so plateaued curves don't drop to 0.
    """
    ys: List[float] = []
    last = 0.0
    for round_number in range(1, max_round + 1):
        raw = cumulative.get(str(round_number))
        if raw is None:
            raw = cumulative.get(round_number)
        if raw is not None:
            last = float(raw)
        ys.append(100.0 * last)
    return ys


def _pad_axis_for_labels(ax, *, pct: bool = False, min_top: float = 1.0) -> None:
    """Ensure bar-label annotations rendered above bars stay inside the axes."""
    try:
        top = float(ax.get_ylim()[1])
    except Exception:
        return
    headroom = max(top * 0.12, 1.0 if pct else 0.1)
    ax.set_ylim(0, max(min_top, top + headroom))


def _annotate_bar_values(ax, bars, *, pct: bool = False, min_value: float = 0.0) -> None:
    for bar in bars:
        value = float(bar.get_height())
        if value < min_value:
            continue
        label = f"{value:.0f}%" if pct else f"{value:.1f}".rstrip("0").rstrip(".")
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + (1.0 if pct else max(0.05, value * 0.03)),
            label,
            ha="center",
            va="bottom",
            fontsize=8,
        )


def _plot_first_hit_rounds_condition(
    plt,
    rows: List[dict],
    output_path: Path,
    *,
    label_formatter: Optional[Callable[[dict], str]] = None,
) -> None:
    if not rows:
        return
    ordered_conditions = sorted({str(row.get("condition", "")) for row in rows}, key=_condition_sort_key)
    condition_rows = {}
    for row in rows:
        condition_rows.setdefault(str(row.get("condition", "")), row)
    display_labels = {
        condition: _condition_display_label(condition_rows.get(condition, {}), label_formatter)
        for condition in ordered_conditions
    }
    max_round = max(int(row.get("max_round", 0) or 0) for row in rows)
    if max_round <= 0:
        return

    fig, axes = plt.subplots(1, len(SEARCH_BOTTLENECK_CUTOFFS), figsize=(18, 5), sharey=True)
    for ax, cutoff in zip(axes, SEARCH_BOTTLENECK_CUTOFFS):
        cutoff_rows = {str(row["condition"]): row for row in rows if int(row.get("cutoff", 0) or 0) == cutoff}
        for condition in ordered_conditions:
            row = cutoff_rows.get(condition)
            if row is None:
                continue
            xs = list(range(1, max_round + 1))
            ys = _cumulative_series(row.get("cumulative_found_rates", {}), max_round)
            not_found_pct = 100.0 * float(row.get("not_found_rate", 0.0) or 0.0)
            ax.plot(
                xs,
                ys,
                marker="o",
                linewidth=2,
                label=f"{display_labels.get(condition, condition)} (nf {not_found_pct:.0f}%)",
            )
        ax.set_title(f"Top-{cutoff}")
        ax.set_xlabel("Search Round")
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.25, linestyle="--", linewidth=0.7)
        if ax is axes[0]:
            ax.set_ylabel("Tasks Found by Round (%)")
    handles, labels = axes[-1].get_legend_handles_labels()
    legend_artist = None
    if handles:
        legend_artist = fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
    fig.suptitle("First Gold Hit by Search Round and Condition")
    fig.tight_layout(rect=(0, 0, 0.83, 0.95))
    extra_artists = (legend_artist,) if legend_artist is not None else ()
    fig.savefig(output_path, bbox_inches="tight", bbox_extra_artists=extra_artists)
    plt.close(fig)


def _plot_cumulative_gold_recall_condition(
    plt,
    rows: List[dict],
    output_path: Path,
    *,
    recall_field: str,
    title: str,
    label_formatter: Optional[Callable[[dict], str]] = None,
) -> None:
    if not rows:
        return

    ordered_conditions = sorted({str(row.get("condition", "")) for row in rows}, key=_condition_sort_key)
    condition_rows = {}
    for row in rows:
        condition_rows.setdefault(str(row.get("condition", "")), row)
    display_labels = {
        condition: _condition_display_label(condition_rows.get(condition, {}), label_formatter)
        for condition in ordered_conditions
    }

    series_by_condition: Dict[str, Dict[str, Dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
    global_max_call = 0
    for row in rows:
        condition = str(row.get("condition", ""))
        task_id = str(row.get("task_id", ""))
        search_call_index = int(row.get("search_call_index", 0) or 0)
        if not condition or not task_id or search_call_index <= 0:
            continue
        recall = row.get(recall_field)
        if recall in (None, ""):
            continue
        series_by_condition[condition][task_id][search_call_index] = float(recall)
        global_max_call = max(global_max_call, search_call_index)

    if global_max_call <= 0:
        return

    fig, ax = plt.subplots(figsize=(11, 6))
    for condition in ordered_conditions:
        task_series = series_by_condition.get(condition, {})
        if not task_series:
            continue
        xs = list(range(1, global_max_call + 1))
        ys: List[float] = []
        for search_call_index in xs:
            recalls_at_k: List[float] = []
            for series in task_series.values():
                last = 0.0
                for call_idx in range(1, search_call_index + 1):
                    if call_idx in series:
                        last = float(series[call_idx])
                recalls_at_k.append(last)
            ys.append(100.0 * (sum(recalls_at_k) / len(recalls_at_k)) if recalls_at_k else 0.0)
        ax.plot(xs, ys, marker="o", linewidth=2, label=display_labels.get(condition, condition))

    ax.set_xlabel("Search Call")
    ax.set_ylabel("Mean Cumulative Gold Recall (%)")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.7)
    ax.set_title(title)

    handles, labels = ax.get_legend_handles_labels()
    legend_artist = None
    if handles:
        legend_artist = fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
    fig.tight_layout(rect=(0, 0, 0.83, 1.0))
    extra_artists = (legend_artist,) if legend_artist is not None else ()
    fig.savefig(output_path, bbox_inches="tight", bbox_extra_artists=extra_artists)
    plt.close(fig)


def _plot_first_hit_rounds_tool(plt, rows: List[dict], output_path: Path) -> None:
    if not rows:
        return
    ordered_tools = sorted({str(row.get("search_tool", "")) for row in rows}, key=_tool_sort_key)
    max_round = max(int(row.get("max_round", 0) or 0) for row in rows)
    if max_round <= 0:
        return

    fig, axes = plt.subplots(1, len(SEARCH_BOTTLENECK_CUTOFFS), figsize=(18, 5), sharey=True)
    for ax, cutoff in zip(axes, SEARCH_BOTTLENECK_CUTOFFS):
        cutoff_rows = {str(row["search_tool"]): row for row in rows if int(row.get("cutoff", 0) or 0) == cutoff}
        for tool_name in ordered_tools:
            row = cutoff_rows.get(tool_name)
            if row is None:
                continue
            xs = list(range(1, max_round + 1))
            ys = _cumulative_series(row.get("cumulative_found_rates", {}), max_round)
            not_found_pct = 100.0 * float(row.get("not_found_rate", 0.0) or 0.0)
            ax.plot(
                xs,
                ys,
                marker="o",
                linewidth=2,
                label=f"{tool_name} (nf {not_found_pct:.0f}%)",
                color=SEARCH_TOOL_COLORS.get(tool_name),
            )
        ax.set_title(f"Top-{cutoff}")
        ax.set_xlabel("Tool-Local Search Round")
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.25, linestyle="--", linewidth=0.7)
        if ax is axes[0]:
            ax.set_ylabel("Tasks Found by Round (%)")
    handles, labels = axes[-1].get_legend_handles_labels()
    legend_artist = None
    if handles:
        legend_artist = fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
    fig.suptitle("First Gold Hit by Search Tool and Tool-Local Round")
    fig.tight_layout(rect=(0, 0, 0.83, 0.95))
    extra_artists = (legend_artist,) if legend_artist is not None else ()
    fig.savefig(output_path, bbox_inches="tight", bbox_extra_artists=extra_artists)
    plt.close(fig)


def _plot_topk_miss_by_tool(plt, rows: List[dict], output_path: Path) -> None:
    if not rows:
        return
    ordered_tools = sorted({str(row.get("search_tool", "")) for row in rows}, key=_tool_sort_key)
    cutoff_to_rows = {
        cutoff: {str(row["search_tool"]): row for row in rows if int(row.get("cutoff", 0) or 0) == cutoff}
        for cutoff in SEARCH_BOTTLENECK_CUTOFFS
    }
    fig, ax = plt.subplots(figsize=(max(9, len(ordered_tools) * 1.4), 6))
    x_positions = list(range(len(ordered_tools)))
    width = 0.22
    for idx, cutoff in enumerate(SEARCH_BOTTLENECK_CUTOFFS):
        offsets = [x + (idx - 1) * width for x in x_positions]
        values = [
            100.0 * float(cutoff_to_rows[cutoff].get(tool_name, {}).get("miss_rate", 0.0) or 0.0)
            for tool_name in ordered_tools
        ]
        bars = ax.bar(offsets, values, width=width * 0.92, label=f"Top-{cutoff}")
        _annotate_bar_values(ax, bars, pct=True, min_value=4.0)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(ordered_tools, rotation=20, ha="right")
    ax.set_ylabel("Miss Rate (%)")
    ax.set_title("Gold Miss Rate by Search Tool")
    ax.legend()
    ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.7)
    _pad_axis_for_labels(ax, pct=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_search_tool_coverage_waste(plt, rows: List[dict], output_path: Path) -> None:
    if not rows:
        return
    ordered_tools = sorted({str(row.get("search_tool", "")) for row in rows}, key=_tool_sort_key)
    cutoff_to_rows = {
        cutoff: {str(row["search_tool"]): row for row in rows if int(row.get("cutoff", 0) or 0) == cutoff}
        for cutoff in SEARCH_BOTTLENECK_CUTOFFS
    }
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    x_positions = list(range(len(ordered_tools)))
    width = 0.22

    for idx, cutoff in enumerate(SEARCH_BOTTLENECK_CUTOFFS):
        offsets = [x + (idx - 1) * width for x in x_positions]
        coverage = [
            100.0 * float(cutoff_to_rows[cutoff].get(tool_name, {}).get("ever_found_rate", 0.0) or 0.0)
            for tool_name in ordered_tools
        ]
        wasted = [
            float(cutoff_to_rows[cutoff].get(tool_name, {}).get("avg_wasted_rounds", 0.0) or 0.0)
            for tool_name in ordered_tools
        ]
        bars_cov = axes[0].bar(offsets, coverage, width=width * 0.92, label=f"Top-{cutoff}")
        bars_waste = axes[1].bar(offsets, wasted, width=width * 0.92, label=f"Top-{cutoff}")
        _annotate_bar_values(axes[0], bars_cov, pct=True, min_value=4.0)
        _annotate_bar_values(axes[1], bars_waste, pct=False, min_value=0.15)

    axes[0].set_title("Ever Found Rate by Search Tool")
    axes[0].set_ylabel("Tasks Ever Found (%)")
    axes[1].set_title("Average Wasted Rounds Before First Hit")
    axes[1].set_ylabel("Wasted Rounds")
    for ax in axes:
        ax.set_xticks(x_positions)
        ax.set_xticklabels(ordered_tools, rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.7)
    axes[0].legend()
    _pad_axis_for_labels(axes[0], pct=True)
    _pad_axis_for_labels(axes[1], pct=False)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_topk_miss_by_condition(
    plt,
    rows: List[dict],
    output_path: Path,
    *,
    label_formatter: Optional[Callable[[dict], str]] = None,
) -> None:
    if not rows:
        return
    ordered_conditions = sorted({str(row.get("condition", "")) for row in rows}, key=_condition_sort_key)
    condition_rows = {}
    for row in rows:
        condition_rows.setdefault(str(row.get("condition", "")), row)
    display_labels = [
        _condition_display_label(condition_rows.get(condition, {}), label_formatter)
        for condition in ordered_conditions
    ]
    cutoff_to_rows = {
        cutoff: {str(row["condition"]): row for row in rows if int(row.get("cutoff", 0) or 0) == cutoff}
        for cutoff in SEARCH_BOTTLENECK_CUTOFFS
    }
    fig, ax = plt.subplots(figsize=(max(9, len(ordered_conditions) * 1.8), 6))
    x_positions = list(range(len(ordered_conditions)))
    width = 0.22
    for idx, cutoff in enumerate(SEARCH_BOTTLENECK_CUTOFFS):
        offsets = [x + (idx - 1) * width for x in x_positions]
        values = [
            100.0 * float(cutoff_to_rows[cutoff].get(condition, {}).get("not_found_rate", 0.0) or 0.0)
            for condition in ordered_conditions
        ]
        bars = ax.bar(offsets, values, width=width * 0.92, label=f"Top-{cutoff}")
        _annotate_bar_values(ax, bars, pct=True, min_value=4.0)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(display_labels, rotation=20, ha="right")
    ax.set_ylabel("Miss Rate (%)")
    ax.set_title("Gold Miss Rate by Variant")
    ax.legend()
    ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.7)
    _pad_axis_for_labels(ax, pct=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_search_condition_coverage_waste(
    plt,
    rows: List[dict],
    output_path: Path,
    *,
    label_formatter: Optional[Callable[[dict], str]] = None,
) -> None:
    if not rows:
        return
    ordered_conditions = sorted({str(row.get("condition", "")) for row in rows}, key=_condition_sort_key)
    condition_rows = {}
    for row in rows:
        condition_rows.setdefault(str(row.get("condition", "")), row)
    display_labels = [
        _condition_display_label(condition_rows.get(condition, {}), label_formatter)
        for condition in ordered_conditions
    ]
    cutoff_to_rows = {
        cutoff: {str(row["condition"]): row for row in rows if int(row.get("cutoff", 0) or 0) == cutoff}
        for cutoff in SEARCH_BOTTLENECK_CUTOFFS
    }
    fig, axes = plt.subplots(1, 2, figsize=(max(16, len(ordered_conditions) * 2.1), 6))
    x_positions = list(range(len(ordered_conditions)))
    width = 0.22

    for idx, cutoff in enumerate(SEARCH_BOTTLENECK_CUTOFFS):
        offsets = [x + (idx - 1) * width for x in x_positions]
        coverage = [
            100.0 * float(cutoff_to_rows[cutoff].get(condition, {}).get("ever_found_rate", 0.0) or 0.0)
            for condition in ordered_conditions
        ]
        wasted = [
            float(cutoff_to_rows[cutoff].get(condition, {}).get("avg_wasted_rounds", 0.0) or 0.0)
            for condition in ordered_conditions
        ]
        bars_cov = axes[0].bar(offsets, coverage, width=width * 0.92, label=f"Top-{cutoff}")
        bars_waste = axes[1].bar(offsets, wasted, width=width * 0.92, label=f"Top-{cutoff}")
        _annotate_bar_values(axes[0], bars_cov, pct=True, min_value=4.0)
        _annotate_bar_values(axes[1], bars_waste, pct=False, min_value=0.15)

    axes[0].set_title("Ever Found Rate by Variant")
    axes[0].set_ylabel("Tasks Ever Found (%)")
    axes[1].set_title("Average Wasted Rounds Before First Hit")
    axes[1].set_ylabel("Wasted Rounds")
    for ax in axes:
        ax.set_xticks(x_positions)
        ax.set_xticklabels(display_labels, rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.7)
    axes[0].legend()
    _pad_axis_for_labels(axes[0], pct=True)
    _pad_axis_for_labels(axes[1], pct=False)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def generate_search_bottleneck_figures(
    search_bottleneck: dict,
    output_dir: Path,
    *,
    include_condition_breakouts: bool = False,
    condition_label_formatter: Optional[Callable[[dict], str]] = None,
) -> None:
    plt = _import_plot_libs()
    if plt is None:
        return
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    _plot_cumulative_gold_recall_condition(
        plt,
        search_bottleneck.get("per_call_rows", []),
        fig_dir / "search_cumulative_retrieval_recall_by_condition.pdf",
        recall_field="cumulative_search_gold_recall",
        title="Cumulative Gold Retrieval Recall by Search Call and Variant",
        label_formatter=condition_label_formatter,
    )
    _plot_cumulative_gold_recall_condition(
        plt,
        search_bottleneck.get("per_call_rows", []),
        fig_dir / "search_cumulative_access_recall_by_condition.pdf",
        recall_field="cumulative_read_gold_recall",
        title="Cumulative Gold Access Recall by Search Call and Variant",
        label_formatter=condition_label_formatter,
    )
    _plot_topk_miss_by_tool(
        plt,
        search_bottleneck.get("tool_miss_rows", []),
        fig_dir / "search_topk_miss_by_tool.pdf",
    )
    _plot_search_tool_coverage_waste(
        plt,
        search_bottleneck.get("tool_efficiency_rows", []),
        fig_dir / "search_tool_coverage_waste.pdf",
    )
    if include_condition_breakouts:
        _plot_topk_miss_by_condition(
            plt,
            search_bottleneck.get("condition_summary_rows", []),
            fig_dir / "search_topk_miss_by_condition.pdf",
            label_formatter=condition_label_formatter,
        )
        _plot_search_condition_coverage_waste(
            plt,
            search_bottleneck.get("condition_summary_rows", []),
            fig_dir / "search_condition_coverage_waste.pdf",
            label_formatter=condition_label_formatter,
        )
