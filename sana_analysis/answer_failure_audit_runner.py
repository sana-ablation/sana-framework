#!/usr/bin/env python3
"""Run answer-failure audits through external model calls and temp JSON files."""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sana_analysis.running_analysis.answer_failure_taxonomy import (
    format_answer_failure_type_definitions,
    format_blocker_subtypes,
    format_boundary_rules,
    format_failure_stages,
)
from sana_analysis.running_analysis.answer_failure_validation import (
    EVENT_COLUMNS,
    MODEL_VALIDATION_NOTES_FIELD,
    MODEL_VALIDATION_STATUS_FIELD,
    ROW_OUTPUT_COLUMNS,
    VALIDATION_NOTES_FIELD,
    VALIDATION_STATUS_FIELD,
    is_non_correct_row,
    row_log_path,
    validate_answer_failure_root,
)
from sana_analysis.report_generator.build_answer_failure_report import build_answer_failure_report


SOURCE_ROOTS = {
    "results_semantic": ("results_semantic_answer_failures", Path("logs/modes")),
    "results-ec2_semantic": ("results-ec2_semantic_answer_failures", Path("logs-ec2/modes")),
    "sana-results_semantic": ("sana-results_semantic_answer_failures", Path("sana-results/logs/modes")),
    "results-kramabench_semantic": ("results-kramabench_semantic_answer_failures", Path("log-kramabench/modes")),
}

PLAN_ROOTS = {
    "tasks_mini": "plans_mini",
    "tasks_mini_core": "plans_mini_core",
    "tasks-mini-kramabench": "plans-mini-kramabench",
    "tasks_core_quality": "plans_core_quality",
}

DEFAULT_ROW_MODEL = "gpt-5.4-mini"
DEFAULT_ROW_REASONING = "high"
DEFAULT_AUDITOR_MODEL = "gpt-5.4-mini"
DEFAULT_AUDITOR_REASONING = "high"
DEFAULT_ROW_AUDIT_PIPELINE = "two-stage"
CURRENT_TRACE_SCHEMA_VERSION = "answer_failure_trace_v2"
CURRENT_AUDIT_SCHEMA_VERSION = "answer_failure_audit_v2"

TRACE_ROLE_TO_FAILURE_STAGE = {
    "source_choice": "source_selection",
    "extraction": "extraction",
    "computation": "computation",
    "final_answer": "finalization",
    "blocker": "data_access",
    "plan_mismatch": "task_understanding",
    "other": "evidence_gathering",
}

TRACE_UNCERTAINTY_TO_CONFIDENCE = {
    "low": "high",
    "medium": "medium",
    "high": "low",
}

ANSWER_FAILURE_TYPE_ALIASES = {
    "computation": "computation_or_aggregation_error",
    "computation_error": "computation_or_aggregation_error",
    "computation_failure": "computation_or_aggregation_error",
    "data_selection_or_computation_error": "wrong_scope_or_filter",
    "filter_error": "wrong_scope_or_filter",
    "reasoning": "computation_or_aggregation_error",
    "reasoning_error": "computation_or_aggregation_error",
    "scope_error": "wrong_scope_or_filter",
    "wrong_filter_or_row_subset": "wrong_scope_or_filter",
    "wrong_scope_or_population": "wrong_scope_or_filter",
    "wrong_source_or_scope": "wrong_scope_or_filter",
    "data_access": "tool_or_data_blocker",
    "data_access_error": "tool_or_data_blocker",
    "data_access_failure": "tool_or_data_blocker",
    "system_failure": "tool_or_data_blocker",
    "tool_failure": "tool_or_data_blocker",
    "tool_misuse": "tool_or_data_blocker",
    "incomplete_evidence_not_enough_turns": "incomplete_evidence_budget_exhausted",
    "tool_limit": "incomplete_evidence_budget_exhausted",
    "tool_exhaustion": "incomplete_evidence_budget_exhausted",
    "evidence_gap": "incomplete_evidence_early_answer",
    "missing_evidence": "incomplete_evidence_early_answer",
    "source_selection": "wrong_source_or_dataset",
    "source_selection_error": "wrong_source_or_dataset",
    "source_selection_failure": "wrong_source_or_dataset",
    "source_mistake": "wrong_source_or_dataset",
    "source_mismatch": "wrong_source_or_dataset",
    "wrong_source": "wrong_source_or_dataset",
    "evidence_gathering_error": "incomplete_evidence_early_answer",
    "task_understanding": "question_or_constraint_misread",
    "entity_confusion": "question_or_constraint_misread",
    "extraction": "extraction_or_parsing_error",
    "extraction_error": "extraction_or_parsing_error",
    "submission": "evidence_available_answer_error",
    "submission_error": "evidence_available_answer_error",
    "submission_failure": "evidence_available_answer_error",
    "finalization": "evidence_available_answer_error",
    "finalization_error": "evidence_available_answer_error",
    "formatting": "evidence_available_answer_error",
    "hallucination": "evidence_available_answer_error",
    "hallucinated_answer": "evidence_available_answer_error",
    "unsupported_estimate": "evidence_available_answer_error",
    "wrong_answer": "evidence_available_answer_error",
    "retrieval": "incomplete_evidence_early_answer",
    "evidence_gathering": "incomplete_evidence_early_answer",
}


@dataclass(frozen=True)
class AuditLayout:
    repo_root: Path
    source_root: Path
    output_root: Path
    logs_root: Path
    eval_path: Path
    mirrored_eval_path: Path
    mirrored_events_path: Path
    mirrored_report_path: Path
    model_variant: str
    mode_variant: str


@dataclass(frozen=True)
class PendingEvalStats:
    eval_path: Path
    eligible_rows: int
    pending_rows: int
    invalid_rows: int
    has_all_artifacts: bool
    pending_task_ids: tuple[str, ...]
    invalid_task_ids: tuple[str, ...]


def infer_layout(eval_path: str | Path) -> AuditLayout:
    eval_path = Path(eval_path).resolve()
    source_root: Optional[Path] = None
    output_root_name = ""
    logs_rel = Path()

    for parent in [eval_path.parent, *eval_path.parents]:
        if parent.name in SOURCE_ROOTS:
            source_root = parent
            output_root_name, logs_rel = SOURCE_ROOTS[parent.name]
            break
    if source_root is None:
        known = ", ".join(sorted(SOURCE_ROOTS))
        raise ValueError(f"Could not infer source root for {eval_path}; expected one of: {known}")

    rel_path = eval_path.relative_to(source_root)
    parts = rel_path.parts
    if "modes" not in parts:
        raise ValueError(f"Expected eval path under a modes/ tree: {eval_path}")
    modes_index = parts.index("modes")
    after_modes = parts[modes_index + 1 :]
    if len(after_modes) < 3 or after_modes[-1] != "eval_results.csv":
        raise ValueError(f"Expected .../modes/{{model_variant}}/{{mode_variant}}/eval_results.csv: {eval_path}")

    repo_root = source_root.parent
    output_root = repo_root / output_root_name
    mirrored_eval_path = output_root / rel_path
    model_variant = after_modes[0]
    mode_variant = after_modes[1]
    return AuditLayout(
        repo_root=repo_root,
        source_root=source_root,
        output_root=output_root,
        logs_root=repo_root / logs_rel,
        eval_path=eval_path,
        mirrored_eval_path=mirrored_eval_path,
        mirrored_events_path=mirrored_eval_path.parent / "answer_failure_events.csv",
        mirrored_report_path=mirrored_eval_path.parent / "answer_failure_report.md",
        model_variant=model_variant,
        mode_variant=mode_variant,
    )


def _escape_quotes_in_backtick_spans_inside_json_strings(text: str) -> str:
    out: list[str] = []
    in_string = False
    in_backtick_span = False
    escaped = False
    for char in text:
        if escaped:
            out.append(char)
            escaped = False
            continue
        if in_string and char == "\\":
            out.append(char)
            escaped = True
            continue
        if in_string and char == "`":
            out.append(char)
            in_backtick_span = not in_backtick_span
            continue
        if in_string and in_backtick_span and char == '"':
            out.append('\\"')
            continue
        if char == '"':
            out.append(char)
            in_string = not in_string
            if not in_string:
                in_backtick_span = False
            continue
        out.append(char)
    return "".join(out)


def parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fence:
        text = fence.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end >= start:
            text = text[start : end + 1]
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        repaired = _escape_quotes_in_backtick_spans_inside_json_strings(text)
        if repaired == text:
            raise
        try:
            parsed = json.loads(repaired)
        except json.JSONDecodeError:
            raise exc
    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object")
    return parsed


def _read_source_rows(eval_path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with eval_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def _read_text_if_exists(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(errors="replace")


def _resolve_plan_path(repo_root: Path, task_id: str) -> Optional[Path]:
    task_path = Path(task_id)
    if not task_path.parts:
        return None
    plan_root = PLAN_ROOTS.get(task_path.parts[0])
    if plan_root is None:
        return None
    plan_path = repo_root / plan_root / Path(*task_path.parts[1:])
    return plan_path if plan_path.exists() else None


def _load_prompt_artifacts(
    *,
    layout: AuditLayout,
    source_row: dict[str, str],
    task_id: str,
    log_path: Path,
) -> dict[str, Any]:
    task_path = layout.repo_root / task_id
    plan_path = _resolve_plan_path(layout.repo_root, task_id)
    return {
        "source_row_text": json.dumps(source_row, ensure_ascii=False, indent=2),
        "task_path": task_path,
        "task_text": _read_text_if_exists(task_path),
        "plan_path": plan_path,
        "plan_text": _read_text_if_exists(plan_path) if plan_path is not None else "",
        "log_path": log_path,
        "log_text": _read_text_if_exists(log_path),
    }


def _mirrored_output_complete(layout: AuditLayout) -> bool:
    required = [layout.mirrored_eval_path, layout.mirrored_events_path, layout.mirrored_report_path]
    if any(not path.exists() for path in required):
        return False

    try:
        with layout.mirrored_eval_path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if is_non_correct_row(row) and row.get(VALIDATION_STATUS_FIELD) in {"invalid", "missing_log"}:
                    return False
    except OSError:
        return False
    return True


def _default_tmp_root_for_layout(layout: AuditLayout) -> Path:
    return layout.repo_root / ".tmp_answer_failure_audits"


def _cached_audit_task_ids(
    layout: AuditLayout,
    *,
    tmp_root: Optional[str | Path] = None,
    journal_path: Optional[str | Path] = None,
    task_ids: Optional[list[str]] = None,
) -> set[str]:
    candidates = set(task_ids or [])
    cached: set[str] = set()
    resolved_tmp_root = Path(tmp_root) if tmp_root is not None else _default_tmp_root_for_layout(layout)
    if not resolved_tmp_root.is_absolute():
        resolved_tmp_root = layout.repo_root / resolved_tmp_root
    for task_id in candidates:
        path = _audit_json_path(resolved_tmp_root, layout, task_id)
        if not path.exists():
            continue
        try:
            parsed = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(parsed, dict) and _audit_has_current_schema(parsed):
            cached.add(task_id)

    resolved_journal_path = Path(journal_path) if journal_path is not None else _default_journal_path(layout)
    cached.update(task_id for task_id in _load_journal_audits(resolved_journal_path, layout) if not candidates or task_id in candidates)
    return cached


def eval_pending_stats(
    eval_path: str | Path,
    *,
    tmp_root: Optional[str | Path] = None,
    journal_path: Optional[str | Path] = None,
) -> PendingEvalStats:
    layout = infer_layout(eval_path)
    source_rows, _ = _read_source_rows(layout.eval_path)
    eligible_task_ids = [str(row.get("task_id", "")) for row in source_rows if is_non_correct_row(row)]
    cached_task_ids = _cached_audit_task_ids(
        layout,
        tmp_root=tmp_root,
        journal_path=journal_path,
        task_ids=eligible_task_ids,
    )
    required = [layout.mirrored_eval_path, layout.mirrored_events_path, layout.mirrored_report_path]
    has_all_artifacts = all(path.exists() for path in required)
    if not has_all_artifacts or not layout.mirrored_eval_path.exists():
        pending_task_ids = [task_id for task_id in eligible_task_ids if task_id not in cached_task_ids]
        return PendingEvalStats(
            eval_path=layout.eval_path,
            eligible_rows=len(eligible_task_ids),
            pending_rows=len(pending_task_ids),
            invalid_rows=0,
            has_all_artifacts=has_all_artifacts,
            pending_task_ids=tuple(pending_task_ids),
            invalid_task_ids=(),
        )

    pending_task_ids: list[str] = []
    invalid_task_ids: list[str] = []
    try:
        with layout.mirrored_eval_path.open(newline="") as handle:
            mirrored_by_task = {str(row.get("task_id", "")): row for row in csv.DictReader(handle)}
    except OSError:
        mirrored_by_task = {}
    for task_id in eligible_task_ids:
        mirrored = mirrored_by_task.get(task_id)
        status = str((mirrored or {}).get(VALIDATION_STATUS_FIELD, ""))
        if status in {"invalid", "missing_log"}:
            invalid_task_ids.append(task_id)
            continue
        if task_id in cached_task_ids:
            continue
        if mirrored is None or not status:
            pending_task_ids.append(task_id)

    return PendingEvalStats(
        eval_path=layout.eval_path,
        eligible_rows=len(eligible_task_ids),
        pending_rows=len(pending_task_ids),
        invalid_rows=len(invalid_task_ids),
        has_all_artifacts=has_all_artifacts,
        pending_task_ids=tuple(pending_task_ids),
        invalid_task_ids=tuple(invalid_task_ids),
    )


def find_pending_eval_stats(
    repo_root: str | Path,
    *,
    source_root_name: str = "results_semantic",
    tmp_root: Optional[str | Path] = None,
    journal_path: Optional[str | Path] = None,
) -> list[PendingEvalStats]:
    repo_root = Path(repo_root).resolve()
    source_root = repo_root / source_root_name
    if source_root_name not in SOURCE_ROOTS:
        known = ", ".join(sorted(SOURCE_ROOTS))
        raise ValueError(f"Unknown source root {source_root_name}; expected one of: {known}")
    if not source_root.exists():
        return []
    pending: list[PendingEvalStats] = []
    for eval_path in sorted((source_root / "modes").rglob("eval_results.csv")):
        stats = eval_pending_stats(eval_path, tmp_root=tmp_root, journal_path=journal_path)
        if not stats.has_all_artifacts or stats.pending_rows or stats.invalid_rows:
            pending.append(stats)
    return pending


def find_pending_eval_paths(
    repo_root: str | Path,
    *,
    source_root_name: str = "results_semantic",
    tmp_root: Optional[str | Path] = None,
    journal_path: Optional[str | Path] = None,
) -> list[Path]:
    return [
        stats.eval_path
        for stats in find_pending_eval_stats(
            repo_root,
            source_root_name=source_root_name,
            tmp_root=tmp_root,
            journal_path=journal_path,
        )
    ]


def _load_audits_from_mirrored_outputs(layout: AuditLayout) -> dict[str, dict[str, Any]]:
    if not layout.mirrored_eval_path.exists() or not layout.mirrored_events_path.exists():
        return {}
    with layout.mirrored_eval_path.open(newline="") as handle:
        eval_by_task = {str(row.get("task_id", "")): row for row in csv.DictReader(handle)}
    events_by_task: dict[str, list[dict[str, Any]]] = {}
    with layout.mirrored_events_path.open(newline="") as handle:
        for event in csv.DictReader(handle):
            task_id = str(event.get("task_id", ""))
            events_by_task.setdefault(task_id, []).append(
                {
                    "event_index": event.get("event_index", ""),
                    "answer_failure_type": event.get("answer_failure_type", ""),
                    "answer_failure_subtype": event.get("answer_failure_subtype", ""),
                    "failure_stage": event.get("failure_stage", ""),
                    "failure_summary": event.get("failure_summary", ""),
                    "failure_evidence": event.get("failure_evidence", ""),
                    "confidence": event.get("confidence", ""),
                }
            )
    audits: dict[str, dict[str, Any]] = {}
    for task_id, events in events_by_task.items():
        row = eval_by_task.get(task_id, {})
        audits[task_id] = {
            "answer_failure_audit_schema_version": CURRENT_AUDIT_SCHEMA_VERSION,
            "task_id": task_id,
            "answer_failure_summary": row.get("answer_failure_summary", ""),
            "events": events,
        }
    return audits


def _task_slug(task_id: str) -> str:
    return task_id.replace("/", "__").replace("\\", "__").replace(".json", "")


def _temp_dir_for_layout(tmp_root: Path, layout: AuditLayout) -> Path:
    return tmp_root / layout.source_root.name / f"{layout.model_variant}__{layout.mode_variant}"


def _audit_json_path(tmp_root: Path, layout: AuditLayout, task_id: str) -> Path:
    return _temp_dir_for_layout(tmp_root, layout) / f"{_task_slug(task_id)}.json"


def _trace_json_path(tmp_root: Path, layout: AuditLayout, task_id: str) -> Path:
    return _audit_json_path(tmp_root, layout, task_id).with_suffix(".trace.json")


def _audit_has_current_schema(audit: dict[str, Any]) -> bool:
    return audit.get("answer_failure_audit_schema_version") == CURRENT_AUDIT_SCHEMA_VERSION


def _trace_has_current_schema(trace: dict[str, Any]) -> bool:
    return trace.get("answer_failure_trace_schema_version") == CURRENT_TRACE_SCHEMA_VERSION


def _mark_current_audit_schema(audit: dict[str, Any]) -> dict[str, Any]:
    audit["answer_failure_audit_schema_version"] = CURRENT_AUDIT_SCHEMA_VERSION
    return audit


def _mark_current_trace_schema(trace: dict[str, Any]) -> dict[str, Any]:
    trace["answer_failure_trace_schema_version"] = CURRENT_TRACE_SCHEMA_VERSION
    return trace


def _slug_for_journal(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")


def _compact_slug_for_journal(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "", value)


def _short_mode_variant(mode_variant: str) -> str:
    tokens = mode_variant.split("_")
    kept: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "results":
            index += 2
            continue
        if re.fullmatch(r"k\d+", token) or token in {"skills", "off", "on"}:
            index += 1
            continue
        kept.append(token)
        index += 1
    return "_".join(kept)


def _default_journal_path(layout: AuditLayout) -> Path:
    source_slug = _slug_for_journal(layout.source_root.name)
    model_slug = _compact_slug_for_journal(layout.model_variant)
    mode_slug = _short_mode_variant(layout.mode_variant)
    return Path("/tmp") / f"answer_failure_audits_{source_slug}_{model_slug}_{mode_slug}.jsonl"


def _append_journal_record(
    journal_path: Path,
    *,
    layout: AuditLayout,
    task_id: str,
    status: str,
    audit: Optional[dict[str, Any]] = None,
    error: str = "",
) -> None:
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_variant": layout.model_variant,
        "mode_variant": layout.mode_variant,
        "source_eval": str(layout.eval_path),
        "task_id": task_id,
        "status": status,
    }
    if audit is not None:
        record["audit"] = audit
    if error:
        record["error"] = error
    with journal_path.open("a") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _load_journal_audits(journal_path: Path, layout: AuditLayout) -> dict[str, dict[str, Any]]:
    if not journal_path.exists():
        return {}
    audits: dict[str, dict[str, Any]] = {}
    with journal_path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"warning: ignoring malformed journal line {line_number} in {journal_path}", file=sys.stderr)
                continue
            if record.get("model_variant") != layout.model_variant or record.get("mode_variant") != layout.mode_variant:
                continue
            if record.get("source_eval") != str(layout.eval_path):
                continue
            if record.get("status") != "ok" or not isinstance(record.get("audit"), dict):
                continue
            if not _audit_has_current_schema(record["audit"]):
                continue
            task_id = str(record.get("task_id", ""))
            if task_id:
                audits[task_id] = record["audit"]
    return audits


def _row_summary_from_events(events: list[dict[str, Any]]) -> dict[str, str]:
    types: list[str] = []
    subtypes: list[str] = []
    evidence: list[str] = []
    summaries: list[str] = []
    for event in sorted(events, key=lambda row: int(str(row.get("event_index") or 0))):
        failure_type = " ".join(str(event.get("answer_failure_type", "")).split())
        subtype = " ".join(str(event.get("answer_failure_subtype", "")).split())
        if failure_type and failure_type not in types:
            types.append(failure_type)
        if subtype and subtype not in subtypes:
            subtypes.append(subtype)
        if str(event.get("failure_evidence", "")).strip():
            evidence.append(" ".join(str(event.get("failure_evidence", "")).split()))
        if str(event.get("failure_summary", "")).strip():
            summaries.append(" ".join(str(event.get("failure_summary", "")).split()))
    return {
        "answer_failure_types": "; ".join(types),
        "answer_failure_subtypes": "; ".join(subtypes),
        "answer_failure_event_count": str(len(events)),
        "answer_failure_evidence": "; ".join(evidence[:4]),
        "answer_failure_summary": "; ".join(summaries[:3]),
    }


def _normalized_events(audit: dict[str, Any]) -> list[dict[str, str]]:
    events = audit.get("events") or []
    if not isinstance(events, list):
        raise ValueError(f"Audit for {audit.get('task_id')} has non-list events")
    normalized: list[dict[str, str]] = []
    for index, event in enumerate(events, 1):
        if not isinstance(event, dict):
            raise ValueError(f"Audit for {audit.get('task_id')} has non-object event {index}")
        event_index = event.get("event_index") or index
        failure_type = str(event.get("answer_failure_type", ""))
        normalized_type = ANSWER_FAILURE_TYPE_ALIASES.get(failure_type, failure_type)
        subtype = str(event.get("answer_failure_subtype", ""))
        if normalized_type == "tool_or_data_blocker" and not subtype:
            subtype = "data_source_missing_or_unavailable"
        if failure_type in {"tool_limit", "tool_exhaustion"} and not subtype:
            subtype = "tool_budget_exhausted"
        normalized.append(
            {
                "event_index": str(event_index),
                "answer_failure_type": normalized_type,
                "answer_failure_subtype": subtype,
                "failure_stage": str(event.get("failure_stage", "")),
                "failure_summary": str(event.get("failure_summary", "")),
                "failure_evidence": str(event.get("failure_evidence", "")),
                "confidence": str(event.get("confidence", "")),
            }
        )
    return normalized


def write_mirrored_outputs(
    layout: AuditLayout,
    audits_by_task: dict[str, dict[str, Any]],
    *,
    model_validation_by_task: Optional[dict[str, dict[str, str]]] = None,
) -> dict[str, Any]:
    rows, fieldnames = _read_source_rows(layout.eval_path)
    fieldnames = list(fieldnames)
    for column in ROW_OUTPUT_COLUMNS:
        if column not in fieldnames:
            fieldnames.append(column)

    model_validation_by_task = model_validation_by_task or {}
    event_rows: list[dict[str, str]] = []
    output_rows: list[dict[str, str]] = []

    for row in rows:
        output_row = dict(row)
        task_id = str(row.get("task_id", ""))
        if is_non_correct_row(row):
            audit = audits_by_task.get(task_id)
            if audit is None:
                raise ValueError(f"Missing audit JSON for eligible task_id: {task_id}")
            events = _normalized_events(audit)
            summary = _row_summary_from_events(events)
            if str(audit.get("answer_failure_summary", "")).strip():
                summary["answer_failure_summary"] = str(audit.get("answer_failure_summary", "")).strip()
            output_row.update(summary)
            output_row[VALIDATION_STATUS_FIELD] = ""
            output_row[VALIDATION_NOTES_FIELD] = ""
            validation = model_validation_by_task.get(task_id, {})
            output_row[MODEL_VALIDATION_STATUS_FIELD] = validation.get("status", "")
            output_row[MODEL_VALIDATION_NOTES_FIELD] = validation.get("notes", "")
            for event in events:
                event_rows.append(
                    {
                        "task_id": task_id,
                        "model_variant": layout.model_variant,
                        "mode_variant": layout.mode_variant,
                        **event,
                    }
                )
        else:
            for column in ROW_OUTPUT_COLUMNS:
                output_row[column] = ""
        output_rows.append(output_row)

    layout.mirrored_eval_path.parent.mkdir(parents=True, exist_ok=True)
    with layout.mirrored_eval_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    event_rows.sort(key=lambda row: (row["task_id"], int(row["event_index"] or 0)))
    with layout.mirrored_events_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EVENT_COLUMNS)
        writer.writeheader()
        writer.writerows(event_rows)

    return {
        "eval_path": layout.mirrored_eval_path,
        "events_path": layout.mirrored_events_path,
        "row_count": len(output_rows),
        "event_count": len(event_rows),
    }


def _answer_failure_output_summary(eval_path: Path, events_path: Path) -> dict[str, int]:
    with eval_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    events: list[dict[str, str]] = []
    if events_path.exists():
        with events_path.open(newline="") as handle:
            events = list(csv.DictReader(handle))

    event_counts_by_task: dict[str, int] = defaultdict(int)
    for event in events:
        event_counts_by_task[str(event.get("task_id", ""))] += 1

    eligible_rows = [row for row in rows if is_non_correct_row(row)]
    invalid_rows = [
        row
        for row in eligible_rows
        if str(row.get(VALIDATION_STATUS_FIELD, "")) == "invalid"
    ]
    missing_log_rows = [
        row
        for row in eligible_rows
        if str(row.get(VALIDATION_STATUS_FIELD, "")) == "missing_log"
    ]
    valid_missing_model_validation_rows = [
        row
        for row in eligible_rows
        if str(row.get(VALIDATION_STATUS_FIELD, "")) in {"valid", "valid_with_warnings"}
        and not str(row.get(MODEL_VALIDATION_STATUS_FIELD, "")).strip()
    ]
    trusted_rows = [
        row
        for row in eligible_rows
        if str(row.get(VALIDATION_STATUS_FIELD, "")) in {"valid", "valid_with_warnings"}
        and str(row.get(MODEL_VALIDATION_STATUS_FIELD, "")) in {"pass", "repaired_pass"}
    ]

    return {
        "selected_eval_rows": len(rows),
        "selected_eligible_rows": len(eligible_rows),
        "selected_rows_with_events": sum(
            1
            for row in eligible_rows
            if str(row.get("task_id", "")) in event_counts_by_task
            or str(row.get("answer_failure_event_count", "")).strip()
        ),
        "selected_event_rows": len(events),
        "selected_invalid_rows": len(invalid_rows),
        "selected_missing_log_rows": len(missing_log_rows),
        "selected_valid_missing_model_validation_rows": len(valid_missing_model_validation_rows),
        "selected_trusted_rows": len(trusted_rows),
        "selected_trusted_event_rows": sum(event_counts_by_task.get(str(row.get("task_id", "")), 0) for row in trusted_rows),
    }


def _missing_log_audit(task_id: str) -> dict[str, Any]:
    return {
        "answer_failure_audit_schema_version": CURRENT_AUDIT_SCHEMA_VERSION,
        "task_id": task_id,
        "answer_failure_summary": "The raw log is missing, so no grounded diagnosis is possible.",
        "events": [
            {
                "event_index": 1,
                "answer_failure_type": "ungroundable",
                "answer_failure_subtype": "missing_or_malformed_log",
                "failure_stage": "validation",
                "failure_summary": "The raw log is missing, so no grounded diagnosis is possible.",
                "failure_evidence": "matching raw log missing from logs root",
                "confidence": "high",
            }
        ],
    }


def _build_row_prompt(
    *,
    layout: AuditLayout,
    source_row: dict[str, str],
    task_id: str,
    log_path: Path,
    backend: str,
    repair_notes: str = "",
) -> str:
    raw_log_root = layout.logs_root / layout.model_variant / layout.mode_variant
    artifacts = _load_prompt_artifacts(
        layout=layout,
        source_row=source_row,
        task_id=task_id,
        log_path=log_path,
    )
    base = f"""You are auditing one failed QA run.
Audit exactly this task_id and no other row: {task_id}

Use the CSV row, task JSON, plan JSON, and raw log embedded below as the source of truth.
Source eval: {layout.eval_path}
Task id: {task_id}
Raw log root: {raw_log_root}/ (.json -> .log)

Do not estimate turn waste. Do not choose one dominant error if multiple material failures are visible.

Resolved matching raw log: {log_path}

Return JSON only with keys: task_id, answer_failure_summary, events.
A row can have multiple events. Each event is one material answer-failure cause.
Each event must include event_index, answer_failure_type, answer_failure_subtype, failure_stage, failure_summary, failure_evidence, confidence.
failure_stage must be exactly one of: {format_failure_stages()}.
Evidence must be raw-log anchored and start with Turn N |  when citing a turn.
If the log is missing/unusable, return one ungroundable event with subtype missing_or_malformed_log.

Allowed answer_failure_type values:
{format_answer_failure_type_definitions()}

Use answer_failure_subtype for concrete detail. For tool_or_data_blocker events, use one of these subtypes when applicable:
{format_blocker_subtypes()}

Boundary rules:
{format_boundary_rules()}

{{
  "task_id": "{task_id}",
  "answer_failure_summary": "short row-level summary",
  "events": [
    {{
      "event_index": 1,
      "answer_failure_type": "one allowed type",
      "answer_failure_subtype": "short subtype or empty string",
      "failure_stage": "task_understanding",
      "failure_summary": "short string",
      "failure_evidence": "Turn N | exact or tightly grounded log fragment",
      "confidence": "high | medium | low"
    }}
  ]
}}
"""
    plan_path = artifacts["plan_path"] or "(no matching plan file found)"
    repair_block = ""
    if repair_notes:
        repair_block = (
            "\nPrevious deterministic validation failed for this row:\n"
            + repair_notes
            + "\n\nRepair requirements:\n"
            + "- Use only allowed answer_failure_type taxonomy labels from the allowed list above.\n"
            + "- If citing an exact tool call/result/exception prefix like `Executing:` or `Tool result:`, copy the raw-log text exactly.\n"
            + "- If you cannot copy the exact raw-log text, keep the `Turn N | ` anchor but paraphrase without starting the evidence snippet with `Executing:`, `Tool result:`, `WARNING`, `ERROR`, `Traceback`, or `Exception`.\n"
            + "- Return at least one event for this non-correct row unless the log is missing/unusable.\n"
        )
    return (
        base
        + repair_block
        + "\nAssigned artifacts are embedded below. Use these embedded artifacts as the audit source of truth.\n\n"
        + "CSV row JSON:\n```json\n"
        + artifacts["source_row_text"]
        + "\n```\n\nTask JSON path: "
        + str(artifacts["task_path"])
        + "\nTask JSON:\n```json\n"
        + artifacts["task_text"]
        + "\n```\n\nPlan JSON path: "
        + str(plan_path)
        + "\nPlan JSON:\n```json\n"
        + artifacts["plan_text"]
        + "\n```\n\nRaw log path: "
        + str(artifacts["log_path"])
        + "\nRaw log:\n```text\n"
        + artifacts["log_text"]
        + "\n```"
    )


def _build_trace_extractor_prompt(
    *,
    layout: AuditLayout,
    source_row: dict[str, str],
    task_id: str,
    log_path: Path,
    repair_notes: str = "",
) -> str:
    raw_log_root = layout.logs_root / layout.model_variant / layout.mode_variant
    artifacts = _load_prompt_artifacts(
        layout=layout,
        source_row=source_row,
        task_id=task_id,
        log_path=log_path,
    )
    plan_path = artifacts["plan_path"] or "(no matching plan file found)"
    repair_block = ""
    if repair_notes:
        repair_block = (
            "\nPrevious validation notes for the final audit:\n"
            + repair_notes
            + "\nUse these notes only to look for better raw-log anchors. Do not produce taxonomy labels.\n"
        )
    return f"""Stage A: extract a compact evidence trace for one failed QA run.
Audit exactly this task_id and no other row: {task_id}

Use the CSV row, task JSON, plan JSON, and raw log embedded below as the source of truth.
Source eval: {layout.eval_path}
Task id: {task_id}
Raw log root: {raw_log_root}/ (.json -> .log)
Resolved matching raw log: {log_path}

Do not assign answer_failure_type labels or any taxonomy labels.
Do not decide one dominant failure if multiple material failure moments are visible.
Do not estimate turn waste.

Return JSON only with keys: task_id, trace_summary, final_answer, evidence_items, material_observations, open_questions.
The trace should be compact but complete enough for a separate labeler to classify failures without reading the full raw log.
Each evidence item must include evidence_id, turn, raw_log_excerpt, role, why_relevant.
Each raw_log_excerpt must start with Turn N |  when citing a turn.
Each evidence item role must be exactly one of: source_choice, extraction, computation, final_answer, blocker, plan_mismatch, other.
Each material observation must include observation_id, evidence_ids, observation, materiality, uncertainty.
Each material observation uncertainty must be exactly one of: low, medium, high.
These are factual observations, not taxonomy labels.

{{
  "task_id": "{task_id}",
  "trace_summary": "short evidence-only summary of what went wrong",
  "final_answer": {{
    "turn": 1,
    "raw_log_excerpt": "Turn N | final answer or submission excerpt",
    "answer_text": "submitted answer or empty string"
  }},
  "evidence_items": [
    {{
      "evidence_id": "E1",
      "turn": 1,
      "raw_log_excerpt": "Turn N | exact or tightly grounded log fragment",
      "role": "source_choice",
      "why_relevant": "why this line matters for diagnosing the failed answer"
    }}
  ],
  "material_observations": [
    {{
      "observation_id": "O1",
      "evidence_ids": ["E1"],
      "observation": "plain-language factual observation grounded in evidence",
      "materiality": "why this likely affected the final answer",
      "uncertainty": "low"
    }}
  ],
  "open_questions": []
}}

Example shape for Stage A reasoning, using illustrative values only:
```json
{{
  "evidence_items": [
    {{
      "evidence_id": "E1",
      "turn": 12,
      "raw_log_excerpt": "Turn 12 | Executing: query_ideal(... year=2022 ...)",
      "role": "source_choice",
      "why_relevant": "The task asks for 2023, but this query uses 2022."
    }}
  ],
  "material_observations": [
    {{
      "observation_id": "O1",
      "evidence_ids": ["E1"],
      "observation": "The run used 2022 data for a 2023 question.",
      "materiality": "The final answer was computed from that wrong-year query.",
      "uncertainty": "low"
    }}
  ]
}}
```
{repair_block}
Assigned artifacts are embedded below. Use these embedded artifacts as the trace source of truth.

CSV row JSON:
```json
{artifacts["source_row_text"]}
```

Task JSON path: {artifacts["task_path"]}
Task JSON:
```json
{artifacts["task_text"]}
```

Plan JSON path: {plan_path}
Plan JSON:
```json
{artifacts["plan_text"]}
```

Raw log path: {artifacts["log_path"]}
Raw log:
```text
{artifacts["log_text"]}
```"""


def _build_trace_labeler_prompt(
    *,
    layout: AuditLayout,
    source_row: dict[str, str],
    task_id: str,
    trace: dict[str, Any],
    log_path: Path,
    repair_notes: str = "",
) -> str:
    repair_block = ""
    if repair_notes:
        repair_block = (
            "\nPrevious deterministic validation failed for this row:\n"
            + repair_notes
            + "\n\nRepair requirements:\n"
            + "- Use only allowed answer_failure_type taxonomy labels from the allowed list below.\n"
            + "- Prefer omitting weak events over inventing stronger narratives.\n"
            + "- Evidence must come from Trace JSON evidence_ids.\n"
        )
    return f"""Stage B: label answer-failure events from a compact trace.
Audit exactly this task_id and no other row: {task_id}

Your primary job is classification, not summarization.
Do not read or request the full raw log. Use only the Trace JSON and the taxonomy below.
Do not choose answer_failure_figure_group; it is derived later from answer_failure_type.
Source eval: {layout.eval_path}
Task id: {task_id}
Resolved log path for trace provenance: {log_path}

Return JSON only with keys: task_id, events.
Each event must include event_index, observation_id, answer_failure_type, answer_failure_subtype, evidence_ids.
Use observation_id values only from Trace JSON material_observations.
Use evidence_ids only from Trace JSON evidence_items. The runner will copy evidence text from those ids.
The runner will fill validation-only boilerplate later.
If no material observation should become an event, return an empty events list.

Allowed answer_failure_type values:
{format_answer_failure_type_definitions()}

Use answer_failure_subtype for concrete detail. For tool_or_data_blocker events, use one of these subtypes when applicable:
{format_blocker_subtypes()}

Boundary rules:
{format_boundary_rules()}

{{
  "task_id": "{task_id}",
  "events": [
    {{
      "event_index": 1,
      "observation_id": "O1",
      "answer_failure_type": "one allowed type",
      "answer_failure_subtype": "short subtype or empty string",
      "evidence_ids": ["E1"]
    }}
  ]
}}
{repair_block}
Trace JSON:
```json
{json.dumps(trace, ensure_ascii=False, indent=2)}
```"""


def _call_codex(
    prompt: str,
    *,
    cwd: Path,
    model: str,
    reasoning_effort: str,
    last_message_path: Path,
    stdout_path: Path,
    timeout: int,
) -> str:
    last_message_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "codex",
        "exec",
        "-m",
        model,
        "-c",
        f'model_reasoning_effort="{reasoning_effort}"',
        "-C",
        str(cwd),
        "-s",
        "workspace-write",
        "--output-last-message",
        str(last_message_path),
        "-",
    ]
    proc = subprocess.run(
        cmd,
        input=prompt,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    stdout_path.write_text(proc.stdout)
    if proc.returncode != 0:
        raise RuntimeError(f"codex exec failed with exit code {proc.returncode}; see {stdout_path}")
    if last_message_path.exists():
        return last_message_path.read_text()
    return proc.stdout


def _extract_response_text(payload: dict[str, Any]) -> str:
    if isinstance(payload.get("output_text"), str):
        return payload["output_text"]
    pieces: list[str] = []
    for item in payload.get("output", []) or []:
        for content in item.get("content", []) or []:
            text = content.get("text")
            if isinstance(text, str):
                pieces.append(text)
    return "\n".join(pieces)


def _openai_api_key(repo_root: Path) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        return api_key
    env_path = repo_root / ".env"
    if env_path.exists():
        with env_path.open() as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped or stripped.startswith("#") or "=" not in stripped:
                    continue
                key, value = stripped.split("=", 1)
                if key == "OPENAI_API_KEY":
                    return value.strip().strip('"').strip("'")
    return ""


def _call_openai(prompt: str, *, cwd: Path, model: str, reasoning_effort: str, timeout: int) -> str:
    api_key = _openai_api_key(cwd)
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for --backend openai")
    body = {
        "model": model,
        "input": prompt,
        "reasoning": {"effort": reasoning_effort},
    }
    request = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    text = _extract_response_text(payload)
    if not text:
        raise RuntimeError("OpenAI response did not contain output text")
    return text


def _call_model(
    prompt: str,
    *,
    backend: str,
    cwd: Path,
    model: str,
    reasoning_effort: str,
    last_message_path: Path,
    stdout_path: Path,
    timeout: int,
) -> str:
    if backend == "codex":
        return _call_codex(
            prompt,
            cwd=cwd,
            model=model,
            reasoning_effort=reasoning_effort,
            last_message_path=last_message_path,
            stdout_path=stdout_path,
            timeout=timeout,
        )
    if backend == "openai":
        text = _call_openai(prompt, cwd=cwd, model=model, reasoning_effort=reasoning_effort, timeout=timeout)
        last_message_path.parent.mkdir(parents=True, exist_ok=True)
        last_message_path.write_text(text)
        stdout_path.write_text("")
        return text
    raise ValueError(f"Unsupported backend: {backend}")


def _missing_log_trace(task_id: str, log_path: Path) -> dict[str, Any]:
    return {
        "answer_failure_trace_schema_version": CURRENT_TRACE_SCHEMA_VERSION,
        "task_id": task_id,
        "trace_summary": "The matching raw log is missing, so no grounded failure trace can be extracted.",
        "log_status": "missing_or_malformed_log",
        "log_path": str(log_path),
        "final_answer": None,
        "evidence_items": [],
        "material_observations": [
            {
                "observation_id": "O1",
                "evidence_ids": [],
                "observation": "The raw log is missing or unusable.",
                "materiality": "Without the raw log, the answer failure cannot be grounded in the run.",
                "uncertainty": "low",
            }
        ],
        "open_questions": [],
    }


def _clean_trace_value(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [cleaned for item in value if (cleaned := _clean_trace_value(item))]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _trace_items_by_id(trace: dict[str, Any], key: str, id_field: str) -> dict[str, dict[str, Any]]:
    items: dict[str, dict[str, Any]] = {}
    for item in trace.get(key) or []:
        if not isinstance(item, dict):
            continue
        item_id = _clean_trace_value(item.get(id_field))
        if item_id:
            items[item_id] = item
    return items


def _failure_stage_from_evidence(evidence_items: list[dict[str, Any]]) -> str:
    for item in evidence_items:
        role = _clean_trace_value(item.get("role"))
        if role in TRACE_ROLE_TO_FAILURE_STAGE:
            return TRACE_ROLE_TO_FAILURE_STAGE[role]
    return "evidence_gathering"


def _confidence_from_observation(observation: dict[str, Any]) -> str:
    uncertainty = _clean_trace_value(observation.get("uncertainty")).lower()
    return TRACE_UNCERTAINTY_TO_CONFIDENCE.get(uncertainty, "medium")


def _evidence_text_from_items(evidence_items: list[dict[str, Any]]) -> str:
    excerpts: list[str] = []
    for item in evidence_items:
        excerpt = " ".join(str(item.get("raw_log_excerpt", "")).split())
        if excerpt:
            excerpts.append(excerpt)
    return "; ".join(excerpts)


def _expand_trace_labeler_output(
    *,
    task_id: str,
    trace: dict[str, Any],
    label_payload: dict[str, Any],
) -> dict[str, Any]:
    evidence_by_id = _trace_items_by_id(trace, "evidence_items", "evidence_id")
    observation_by_id = _trace_items_by_id(trace, "material_observations", "observation_id")
    expanded_events: list[dict[str, Any]] = []

    for index, label_event in enumerate(label_payload.get("events") or [], 1):
        if not isinstance(label_event, dict):
            continue
        observation_id = _clean_trace_value(label_event.get("observation_id"))
        observation = observation_by_id.get(observation_id, {})
        evidence_ids = _string_list(label_event.get("evidence_ids")) or _string_list(observation.get("evidence_ids"))
        evidence_items = [evidence_by_id[evidence_id] for evidence_id in evidence_ids if evidence_id in evidence_by_id]
        failure_summary = (
            _clean_trace_value(observation.get("observation"))
            or _clean_trace_value(label_event.get("failure_summary"))
            or _clean_trace_value(label_event.get("answer_failure_type"))
            or "Unspecified answer failure."
        )
        expanded_events.append(
            {
                "event_index": label_event.get("event_index") or index,
                "answer_failure_type": _clean_trace_value(label_event.get("answer_failure_type")),
                "answer_failure_subtype": _clean_trace_value(label_event.get("answer_failure_subtype")),
                "failure_stage": _failure_stage_from_evidence(evidence_items),
                "failure_summary": failure_summary,
                "failure_evidence": _evidence_text_from_items(evidence_items),
                "confidence": _confidence_from_observation(observation),
            }
        )

    summaries = [
        str(event.get("failure_summary", "")).strip()
        for event in expanded_events
        if str(event.get("failure_summary", "")).strip()
    ]
    return {
        "answer_failure_audit_schema_version": CURRENT_AUDIT_SCHEMA_VERSION,
        "task_id": task_id,
        "answer_failure_summary": "; ".join(summaries[:3]) or str(trace.get("trace_summary", "")).strip(),
        "events": expanded_events,
    }


def run_two_stage_row_audit(
    *,
    layout: AuditLayout,
    source_row: dict[str, str],
    tmp_root: Path,
    backend: str,
    model: str,
    reasoning_effort: str,
    timeout: int,
    force: bool,
    repair_notes: str = "",
) -> tuple[str, dict[str, Any]]:
    task_id = str(source_row.get("task_id", ""))
    out_path = _audit_json_path(tmp_root, layout, task_id)
    trace_path = _trace_json_path(tmp_root, layout, task_id)
    if out_path.exists() and not force:
        cached_audit = json.loads(out_path.read_text())
        if isinstance(cached_audit, dict) and _audit_has_current_schema(cached_audit):
            print(f"row audit cache hit: task={task_id}", flush=True)
            return task_id, cached_audit
        print(f"row audit cache stale: task={task_id}; rerunning", flush=True)

    log_path = row_log_path(layout.logs_root, layout.model_variant, layout.mode_variant, task_id)
    if not log_path.exists():
        print(f"row audit missing log: task={task_id} log={log_path}", flush=True)
        trace = _missing_log_trace(task_id, log_path)
        audit = _missing_log_audit(task_id)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text(json.dumps(trace, indent=2, ensure_ascii=False) + "\n")
        out_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")
        return task_id, audit

    print(
        f"two-stage row audit start: task={task_id} model={layout.model_variant} "
        f"mode={layout.mode_variant} backend={backend}",
        flush=True,
    )
    stem = out_path.with_suffix("")
    if trace_path.exists() and not force:
        cached_trace = json.loads(trace_path.read_text())
        if isinstance(cached_trace, dict) and _trace_has_current_schema(cached_trace):
            print(f"stage A trace cache hit: task={task_id}", flush=True)
            trace = cached_trace
        else:
            print(f"stage A trace cache stale: task={task_id}; rerunning", flush=True)
            trace = {}
    else:
        trace = {}

    if not trace:
        trace_prompt = _build_trace_extractor_prompt(
            layout=layout,
            source_row=source_row,
            task_id=task_id,
            log_path=log_path,
            repair_notes=repair_notes,
        )
        trace_text = _call_model(
            trace_prompt,
            backend=backend,
            cwd=layout.repo_root,
            model=model,
            reasoning_effort=reasoning_effort,
            last_message_path=stem.with_suffix(".trace.last_message.txt"),
            stdout_path=stem.with_suffix(f".trace.{backend}_stdout.log"),
            timeout=timeout,
        )
        trace = parse_json_object(trace_text)
        trace["task_id"] = task_id
        _mark_current_trace_schema(trace)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text(json.dumps(trace, indent=2, ensure_ascii=False) + "\n")

    labeler_prompt = _build_trace_labeler_prompt(
        layout=layout,
        source_row=source_row,
        task_id=task_id,
        trace=trace,
        log_path=log_path,
        repair_notes=repair_notes,
    )
    audit_text = _call_model(
        labeler_prompt,
        backend=backend,
        cwd=layout.repo_root,
        model=model,
        reasoning_effort=reasoning_effort,
        last_message_path=stem.with_suffix(".labeler.last_message.txt"),
        stdout_path=stem.with_suffix(f".labeler.{backend}_stdout.log"),
        timeout=timeout,
    )
    label_payload = parse_json_object(audit_text)
    label_payload["task_id"] = task_id
    audit = _expand_trace_labeler_output(task_id=task_id, trace=trace, label_payload=label_payload)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")
    return task_id, audit


def run_row_audit(
    *,
    layout: AuditLayout,
    source_row: dict[str, str],
    tmp_root: Path,
    backend: str,
    model: str,
    reasoning_effort: str,
    timeout: int,
    force: bool,
    repair_notes: str = "",
    pipeline: str = DEFAULT_ROW_AUDIT_PIPELINE,
) -> tuple[str, dict[str, Any]]:
    if pipeline == "two-stage":
        return run_two_stage_row_audit(
            layout=layout,
            source_row=source_row,
            tmp_root=tmp_root,
            backend=backend,
            model=model,
            reasoning_effort=reasoning_effort,
            timeout=timeout,
            force=force,
            repair_notes=repair_notes,
        )
    if pipeline != "single":
        raise ValueError(f"Unsupported row audit pipeline: {pipeline}")

    task_id = str(source_row.get("task_id", ""))
    out_path = _audit_json_path(tmp_root, layout, task_id)
    if out_path.exists() and not force:
        cached_audit = json.loads(out_path.read_text())
        if isinstance(cached_audit, dict) and _audit_has_current_schema(cached_audit):
            print(f"row audit cache hit: task={task_id}", flush=True)
            return task_id, cached_audit
        print(f"row audit cache stale: task={task_id}; rerunning", flush=True)

    log_path = row_log_path(layout.logs_root, layout.model_variant, layout.mode_variant, task_id)
    if not log_path.exists():
        print(f"row audit missing log: task={task_id} log={log_path}", flush=True)
        audit = _missing_log_audit(task_id)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(audit, indent=2) + "\n")
        return task_id, audit

    print(
        f"row audit start: task={task_id} model={layout.model_variant} "
        f"mode={layout.mode_variant} backend={backend}",
        flush=True,
    )
    if backend == "codex":
        print(f"  spawning codex row-audit subagent for task={task_id}", flush=True)
    else:
        print(f"  calling {backend} row-audit judge for task={task_id}", flush=True)
    prompt = _build_row_prompt(
        layout=layout,
        source_row=source_row,
        task_id=task_id,
        log_path=log_path,
        backend=backend,
        repair_notes=repair_notes,
    )
    stem = out_path.with_suffix("")
    text = _call_model(
        prompt,
        backend=backend,
        cwd=layout.repo_root,
        model=model,
        reasoning_effort=reasoning_effort,
        last_message_path=stem.with_suffix(".last_message.txt"),
        stdout_path=stem.with_suffix(f".{backend}_stdout.log"),
        timeout=timeout,
    )
    audit = parse_json_object(text)
    audit["task_id"] = task_id
    _mark_current_audit_schema(audit)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")
    return task_id, audit


def _load_audits_for_rows(layout: AuditLayout, tmp_root: Path, rows: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    audits: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not is_non_correct_row(row):
            continue
        task_id = str(row.get("task_id", ""))
        path = _audit_json_path(tmp_root, layout, task_id)
        if path.exists():
            audit = json.loads(path.read_text())
            if isinstance(audit, dict) and _audit_has_current_schema(audit):
                audits[task_id] = audit
    return audits


def _deterministic_status_by_task(layout: AuditLayout) -> dict[str, dict[str, str]]:
    if not layout.mirrored_eval_path.exists():
        return {}
    with layout.mirrored_eval_path.open(newline="") as handle:
        return {
            str(row.get("task_id", "")): {
                "status": str(row.get(VALIDATION_STATUS_FIELD, "")),
                "notes": str(row.get(VALIDATION_NOTES_FIELD, "")),
            }
            for row in csv.DictReader(handle)
        }


def _model_validation_status_by_task(layout: AuditLayout) -> dict[str, dict[str, str]]:
    if not layout.mirrored_eval_path.exists():
        return {}
    with layout.mirrored_eval_path.open(newline="") as handle:
        return {
            str(row.get("task_id", "")): {
                "status": str(row.get(MODEL_VALIDATION_STATUS_FIELD, "")),
                "notes": str(row.get(MODEL_VALIDATION_NOTES_FIELD, "")),
            }
            for row in csv.DictReader(handle)
        }


def _model_validator_stale_task_ids(
    *,
    audits_by_task: dict[str, dict[str, Any]],
    deterministic_by_task: dict[str, dict[str, str]],
    existing_model_validation_by_task: dict[str, dict[str, str]],
    only_task_ids: set[str] | None = None,
) -> list[str]:
    stale_statuses = {"", "invalid_untrusted"}
    task_ids: list[str] = []
    for task_id in sorted(audits_by_task):
        if only_task_ids is not None and task_id not in only_task_ids:
            continue
        deterministic_status = deterministic_by_task.get(task_id, {}).get("status")
        if deterministic_status not in {"valid", "valid_with_warnings"}:
            continue
        model_status = existing_model_validation_by_task.get(task_id, {}).get("status", "")
        if model_status in stale_statuses:
            task_ids.append(task_id)
    return task_ids


def _task_ids_for_deterministic_repair(validation_outputs: dict[str, Any]) -> list[str]:
    task_ids: list[str] = []
    for row in validation_outputs.get("invalid_rows", []):
        if row.get(VALIDATION_STATUS_FIELD) != "invalid":
            continue
        task_id = str(row.get("task_id", ""))
        if task_id and task_id not in task_ids:
            task_ids.append(task_id)
    return task_ids


def _build_row_model_validator_prompt(
    *,
    layout: AuditLayout,
    source_row: dict[str, str],
    task_id: str,
    audit: dict[str, Any],
    deterministic_status: dict[str, str],
    log_path: Path,
) -> str:
    artifacts = _load_prompt_artifacts(
        layout=layout,
        source_row=source_row,
        task_id=task_id,
        log_path=log_path,
    )
    plan_path = artifacts["plan_path"] or "(no matching plan file found)"
    return f"""You are validating one answer-failure row audit.
Do not validate any other task_id.

Task id: {task_id}
Source eval: {layout.eval_path}
Mirrored eval: {layout.mirrored_eval_path}
Mirrored events: {layout.mirrored_events_path}
Logs root: {layout.logs_root}
Resolved matching raw log: {log_path}

Return pass if this one row audit is grounded enough to keep, needs_repair if it should be replaced by a smaller grounded version, or invalid_untrusted if it should not be summarized.
Prefer deleting weak events over inventing stronger narratives.
Return JSON only:
{{
  "task_id": "{task_id}",
  "verdict": "pass | needs_repair | invalid_untrusted",
  "problems": ["short string"],
  "supported_facts": ["short string"],
  "repaired_row": {{
    "task_id": "{task_id}",
    "answer_failure_summary": "short string",
    "events": []
  }}
}}

Rules:
- Validate only whether the supplied row audit is grounded in the embedded task, plan, CSV row, and raw log.
- Use needs_repair only when a non-empty repaired_row can replace the original audit.
- Use invalid_untrusted when the audit is not grounded enough and cannot be safely repaired.
- A repaired_row must keep the same task_id and include at least one event.

Deterministic validation:
```json
{json.dumps(deterministic_status, ensure_ascii=False, indent=2)}
```

Row audit JSON:
```json
{json.dumps(audit, ensure_ascii=False, indent=2)}
```

CSV row JSON:
```json
{artifacts["source_row_text"]}
```

Task JSON path: {artifacts["task_path"]}
Task JSON:
```json
{artifacts["task_text"]}
```

Plan JSON path: {plan_path}
Plan JSON:
```json
{artifacts["plan_text"]}
```

Raw log path: {artifacts["log_path"]}
Raw log:
```text
{artifacts["log_text"]}
```"""


def _validated_batch_repair(original: dict[str, Any], repaired: dict[str, Any]) -> tuple[dict[str, Any], str]:
    events = repaired.get("events")
    if not isinstance(events, list) or not events:
        return original, "empty repaired_row rejected; kept original row audit"
    return repaired, ""


def _apply_model_validator_item(
    *,
    task_id: str,
    item: dict[str, Any],
    audits_by_task: dict[str, dict[str, Any]],
    repaired_audits: dict[str, dict[str, Any]],
    allow_downgrade: bool,
) -> dict[str, str]:
    verdict = str(item.get("verdict", "invalid_untrusted"))
    problems = item.get("problems") or []
    notes = "; ".join(str(problem) for problem in problems if str(problem).strip())
    if verdict == "pass":
        return {"status": "pass", "notes": notes}
    if verdict == "needs_repair" and isinstance(item.get("repaired_row"), dict):
        repaired = item["repaired_row"]
        repaired["task_id"] = task_id
        accepted, repair_note = _validated_batch_repair(audits_by_task.get(task_id, {}), repaired)
        repaired_audits[task_id] = accepted
        if repair_note:
            return {
                "status": "invalid_untrusted" if allow_downgrade else "pass",
                "notes": "; ".join(part for part in [notes, repair_note] if part),
            }
        return {"status": "repaired_pass", "notes": notes or "row model validator repaired row"}
    return {
        "status": "invalid_untrusted" if allow_downgrade else "pass",
        "notes": notes,
    }


def run_model_validators(
    *,
    layout: AuditLayout,
    tmp_root: Path,
    audits_by_task: dict[str, dict[str, Any]],
    deterministic_by_task: dict[str, dict[str, str]],
    backend: str,
    model: str,
    reasoning_effort: str,
    timeout: int,
    allow_downgrade: bool,
    concurrency: int = 2,
    task_ids_to_validate: set[str] | None = None,
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, Any]]]:
    print(
        f"row model validators start: model={layout.model_variant} mode={layout.mode_variant} "
        f"backend={backend} rows={len(audits_by_task)}",
        flush=True,
    )
    source_rows, _ = _read_source_rows(layout.eval_path)
    source_by_task = {str(row.get("task_id", "")): row for row in source_rows}
    task_ids = [
        task_id
        for task_id in sorted(audits_by_task)
        if deterministic_by_task.get(task_id, {}).get("status") in {"valid", "valid_with_warnings"}
        and (task_ids_to_validate is None or task_id in task_ids_to_validate)
    ]
    validation_by_task: dict[str, dict[str, str]] = {}
    repaired_audits = dict(audits_by_task)

    def validate_one(task_id: str) -> tuple[str, dict[str, str], dict[str, Any] | None]:
        log_path = row_log_path(layout.logs_root, layout.model_variant, layout.mode_variant, task_id)
        prompt = _build_row_model_validator_prompt(
            layout=layout,
            source_row=source_by_task.get(task_id, {"task_id": task_id}),
            task_id=task_id,
            audit=audits_by_task[task_id],
            deterministic_status=deterministic_by_task.get(task_id, {}),
            log_path=log_path,
        )
        stem = _audit_json_path(tmp_root, layout, task_id).with_suffix("")
        text = _call_model(
            prompt,
            backend=backend,
            cwd=layout.repo_root,
            model=model,
            reasoning_effort=reasoning_effort,
            last_message_path=stem.with_suffix(".validator.last_message.txt"),
            stdout_path=stem.with_suffix(f".validator.{backend}_stdout.log"),
            timeout=timeout,
        )
        payload = parse_json_object(text)
        if isinstance(payload.get("rows"), list):
            rows = payload["rows"]
            if len(rows) != 1 or not isinstance(rows[0], dict):
                raise ValueError(f"Row model validator for {task_id} returned invalid rows payload")
            payload = rows[0]
        payload["task_id"] = task_id
        repaired_for_task = dict(repaired_audits)
        validation = _apply_model_validator_item(
            task_id=task_id,
            item=payload,
            audits_by_task=audits_by_task,
            repaired_audits=repaired_for_task,
            allow_downgrade=allow_downgrade,
        )
        return task_id, validation, repaired_for_task.get(task_id)

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(validate_one, task_id): task_id for task_id in task_ids}
        for future in as_completed(futures):
            task_id = futures[future]
            try:
                task_id, validation, repaired = future.result()
            except Exception as exc:
                validation = {
                    "status": "invalid_untrusted",
                    "notes": f"row model validator failed: {type(exc).__name__}: {exc}",
                }
                repaired = None
            validation_by_task[task_id] = validation
            if repaired is not None:
                repaired_audits[task_id] = repaired
            print(f"row model validator complete: {task_id} status={validation.get('status', '')}")

    for task_id, status in deterministic_by_task.items():
        if status.get("status") in {"invalid", "missing_log"}:
            validation_by_task[task_id] = {
                "status": "invalid_untrusted",
                "notes": status.get("notes", ""),
            }
        elif (
            task_ids_to_validate is None
            and status.get("status") in {"valid", "valid_with_warnings"}
            and task_id not in validation_by_task
        ):
            validation_by_task[task_id] = {
                "status": "invalid_untrusted" if allow_downgrade else "pass",
                "notes": "row model validator did not return a verdict for this row",
            }

    return validation_by_task, repaired_audits


def run_batch_auditor(
    *,
    layout: AuditLayout,
    tmp_root: Path,
    audits_by_task: dict[str, dict[str, Any]],
    deterministic_by_task: dict[str, dict[str, str]],
    backend: str,
    model: str,
    reasoning_effort: str,
    timeout: int,
    allow_downgrade: bool,
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, Any]]]:
    return run_model_validators(
        layout=layout,
        tmp_root=tmp_root,
        audits_by_task=audits_by_task,
        deterministic_by_task=deterministic_by_task,
        backend=backend,
        model=model,
        reasoning_effort=reasoning_effort,
        timeout=timeout,
        allow_downgrade=allow_downgrade,
    )


def _run_row_jobs(
    *,
    rows_to_run: list[dict[str, str]],
    layout: AuditLayout,
    tmp_root: Path,
    journal_path: Path,
    backend: str,
    model: str,
    reasoning_effort: str,
    timeout: int,
    concurrency: int,
    force: bool,
    pipeline: str,
    repair_notes_by_task: Optional[dict[str, str]] = None,
) -> dict[str, dict[str, Any]]:
    repair_notes_by_task = repair_notes_by_task or {}
    audits_by_task: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(
                run_row_audit,
                layout=layout,
                source_row=row,
                tmp_root=tmp_root,
                backend=backend,
                model=model,
                reasoning_effort=reasoning_effort,
                timeout=timeout,
                force=force,
                repair_notes=repair_notes_by_task.get(str(row.get("task_id", "")), ""),
                pipeline=pipeline,
            ): str(row.get("task_id", ""))
            for row in rows_to_run
        }
        for future in as_completed(futures):
            task_id = futures[future]
            try:
                task_id, audit = future.result()
            except Exception as exc:
                _append_journal_record(journal_path, layout=layout, task_id=task_id, status="error", error=str(exc))
                raise
            audits_by_task[task_id] = audit
            _append_journal_record(journal_path, layout=layout, task_id=task_id, status="ok", audit=audit)
            print(f"row audit complete: {task_id}")
    return audits_by_task


def run_audit_file(args: argparse.Namespace) -> dict[str, Any]:
    eval_path = args.eval_path
    selected_stats: PendingEvalStats | None = None
    tmp_root = Path(args.tmp_root).resolve()
    explicit_journal_path = Path(args.journal_path).resolve() if args.journal_path else None
    if eval_path is None:
        eval_path = choose_pending_eval_path(
            repo_root=Path(args.repo_root).resolve(),
            source_root_name=args.source_root,
            tmp_root=tmp_root,
            journal_path=explicit_journal_path,
        )
        selected_stats = eval_pending_stats(eval_path, tmp_root=tmp_root, journal_path=explicit_journal_path)
    layout = infer_layout(eval_path)
    if selected_stats is None:
        selected_stats = eval_pending_stats(layout.eval_path, tmp_root=tmp_root, journal_path=explicit_journal_path)
    journal_path = explicit_journal_path or _default_journal_path(layout)
    rows, _ = _read_source_rows(layout.eval_path)
    eligible_rows = [row for row in rows if is_non_correct_row(row)]
    limited_smoke = args.limit is not None
    if args.limit is not None:
        eligible_rows = eligible_rows[: args.limit]

    print(f"Source eval: {layout.eval_path}")
    print(f"Eligible failed task rows in selected file: {len(eligible_rows)}")
    print(f"Temp JSON dir: {_temp_dir_for_layout(tmp_root, layout)}")
    print(f"JSONL journal: {journal_path}")

    audits_by_task: dict[str, dict[str, Any]] = _load_journal_audits(journal_path, layout)
    audits_by_task.update(_load_audits_for_rows(layout, tmp_root, eligible_rows))
    audits_by_task.update(_load_audits_from_mirrored_outputs(layout))
    for task_id, audit in sorted(audits_by_task.items()):
        out_path = _audit_json_path(tmp_root, layout, task_id)
        if not out_path.exists():
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")
    if audits_by_task:
        print(f"Resumed {len(audits_by_task)} completed row audits from cache")

    selected_pending_ids = set(selected_stats.pending_task_ids)
    rows_to_run = [
        row
        for row in eligible_rows
        if args.force
        or str(row.get("task_id", "")) not in audits_by_task
        or str(row.get("task_id", "")) in selected_pending_ids
    ]
    if args.force:
        audits_by_task = {}

    audits_by_task.update(
        _run_row_jobs(
            rows_to_run=rows_to_run,
            layout=layout,
            tmp_root=tmp_root,
            journal_path=journal_path,
            backend=args.backend,
            model=args.row_model,
            reasoning_effort=args.row_reasoning_effort,
            timeout=args.timeout,
            concurrency=args.concurrency,
            force=args.force,
            pipeline=args.row_audit_pipeline,
        )
    )

    if args.limit is None:
        audits_by_task.update(_load_audits_for_rows(layout, tmp_root, rows))

    if limited_smoke:
        return {
            "layout": layout,
            "eligible_rows": len(eligible_rows),
            "initial_invalid_rows": 0,
            "final_invalid_rows": 0,
            "event_count": sum(len(audit.get("events") or []) for audit in audits_by_task.values()),
            "eval_path": layout.mirrored_eval_path,
            "events_path": layout.mirrored_events_path,
            "report_path": layout.mirrored_report_path,
            "smoke_only": True,
            "journal_path": journal_path,
        }

    write_mirrored_outputs(layout, audits_by_task)
    validation_outputs = validate_answer_failure_root(
        source_root=layout.output_root,
        logs_dir=layout.logs_root,
        rewrite=True,
    )
    selected_initial_summary = _answer_failure_output_summary(layout.mirrored_eval_path, layout.mirrored_events_path)
    row_by_task = {str(row.get("task_id", "")): row for row in eligible_rows}
    repaired_task_ids: set[str] = set()
    if selected_stats.invalid_rows:
        print(f"Selected file has invalid_rows={selected_stats.invalid_rows}; repair mode will target those rows.")
    for repair_round in range(1, args.repair_invalid_rounds + 1):
        repair_ids = _task_ids_for_deterministic_repair(validation_outputs)
        if not repair_ids:
            break
        repair_rows = [row_by_task[task_id] for task_id in repair_ids if task_id in row_by_task]
        repair_notes_by_task = {
            str(row.get("task_id", "")): str(row.get(VALIDATION_NOTES_FIELD, ""))
            for row in validation_outputs.get("invalid_rows", [])
        }
        print(f"Repair round {repair_round}: rerunning {len(repair_rows)} deterministic-invalid rows")
        repaired = _run_row_jobs(
            rows_to_run=repair_rows,
            layout=layout,
            tmp_root=tmp_root,
            journal_path=journal_path,
            backend=args.backend,
            model=args.row_model,
            reasoning_effort=args.row_reasoning_effort,
            timeout=args.timeout,
            concurrency=args.concurrency,
            force=True,
            pipeline=args.row_audit_pipeline,
            repair_notes_by_task=repair_notes_by_task,
        )
        audits_by_task.update(repaired)
        repaired_task_ids.update(repaired)
        write_mirrored_outputs(layout, audits_by_task)
        validation_outputs = validate_answer_failure_root(
            source_root=layout.output_root,
            logs_dir=layout.logs_root,
            rewrite=True,
        )
        selected_initial_summary = _answer_failure_output_summary(layout.mirrored_eval_path, layout.mirrored_events_path)
    deterministic_by_task = _deterministic_status_by_task(layout)

    if args.no_batch_auditor:
        model_validation_by_task = {
            task_id: {
                "status": "pass" if status.get("status") in {"valid", "valid_with_warnings"} else "invalid_untrusted",
                "notes": "row model validator skipped by --no-batch-auditor",
            }
            for task_id, status in deterministic_by_task.items()
            if task_id in audits_by_task
        }
        final_audits = audits_by_task
    else:
        existing_model_validation_by_task = _model_validation_status_by_task(layout)
        task_ids_to_validate = None
        if args.model_validator_stale_only or args.model_validator_repaired_only:
            only_task_ids = repaired_task_ids if args.model_validator_repaired_only else None
            task_ids_to_validate = set(
                _model_validator_stale_task_ids(
                    audits_by_task=audits_by_task,
                    deterministic_by_task=deterministic_by_task,
                    existing_model_validation_by_task=existing_model_validation_by_task,
                    only_task_ids=only_task_ids,
                )
            )
            if args.model_validator_repaired_only:
                print(
                    "Model-validator repaired-only mode: "
                    f"repaired valid/valid-with-warnings rows needing validator={len(task_ids_to_validate)}",
                    flush=True,
                )
            else:
                print(
                    "Model-validator stale-only mode: "
                    f"valid/valid-with-warnings rows needing validator rerun={len(task_ids_to_validate)}",
                    flush=True,
                )
        updated_model_validation_by_task, final_audits = run_model_validators(
            layout=layout,
            tmp_root=tmp_root,
            audits_by_task=audits_by_task,
            deterministic_by_task=deterministic_by_task,
            backend=args.auditor_backend or args.backend,
            model=args.auditor_model,
            reasoning_effort=args.auditor_reasoning_effort,
            timeout=args.timeout,
            allow_downgrade=args.strict_batch_auditor,
            concurrency=args.concurrency,
            task_ids_to_validate=task_ids_to_validate,
        )
        if args.model_validator_stale_only or args.model_validator_repaired_only:
            model_validation_by_task = dict(existing_model_validation_by_task)
            model_validation_by_task.update(updated_model_validation_by_task)
        else:
            model_validation_by_task = updated_model_validation_by_task

    write_mirrored_outputs(
        layout,
        final_audits,
        model_validation_by_task=model_validation_by_task,
    )
    final_validation_outputs = validate_answer_failure_root(
        source_root=layout.output_root,
        logs_dir=layout.logs_root,
        rewrite=True,
    )
    selected_final_summary = _answer_failure_output_summary(layout.mirrored_eval_path, layout.mirrored_events_path)
    report_outputs = build_answer_failure_report(
        source_root=layout.output_root,
        events_path=layout.mirrored_events_path,
        logs_dir=layout.logs_root,
    )
    return {
        "layout": layout,
        "eligible_rows": len(eligible_rows),
        "initial_invalid_rows": selected_initial_summary["selected_invalid_rows"]
        + selected_initial_summary["selected_missing_log_rows"],
        "final_invalid_rows": selected_final_summary["selected_invalid_rows"]
        + selected_final_summary["selected_missing_log_rows"],
        "root_initial_invalid_rows": len(validation_outputs["invalid_rows"]),
        "root_final_invalid_rows": len(final_validation_outputs["invalid_rows"]),
        "event_count": report_outputs["event_count"],
        **selected_final_summary,
        "eval_path": layout.mirrored_eval_path,
        "events_path": layout.mirrored_events_path,
        "report_path": layout.mirrored_report_path,
        "journal_path": journal_path,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-path", help="Source semantic eval_results.csv to audit")
    parser.add_argument("--repo-root", default=".", help="Repo root to scan when --eval-path is omitted")
    parser.add_argument(
        "--source-root",
        choices=sorted(SOURCE_ROOTS),
        default="results_semantic",
        help="Semantic results root to scan when --eval-path is omitted",
    )
    parser.add_argument("--backend", choices=["codex", "openai"], default="codex")
    parser.add_argument("--auditor-backend", choices=["codex", "openai"], default=None)
    parser.add_argument(
        "--row-audit-pipeline",
        choices=["single", "two-stage"],
        default=DEFAULT_ROW_AUDIT_PIPELINE,
        help="Row audit pipeline; two-stage extracts an evidence trace before labeling and is the default",
    )
    parser.add_argument("--row-model", default=DEFAULT_ROW_MODEL)
    parser.add_argument("--row-reasoning-effort", default=DEFAULT_ROW_REASONING)
    parser.add_argument("--auditor-model", default=DEFAULT_AUDITOR_MODEL)
    parser.add_argument("--auditor-reasoning-effort", default=DEFAULT_AUDITOR_REASONING)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--tmp-root", default=".tmp_answer_failure_audits")
    parser.add_argument("--journal-path", help="Append/resume row audit JSONL journal; defaults to /tmp/answer_failure_audits_{source_root}_{model}_{mode}.jsonl")
    parser.add_argument("--force", action="store_true", help="Rerun row audits even when temp JSON exists")
    parser.add_argument("--no-batch-auditor", action="store_true", help="Skip row model validation and trust deterministic validation")
    parser.add_argument(
        "--model-validator-stale-only",
        action="store_true",
        help=(
            "Run row model validators only for deterministic-valid rows whose existing "
            "model-validator status is blank or invalid_untrusted"
        ),
    )
    parser.add_argument(
        "--model-validator-repaired-only",
        action="store_true",
        help="Run row model validators only for deterministic-invalid rows repaired during this invocation.",
    )
    parser.add_argument("--strict-batch-auditor", action="store_true", help="Allow row model validators to downgrade deterministic-valid rows to invalid_untrusted")
    parser.add_argument("--repair-invalid-rounds", type=int, default=1, help="Rerun deterministic-invalid rows this many times before model validation")
    parser.add_argument("--limit", type=int, help="Audit only the first N eligible rows and stop before mirrored output writes")
    return parser


def choose_pending_eval_path(
    *,
    repo_root: Path,
    source_root_name: str,
    tmp_root: Optional[str | Path] = None,
    journal_path: Optional[str | Path] = None,
) -> Path:
    pending = find_pending_eval_stats(
        repo_root,
        source_root_name=source_root_name,
        tmp_root=tmp_root,
        journal_path=journal_path,
    )
    if not pending:
        raise SystemExit(f"No pending eval_results.csv files found under {repo_root / source_root_name}/modes")

    print(f"Pending answer-failure files under {repo_root / source_root_name}:")
    for index, stats in enumerate(pending, 1):
        artifact_note = "" if stats.has_all_artifacts else " missing_artifacts=yes"
        print(
            f"{index}. {stats.eval_path.relative_to(repo_root)} "
            f"pending_rows={stats.pending_rows} invalid_rows={stats.invalid_rows}{artifact_note}"
        )

    while True:
        choice = input("Pick a file number: ").strip()
        try:
            selected = int(choice)
        except ValueError:
            print("Enter a number from the list.")
            continue
        if 1 <= selected <= len(pending):
            return pending[selected - 1].eval_path
        print(f"Enter a number between 1 and {len(pending)}.")


def main() -> None:
    args = build_arg_parser().parse_args()
    outputs = run_audit_file(args)
    if outputs.get("smoke_only"):
        print("Smoke run only: skipped mirrored CSV/report writes")
    else:
        print(f"Wrote {outputs['eval_path']}")
        print(f"Wrote {outputs['events_path']}")
        print(f"Wrote {outputs['report_path']}")
    print(f"Selected file eval rows: {outputs.get('selected_eval_rows', '')}")
    print(f"Selected file eligible failed task rows: {outputs['eligible_rows']}")
    print(f"Selected file rows with answer-failure events: {outputs.get('selected_rows_with_events', '')}")
    print(f"Selected file answer-failure event rows: {outputs['event_count']}")
    print(f"Selected file invalid/missing-log rows before repair: {outputs['initial_invalid_rows']}")
    print(f"Selected file invalid/missing-log rows after validation: {outputs['final_invalid_rows']}")
    print(
        "Selected file valid rows still missing model-validator status: "
        f"{outputs.get('selected_valid_missing_model_validation_rows', '')}"
    )
    print(f"Selected file trusted rows for combiner: {outputs.get('selected_trusted_rows', '')}")
    print(f"Selected file trusted event rows for combiner: {outputs.get('selected_trusted_event_rows', '')}")
    if "root_final_invalid_rows" in outputs:
        print(f"Whole answer-failure root invalid/missing-log rows: {outputs['root_final_invalid_rows']}")
    print(f"JSONL journal: {outputs['journal_path']}")


if __name__ == "__main__":
    main()
