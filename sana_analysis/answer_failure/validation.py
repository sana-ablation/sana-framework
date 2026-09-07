#!/usr/bin/env python3
"""Validation helpers for answer-failure audit outputs."""
from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

from sana_analysis.running_analysis.answer_failure_taxonomy import ANSWER_FAILURE_TYPES, FAILURE_STAGES


CONFIDENCE_VALUES = {"high", "medium", "low"}

TRUSTED_MODEL_VALIDATION_STATUSES = {"pass", "repaired_pass"}

ROW_OUTPUT_COLUMNS = [
    "answer_failure_summary",
    "answer_failure_types",
    "answer_failure_subtypes",
    "answer_failure_event_count",
    "answer_failure_evidence",
    "answer_failure_validation_status",
    "answer_failure_validation_notes",
    "answer_failure_log_max_turn",
    "answer_failure_model_validation_status",
    "answer_failure_model_validation_notes",
]

EVENT_COLUMNS = [
    "task_id",
    "model_variant",
    "mode_variant",
    "event_index",
    "answer_failure_type",
    "answer_failure_subtype",
    "failure_stage",
    "failure_summary",
    "failure_evidence",
    "confidence",
]

VALIDATION_STATUS_FIELD = "answer_failure_validation_status"
VALIDATION_NOTES_FIELD = "answer_failure_validation_notes"
VALIDATION_MAX_TURN_FIELD = "answer_failure_log_max_turn"
MODEL_VALIDATION_STATUS_FIELD = "answer_failure_model_validation_status"
MODEL_VALIDATION_NOTES_FIELD = "answer_failure_model_validation_notes"

TURN_HEADER_RE = re.compile(r"--- Turn (\d+)")
TURN_REF_RE = re.compile(r"\bTurns?\s+(\d+)(?:-(\d+))?")
EXACT_EVIDENCE_RE = re.compile(r"\bTurn\s+(\d+)\s*[:|]\s*(.+)")

_EXACT_SNIPPET_PREFIXES = (
    "Executing:",
    "Tool result:",
    "WARNING",
    "ERROR",
    "MaxTokensReachedException",
    "Traceback",
    "Exception",
)


def _normalize_text(value: str | None) -> str:
    return " ".join((value or "").strip().split())


def _safe_int(value: str | int | float | None) -> Optional[int]:
    if value in ("", None):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def is_non_correct_row(row: dict) -> bool:
    semantic_bucket = _normalize_text(str(row.get("semantic_bucket", "")))
    semantic_match = _normalize_text(str(row.get("semantic_match", "")))
    return semantic_bucket in {"semantic_incorrect", "answer_unknown_blank"} or semantic_match == "0"


def row_log_path(logs_root: Path, model_variant: str, mode_variant: str, task_id: str) -> Path:
    task_path = Path(task_id)
    primary = logs_root / model_variant / mode_variant / task_path.with_suffix(".log")
    if primary.exists():
        return primary
    legacy = logs_root / model_variant / mode_variant / task_path.parent.name / f"{task_path.stem}.log"
    if legacy.exists():
        return legacy
    return primary


def parse_log_turn_blocks(log_path: Path) -> tuple[int, dict[int, str]]:
    max_turn = 0
    blocks: dict[int, list[str]] = {}
    current_turn: Optional[int] = None

    with log_path.open() as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            turn_match = TURN_HEADER_RE.search(line)
            if turn_match:
                current_turn = int(turn_match.group(1))
                max_turn = max(max_turn, current_turn)
                blocks.setdefault(current_turn, [])
                continue
            if current_turn is None:
                continue
            parts = line.split(" | ", 4)
            message = parts[-1] if len(parts) >= 5 else line.strip()
            if message:
                blocks[current_turn].append(_normalize_text(message))

    return max_turn, {turn: " ".join(lines) for turn, lines in blocks.items()}


def parse_turn_refs(text: str | None) -> list[int]:
    turns: list[int] = []
    for match in TURN_REF_RE.finditer(text or ""):
        start = int(match.group(1))
        end = int(match.group(2) or start)
        if end < start:
            start, end = end, start
        for turn in range(start, end + 1):
            if turn not in turns:
                turns.append(turn)
    return turns


def _variants_from_eval_path(source_root: Path, eval_path: Path) -> tuple[str, str] | None:
    rel_path = eval_path.relative_to(source_root)
    parts = rel_path.parts
    if "modes" in parts:
        modes_index = parts.index("modes")
        after_modes = parts[modes_index + 1 :]
        if len(after_modes) >= 3:
            return after_modes[0], after_modes[1]
    if len(parts) >= 4:
        return parts[-3], parts[-2]
    return None


def validate_answer_failure_row(row: dict, events: list[dict], log_path: Path | None) -> dict:
    issues: list[str] = []
    warnings: list[str] = []
    non_correct = is_non_correct_row(row)

    if not non_correct:
        if events:
            issues.append("semantic-correct row has answer failure events")
        for field in ROW_OUTPUT_COLUMNS:
            if field.startswith("answer_failure_validation"):
                continue
            if _normalize_text(str(row.get(field, ""))):
                issues.append(f"semantic-correct row has non-blank {field}")
                break
        return {
            "status": "invalid" if issues else "not_audited",
            "notes": "; ".join(issues),
            "log_max_turn": "",
            "issues": issues,
            "warnings": warnings,
        }

    if log_path is None or not log_path.exists():
        if len(events) == 1 and event_is_missing_log_ungroundable(events[0]):
            event = events[0]
            for required in ("failure_stage", "failure_summary", "failure_evidence", "confidence"):
                if not _normalize_text(str(event.get(required, ""))):
                    issues.append(f"event 1: missing {required}")

            failure_stage = _normalize_text(str(event.get("failure_stage", "")))
            if failure_stage and failure_stage not in FAILURE_STAGES:
                issues.append(f"event 1: invalid failure_stage `{failure_stage}`")

            confidence = _normalize_text(str(event.get("confidence", "")))
            if confidence and confidence not in CONFIDENCE_VALUES:
                issues.append(f"event 1: invalid confidence `{confidence}`")

            if not issues:
                warnings.append("matching raw log is missing; row is marked ungroundable")
                return {
                    "status": "valid_with_warnings",
                    "notes": "; ".join(warnings),
                    "log_max_turn": "",
                    "issues": issues,
                    "warnings": warnings,
                }
        elif not events:
            issues.append("matching raw log is missing and no ungroundable event was provided")
        elif any(event.get("answer_failure_type") == "ungroundable" for event in events):
            issues.append("matching raw log is missing; use exactly one ungroundable event with subtype missing_or_malformed_log")
        else:
            issues.append("matching raw log is missing; non-correct row must be marked ungroundable")
        return {
            "status": "missing_log",
            "notes": "; ".join(issues) or "matching raw log is missing",
            "log_max_turn": "",
            "issues": issues,
            "warnings": warnings,
        }

    log_max_turn, turn_blocks = parse_log_turn_blocks(log_path)

    if not events:
        issues.append("non-correct row has no answer failure events")

    declared_event_count = _safe_int(row.get("answer_failure_event_count"))
    if declared_event_count is not None and declared_event_count != len(events):
        issues.append(
            f"answer_failure_event_count={declared_event_count} but answer_failure_events.csv has {len(events)} events"
        )

    for index, event in enumerate(events, 1):
        failure_type = _normalize_text(str(event.get("answer_failure_type", "")))
        if failure_type not in ANSWER_FAILURE_TYPES:
            issues.append(f"event {index}: invalid answer_failure_type `{failure_type}`")

        if failure_type == "tool_or_data_blocker" and not _normalize_text(str(event.get("answer_failure_subtype", ""))):
            issues.append(f"event {index}: tool_or_data_blocker requires answer_failure_subtype")

        for required in ("failure_stage", "failure_summary", "failure_evidence", "confidence"):
            if not _normalize_text(str(event.get(required, ""))):
                issues.append(f"event {index}: missing {required}")

        failure_stage = _normalize_text(str(event.get("failure_stage", "")))
        if failure_stage and failure_stage not in FAILURE_STAGES:
            issues.append(f"event {index}: invalid failure_stage `{failure_stage}`")

        confidence = _normalize_text(str(event.get("confidence", "")))
        if confidence and confidence not in CONFIDENCE_VALUES:
            issues.append(f"event {index}: invalid confidence `{confidence}`")

        evidence = str(event.get("failure_evidence", ""))
        normalized_evidence = _normalize_text(evidence)
        if normalized_evidence and not re.match(r"^Turn\s+\d+\s+\|", normalized_evidence):
            warnings.append(f"event {index}: failure_evidence must start with `Turn N |` for rows with existing raw logs")
        turn_refs = parse_turn_refs(evidence)
        if turn_refs and max(turn_refs) > log_max_turn:
            issues.append(f"event {index}: failure_evidence references turns beyond log max turn {log_max_turn}")

        for entry in evidence.split(";"):
            match = EXACT_EVIDENCE_RE.search(entry.strip())
            if not match:
                continue
            turn = int(match.group(1))
            snippet = _normalize_text(match.group(2))
            if not snippet.startswith(_EXACT_SNIPPET_PREFIXES):
                continue
            if turn > log_max_turn:
                continue
            block_text = turn_blocks.get(turn, "")
            if snippet and snippet not in block_text:
                warnings.append(f"event {index}: exact evidence snippet for turn {turn} does not match the raw log")

    status = "valid"
    if issues:
        status = "invalid"
    elif warnings:
        status = "valid_with_warnings"

    return {
        "status": status,
        "notes": "; ".join(issues + warnings),
        "log_max_turn": log_max_turn,
        "issues": issues,
        "warnings": warnings,
    }


def _summarize_events(events: list[dict]) -> dict[str, str]:
    types: list[str] = []
    subtypes: list[str] = []
    evidence: list[str] = []
    summaries: list[str] = []
    for event in sorted(events, key=lambda row: _safe_int(row.get("event_index")) or 0):
        failure_type = _normalize_text(str(event.get("answer_failure_type", "")))
        subtype = _normalize_text(str(event.get("answer_failure_subtype", "")))
        if failure_type and failure_type not in types:
            types.append(failure_type)
        if subtype and subtype not in subtypes:
            subtypes.append(subtype)
        if _normalize_text(str(event.get("failure_evidence", ""))):
            evidence.append(_normalize_text(str(event.get("failure_evidence", ""))))
        if _normalize_text(str(event.get("failure_summary", ""))):
            summaries.append(_normalize_text(str(event.get("failure_summary", ""))))
    return {
        "answer_failure_summary": "; ".join(summaries[:3]),
        "answer_failure_types": "; ".join(types),
        "answer_failure_subtypes": "; ".join(subtypes),
        "answer_failure_event_count": str(len(events)),
        "answer_failure_evidence": "; ".join(evidence[:4]),
    }


def event_is_missing_log_ungroundable(event: dict) -> bool:
    return (
        _normalize_text(str(event.get("answer_failure_type", ""))) == "ungroundable"
        and _normalize_text(str(event.get("answer_failure_subtype", ""))) == "missing_or_malformed_log"
    )


def validate_answer_failure_root(
    source_root: str | Path = "results_semantic_answer_failures",
    *,
    logs_dir: str | Path = "logs/modes",
    rewrite: bool = False,
) -> dict:
    source_root = Path(source_root)
    logs_root = Path(logs_dir)
    audited_rows = 0
    invalid_rows: list[dict] = []
    file_rows: list[tuple[Path, list[dict], list[str]]] = []

    for eval_path in sorted(source_root.rglob("eval_results.csv")):
        variants = _variants_from_eval_path(source_root, eval_path)
        if variants is None:
            continue
        model_variant, mode_variant = variants
        events_path = eval_path.parent / "answer_failure_events.csv"

        with eval_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            fieldnames = list(reader.fieldnames or [])

        for required in ROW_OUTPUT_COLUMNS:
            if required not in fieldnames:
                fieldnames.append(required)

        events_by_task: dict[str, list[dict]] = defaultdict(list)
        if events_path.exists():
            with events_path.open(newline="") as handle:
                for event in csv.DictReader(handle):
                    events_by_task[str(event.get("task_id", ""))].append(event)

        for row in rows:
            task_id = str(row.get("task_id", ""))
            events = events_by_task.get(task_id, [])
            non_correct = is_non_correct_row(row)
            if non_correct:
                audited_rows += 1

            log_path = row_log_path(logs_root, model_variant, mode_variant, task_id)
            result = validate_answer_failure_row(row, events, log_path)
            if non_correct:
                for key, value in _summarize_events(events).items():
                    row[key] = value
                row[VALIDATION_STATUS_FIELD] = result["status"]
                row[VALIDATION_NOTES_FIELD] = result["notes"]
                row[VALIDATION_MAX_TURN_FIELD] = result["log_max_turn"]
            else:
                for field in ROW_OUTPUT_COLUMNS:
                    row[field] = ""

            if result["status"] in {"invalid", "missing_log"}:
                invalid_rows.append(
                    {
                        "model_variant": model_variant,
                        "mode_variant": mode_variant,
                        "task_id": task_id,
                        VALIDATION_STATUS_FIELD: result["status"],
                        VALIDATION_NOTES_FIELD: result["notes"],
                        VALIDATION_MAX_TURN_FIELD: result["log_max_turn"],
                        "answer_failure_types": row.get("answer_failure_types", ""),
                        "answer_failure_summary": row.get("answer_failure_summary", ""),
                    }
                )

        file_rows.append((eval_path, rows, fieldnames))

    if rewrite:
        for eval_path, rows, fieldnames in file_rows:
            with eval_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

    failures_path = source_root / "answer_failure_validation_failures.csv"
    report_path = source_root / "answer_failure_validation_report.md"
    failures_path.parent.mkdir(parents=True, exist_ok=True)

    failure_fieldnames = [
        "model_variant",
        "mode_variant",
        "task_id",
        VALIDATION_STATUS_FIELD,
        VALIDATION_NOTES_FIELD,
        VALIDATION_MAX_TURN_FIELD,
        "answer_failure_types",
        "answer_failure_summary",
    ]
    with failures_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=failure_fieldnames)
        writer.writeheader()
        writer.writerows(invalid_rows)

    status_counts: Counter[str] = Counter()
    for _, rows, _ in file_rows:
        for row in rows:
            status = str(row.get(VALIDATION_STATUS_FIELD, ""))
            if status:
                status_counts[status] += 1

    lines = [
        "# Answer Failure Validation Report",
        "",
        f"- Source root: `{source_root}`",
        f"- Logs root: `{logs_root}`",
        f"- Non-correct rows checked: {audited_rows}",
        f"- Invalid or missing-log rows: {len(invalid_rows)}",
        "",
        "## Status Counts",
        "",
    ]
    for status, count in sorted(status_counts.items()):
        lines.append(f"- `{status}`: {count}")

    if invalid_rows:
        lines.extend(["", "## Invalid Rows", ""])
        for row in invalid_rows[:20]:
            lines.append(
                f"- `{row['task_id']}` in `{row['model_variant']}/{row['mode_variant']}`: "
                f"{row[VALIDATION_STATUS_FIELD]} - {row[VALIDATION_NOTES_FIELD]}"
            )

    report_path.write_text("\n".join(lines) + "\n")
    return {
        "audited_rows": audited_rows,
        "invalid_rows": invalid_rows,
        "failures_path": failures_path,
        "report_path": report_path,
    }
