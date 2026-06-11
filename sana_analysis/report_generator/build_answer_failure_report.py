#!/usr/bin/env python3
"""Build Markdown reports from answer-failure audit event CSVs."""
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sana_analysis.running_analysis.answer_failure_validation import (
    MODEL_VALIDATION_STATUS_FIELD,
    TRUSTED_MODEL_VALIDATION_STATUSES,
    VALIDATION_STATUS_FIELD,
    row_log_path,
)

TRUSTED_DETERMINISTIC_STATUSES = {"valid", "valid_with_warnings"}


def _normalize_text(value: str | None) -> str:
    return " ".join((value or "").strip().split())


def _truncate(value: str, limit: int = 220) -> str:
    text = _normalize_text(value)
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _relative_link(from_dir: Path, target: Path) -> str:
    return os.path.relpath(target, start=from_dir)


def _markdown_link(label: str, from_dir: Path, target: Optional[Path]) -> str:
    if target is None:
        return label
    return f"[{label}]({_relative_link(from_dir, target)})"


def _repo_root_from_source_root(source_root: Path) -> Path:
    for candidate in [source_root, *source_root.parents]:
        if (candidate / ".git").exists() or (candidate / "tasks_mini").exists():
            return candidate
    return source_root.parent


def _variants_from_events_path(source_root: Path, events_path: Path) -> tuple[str, str] | None:
    rel_path = events_path.relative_to(source_root)
    parts = rel_path.parts
    if "modes" in parts:
        modes_index = parts.index("modes")
        after_modes = parts[modes_index + 1 :]
        if len(after_modes) >= 3:
            return after_modes[0], after_modes[1]
    if len(parts) >= 4:
        return parts[-3], parts[-2]
    return None


def _resolve_task_path(repo_root: Path, task_id: str) -> Optional[Path]:
    task_path = repo_root / Path(task_id)
    if task_path.exists():
        return task_path
    return None


def _resolve_plan_path(repo_root: Path, task_id: str) -> Optional[Path]:
    task_path = Path(task_id)
    if not task_path.parts:
        return None
    plan_roots = {
        "tasks_mini": "plans_mini",
        "tasks_mini_core": "plans_mini_core",
        "tasks-mini-kramabench": "plans-mini-kramabench",
        "tasks_core_quality": "plans_core_quality",
    }
    plan_root = plan_roots.get(task_path.parts[0])
    if plan_root is None:
        return None
    plan_path = repo_root / plan_root / Path(*task_path.parts[1:])
    return plan_path if plan_path.exists() else None


def _add_table(lines: list[str], headers: list[str], table_rows: list[list[str]]) -> None:
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in table_rows:
        lines.append("| " + " | ".join(str(cell).replace("\n", " ") for cell in row) + " |")
    lines.append("")


def _load_rows_for_events_file(source_root: Path, events_path: Path) -> tuple[list[dict], list[dict]]:
    eval_path = events_path.parent / "eval_results.csv"
    eval_rows: list[dict] = []
    with eval_path.open(newline="") as handle:
        eval_rows = list(csv.DictReader(handle))
    eval_by_task = {str(row.get("task_id", "")): row for row in eval_rows}

    variants = _variants_from_events_path(source_root, events_path)
    model_variant, mode_variant = variants if variants else ("", "")
    events: list[dict] = []
    with events_path.open(newline="") as handle:
        for event in csv.DictReader(handle):
            enriched = dict(event)
            enriched["model_variant"] = enriched.get("model_variant") or model_variant
            enriched["mode_variant"] = enriched.get("mode_variant") or mode_variant
            enriched["eval_path"] = eval_path
            enriched["events_path"] = events_path
            source_row = eval_by_task.get(str(enriched.get("task_id", "")), {})
            enriched["row_validation_status"] = source_row.get(VALIDATION_STATUS_FIELD, "")
            enriched["row_model_validation_status"] = source_row.get(MODEL_VALIDATION_STATUS_FIELD, "")
            events.append(enriched)
    return eval_rows, events


def _trusted_events(events: list[dict]) -> list[dict]:
    trusted: list[dict] = []
    for event in events:
        deterministic = str(event.get("row_validation_status", ""))
        model_status = str(event.get("row_model_validation_status", ""))
        if deterministic not in TRUSTED_DETERMINISTIC_STATUSES:
            continue
        if model_status not in TRUSTED_MODEL_VALIDATION_STATUSES:
            continue
        trusted.append(event)
    return trusted


def _representative_cell(row: dict, *, report_dir: Path, repo_root: Path, logs_root: Optional[Path]) -> str:
    task_id = str(row.get("task_id", ""))
    task_path = _resolve_task_path(repo_root, task_id)
    plan_path = _resolve_plan_path(repo_root, task_id)
    log_path = None
    if logs_root is not None:
        candidate = row_log_path(
            logs_root,
            str(row.get("model_variant", "")),
            str(row.get("mode_variant", "")),
            task_id,
        )
        log_path = candidate if candidate.exists() else None
    return " / ".join(
        [
            _markdown_link("task", report_dir, task_path),
            _markdown_link("plan", report_dir, plan_path),
            _markdown_link("log", report_dir, log_path),
        ]
    )


def _count_by_key(events: list[dict], key_fn) -> list[tuple[tuple[str, ...], int]]:
    counts: Counter[tuple[str, ...]] = Counter()
    for event in events:
        counts[key_fn(event)] += 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))


def _representatives_by_key(events: list[dict], key_fn) -> dict[tuple[str, ...], dict]:
    grouped: dict[tuple[str, ...], list[dict]] = defaultdict(list)
    for event in events:
        grouped[key_fn(event)].append(event)
    return {key: sorted(rows, key=lambda row: str(row.get("task_id", "")))[0] for key, rows in grouped.items()}


def build_answer_failure_report(
    source_root: str | Path = "results_semantic_answer_failures",
    *,
    events_path: str | Path | None = None,
    logs_dir: str | Path | None = None,
    representative_limit: int = 5,
) -> dict:
    source_root = Path(source_root)
    repo_root = _repo_root_from_source_root(source_root)
    logs_root = Path(logs_dir) if logs_dir is not None else None
    event_files = [Path(events_path)] if events_path is not None else sorted(source_root.rglob("answer_failure_events.csv"))
    written_reports: list[Path] = []
    all_events: list[dict] = []

    for event_file in event_files:
        eval_rows, events = _load_rows_for_events_file(source_root, event_file)
        trusted = _trusted_events(events)
        all_events.extend(events)
        report_path = event_file.parent / "answer_failure_report.md"
        _write_report(
            report_path,
            title="Answer Failure Report",
            source_root=source_root,
            eval_rows=eval_rows,
            events=events,
            trusted_events=trusted,
            repo_root=repo_root,
            logs_root=logs_root,
            representative_limit=representative_limit,
        )
        written_reports.append(report_path)

    aggregate_path = source_root / "answer_failure_report.md"
    if event_files:
        _write_report(
            aggregate_path,
            title="Answer Failure Aggregate Report",
            source_root=source_root,
            eval_rows=[],
            events=all_events,
            trusted_events=_trusted_events(all_events),
            repo_root=repo_root,
            logs_root=logs_root,
            representative_limit=representative_limit,
        )
        written_reports.append(aggregate_path)

    return {"report_paths": written_reports, "event_files": event_files, "event_count": len(all_events)}


def _write_report(
    report_path: Path,
    *,
    title: str,
    source_root: Path,
    eval_rows: list[dict],
    events: list[dict],
    trusted_events: list[dict],
    repo_root: Path,
    logs_root: Optional[Path],
    representative_limit: int,
) -> None:
    report_dir = report_path.parent
    non_correct_count = sum(
        str(row.get("semantic_bucket", "")) in {"semantic_incorrect", "answer_unknown_blank"}
        or str(row.get("semantic_match", "")) == "0"
        for row in eval_rows
    )
    rejected_count = len(events) - len(trusted_events)
    lines = [
        f"# {title}",
        "",
        f"- Source root: `{source_root}`",
        f"- Event rows: {len(events)}",
        f"- Trusted event rows: {len(trusted_events)}",
    ]
    if eval_rows:
        lines.append(f"- Non-correct eval rows: {non_correct_count}")
    if rejected_count:
        lines.append(f"- Events excluded by validation/model-validation: {rejected_count}")
    lines.append("")

    lines.extend(["## Counts by Failure Type", ""])
    type_rows = _count_by_key(trusted_events, lambda row: (_normalize_text(str(row.get("answer_failure_type", ""))) or "(blank)",))
    _add_table(lines, ["Failure Type", "Events"], [[key[0], str(total)] for key, total in type_rows])

    lines.extend(["## Counts by Failure Type and Subtype", ""])
    type_subtype_key = lambda row: (
        _normalize_text(str(row.get("answer_failure_type", ""))) or "(blank)",
        _normalize_text(str(row.get("answer_failure_subtype", ""))) or "(none)",
    )
    subtype_rows = _count_by_key(trusted_events, type_subtype_key)
    subtype_representatives = _representatives_by_key(trusted_events, type_subtype_key)
    _add_table(
        lines,
        ["Failure Type", "Subtype", "Events", "Representative"],
        [
            [
                key[0],
                key[1],
                str(total),
                _representative_cell(
                    subtype_representatives[key],
                    report_dir=report_dir,
                    repo_root=repo_root,
                    logs_root=logs_root,
                ),
            ]
            for key, total in subtype_rows
        ],
    )

    lines.extend(["## Co-occurring Failure Types", ""])
    by_task: dict[str, set[str]] = defaultdict(set)
    for event in trusted_events:
        by_task[str(event.get("task_id", ""))].add(_normalize_text(str(event.get("answer_failure_type", ""))) or "(blank)")
    cooccurrence = Counter("; ".join(sorted(types)) for types in by_task.values())
    _add_table(
        lines,
        ["Failure Type Set", "Rows"],
        [[key, str(total)] for key, total in sorted(cooccurrence.items(), key=lambda item: (-item[1], item[0]))],
    )

    lines.extend(["## Representative Evidence", ""])
    for event in trusted_events[:representative_limit]:
        lines.append(
            f"- `{event.get('task_id', '')}` "
            f"`{event.get('answer_failure_type', '')}`/"
            f"`{event.get('answer_failure_subtype', '') or '(none)'}`: "
            f"{_truncate(str(event.get('failure_evidence', '')))}"
        )

    report_path.write_text("\n".join(lines).rstrip() + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default="results_semantic_answer_failures")
    parser.add_argument("--events-path")
    parser.add_argument("--logs-dir")
    parser.add_argument("--representative-limit", type=int, default=5)
    args = parser.parse_args()
    outputs = build_answer_failure_report(
        source_root=args.source_root,
        events_path=args.events_path,
        logs_dir=args.logs_dir,
        representative_limit=args.representative_limit,
    )
    for report_path in outputs["report_paths"]:
        print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
