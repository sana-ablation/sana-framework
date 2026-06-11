#!/usr/bin/env python3
"""Migrate answer-failure audit outputs to the current taxonomy."""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sana_analysis.running_analysis.answer_failure_taxonomy import ANSWER_FAILURE_TYPES
from sana_analysis.running_analysis.answer_failure_validation import validate_answer_failure_root
from sana_analysis.report_generator.build_answer_failure_report import build_answer_failure_report
from sana_analysis.report_generator.combine_answer_failure_grouped_models import combine_answer_failures


AMBIGUOUS_OLD_TYPES = {
    "wrong_source_or_scope",
}

AUTOMATIC_TYPE_MAP = {
    "answer_computation_mismatch": "computation_or_aggregation_error",
    "data_selection_or_computation_error": "wrong_scope_or_filter",
    "final_answer_mismatch": "evidence_available_answer_error",
    "incomplete_evidence_not_enough_turns": "incomplete_evidence_budget_exhausted",
    "wrong_filter_or_row_subset": "wrong_scope_or_filter",
    "wrong_scope_or_population": "wrong_scope_or_filter",
}

LEGACY_SUBTYPE_MAP = {
    "incomplete_evidence_not_enough_turns": "budget_exhausted",
    "wrong_source_or_scope": "source_or_scope_error",
}

DECISION_COLUMNS = [
    "source_events_path",
    "task_id",
    "event_index",
    "old_type",
    "new_type",
    "new_subtype",
    "review_required",
    "confidence",
    "rationale",
]

REVIEW_COLUMNS = [
    "source_events_path",
    "task_id",
    "event_index",
    "old_type",
    "old_subtype",
    "failure_stage",
    "failure_summary",
    "failure_evidence",
    "confidence",
    "eval_answer_failure_types",
    "eval_answer_failure_subtypes",
    "eval_answer_failure_summary",
    "suggested_new_type",
    "suggested_new_subtype",
    "migration_instruction",
]

SOURCE_DATASET_KEYWORDS = (
    "database",
    "dataset",
    "external source",
    "file",
    "source family",
    "source version",
    "table",
)

SCOPE_FILTER_KEYWORDS = (
    "branch",
    "city",
    "county",
    "date",
    "district",
    "entity",
    "filter",
    "geograph",
    "inclusion",
    "jurisdiction",
    "predicate",
    "population",
    "row subset",
    "scope",
    "slice",
    "time period",
    "ward",
    "where",
    "year",
)

COMPUTATION_KEYWORDS = (
    "aggregation",
    "arithmetic",
    "average",
    "counted",
    "dedup",
    "denominator",
    "distinct",
    "group",
    "join",
    "math",
    "rank",
    "ratio",
    "round",
    "sum",
)

EXTRACTION_KEYWORDS = (
    "cell",
    "column",
    "extracted",
    "field",
    "nested value",
    "read",
    "span",
)


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _event_files(root: Path) -> list[Path]:
    return sorted(root.rglob("answer_failure_events.csv"))


def _relative_events_path(root: Path, events_path: Path) -> str:
    return events_path.relative_to(root).as_posix()


def _eval_rows_by_task(events_path: Path) -> dict[str, dict[str, str]]:
    eval_path = events_path.parent / "eval_results.csv"
    if not eval_path.exists():
        return {}
    _, rows = _read_csv(eval_path)
    return {str(row.get("task_id", "")): row for row in rows}


def _needs_review(event: dict[str, str], *, include_all: bool) -> bool:
    if include_all:
        return True
    failure_type = str(event.get("answer_failure_type", ""))
    return failure_type in AMBIGUOUS_OLD_TYPES


def _suggested_new_type(old_type: str) -> str:
    return AUTOMATIC_TYPE_MAP.get(old_type, "")


def _event_text(event: dict[str, str]) -> str:
    return " ".join(
        str(event.get(field, ""))
        for field in (
            "answer_failure_subtype",
            "failure_stage",
            "failure_summary",
            "failure_evidence",
        )
    ).lower()


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in text for keyword in keywords)


def infer_ambiguous_event_decision(
    source_events_path: str,
    event: dict[str, str],
) -> dict[str, str]:
    """Create a transparent one-shot decision for an old ambiguous event row."""
    old_type = str(event.get("answer_failure_type", ""))
    if old_type not in AMBIGUOUS_OLD_TYPES:
        raise ValueError(f"expected ambiguous old type, got `{old_type}`")

    text = _event_text(event)
    if _contains_any(text, SOURCE_DATASET_KEYWORDS):
        new_type = "wrong_source_or_dataset"
        rationale = "Mentions a dataset/file/table/database/source-family/source-version issue."
        review_required = "false"
    elif _contains_any(text, EXTRACTION_KEYWORDS) and not _contains_any(text, SCOPE_FILTER_KEYWORDS):
        new_type = "extraction_or_parsing_error"
        rationale = "Mentions a wrong field/cell/column/span read without a scope/filter cue."
        review_required = "true"
    elif _contains_any(text, COMPUTATION_KEYWORDS) and not _contains_any(text, SCOPE_FILTER_KEYWORDS):
        new_type = "computation_or_aggregation_error"
        rationale = "Mentions aggregation/math/ranking/counting without a scope/filter cue."
        review_required = "true"
    else:
        new_type = "wrong_scope_or_filter"
        rationale = "Default for old source/scope rows that describe branch, entity, year, geography, filter, population, or query-scope problems."
        review_required = "false"

    return {
        "source_events_path": source_events_path,
        "task_id": str(event.get("task_id", "")),
        "event_index": str(event.get("event_index", "")),
        "old_type": old_type,
        "new_type": new_type,
        "new_subtype": str(event.get("answer_failure_subtype", "")),
        "review_required": review_required,
        "confidence": str(event.get("confidence", "")) or "medium",
        "rationale": rationale,
    }


def export_review_packet(
    source_root: str | Path,
    review_path: str | Path,
    *,
    include_all: bool = False,
) -> dict[str, int | Path]:
    """Export rows that need agent taxonomy review."""
    source_root = Path(source_root)
    review_path = Path(review_path)
    rows: list[dict[str, str]] = []

    for events_path in _event_files(source_root):
        rel_events_path = _relative_events_path(source_root, events_path)
        _, events = _read_csv(events_path)
        eval_by_task = _eval_rows_by_task(events_path)
        for event in events:
            if not _needs_review(event, include_all=include_all):
                continue
            old_type = str(event.get("answer_failure_type", ""))
            task_id = str(event.get("task_id", ""))
            eval_row = eval_by_task.get(task_id, {})
            rows.append(
                {
                    "source_events_path": rel_events_path,
                    "task_id": task_id,
                    "event_index": str(event.get("event_index", "")),
                    "old_type": old_type,
                    "old_subtype": str(event.get("answer_failure_subtype", "")),
                    "failure_stage": str(event.get("failure_stage", "")),
                    "failure_summary": str(event.get("failure_summary", "")),
                    "failure_evidence": str(event.get("failure_evidence", "")),
                    "confidence": str(event.get("confidence", "")),
                    "eval_answer_failure_types": str(eval_row.get("answer_failure_types", "")),
                    "eval_answer_failure_subtypes": str(eval_row.get("answer_failure_subtypes", "")),
                    "eval_answer_failure_summary": str(eval_row.get("answer_failure_summary", "")),
                    "suggested_new_type": _suggested_new_type(old_type),
                    "suggested_new_subtype": str(event.get("answer_failure_subtype", "")),
                    "migration_instruction": (
                        "Choose exactly one new_type from: wrong_source_or_dataset, wrong_scope_or_filter, "
                        "computation_or_aggregation_error, extraction_or_parsing_error, "
                        "evidence_available_answer_error, question_or_constraint_misread, "
                        "planning_decomposition_mismatch, incomplete_evidence_early_answer, "
                        "incomplete_evidence_budget_exhausted, tool_or_data_blocker, "
                        "semantic_or_gold_label_issue, other_or_unclear, ungroundable."
                    ),
                }
            )

    _write_csv(review_path, REVIEW_COLUMNS, rows)
    return {"rows_written": len(rows), "review_path": review_path}


def write_auto_decisions(
    source_root: str | Path,
    decisions_path: str | Path,
) -> dict[str, int | Path]:
    source_root = Path(source_root)
    decisions_path = Path(decisions_path)
    decisions: list[dict[str, str]] = []
    review_required = 0

    for events_path in _event_files(source_root):
        rel_events_path = _relative_events_path(source_root, events_path)
        _, events = _read_csv(events_path)
        for event in events:
            if str(event.get("answer_failure_type", "")) not in AMBIGUOUS_OLD_TYPES:
                continue
            decision = infer_ambiguous_event_decision(rel_events_path, event)
            if decision["review_required"].lower() == "true":
                review_required += 1
            decisions.append(decision)

    _write_csv(decisions_path, DECISION_COLUMNS, decisions)
    return {
        "auto_decisions_written": len(decisions),
        "review_required": review_required,
        "decisions_path": decisions_path,
    }


def _load_decisions(decisions_path: Path | None) -> dict[tuple[str, str, str], dict[str, str]]:
    if decisions_path is None:
        return {}
    _, rows = _read_csv(decisions_path)
    decisions: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in rows:
        key = (
            str(row.get("source_events_path", "")),
            str(row.get("task_id", "")),
            str(row.get("event_index", "")),
        )
        if not all(key):
            raise ValueError(f"decision row has incomplete key: {row}")
        old_type = str(row.get("old_type", ""))
        new_type = str(row.get("new_type", ""))
        if new_type and new_type not in ANSWER_FAILURE_TYPES:
            raise ValueError(f"decision for {key} has invalid new_type `{new_type}`")
        if not new_type and old_type in AMBIGUOUS_OLD_TYPES:
            raise ValueError(f"decision for {key} is missing new_type")
        decisions[key] = row
    return decisions


def _copy_source_root(source_root: Path, output_root: Path, *, overwrite: bool) -> None:
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"output root already exists: {output_root}")
        shutil.rmtree(output_root)
    shutil.copytree(source_root, output_root)


def _migrated_type(
    *,
    rel_events_path: str,
    event: dict[str, str],
    decisions: dict[tuple[str, str, str], dict[str, str]],
) -> tuple[str, str, bool]:
    old_type = str(event.get("answer_failure_type", ""))
    old_subtype = str(event.get("answer_failure_subtype", ""))
    key = (rel_events_path, str(event.get("task_id", "")), str(event.get("event_index", "")))
    decision = decisions.get(key)

    if decision is not None:
        if decision.get("old_type") and str(decision.get("old_type")) != old_type:
            raise ValueError(f"decision old_type mismatch for {key}: {decision.get('old_type')} != {old_type}")
        new_type = str(decision.get("new_type", "")) or old_type
        if new_type not in ANSWER_FAILURE_TYPES:
            raise ValueError(f"decision for {key} has invalid new_type `{new_type}`")
        new_subtype = str(decision.get("new_subtype", "")) or old_subtype
        new_subtype = LEGACY_SUBTYPE_MAP.get(new_subtype, new_subtype)
        return new_type, new_subtype, (new_type != old_type or new_subtype != old_subtype)

    if old_type in AMBIGUOUS_OLD_TYPES:
        raise ValueError(f"missing migration decision for ambiguous event {key}")

    new_type = AUTOMATIC_TYPE_MAP.get(old_type, old_type)
    if new_type not in ANSWER_FAILURE_TYPES:
        raise ValueError(f"event {key} has invalid answer_failure_type `{old_type}`")
    new_subtype = LEGACY_SUBTYPE_MAP.get(old_subtype, old_subtype)
    return new_type, new_subtype, (new_type != old_type or new_subtype != old_subtype)


def _row_summary_from_events(events: list[dict[str, str]]) -> dict[str, str]:
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


def _rewrite_eval_summary(events_path: Path) -> bool:
    eval_path = events_path.parent / "eval_results.csv"
    if not eval_path.exists():
        return False
    fieldnames, eval_rows = _read_csv(eval_path)
    if not fieldnames:
        return False
    _, events = _read_csv(events_path)
    events_by_task: dict[str, list[dict[str, str]]] = defaultdict(list)
    for event in events:
        events_by_task[str(event.get("task_id", ""))].append(event)

    needed_fields = [
        "answer_failure_summary",
        "answer_failure_types",
        "answer_failure_subtypes",
        "answer_failure_event_count",
        "answer_failure_evidence",
    ]
    for field in needed_fields:
        if field not in fieldnames:
            fieldnames.append(field)

    changed = False
    for row in eval_rows:
        task_id = str(row.get("task_id", ""))
        if task_id not in events_by_task:
            continue
        summary = _row_summary_from_events(events_by_task[task_id])
        for field, value in summary.items():
            if row.get(field, "") != value:
                row[field] = value
                changed = True

    if changed:
        _write_csv(eval_path, fieldnames, eval_rows)
    return changed


def apply_migration(
    source_root: str | Path,
    output_root: str | Path,
    *,
    decisions_path: str | Path | None = None,
    overwrite: bool = False,
    validate_logs_dir: str | Path | None = None,
    rebuild_reports: bool = False,
    combine_output_root: str | Path | None = None,
) -> dict[str, int | Path]:
    source_root = Path(source_root)
    output_root = Path(output_root)
    decisions = _load_decisions(Path(decisions_path) if decisions_path is not None else None)

    _copy_source_root(source_root, output_root, overwrite=overwrite)

    event_files = _event_files(output_root)
    events_changed = 0
    event_files_rewritten = 0
    eval_files_rewritten = 0

    for events_path in event_files:
        rel_events_path = _relative_events_path(output_root, events_path)
        fieldnames, events = _read_csv(events_path)
        changed = False
        for event in events:
            new_type, new_subtype, event_changed = _migrated_type(
                rel_events_path=rel_events_path,
                event=event,
                decisions=decisions,
            )
            if event_changed:
                event["answer_failure_type"] = new_type
                event["answer_failure_subtype"] = new_subtype
                events_changed += 1
                changed = True
        if changed:
            _write_csv(events_path, fieldnames, events)
            event_files_rewritten += 1
        if _rewrite_eval_summary(events_path):
            eval_files_rewritten += 1

    validation_invalid_rows = 0
    if validate_logs_dir is not None:
        validation = validate_answer_failure_root(
            source_root=output_root,
            logs_dir=Path(validate_logs_dir),
            rewrite=True,
        )
        validation_invalid_rows = len(validation["invalid_rows"])

    report_count = 0
    if rebuild_reports:
        reports = build_answer_failure_report(source_root=output_root, logs_dir=validate_logs_dir)
        report_count = len(reports["report_paths"])

    combined_csv = Path("")
    if combine_output_root is not None:
        combined = combine_answer_failures(source_root=output_root, output_root=Path(combine_output_root))
        combined_csv = combined["csv_path"]

    return {
        "output_root": output_root,
        "event_files": len(event_files),
        "event_files_rewritten": event_files_rewritten,
        "events_changed": events_changed,
        "eval_files_rewritten": eval_files_rewritten,
        "validation_invalid_rows": validation_invalid_rows,
        "reports_written": report_count,
        "combined_csv": combined_csv,
    }


def run_one_shot_migration(
    source_root: str | Path = "results_semantic_answer_failures",
    output_root: str | Path = "results_semantic_answer_failures_taxonomy_v3",
    *,
    decisions_path: str | Path = "answer_failure_taxonomy_migration_decisions.csv",
    review_path: str | Path = "answer_failure_taxonomy_migration_review.csv",
    overwrite: bool = False,
    validate_logs_dir: str | Path | None = None,
    rebuild_reports: bool = False,
    combine_output_root: str | Path | None = None,
) -> dict[str, int | Path]:
    source_root = Path(source_root)
    decisions_path = Path(decisions_path)
    review_path = Path(review_path)

    review_outputs = export_review_packet(source_root, review_path)
    decision_outputs = write_auto_decisions(source_root, decisions_path)
    apply_outputs = apply_migration(
        source_root,
        output_root,
        decisions_path=decisions_path,
        overwrite=overwrite,
        validate_logs_dir=validate_logs_dir,
        rebuild_reports=rebuild_reports,
        combine_output_root=combine_output_root,
    )
    return {
        **review_outputs,
        **decision_outputs,
        **apply_outputs,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default="results_semantic_answer_failures", type=Path)
    parser.add_argument("--output-root", default="results_semantic_answer_failures_taxonomy_v3", type=Path)
    parser.add_argument("--decisions-path", default="answer_failure_taxonomy_migration_decisions.csv", type=Path)
    parser.add_argument("--review-path", default="answer_failure_taxonomy_migration_review.csv", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-logs-dir", type=Path)
    parser.add_argument("--rebuild-reports", action="store_true")
    parser.add_argument("--combine-output-root", type=Path)
    subparsers = parser.add_subparsers(dest="command")

    export_parser = subparsers.add_parser("export-review")
    export_parser.add_argument("--source-root", default="results_semantic_answer_failures", type=Path)
    export_parser.add_argument("--review-path", required=True, type=Path)
    export_parser.add_argument("--all-events", action="store_true")

    apply_parser = subparsers.add_parser("apply")
    apply_parser.add_argument("--source-root", default="results_semantic_answer_failures", type=Path)
    apply_parser.add_argument("--output-root", required=True, type=Path)
    apply_parser.add_argument("--decisions-path", type=Path)
    apply_parser.add_argument("--overwrite", action="store_true")
    apply_parser.add_argument("--validate-logs-dir", type=Path)
    apply_parser.add_argument("--rebuild-reports", action="store_true")
    apply_parser.add_argument("--combine-output-root", type=Path)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.command in (None, "run"):
        outputs = run_one_shot_migration(
            args.source_root,
            args.output_root,
            decisions_path=args.decisions_path,
            review_path=args.review_path,
            overwrite=args.overwrite,
            validate_logs_dir=args.validate_logs_dir,
            rebuild_reports=args.rebuild_reports,
            combine_output_root=args.combine_output_root,
        )
    elif args.command == "export-review":
        outputs = export_review_packet(args.source_root, args.review_path, include_all=args.all_events)
    elif args.command == "apply":
        outputs = apply_migration(
            args.source_root,
            args.output_root,
            decisions_path=args.decisions_path,
            overwrite=args.overwrite,
            validate_logs_dir=args.validate_logs_dir,
            rebuild_reports=args.rebuild_reports,
            combine_output_root=args.combine_output_root,
        )
    else:
        raise ValueError(f"Unknown command: {args.command}")

    for key, value in outputs.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
