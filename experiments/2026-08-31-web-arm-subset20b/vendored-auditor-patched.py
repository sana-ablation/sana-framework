#!/usr/bin/env python3
"""Rewrite eval_results.csv files with agent-based semantic judgments."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


SEMANTIC_COLUMNS = [
    "semantic_match",
    "semantic_reason",
    "semantic_bucket",
    "log_error_bucket",
    "log_error_evidence",
]

SEMANTIC_BUCKETS = {
    "semantic_correct",
    "semantic_incorrect",
    "answer_unknown_blank",
}

LOG_ERROR_BUCKETS = {
    "",
    "error_turns_exhausted",
    "error_tools_limit",
    "error_tokens_reached",
    "error_context_overflow",
    "error_event_loop",
    "error_unknown",
}

LOG_SIGNAL_MARKERS = [
    "MaxTokensReachedException",
    "max_tokens limit",
    "Context window overflow",
    "ValidationException",
    "EventLoopException",
    "Tool limit reached",
    "Timeout reached",
    "Search call budget exhausted",
    "Call submit_answer NOW",
    "Answer submitted",
]

RUNTIME_BUCKET_MARKERS = [
    ("error_event_loop", ["eventloopexception"]),
    ("error_context_overflow", ["context window overflow", "validationexception"]),
    ("error_tools_limit", ["tool limit reached"]),
    ("error_turns_exhausted", ["search call budget exhausted", "max turns", "turns exhausted"]),
    (
        "error_tokens_reached",
        [
            "maxtokensreachedexception",
            "max_tokens limit",
            "timeout reached",
            "call submit_answer now",
            "you have already been warned. call submit_answer now",
        ],
    ),
]

RUNTIME_GENERIC_MARKERS = [
    "hard-stopping after second limit trigger",
]

UNKNOWN_ANSWER_MARKERS = {
    "",
    "unknown",
    "[unknown]",
    "n/a",
    "na",
    "none",
    "null",
    "[null]",
    "cannot determine",
    "can't determine",
    "not sure",
    "unsure",
}

FILE_AUDIT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "status": {"type": "string", "enum": ["ready"]},
        "row_count": {"type": "integer", "minimum": 0},
        "notes": {"type": "string"},
    },
    "required": ["status", "row_count", "notes"],
}

ROW_AUDIT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "semantic_match": {"type": "integer", "enum": [0, 1]},
        "semantic_reason": {"type": "string"},
        "semantic_bucket": {
            "type": "string",
            "enum": sorted(SEMANTIC_BUCKETS),
        },
        "log_error_bucket": {
            "type": "string",
            "enum": sorted(LOG_ERROR_BUCKETS),
        },
        "log_error_evidence": {"type": "string"},
    },
    "required": [
        "semantic_match",
        "semantic_reason",
        "semantic_bucket",
        "log_error_bucket",
        "log_error_evidence",
    ],
}


@dataclass(frozen=True)
class FileAuditResult:
    status: str
    row_count: int
    notes: str


@dataclass(frozen=True)
class RowAuditResult:
    semantic_match: int
    semantic_reason: str
    semantic_bucket: str
    log_error_bucket: str
    log_error_evidence: str


class SemanticJudge(Protocol):
    def audit_file(
        self,
        *,
        source_eval_path: Path,
        output_eval_path: Path,
        fieldnames: list[str],
        row_count: int,
    ) -> FileAuditResult: ...

    def audit_row(
        self,
        *,
        source_eval_path: Path,
        row: dict[str, str],
        log_tail: str,
        log_tail_lines_used: int,
        log_required: bool,
    ) -> RowAuditResult: ...


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="results-ec2")
    parser.add_argument("--output", default="")
    parser.add_argument("--logs", default="")
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--file-max-output-tokens", type=int, default=220)
    parser.add_argument("--row-max-output-tokens", type=int, default=320)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--tail-lines", type=int, default=20)
    parser.add_argument("--tail-lines-fallback", type=int, default=50)
    parser.add_argument("--limit-files", type=int, default=0)
    parser.add_argument("--limit-rows", type=int, default=0)
    return parser.parse_args()


def default_output_root(source_root: Path) -> Path:
    return source_root.with_name(f"{source_root.name}_semantic")


def default_logs_root(source_root: Path) -> Path:
    if source_root.name == "sana-results":
        return source_root / "logs"
    if source_root.name.startswith("results"):
        return source_root.with_name(source_root.name.replace("results", "logs", 1))
    return source_root.with_name("logs")


def eval_search_root(source_root: Path) -> Path:
    modes_root = source_root / "modes"
    if modes_root.is_dir():
        return modes_root
    return source_root


def sanitize_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def is_unknown_blank_answer(predicted_answer: str) -> bool:
    lowered = str(predicted_answer or "").strip().lower()
    return lowered in UNKNOWN_ANSWER_MARKERS


def as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def needs_log_review(row: dict[str, str]) -> bool:
    if is_unknown_blank_answer(row.get("predicted_answer", "")):
        return True
    if row.get("error", "").strip():
        return True
    return as_float(row.get("exact_match")) < 1.0


def tail_has_signal(text: str) -> bool:
    lowered = text.lower()
    return any(marker.lower() in lowered for marker in LOG_SIGNAL_MARKERS)


def infer_runtime_bucket(*texts: str) -> str:
    lowered = "\n".join(str(text or "").lower() for text in texts)
    for bucket, markers in RUNTIME_BUCKET_MARKERS:
        if any(marker in lowered for marker in markers):
            return bucket
    return ""


def has_runtime_issue(*texts: str, success_value: str = "") -> bool:
    lowered = "\n".join(str(text or "").lower() for text in texts)
    if infer_runtime_bucket(*texts):
        return True
    if any(marker in lowered for marker in RUNTIME_GENERIC_MARKERS):
        return True
    return str(success_value or "").strip().lower() == "false"


def infer_runtime_evidence(*texts: str) -> str:
    known_markers = [marker for _, bucket_markers in RUNTIME_BUCKET_MARKERS for marker in bucket_markers]
    known_markers.extend(RUNTIME_GENERIC_MARKERS)
    for text in texts:
        if not text:
            continue
        for line in str(text).splitlines():
            lowered = line.lower()
            if any(marker in lowered for marker in known_markers):
                return sanitize_text(line)
    return sanitize_text(next((text for text in texts if str(text or "").strip()), ""))


def read_log_tail_for_review(path: Path, tail_lines: int, fallback_lines: int) -> tuple[str, int]:
    if not path.exists():
        return "", 0
    lines = path.read_text(errors="ignore").splitlines()
    short_tail = "\n".join(lines[-tail_lines:])
    if len(lines) <= tail_lines or tail_has_signal(short_tail):
        return short_tail, min(len(lines), tail_lines)
    long_tail = "\n".join(lines[-fallback_lines:])
    return long_tail, min(len(lines), fallback_lines)


def relative_task_log_path(task_id: str) -> Path:
    task_path = Path(str(task_id))
    parts = list(task_path.parts)
    if parts and parts[0].startswith("tasks"):
        parts = parts[1:]
    relative = Path(*parts) if parts else task_path
    return relative.with_suffix(".log")


def infer_mode_context(source_root: Path, eval_path: Path) -> tuple[str, str]:
    relative = eval_path.relative_to(source_root)
    parts = list(relative.parts)
    if len(parts) < 4 or parts[0] != "modes" or parts[-1] != "eval_results.csv":
        raise ValueError(f"Expected modes/<model>/<variant>/eval_results.csv layout, got {eval_path}")
    return parts[1], parts[2]


def mirrored_output_path(source_root: Path, output_root: Path, eval_path: Path) -> Path:
    return output_root / eval_path.relative_to(source_root)


def ordered_fieldnames(existing_fieldnames: list[str] | None) -> list[str]:
    ordered = list(existing_fieldnames or [])
    for column in SEMANTIC_COLUMNS:
        if column in ordered:
            ordered.remove(column)
    if "exact_match" in ordered:
        idx = ordered.index("exact_match") + 1
        return ordered[:idx] + SEMANTIC_COLUMNS + ordered[idx:]
    return ordered + SEMANTIC_COLUMNS


def maybe_load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def extract_output_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return str(output_text)

    chunks: list[str] = []
    for item in getattr(response, "output", None) or []:
        for part in getattr(item, "content", None) or []:
            text = getattr(part, "text", None)
            if text:
                chunks.append(str(text))
    return "\n".join(chunks)


def validate_row_audit(result: RowAuditResult, *, log_required: bool) -> None:
    if result.semantic_bucket not in SEMANTIC_BUCKETS:
        raise ValueError(f"Unexpected semantic bucket: {result.semantic_bucket}")
    if result.log_error_bucket not in LOG_ERROR_BUCKETS:
        raise ValueError(f"Unexpected log error bucket: {result.log_error_bucket}")
    if result.semantic_bucket == "semantic_correct" and result.semantic_match != 1:
        raise ValueError("semantic_correct rows must set semantic_match=1")
    if result.semantic_bucket in {"semantic_incorrect", "answer_unknown_blank"} and result.semantic_match != 0:
        raise ValueError("Incorrect or blank rows must set semantic_match=0")
    if not result.log_error_bucket and result.log_error_evidence:
        raise ValueError("Rows without a log_error_bucket must leave log_error_evidence blank")


class OpenAIResponsesJudge:
    def __init__(
        self,
        *,
        model: str,
        reasoning_effort: str,
        file_max_output_tokens: int,
        row_max_output_tokens: int,
        max_retries: int,
    ) -> None:
        maybe_load_dotenv()
        from openai import OpenAI

        self.client = OpenAI()
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.file_max_output_tokens = file_max_output_tokens
        self.row_max_output_tokens = row_max_output_tokens
        self.max_retries = max_retries

    def _call_json(
        self,
        *,
        instructions: str,
        prompt: str,
        schema_name: str,
        schema_description: str,
        schema: dict[str, Any],
        max_output_tokens: int,
    ) -> dict[str, Any]:
        last_error: Exception | None = None
        for _ in range(self.max_retries):
            try:
                response = self.client.responses.create(
                    model=self.model,
                    instructions=instructions,
                    input=prompt,
                    reasoning={"effort": self.reasoning_effort},
                    # No temperature: the gpt-5 reasoning models reject it
                    # ("Unsupported parameter"), and `reasoning=` above already
                    # makes this call reasoning-model-only.
                    max_output_tokens=max_output_tokens,
                    text={
                        "verbosity": "low",
                        "format": {
                            "type": "json_schema",
                            "name": schema_name,
                            "description": schema_description,
                            "schema": schema,
                            "strict": True,
                        },
                    },
                    truncation="disabled",
                    store=False,
                )
                return json.loads(extract_output_text(response))
            except Exception as exc:  # pragma: no cover - exercised by live runs
                last_error = exc
        assert last_error is not None
        raise last_error

    def audit_file(
        self,
        *,
        source_eval_path: Path,
        output_eval_path: Path,
        fieldnames: list[str],
        row_count: int,
    ) -> FileAuditResult:
        instructions = (
            "You are auditing one eval_results.csv file before row-by-row semantic labeling. "
            "Acknowledge the rules exactly. You must process every row individually, preserve lexical exact_match, "
            "add semantic fields, inspect log tails for blank or semantically incorrect answers, and not rely on "
            "hardcoded normalization shortcuts as the primary semantic judge."
        )
        prompt = (
            f"Source file: {source_eval_path}\n"
            f"Output file: {output_eval_path}\n"
            f"Columns: {', '.join(fieldnames)}\n"
            f"Row count: {row_count}\n\n"
            "Return ready only if you understand that every row will be audited individually."
        )
        payload = self._call_json(
            instructions=instructions,
            prompt=prompt,
            schema_name="semantic_eval_file_audit",
            schema_description="Confirms file-level semantic audit instructions and row count.",
            schema=FILE_AUDIT_SCHEMA,
            max_output_tokens=self.file_max_output_tokens,
        )
        return FileAuditResult(
            status=str(payload["status"]),
            row_count=int(payload["row_count"]),
            notes=sanitize_text(payload["notes"]),
        )

    def audit_row(
        self,
        *,
        source_eval_path: Path,
        row: dict[str, str],
        log_tail: str,
        log_tail_lines_used: int,
        log_required: bool,
    ) -> RowAuditResult:
        instructions = (
            "You are auditing exactly one evaluation row. Judge semantic equivalence from meaning, not string hacks. "
            "Keep lexical exact_match unchanged in the caller; you only return semantic fields. "
            "If a log tail is provided, you must inspect it before assigning log_error_bucket. "
            "Use only the allowed bucket labels."
        )
        prompt = (
            f"Source file: {source_eval_path}\n"
            f"Task id: {row.get('task_id', '')}\n"
            f"Expected answer: {row.get('expected_answer', row.get('ground_truth', ''))}\n"
            f"Predicted answer: {row.get('predicted_answer', '')}\n"
            f"Lexical exact_match: {row.get('exact_match', '')}\n"
            f"Existing error field: {row.get('error', '')}\n"
            f"Log review required: {'yes' if log_required else 'no'}\n"
            f"Log tail lines checked: {log_tail_lines_used}\n"
            f"Allowed semantic buckets: semantic_correct, semantic_incorrect, answer_unknown_blank\n"
            f"Allowed log error buckets: error_turns_exhausted, error_tools_limit, error_tokens_reached, "
            f"error_context_overflow, error_event_loop, error_unknown\n\n"
            "Rules:\n"
            "- semantic_correct means the predicted answer refers to the same final answer as the expected answer.\n"
            "- semantic_incorrect means the predicted answer is substantive but wrong.\n"
            "- answer_unknown_blank means the prediction is blank, unknown, malformed, or otherwise not an answer.\n"
            "- If the referent is unchanged, omitted administrative suffixes do not make the answer wrong.\n"
            "- Examples of semantic_correct: Erie County vs Erie, Dallas County vs Dallas, O. W. Wilson vs Orlando W. Wilson.\n"
            "- For list answers, preserve set meaning rather than exact formatting; Spokane County, Whitman County vs Spokane, Whitman is semantic_correct.\n"
            "- If log review is required and the row is not semantic_correct, inspect the existing error field and log tail for actual execution failures.\n"
            "- Leave log_error_bucket and log_error_evidence blank when the row is simply wrong but the run still completed without a runtime failure.\n"
            "- Use error_unknown only when there is a real execution failure signal that does not fit the named buckets.\n"
            "- Keep reasons short and concrete.\n\n"
            f"Log tail:\n{log_tail or '<no log tail provided>'}"
        )
        payload = self._call_json(
            instructions=instructions,
            prompt=prompt,
            schema_name="semantic_eval_row_audit",
            schema_description="Structured semantic judgment and log-based failure classification for one eval row.",
            schema=ROW_AUDIT_SCHEMA,
            max_output_tokens=self.row_max_output_tokens,
        )
        result = RowAuditResult(
            semantic_match=int(payload["semantic_match"]),
            semantic_reason=sanitize_text(payload["semantic_reason"]),
            semantic_bucket=str(payload["semantic_bucket"]),
            log_error_bucket=str(payload["log_error_bucket"]),
            log_error_evidence=sanitize_text(payload["log_error_evidence"]),
        )
        validate_row_audit(result, log_required=log_required)
        return result


class SemanticEvalAuditor:
    def __init__(
        self,
        *,
        source_root: Path,
        output_root: Path,
        logs_root: Path,
        judge: SemanticJudge,
        tail_lines: int,
        tail_lines_fallback: int,
        limit_files: int = 0,
        limit_rows: int = 0,
    ) -> None:
        self.source_root = source_root
        self.output_root = output_root
        self.logs_root = logs_root
        self.judge = judge
        self.tail_lines = tail_lines
        self.tail_lines_fallback = tail_lines_fallback
        self.limit_files = limit_files
        self.limit_rows = limit_rows

    def _iter_eval_paths(self) -> list[Path]:
        paths = sorted(eval_search_root(self.source_root).rglob("eval_results.csv"))
        if self.limit_files > 0:
            return paths[: self.limit_files]
        return paths

    def _augment_row(
        self,
        *,
        source_eval_path: Path,
        model: str,
        variant: str,
        row: dict[str, str],
    ) -> dict[str, str]:
        log_required = needs_log_review(row)
        log_tail = ""
        log_tail_lines_used = 0
        if log_required:
            log_path = self.logs_root / "modes" / model / variant / relative_task_log_path(row.get("task_id", ""))
            log_tail, log_tail_lines_used = read_log_tail_for_review(
                log_path,
                tail_lines=self.tail_lines,
                fallback_lines=self.tail_lines_fallback,
            )
            if not log_tail:
                log_tail = f"Missing log file: {log_path}"
        audit = self.judge.audit_row(
            source_eval_path=source_eval_path,
            row=row,
            log_tail=log_tail,
            log_tail_lines_used=log_tail_lines_used,
            log_required=log_required,
        )
        runtime_bucket = infer_runtime_bucket(row.get("error", ""), log_tail)
        runtime_issue = has_runtime_issue(
            row.get("error", ""),
            log_tail,
            success_value=row.get("success", ""),
        )
        log_error_bucket = audit.log_error_bucket
        log_error_evidence = audit.log_error_evidence
        if not runtime_issue:
            log_error_bucket = ""
            log_error_evidence = ""
        elif log_error_bucket in {"", "error_unknown"}:
            log_error_bucket = runtime_bucket or "error_unknown"
            if not log_error_evidence:
                log_error_evidence = infer_runtime_evidence(row.get("error", ""), log_tail)
        enriched = dict(row)
        enriched["semantic_match"] = str(audit.semantic_match)
        enriched["semantic_reason"] = audit.semantic_reason
        enriched["semantic_bucket"] = audit.semantic_bucket
        enriched["log_error_bucket"] = log_error_bucket
        enriched["log_error_evidence"] = log_error_evidence
        return enriched

    def _process_file(self, eval_path: Path) -> tuple[int, Counter]:
        model, variant = infer_mode_context(self.source_root, eval_path)
        output_path = mirrored_output_path(self.source_root, self.output_root, eval_path)
        with eval_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or [])
            rows = list(reader)
        if self.limit_rows > 0:
            rows = rows[: self.limit_rows]
        file_audit = self.judge.audit_file(
            source_eval_path=eval_path,
            output_eval_path=output_path,
            fieldnames=fieldnames,
            row_count=len(rows),
        )
        if file_audit.status != "ready":
            raise ValueError(f"Unexpected file audit status for {eval_path}: {file_audit.status}")
        if file_audit.row_count != len(rows):
            raise ValueError(
                f"File audit row count mismatch for {eval_path}: model saw {file_audit.row_count}, expected {len(rows)}"
            )

        counts: Counter = Counter()
        updated_rows = [
            self._augment_row(source_eval_path=eval_path, model=model, variant=variant, row=row)
            for row in rows
        ]
        for row in updated_rows:
            counts[row["semantic_bucket"]] += 1

        output_path.parent.mkdir(parents=True, exist_ok=True)
        ordered = ordered_fieldnames(fieldnames)
        with output_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=ordered)
            writer.writeheader()
            writer.writerows(updated_rows)
        return len(updated_rows), counts

    def run(self) -> None:
        if not self.source_root.exists():
            raise FileNotFoundError(f"Source directory does not exist: {self.source_root}")
        eval_paths = self._iter_eval_paths()
        if not eval_paths:
            raise FileNotFoundError(f"No eval_results.csv files found under {self.source_root}")
        total_rows = 0
        aggregate_counts: Counter = Counter()
        for idx, eval_path in enumerate(eval_paths, start=1):
            output_path = mirrored_output_path(self.source_root, self.output_root, eval_path)
            print(f"[{idx}/{len(eval_paths)}] Auditing {eval_path} -> {output_path}")
            row_count, counts = self._process_file(eval_path)
            total_rows += row_count
            aggregate_counts.update(counts)
        print(f"Audited {total_rows} rows across {len(eval_paths)} files into {self.output_root}")
        if aggregate_counts:
            print("Semantic buckets:", dict(sorted(aggregate_counts.items())))


def main() -> None:
    args = parse_args()
    source_root = Path(args.source).resolve()
    output_root = Path(args.output).resolve() if args.output else default_output_root(source_root)
    logs_root = Path(args.logs).resolve() if args.logs else default_logs_root(source_root)
    judge = OpenAIResponsesJudge(
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        file_max_output_tokens=args.file_max_output_tokens,
        row_max_output_tokens=args.row_max_output_tokens,
        max_retries=args.max_retries,
    )
    SemanticEvalAuditor(
        source_root=source_root,
        output_root=output_root,
        logs_root=logs_root,
        judge=judge,
        tail_lines=args.tail_lines,
        tail_lines_fallback=args.tail_lines_fallback,
        limit_files=args.limit_files,
        limit_rows=args.limit_rows,
    ).run()


if __name__ == "__main__":
    main()
