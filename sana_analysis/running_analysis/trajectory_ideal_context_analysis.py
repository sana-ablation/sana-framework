#!/usr/bin/env python3
"""Judge each trajectory against task ideal reasoning context."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sana_analysis.running_analysis.trajectory_pair_analysis import (
    _append_journal_record,
    _extract_runner_model,
    _first_executing_line,
    _mode_log_paths,
    _parse_json_object,
    build_repair_prompt,
    call_judge_model,
    extract_trajectory_excerpt,
    resolve_judge_limit,
    write_csv,
)


TARGET_MODES = {
    "nii": "search_i_results_i_plann_computei_k5_skills_off",
    "dii": "search_i_results_i_pland_computei_k5_skills_off",
    "iii": "search_i_results_i_plani_computei_k5_skills_off",
}
BENCHMARK_LOG_ROOTS = {
    "lakeqa": Path("logs"),
    "kramabench": Path("log-kramabench"),
}
TASK_ROOT_TO_PLAN_ROOT = {
    "tasks_mini": "plans_mini",
    "tasks_core": "plans_mini/tasks_core",
    "tasks-mini-kramabench": "plans-mini-kramabench",
}
LABELS = [
    "followed",
    "mostly_followed",
    "partially_followed",
    "missed_crucial_steps",
    "did_not_follow",
    "not_comparable",
]
FOLLOWED_STRICT = {"followed"}
FOLLOWED_BROAD = {"followed", "mostly_followed"}
DEFAULT_OUTPUT_STEM = "trajectory_ideal_context"
DEFAULT_TMP_ROOT = Path("agent_analysis/trajectory_ideal_context_analysis/tmp")
OUTPUT_COLUMNS = [
    "benchmark",
    "mode_label",
    "model_variant",
    "runner_model",
    "task_id",
    "mode",
    "log_path",
    "start_line",
    "trajectory_event_count",
    "run_status",
    "limit_reached",
    "log_max_turn",
    "task_path",
    "plan_path",
    "question",
    "ideal_answer",
    "ideal_reasoning_chain",
    "node_sequence",
    "plan_dataset_sequence",
    "plan_source_sequence",
    "plan_reasoning_text",
    "trajectory_alignment",
    "divergence_type",
    "trajectory_summary",
    "followed_steps",
    "missed_or_divergent_steps",
    "alignment_reason",
    "evidence",
    "auditor_model",
    "audit_status",
]
AUDIT_UPDATE_COLUMNS = [
    "trajectory_alignment",
    "divergence_type",
    "trajectory_summary",
    "followed_steps",
    "missed_or_divergent_steps",
    "alignment_reason",
    "evidence",
    "auditor_model",
    "audit_status",
]


def _run_id_slug(run_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", run_id.strip()).strip("_")


def output_stem(run_id: str = "") -> str:
    slug = _run_id_slug(run_id)
    return f"{DEFAULT_OUTPUT_STEM}_{slug}" if slug else DEFAULT_OUTPUT_STEM


def default_journal_path(output_dir: Path, run_id: str = "") -> Path:
    return output_dir / f"{output_stem(run_id)}_journal.jsonl"


def default_tmp_root(output_dir: Path, run_id: str = "") -> Path:
    slug = _run_id_slug(run_id)
    return output_dir / "tmp" / slug if slug else DEFAULT_TMP_ROOT


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _compact_json_value(value: Any, *, max_chars: int = 1200) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, list):
        return "\n".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return _truncate(str(value), max_chars)


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


TURN_HEADER_RE = re.compile(r"--- Turn (\d+)")


def parse_log_run_status(log_path: Path) -> dict[str, str]:
    max_turn = 0
    saw_answer = False
    saw_tool_limit = False
    saw_timeout = False
    saw_max_tokens = False
    saw_hard_stop = False
    if not log_path.exists():
        return {"run_status": "log_missing", "limit_reached": "false", "log_max_turn": ""}
    try:
        lines = log_path.read_text(errors="replace").splitlines()
    except OSError:
        return {"run_status": "log_unreadable", "limit_reached": "false", "log_max_turn": ""}
    for line in lines:
        turn_match = TURN_HEADER_RE.search(line)
        if turn_match:
            max_turn = max(max_turn, int(turn_match.group(1)))
        saw_answer = saw_answer or "ANSWER:" in line
        saw_tool_limit = saw_tool_limit or "Tool limit reached" in line
        saw_timeout = saw_timeout or "Timeout reached" in line
        saw_max_tokens = saw_max_tokens or "MaxTokensReachedException" in line
        saw_hard_stop = saw_hard_stop or "Hard-stopping after second limit trigger" in line
    if saw_hard_stop and saw_tool_limit:
        run_status = "hard_stopped_tool_limit"
    elif saw_hard_stop and saw_timeout:
        run_status = "hard_stopped_timeout"
    elif saw_tool_limit:
        run_status = "tool_limit_reached"
    elif saw_timeout:
        run_status = "timeout_reached"
    elif saw_max_tokens:
        run_status = "max_tokens_reached"
    elif saw_answer:
        run_status = "answered"
    else:
        run_status = "unknown"
    limit_statuses = {
        "tool_limit_reached",
        "timeout_reached",
        "max_tokens_reached",
        "hard_stopped_tool_limit",
        "hard_stopped_timeout",
    }
    return {
        "run_status": run_status,
        "limit_reached": "true" if run_status in limit_statuses else "false",
        "log_max_turn": str(max_turn) if max_turn else "",
    }


def _plan_path_for_task(repo_root: Path, task_id: str) -> Path:
    parts = Path(task_id).parts
    if not parts:
        return repo_root / "plans_mini" / task_id
    plan_root = TASK_ROOT_TO_PLAN_ROOT.get(parts[0])
    if plan_root:
        return repo_root / plan_root / Path(*parts[1:])
    return repo_root / "plans_mini" / Path(*parts[1:])


def _node_sort_key(node_id: str) -> tuple[int, str]:
    return (int(node_id), node_id) if node_id.isdigit() else (10**9, node_id)


def _format_node_sequence(task_payload: dict[str, Any]) -> str:
    nodes = task_payload.get("nodes")
    if not isinstance(nodes, dict):
        return ""
    lines: list[str] = []
    for node_id, raw_node in sorted(nodes.items(), key=lambda item: _node_sort_key(str(item[0]))):
        if not isinstance(raw_node, dict):
            continue
        source = str(raw_node.get("source", "")).strip()
        subquestion = _truncate(str(raw_node.get("subquestion", "")).strip(), 500)
        answer = _truncate(str(raw_node.get("answer", "")).strip(), 300)
        fact = _truncate(str(raw_node.get("fact", "")).strip(), 700)
        line = f"Node {node_id} | source={source} | subquestion={subquestion} | answer={answer}"
        if fact:
            line += f" | fact={fact}"
        lines.append(line)
    return "\n".join(lines)


def _context_for_task(repo_root: Path, task_id: str) -> tuple[dict[str, str], bool]:
    task_path = repo_root / task_id
    plan_path = _plan_path_for_task(repo_root, task_id)
    task_payload = _load_json(task_path)
    plan_payload = _load_json(plan_path)
    context = {
        "task_path": str(task_path.resolve(strict=False)),
        "plan_path": str(plan_path.resolve(strict=False)) if plan_path.exists() else "",
        "question": "",
        "ideal_answer": "",
        "ideal_reasoning_chain": "",
        "node_sequence": "",
        "plan_dataset_sequence": "",
        "plan_source_sequence": "",
        "plan_reasoning_text": "",
    }
    if task_payload:
        context.update(
            {
                "question": str(task_payload.get("question", "")),
                "ideal_answer": str(task_payload.get("answer", "")),
                "ideal_reasoning_chain": _compact_json_value(task_payload.get("reasoning_chain")),
                "node_sequence": _format_node_sequence(task_payload),
            }
        )
    if plan_payload:
        context.update(
            {
                "plan_path": str(plan_path.resolve(strict=False)),
                "plan_dataset_sequence": _compact_json_value(plan_payload.get("dataset_sequence")),
                "plan_source_sequence": _compact_json_value(plan_payload.get("source_sequence")),
                "plan_reasoning_text": _compact_json_value(
                    plan_payload.get("reasoning_chain_text")
                    or plan_payload.get("original_reasoning_chain")
                ),
            }
        )
    has_context = bool(
        task_payload
        and (context["ideal_reasoning_chain"] or context["node_sequence"])
    )
    return context, has_context


def _prefill_not_comparable(row: dict[str, str], divergence_type: str, reason: str, evidence: str) -> None:
    row.update(
        {
            "trajectory_alignment": "not_comparable",
            "divergence_type": divergence_type,
            "alignment_reason": reason,
            "evidence": evidence,
            "auditor_model": "mechanical",
            "audit_status": "complete",
        }
    )


def build_trajectory_rows(
    *,
    benchmark: str,
    log_root: Path,
    repo_root: Path,
    mode_labels: Iterable[str] = TARGET_MODES.keys(),
    model_filter: str | None = None,
    task_filter: str | None = None,
) -> list[dict[str, str]]:
    paths_by_model_mode = _mode_log_paths(log_root)
    rows: list[dict[str, str]] = []
    for mode_label in mode_labels:
        mode = TARGET_MODES[mode_label]
        models = sorted(model for model, found_mode in paths_by_model_mode if found_mode == mode)
        for model_variant in models:
            if model_filter and model_variant != model_filter:
                continue
            logs = paths_by_model_mode.get((model_variant, mode), {})
            for task_id, log_path in sorted(logs.items()):
                if task_filter and task_filter not in task_id:
                    continue
                start_line = 0
                if mode_label == "dii":
                    start_line = _first_executing_line(log_path, "plan") or 0
                elif mode_label == "iii":
                    start_line = _first_executing_line(log_path, "plan_ideal") or 0
                events = extract_trajectory_excerpt(log_path, start_after_line=start_line)
                context, has_context = _context_for_task(repo_root, task_id)
                run_status = parse_log_run_status(log_path)
                row = {
                    "benchmark": benchmark,
                    "mode_label": mode_label,
                    "model_variant": model_variant,
                    "runner_model": _extract_runner_model(log_path),
                    "task_id": task_id,
                    "mode": mode,
                    "log_path": str(log_path.resolve(strict=False)),
                    "start_line": str(start_line),
                    "trajectory_event_count": str(len(events)),
                    **run_status,
                    **context,
                    "trajectory_alignment": "",
                    "divergence_type": "",
                    "trajectory_summary": "",
                    "followed_steps": "",
                    "missed_or_divergent_steps": "",
                    "alignment_reason": "",
                    "evidence": "",
                    "auditor_model": "",
                    "audit_status": "pending",
                }
                if not has_context:
                    _prefill_not_comparable(
                        row,
                        "task_context_missing",
                        "The task JSON is missing or lacks ideal reasoning/node context.",
                        row["task_path"],
                    )
                elif not events:
                    _prefill_not_comparable(
                        row,
                        "trajectory_missing",
                        "The log has no comparable trajectory events after the start line.",
                        row["log_path"],
                    )
                rows.append(row)
    return rows


def load_existing_audits(path: Path) -> dict[tuple[str, str, str, str], dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="") as handle:
        existing = {}
        for row in csv.DictReader(handle):
            key = (
                row.get("benchmark", ""),
                row.get("mode_label", ""),
                row.get("model_variant", ""),
                row.get("task_id", ""),
            )
            existing[key] = dict(row)
        return existing


def merge_existing_audits(rows: list[dict[str, str]], existing: dict[tuple[str, str, str, str], dict[str, str]]) -> None:
    for row in rows:
        key = (row["benchmark"], row["mode_label"], row["model_variant"], row["task_id"])
        old = existing.get(key)
        if old and old.get("audit_status") not in ("", "pending"):
            for column in AUDIT_UPDATE_COLUMNS:
                row[column] = old.get(column, row.get(column, ""))


def load_tmp_audits(tmp_root: Path) -> dict[tuple[str, str, str, str], dict[str, str]]:
    existing: dict[tuple[str, str, str, str], dict[str, str]] = {}
    if not tmp_root.exists():
        return existing
    for path in sorted(tmp_root.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        row_key = payload.get("row_key")
        row_update = payload.get("row_update")
        if not isinstance(row_key, dict) or not isinstance(row_update, dict):
            continue
        key = (
            str(row_key.get("benchmark", "")),
            str(row_key.get("mode_label", "")),
            str(row_key.get("model_variant", "")),
            str(row_key.get("task_id", "")),
        )
        if not all(key):
            continue
        existing[key] = {column: str(row_update.get(column, "")) for column in AUDIT_UPDATE_COLUMNS}
    return existing


def _pct(numerator: int, denominator: int) -> float:
    return round(100.0 * numerator / denominator, 1) if denominator else 0.0


def _fraction(numerator: int, denominator: int) -> str:
    return f"{numerator}/{denominator}" if denominator else "0/0"


def summarize_rows(rows: Iterable[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["benchmark"], row["model_variant"], row["mode_label"])].append(row)

    summary_rows: list[dict[str, Any]] = []
    breakdown_rows: list[dict[str, Any]] = []
    for (benchmark, model_variant, mode_label), group_rows in sorted(grouped.items()):
        labels = Counter(row.get("trajectory_alignment", "") or "pending" for row in group_rows)
        statuses = Counter(row.get("audit_status", "") or "pending" for row in group_rows)
        n_total = len(group_rows)
        n_not_comparable = labels.get("not_comparable", 0)
        n_comparable = n_total - n_not_comparable
        n_strict = sum(labels.get(label, 0) for label in FOLLOWED_STRICT)
        n_broad = sum(labels.get(label, 0) for label in FOLLOWED_BROAD)
        row = {
            "benchmark": benchmark,
            "model_variant": model_variant,
            "mode_label": mode_label,
            "mode": TARGET_MODES[mode_label],
            "n_total": n_total,
            "n_complete": statuses.get("complete", 0),
            "n_pending": statuses.get("pending", 0),
            "n_comparable": n_comparable,
            "n_not_comparable": n_not_comparable,
            "n_followed_strict": n_strict,
            "followed_strict_fraction_all": _fraction(n_strict, n_total),
            "followed_strict_pct_all": _pct(n_strict, n_total),
            "followed_strict_fraction_comparable": _fraction(n_strict, n_comparable),
            "followed_strict_pct_comparable": _pct(n_strict, n_comparable),
            "n_followed_or_mostly": n_broad,
            "followed_or_mostly_fraction_all": _fraction(n_broad, n_total),
            "followed_or_mostly_pct_all": _pct(n_broad, n_total),
            "followed_or_mostly_fraction_comparable": _fraction(n_broad, n_comparable),
            "followed_or_mostly_pct_comparable": _pct(n_broad, n_comparable),
        }
        for label in LABELS:
            row[f"n_{label}"] = labels.get(label, 0)
            row[f"{label}_fraction_all"] = _fraction(labels.get(label, 0), n_total)
            row[f"{label}_pct_all"] = _pct(labels.get(label, 0), n_total)
        row["n_pending_label"] = labels.get("pending", 0)
        row["pending_fraction_all"] = _fraction(labels.get("pending", 0), n_total)
        summary_rows.append(row)

        for label, count in sorted(labels.items()):
            breakdown_rows.append(
                {
                    "benchmark": benchmark,
                    "model_variant": model_variant,
                    "mode_label": mode_label,
                    "mode": TARGET_MODES[mode_label],
                    "trajectory_alignment": label,
                    "n": count,
                    "fraction_all": _fraction(count, n_total),
                    "pct_all": _pct(count, n_total),
                    "fraction_comparable": _fraction(count, n_comparable) if label != "not_comparable" else "",
                    "pct_comparable": _pct(count, n_comparable) if label != "not_comparable" else "",
                }
            )
    return summary_rows, breakdown_rows


def write_outputs(output_dir: Path, rows: list[dict[str, str]], *, run_id: str = "") -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_stem(run_id)
    audit_path = output_dir / f"{stem}_audit.csv"
    summary_path = output_dir / f"{stem}_summary.csv"
    breakdown_path = output_dir / f"{stem}_label_breakdown.csv"
    json_path = output_dir / f"{stem}_summary.json"
    summary_rows, breakdown_rows = summarize_rows(rows)
    write_csv(audit_path, rows, OUTPUT_COLUMNS)
    write_csv(summary_path, summary_rows)
    write_csv(breakdown_path, breakdown_rows)
    json_path.write_text(
        json.dumps(
            {
                "summary": summary_rows,
                "label_breakdown": breakdown_rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return [audit_path, summary_path, breakdown_path, json_path]


def _row_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (row["benchmark"], row["mode_label"], row["model_variant"], row["task_id"])


def _row_slug(row: dict[str, str]) -> str:
    benchmark, mode_label, model_variant, task_id = _row_key(row)
    task_stem = Path(task_id).with_suffix("").as_posix()
    raw = "__".join([benchmark, mode_label, model_variant, task_stem])
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("_") or "row"


def _audit_json_path(tmp_root: Path, row: dict[str, str]) -> Path:
    return tmp_root / row["benchmark"] / row["model_variant"] / row["mode_label"] / f"{_row_slug(row)}.json"


def validate_judge_payload(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    label = payload.get("trajectory_alignment")
    if label not in LABELS:
        errors.append(f"trajectory_alignment must be one of {LABELS}, got {label!r}")
    status = payload.get("audit_status")
    if status != "complete":
        errors.append("audit_status must be 'complete'")
    for field in [
        "divergence_type",
        "trajectory_summary",
        "alignment_reason",
        "evidence",
    ]:
        if not isinstance(payload.get(field), str) or not str(payload.get(field)).strip():
            errors.append(f"{field} must be a non-empty string")
    for field in ["followed_steps", "missed_or_divergent_steps"]:
        if not isinstance(payload.get(field), str):
            errors.append(f"{field} must be a string")
    return errors


def build_judge_prompt(row: dict[str, str]) -> str:
    events = extract_trajectory_excerpt(
        Path(row["log_path"]),
        start_after_line=int(row.get("start_line") or 0),
    )
    return f"""You are judging one actual agent trajectory against the task's ideal reasoning context.

Decide whether the trajectory follows the ideal reasoning chain, ideal reasoning text, and node sequence.
Do not compare this trajectory to any other executed trajectory. The task context is the reference.
Credit semantically equivalent paths that use the right sources, dependencies, filters, operations, and finalization.
Penalize skipped nodes, wrong source families, missing dependency handoffs, wrong computations, or answers reached without the required reasoning.

Labels:
- followed: all key ideal nodes/hops and final computation are followed; only trivial retries or harmless ordering differences.
- mostly_followed: same answer-producing reasoning path with minor detours, repairs, or small omissions that do not change the core chain.
- partially_followed: several major ideal steps are present, but at least one meaningful source, filter, operation, hop, or finalization step is changed or missing.
- missed_crucial_steps: the trajectory is related but skips one or more crucial nodes/hops/computations needed by the ideal chain.
- did_not_follow: materially different reasoning path, source family, or computation.
- not_comparable: insufficient log/context, malformed trajectory, or impossible to judge from the provided evidence.

Return JSON only:
{{
  "trajectory_alignment": "followed|mostly_followed|partially_followed|missed_crucial_steps|did_not_follow|not_comparable",
  "divergence_type": "none|minor_detour|source_mismatch|filter_mismatch|operation_mismatch|hop_skip|output_mismatch|trajectory_missing|other",
  "trajectory_summary": "one sentence",
  "followed_steps": "semicolon-separated ideal steps that the trajectory followed",
  "missed_or_divergent_steps": "semicolon-separated ideal steps that were missed or diverged",
  "alignment_reason": "short evidence-grounded reason",
  "evidence": "cite log path and relevant line numbers from the excerpt, plus task/plan context where useful",
  "audit_status": "complete"
}}

Metadata:
- benchmark: {row['benchmark']}
- task_id: {row['task_id']}
- model_variant: {row['model_variant']}
- runner_model: {row['runner_model']}
- mode_label: {row['mode_label']}
- mode: {row['mode']}
- log_path: {row['log_path']}
- run_status: {row.get('run_status', '')}
- limit_reached: {row.get('limit_reached', '')}
- log_max_turn: {row.get('log_max_turn', '')}
- task_path: {row['task_path']}
- plan_path: {row['plan_path']}

RUN STATUS:
run_status={row.get('run_status', '')}; limit_reached={row.get('limit_reached', '')}; log_max_turn={row.get('log_max_turn', '')}

QUESTION:
{row['question']}

IDEAL ANSWER:
{row['ideal_answer']}

IDEAL REASONING CHAIN:
{row['ideal_reasoning_chain']}

NODE SEQUENCE:
{row['node_sequence']}

PLAN CONTEXT:
dataset_sequence:
{row['plan_dataset_sequence']}

source_sequence:
{row['plan_source_sequence']}

reasoning_chain_text:
{row['plan_reasoning_text']}

ACTUAL TRAJECTORY:
{chr(10).join(events)}
"""


def judge_pending_rows(
    rows: list[dict[str, str]],
    *,
    backend: str,
    output_dir: Path,
    repo_root: Path,
    model: str,
    reasoning_effort: str,
    limit: int,
    timeout: int,
    tmp_root: Path,
    journal_path: Path,
    max_retries: int,
    run_id: str = "",
) -> int:
    judged = 0
    for row in rows:
        if row.get("audit_status") != "pending":
            continue
        if limit and judged >= limit:
            break
        print(
            "Judging trajectory ideal-context "
            f"#{judged + 1}: benchmark={row['benchmark']} model={row['model_variant']} "
            f"mode={row['mode_label']} task={row['task_id']}",
            flush=True,
        )
        prompt = build_judge_prompt(row)
        out_path = _audit_json_path(tmp_root, row)
        stem = out_path.with_suffix("")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_path = stem.with_suffix(".prompt.txt")
        last_message_path = stem.with_suffix(".last_message.txt")
        stdout_path = stem.with_suffix(f".{backend}_stdout.log")
        attempt_prompt = prompt
        parsed: dict[str, Any] | None = None
        text = ""
        validation_errors: list[str] = []
        for attempt in range(1, max_retries + 2):
            prompt_path.write_text(attempt_prompt, encoding="utf-8")
            stem.with_suffix(f".attempt{attempt}.prompt.txt").write_text(attempt_prompt, encoding="utf-8")
            print(
                f"  attempt {attempt}: spawning {backend} judge for task={row['task_id']} mode={row['mode_label']}",
                flush=True,
            )
            try:
                text = call_judge_model(
                    attempt_prompt,
                    backend=backend,
                    repo_root=repo_root,
                    model=model,
                    reasoning_effort=reasoning_effort,
                    last_message_path=last_message_path,
                    stdout_path=stdout_path,
                    timeout=timeout,
                )
            except Exception as exc:
                _append_journal_record(
                    journal_path,
                    {
                        "status": "error",
                        "error": str(exc),
                        "attempt": attempt,
                        "benchmark": row["benchmark"],
                        "mode_label": row["mode_label"],
                        "model_variant": row["model_variant"],
                        "task_id": row["task_id"],
                        "prompt_path": str(prompt_path),
                        "stdout_path": str(stdout_path),
                    },
                )
                raise
            stem.with_suffix(f".attempt{attempt}.last_message.txt").write_text(text, encoding="utf-8")
            try:
                parsed = _parse_json_object(text)
            except (json.JSONDecodeError, ValueError) as exc:
                validation_errors = [f"response must be a JSON object: {exc}"]
            else:
                validation_errors = validate_judge_payload(parsed)
            if not validation_errors:
                break
            _append_journal_record(
                journal_path,
                {
                    "status": "retry",
                    "attempt": attempt,
                    "errors": validation_errors,
                    "benchmark": row["benchmark"],
                    "mode_label": row["mode_label"],
                    "model_variant": row["model_variant"],
                    "task_id": row["task_id"],
                    "prompt_path": str(prompt_path),
                    "last_message_path": str(last_message_path),
                    "stdout_path": str(stdout_path),
                },
            )
            if attempt > max_retries:
                raise RuntimeError(
                    "Trajectory ideal-context judge response failed validation after retries: "
                    + "; ".join(validation_errors)
                )
            attempt_prompt = build_repair_prompt(prompt, text, validation_errors)
        assert parsed is not None
        label = str(parsed.get("trajectory_alignment", "not_comparable"))
        row.update(
            {
                "trajectory_alignment": label,
                "divergence_type": str(parsed.get("divergence_type", "other")),
                "trajectory_summary": str(parsed.get("trajectory_summary", "")),
                "followed_steps": str(parsed.get("followed_steps", "")),
                "missed_or_divergent_steps": str(parsed.get("missed_or_divergent_steps", "")),
                "alignment_reason": str(parsed.get("alignment_reason", "")),
                "evidence": str(parsed.get("evidence", "")),
                "auditor_model": model,
                "audit_status": str(parsed.get("audit_status", "complete") or "complete"),
            }
        )
        audit_payload = {
            "row_key": {
                "benchmark": row["benchmark"],
                "mode_label": row["mode_label"],
                "model_variant": row["model_variant"],
                "task_id": row["task_id"],
            },
            "backend": backend,
            "judge_model": model,
            "reasoning_effort": reasoning_effort,
            "prompt_path": str(prompt_path),
            "last_message_path": str(last_message_path),
            "stdout_path": str(stdout_path),
            "parsed": parsed,
            "row_update": {key: row[key] for key in AUDIT_UPDATE_COLUMNS},
        }
        out_path.write_text(json.dumps(audit_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        _append_journal_record(
            journal_path,
            {
                "status": "ok",
                "benchmark": row["benchmark"],
                "mode_label": row["mode_label"],
                "model_variant": row["model_variant"],
                "task_id": row["task_id"],
                "label": label,
                "audit_json_path": str(out_path),
                "prompt_path": str(prompt_path),
                "last_message_path": str(last_message_path),
                "stdout_path": str(stdout_path),
            },
        )
        judged += 1
        write_outputs(output_dir, rows, run_id=run_id)
        time.sleep(0.1)
    return judged


def build_all_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    benchmarks = list(BENCHMARK_LOG_ROOTS) if args.benchmark == "all" else [args.benchmark]
    mode_labels = list(TARGET_MODES) if args.mode == "all" else [args.mode]
    rows: list[dict[str, str]] = []
    for benchmark in benchmarks:
        repo_root = Path(args.input_root)
        log_root = repo_root / BENCHMARK_LOG_ROOTS[benchmark]
        rows.extend(
            build_trajectory_rows(
                benchmark=benchmark,
                log_root=log_root,
                repo_root=repo_root,
                mode_labels=mode_labels,
                model_filter=args.model or None,
                task_filter=args.task or None,
            )
        )
    return rows


def row_matches_filters(row: dict[str, str], args: argparse.Namespace) -> bool:
    if args.benchmark != "all" and row["benchmark"] != args.benchmark:
        return False
    if args.mode != "all" and row["mode_label"] != args.mode:
        return False
    if args.model and row["model_variant"] != args.model:
        return False
    if args.task and args.task not in row["task_id"]:
        return False
    return True


def full_build_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        **{
            **vars(args),
            "benchmark": "all",
            "mode": "all",
            "model": "",
            "task": "",
        }
    )


def print_summary(rows: list[dict[str, str]]) -> None:
    summary_rows, _ = summarize_rows(rows)
    print("Trajectory ideal-context summary:")
    for row in summary_rows:
        print(
            "  "
            f"{row['benchmark']:<11} {row['model_variant']:<18} {row['mode_label']:<3} "
            f"followed+mostly={row['followed_or_mostly_pct_all']:>5.1f}% "
            f"({row['n_followed_or_mostly']}/{row['n_total']}), "
            f"strict={row['followed_strict_pct_all']:>5.1f}% "
            f"({row['n_followed_strict']}/{row['n_total']}), "
            f"pending={row['n_pending']} n/c={row['n_not_comparable']}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default=".", help="Repository/input root containing logs and task JSON.")
    parser.add_argument("--output-dir", default="agent_analysis/trajectory_ideal_context_analysis")
    parser.add_argument(
        "--run-id",
        default="",
        help=(
            "Optional run id used to suffix output files and the default judge temp directory, "
            "e.g. --run-id run2 writes trajectory_ideal_context_run2_*.csv."
        ),
    )
    parser.add_argument("--benchmark", choices=["all", *BENCHMARK_LOG_ROOTS.keys()], default="all")
    parser.add_argument("--mode", choices=["all", *TARGET_MODES.keys()], default="all")
    parser.add_argument("--model", default="", help="Optional model folder, e.g. openai_gpt-5-mini.")
    parser.add_argument("--task", default="", help="Optional task-id substring filter.")
    parser.add_argument("--judge", action="store_true", help="Judge pending trajectories.")
    parser.add_argument("--backend", choices=["codex", "openai"], default="codex", help="Judge backend; defaults to codex exec.")
    parser.add_argument("--judge-model", default="gpt-5.4-mini")
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max pending rows to judge. Omit for an interactive prompt; 0 means all.",
    )
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-retries", type=int, default=2, help="Retry invalid judge JSON this many times.")
    parser.add_argument(
        "--tmp-root",
        default="",
        help=(
            "Directory for per-row prompt, raw response, and parsed judge JSON files. "
            "Defaults to the legacy temp dir without --run-id, or <output-dir>/tmp/<run-id> with --run-id."
        ),
    )
    parser.add_argument(
        "--journal-path",
        default="",
        help=(
            "Optional JSONL journal path. Defaults to <output-dir>/trajectory_ideal_context_journal.jsonl "
            "without --run-id, or <output-dir>/trajectory_ideal_context_<run-id>_journal.jsonl with --run-id."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    audit_path = output_dir / f"{output_stem(args.run_id)}_audit.csv"
    tmp_root = Path(args.tmp_root).resolve() if args.tmp_root else default_tmp_root(output_dir, args.run_id).resolve()
    rows = build_all_rows(full_build_args(args))
    merge_existing_audits(rows, load_existing_audits(audit_path))
    merge_existing_audits(rows, load_tmp_audits(tmp_root))
    written = write_outputs(output_dir, rows, run_id=args.run_id)
    if args.judge:
        journal_path = (
            Path(args.journal_path).resolve()
            if args.journal_path
            else default_journal_path(output_dir, args.run_id)
        )
        print(f"Temp judge dir: {tmp_root}")
        print(f"JSONL journal: {journal_path}")
        print(f"Judge backend: {args.backend}")
        rows_to_judge = [row for row in rows if row_matches_filters(row, args)]
        pending_count = sum(1 for row in rows_to_judge if row.get("audit_status") == "pending")
        judge_limit = resolve_judge_limit(
            requested_limit=args.limit,
            pending_count=pending_count,
            is_interactive=sys.stdin.isatty(),
        )
        if judge_limit is None:
            print("Aborted without judging rows.")
            print_summary(rows)
            for path in written:
                print(f"Wrote {path}")
            return 0
        judged = judge_pending_rows(
            rows_to_judge,
            backend=args.backend,
            output_dir=output_dir,
            repo_root=Path(args.input_root).resolve(),
            model=args.judge_model,
            reasoning_effort=args.reasoning_effort,
            limit=judge_limit,
            timeout=args.timeout,
            tmp_root=tmp_root,
            journal_path=journal_path,
            max_retries=args.max_retries,
            run_id=args.run_id,
        )
        print(f"Judged {judged} pending row(s).")
        rows = build_all_rows(full_build_args(args))
        merge_existing_audits(rows, load_existing_audits(audit_path))
        merge_existing_audits(rows, load_tmp_audits(tmp_root))
        written = write_outputs(output_dir, rows, run_id=args.run_id)
    print_summary(rows)
    for path in written:
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
