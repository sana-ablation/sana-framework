#!/usr/bin/env python3
"""Prepare, judge, and summarize paired trajectory similarity audits."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


IDEAL_MODE = "search_i_results_i_plani_computei_k5_skills_off"
PAIR_MODES = {
    "nii_vs_iii": "search_i_results_i_plann_computei_k5_skills_off",
    "dii_vs_iii": "search_i_results_i_pland_computei_k5_skills_off",
}
BENCHMARK_LOG_ROOTS = {
    "lakeqa": Path("logs"),
    "kramabench": Path("log-kramabench"),
}
LABELS = [
    "resembles_completely",
    "closely_resembles",
    "partially_resembles",
    "misses_crucial_steps",
    "not_similar",
    "not_comparable",
]
GOOD_LABELS = {"resembles_completely", "closely_resembles"}
PARTIAL_OR_BETTER_LABELS = GOOD_LABELS | {"partially_resembles"}
INTERNAL_TOOLS = {
    "pick",
    "select_match",
    "record_intent_match",
    "submit_repaired_sql",
    "submit_repaired_code",
}
OUTPUT_COLUMNS = [
    "benchmark",
    "pair_label",
    "model_variant",
    "runner_model",
    "task_id",
    "comparison_mode",
    "ideal_mode",
    "comparison_log",
    "ideal_log",
    "comparison_start_line",
    "ideal_start_line",
    "comparison_event_count",
    "ideal_event_count",
    "trajectory_similarity",
    "divergence_type",
    "comparison_trajectory_summary",
    "ideal_trajectory_summary",
    "aligned_steps",
    "divergent_steps",
    "similarity_reason",
    "evidence",
    "auditor_model",
    "audit_status",
]
AUDIT_UPDATE_COLUMNS = [
    "trajectory_similarity",
    "divergence_type",
    "comparison_trajectory_summary",
    "ideal_trajectory_summary",
    "aligned_steps",
    "divergent_steps",
    "similarity_reason",
    "evidence",
    "auditor_model",
    "audit_status",
]


RUNNER_MODEL_RE = re.compile(r"\bNEW TASK:\s*([A-Za-z0-9_.-]+)")
EXECUTING_RE = re.compile(r"Executing:\s*([A-Za-z_][A-Za-z0-9_]*)\(")


def _mode_log_paths(log_root: Path) -> dict[tuple[str, str], dict[str, Path]]:
    modes_root = log_root / "modes"
    out: dict[tuple[str, str], dict[str, Path]] = defaultdict(dict)
    if not modes_root.is_dir():
        return out
    for log_path in sorted(modes_root.glob("*/*/**/*.log")):
        rel_parts = log_path.relative_to(modes_root).parts
        if len(rel_parts) < 4:
            continue
        model_variant, mode = rel_parts[0], rel_parts[1]
        task_id = Path(*rel_parts[2:]).with_suffix(".json").as_posix()
        out[(model_variant, mode)][task_id] = log_path.resolve(strict=False)
    return out


def _extract_runner_model(log_path: Path) -> str:
    if not log_path.exists():
        return ""
    try:
        lines = log_path.read_text(errors="replace").splitlines()
    except OSError:
        return ""
    for line in lines:
        match = RUNNER_MODEL_RE.search(line)
        if match:
            return match.group(1)
    return ""


def _first_executing_line(log_path: Path, tool_name: str) -> int | None:
    if not log_path.exists():
        return None
    try:
        lines = log_path.read_text(errors="replace").splitlines()
    except OSError:
        return None
    marker = f"Executing: {tool_name}("
    for line_number, line in enumerate(lines, start=1):
        if marker in line:
            return line_number
    return None


def _strip_log_prefix(line: str) -> str:
    if " | " in line:
        parts = line.split(" | ", 3)
        if len(parts) == 4:
            line = parts[3]
    return " ".join(line.strip().split())


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def extract_trajectory_excerpt(
    log_path: Path,
    *,
    start_after_line: int | None = None,
    max_events: int = 80,
    max_chars_per_event: int = 700,
) -> list[str]:
    if not log_path.exists():
        return []
    try:
        lines = log_path.read_text(errors="replace").splitlines()
    except OSError:
        return []

    events: list[str] = []
    last_kept_tool: str | None = None
    start_line = start_after_line or 0
    for line_number, line in enumerate(lines, start=1):
        if line_number <= start_line:
            continue
        if "Executing:" in line:
            match = EXECUTING_RE.search(line)
            tool = match.group(1) if match else ""
            if tool in INTERNAL_TOOLS:
                last_kept_tool = None
                continue
            last_kept_tool = tool or None
            events.append(f"L{line_number}: {_truncate(_strip_log_prefix(line), max_chars_per_event)}")
        elif "Tool result:" in line and last_kept_tool:
            events.append(f"L{line_number}: {_truncate(_strip_log_prefix(line), max_chars_per_event)}")
            last_kept_tool = None
        elif "ANSWER:" in line:
            events.append(f"L{line_number}: {_truncate(_strip_log_prefix(line), max_chars_per_event)}")
        if len(events) >= max_events:
            break
    return events


def _prefill_not_comparable(row: dict[str, str], divergence_type: str, reason: str, evidence: str) -> None:
    row.update(
        {
            "trajectory_similarity": "not_comparable",
            "divergence_type": divergence_type,
            "similarity_reason": reason,
            "evidence": evidence,
            "auditor_model": "mechanical",
            "audit_status": "complete",
        }
    )


def build_pair_rows(
    *,
    benchmark: str,
    log_root: Path,
    pair_labels: Iterable[str] = PAIR_MODES.keys(),
    model_filter: str | None = None,
    task_filter: str | None = None,
) -> list[dict[str, str]]:
    paths_by_model_mode = _mode_log_paths(log_root)
    rows: list[dict[str, str]] = []
    for pair_label in pair_labels:
        comparison_mode = PAIR_MODES[pair_label]
        models = sorted(
            {
                model
                for model, mode in paths_by_model_mode
                if mode in {comparison_mode, IDEAL_MODE}
            }
        )
        for model_variant in models:
            if model_filter and model_variant != model_filter:
                continue
            comparison_logs = paths_by_model_mode.get((model_variant, comparison_mode), {})
            ideal_logs = paths_by_model_mode.get((model_variant, IDEAL_MODE), {})
            task_ids = sorted(set(comparison_logs) | set(ideal_logs))
            for task_id in task_ids:
                if task_filter and task_filter not in task_id:
                    continue
                comparison_log = comparison_logs.get(task_id)
                ideal_log = ideal_logs.get(task_id)
                comparison_log_str = str(comparison_log.resolve(strict=False)) if comparison_log else ""
                ideal_log_str = str(ideal_log.resolve(strict=False)) if ideal_log else ""
                comparison_plan_line = (
                    _first_executing_line(comparison_log, "plan")
                    if comparison_log and pair_label == "dii_vs_iii"
                    else None
                )
                ideal_plan_line = _first_executing_line(ideal_log, "plan_ideal") if ideal_log else None
                comparison_start_line = comparison_plan_line or 0
                ideal_start_line = ideal_plan_line or 0
                comparison_events = (
                    extract_trajectory_excerpt(comparison_log, start_after_line=comparison_start_line)
                    if comparison_log
                    else []
                )
                ideal_events = (
                    extract_trajectory_excerpt(ideal_log, start_after_line=ideal_start_line)
                    if ideal_log
                    else []
                )
                comparison_model = _extract_runner_model(comparison_log) if comparison_log else ""
                ideal_model = _extract_runner_model(ideal_log) if ideal_log else ""
                runner_model = comparison_model or ideal_model
                if comparison_model and ideal_model and comparison_model != ideal_model:
                    runner_model = f"{comparison_model}|{ideal_model}"

                row = {
                    "benchmark": benchmark,
                    "pair_label": pair_label,
                    "model_variant": model_variant,
                    "runner_model": runner_model,
                    "task_id": task_id,
                    "comparison_mode": comparison_mode,
                    "ideal_mode": IDEAL_MODE,
                    "comparison_log": comparison_log_str,
                    "ideal_log": ideal_log_str,
                    "comparison_start_line": str(comparison_start_line),
                    "ideal_start_line": str(ideal_start_line),
                    "comparison_event_count": str(len(comparison_events)),
                    "ideal_event_count": str(len(ideal_events)),
                    "trajectory_similarity": "",
                    "divergence_type": "",
                    "comparison_trajectory_summary": "",
                    "ideal_trajectory_summary": "",
                    "aligned_steps": "",
                    "divergent_steps": "",
                    "similarity_reason": "",
                    "evidence": "",
                    "auditor_model": "",
                    "audit_status": "pending",
                }

                if not comparison_log:
                    _prefill_not_comparable(
                        row,
                        "comparison_log_missing",
                        "The comparison trajectory log is missing.",
                        ideal_log_str,
                    )
                elif not ideal_log:
                    _prefill_not_comparable(
                        row,
                        "ideal_log_missing",
                        "The ideal trajectory log is missing.",
                        comparison_log_str,
                    )
                elif comparison_model and ideal_model and comparison_model != ideal_model:
                    _prefill_not_comparable(
                        row,
                        "model_mismatch",
                        f"Runner model mismatch: comparison log has {comparison_model}, ideal log has {ideal_model}.",
                        f"{comparison_log_str}; {ideal_log_str}",
                    )
                elif not ideal_plan_line:
                    _prefill_not_comparable(
                        row,
                        "ideal_plan_missing",
                        "No explicit plan_ideal() call was found in the ideal trajectory log.",
                        ideal_log_str,
                    )
                elif pair_label == "dii_vs_iii" and not comparison_plan_line:
                    _prefill_not_comparable(
                        row,
                        "comparison_plan_missing",
                        "No explicit plan() call was found in the default-plan comparison log.",
                        comparison_log_str,
                    )
                elif not comparison_events or not ideal_events:
                    _prefill_not_comparable(
                        row,
                        "trajectory_missing",
                        "One side has no comparable post-plan tool trajectory events.",
                        f"{comparison_log_str}; {ideal_log_str}",
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
                row.get("pair_label", ""),
                row.get("model_variant", ""),
                row.get("task_id", ""),
            )
            existing[key] = dict(row)
        return existing


def merge_existing_audits(rows: list[dict[str, str]], existing: dict[tuple[str, str, str, str], dict[str, str]]) -> None:
    for row in rows:
        key = (row["benchmark"], row["pair_label"], row["model_variant"], row["task_id"])
        old = existing.get(key)
        if old and old.get("audit_status") not in ("", "pending"):
            for column in AUDIT_UPDATE_COLUMNS:
                row[column] = old.get(column, row.get(column, ""))


def load_tmp_audits(tmp_root: Path) -> dict[tuple[str, str, str, str], dict[str, str]]:
    """Load durable per-row judge JSON outputs as the source of truth."""
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
            str(row_key.get("pair_label", "")),
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
        grouped[(row["benchmark"], row["model_variant"], row["pair_label"])].append(row)

    summary_rows: list[dict[str, Any]] = []
    breakdown_rows: list[dict[str, Any]] = []
    for (benchmark, model_variant, pair_label), group_rows in sorted(grouped.items()):
        labels = Counter(row.get("trajectory_similarity", "") or "pending" for row in group_rows)
        statuses = Counter(row.get("audit_status", "") or "pending" for row in group_rows)
        n_total = len(group_rows)
        n_not_comparable = labels.get("not_comparable", 0)
        n_comparable = n_total - n_not_comparable
        n_good = sum(labels.get(label, 0) for label in GOOD_LABELS)
        n_partial_or_better = sum(labels.get(label, 0) for label in PARTIAL_OR_BETTER_LABELS)
        row = {
            "benchmark": benchmark,
            "model_variant": model_variant,
            "pair_label": pair_label,
            "comparison_mode": PAIR_MODES[pair_label],
            "ideal_mode": IDEAL_MODE,
            "n_total": n_total,
            "n_complete": statuses.get("complete", 0),
            "n_pending": statuses.get("pending", 0),
            "n_comparable": n_comparable,
            "n_not_comparable": n_not_comparable,
            "n_resembles_or_close": n_good,
            "resembles_or_close_fraction_all": _fraction(n_good, n_total),
            "resembles_or_close_pct_all": _pct(n_good, n_total),
            "resembles_or_close_fraction_comparable": _fraction(n_good, n_comparable),
            "resembles_or_close_pct_comparable": _pct(n_good, n_comparable),
            "n_partial_or_better": n_partial_or_better,
            "partial_or_better_fraction_all": _fraction(n_partial_or_better, n_total),
            "partial_or_better_pct_all": _pct(n_partial_or_better, n_total),
            "partial_or_better_fraction_comparable": _fraction(n_partial_or_better, n_comparable),
            "partial_or_better_pct_comparable": _pct(n_partial_or_better, n_comparable),
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
                    "pair_label": pair_label,
                    "trajectory_similarity": label,
                    "n": count,
                    "fraction_all": _fraction(count, n_total),
                    "pct_all": _pct(count, n_total),
                    "fraction_comparable": _fraction(count, n_comparable) if label != "not_comparable" else "",
                    "pct_comparable": _pct(count, n_comparable) if label != "not_comparable" else "",
                }
            )
    return summary_rows, breakdown_rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(output_dir: Path, rows: list[dict[str, str]]) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = output_dir / "trajectory_pair_audit.csv"
    summary_path = output_dir / "trajectory_pair_summary.csv"
    breakdown_path = output_dir / "trajectory_pair_label_breakdown.csv"
    json_path = output_dir / "trajectory_pair_summary.json"
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


def call_openai(prompt: str, *, repo_root: Path, model: str, reasoning_effort: str, timeout: int) -> str:
    api_key = _openai_api_key(repo_root)
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


def call_codex(
    prompt: str,
    *,
    repo_root: Path,
    model: str,
    reasoning_effort: str,
    last_message_path: Path,
    stdout_path: Path,
    timeout: int,
) -> str:
    last_message_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "codex",
        "exec",
        "-m",
        model,
        "-c",
        f'model_reasoning_effort="{reasoning_effort}"',
        "-C",
        str(repo_root),
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
    stdout_path.write_text(proc.stdout, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"codex exec failed with exit code {proc.returncode}; see {stdout_path}")
    if last_message_path.exists():
        return last_message_path.read_text(encoding="utf-8")
    return proc.stdout


def call_judge_model(
    prompt: str,
    *,
    backend: str,
    repo_root: Path,
    model: str,
    reasoning_effort: str,
    last_message_path: Path,
    stdout_path: Path,
    timeout: int,
) -> str:
    if backend == "codex":
        return call_codex(
            prompt,
            repo_root=repo_root,
            model=model,
            reasoning_effort=reasoning_effort,
            last_message_path=last_message_path,
            stdout_path=stdout_path,
            timeout=timeout,
        )
    if backend == "openai":
        text = call_openai(prompt, repo_root=repo_root, model=model, reasoning_effort=reasoning_effort, timeout=timeout)
        last_message_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        last_message_path.write_text(text, encoding="utf-8")
        stdout_path.write_text("", encoding="utf-8")
        return text
    raise ValueError(f"Unsupported judge backend: {backend}")


def _row_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (row["benchmark"], row["pair_label"], row["model_variant"], row["task_id"])


def _row_slug(row: dict[str, str]) -> str:
    benchmark, pair_label, model_variant, task_id = _row_key(row)
    task_stem = Path(task_id).with_suffix("").as_posix()
    raw = "__".join([benchmark, pair_label, model_variant, task_stem])
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("_") or "row"


def _audit_json_path(tmp_root: Path, row: dict[str, str]) -> Path:
    return tmp_root / row["benchmark"] / row["model_variant"] / row["pair_label"] / f"{_row_slug(row)}.json"


def _append_journal_record(journal_path: Path, record: dict[str, Any]) -> None:
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"timestamp_ms": int(time.time() * 1000)}
    payload.update(record)
    with journal_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end < start:
            raise
        value = json.loads(stripped[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("Expected JSON object from trajectory judge")
    return value


def validate_judge_payload(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    label = payload.get("trajectory_similarity")
    if label not in LABELS:
        errors.append(f"trajectory_similarity must be one of {LABELS}, got {label!r}")
    status = payload.get("audit_status")
    if status != "complete":
        errors.append("audit_status must be 'complete'")
    for field in [
        "divergence_type",
        "comparison_trajectory_summary",
        "ideal_trajectory_summary",
        "similarity_reason",
        "evidence",
    ]:
        if not isinstance(payload.get(field), str) or not str(payload.get(field)).strip():
            errors.append(f"{field} must be a non-empty string")
    for field in ["aligned_steps", "divergent_steps"]:
        if not isinstance(payload.get(field), str):
            errors.append(f"{field} must be a string")
    return errors


def build_repair_prompt(original_prompt: str, previous_text: str, errors: list[str]) -> str:
    return (
        original_prompt.rstrip()
        + "\n\nYour previous response failed deterministic validation.\n"
        + "Return JSON only, with exactly the required schema. Fix these errors:\n"
        + "\n".join(f"- {error}" for error in errors)
        + "\n\nPrevious response:\n```text\n"
        + previous_text
        + "\n```"
    )


def build_judge_prompt(row: dict[str, str]) -> str:
    comparison_events = extract_trajectory_excerpt(
        Path(row["comparison_log"]),
        start_after_line=int(row.get("comparison_start_line") or 0),
    )
    ideal_events = extract_trajectory_excerpt(
        Path(row["ideal_log"]),
        start_after_line=int(row.get("ideal_start_line") or 0),
    )
    return f"""You are judging whether two actual agent trajectories for the same task resemble each other.

Compare only the executed trajectory after planning. Do not judge written-plan similarity.
The ideal side is the reference trajectory. The comparison side is either plan-naive or plan-default.

Labels:
- resembles_completely: same source path, same key filters/hops/computation, same final output shape; only trivial retry/order differences.
- closely_resembles: same answer-producing path with harmless detours or small repair loops.
- partially_resembles: shares several major steps but changes or omits a meaningful source, filter, operation, hop, or finalization step.
- misses_crucial_steps: starts on a related path but skips one or more crucial hops/computations needed by the ideal trajectory.
- not_similar: materially different trajectory/source family/reasoning chain.
- not_comparable: insufficient logs, malformed trajectories, or model/task mismatch.

Return JSON only:
{{
  "trajectory_similarity": "resembles_completely|closely_resembles|partially_resembles|misses_crucial_steps|not_similar|not_comparable",
  "divergence_type": "none|minor_detour|source_mismatch|filter_mismatch|operation_mismatch|hop_skip|output_mismatch|trajectory_missing|other",
  "comparison_trajectory_summary": "one sentence",
  "ideal_trajectory_summary": "one sentence",
  "aligned_steps": "semicolon-separated aligned trajectory steps",
  "divergent_steps": "semicolon-separated divergent or missing steps",
  "similarity_reason": "short evidence-grounded reason",
  "evidence": "cite both log paths and relevant line numbers from the excerpts",
  "audit_status": "complete"
}}

Metadata:
- benchmark: {row['benchmark']}
- task_id: {row['task_id']}
- model_variant: {row['model_variant']}
- runner_model: {row['runner_model']}
- pair_label: {row['pair_label']}
- comparison_mode: {row['comparison_mode']}
- ideal_mode: {row['ideal_mode']}
- comparison_log: {row['comparison_log']}
- ideal_log: {row['ideal_log']}

COMPARISON TRAJECTORY:
{chr(10).join(comparison_events)}

IDEAL TRAJECTORY:
{chr(10).join(ideal_events)}
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
) -> int:
    judged = 0
    for row in rows:
        if row.get("audit_status") != "pending":
            continue
        if limit and judged >= limit:
            break
        print(
            "Judging trajectory pair "
            f"#{judged + 1}: benchmark={row['benchmark']} model={row['model_variant']} "
            f"pair={row['pair_label']} task={row['task_id']}",
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
                f"  attempt {attempt}: spawning {backend} judge for task={row['task_id']} pair={row['pair_label']}",
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
                        "pair_label": row["pair_label"],
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
                    "pair_label": row["pair_label"],
                    "model_variant": row["model_variant"],
                    "task_id": row["task_id"],
                    "prompt_path": str(prompt_path),
                    "last_message_path": str(last_message_path),
                    "stdout_path": str(stdout_path),
                },
            )
            if attempt > max_retries:
                raise RuntimeError(
                    "Trajectory judge response failed validation after retries: "
                    + "; ".join(validation_errors)
                )
            attempt_prompt = build_repair_prompt(prompt, text, validation_errors)
        assert parsed is not None
        label = str(parsed.get("trajectory_similarity", "not_comparable"))
        row.update(
            {
                "trajectory_similarity": label,
                "divergence_type": str(parsed.get("divergence_type", "other")),
                "comparison_trajectory_summary": str(parsed.get("comparison_trajectory_summary", "")),
                "ideal_trajectory_summary": str(parsed.get("ideal_trajectory_summary", "")),
                "aligned_steps": str(parsed.get("aligned_steps", "")),
                "divergent_steps": str(parsed.get("divergent_steps", "")),
                "similarity_reason": str(parsed.get("similarity_reason", "")),
                "evidence": str(parsed.get("evidence", "")),
                "auditor_model": model,
                "audit_status": str(parsed.get("audit_status", "complete") or "complete"),
            }
        )
        audit_payload = {
            "row_key": {
                "benchmark": row["benchmark"],
                "pair_label": row["pair_label"],
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
            "row_update": {
                key: row[key]
                for key in [
                    "trajectory_similarity",
                    "divergence_type",
                    "comparison_trajectory_summary",
                    "ideal_trajectory_summary",
                    "aligned_steps",
                    "divergent_steps",
                    "similarity_reason",
                    "evidence",
                    "auditor_model",
                    "audit_status",
                ]
            },
        }
        out_path.write_text(json.dumps(audit_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        _append_journal_record(
            journal_path,
            {
                "status": "ok",
                "benchmark": row["benchmark"],
                "pair_label": row["pair_label"],
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
        write_outputs(output_dir, rows)
        time.sleep(0.1)
    return judged


def build_all_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    benchmarks = list(BENCHMARK_LOG_ROOTS) if args.benchmark == "all" else [args.benchmark]
    pair_labels = list(PAIR_MODES) if args.pair == "all" else [args.pair]
    rows: list[dict[str, str]] = []
    for benchmark in benchmarks:
        root = Path(args.input_root) / BENCHMARK_LOG_ROOTS[benchmark]
        rows.extend(
            build_pair_rows(
                benchmark=benchmark,
                log_root=root,
                pair_labels=pair_labels,
                model_filter=args.model or None,
                task_filter=args.task or None,
            )
        )
    return rows


def row_matches_filters(row: dict[str, str], args: argparse.Namespace) -> bool:
    if args.benchmark != "all" and row["benchmark"] != args.benchmark:
        return False
    if args.pair != "all" and row["pair_label"] != args.pair:
        return False
    if args.model and row["model_variant"] != args.model:
        return False
    if args.task and args.task not in row["task_id"]:
        return False
    return True


def resolve_judge_limit(
    *,
    requested_limit: int | None,
    pending_count: int,
    is_interactive: bool,
    read_input=input,
    print_fn=print,
) -> int | None:
    if pending_count == 0:
        print_fn("Pending rows matching filters: 0")
        return requested_limit or 0
    print_fn(f"Pending rows matching filters: {pending_count}")
    if requested_limit is not None:
        if requested_limit == 0:
            print_fn(f"Judging all {pending_count} pending row(s) because --limit=0.")
            return 0
        capped = min(requested_limit, pending_count)
        print_fn(f"Judging up to {capped} pending row(s) because --limit={requested_limit}.")
        return requested_limit
    if not is_interactive:
        print_fn(
            "No --limit provided and stdin is non-interactive, so no judge calls were made. "
            f"Re-run with --limit N, or --limit 0 to judge all {pending_count} pending row(s)."
        )
        return None
    print_fn(
        f"No --limit provided. Press Enter to judge all {pending_count} pending row(s), "
        "type a number to judge only that many, or type q to abort."
    )
    choice = read_input("Rows to judge [all]: ").strip().lower()
    if choice in {"q", "quit", "abort", "n", "no"}:
        return None
    if not choice:
        return 0
    if choice.isdigit() and int(choice) > 0:
        return int(choice)
    raise SystemExit(f"Invalid row count: {choice!r}")


def full_build_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        **{
            **vars(args),
            "benchmark": "all",
            "pair": "all",
            "model": "",
            "task": "",
        }
    )


def print_summary(rows: list[dict[str, str]]) -> None:
    summary_rows, _ = summarize_rows(rows)
    print("Trajectory-pair summary:")
    for row in summary_rows:
        print(
            "  "
            f"{row['benchmark']:<11} {row['model_variant']:<18} {row['pair_label']:<10} "
            f"close+complete={row['resembles_or_close_pct_all']:>5.1f}% "
            f"({row['n_resembles_or_close']}/{row['n_total']}), "
            f"partial+={row['partial_or_better_pct_all']:>5.1f}% "
            f"({row['n_partial_or_better']}/{row['n_total']}), "
            f"pending={row['n_pending']} n/c={row['n_not_comparable']}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default=".", help="Repository/input root containing logs and log-kramabench.")
    parser.add_argument("--output-dir", default="agent_analysis/trajectory_pair_analysis")
    parser.add_argument("--benchmark", choices=["all", *BENCHMARK_LOG_ROOTS.keys()], default="all")
    parser.add_argument("--pair", choices=["all", *PAIR_MODES.keys()], default="all")
    parser.add_argument("--model", default="", help="Optional model folder, e.g. openai_gpt-5-mini.")
    parser.add_argument("--task", default="", help="Optional task-id substring filter.")
    parser.add_argument("--judge", action="store_true", help="Judge pending pairs.")
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
        default="agent_analysis/trajectory_pair_analysis/tmp",
        help="Directory for per-row prompt, raw response, and parsed judge JSON files.",
    )
    parser.add_argument(
        "--journal-path",
        default="",
        help="Optional JSONL journal path. Defaults to <output-dir>/trajectory_pair_journal.jsonl.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    audit_path = output_dir / "trajectory_pair_audit.csv"
    tmp_root = Path(args.tmp_root).resolve()
    rows = build_all_rows(full_build_args(args))
    # CSV is a legacy fallback; per-row JSON under tmp_root is the source of truth.
    merge_existing_audits(rows, load_existing_audits(audit_path))
    merge_existing_audits(rows, load_tmp_audits(tmp_root))
    written = write_outputs(output_dir, rows)
    if args.judge:
        journal_path = Path(args.journal_path).resolve() if args.journal_path else output_dir / "trajectory_pair_journal.jsonl"
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
        )
        print(f"Judged {judged} pending row(s).")
        rows = build_all_rows(full_build_args(args))
        merge_existing_audits(rows, load_existing_audits(audit_path))
        merge_existing_audits(rows, load_tmp_audits(tmp_root))
        written = write_outputs(output_dir, rows)
    print_summary(rows)
    for path in written:
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
