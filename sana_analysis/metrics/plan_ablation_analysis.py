#!/usr/bin/env python3
"""Judge the standard-plan arm's plan text against the ideal arm's.

The plan-similarity half of the plan ablation. Shaped like
`metrics/trajectory_ideal_context_analysis`, which does prepare/judge/summarize
for the project's other model-judged measurement, so the two read alike.

`plan_d_*` / `plan_i_*` field names are the output contract that
`paper/plan_ablation_figure._summarize_rows` reads. "plan_d" is the retired
spelling of `plan=standard`; the names stay, the selection does not -- arms are
resolved through `variants.find_variant`, never matched against literals.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

from sana_analysis.paper.plan_ablation_figure import BUCKET_ORDER, PLAN_D_AXES, PLAN_I_AXES
from sana_analysis.variants import find_variant, try_parse_variant

RUNNER_MODEL_RE = re.compile(r"\bNEW TASK:\s*([A-Za-z0-9_.-]+)")
PLAN_TOOLS = ("plan_ideal", "plan")

PAIR_COLUMNS = [
    "model_variant",
    "runner_model",
    "task_id",
    "plan_d_mode",
    "plan_i_mode",
    "plan_d_log",
    "plan_i_log",
    "plan_d_found",
    "plan_i_found",
    "plan_d_line",
    "plan_i_line",
    "plan_d_text",
    "plan_i_text",
    "missing_plan_type",
    "prefill_plan_similarity",
    "prefill_divergence_type",
    "prefill_similarity_reason",
]


def iter_plan_payloads(line: str) -> Iterable[tuple[str, dict]]:
    decoder = json.JSONDecoder()
    for tool_name in PLAN_TOOLS:
        search_from = 0
        marker = f"{tool_name}("
        while True:
            marker_index = line.find(marker, search_from)
            if marker_index == -1:
                break
            payload_start = marker_index + len(marker)
            try:
                payload, _ = decoder.raw_decode(line[payload_start:])
            except json.JSONDecodeError:
                search_from = payload_start
                continue
            if isinstance(payload, dict):
                yield tool_name, payload
            search_from = payload_start + 1


def extract_plan_calls(log_path: Path) -> list[tuple[str, str, int]]:
    """(tool_name, plan_text, line_number) for every explicit planning call."""
    calls: list[tuple[str, str, int]] = []
    for line_number, line in enumerate(log_path.read_text(errors="replace").splitlines(), start=1):
        for tool_name, payload in iter_plan_payloads(line):
            plan_text = payload.get("plan_text")
            if isinstance(plan_text, str):
                calls.append((tool_name, plan_text, line_number))
    return calls


def extract_runner_model(log_path: Path) -> str:
    for line in log_path.read_text(errors="replace").splitlines():
        match = RUNNER_MODEL_RE.search(line)
        if match:
            return match.group(1)
    return ""


def _first_call(calls: list[tuple[str, str, int]], tool_name: str) -> tuple[str, str, int] | None:
    for call in calls:
        if call[0] == tool_name:
            return call
    return None


def _mode_log_paths(log_root: Path) -> Iterable[tuple[str, str, str, Path]]:
    modes_root = log_root / "modes"
    if not modes_root.is_dir():
        return
    for log_path in sorted(modes_root.glob("*/*/**/*.log")):
        rel_parts = log_path.relative_to(modes_root).parts
        if len(rel_parts) < 4:
            continue
        model_variant, mode_variant = rel_parts[0], rel_parts[1]
        task_id = Path(*rel_parts[2:]).as_posix()
        yield model_variant, mode_variant, task_id, log_path


def observed_modes(log_root: Path) -> list[str]:
    """Every variant directory name present under `log_root/modes/*/`."""
    modes_root = log_root / "modes"
    if not modes_root.is_dir():
        return []
    names = {
        variant_dir.name
        for model_dir in modes_root.iterdir()
        if model_dir.is_dir()
        for variant_dir in model_dir.iterdir()
        if variant_dir.is_dir()
    }
    return sorted(names)


def resolve_plan_modes(
    log_root: Path,
    *,
    plan_d_mode: str | None = None,
    plan_i_mode: str | None = None,
) -> tuple[str, str]:
    """The two arms of the plan ablation, as they are spelled in this tree.

    Resolved by axes rather than by name, so every naming generation on disk
    reads. `find_variant` raises rather than guessing when two names share the
    three ablation axes; pass the mode explicitly to break such a tie.
    """
    names = observed_modes(log_root)
    resolved_d = plan_d_mode or find_variant(names, **PLAN_D_AXES)
    resolved_i = plan_i_mode or find_variant(names, **PLAN_I_AXES)
    for label, axes, resolved in (
        ("standard-plan", PLAN_D_AXES, resolved_d),
        ("ideal-plan", PLAN_I_AXES, resolved_i),
    ):
        if resolved is None:
            raise ValueError(
                f"No {label} arm under {log_root}: no variant matches {axes!r}. "
                f"Observed modes: {names or '(none)'}"
            )
    return resolved_d, resolved_i


def missing_plan_type(plan_d_found: bool, plan_i_found: bool) -> str:
    if plan_d_found and plan_i_found:
        return ""
    if not plan_d_found and not plan_i_found:
        return "missing_both"
    if not plan_d_found:
        return "missing_plan_d"
    return "missing_plan_i"


def prefill_not_comparable_reason(missing_type: str, d_log: Path, i_log: Path) -> str:
    if missing_type == "missing_plan_d":
        return f"No explicit plan() call was found in standard-plan log: {d_log.resolve(strict=False)}"
    if missing_type == "missing_plan_i":
        return f"No explicit plan_ideal() call was found in ideal-plan log: {i_log.resolve(strict=False)}"
    if missing_type == "missing_both":
        return (
            "Neither paired log has the required explicit planning call: "
            f"{d_log.resolve(strict=False)} ; {i_log.resolve(strict=False)}"
        )
    return ""


def _accepts(mode_variant: str, *, plan: str) -> bool:
    decoded = try_parse_variant(mode_variant)
    return decoded is not None and decoded.plan == plan


def build_plan_pair_rows(
    log_root: Path,
    *,
    model: str | None = None,
    plan_d_mode: str | None = None,
    plan_i_mode: str | None = None,
    task: str | None = None,
    include_all_modes: bool = False,
) -> list[dict[str, str]]:
    """One row per (model, task) pairing of the standard and ideal plan arms."""
    if include_all_modes:
        def accepts_d(mode: str) -> bool:
            return _accepts(mode, plan="standard")

        def accepts_i(mode: str) -> bool:
            return _accepts(mode, plan="ideal")
    else:
        resolved_d, resolved_i = resolve_plan_modes(
            log_root, plan_d_mode=plan_d_mode, plan_i_mode=plan_i_mode
        )

        def accepts_d(mode: str) -> bool:
            return mode == resolved_d

        def accepts_i(mode: str) -> bool:
            return mode == resolved_i

    defaults: dict[tuple[str, str], list[tuple[str, Path]]] = {}
    ideals: dict[tuple[str, str], list[tuple[str, Path]]] = {}
    for model_variant, mode_variant, task_id, log_path in _mode_log_paths(log_root):
        if model and model_variant != model:
            continue
        if task and task not in task_id:
            continue
        key = (model_variant, task_id)
        if accepts_d(mode_variant):
            defaults.setdefault(key, []).append((mode_variant, log_path))
        if accepts_i(mode_variant):
            ideals.setdefault(key, []).append((mode_variant, log_path))

    rows: list[dict[str, str]] = []
    for key in sorted(defaults.keys() & ideals.keys()):
        model_variant, task_id = key
        for d_mode, d_log in sorted(defaults[key]):
            d_log = d_log.resolve(strict=False)
            d_call = _first_call(extract_plan_calls(d_log), "plan")
            d_model = extract_runner_model(d_log)
            for i_mode, i_log in sorted(ideals[key]):
                i_log = i_log.resolve(strict=False)
                i_call = _first_call(extract_plan_calls(i_log), "plan_ideal")
                i_model = extract_runner_model(i_log)
                runner_model = d_model or i_model
                if d_model and i_model and d_model != i_model:
                    runner_model = f"{d_model}|{i_model}"
                missing_type = missing_plan_type(d_call is not None, i_call is not None)
                model_mismatch = bool(d_model and i_model and d_model != i_model)
                prefill_similarity = "not_comparable" if missing_type or model_mismatch else ""
                prefill_divergence = "model_mismatch" if model_mismatch else missing_type
                prefill_reason = (
                    f"Runner model mismatch: standard-plan log has {d_model}, ideal log has {i_model}."
                    if model_mismatch
                    else prefill_not_comparable_reason(missing_type, d_log, i_log)
                )
                rows.append({
                    "model_variant": model_variant,
                    "runner_model": runner_model,
                    "task_id": task_id,
                    "plan_d_mode": d_mode,
                    "plan_i_mode": i_mode,
                    "plan_d_log": str(d_log),
                    "plan_i_log": str(i_log),
                    "plan_d_found": str(d_call is not None),
                    "plan_i_found": str(i_call is not None),
                    "plan_d_line": str(d_call[2]) if d_call else "",
                    "plan_i_line": str(i_call[2]) if i_call else "",
                    "plan_d_text": d_call[1] if d_call else "",
                    "plan_i_text": i_call[1] if i_call else "",
                    "missing_plan_type": missing_type,
                    "prefill_plan_similarity": prefill_similarity,
                    "prefill_divergence_type": prefill_divergence,
                    "prefill_similarity_reason": prefill_reason,
                })
    return rows


def render_text(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "No paired standard-plan / ideal-plan logs found."
    lines = ["Paired standard-plan / ideal-plan logs:"]
    for idx, row in enumerate(rows, start=1):
        status = [
            "plan_d found" if row["plan_d_found"] == "True" else "plan_d missing",
            "plan_i found" if row["plan_i_found"] == "True" else "plan_i missing",
        ]
        lines.append(
            f"{idx}. {row['model_variant']} ({row['runner_model']}) | "
            f"{row['plan_d_mode']} vs {row['plan_i_mode']} | {row['task_id']} | "
            f"{', '.join(status)}"
        )
    return "\n".join(lines)


# The labels a judge may return. `no_plan` is deliberately absent: it is not a
# judgment. `plan_ablation_figure._normalize_plan_similarity` derives it from
# `missing_plan_type == "missing_plan_d"`, which the prepare stage establishes
# mechanically. The drift test asserts that these plus DERIVED_LABELS are exactly
# `plan_ablation_figure.BUCKET_ORDER`.
PLAN_SIMILARITY_DEFINITIONS = {
    "similar": (
        "Same answer-producing strategy. The standard plan may be less specific -- it may "
        "say 'find the relevant dataset' or omit exact column names, file names or "
        "indicator labels -- and still be similar, as long as it points to the same source "
        "family, required hops, core computation and final output."
    ),
    "missing_details": (
        "Same source family, hops, broad computation and output, but one plan omits a "
        "material detail that could change the result: exact period, indicator, excluded "
        "aggregate, geography, category, threshold, tie-break, denominator definition, "
        "unit conversion or final formatting constraint. Not for harmless abstraction."
    ),
    "incomplete_plan": (
        "Same broad objective, but one plan omits, compresses away or reverses a required "
        "intermediate step needed to derive a later entity or filter -- 'find the sector, "
        "then count incidents in that sector'."
    ),
    "operation_mismatch": (
        "Same topic and objective, but one step is materially different: computation, "
        "filter, comparison direction, grouping key, threshold, ratio direction, min/max "
        "choice, conversion or final output shape would likely produce a different answer."
    ),
    "not_similar": (
        "The plans would answer materially different questions: different source family, "
        "required hop, target population, computation, comparison direction or final "
        "selection."
    ),
    "not_comparable": "Missing plan text, malformed logs, or a runner model mismatch.",
}

PLAN_SIMILARITY_LABELS = list(PLAN_SIMILARITY_DEFINITIONS)

# Derived by the figure from `missing_plan_type`, never asked of a judge.
DERIVED_LABELS = frozenset({"no_plan"})

AUDIT_COLUMNS = [
    "plan_similarity",
    "divergence_type",
    "similarity_reason",
    "aligned_steps",
    "divergent_steps",
    "provenance_note",
    "auditor_model",
    "audit_status",
]

OUTPUT_COLUMNS = PAIR_COLUMNS + AUDIT_COLUMNS

DEFAULT_OUTPUT_DIR = Path("agent_analysis/plan_default_analysis")


def format_plan_similarity_definitions() -> str:
    return "\n".join(
        f"- {label}: {definition}"
        for label, definition in PLAN_SIMILARITY_DEFINITIONS.items()
    )


import time
from typing import Any, Callable

from sana_analysis.metrics.trajectory_pair_analysis import (
    _append_journal_record,
    _parse_json_object,
    build_repair_prompt,
    call_judge_model,
    write_csv,
)


def build_judge_prompt(row: dict[str, str]) -> str:
    return f"""You are judging whether two plans for the same task describe the same
answer-producing strategy.

Judge the plans as intended algorithms. Ignore whether either run later
succeeded. The standard plan is allowed to be less exact than the ideal plan;
do not penalise harmless abstraction.

Task: {row['task_id']}
Model variant: {row['model_variant']}
Runner model: {row['runner_model']}
Standard-plan log: {row['plan_d_log']} (line {row['plan_d_line']})
Ideal-plan log: {row['plan_i_log']} (line {row['plan_i_line']})

Standard plan (`plan`):
\"\"\"
{row['plan_d_text']}
\"\"\"

Ideal plan (`plan_ideal`):
\"\"\"
{row['plan_i_text']}
\"\"\"

Compare, in order: question target, source path, required hops, filters and
cohorts, computation, final selection.

Choose exactly one plan_similarity label:
{format_plan_similarity_definitions()}

Return JSON only:
{{
  "plan_similarity": "one of the labels above",
  "divergence_type": "short string",
  "similarity_reason": "short evidence-grounded explanation",
  "aligned_steps": "semicolon-separated",
  "divergent_steps": "semicolon-separated",
  "provenance_note": "where the model identity came from",
  "audit_status": "complete"
}}
"""


def validate_judge_payload(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    label = payload.get("plan_similarity")
    if label not in PLAN_SIMILARITY_LABELS:
        errors.append(f"plan_similarity must be one of {PLAN_SIMILARITY_LABELS}, got {label!r}")
    if payload.get("audit_status") != "complete":
        errors.append("audit_status must be 'complete'")
    for field in ("divergence_type", "similarity_reason", "provenance_note"):
        if not isinstance(payload.get(field), str) or not str(payload.get(field)).strip():
            errors.append(f"{field} must be a non-empty string")
    for field in ("aligned_steps", "divergent_steps"):
        if not isinstance(payload.get(field), str):
            errors.append(f"{field} must be a string")
    return errors


def _prefill(row: dict[str, str]) -> None:
    """Copy the prepare stage's mechanical verdict into the audit columns."""
    row["plan_similarity"] = row["prefill_plan_similarity"]
    row["divergence_type"] = row["prefill_divergence_type"]
    row["similarity_reason"] = row["prefill_similarity_reason"]
    row.setdefault("aligned_steps", "")
    row.setdefault("divergent_steps", "")
    row["provenance_note"] = f"prefilled by prepare stage: {row['missing_plan_type'] or 'model_mismatch'}"
    row["auditor_model"] = ""
    row["audit_status"] = "prefilled"


def _pair_dir(output_dir: Path, log_root_name: str, row: dict[str, str]) -> Path:
    return (
        output_dir
        / log_root_name
        / "modes"
        / row["model_variant"]
        / f"{row['plan_d_mode']}__vs__{row['plan_i_mode']}"
    )


def write_outputs(output_dir: Path, rows: list[dict[str, str]], log_root_name: str) -> list[Path]:
    """One plan_similarity.csv per (model, arm-pair), where the figure rglobs for them."""
    grouped: dict[Path, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(_pair_dir(output_dir, log_root_name, row), []).append(row)
    written = []
    for pair_dir, pair_rows in sorted(grouped.items()):
        path = pair_dir / "plan_similarity.csv"
        write_csv(path, [{key: row.get(key, "") for key in OUTPUT_COLUMNS} for row in pair_rows], OUTPUT_COLUMNS)
        written.append(path)
    return written


def judge_pending_rows(
    rows: list[dict[str, str]],
    *,
    output_dir: Path,
    log_root_name: str,
    repo_root: Path,
    backend: str,
    model: str,
    reasoning_effort: str,
    limit: int,
    timeout: int,
    tmp_root: Path,
    journal_path: Path,
    max_retries: int,
    call: Callable[..., str] = call_judge_model,
) -> int:
    """Judge every row that is neither prefilled nor already complete.

    `call` is the stub-judge seam, mirroring the `SemanticJudge` Protocol the
    semantic auditor uses: tests substitute it so nothing reaches a real API.
    """
    judged = 0
    for row in rows:
        if row.get("prefill_plan_similarity"):
            if row.get("audit_status") != "prefilled":
                _prefill(row)
            continue
        if row.get("audit_status") == "complete":
            continue
        if limit and judged >= limit:
            break

        row.setdefault("audit_status", "pending")
        print(
            f"Judging plan pair #{judged + 1}: model={row['model_variant']} task={row['task_id']}",
            flush=True,
        )
        prompt = build_judge_prompt(row)
        stem = tmp_root / row["model_variant"] / re.sub(r"[^A-Za-z0-9_.-]+", "_", row["task_id"]).strip("_")
        stem.parent.mkdir(parents=True, exist_ok=True)
        last_message_path = stem.with_suffix(".last_message.txt")
        stdout_path = stem.with_suffix(f".{backend}_stdout.log")

        attempt_prompt = prompt
        parsed: dict[str, Any] | None = None
        for attempt in range(1, max_retries + 2):
            text = call(
                attempt_prompt,
                backend=backend,
                repo_root=repo_root,
                model=model,
                reasoning_effort=reasoning_effort,
                last_message_path=last_message_path,
                stdout_path=stdout_path,
                timeout=timeout,
            )
            try:
                parsed = _parse_json_object(text)
            except (json.JSONDecodeError, ValueError) as exc:
                errors = [f"response must be a JSON object: {exc}"]
            else:
                errors = validate_judge_payload(parsed)
            if not errors:
                break
            _append_journal_record(journal_path, {
                "status": "retry",
                "attempt": attempt,
                "errors": errors,
                "model_variant": row["model_variant"],
                "task_id": row["task_id"],
            })
            if attempt > max_retries:
                raise RuntimeError(
                    "Plan-similarity judge response failed validation after retries: " + "; ".join(errors)
                )
            attempt_prompt = build_repair_prompt(prompt, text, errors)

        assert parsed is not None
        row.update({
            "plan_similarity": str(parsed["plan_similarity"]),
            "divergence_type": str(parsed.get("divergence_type", "")),
            "similarity_reason": str(parsed.get("similarity_reason", "")),
            "aligned_steps": str(parsed.get("aligned_steps", "")),
            "divergent_steps": str(parsed.get("divergent_steps", "")),
            "provenance_note": str(parsed.get("provenance_note", "")),
            "auditor_model": model,
            "audit_status": "complete",
        })
        _append_journal_record(journal_path, {
            "status": "ok",
            "model_variant": row["model_variant"],
            "task_id": row["task_id"],
            "label": row["plan_similarity"],
        })
        judged += 1
        write_outputs(output_dir, rows, log_root_name)
        time.sleep(0.1)

    write_outputs(output_dir, rows, log_root_name)
    return judged


def summarize_rows(rows: Iterable[dict[str, str]]) -> list[dict[str, Any]]:
    """Counts per (model, label), in BUCKET_ORDER, for the run's console report."""
    from collections import Counter

    grouped: dict[str, Counter] = {}
    for row in rows:
        label = row.get("plan_similarity", "")
        if row.get("missing_plan_type") == "missing_plan_d":
            label = "no_plan"
        if not label:
            continue
        grouped.setdefault(row["model_variant"], Counter())[label] += 1
    return [
        {"model_variant": model, "n": sum(counts.values()),
         **{label: counts.get(label, 0) for label in BUCKET_ORDER}}
        for model, counts in sorted(grouped.items())
    ]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_root", help="Log root such as logs or log-kramabench")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--model", default="", help="Optional model folder, e.g. openai_gpt-5-mini.")
    parser.add_argument("--task", default="", help="Optional task-id substring filter.")
    parser.add_argument("--plan-d-mode", default="")
    parser.add_argument("--plan-i-mode", default="")
    parser.add_argument("--all-modes", action="store_true")
    parser.add_argument("--judge", action="store_true", help="Judge pending pairs.")
    parser.add_argument("--backend", choices=["codex", "openai"], default="codex")
    parser.add_argument("--judge-model", default="gpt-5.4-mini")
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--limit-files", type=int, default=0, help="Judge at most this many models.")
    parser.add_argument("--limit-rows", type=int, default=0, help="Judge at most this many pairs.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    log_root = Path(args.log_root).resolve()
    output_dir = Path(args.output_dir)
    rows = build_plan_pair_rows(
        log_root,
        model=args.model or None,
        plan_d_mode=args.plan_d_mode or None,
        plan_i_mode=args.plan_i_mode or None,
        task=args.task or None,
        include_all_modes=args.all_modes,
    )
    if args.limit_files > 0:
        keep = sorted({row["model_variant"] for row in rows})[: args.limit_files]
        rows = [row for row in rows if row["model_variant"] in keep]
    print(f"Prepared {len(rows)} plan pairs from {log_root}")

    if args.judge:
        judge_pending_rows(
            rows,
            output_dir=output_dir,
            log_root_name=log_root.name,
            repo_root=Path.cwd(),
            backend=args.backend,
            model=args.judge_model,
            reasoning_effort=args.reasoning_effort,
            limit=args.limit_rows,
            timeout=args.timeout,
            tmp_root=output_dir / "tmp",
            journal_path=output_dir / "journal.jsonl",
            max_retries=args.max_retries,
        )
    else:
        for row in rows:
            if row.get("prefill_plan_similarity"):
                _prefill(row)
            else:
                row.setdefault("audit_status", "pending")
        write_outputs(output_dir, rows, log_root.name)
        print("Prepared only. Pass --judge to run the judging stage.")

    for summary_row in summarize_rows(rows):
        print(summary_row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
