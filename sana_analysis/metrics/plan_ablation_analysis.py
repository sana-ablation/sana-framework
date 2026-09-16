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

from sana_analysis.paper.plan_ablation_figure import PLAN_D_AXES, PLAN_I_AXES
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
