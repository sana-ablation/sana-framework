#!/usr/bin/env python3
"""Export SANA semantic-mode results into paper-ready LaTeX and figures."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path
from typing import Iterable, Mapping


MODEL_SPECS = [
    ("gpt-5.4-nano", ("gpt-5.4-nano", "openai_gpt-5.4-nano", "openai/gpt-5.4-nano")),
    ("gpt-5-mini", ("gpt-5-mini", "openai_gpt-5-mini", "openai/gpt-5-mini")),
]

TOOL_CALL_CAPTION_NOTE = (
    "Ret Tool Call and Acc Tool Call are average calls per task; "
    "Acc Tool Call excludes auxiliary ideal repair bookkeeping traces."
)

PLANNED_CONDITIONS = [
    ("N-I-I-I", "naive", "ideal", "ideal", "ideal"),
    ("S-I-I-I", "standard", "ideal", "ideal", "ideal"),
    ("I-N-I-I", "ideal", "naive", "ideal", "ideal"),
    ("I-S-I-I", "ideal", "standard", "ideal", "ideal"),
    ("I-P-I-I", "ideal", "preloaded", "ideal", "ideal"),
    ("I-I-S-I", "ideal", "ideal", "standard", "ideal"),
    ("I-I-I-I", "ideal", "ideal", "ideal", "ideal"),
    ("N-N-S-N", "naive", "naive", "standard", "naive"),
    ("S-S-S-I", "standard", "standard", "standard", "ideal"),
    ("I-I-I-N", "ideal", "ideal", "ideal", "naive"),
]

CANONICAL_MODE_SPECS = [
    (
        "Naive",
        {"agent_management": "naive", "search_tool": "naive", "search_results": "ideal", "computation_tool": "standard"},
    ),
    (
        "Standard",
        {
            "agent_management": "standard",
            "search_tool": "standard",
            "search_results": "ideal",
            "computation_tool": "standard",
        },
    ),
    (
        "Ideal",
        {"agent_management": "ideal", "search_tool": "ideal", "search_results": "ideal", "computation_tool": "ideal"},
    ),
]

MAIN_ABLATION_TABLE_SPECS = [
    ("naive", "ideal", "ideal"),
    ("standard", "ideal", "ideal"),
    ("ideal", "naive", "ideal"),
    ("ideal", "standard", "ideal"),
    ("ideal", "ideal", "standard"),
    ("ideal", "ideal", "ideal"),
    ("ideal", "preloaded", "ideal"),
]

BENCHMARK_TABLE_META = {
    "lakeqa": {
        "display": "LakeQA",
        "label": "tab:lakeqa-canonical-modes",
        "filename": "lakeqa_canonical_modes.tex",
    },
    "kramabench": {
        "display": "Kramabench",
        "label": "tab:kramabench-canonical-modes",
        "filename": "kramabench_canonical_modes.tex",
    },
}

MODE_DISPLAY = {
    "naive": "Naive",
    "standard": "Standard",
    "ideal": "Ideal",
    "preloaded": "Preloaded",
}

LETTER_TO_MODE = {"n": "naive", "d": "standard", "s": "standard", "i": "ideal", "p": "preloaded"}

SUPPORTING_FIGURES = {
    "results_cost_vs_semantic.pdf": ("cost_vs_semantic.pdf", "fig6_cost_vs_semantic.pdf"),
    "results_semantic_x_error_variant.pdf": (
        "semantic_error_crosstab_by_variant.pdf",
        "fig03_semantic_x_error_variant.pdf",
    ),
    "results_semantic_buckets_variant.pdf": (
        "semantic_buckets_by_variant.pdf",
        "fig01_semantic_buckets_variant.pdf",
    ),
    "results_recall_semantic_combined.pdf": (
        "retrieval_vs_semantic_combined.pdf",
        "fig2a_recall_semantic_combined.pdf",
    ),
}


def _normalize_model_name(model: str) -> str:
    normalized = model.strip().replace("openai_", "").replace("openai/", "")
    for display, aliases in MODEL_SPECS:
        if normalized == display or model in aliases:
            return display
    return normalized


def _short_model_name(model: str) -> str:
    normalized = _normalize_model_name(model)
    if normalized == "gpt-5.4-nano":
        return "nano"
    if normalized == "gpt-5-mini":
        return "mini"
    return normalized


def _parse_variant_axes(variant: str) -> dict[str, str | None]:
    axes: dict[str, str | None] = {
        "agent_management": None,
        "search_tool": None,
        "computation_tool": "standard",
        "search_results": None,
    }
    parts = variant.split("_")
    for idx, token in enumerate(parts):
        if token == "search" and idx + 1 < len(parts):
            axes["search_tool"] = LETTER_TO_MODE.get(parts[idx + 1])
        elif token == "results" and idx + 1 < len(parts):
            axes["search_results"] = LETTER_TO_MODE.get(parts[idx + 1])
        elif token.startswith("plan") and len(token) > 4:
            axes["agent_management"] = LETTER_TO_MODE.get(token[4:])
        elif token.startswith("compute") and len(token) > 7:
            axes["computation_tool"] = LETTER_TO_MODE.get(token[7:])
        elif len(token) == 2 and token[0] in {"s", "r", "p", "c"}:
            mode = LETTER_TO_MODE.get(token[1])
            if token[0] == "s":
                axes["search_tool"] = mode
            elif token[0] == "r":
                axes["search_results"] = mode
            elif token[0] == "p":
                axes["agent_management"] = mode
            elif token[0] == "c":
                axes["computation_tool"] = mode
    return axes


def _axes_for_summary_row(row: Mapping[str, object]) -> dict[str, str | None]:
    axes = _parse_variant_axes(str(row.get("variant") or ""))
    for field in ("agent_management", "search_tool", "computation_tool", "search_results"):
        value = row.get(field)
        if value:
            axes[field] = str(value).lower()
    if not axes.get("computation_tool"):
        axes["computation_tool"] = "standard"
    return axes


def _format_rate(value: object, *, pending: bool = False) -> str:
    if pending:
        return "Pending"
    if value is None:
        return "N/A"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if math.isnan(number):
        return "N/A"
    return f"{number * 100.0:.1f}"


def _format_number(value: object, *, digits: int, pending: bool = False) -> str:
    if pending:
        return "Pending"
    if value is None:
        return "N/A"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if math.isnan(number):
        return "N/A"
    return f"{number:.{digits}f}"


def _format_cost_cents(value: object, *, pending: bool = False) -> str:
    if pending:
        return "Pending"
    if value is None:
        return "N/A"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if math.isnan(number):
        return "N/A"
    return f"{number * 100.0:.1f}"


def _format_cost_dollars(value: object, *, pending: bool = False) -> str:
    if pending:
        return "Pending"
    if value is None:
        return "N/A"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if math.isnan(number):
        return "N/A"
    return f"{number:.2f}"


def load_uncached_cost_summary(path: Path | None) -> dict[tuple[str, str], dict[str, float]]:
    if path is None or not path.exists():
        return {}
    out: dict[tuple[str, str], dict[str, float]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            model = _normalize_model_name(str(row.get("model") or ""))
            variant = str(row.get("variant") or "")
            if not model or not variant:
                continue
            values: dict[str, float] = {}
            for key in (
                "avg_uncached_main_cost_usd",
                "avg_uncached_total_cost_usd",
                "avg_delta_total_cost_usd",
                "uncached_main_cost_usd",
                "uncached_total_cost_usd",
                "n",
            ):
                try:
                    values[key] = float(row.get(key) or 0.0)
                except (TypeError, ValueError):
                    values[key] = 0.0
            if not values["avg_uncached_main_cost_usd"] and values["uncached_main_cost_usd"] and values["n"]:
                values["avg_uncached_main_cost_usd"] = values["uncached_main_cost_usd"] / values["n"]
            out[(model, variant)] = values
    return out


def _apply_uncached_costs(
    summary_rows: list[Mapping[str, object]],
    uncached_costs: Mapping[tuple[str, str], Mapping[str, float]],
) -> list[dict[str, object]]:
    if not uncached_costs:
        return [dict(row) for row in summary_rows]
    rows: list[dict[str, object]] = []
    for row in summary_rows:
        row_copy = dict(row)
        model = _normalize_model_name(str(row_copy.get("model") or ""))
        variant = str(row_copy.get("variant") or "")
        cost_row = uncached_costs.get((model, variant))
        if cost_row:
            row_copy["avg_uncached_total_cost_usd"] = cost_row.get("avg_uncached_total_cost_usd")
            row_copy["avg_uncached_base_cost_usd"] = cost_row.get("avg_uncached_main_cost_usd")
            row_copy["avg_uncached_delta_cost_usd"] = cost_row.get("avg_delta_total_cost_usd")
        rows.append(row_copy)
    return rows


def _as_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def _rate_value(row: Mapping[str, object] | None, field: str) -> object:
    if row is None:
        return None
    if field == "D_acc":
        value = row.get("D_acc_recall")
        return row.get("D_acc") if value in (None, "") else value
    return row.get(field)


def _format_delta(value: object, baseline: object) -> str | None:
    value_float = _as_float(value)
    baseline_float = _as_float(baseline)
    if value_float is None or baseline_float is None:
        return None
    delta = (value_float - baseline_float) * 100.0
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.1f}"


def _format_delta_latex(delta: str | None) -> str:
    if delta in (None, ""):
        return ""
    try:
        delta_float = float(delta)
    except ValueError:
        return ""
    macro = "\\dgood" if delta_float >= 0 else "\\dbad"
    return f"\\,{macro}{{(${delta}$)}}"


def _metric_cell(row: Mapping[str, str], value_field: str, delta_field: str) -> str:
    value = str(row[value_field])
    delta = _format_delta_latex(row.get(delta_field))
    if row.get("mode") == "Ideal" and value not in {"N/A", "Pending"}:
        value = f"\\textbf{{{value}}}"
    return f"{value}{delta}"


def _find_observed_row(
    summary_rows: Iterable[Mapping[str, object]],
    *,
    model: str,
    plan: str,
    search: str,
    compute: str,
    results: str,
) -> Mapping[str, object] | None:
    for row in summary_rows:
        if _normalize_model_name(str(row.get("model") or "")) != model:
            continue
        axes = _axes_for_summary_row(row)
        if (
            axes.get("agent_management") == plan
            and axes.get("search_tool") == search
            and axes.get("computation_tool") == compute
            and axes.get("search_results") == results
        ):
            return row
    return None


def build_main_result_rows(summary_rows: list[Mapping[str, object]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for model, _aliases in MODEL_SPECS:
        for condition, plan, search, compute, results in PLANNED_CONDITIONS:
            observed = _find_observed_row(
                summary_rows,
                model=model,
                plan=plan,
                search=search,
                compute=compute,
                results=results,
            )
            pending = observed is None
            d_acc_value = None if pending else (observed.get("D_acc_recall") or observed.get("D_acc"))
            d_ret_value = None if pending else observed.get("D_ret")
            cost_value = None if pending else (
                observed.get("avg_uncached_total_cost_usd")
                or observed.get("avg_total_cost_with_ideal_subagents_usd")
                or observed.get("avg_cost_usd")
            )
            calls_value = None if pending else (
                observed.get("avg_tool_calls_total") or observed.get("avg_tool_calls")
            )
            out.append(
                {
                    "model": model,
                    "condition": condition,
                    "plan": MODE_DISPLAY[plan],
                    "search": MODE_DISPLAY[search],
                    "compute": MODE_DISPLAY[compute],
                    "results": MODE_DISPLAY[results],
                    "n": "Pending" if pending else str(int(observed.get("n") or observed.get("n_total") or 0)),
                    "semantic_match": _format_rate(None if pending else observed.get("semantic_match"), pending=pending),
                    "avg_cost": _format_cost_dollars(cost_value, pending=pending),
                    "avg_calls": _format_number(calls_value, digits=1, pending=pending),
                    "ret_tool_calls": _format_number(
                        None if pending else observed.get("avg_search_calls"),
                        digits=1,
                        pending=pending,
                    ),
                    "acc_tool_calls": _format_number(
                        None if pending else observed.get("avg_read_calls"),
                        digits=1,
                        pending=pending,
                    ),
                    "D_acc": _format_rate(d_acc_value, pending=pending),
                    "D_ret": _format_rate(d_ret_value, pending=pending),
                    "status": "pending" if pending else "observed",
                }
            )
    return out


def build_canonical_mode_rows(summary_rows: list[Mapping[str, object]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for model, _aliases in MODEL_SPECS:
        observed_by_mode: dict[str, Mapping[str, object] | None] = {}
        for mode, axes in CANONICAL_MODE_SPECS:
            observed_by_mode[mode] = _find_observed_row(
                summary_rows,
                model=model,
                plan=str(axes["agent_management"]),
                search=str(axes["search_tool"]),
                compute=str(axes["computation_tool"]),
                results=str(axes["search_results"]),
            )
        baseline = observed_by_mode.get("Naive")
        for mode, _axes in CANONICAL_MODE_SPECS:
            observed = observed_by_mode.get(mode)
            pending = observed is None
            out.append(
                {
                    "model": model,
                    "mode": mode,
                    "n": "Pending" if pending else str(int(observed.get("n") or observed.get("n_total") or 0)),
                    "semantic_match": _format_rate(None if pending else observed.get("semantic_match"), pending=pending),
                    "semantic_match_delta": None
                    if mode == "Naive" or pending
                    else _format_delta(observed.get("semantic_match"), baseline.get("semantic_match") if baseline else None),
                    "D_ret": _format_rate(None if pending else _rate_value(observed, "D_ret"), pending=pending),
                    "D_ret_delta": None
                    if mode == "Naive" or pending
                    else _format_delta(_rate_value(observed, "D_ret"), _rate_value(baseline, "D_ret")),
                    "D_acc": _format_rate(None if pending else _rate_value(observed, "D_acc"), pending=pending),
                    "D_acc_delta": None
                    if mode == "Naive" or pending
                    else _format_delta(_rate_value(observed, "D_acc"), _rate_value(baseline, "D_acc")),
                    "ret_tool_calls": _format_number(
                        None if pending else observed.get("avg_search_calls"),
                        digits=1,
                        pending=pending,
                    ),
                    "acc_tool_calls": _format_number(
                        None if pending else observed.get("avg_read_calls"),
                        digits=1,
                        pending=pending,
                    ),
                }
            )
    return out


def build_main_ablation_table_rows(summary_rows: list[Mapping[str, object]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for model, _aliases in MODEL_SPECS:
        for plan, search, compute in MAIN_ABLATION_TABLE_SPECS:
            observed = _find_observed_row(
                summary_rows,
                model=model,
                plan=plan,
                search=search,
                compute=compute,
                results="ideal",
            )
            if observed is None:
                continue
            d_acc_value = observed.get("D_acc_recall") or observed.get("D_acc")
            d_ret_value = "--" if search == "preloaded" else _format_rate(observed.get("D_ret"))
            out.append(
                {
                    "model": model,
                    "n": str(int(observed.get("n") or observed.get("n_total") or 0)),
                    "plan": MODE_DISPLAY[plan],
                    "search": MODE_DISPLAY[search],
                    "compute": MODE_DISPLAY[compute],
                    "semantic_match": _format_rate(observed.get("semantic_match")),
                    "D_acc": _format_rate(d_acc_value),
                    "D_ret": d_ret_value,
                    "avg_base_cost": _format_cost_dollars(
                        observed.get("avg_uncached_base_cost_usd") or observed.get("avg_cost_usd")
                    ),
                    "avg_base_ideal_cost": _format_cost_dollars(
                        observed.get("avg_uncached_total_cost_usd")
                        or observed.get("avg_total_cost_with_ideal_subagents_usd")
                        or observed.get("avg_cost_usd")
                    ),
                    "avg_calls": _format_number(
                        observed.get("avg_tool_calls_total") or observed.get("avg_tool_calls"),
                        digits=1,
                    ),
                    "ret_tool_calls": _format_number(observed.get("avg_search_calls"), digits=1),
                    "acc_tool_calls": _format_number(observed.get("avg_read_calls"), digits=1),
                }
            )
    return out


def _latex_escape(value: str) -> str:
    return (
        value.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
        .replace("#", "\\#")
    )


def _wrapped_header(*parts: str, bold: bool = False) -> str:
    body = "\\shortstack{" + "\\\\".join(parts) + "}"
    return f"\\textbf{{{body}}}" if bold else body


def _wrapped_model_cell(model: str) -> str:
    if model == "gpt-5.4-nano":
        return "\\shortstack[l]{\\texttt{gpt-5.4-}\\\\\\texttt{nano}}"
    if model == "gpt-5-mini":
        return "\\shortstack[l]{\\texttt{gpt-5-}\\\\\\texttt{mini}}"
    return f"\\texttt{{{_latex_escape(model)}}}"


def render_results_table(rows: list[Mapping[str, str]]) -> str:
    lines = [
        "\\begin{table*}[t]",
        "  \\caption{Main SANA ablation results. Pending cells mark planned runs that are not yet present in the current semantic analysis bundle.}",
        "  \\label{tab:sana-main-results}",
        "  \\centering",
        "  \\scriptsize",
        "  \\setlength{\\tabcolsep}{3pt}",
        "  \\begin{tabular}{llllllllrrrrrl}",
        "    \\toprule",
        "    Model & ID & Plan & Search & Data An. & Results & $n$ & SM (\\%) & Cost (\\$) & Calls & Ret Tool Call & Acc Tool Call & $D_{acc}$ (\\%) & $D_{ret}$ (\\%) \\\\",
        "    \\midrule",
    ]
    for row in rows:
        cells = [
            row["model"],
            row["condition"],
            row["plan"],
            row["search"],
            row["compute"],
            row["results"],
            row["n"],
            row["semantic_match"],
            row["avg_cost"],
            row["avg_calls"],
            row["ret_tool_calls"],
            row["acc_tool_calls"],
            row["D_acc"],
            row["D_ret"],
        ]
        lines.append("    " + " & ".join(_latex_escape(str(cell)) for cell in cells) + " \\\\")
    lines.extend(
        [
            "    \\bottomrule",
            "  \\end{tabular}",
            "\\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def render_main_ablation_table(rows: list[Mapping[str, str]], *, benchmark: str, n_tasks: int | None = None) -> str:
    display = BENCHMARK_TABLE_META[benchmark]["display"]
    label = "tab:sana-lakeqa" if benchmark == "lakeqa" else "tab:sana-kramabench"
    n_text = f" (${n_tasks}$ tasks)" if n_tasks is not None and benchmark == "lakeqa" else ""
    if benchmark == "kramabench" and n_tasks is not None:
        n_text = f" (${n_tasks}$ tasks per cell)"
    caption = (
        f"{display} SANA per-axis ablation matrix{n_text}. "
        + TOOL_CALL_CAPTION_NOTE
    )
    if benchmark == "kramabench":
        caption += " Same column conventions as Table~\\ref{tab:sana-lakeqa}."
    rows_by_model: dict[str, list[Mapping[str, str]]] = {}
    for row in rows:
        rows_by_model.setdefault(str(row["model"]), []).append(row)
    lines = [
        "\\begin{table}[h]",
        "  \\centering",
        f"  \\caption{{{caption} Base is the average uncached estimated API dollars per task for the agent run; Base+Ideal adds the ideal-tool subagent calls.}}",
        f"  \\label{{{label}}}",
        "  \\scriptsize",
        "  \\setlength{\\tabcolsep}{2.5pt}",
        "  \\renewcommand{\\arraystretch}{0.9}",
        "\\resizebox{\\columnwidth}{!}{%",
        "  \\begin{tabular}{llllrrrrrrr}",
        "    \\toprule",
        "    Model & Plan & Search & "
        f"{_wrapped_header('Data', 'An.')} & "
        "SM (\\%) & "
        "$D_{acc}$ (\\%) & "
        "$D_{ret}$ (\\%) & "
        "Base (\\$) & "
        "\\shortstack{Base+Ideal\\\\(\\$)} & "
        f"{_wrapped_header('Ret Tool', 'Call')} & "
        f"{_wrapped_header('Acc Tool', 'Call')} \\\\",
        "    \\midrule",
    ]
    for model_index, (model, _aliases) in enumerate(MODEL_SPECS):
        model_rows = rows_by_model.get(model, [])
        if not model_rows:
            continue
        if model_index > 0:
            lines.append("    \\midrule")
        lines.append(f"    \\multirow{{{len(model_rows)}}}{{*}}{{{_wrapped_model_cell(model)}}}")
        for row in model_rows:
            semantic = f"\\textbf{{{row['semantic_match']}}}" if row["search"] == "Preloaded" or (
                row["plan"] == "Ideal" and row["search"] == "Ideal" and row["compute"] == "Ideal"
            ) else row["semantic_match"]
            d_acc = f"\\textbf{{{row['D_acc']}}}" if row["search"] == "Preloaded" or (
                row["plan"] == "Ideal" and row["search"] == "Ideal" and row["compute"] == "Ideal"
            ) else row["D_acc"]
            cells = [
                "",
                row["plan"],
                row["search"],
                row["compute"],
                semantic,
                d_acc,
                row["D_ret"],
                row["avg_base_cost"],
                row["avg_base_ideal_cost"],
                row["ret_tool_calls"],
                row["acc_tool_calls"],
            ]
            lines.append("      " + " & ".join(str(cell) for cell in cells) + " \\\\")
    lines.extend(
        [
            "    \\bottomrule",
            "  \\end{tabular}",
            "}",
            "",
            "\\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def render_canonical_modes_table(rows: list[Mapping[str, str]], *, benchmark: str, n_tasks: int | None = None) -> str:
    meta = BENCHMARK_TABLE_META[benchmark]
    display = str(meta["display"])
    n_text = f" ($n={n_tasks}$)" if n_tasks is not None else ""
    lines = [
        "\\begin{table}[h]",
        "  \\centering",
        "  \\scriptsize",
        "  \\setlength{\\tabcolsep}{2.5pt}",
        "  \\renewcommand{\\arraystretch}{0.9}",
        f"  \\caption{{{display} end-to-end mode comparison{n_text}. Deltas are relative to each model's Baseline. {TOOL_CALL_CAPTION_NOTE}}}",
        f"  \\label{{{meta['label']}}}",
        "  \\begin{tabular}{@{}llrrrrr@{}}",
        "    \\toprule",
        "    \\textbf{Model} & \\textbf{Mode} & "
        "\\textbf{SM (\\%)} & "
        "\\textbf{$D_{ret}$ (\\%)} & "
        "\\textbf{$D_{acc}$ (\\%)} & "
        "\\textbf{Ret} & "
        "\\textbf{Acc} \\\\",
        "    \\midrule",
    ]
    rows_by_model: dict[str, list[Mapping[str, str]]] = {}
    for row in rows:
        rows_by_model.setdefault(str(row["model"]), []).append(row)
    for model_index, (model, _aliases) in enumerate(MODEL_SPECS):
        model_rows = rows_by_model.get(model, [])
        if not model_rows:
            continue
        if model_index > 0:
            lines.append("    \\midrule")
        for row_index, row in enumerate(model_rows):
            model_cell = f"\\multirow{{3}}{{*}}{{{_wrapped_model_cell(model)}}}" if row_index == 0 else ""
            cells = [
                model_cell,
                row["mode"],
                _metric_cell(row, "semantic_match", "semantic_match_delta"),
                _metric_cell(row, "D_ret", "D_ret_delta"),
                _metric_cell(row, "D_acc", "D_acc_delta"),
                row["ret_tool_calls"],
                row["acc_tool_calls"],
            ]
            lines.append("    " + " & ".join(str(cell) for cell in cells) + " \\\\")
    lines.extend(
        [
            "    \\bottomrule",
            "  \\end{tabular}",
            "",
            "\\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def write_semantic_heatmap(rows: list[Mapping[str, str]], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    models = [model for model, _aliases in MODEL_SPECS]
    condition_ids = [condition for condition, *_rest in PLANNED_CONDITIONS]
    values = np.full((len(models), len(condition_ids)), np.nan)
    for row in rows:
        if row["semantic_match"] == "Pending":
            continue
        model_idx = models.index(row["model"])
        condition_idx = condition_ids.index(row["condition"])
        values[model_idx, condition_idx] = float(row["semantic_match"])

    cmap = plt.cm.Greens.copy()
    cmap.set_bad("#ECEFF3")
    fig, ax = plt.subplots(figsize=(8.6, 2.4))
    im = ax.imshow(values, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(condition_ids)))
    ax.set_xticklabels(condition_ids, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=9)
    ax.set_title("Semantic Match by SANA Ablation Cell", fontsize=11)
    for i in range(len(models)):
        for j in range(len(condition_ids)):
            if np.isnan(values[i, j]):
                label = "pending"
                color = "#667085"
            else:
                label = f"{values[i, j]:.1f}%"
                color = "white" if values[i, j] >= 45 else "#1f2937"
            ax.text(j, i, label, ha="center", va="center", fontsize=7, color=color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.ax.set_ylabel("Semantic match (%)", rotation=270, labelpad=14, fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def copy_supporting_figures(analysis_dir: Path, paper_figures_dir: Path) -> list[Path]:
    copied: list[Path] = []
    source_dir = analysis_dir / "figures"
    paper_figures_dir.mkdir(parents=True, exist_ok=True)
    for target_name, source_names in SUPPORTING_FIGURES.items():
        source = next((source_dir / source_name for source_name in source_names if (source_dir / source_name).exists()), None)
        if source is not None:
            target = paper_figures_dir / target_name
            shutil.copyfile(source, target)
            copied.append(target)
    return copied


def _infer_benchmark(analysis_dir: Path) -> str:
    name = str(analysis_dir).lower()
    if "kramabench" in name or "krama" in name:
        return "kramabench"
    return "lakeqa"


def _canonical_n_tasks(rows: list[Mapping[str, str]]) -> int | None:
    observed = [int(row["n"]) for row in rows if str(row.get("n", "")).isdigit()]
    return observed[0] if observed else None


def _default_uncached_cost_summary_path(benchmark: str) -> Path:
    return Path("analysis_results_uncached_costs") / benchmark / "variant_uncached_cost_summary.csv"


def export_paper_results(
    analysis_dir: Path,
    paper_dir: Path,
    *,
    benchmark: str | None = None,
    uncached_cost_summary: Path | None = None,
) -> dict[str, Path | list[Path]]:
    summary_path = analysis_dir / "summary.json"
    variant_summary_path = analysis_dir / "variant_summary.json"
    with summary_path.open() as handle:
        summary_rows = json.load(handle)
    if variant_summary_path.exists():
        with variant_summary_path.open() as handle:
            json.load(handle)

    benchmark = benchmark or _infer_benchmark(analysis_dir)
    if uncached_cost_summary is None:
        default_cost_path = _default_uncached_cost_summary_path(benchmark)
        uncached_cost_summary = default_cost_path if default_cost_path.exists() else None
    summary_rows = _apply_uncached_costs(summary_rows, load_uncached_cost_summary(uncached_cost_summary))

    rows = build_main_result_rows(summary_rows)
    generated_tex = paper_dir / "subsections" / "generated_main_results.tex"
    generated_tex.parent.mkdir(parents=True, exist_ok=True)
    generated_tex.write_text(render_results_table(rows), encoding="utf-8")

    main_ablation_rows = build_main_ablation_table_rows(summary_rows)
    main_ablation_tex = paper_dir / "subsections" / (
        "lakeqa_main_results.tex" if benchmark == "lakeqa" else "kramabench_main_results.tex"
    )
    main_ablation_tex.write_text(
        render_main_ablation_table(main_ablation_rows, benchmark=benchmark, n_tasks=_canonical_n_tasks(main_ablation_rows)),
        encoding="utf-8",
    )

    canonical_rows = build_canonical_mode_rows(summary_rows)
    canonical_meta = BENCHMARK_TABLE_META[benchmark]
    canonical_tex = paper_dir / "subsections" / str(canonical_meta["filename"])
    canonical_tex.write_text(
        render_canonical_modes_table(canonical_rows, benchmark=benchmark, n_tasks=_canonical_n_tasks(canonical_rows)),
        encoding="utf-8",
    )

    heatmap_path = paper_dir / "figures" / "results_semantic_match_heatmap.pdf"
    write_semantic_heatmap(rows, heatmap_path)
    copied = copy_supporting_figures(analysis_dir, paper_dir / "figures")
    return {
        "table": generated_tex,
        "main_ablation_table": main_ablation_tex,
        "canonical_table": canonical_tex,
        "heatmap": heatmap_path,
        "supporting_figures": copied,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", default="analysis_results_mode_semantic")
    parser.add_argument("--paper-dir", default="sana_framework_paper")
    parser.add_argument("--benchmark", choices=sorted(BENCHMARK_TABLE_META), default=None)
    parser.add_argument("--uncached-cost-summary", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = export_paper_results(
        Path(args.analysis_dir),
        Path(args.paper_dir),
        benchmark=args.benchmark,
        uncached_cost_summary=Path(args.uncached_cost_summary) if args.uncached_cost_summary else None,
    )
    print(f"Wrote {outputs['table']}")
    print(f"Wrote {outputs['main_ablation_table']}")
    print(f"Wrote {outputs['canonical_table']}")
    print(f"Wrote {outputs['heatmap']}")
    for figure in outputs["supporting_figures"]:
        print(f"Wrote {figure}")


if __name__ == "__main__":
    main()
