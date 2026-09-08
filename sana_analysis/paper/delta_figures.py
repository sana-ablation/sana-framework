#!/usr/bin/env python3
"""Generate semantic delta and paired-mode comparison figures for mode analyses."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


PLAN_ABLATION = [
    ("n", "No Plan"),
    ("d", "Default Plan"),
    ("i", "Ideal Plan"),
]

SEARCH_ABLATION = [
    ("n", "BM25 Search"),
    ("d", "PNEUMA Hybrid Search"),
    ("i", "Ideal Search"),
    ("p", "Preloaded Sources"),
]

EXECUTION_ABLATION = [
    ("d", "Standard Data Analysis"),
    ("i", "Ideal Data Analysis"),
]

ABLATIONS = [
    ("Plan Ablation", "plan", "n", PLAN_ABLATION, {"search": "i", "results": "i", "compute": "i"}),
    ("Search Ablation", "search", "n", SEARCH_ABLATION, {"plan": "i", "results": "i", "compute": "i"}),
    ("Data Analysis Ablation", "compute", "d", EXECUTION_ABLATION, {"plan": "i", "search": "i", "results": "i"}),
]

COMPARISON_MODELS = ["gpt-5.4-nano", "gpt-5-mini"]

SEMANTIC_DELTA_TITLE_FONTSIZE = 13.0
SEMANTIC_DELTA_LABEL_FONTSIZE = 11.5
SEMANTIC_DELTA_VALUE_FONTSIZE = 11.0
SEMANTIC_DELTA_SUPTITLE_FONTSIZE = 16.0

SEMANTIC_DELTA_COMPACT_TITLE_FONTSIZE = 15.5
SEMANTIC_DELTA_COMPACT_LABEL_FONTSIZE = 13.2
SEMANTIC_DELTA_COMPACT_VALUE_FONTSIZE = 12.0
SEMANTIC_DELTA_COMPACT_MODEL_FONTSIZE = 15.0
SEMANTIC_DELTA_COMPACT_AXIS_LABEL_FONTSIZE = 0.0

PAIRED_CONDITIONS = [
    (
        "nns",
        "BM25 Search, No Plan, Standard Data Analysis",
        "BM25 Search, No Plan, Standard Data Analysis",
        {"plan": "n", "search": "n", "results": "i", "compute": "d"},
    ),
    (
        "sss",
        "Pneuma Search, Plan, Standard Data Analysis",
        "Pneuma Search, Plan, Standard Data Analysis",
        {"plan": "d", "search": "d", "results": "i", "compute": "d"},
    ),
    (
        "iii",
        "Ideal Search, Ideal Plan, Ideal Data Analysis",
        "Ideal Search, Ideal Plan, Ideal Data Analysis",
        {"plan": "i", "search": "i", "results": "i", "compute": "i"},
    ),
]

PAIRED_CONDITION_AXIS_LABELS = {
    "nns": "BM25 / No Plan / Std DA",
    "sss": "Pneuma / Plan / Std DA",
    "iii": "Ideal / Ideal / Ideal",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", default="analysis_results_mode_semantic")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def _as_float(value) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _round(value: float) -> float:
    return round(value, 6)


def _mean(values: List[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def _safe_slug(value: str) -> str:
    chars = [char.lower() if char.isalnum() else "_" for char in str(value)]
    slug = "".join(chars)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_") or "unknown"


def _parse_variant_codes(variant: str) -> Dict[str, Optional[str]]:
    codes: Dict[str, Optional[str]] = {
        "search": None,
        "results": None,
        "plan": None,
        "compute": "d",
        "skills": None,
        "k": None,
        "sc": None,
    }
    parts = str(variant).split("_")
    for idx, token in enumerate(parts):
        if token == "search" and idx + 1 < len(parts):
            codes["search"] = parts[idx + 1]
        elif token == "results" and idx + 1 < len(parts):
            codes["results"] = parts[idx + 1]
        elif token.startswith("plan") and len(token) > 4:
            codes["plan"] = token[4:]
        elif token.startswith("compute") and len(token) > 7:
            codes["compute"] = token[7:]
        elif token == "skills" and idx + 1 < len(parts):
            codes["skills"] = parts[idx + 1]
        elif token.startswith("k") and token[1:].isdigit():
            codes["k"] = token[1:]
        elif token.startswith("sc") and token[2:].isdigit():
            codes["sc"] = token[2:]
    return codes


def _context_key(codes: Dict[str, Optional[str]], axis: str) -> Tuple[Tuple[str, Optional[str]], ...]:
    return tuple(
        (field, value)
        for field, value in sorted(codes.items())
        if field != axis
    )


def _matches_codes(codes: Dict[str, Optional[str]], expected: Dict[str, str]) -> bool:
    return all(codes.get(field) == value for field, value in expected.items())


def _metric_value(row: dict, metric: str) -> Optional[float]:
    if metric == "em":
        return _as_float(row.get("em", row.get("exact_match")))
    return _as_float(row.get(metric))


def _model_from_row(row: dict) -> str:
    model = row.get("model") or row.get("model_dir")
    if model:
        return str(model)
    condition_model = str(row.get("condition_model", ""))
    if "/" in condition_model:
        return condition_model.split("/", 1)[0]
    return "unknown"


def _model_sort_key(model: str) -> Tuple[int, str]:
    model_text = str(model)
    lowered = model_text.lower()
    for idx, token in enumerate(COMPARISON_MODELS):
        if token.lower() in lowered:
            return (idx, model_text)
    return (len(COMPARISON_MODELS), model_text)


def load_summary_rows(analysis_dir: Path | str) -> List[dict]:
    root = Path(analysis_dir)
    summary_path = root / "summary.json"
    with summary_path.open() as handle:
        rows = json.load(handle)

    if not isinstance(rows, list):
        raise ValueError(f"Expected {summary_path} to contain a list of rows")

    rows = [dict(row) for row in rows]
    em_by_condition = _load_em_from_per_task_semantic(root / "per_task_semantic.csv")
    for row in rows:
        condition_model = str(row.get("condition_model", ""))
        if _metric_value(row, "em") is None and condition_model in em_by_condition:
            row["em"] = em_by_condition[condition_model]
            row["exact_match"] = em_by_condition[condition_model]
    return rows


def _load_em_from_per_task_semantic(path: Path) -> Dict[str, float]:
    if not path.exists():
        return {}
    values: Dict[str, List[float]] = defaultdict(list)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            condition_model = str(row.get("condition_model", ""))
            exact_match = _as_float(row.get("exact_match"))
            if condition_model and exact_match is not None:
                values[condition_model].append(exact_match)
    return {key: _round(sum(vals) / len(vals)) for key, vals in values.items() if vals}


def build_semantic_delta_rows(summary_rows: List[dict]) -> List[dict]:
    output_rows: List[dict] = []
    for ablation_name, axis, baseline_code, members, fixed_context in ABLATIONS:
        grouped: Dict[str, Dict[str, List[dict]]] = defaultdict(lambda: defaultdict(list))
        for row in summary_rows:
            value = _metric_value(row, "semantic_match")
            if value is None:
                continue
            codes = _parse_variant_codes(str(row.get("variant", "")))
            if not _matches_codes(codes, fixed_context):
                continue
            code = codes.get(axis)
            if code is None:
                continue
            model = _model_from_row(row)
            grouped[model][code].append(row)

        deltas_by_model_label: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
        for model, rows_by_code in grouped.items():
            if baseline_code not in rows_by_code:
                continue
            if not any(code != baseline_code and code in rows_by_code for code, _label in members):
                continue
            baseline_rows = rows_by_code.get(baseline_code, [])
            baseline_values = [
                value
                for value in (_metric_value(row, "semantic_match") for row in baseline_rows)
                if value is not None
            ]
            baseline = _mean(baseline_values)
            if baseline is None:
                continue
            for code, label in members:
                if code not in rows_by_code:
                    continue
                values = [
                    value
                    for value in (_metric_value(row, "semantic_match") for row in rows_by_code.get(code, []))
                    if value is not None
                ]
                value = _mean(values)
                if value is None:
                    continue
                deltas_by_model_label[(model, label)].append(
                    {
                        "value": value,
                        "baseline": baseline,
                        "delta": value - baseline,
                    }
                )

        for model in sorted({model for model, _label in deltas_by_model_label}, key=_model_sort_key):
            for code, label in members:
                entries = deltas_by_model_label.get((model, label), [])
                if not entries:
                    continue
                values = [entry["value"] for entry in entries]
                baselines = [entry["baseline"] for entry in entries]
                deltas = [entry["delta"] for entry in entries]
                output_rows.append(
                    {
                        "model": model,
                        "ablation": ablation_name,
                        "axis": axis,
                        "code": code,
                        "label": label,
                        "semantic_match": _round(_mean(values) or 0.0),
                        "baseline_semantic_match": _round(_mean(baselines) or 0.0),
                        "delta": _round(_mean(deltas) or 0.0),
                        "n_contexts": len(entries),
                    }
                )
    return output_rows


def build_paired_mode_metric_rows(summary_rows: List[dict]) -> List[dict]:
    grouped: Dict[Tuple[str, str], dict] = {}
    for row in summary_rows:
        codes = _parse_variant_codes(str(row.get("variant", "")))
        matched_condition = None
        for condition_id, condition_code, condition_label, expected_codes in PAIRED_CONDITIONS:
            if _matches_codes(codes, expected_codes):
                matched_condition = (condition_id, condition_code, condition_label)
                break
        if matched_condition is None:
            continue
        condition_id, condition_code, condition_label = matched_condition
        metrics = {
            "semantic_match": _metric_value(row, "semantic_match"),
            "em": _metric_value(row, "em"),
            "D_ret": _metric_value(row, "D_ret"),
            "D_acc": _metric_value(row, "D_acc"),
        }
        if all(value is None for value in metrics.values()):
            continue
        model = _model_from_row(row)
        group = grouped.setdefault(
            (model, condition_id),
            {
                "model": model,
                "variants": [],
                "condition_id": condition_id,
                "condition_code": condition_code,
                "condition_label": condition_label,
                "_semantic_match": [],
                "_em": [],
                "_D_ret": [],
                "_D_acc": [],
            },
        )
        group["variants"].append(str(row.get("variant", "")))
        for metric, value in metrics.items():
            if value is not None:
                group[f"_{metric}"].append(value)

    condition_order = {
        condition_id: idx for idx, (condition_id, _code, _label, _expected) in enumerate(PAIRED_CONDITIONS)
    }
    rows: List[dict] = []
    for (model, _condition_id), group in grouped.items():
        rows.append(
            {
                "model": group["model"],
                "variant": ",".join(sorted(set(group["variants"]))),
                "condition_id": group["condition_id"],
                "condition_code": group["condition_code"],
                "condition_label": group["condition_label"],
                "semantic_match": _round(_mean(group["_semantic_match"])) if group["_semantic_match"] else None,
                "em": _round(_mean(group["_em"])) if group["_em"] else None,
                "D_ret": _round(_mean(group["_D_ret"])) if group["_D_ret"] else None,
                "D_acc": _round(_mean(group["_D_acc"])) if group["_D_acc"] else None,
                "n_variants": len(set(group["variants"])),
            }
        )
    return sorted(
        rows,
        key=lambda item: (
            _model_sort_key(str(item["model"])),
            condition_order.get(str(item["condition_id"]), 99),
            str(item["variant"]),
        ),
    )


def write_delta_csv(rows: List[dict], output_path: Path) -> None:
    fieldnames = [
        "model",
        "ablation",
        "axis",
        "code",
        "label",
        "semantic_match",
        "baseline_semantic_match",
        "delta",
        "n_contexts",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_paired_csv(rows: List[dict], output_path: Path) -> None:
    fieldnames = [
        "model",
        "variant",
        "condition_id",
        "condition_code",
        "condition_label",
        "semantic_match",
        "em",
        "D_ret",
        "D_acc",
        "n_variants",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _import_plot_libs():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        print("Skipping delta figures: matplotlib is not installed.")
        return None


def _cleanup_delta_figure_pdfs(output_dir: Path) -> None:
    for pattern in (
        "semantic_delta_ablation_*.pdf",
        "paired_mode_metrics_*.pdf",
        "semantic_delta_ablation_comparison.pdf",
        "semantic_delta_ablation_compact.pdf",
        "paired_mode_metrics_comparison.pdf",
        "fig21_*_semantic_delta_ablation.pdf",
        "fig22_*_paired_modes_metrics.pdf",
        "fig21a_*_comparison.pdf",
        "fig21b_*_compact.pdf",
        "fig22a_*_comparison.pdf",
    ):
        for path in output_dir.glob(pattern):
            path.unlink()


def generate_delta_figures(summary_rows: List[dict], output_dir: Path) -> Dict[str, List[dict]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    _cleanup_delta_figure_pdfs(output_dir)
    delta_rows = build_semantic_delta_rows(summary_rows)
    paired_rows = build_paired_mode_metric_rows(summary_rows)
    write_delta_csv(delta_rows, output_dir / "semantic_delta_ablation.csv")
    write_paired_csv(paired_rows, output_dir / "paired_mode_metrics.csv")

    plt = _import_plot_libs()
    if plt is not None:
        _plot_semantic_delta_ablation(plt, delta_rows, output_dir)
        _plot_paired_mode_metrics(plt, paired_rows, output_dir)
        _plot_semantic_delta_ablation_comparison(plt, delta_rows, output_dir)
        _plot_semantic_delta_ablation_compact(plt, delta_rows, output_dir)
        _plot_paired_mode_metrics_comparison(plt, paired_rows, output_dir)
    return {"semantic_delta_rows": delta_rows, "paired_mode_rows": paired_rows}


def _delta_color(delta: Optional[float]) -> str:
    if delta is None:
        return "#8A8F98"
    if delta > 0.0001:
        return "#2E8B57"
    if delta < -0.0001:
        return "#C44E52"
    return "#8A8F98"


def _delta_label(delta: Optional[float]) -> str:
    if delta is None:
        return "n/a"
    return f"{delta:+.1f}%"


def _delta_value_label(value: float, delta: Optional[float]) -> str:
    return f"{value:.1f}% ({_delta_label(delta)})"


def _plot_horizontal_delta_bars(
    ax,
    labels: List[str],
    values: List[Optional[float]],
    deltas: List[Optional[float]],
    title: str,
    *,
    label_fontsize: float = 9,
    show_y_labels: bool = True,
    value_fontsize: float = 9,
    title_fontsize: Optional[float] = None,
    x_limit: float = 124.0,
    inside_label_reserved: float = 45.0,
) -> None:
    if not labels:
        ax.set_title(title, fontsize=title_fontsize)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center", fontsize=9)
        for spine in ("top", "right", "bottom"):
            ax.spines[spine].set_visible(False)
        return

    y_positions = list(range(len(labels)))
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels if show_y_labels else [""] * len(labels), fontsize=label_fontsize)
    if not show_y_labels:
        ax.tick_params(axis="y", length=0)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlim(0, x_limit)
    ax.set_xticks([])
    ax.tick_params(axis="x", length=0)
    for spine in ("top", "right", "bottom"):
        ax.spines[spine].set_visible(False)

    for y_position, value, delta in zip(y_positions, values, deltas):
        if value is None:
            ax.text(
                1.2,
                y_position,
                "n/a",
                ha="left",
                va="center",
                fontsize=value_fontsize,
                color="#5C6168",
            )
            continue
        bar = ax.barh(
            [y_position],
            [value],
            color=_delta_color(delta),
            alpha=0.42,
            height=0.62,
        )[0]
        label_x = value + 1.2
        label_ha = "left"
        if label_x > x_limit - inside_label_reserved:
            label_x = max(1.5, value - 1.4)
            label_ha = "right"
        ax.text(
            label_x,
            bar.get_y() + bar.get_height() / 2,
            _delta_value_label(value, delta),
            ha=label_ha,
            va="center",
            fontsize=value_fontsize,
            fontweight="medium",
            color="#111827",
        )
    ax.set_ylim(len(labels) - 0.5, -0.5)


def _comparison_models(rows_by_model: Dict[str, List[dict]]) -> List[str]:
    return sorted(rows_by_model, key=_model_sort_key)[:2]


def _compact_model_label(model: str) -> str:
    return str(model).replace("openai_", "").replace("_", "-")


def _paired_condition_axis_label(condition_id: str) -> str:
    return PAIRED_CONDITION_AXIS_LABELS.get(str(condition_id), str(condition_id))


def _compact_ablation_label(label: str) -> str:
    replacements = {
        "Default Plan": "Default",
        "Ideal Plan": "Ideal",
        "BM25 Search": "BM25",
        "PNEUMA Hybrid Search": "PNEUMA",
        "Ideal Search": "Ideal",
        "Preloaded Sources": "Preloaded",
        "Standard Data Analysis": "Standard",
        "Ideal Data Analysis": "Ideal",
    }
    return replacements.get(label, label)


def _plot_semantic_delta_ablation(plt, delta_rows: List[dict], output_dir: Path) -> None:
    if not delta_rows:
        return
    rows_by_model: Dict[str, List[dict]] = defaultdict(list)
    for row in delta_rows:
        rows_by_model[str(row["model"])].append(row)

    for model, model_rows in sorted(rows_by_model.items(), key=lambda item: _model_sort_key(item[0])):
        fig, axes = plt.subplots(3, 1, figsize=(13.2, 9.2))
        for ax, (ablation_name, _axis, _baseline_code, members, _fixed_context) in zip(axes, ABLATIONS):
            by_label = {row["label"]: row for row in model_rows if row["ablation"] == ablation_name}
            labels = [label for _code, label in members]
            values = [
                float(by_label[label]["semantic_match"]) * 100.0 if label in by_label else None
                for label in labels
            ]
            deltas = [float(by_label[label]["delta"]) * 100.0 if label in by_label else None for label in labels]
            _plot_horizontal_delta_bars(
                ax,
                labels,
                values,
                deltas,
                ablation_name,
                label_fontsize=SEMANTIC_DELTA_LABEL_FONTSIZE,
                value_fontsize=SEMANTIC_DELTA_VALUE_FONTSIZE,
                title_fontsize=SEMANTIC_DELTA_TITLE_FONTSIZE,
            )
        fig.suptitle(
            f"Semantic Match Ablations vs. Naive Baseline - {model}",
            fontsize=SEMANTIC_DELTA_SUPTITLE_FONTSIZE,
        )
        fig.subplots_adjust(left=0.20, right=0.975, bottom=0.06, top=0.91, hspace=0.42)
        fig.savefig(output_dir / f"semantic_delta_ablation_{_safe_slug(model)}.pdf")
        plt.close(fig)


def _plot_semantic_delta_ablation_comparison(plt, delta_rows: List[dict], output_dir: Path) -> None:
    if not delta_rows:
        return
    rows_by_model: Dict[str, List[dict]] = defaultdict(list)
    for row in delta_rows:
        rows_by_model[str(row["model"])].append(row)
    models = _comparison_models(rows_by_model)
    if len(models) < 2:
        return

    fig, axes = plt.subplots(3, 2, figsize=(15.6, 7.6), squeeze=False)
    for col_idx, model in enumerate(models):
        model_rows = rows_by_model.get(model, [])
        for row_idx, (ablation_name, _axis, _baseline_code, members, _fixed_context) in enumerate(ABLATIONS):
            ax = axes[row_idx][col_idx]
            by_label = {row["label"]: row for row in model_rows if row["ablation"] == ablation_name}
            labels = [label for _code, label in members]
            values = [
                float(by_label[label]["semantic_match"]) * 100.0 if label in by_label else None
                for label in labels
            ]
            deltas = [float(by_label[label]["delta"]) * 100.0 if label in by_label else None for label in labels]
            title = f"{model}\n{ablation_name}" if row_idx == 0 else ablation_name
            _plot_horizontal_delta_bars(
                ax,
                labels,
                values,
                deltas,
                title,
                label_fontsize=SEMANTIC_DELTA_LABEL_FONTSIZE,
                show_y_labels=col_idx == 0,
                value_fontsize=SEMANTIC_DELTA_VALUE_FONTSIZE,
                title_fontsize=SEMANTIC_DELTA_TITLE_FONTSIZE,
            )

    fig.suptitle("Semantic Match Ablations", fontsize=SEMANTIC_DELTA_SUPTITLE_FONTSIZE)
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.075, top=0.89, hspace=0.34, wspace=0.15)
    fig.savefig(output_dir / "semantic_delta_ablation_comparison.pdf")
    plt.close(fig)


def _plot_semantic_delta_ablation_compact(plt, delta_rows: List[dict], output_dir: Path) -> None:
    if not delta_rows:
        return
    rows_by_model: Dict[str, List[dict]] = defaultdict(list)
    for row in delta_rows:
        rows_by_model[str(row["model"])].append(row)
    models = _comparison_models(rows_by_model)
    if len(models) < 2:
        return

    fig, axes = plt.subplots(len(models), len(ABLATIONS), figsize=(14.2, 4.65), squeeze=False)
    for row_idx, model in enumerate(models):
        model_rows = rows_by_model.get(model, [])
        for col_idx, (ablation_name, _axis, _baseline_code, members, _fixed_context) in enumerate(ABLATIONS):
            ax = axes[row_idx][col_idx]
            by_label = {row["label"]: row for row in model_rows if row["ablation"] == ablation_name}
            labels = [_compact_ablation_label(label) for _code, label in members]
            values = [
                float(by_label[label]["semantic_match"]) * 100.0 if label in by_label else None
                for _code, label in members
            ]
            deltas = [
                float(by_label[label]["delta"]) * 100.0 if label in by_label else None
                for _code, label in members
            ]
            title = ablation_name.replace(" Ablation", "") if row_idx == 0 else ""
            _plot_horizontal_delta_bars(
                ax,
                labels,
                values,
                deltas,
                title,
                label_fontsize=SEMANTIC_DELTA_COMPACT_LABEL_FONTSIZE,
                value_fontsize=SEMANTIC_DELTA_COMPACT_VALUE_FONTSIZE,
                title_fontsize=SEMANTIC_DELTA_COMPACT_TITLE_FONTSIZE,
                x_limit=100.0,
                inside_label_reserved=42.0,
            )

    fig.subplots_adjust(left=0.115, right=0.997, bottom=0.055, top=0.855, hspace=0.22, wspace=0.43)
    for row_idx, model in enumerate(models):
        bbox = axes[row_idx][0].get_position()
        fig.text(
            0.030,
            bbox.y0 + bbox.height / 2,
            _compact_model_label(model),
            rotation=90,
            ha="center",
            va="center",
            fontsize=SEMANTIC_DELTA_COMPACT_MODEL_FONTSIZE,
            fontweight="bold",
        )
    fig.savefig(output_dir / "semantic_delta_ablation_compact.pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def _plot_paired_mode_metrics(plt, paired_rows: List[dict], output_dir: Path) -> None:
    if not paired_rows:
        return
    fields = [
        ("semantic_match", "Semantic Match"),
        ("D_ret", "D_ret"),
        ("D_acc", "D_acc"),
    ]
    rows_by_model: Dict[str, List[dict]] = defaultdict(list)
    for row in paired_rows:
        if any(row.get(field) is not None for field, _label in fields):
            rows_by_model[str(row["model"])].append(row)
    condition_order = {
        condition_id: idx for idx, (condition_id, _code, _label, _expected) in enumerate(PAIRED_CONDITIONS)
    }
    for model, rows in sorted(rows_by_model.items(), key=lambda item: _model_sort_key(item[0])):
        rows = sorted(rows, key=lambda row: condition_order.get(str(row.get("condition_id")), 99))
        rows_by_condition = {str(row.get("condition_id")): row for row in rows}
        fig, axes = plt.subplots(3, 1, figsize=(10.5, 7.2))
        for ax, (field, metric_label) in zip(axes, fields):
            labels = [_paired_condition_axis_label(condition_id) for condition_id, _code, _label, _expected in PAIRED_CONDITIONS]
            values = [
                float(rows_by_condition[condition_id][field]) * 100.0
                if condition_id in rows_by_condition and rows_by_condition[condition_id].get(field) is not None
                else None
                for condition_id, _code, _label, _expected in PAIRED_CONDITIONS
            ]
            baseline_row = rows_by_condition.get("nns")
            baseline = _as_float(baseline_row.get(field)) if baseline_row else None
            deltas = [
                (
                    (float(rows_by_condition[condition_id][field]) - baseline) * 100.0
                    if baseline is not None
                    and condition_id in rows_by_condition
                    and rows_by_condition[condition_id].get(field) is not None
                    else None
                )
                for condition_id, _code, _label, _expected in PAIRED_CONDITIONS
            ]
            _plot_horizontal_delta_bars(ax, labels, values, deltas, metric_label, label_fontsize=9)
        fig.suptitle(f"Reference Mode Metrics (Search / Plan / Data Analysis) - {model}")
        fig.subplots_adjust(left=0.22, right=0.98, bottom=0.055, top=0.9, hspace=0.36)
        fig.savefig(output_dir / f"paired_mode_metrics_{_safe_slug(model)}.pdf")
        plt.close(fig)


def _plot_paired_mode_metrics_comparison(plt, paired_rows: List[dict], output_dir: Path) -> None:
    if not paired_rows:
        return
    fields = [
        ("semantic_match", "Semantic Match"),
        ("D_ret", "D_ret"),
        ("D_acc", "D_acc"),
    ]
    rows_by_model: Dict[str, List[dict]] = defaultdict(list)
    for row in paired_rows:
        if any(row.get(field) is not None for field, _label in fields):
            rows_by_model[str(row["model"])].append(row)
    models = _comparison_models(rows_by_model)
    if len(models) < 2:
        return

    condition_order = {
        condition_id: idx for idx, (condition_id, _code, _label, _expected) in enumerate(PAIRED_CONDITIONS)
    }
    n_models = len(models)
    fig, axes = plt.subplots(n_models, 3, figsize=(14.2, 1.6 + 1.3 * n_models), squeeze=False)
    for row_idx, model in enumerate(models):
        rows = sorted(rows_by_model.get(model, []), key=lambda row: condition_order.get(str(row.get("condition_id")), 99))
        rows_by_condition = {str(row.get("condition_id")): row for row in rows}
        for col_idx, (field, metric_label) in enumerate(fields):
            ax = axes[row_idx][col_idx]
            labels = [_paired_condition_axis_label(condition_id) for condition_id, _code, _label, _expected in PAIRED_CONDITIONS]
            values = [
                float(rows_by_condition[condition_id][field]) * 100.0
                if condition_id in rows_by_condition and rows_by_condition[condition_id].get(field) is not None
                else None
                for condition_id, _code, _label, _expected in PAIRED_CONDITIONS
            ]
            baseline_row = rows_by_condition.get("nns")
            baseline = _as_float(baseline_row.get(field)) if baseline_row else None
            deltas = [
                (
                    (float(rows_by_condition[condition_id][field]) - baseline) * 100.0
                    if baseline is not None
                    and condition_id in rows_by_condition
                    and rows_by_condition[condition_id].get(field) is not None
                    else None
                )
                for condition_id, _code, _label, _expected in PAIRED_CONDITIONS
            ]
            title = metric_label if row_idx == 0 else ""
            _plot_horizontal_delta_bars(
                ax,
                labels,
                values,
                deltas,
                title,
                label_fontsize=9,
                show_y_labels=col_idx == 0,
            )

    fig.suptitle("Canonical Mode Metrics (Search / Plan / Data Analysis)", fontsize=14)
    fig.subplots_adjust(left=0.14, right=0.985, bottom=0.08, top=0.84, hspace=0.18, wspace=0.08)
    for row_idx, model in enumerate(models):
        ax_left = axes[row_idx][0]
        bbox = ax_left.get_position()
        fig.text(
            0.020,
            bbox.y0 + bbox.height / 2,
            model,
            rotation=90,
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
        )
    fig.savefig(output_dir / "paired_mode_metrics_comparison.pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    analysis_dir = Path(args.analysis_dir)
    output_dir = Path(args.output_dir) if args.output_dir else analysis_dir / "figures"
    rows = load_summary_rows(analysis_dir)
    outputs = generate_delta_figures(rows, output_dir)
    print(f"Wrote {len(outputs['semantic_delta_rows'])} semantic delta rows to {output_dir}")
    print(f"Wrote {len(outputs['paired_mode_rows'])} paired mode rows to {output_dir}")


if __name__ == "__main__":
    main()
