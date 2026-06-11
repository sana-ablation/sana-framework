#!/usr/bin/env python3
"""Combine answer-failure audit events into paper-ready artifacts."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sana_analysis.running_analysis.answer_failure_taxonomy import (
    ANSWER_FAILURE_FIGURE_GROUPS as ANSWER_FAILURE_GROUPS,
    OMITTED_ANSWER_FAILURE_TYPES,
)
from sana_analysis.report_generator.build_answer_failure_report import _load_rows_for_events_file, _trusted_events


COMBINED_CSV_NAME = "combined_answer_failure_events.csv"
COMBINED_REPORT_NAME = "combined_answer_failure_report.md"
COMBINED_FIGURE_NAME = "answer_failure_groups_by_model.pdf"
COMBINED_CONDITION_FIGURE_NAME = "answer_failure_groups_by_condition.pdf"

PREFIX_COLUMNS = ["model_variant", "mode_variant", "source_file", "answer_failure_figure_group"]
MODEL_COLORS = {
    "openai_gpt-5-mini": "#4C78A8",
    "gpt-5-mini": "#4C78A8",
    "openai_gpt-5.4-nano": "#F58518",
    "gpt-5.4-nano": "#F58518",
}
GROUP_COLORS = {
    "Task/planning failures": "#7A5C7E",
    "Wrong source target failures": "#4E79A7",
    "Execution/computation failures": "#9C755F",
    "Incomplete evidence failures": "#E15759",
    "Turn-waste failures": "#F28E2B",
    "Finalization failures": "#59A14F",
    "Tool blocker failures": "#76B7B2",
}
PREFERRED_GROUP_ORDER = [
    "Task/planning failures",
    "Wrong source target failures",
    "Execution/computation failures",
    "Incomplete evidence failures",
    "Turn-waste failures",
    "Finalization failures",
    "Tool blocker failures",
]
MODEL_GROUP_TITLE_FONTSIZE = 22.0
MODEL_GROUP_AXIS_FONTSIZE = 17.0
MODEL_GROUP_LABEL_FONTSIZE = 16.0
MODEL_GROUP_TICK_FONTSIZE = 15.0
MODEL_GROUP_VALUE_FONTSIZE = 15.0
MODEL_GROUP_LEGEND_FONTSIZE = 15.0
CONDITION_FIGURE_BAR_LABEL_FONTSIZE = 14.0
CONDITION_FIGURE_TOTAL_FONTSIZE = 14.5
CONDITION_FIGURE_MODEL_LABEL_FONTSIZE = 14.0
CONDITION_FIGURE_CONDITION_LABEL_FONTSIZE = 17.0
CONDITION_FIGURE_YLABEL_FONTSIZE = 17.0
CONDITION_FIGURE_YTICK_FONTSIZE = 15.0
CONDITION_FIGURE_LEGEND_FONTSIZE = 14.0
CONDITION_FIGURE_LEGEND_COLUMNS = 2
CONDITION_FIGURE_BAR_WIDTH = 0.78
CONDITION_FIGURE_SEGMENT_LABEL_MIN_FRACTION = 0.055
CONDITION_FIGURE_YMAX = 260
CONDITION_FIGURE_LEGEND_Y_ANCHOR = 0.925
CONDITION_FIGURE_LEGEND_LABELS = {
    "Task/planning failures": "Task/planning",
    "Wrong source target failures": "Wrong source target",
    "Execution/computation failures": "Execution/computation",
    "Incomplete evidence failures": "Incomplete evidence",
    "Turn-waste failures": "Turn-waste",
    "Finalization failures": "Finalization",
    "Tool blocker failures": "Tool blocker",
}
CONDITION_FIGURE_ORDER = [
    ("No Plan", "search_i_results_i_plann_computei_k5_skills_off"),
    ("Standard Plan", "search_i_results_i_pland_computei_k5_skills_off"),
    ("BM25", "search_n_results_i_plani_computei_k5_skills_off"),
    ("Pneuma Hybrid", "search_d_results_i_plani_computei_k5_skills_off"),
    ("Standard Computation", "search_i_results_i_plani_k5_skills_off"),
    ("Ideal", "search_i_results_i_plani_computei_k5_skills_off"),
    ("Preloaded", "search_p_results_i_plani_computei_k5_skills_off"),
]
CONDITION_FIGURE_MODEL_ORDER = ["openai_gpt-5.4-nano", "openai_gpt-5-mini", "gpt-5.4-nano", "gpt-5-mini"]
CONDITION_FIGURE_MODEL_LABELS = {
    "openai_gpt-5.4-nano": "5.4\nnano",
    "gpt-5.4-nano": "5.4\nnano",
    "openai_gpt-5-mini": "5\nmini",
    "gpt-5-mini": "5\nmini",
}
CONDITION_FIGURE_LABELS = {
    "No Plan": "No\nPlan",
    "Standard Plan": "Standard\nPlan",
    "BM25": "BM25\nSearch",
    "Pneuma Hybrid": "Pneuma\nSearch",
    "Standard Computation": "Standard\nData An.",
    "Ideal": "Ideal",
    "Preloaded": "Preloaded\nSources",
}
@dataclass(frozen=True)
class SourceFile:
    path: Path
    model_variant: str
    mode_variant: str
    relative_path: str


def clean_label(value: str | None, fallback: str = "(blank)") -> str:
    value = (value or "").strip()
    return value if value else fallback


def pretty_model(model: str) -> str:
    return model.replace("openai_", "").replace("openai/", "")


def figure_group_for_type(answer_failure_type: str | None) -> str:
    label = clean_label(answer_failure_type, "other_or_unclear")
    return ANSWER_FAILURE_GROUPS.get(label, label)


def _variants_from_events_path(source_root: Path, events_path: Path) -> tuple[str, str]:
    rel_path = events_path.relative_to(source_root)
    parts = rel_path.parts
    if "modes" in parts:
        modes_index = parts.index("modes")
        after_modes = parts[modes_index + 1 :]
        if len(after_modes) >= 3:
            return after_modes[0], after_modes[1]
    if len(parts) >= 4:
        return parts[-3], parts[-2]
    return "(unknown-model)", events_path.parent.name


def discover_source_files(source_root: Path) -> list[SourceFile]:
    sources: list[SourceFile] = []
    for path in sorted(source_root.rglob("answer_failure_events.csv")):
        model_variant, mode_variant = _variants_from_events_path(source_root, path)
        sources.append(
            SourceFile(
                path=path,
                model_variant=model_variant,
                mode_variant=mode_variant,
                relative_path=path.relative_to(source_root).as_posix(),
            )
        )
    return sources


def _trusted_if_validation_available(events: list[dict]) -> list[dict]:
    if not events:
        return []
    has_validation = any(
        "row_validation_status" in event or "row_model_validation_status" in event
        for event in events
    )
    return _trusted_events(events) if has_validation else events


def read_combined_rows(sources: Iterable[SourceFile], source_root: Path, *, trusted_only: bool = True) -> tuple[list[dict], list[str]]:
    rows: list[dict] = []
    source_columns: list[str] = []
    seen_columns: set[str] = set()

    for source in sources:
        _, events = _load_rows_for_events_file(source_root, source.path)
        selected_events = _trusted_if_validation_available(events) if trusted_only else events
        for event in selected_events:
            for column in event:
                if column not in seen_columns and column not in PREFIX_COLUMNS:
                    source_columns.append(column)
                    seen_columns.add(column)
            answer_failure_type = clean_label(event.get("answer_failure_type"))
            if answer_failure_type in OMITTED_ANSWER_FAILURE_TYPES:
                continue
            combined = {
                "model_variant": clean_label(event.get("model_variant"), source.model_variant),
                "mode_variant": clean_label(event.get("mode_variant"), source.mode_variant),
                "source_file": source.relative_path,
                "answer_failure_figure_group": figure_group_for_type(answer_failure_type),
            }
            combined.update({column: str(event.get(column, "")) for column in source_columns})
            rows.append(combined)

    for row in rows:
        for column in source_columns:
            row.setdefault(column, "")

    rows.sort(
        key=lambda row: (
            row["source_file"],
            clean_label(row.get("answer_failure_figure_group")),
            clean_label(row.get("answer_failure_type")),
            clean_label(row.get("answer_failure_subtype")),
            row.get("task_id", ""),
            row.get("event_index", ""),
        )
    )
    return rows, PREFIX_COLUMNS + source_columns


def write_combined_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _label_text_color(hex_color: str) -> str:
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        return "black"
    r, g, b = (int(hex_color[index : index + 2], 16) for index in (0, 2, 4))
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    return "black" if luminance > 0.55 else "white"


def _ordered_groups_from_counts(group_totals: Counter[str]) -> list[str]:
    preferred = [group for group in PREFERRED_GROUP_ORDER if group_totals.get(group, 0) > 0]
    extras = sorted(group for group, count in group_totals.items() if count > 0 and group not in preferred)
    return preferred + extras


def write_model_group_figure(path: Path, rows: list[dict]) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    counts_by_group_model: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        group_name = clean_label(row.get("answer_failure_figure_group"))
        counts_by_group_model[group_name][row["model_variant"]] += 1

    if not counts_by_group_model:
        return False

    group_totals = Counter({group: sum(counter.values()) for group, counter in counts_by_group_model.items()})
    ordered_groups = _ordered_groups_from_counts(group_totals)
    preferred_models = ["openai_gpt-5.4-nano", "openai_gpt-5-mini", "gpt-5.4-nano", "gpt-5-mini"]
    ordered_models: list[str] = []
    for model in preferred_models:
        if any(counts_by_group_model[group].get(model, 0) for group in ordered_groups) and model not in ordered_models:
            ordered_models.append(model)
    for group_counts in counts_by_group_model.values():
        for model in sorted(group_counts):
            if model not in ordered_models:
                ordered_models.append(model)

    y_positions = list(range(len(ordered_groups)))
    bar_height = 0.34 if len(ordered_models) <= 2 else max(0.18, 0.72 / max(1, len(ordered_models)))
    max_count = max(max(counter.values(), default=0) for counter in counts_by_group_model.values())
    fig, ax = plt.subplots(figsize=(11.2, max(4.4, 0.88 * len(ordered_groups) + 1.8)))

    for model_index, model in enumerate(ordered_models):
        offset = (model_index - (len(ordered_models) - 1) / 2) * bar_height
        values = [counts_by_group_model[group].get(model, 0) for group in ordered_groups]
        bars = ax.barh(
            [y + offset for y in y_positions],
            values,
            height=bar_height * 0.88,
            color=MODEL_COLORS.get(model, "#6F6F6F"),
            label=pretty_model(model),
        )
        for bar, value in zip(bars, values):
            if value <= 0:
                continue
            ax.text(
                value + max(0.15, max_count * 0.012),
                bar.get_y() + bar.get_height() / 2,
                str(value),
                ha="left",
                va="center",
                fontsize=MODEL_GROUP_VALUE_FONTSIZE,
            )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(ordered_groups, fontsize=MODEL_GROUP_LABEL_FONTSIZE)
    ax.invert_yaxis()
    ax.set_xlabel("Answer-failure events", fontsize=MODEL_GROUP_AXIS_FONTSIZE)
    ax.set_title("Answer-Failure Groups by Model", fontsize=MODEL_GROUP_TITLE_FONTSIZE, pad=10)
    ax.grid(axis="x", alpha=0.22, linestyle="--", linewidth=0.7)
    ax.set_xlim(0, max(1, max_count + max(4, max_count * 0.16)))
    ax.tick_params(axis="x", labelsize=MODEL_GROUP_TICK_FONTSIZE)
    ax.legend(loc="upper right", frameon=True, framealpha=0.95, fontsize=MODEL_GROUP_LEGEND_FONTSIZE)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return True


def condition_group_counts(rows: list[dict]) -> tuple[list[tuple[str, str]], list[str], dict[tuple[str, str], Counter[str]]]:
    counts_by_variant_model_group: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    group_totals: Counter[str] = Counter()
    variant_lookup = {variant: label for label, variant in CONDITION_FIGURE_ORDER}

    for row in rows:
        variant = str(row.get("mode_variant", ""))
        if variant not in variant_lookup:
            continue
        group_name = clean_label(row.get("answer_failure_figure_group"))
        counts_by_variant_model_group[(variant, row["model_variant"])][group_name] += 1
        group_totals[group_name] += 1

    ordered_groups = _ordered_groups_from_counts(group_totals)
    active_conditions = [
        (label, variant)
        for label, variant in CONDITION_FIGURE_ORDER
        if any(
            sum(counts.values()) > 0
            for (observed_variant, _), counts in counts_by_variant_model_group.items()
            if observed_variant == variant
        )
    ]
    return active_conditions, ordered_groups, counts_by_variant_model_group


def ordered_condition_models(
    active_conditions: list[tuple[str, str]],
    counts_by_variant_model_group: dict[tuple[str, str], Counter[str]],
) -> list[tuple[str, str, str]]:
    observed_by_variant: dict[str, set[str]] = defaultdict(set)
    observed_models: set[str] = set()
    for variant, model in counts_by_variant_model_group:
        observed_models.add(model)
        if sum(counts_by_variant_model_group[(variant, model)].values()) > 0:
            observed_by_variant[variant].add(model)

    model_order: list[str] = []
    for model in CONDITION_FIGURE_MODEL_ORDER:
        if model in observed_models and model not in model_order:
            model_order.append(model)
    for model in sorted(observed_models):
        if model not in model_order:
            model_order.append(model)

    ordered: list[tuple[str, str, str]] = []
    for label, variant in active_conditions:
        for model in model_order:
            if model in observed_by_variant.get(variant, set()):
                ordered.append((label, variant, model))
    return ordered


def write_condition_group_figure(path: Path, rows: list[dict]) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    active_conditions, ordered_groups, counts_by_variant_model_group = condition_group_counts(rows)
    if not active_conditions or not ordered_groups:
        return False
    condition_models = ordered_condition_models(active_conditions, counts_by_variant_model_group)
    if not condition_models:
        return False

    gap = 1.55
    model_step = 1.0
    x_positions: list[float] = []
    current_x = 0.0
    previous_variant = None
    for _, variant, _ in condition_models:
        if previous_variant is not None and variant != previous_variant:
            current_x += gap
        x_positions.append(current_x)
        current_x += model_step
        previous_variant = variant

    totals = [
        sum(counts_by_variant_model_group.get((variant, model), Counter()).values())
        for _, variant, model in condition_models
    ]
    ymax = max(totals, default=0)
    ylimit = max(1, CONDITION_FIGURE_YMAX)
    fig, ax = plt.subplots(figsize=(14.2, 7.2))
    bottoms = [0 for _ in condition_models]

    for group_name in ordered_groups:
        values = [
            counts_by_variant_model_group.get((variant, model), Counter()).get(group_name, 0)
            for _, variant, model in condition_models
        ]
        color = GROUP_COLORS.get(group_name, "#7F7F7F")
        bars = ax.bar(
            x_positions,
            values,
            bottom=bottoms,
            color=color,
            label=CONDITION_FIGURE_LEGEND_LABELS.get(group_name, group_name),
            width=CONDITION_FIGURE_BAR_WIDTH,
        )
        label_color = _label_text_color(color)
        for bar, value, bottom in zip(bars, values, bottoms):
            if value <= 0:
                continue
            if value >= max(2, ymax * CONDITION_FIGURE_SEGMENT_LABEL_MIN_FRACTION):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bottom + value / 2,
                    str(value),
                    ha="center",
                    va="center",
                    fontsize=CONDITION_FIGURE_BAR_LABEL_FONTSIZE,
                    fontweight="medium",
                    color=label_color,
                )
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]

    run_ids: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        run_ids[(str(row.get("mode_variant", "")), row["model_variant"])].add(str(row.get("task_id", "")))
    for x_pos, total, (_, variant, model) in zip(x_positions, totals, condition_models):
        n_runs = len(run_ids.get((variant, model), set()))
        base_y = min(ylimit - 4.0, total + max(0.25, ylimit * 0.025))
        ax.text(
            x_pos,
            base_y,
            str(total),
            ha="center",
            va="bottom",
            fontsize=CONDITION_FIGURE_TOTAL_FONTSIZE,
            fontweight="medium",
        )
        ax.text(
            x_pos,
            base_y + ylimit * 0.052,
            f"n={n_runs}",
            ha="center",
            va="bottom",
            fontsize=CONDITION_FIGURE_TOTAL_FONTSIZE - 2.0,
            color="#555555",
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [CONDITION_FIGURE_MODEL_LABELS.get(model, pretty_model(model).replace("gpt-", "gpt\n")) for _, _, model in condition_models],
        fontsize=CONDITION_FIGURE_MODEL_LABEL_FONTSIZE,
    )
    for label, variant in active_conditions:
        variant_positions = [x for x, (_, observed_variant, _) in zip(x_positions, condition_models) if observed_variant == variant]
        if not variant_positions:
            continue
        center = sum(variant_positions) / len(variant_positions)
        ax.text(
            center,
            -0.135,
            CONDITION_FIGURE_LABELS.get(label, label),
            ha="center",
            va="top",
            fontsize=CONDITION_FIGURE_CONDITION_LABEL_FONTSIZE,
            transform=ax.get_xaxis_transform(),
        )
    ax.set_ylabel("Answer-failure events", fontsize=CONDITION_FIGURE_YLABEL_FONTSIZE)
    ax.grid(axis="y", alpha=0.22, linestyle="--", linewidth=0.7)
    ax.set_ylim(0, ylimit)
    ax.tick_params(axis="y", labelsize=CONDITION_FIGURE_YTICK_FONTSIZE)
    ax.legend(
        loc="upper right",
        frameon=True,
        framealpha=0.94,
        fontsize=CONDITION_FIGURE_LEGEND_FONTSIZE,
        ncol=CONDITION_FIGURE_LEGEND_COLUMNS,
        columnspacing=1.1,
        handlelength=1.7,
    )
    fig.tight_layout(rect=(0, 0.085, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return True


def _add_table(lines: list[str], headers: list[str], table_rows: list[list[str]]) -> None:
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in table_rows:
        lines.append("| " + " | ".join(str(cell).replace("\n", " ") for cell in row) + " |")
    lines.append("")


def _model_counts_text(counter: Counter[str]) -> str:
    return "; ".join(f"{pretty_model(model)}: {count}" for model, count in sorted(counter.items()))


def build_report(rows: list[dict], source_root: Path, csv_path: Path) -> str:
    lines = [
        "# Combined Answer-Failure Report",
        "",
        f"Source root: `{source_root}`",
        f"Combined CSV: `{csv_path.name}`",
        f"Total answer-failure events: {len(rows)}",
        "",
        "## Counts by Model",
        "",
    ]
    by_model = Counter(row["model_variant"] for row in rows)
    _add_table(lines, ["Model", "Events"], [[pretty_model(model), str(count)] for model, count in sorted(by_model.items())])

    lines.extend(["## Counts by Figure Group", ""])
    group_model_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        group_model_counts[row["answer_failure_figure_group"]][row["model_variant"]] += 1
    group_rows = sorted(
        group_model_counts.items(),
        key=lambda item: (-sum(item[1].values()), item[0]),
    )
    _add_table(
        lines,
        ["Figure Group", "Events", "Events by Model"],
        [[group, str(sum(counter.values())), _model_counts_text(counter)] for group, counter in group_rows],
    )

    lines.extend(["## Counts by Failure Type", ""])
    type_model_counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for row in rows:
        key = (row["answer_failure_figure_group"], clean_label(row.get("answer_failure_type")))
        type_model_counts[key][row["model_variant"]] += 1
    type_rows = sorted(type_model_counts.items(), key=lambda item: (-sum(item[1].values()), item[0]))
    _add_table(
        lines,
        ["Figure Group", "Failure Type", "Events", "Events by Model"],
        [[key[0], key[1], str(sum(counter.values())), _model_counts_text(counter)] for key, counter in type_rows],
    )

    lines.extend(["## Counts by Failure Type and Subtype", ""])
    subtype_counts: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    for row in rows:
        key = (
            row["answer_failure_figure_group"],
            clean_label(row.get("answer_failure_type")),
            clean_label(row.get("answer_failure_subtype"), "(none)"),
        )
        subtype_counts[key][row["model_variant"]] += 1
    subtype_rows = sorted(subtype_counts.items(), key=lambda item: (-sum(item[1].values()), item[0]))
    _add_table(
        lines,
        ["Figure Group", "Failure Type", "Subtype", "Events", "Events by Model"],
        [[key[0], key[1], key[2], str(sum(counter.values())), _model_counts_text(counter)] for key, counter in subtype_rows],
    )
    return "\n".join(lines).rstrip() + "\n"


def combine_answer_failures(
    source_root: str | Path = "results_semantic_answer_failures",
    output_root: str | Path | None = None,
    *,
    trusted_only: bool = True,
) -> dict[str, Path]:
    source_root = Path(source_root)
    output_root = Path(output_root) if output_root is not None else source_root.with_name(source_root.name + "_combined")
    sources = discover_source_files(source_root)
    if not sources:
        raise FileNotFoundError(f"No answer_failure_events.csv files found under {source_root}")

    rows, fieldnames = read_combined_rows(sources, source_root, trusted_only=trusted_only)
    csv_path = output_root / COMBINED_CSV_NAME
    report_path = output_root / COMBINED_REPORT_NAME
    figure_path = output_root / COMBINED_FIGURE_NAME
    condition_figure_path = output_root / COMBINED_CONDITION_FIGURE_NAME

    write_combined_csv(csv_path, rows, fieldnames)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(build_report(rows, source_root, csv_path))
    wrote_figure = write_model_group_figure(figure_path, rows)
    wrote_condition_figure = write_condition_group_figure(condition_figure_path, rows)

    outputs = {"csv_path": csv_path, "report_path": report_path}
    if wrote_figure:
        outputs["figure_path"] = figure_path
    if wrote_condition_figure:
        outputs["condition_figure_path"] = condition_figure_path
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default="results_semantic_answer_failures", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--include-untrusted", action="store_true")
    args = parser.parse_args()

    outputs = combine_answer_failures(
        source_root=args.source_root,
        output_root=args.output_root,
        trusted_only=not args.include_untrusted,
    )
    print(f"Wrote {outputs['csv_path']}")
    print(f"Wrote {outputs['report_path']}")
    if "figure_path" in outputs:
        print(f"Wrote {outputs['figure_path']}")
    if "condition_figure_path" in outputs:
        print(f"Wrote {outputs['condition_figure_path']}")


if __name__ == "__main__":
    main()
