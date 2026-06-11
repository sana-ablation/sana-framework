#!/usr/bin/env python3
"""Generate the default-plan vs ideal-plan comparison figure."""

from __future__ import annotations

import argparse
import csv
import shutil
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional


CANONICAL_PLAN_D_MODE = "search_i_results_i_pland_computei_k5_skills_off"
CANONICAL_PLAN_I_MODE = "search_i_results_i_plani_computei_k5_skills_off"

BENCHMARK_ROOTS = {
    "lakeqa": Path("agent_analysis/plan_default_analysis/logs"),
    "kramabench": Path("agent_analysis/plan_default_analysis/log-kramabench"),
}
MODEL_ORDER = ["openai_gpt-5.4-nano", "openai_gpt-5-mini"]
BENCHMARK_ORDER = ["lakeqa", "kramabench"]

BUCKET_ORDER = [
    "similar",
    "missing_details",
    "incomplete_plan",
    "operation_mismatch",
    "not_similar",
    "no_plan",
    "not_comparable",
]
BUCKET_LABELS = {
    "similar": "Similar",
    "missing_details": "Missing details",
    "incomplete_plan": "Incomplete",
    "operation_mismatch": "Mismatch",
    "not_similar": "Not similar",
    "no_plan": "No plan",
    "not_comparable": "Other n/c",
}
BUCKET_COLORS = {
    "similar": "#4C9F70",
    "missing_details": "#F2C14E",
    "incomplete_plan": "#E17C05",
    "operation_mismatch": "#C44E52",
    "not_similar": "#8B1E3F",
    "no_plan": "#7F7F7F",
    "not_comparable": "#B8B8B8",
}


def _normalize_plan_similarity(row: dict) -> Optional[str]:
    missing_type = str(row.get("missing_plan_type", "") or "").strip()
    if missing_type in {"missing_both", "missing_plan_i"}:
        return None
    if missing_type == "missing_plan_d":
        return "no_plan"
    label = str(row.get("plan_similarity", "") or "").strip()
    if label in BUCKET_ORDER:
        return label
    return "not_comparable"


def _pretty_model(model: str) -> str:
    return str(model).replace("openai_", "")


def _read_plan_similarity_csv(path: Path, benchmark: str) -> list[dict]:
    with path.open(newline="") as handle:
        rows = []
        for row in csv.DictReader(handle):
            normalized = dict(row)
            normalized["benchmark"] = benchmark
            rows.append(normalized)
        return rows


def load_plan_default_rows(input_root: Path) -> list[dict]:
    rows: list[dict] = []
    for benchmark, benchmark_root in BENCHMARK_ROOTS.items():
        root = input_root / benchmark_root if not benchmark_root.is_absolute() else benchmark_root
        csv_paths = sorted(root.rglob("plan_similarity.csv"))
        for path in csv_paths:
            rows.extend(_read_plan_similarity_csv(path, benchmark))
    return rows


def _summarize_rows(rows: Iterable[dict]) -> dict[tuple[str, str], dict]:
    summary: dict[tuple[str, str], dict] = {}
    for row in rows:
        if str(row.get("plan_d_mode", "")) != CANONICAL_PLAN_D_MODE:
            continue
        if str(row.get("plan_i_mode", "")) != CANONICAL_PLAN_I_MODE:
            continue
        bucket = _normalize_plan_similarity(row)
        if bucket is None:
            continue
        benchmark = str(row.get("benchmark", "unknown"))
        model = str(row.get("model_variant", "") or row.get("runner_model", "") or "unknown")
        entry = summary.setdefault((benchmark, model), {"benchmark": benchmark, "model": model, "n": 0, "counts": Counter()})
        entry["n"] += 1
        entry["counts"][bucket] += 1
    return summary


def _import_plot_libs():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _ordered_models(summary: dict[tuple[str, str], dict]) -> list[str]:
    models = []
    for model in MODEL_ORDER:
        if any(key[1] == model for key in summary):
            models.append(model)
    for _benchmark, model in sorted(summary):
        if model not in models:
            models.append(model)
    return models


def render_plan_default_similarity_figure(rows: Iterable[dict], output_path: Path) -> Path:
    summary = _summarize_rows(rows)
    if not summary:
        raise ValueError("No canonical plan_d vs plan_i rows found.")

    models = _ordered_models(summary)
    bar_labels: list[str] = []
    bar_entries: list[dict] = []
    x_positions: list[float] = []
    x = 0.0
    model_centers: list[tuple[float, str]] = []

    for model in models:
        start = x
        for benchmark in BENCHMARK_ORDER:
            entry = summary.get((benchmark, model))
            if entry is None:
                continue
            bar_entries.append(entry)
            bar_labels.append("LakeQA" if benchmark == "lakeqa" else "Kramabench")
            x_positions.append(x)
            x += 0.82
        if x > start:
            model_centers.append(((start + x - 0.82) / 2.0, model))
            x += 0.62

    plt = _import_plot_libs()
    fig, ax = plt.subplots(figsize=(8.8, 5.6))

    bottoms = [0.0 for _ in bar_entries]
    for bucket in BUCKET_ORDER:
        values = []
        for entry in bar_entries:
            n = int(entry["n"])
            values.append(100.0 * int(entry["counts"].get(bucket, 0)) / n if n else 0.0)
        if not any(value > 0 for value in values):
            continue
        bars = ax.bar(
            x_positions,
            values,
            bottom=bottoms,
            width=0.68,
            color=BUCKET_COLORS[bucket],
            label=BUCKET_LABELS[bucket],
            edgecolor="white",
            linewidth=0.6,
        )
        for bar, value, bottom in zip(bars, values, bottoms):
            if value < 7.0:
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bottom + value / 2,
                f"{value:.0f}%",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if bucket not in {"missing_details", "not_comparable"} else "black",
            )
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]

    for xpos, entry in zip(x_positions, bar_entries):
        ax.text(
            xpos,
            102.0,
            f"n={entry['n']}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_ylim(0, 108)
    ax.set_ylabel("Rows with ideal plan available (%)")
    ax.set_title("Default Plan vs Ideal Plan Similarity")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(bar_labels, rotation=0, ha="center")
    ax.grid(axis="y", alpha=0.22, linestyle="--", linewidth=0.7)

    for center, model in model_centers:
        ax.text(
            center,
            -0.16,
            _pretty_model(model),
            ha="center",
            va="top",
            transform=ax.get_xaxis_transform(),
            fontsize=10,
            fontweight="bold",
        )

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True, fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 0.82, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def export_figure(source_path: Path, destinations: Iterable[Path]) -> list[Path]:
    copied = []
    for destination in destinations:
        destination.mkdir(parents=True, exist_ok=True)
        target = destination / source_path.name
        if source_path.resolve() != target.resolve():
            shutil.copy2(source_path, target)
        copied.append(target)
    return copied


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", default=".")
    parser.add_argument("--output", default="agent_analysis/plan_default_analysis/figures/plan_default_similarity_by_benchmark_model.pdf")
    parser.add_argument("--paper-dir", default="sana_framework_paper/figures")
    parser.add_argument("--mirror-dir", default="paper_figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_plan_default_rows(Path(args.input_root))
    output_path = render_plan_default_similarity_figure(rows, Path(args.output))
    copied = export_figure(output_path, [Path(args.paper_dir), Path(args.mirror_dir)])
    print(f"Wrote {output_path}")
    for path in copied:
        print(f"Copied {path}")


if __name__ == "__main__":
    main()
