#!/usr/bin/env python3
"""Generate and export paper figures for semantic mode analyses."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from sana_analysis.run_mode_analysis_semantic import run_analysis
from sana_analysis.report_generator.combine_answer_failure_grouped_models import (
    COMBINED_CSV_NAME,
    COMBINED_CONDITION_FIGURE_NAME,
    COMBINED_FIGURE_NAME,
    write_condition_group_figure,
    write_model_group_figure,
)


SEARCH_VARIANTS = {
    "search_n_results_i_plani_computei_k5_skills_off": ("NII", "BM25"),
    "search_d_results_i_plani_computei_k5_skills_off": ("DII", "Pneuma"),
    "search_i_results_i_plani_computei_k5_skills_off": ("III", "Ideal"),
}
SEARCH_ORDER = ["NII", "DII", "III"]
SEARCH_COLORS = {"NII": "#4C78A8", "DII": "#F58518", "III": "#54A24B"}
SEARCH_DISPLAY_LABELS = {"NII": "BM25", "DII": "PNEUMA", "III": "Ideal"}
SEARCH_LABEL_OFFSETS = {"NII": (8, -16), "DII": (8, 0), "III": (8, 12)}


def _search_label_offset(model: str, label: str) -> tuple[int, int]:
    if label == "DII" and "gpt-5-mini" in model.lower():
        return (-10, 0)
    return SEARCH_LABEL_OFFSETS.get(label, (7, 7))


def _search_label_alignment(model: str, label: str) -> str:
    if label == "DII" and "gpt-5-mini" in model.lower():
        return "right"
    return "left"


@dataclass(frozen=True)
class BenchmarkDefaults:
    benchmark: str
    results_dir: Path
    base_results_dir: Path
    traces_dir: Path
    tasks_dir: Path
    analysis_dir: Path
    turn_waste_grouped_dir: Path
    answer_failure_combined_dir: Path
    semantic_delta_name: str


@dataclass(frozen=True)
class GeneratorConfig:
    benchmark: str
    results_dir: Path
    base_results_dir: Path
    traces_dir: Path
    tasks_dir: Path
    analysis_dir: Path
    turn_waste_grouped_dir: Path
    answer_failure_combined_dir: Path
    model_filter: Optional[str]
    force: bool
    paper_dir: Path
    mirror_dir: Path
    agent_analysis_dir: Path


def _benchmark_defaults(benchmark: str) -> BenchmarkDefaults:
    if benchmark == "lakeqa":
        return BenchmarkDefaults(
            benchmark="lakeqa",
            results_dir=Path("results_semantic/modes"),
            base_results_dir=Path("results/modes"),
            traces_dir=Path("results/traces/modes"),
            tasks_dir=Path("benchmarks/lakeqa/tasks-mini/tasks"),
            analysis_dir=Path("analysis_results_mode_semantic"),
            turn_waste_grouped_dir=Path("results_semantic_turn_waste_grouped"),
            answer_failure_combined_dir=Path("results_semantic_answer_failures_combined"),
            semantic_delta_name="lakeqa_semantic_delta_ablation.pdf",
        )
    if benchmark == "kramabench":
        return BenchmarkDefaults(
            benchmark="kramabench",
            results_dir=Path("results-kramabench_semantic/modes"),
            base_results_dir=Path("results-kramabench/modes"),
            traces_dir=Path("results-kramabench/traces/modes"),
            tasks_dir=Path("benchmarks/kramabench/tasks-mini/tasks"),
            analysis_dir=Path("analysis_results_mode_kramabench_semantic"),
            turn_waste_grouped_dir=Path("results-kramabench_semantic_turn_waste_grouped"),
            answer_failure_combined_dir=Path("results-kramabench_semantic_answer_failures_combined"),
            semantic_delta_name="kramabench_semantic_delta_ablation.pdf",
        )
    raise ValueError(f"Unsupported benchmark: {benchmark}")


def _selected_existing_figures(benchmark: str) -> list[str]:
    defaults = _benchmark_defaults(benchmark)
    return [
        "answer_failure_groups_by_model.pdf",
        "answer_failure_groups_by_condition.pdf",
        defaults.semantic_delta_name,
    ]


LEGACY_FIGURE_FALLBACKS = {
    "answer_failure_groups_by_model.pdf": ["fig06_answer_failure_groups_by_model.pdf"],
    "answer_failure_groups_by_condition.pdf": ["fig06b_answer_failure_groups_by_condition.pdf"],
    "lakeqa_semantic_delta_ablation.pdf": [
        "fig21b_lakeqa_semantic_delta_ablation.pdf",
        "fig21b_semantic_delta_ablation_compact.pdf",
        "semantic_delta_ablation_compact.pdf",
    ],
    "kramabench_semantic_delta_ablation.pdf": [
        "fig21b_krama_semantic_delta_ablation.pdf",
        "fig21b_semantic_delta_ablation_compact.pdf",
        "semantic_delta_ablation_compact.pdf",
    ],
}


def _search_variant_label(variant: str) -> Optional[str]:
    payload = SEARCH_VARIANTS.get(str(variant))
    return payload[0] if payload else None


def _parse_model_filters(model_filter: Optional[str]) -> Optional[list[str]]:
    if not model_filter:
        return None
    filters = [token.strip().lower() for token in model_filter.split(",") if token.strip()]
    return filters or None


def _keep_model(model: str, model_filters: Optional[list[str]]) -> bool:
    if not model_filters:
        return True
    lowered = model.lower()
    return any(token in lowered for token in model_filters)


def _condition_keys_from_eval_results(root: Path, model_filter: Optional[str]) -> set[str]:
    model_filters = _parse_model_filters(model_filter)
    keys: set[str] = set()
    if not root.exists():
        return keys
    for csv_path in root.rglob("eval_results.csv"):
        rel = csv_path.relative_to(root)
        parts = rel.parts
        if len(parts) < 3:
            continue
        model = parts[-3]
        variant = parts[-2]
        if _keep_model(model, model_filters):
            keys.add(f"{model}/{variant}")
    return keys


def _turn_waste_scope_complete(
    results_dir: Path,
    turn_waste_grouped_dir: Path,
    model_filter: Optional[str],
) -> bool:
    expected = _condition_keys_from_eval_results(results_dir, model_filter)
    if not expected:
        return False
    available = _condition_keys_from_eval_results(turn_waste_grouped_dir, model_filter)
    return expected.issubset(available)


def _as_float(value: object) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def _model_sort_key(model: str) -> tuple[int, str]:
    lowered = model.lower()
    if "gpt-5-mini" in lowered:
        return (0, model)
    if "gpt-5.4-nano" in lowered:
        return (1, model)
    return (2, model)


def _pretty_model(model: str) -> str:
    return str(model).replace("openai_", "").replace("openai/", "")


def _search_panel_xlim(model: str, values: list[float], fallback: tuple[float, float]) -> tuple[float, float]:
    if "gpt-5-mini" not in model.lower():
        return fallback
    return _padded_limits(
        values,
        lower_bound=0.0,
        upper_bound=max(10.0, max(values, default=1.0) + 5.0),
        pad_fraction=0.35,
        min_span=1.2,
    )


def _load_json_rows(path: Path) -> list[dict]:
    with path.open() as handle:
        rows = json.load(handle)
    if not isinstance(rows, list):
        raise ValueError(f"Expected {path} to contain a list")
    return [dict(row) for row in rows]


def _load_search_figure_data(analysis_dir: Path) -> dict:
    summary_path = analysis_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)

    summary_by_model: dict[str, list[dict]] = {}
    for row in _load_json_rows(summary_path):
        label = _search_variant_label(str(row.get("variant", "")))
        if label is None:
            continue
        model = str(row.get("model", "unknown"))
        avg_search_calls = _as_float(row.get("avg_search_calls"))
        d_ret = _as_float(row.get("D_ret"))
        d_acc = _as_float(row.get("D_acc_recall", row.get("D_acc")))
        if avg_search_calls is None or d_ret is None or d_acc is None:
            continue
        summary_by_model.setdefault(model, []).append(
            {
                "label": label,
                "variant": str(row.get("variant", "")),
                "avg_search_calls": avg_search_calls,
                "d_ret": d_ret,
                "d_acc": d_acc,
            }
        )

    for rows in summary_by_model.values():
        rows.sort(key=lambda row: SEARCH_ORDER.index(str(row["label"])))

    per_task_calls: dict[tuple[str, str, str], dict[int, float]] = {}
    call_csv = analysis_dir / "search_call_cumulative_retrieval.csv"
    if call_csv.exists():
        with call_csv.open(newline="") as handle:
            for row in csv.DictReader(handle):
                label = _search_variant_label(str(row.get("variant", "")))
                if label is None:
                    continue
                model = str(row.get("model", "unknown"))
                task_id = str(row.get("task_id", ""))
                recall = _as_float(row.get("cumulative_search_gold_recall"))
                if not task_id or recall is None:
                    continue
                try:
                    call_index = int(float(row.get("search_call_index")))
                except (TypeError, ValueError):
                    continue
                per_task_calls.setdefault((model, label, task_id), {})[call_index] = recall

    curves: dict[tuple[str, str], dict[int, float]] = {}
    tasks_by_model_label: dict[tuple[str, str], list[dict[int, float]]] = {}
    for (model, label, _task_id), calls in per_task_calls.items():
        tasks_by_model_label.setdefault((model, label), []).append(calls)

    for key, task_calls in tasks_by_model_label.items():
        max_call = max((max(calls.keys()) for calls in task_calls if calls), default=0)
        if max_call <= 0:
            continue
        dense_curve: dict[int, float] = {}
        for call_index in range(1, max_call + 1):
            values: list[float] = []
            for calls in task_calls:
                last_value = 0.0
                for candidate in range(1, call_index + 1):
                    if candidate in calls:
                        last_value = calls[candidate]
                values.append(last_value)
            dense_curve[call_index] = sum(values) / len(values) if values else 0.0
        curves[key] = dense_curve

    return {"summary_by_model": summary_by_model, "curves": curves}


def _import_plot_libs():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception as exc:
        raise RuntimeError("matplotlib is required to render paper figures") from exc


def _padded_limits(
    values: Iterable[float],
    *,
    lower_bound: float,
    upper_bound: float,
    pad_fraction: float = 0.18,
    min_span: float = 8.0,
) -> tuple[float, float]:
    values = [float(value) for value in values]
    if not values:
        return lower_bound, upper_bound
    low = min(values)
    high = max(values)
    span = max(high - low, min_span)
    center = (low + high) / 2.0
    low = center - span / 2.0
    high = center + span / 2.0
    pad = span * pad_fraction
    return max(lower_bound, low - pad), min(upper_bound, high + pad)


def render_search_efficiency_figure(analysis_dir: Path, benchmark: str, output_path: Path) -> Path:
    data = _load_search_figure_data(analysis_dir)
    summary_by_model: dict[str, list[dict]] = data["summary_by_model"]
    # curves: dict[tuple[str, str], dict[int, float]] = data["curves"]
    models = [model for model in sorted(summary_by_model, key=_model_sort_key) if summary_by_model[model]]
    if not models:
        raise ValueError(f"No canonical NII/DII/III rows found in {analysis_dir / 'summary.json'}")

    plt = _import_plot_libs()
    all_x_values = [
        float(item["avg_search_calls"])
        for model in models
        for item in summary_by_model[model]
    ]
    all_y_values = [
        float(item["d_ret"]) * 100.0
        for model in models
        for item in summary_by_model[model]
    ]
    scatter_xlim = _padded_limits(
        all_x_values,
        lower_bound=0.0,
        upper_bound=max(10.0, max(all_x_values, default=1.0) + 5.0),
        pad_fraction=0.22,
        min_span=3.0,
    )
    scatter_ylim = _padded_limits(
        all_y_values,
        lower_bound=0.0,
        upper_bound=100.0,
        pad_fraction=0.22,
        min_span=14.0,
    )
    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(models),
        figsize=(4.35 * len(models), 3.25),
        squeeze=False,
    )
    for col_index, model in enumerate(models):
        scatter_ax = axes[0][col_index]
        rows = summary_by_model[model]
        scatter_x_values = [float(item["avg_search_calls"]) for item in rows]

        for item in rows:
            label = str(item["label"])
            display_label = SEARCH_DISPLAY_LABELS.get(label, label)
            color = SEARCH_COLORS[label]
            x = float(item["avg_search_calls"])
            d_ret_pct = float(item["d_ret"]) * 100.0
            d_acc_pct = float(item["d_acc"]) * 100.0
            ring_size = 320.0 + 1600.0 * max(0.0, min(1.0, float(item["d_acc"])))
            scatter_ax.scatter([x], [d_ret_pct], s=125, color=color, zorder=4)
            scatter_ax.scatter(
                [x],
                [d_ret_pct],
                s=ring_size * 1.08,
                facecolors="none",
                edgecolors=color,
                linewidths=1.8,
                alpha=0.45,
                zorder=3,
            )
            scatter_ax.annotate(
                f"{display_label}\nD_ret {d_ret_pct:.0f}%\nD_acc {d_acc_pct:.0f}%",
                (x, d_ret_pct),
                textcoords="offset points",
                xytext=_search_label_offset(model, label),
                fontsize=9.5,
                ha=_search_label_alignment(model, label),
            )

        # Cumulative recall panel intentionally disabled. The previous version
        # forward-filled each task after its final search call, which made the
        # curve depend on an interpretation that is not needed for this figure.
        # for label in SEARCH_ORDER:
        #     curve = curves.get((model, label), {})
        #     if not curve:
        #         continue
        #     xs = sorted(curve)
        #     ys = [curve[x] * 100.0 for x in xs]
        #     model_curve_calls.extend(xs)
        #     model_curve_values.extend(ys)
        #     curve_ax.plot(xs, ys, marker="o", linewidth=2.0, markersize=4, color=SEARCH_COLORS[label], label=label)

        scatter_ax.set_xlim(*_search_panel_xlim(model, scatter_x_values, scatter_xlim))
        scatter_ax.set_ylim(*scatter_ylim)
        scatter_ax.grid(True, alpha=0.25, linewidth=0.6)
        scatter_ax.set_title(f"{_pretty_model(model)}: Retrieval Coverage", fontsize=12, pad=7)
        scatter_ax.set_xlabel("Avg search calls / task", fontsize=11)
        scatter_ax.set_ylabel("Avg D_ret (%)", fontsize=11)
        scatter_ax.tick_params(axis="both", labelsize=10)

        # Cumulative recall axis setup intentionally disabled with the plot.
        # curve_xmax = max(model_curve_calls, default=1)
        # curve_ax.set_xlim(1, max(1.0, float(curve_xmax)))
        # curve_ax.set_ylim(
        #     *_padded_limits(
        #         model_curve_values,
        #         lower_bound=0.0,
        #         upper_bound=100.0,
        #         pad_fraction=0.16,
        #         min_span=20.0,
        #     )
        # )
        # curve_ax.grid(True, alpha=0.25, linewidth=0.6)
        # curve_ax.set_title(f"{_pretty_model(model)}: Cumulative D_ret", fontsize=11)
        # curve_ax.set_xlabel("Search call")
        # curve_ax.set_ylabel("Mean cumulative D_ret (%)")
        # curve_ax.legend(loc="lower right", fontsize=8, frameon=True)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def refresh_answer_failure_figures(answer_failure_combined_dir: Path) -> list[Path]:
    combined_csv_path = answer_failure_combined_dir / COMBINED_CSV_NAME
    if not combined_csv_path.exists():
        print(f"Missing optional answer-failure CSV: {combined_csv_path}")
        return []

    with combined_csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        print(f"No rows in optional answer-failure CSV: {combined_csv_path}")
        return []

    refreshed: list[Path] = []
    figure_path = answer_failure_combined_dir / COMBINED_FIGURE_NAME
    if write_model_group_figure(figure_path, rows):
        refreshed.append(figure_path)

    condition_figure_path = answer_failure_combined_dir / COMBINED_CONDITION_FIGURE_NAME
    if write_condition_group_figure(condition_figure_path, rows):
        refreshed.append(condition_figure_path)
    return refreshed


def export_paper_figures(
    *,
    benchmark: str,
    analysis_dir: Path,
    search_figure_path: Path,
    destinations: Iterable[Path],
    fallback_dirs: Iterable[Path] = (),
) -> list[Path]:
    if not search_figure_path.exists():
        raise FileNotFoundError(search_figure_path)

    copied: list[Path] = []
    source_specs: list[tuple[Path, str]] = [(search_figure_path, search_figure_path.name)]
    figure_dir = analysis_dir / "figures"
    fallback_dirs = list(fallback_dirs)
    for filename in _selected_existing_figures(benchmark):
        candidates = [figure_dir / filename]
        candidates.extend(figure_dir / legacy_name for legacy_name in LEGACY_FIGURE_FALLBACKS.get(filename, []))
        candidates.extend(fallback_dir / filename for fallback_dir in fallback_dirs)
        for fallback_dir in fallback_dirs:
            candidates.extend(fallback_dir / legacy_name for legacy_name in LEGACY_FIGURE_FALLBACKS.get(filename, []))
        source = next((path for path in candidates if path.exists()), None)
        if source is not None:
            source_specs.append((source, filename))
            continue
        print(f"Missing optional figure: {candidates[0]}")

    for destination in destinations:
        destination.mkdir(parents=True, exist_ok=True)
        for source, target_name in source_specs:
            target = destination / target_name
            if source.resolve() != target.resolve():
                shutil.copy2(source, target)
            copied.append(target)
    return copied


AGENT_ANALYSIS_EXPORT_SUFFIXES = {".csv", ".json", ".pdf", ".md"}
AGENT_ANALYSIS_SKIP_DIRS = {"tmp", "logs", "log-kramabench", "__pycache__"}
AGENT_ANALYSIS_EXPORT_DIRS = {
    "follow_plan_analysis",
    "plan_default_analysis",
    "trajectory_ideal_context_analysis",
    "trajectory_pair_analysis",
}


def _is_agent_analysis_export(path: Path, root: Path) -> bool:
    if path.name.startswith("."):
        return False
    if path.suffix not in AGENT_ANALYSIS_EXPORT_SUFFIXES:
        return False
    rel_parts = path.relative_to(root).parts
    if not rel_parts or rel_parts[0] not in AGENT_ANALYSIS_EXPORT_DIRS:
        return False
    if any(part in AGENT_ANALYSIS_SKIP_DIRS for part in rel_parts[:-1]):
        return False
    if path.suffix == ".json" and path.name.endswith("_journal.json"):
        return False
    return True


def export_agent_analysis_results(*, agent_analysis_root: Path, destination: Path) -> list[Path]:
    """Mirror paper-relevant agent_analysis summaries/results into paper_figures."""
    if not agent_analysis_root.exists():
        print(f"Missing agent_analysis directory: {agent_analysis_root}")
        return []

    copied: list[Path] = []
    for source in sorted(path for path in agent_analysis_root.rglob("*") if path.is_file()):
        if not _is_agent_analysis_export(source, agent_analysis_root):
            continue
        target = destination / source.relative_to(agent_analysis_root)
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
        copied.append(target)
    return copied


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=["lakeqa", "kramabench"], required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--base-results-dir", default=None)
    parser.add_argument("--traces-dir", default=None)
    parser.add_argument("--tasks-dir", default=None)
    parser.add_argument("--analysis-dir", default=None)
    parser.add_argument("--turn-waste-grouped-dir", default=None)
    parser.add_argument("--answer-failure-combined-dir", default=None)
    parser.add_argument("--model-filter", default=None)
    parser.add_argument("--paper-dir", default="sana_framework_paper/figures")
    parser.add_argument("--mirror-dir", default="paper_figures")
    parser.add_argument("--agent-analysis-dir", default="agent_analysis")
    return parser.parse_args()


def _resolve_config(args: argparse.Namespace) -> GeneratorConfig:
    defaults = _benchmark_defaults(args.benchmark)
    return GeneratorConfig(
        benchmark=args.benchmark,
        results_dir=Path(args.results_dir) if args.results_dir else defaults.results_dir,
        base_results_dir=Path(args.base_results_dir) if args.base_results_dir else defaults.base_results_dir,
        traces_dir=Path(args.traces_dir) if args.traces_dir else defaults.traces_dir,
        tasks_dir=Path(args.tasks_dir) if args.tasks_dir else defaults.tasks_dir,
        analysis_dir=Path(args.analysis_dir) if args.analysis_dir else defaults.analysis_dir,
        turn_waste_grouped_dir=Path(args.turn_waste_grouped_dir)
        if args.turn_waste_grouped_dir
        else defaults.turn_waste_grouped_dir,
        answer_failure_combined_dir=Path(args.answer_failure_combined_dir)
        if args.answer_failure_combined_dir
        else defaults.answer_failure_combined_dir,
        model_filter=args.model_filter,
        force=bool(args.force),
        paper_dir=Path(args.paper_dir),
        mirror_dir=Path(args.mirror_dir),
        agent_analysis_dir=Path(args.agent_analysis_dir),
    )


def ensure_analysis_outputs(config: GeneratorConfig) -> None:
    summary_path = config.analysis_dir / "summary.json"
    should_run = config.force or not summary_path.exists()
    if not should_run:
        print(f"Reusing existing analysis output: {config.analysis_dir}")
        return

    print(f"Running semantic mode analysis for {config.benchmark}...")
    run_analysis(
        results_dir=str(config.results_dir),
        base_results_dir=str(config.base_results_dir),
        turn_waste_grouped_dir=None,
        traces_dir=str(config.traces_dir),
        tasks_dir=str(config.tasks_dir),
        output_dir=str(config.analysis_dir),
        model_filter=config.model_filter,
        no_figures=summary_path.exists() and not config.force,
    )


def main() -> None:
    config = _resolve_config(parse_args())
    ensure_analysis_outputs(config)
    search_figure_path = (
        config.analysis_dir
        / f"search_efficiency_cumulative_retrieval_{config.benchmark}.pdf"
    )
    try:
        render_search_efficiency_figure(config.analysis_dir, config.benchmark, search_figure_path)
    except ValueError as exc:
        fallback_search_figure = (
            Path("paper_figures")
            / f"search_efficiency_cumulative_retrieval_{config.benchmark}.pdf"
        )
        if not fallback_search_figure.exists():
            raise
        print(f"Skipping search-efficiency render: {exc}")
        print(f"Using existing search-efficiency figure: {fallback_search_figure}")
        search_figure_path = fallback_search_figure
    refreshed = refresh_answer_failure_figures(config.answer_failure_combined_dir)
    if refreshed:
        print("Refreshed answer-failure figures:")
        for path in refreshed:
            print(f"  {path}")
    copied = export_paper_figures(
        benchmark=config.benchmark,
        analysis_dir=config.analysis_dir,
        search_figure_path=search_figure_path,
        destinations=[config.paper_dir, config.mirror_dir],
        fallback_dirs=[config.answer_failure_combined_dir, config.paper_dir],
    )
    print("Exported paper figures:")
    for path in copied:
        print(f"  {path}")
    agent_copied = export_agent_analysis_results(
        agent_analysis_root=config.agent_analysis_dir,
        destination=config.mirror_dir / "agent_analysis",
    )
    print("Exported agent_analysis results:")
    for path in agent_copied:
        print(f"  {path}")


if __name__ == "__main__":
    main()
