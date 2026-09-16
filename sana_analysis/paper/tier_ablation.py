#!/usr/bin/env python3
"""Replicate-averaged per-axis ablation figure and table for the model-tier sweeps.

This is the tracked home of two artifacts that were previously produced by
gitignored one-offs inside `experiments/2026-08-31-model-tiers-subset20b/`
(`make_axis_delta_figure.py` and `gen_tex.py`):

  * a grid figure -- one row per model, one column per SANA axis, each cell a
    horizontal bar chart of that axis's modes with the other two held at Ideal;
  * a LaTeX `tabular` of the same seven leave-one-out cells per model, with the
    effect of each degradation relative to the model's reference cell.

Why this does not go through `run_mode_analysis` -> `summary.json`
-----------------------------------------------------------------
`paper.delta_figures` and `paper.export` both read an analysis bundle, which
has exactly one row per (model, variant). These sweeps ran every cell
**three times**, and the whole point of the artifacts is the mean over those
replicate rounds -- a dimension the bundle does not carry. So the round trees
are read directly here. What is shared rather than reimplemented:

  * `sana_analysis.variants.find_variant` resolves each cell by axis predicate.
    It decodes all four naming generations and raises if a predicate matches
    two directories, which is the double-counting this module must not do.
  * `paper.delta_figures.plot_horizontal_delta_bars` (with its palette and its
    `NN.N% (+D.D%)` label) draws the panels, so this figure and the bundle's
    delta figures cannot drift apart visually.

Accuracy is over COMPLETED rows only. An earlier version divided correct
answers by *all* rows, so tasks that crashed before reaching the model counted
as wrong answers and turned two infrastructure failures into a plausible-looking
table. Errored rows are dropped, and a cell with too few completions is refused
rather than printed.

Label vocabulary
----------------
The plan axis value `standard` is printed "Default" here. `delta_figures`,
`run_mode_analysis` and `variants.CONDITION_ORDER` deliberately call the same
condition "Standard Plan"; this module keeps the paper figure's own spelling
rather than changing that canonical vocabulary. Both mappings resolve to the
same axis predicate, which is the part that decides what gets read off disk.

    python -m sana_analysis.paper.tier_ablation --experiment-dir <dir>
"""

from __future__ import annotations

import argparse
import csv
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from sana_analysis.paper.delta_figures import plot_horizontal_delta_bars
from sana_analysis.paper.export import model_display_name
from sana_analysis.variants import find_variant

# The replicate round trees, in order. `<tree>_semantic` is the audited
# parallel tree the semantic-eval-auditor writes.
ROUND_TREES: Tuple[str, ...] = ("results", "results-rep2", "results-rep3")

# Below this a 20-task cell cannot support a number.
MIN_COMPLETE = 15

# One task is worth this many percentage points at n=20.
PP_PER_TASK = 5.0

FIGURE_TITLE = "20 tasks subset"

MODEL_ORDER: Tuple[str, ...] = (
    "openai_gpt-5.4-nano",
    "openai_gpt-5-mini",
    "openai_gpt-5.2",
    "openai_gpt-5.6-luna",
    "openai_gpt-5.6-sol",
)

# ---------------------------------------------------------------------------
# The axis -> label mapping. This is the specification of both artifacts.
#
# Every cell is leave-one-out: one axis degraded, the other two held at
# `ideal`. Only the three ablation axes take part in cell identity -- the
# `results`, `k` and `skills` modifiers are left unconstrained so a predicate
# cannot over-match on a modifier (see `variants.Variant.matches`).
# ---------------------------------------------------------------------------

REFERENCE_AXES: Dict[str, str] = dict(search="ideal", plan="ideal", compute="ideal")

_PLAN_NAIVE = dict(search="ideal", plan="naive", compute="ideal")
_PLAN_STANDARD = dict(search="ideal", plan="standard", compute="ideal")
_SEARCH_NAIVE = dict(search="naive", plan="ideal", compute="ideal")
_SEARCH_STANDARD = dict(search="standard", plan="ideal", compute="ideal")
_SEARCH_PRELOADED = dict(search="preloaded", plan="ideal", compute="ideal")
_COMPUTE_STANDARD = dict(search="ideal", plan="ideal", compute="standard")

# Figure panels: (column heading, [(printed label, axes)]) with the panel's
# BASELINE FIRST -- the weakest mode of that axis. Every bar's delta is against
# that first bar, so the first bar always reads (+0.0%) and neutral grey.
#
# Preloaded is kept in the Search panel: handing over the gold sources is not a
# retrieval condition, but it is the ceiling the retrieval modes are read
# against.
AXIS_PANELS: Tuple[Tuple[str, Tuple[Tuple[str, Dict[str, str]], ...]], ...] = (
    ("Plan", (
        ("No Plan", _PLAN_NAIVE),
        ("Default", _PLAN_STANDARD),
        ("Ideal", REFERENCE_AXES),
    )),
    ("Search", (
        ("BM25", _SEARCH_NAIVE),
        ("PNEUMA", _SEARCH_STANDARD),
        ("Ideal", REFERENCE_AXES),
        ("Preloaded", _SEARCH_PRELOADED),
    )),
    ("Data Analysis", (
        ("Standard", _COMPUTE_STANDARD),
        ("Ideal", REFERENCE_AXES),
    )),
)

# Table rows: the reference cell first, then each degradation. Deltas are
# against the reference, not against a per-axis baseline -- the table and the
# figure answer different questions from the same cells.
TABLE_CONDITIONS: Tuple[Tuple[str, Dict[str, str]], ...] = (
    ("Reference (all ideal)", REFERENCE_AXES),
    ("Search: BM25", _SEARCH_NAIVE),
    ("Search: PNEUMA", _SEARCH_STANDARD),
    ("Search: Preloaded", _SEARCH_PRELOADED),
    ("Plan: No Plan", _PLAN_NAIVE),
    ("Plan: Default", _PLAN_STANDARD),
    ("Data An.: Standard", _COMPUTE_STANDARD),
)

# The bar rows a panel reserves, so a two-mode panel and a four-mode panel
# share one bar pitch across the grid.
PANEL_SLOTS = max(len(modes) for _title, modes in AXIS_PANELS)

# Reproduces the one-off's geometry: a label sits inside its bar once the bar
# is long enough to hold it, which `plot_horizontal_delta_bars` expresses as
# `value + 1.2 > x_limit - inside_label_reserved`, i.e. above ~43.8%.
FIGURE_X_LIMIT = 105.0
FIGURE_INSIDE_LABEL_RESERVED = 60.0

FIGURE_RC_PARAMS = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "pdf.fonttype": 42,
    "figure.facecolor": "#FFFFFF",
    "axes.facecolor": "#FFFFFF",
    "savefig.facecolor": "#FFFFFF",
}
FIGURE_RULE_COLOR = "#333333"

METRIC_COLUMNS = {"exact": "exact_match", "semantic": "semantic_match"}


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Cell:
    """One (model, condition) cell, averaged over the rounds that exist.

    `rounds` is carried explicitly rather than folded away: a cell averaged
    over two rounds and one averaged over three are not the same measurement,
    and both artifacts say which they printed.
    """

    label: str
    percent: float
    rounds: int
    per_round: Tuple[float, ...]
    n_tasks: int
    n_correct: int
    cycles_per_task: Optional[float]
    cost_per_task: Optional[float]

    @property
    def sd(self) -> Optional[float]:
        """Spread across rounds, or None for a cell run once."""
        if len(self.per_round) < 2:
            return None
        return statistics.stdev(self.per_round)


def tree_dir(experiment_dir: Path, tree: str, metric: str) -> Path:
    """The results root for one round under one metric."""
    suffix = "_semantic" if metric == "semantic" else ""
    return Path(experiment_dir) / f"{tree}{suffix}" / "modes"


def discover_models(
    experiment_dir: Path,
    *,
    rounds: Sequence[str] = ROUND_TREES,
    order: Sequence[str] = MODEL_ORDER,
) -> List[str]:
    """Model directory names present in any round, in `order` then alphabetically."""
    seen: set[str] = set()
    for tree in rounds:
        root = tree_dir(experiment_dir, tree, "exact")
        if root.is_dir():
            seen.update(path.name for path in root.iterdir() if path.is_dir())
    ranked = [name for name in order if name in seen]
    return ranked + sorted(seen - set(ranked))


def load_round_rows(
    experiment_dir: Path,
    tree: str,
    model: str,
    axes: Mapping[str, str],
    metric: str,
    *,
    min_complete: int = MIN_COMPLETE,
) -> Optional[List[dict]]:
    """Completed rows for one cell in one round, or None if it cannot be read.

    None covers every way a cell can be absent: the round, the model or the
    variant directory is missing, the CSV has not been written yet (a live
    sweep creates the directory first), the metric column is absent, or too few
    tasks completed to support a percentage.
    """
    model_dir = tree_dir(experiment_dir, tree, metric) / model
    if not model_dir.is_dir():
        return None
    names = [path.name for path in model_dir.iterdir() if path.is_dir()]
    # Raises on an ambiguous match rather than picking one silently.
    variant = find_variant(names, **dict(axes))
    if variant is None:
        return None
    csv_path = model_dir / variant / "eval_results.csv"
    if not csv_path.is_file():
        return None
    with csv_path.open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if not (row.get("error") or "").strip()]
    if len(rows) < min_complete:
        return None
    if METRIC_COLUMNS[metric] not in (rows[0] or {}):
        return None
    return rows


def load_cell(
    experiment_dir: Path,
    model: str,
    label: str,
    axes: Mapping[str, str],
    metric: str,
    *,
    rounds: Sequence[str] = ROUND_TREES,
    min_complete: int = MIN_COMPLETE,
) -> Optional[Cell]:
    """One cell averaged over the rounds that exist, or None if no round does."""
    column = METRIC_COLUMNS[metric]
    per_round: List[float] = []
    n_tasks = n_correct = 0
    cost = cycles = 0.0
    for tree in rounds:
        rows = load_round_rows(
            experiment_dir, tree, model, axes, metric, min_complete=min_complete
        )
        if not rows:
            continue
        correct = sum(1 for row in rows if _as_float(row.get(column)) >= 1)
        per_round.append(100.0 * correct / len(rows))
        n_tasks += len(rows)
        n_correct += correct
        # cost_usd is the main agent only. The ideal modes bill hidden helper
        # agents separately and those dominate: dropping them made
        # compute=ideal look CHEAPER than compute=standard, when it is roughly
        # twice the price.
        cost += sum(
            _as_float(row.get("total_cost_with_all_subagents_usd"))
            or _as_float(row.get("cost_usd"))
            for row in rows
        )
        cycles += sum(_as_float(row.get("cycle_count")) for row in rows)
    if not per_round:
        return None
    return Cell(
        label=label,
        percent=statistics.mean(per_round),
        rounds=len(per_round),
        per_round=tuple(per_round),
        n_tasks=n_tasks,
        n_correct=n_correct,
        cycles_per_task=cycles / n_tasks if n_tasks else None,
        cost_per_task=cost / n_tasks if n_tasks else None,
    )


def resolve_metric(
    experiment_dir: Path,
    models: Sequence[str],
    *,
    rounds: Sequence[str] = ROUND_TREES,
    min_complete: int = MIN_COMPLETE,
) -> str:
    """"semantic" only when every readable raw cell is also audited, else "exact".

    Mixing a semantic cell and an exact cell inside one figure or one table
    column would put incomparable numbers on a shared axis, so the choice is
    made once for all of them.
    """
    conditions = {label: axes for label, axes in TABLE_CONDITIONS}
    seen = 0
    for tree in rounds:
        for model in models:
            for axes in conditions.values():
                raw = load_round_rows(
                    experiment_dir, tree, model, axes, "exact", min_complete=min_complete
                )
                if raw is None:
                    continue
                seen += 1
                audited = load_round_rows(
                    experiment_dir, tree, model, axes, "semantic", min_complete=min_complete
                )
                if audited is None:
                    return "exact"
    return "semantic" if seen else "exact"


def build_panel_cells(
    experiment_dir: Path,
    models: Sequence[str],
    metric: str,
    *,
    rounds: Sequence[str] = ROUND_TREES,
    min_complete: int = MIN_COMPLETE,
) -> Dict[Tuple[str, str], List[Tuple[Cell, Optional[float]]]]:
    """Figure cells keyed by (model, panel title), paired with their deltas.

    Each panel's delta is against the panel's own first mode, so the baseline
    bar reads exactly +0.0. A panel whose baseline never ran yields no bars at
    all rather than bars measured against a different reference.
    """
    out: Dict[Tuple[str, str], List[Tuple[Cell, Optional[float]]]] = {}
    for model in models:
        for title, modes in AXIS_PANELS:
            loaded = [
                load_cell(
                    experiment_dir, model, label, axes, metric,
                    rounds=rounds, min_complete=min_complete,
                )
                for label, axes in modes
            ]
            baseline = loaded[0]
            if baseline is None:
                out[(model, title)] = []
                continue
            out[(model, title)] = [
                (cell, cell.percent - baseline.percent)
                for cell in loaded
                if cell is not None
            ]
    return out


def build_table_rows(
    experiment_dir: Path,
    models: Sequence[str],
    metric: str,
    *,
    rounds: Sequence[str] = ROUND_TREES,
    min_complete: int = MIN_COMPLETE,
) -> List[dict]:
    """One row per (model, condition), with delta against that model's reference."""
    out: List[dict] = []
    for model in models:
        cells = [
            cell
            for cell in (
                load_cell(
                    experiment_dir, model, label, axes, metric,
                    rounds=rounds, min_complete=min_complete,
                )
                for label, axes in TABLE_CONDITIONS
            )
            if cell is not None
        ]
        if not cells or cells[0].label != TABLE_CONDITIONS[0][0]:
            # Without the reference cell there is nothing to take a delta
            # against, so the model is skipped rather than printed against a
            # substitute baseline.
            continue
        reference = cells[0]
        for cell in cells:
            out.append(
                {
                    "model": model,
                    "display_model": model_display_name(model),
                    "condition": cell.label,
                    "mean": cell.percent,
                    "delta": None if cell is reference else cell.percent - reference.percent,
                    "rounds": cell.rounds,
                    "per_round": cell.per_round,
                    "sd": cell.sd,
                    "n_tasks": cell.n_tasks,
                    "cycles_per_task": cell.cycles_per_task,
                    "cost_per_task": cell.cost_per_task,
                }
            )
    return out


def two_sigma_threshold(rows: Iterable[Mapping[str, object]]) -> Optional[float]:
    """Two mean per-cell standard deviations, over cells that have a spread.

    An effect must exceed this to be distinguishable from run-to-run variance.
    A cell run once has no spread of its own and contributes nothing.
    """
    sds = [float(row["sd"]) for row in rows if row.get("sd") is not None]
    return 2.0 * statistics.mean(sds) if sds else None


def _format_delta(row: Mapping[str, object], threshold: Optional[float]) -> str:
    delta = row.get("delta")
    if delta is None:
        return "---"
    text = f"${float(delta):+.1f}$"
    # A cell run once cannot clear a threshold that is defined by spread.
    if threshold is not None and row.get("sd") is not None and abs(float(delta)) > threshold:
        return f"\\textbf{{{text}}}"
    return text


def render_ablation_table(
    rows: Sequence[Mapping[str, object]],
    *,
    metric: str,
    n_rounds: int = len(ROUND_TREES),
    n_tasks: int = 20,
) -> str:
    """The LaTeX `tabular`, grouped by model with a rule between groups."""
    metric_name = "semantic match" if metric == "semantic" else "exact match"
    threshold = two_sigma_threshold(rows)
    by_model: Dict[str, List[Mapping[str, object]]] = {}
    for row in rows:
        by_model.setdefault(str(row["model"]), []).append(row)

    partial = sorted({int(row["rounds"]) for row in rows if int(row["rounds"]) < n_rounds})

    lines: List[str] = [
        "% ============================================================",
        "% GENERATED by sana_analysis/paper/tier_ablation.py -- do not edit by hand.",
        f"% metric: {METRIC_COLUMNS[metric]}; up to {n_rounds} replicate rounds; "
        f"{n_tasks} tasks per cell.",
        "% Requires: booktabs, multirow.",
        "% ============================================================",
        "\\begin{table}[h]",
        "  \\centering",
    ]
    caption = [
        "  \\caption{Leave-one-out ablation across model tiers on LakeQA",
        "  \\textsc{subset20b} (" + str(n_tasks) + " tasks per cell, " + metric_name + ",",
        "  mean over the replicate rounds each row has, run under identical",
        "  configuration). $\\bar{x}$ is that mean and $\\delta$ the effect relative to",
        "  that model's reference cell.",
    ]
    if partial:
        caption.append(
            "  A superscript on $\\bar{x}$ gives the number of replicate rounds"
        )
        caption.append(
            f"  averaged when it is fewer than {n_rounds}; rows without one are over"
        )
        caption.append(f"  all {n_rounds}.")
    if threshold is not None:
        caption.append(
            f"  Effects exceeding the $\\pm{threshold:.0f}$\\,pp two-sigma replicate"
        )
        caption.append(
            "  threshold are set in bold; one task is worth "
            + f"{PP_PER_TASK:.0f}\\,pp at $n={n_tasks}$."
        )
    caption.append(
        "  \\emph{Rounds/task} is the mean number of agent cycles a task took."
    )
    caption.append(
        "  Cost is the mean spend per task over completed rows, including the hidden"
    )
    caption.append(
        "  helper agents the ideal modes delegate to -- these outweigh the visible"
    )
    caption.append(
        "  agent, so a cost counted on the main agent alone would rank the oracle"
    )
    caption.append("  tools as the cheap ones.}")
    lines.extend(caption)

    lines.extend([
        "  \\label{tab:tier-ablation}",
        "  \\scriptsize",
        "  \\setlength{\\tabcolsep}{4pt}",
        "  \\renewcommand{\\arraystretch}{0.95}",
        "  \\resizebox{\\columnwidth}{!}{%",
        "  \\begin{tabular}{llrrrr}",
        "    \\toprule",
        "    Model & Condition & $\\bar{x}$ (\\%) & $\\delta$ (pp) & "
        "Rounds/task & \\$/task \\\\",
        "    \\midrule",
    ])
    for index, (model, model_rows) in enumerate(by_model.items()):
        if index:
            lines.append("    \\midrule")
        display = model_display_name(model)
        lines.append(
            f"    \\multirow{{{len(model_rows)}}}{{*}}{{\\texttt{{{display}}}}}"
        )
        for row in model_rows:
            rounds = int(row["rounds"])
            mean = f"{float(row['mean']):.1f}"
            if rounds < n_rounds:
                mean += f"$^{{{rounds}}}$"
            cycles = row.get("cycles_per_task")
            cost = row.get("cost_per_task")
            cells = [
                str(row["condition"]),
                mean,
                _format_delta(row, threshold),
                "---" if cycles is None else f"{float(cycles):.1f}",
                "---" if cost is None else f"{float(cost):.4f}",
            ]
            # The per-round values are carried as a LaTeX comment so a printed
            # mean can always be traced back to the rounds it averaged.
            lines.append(
                f"      % rounds={rounds} n={int(row['n_tasks'])} "
                f"per_round={[round(v, 1) for v in row['per_round']]}"
            )
            lines.append("      & " + " & ".join(cells) + " \\\\")
    lines.extend([
        "    \\bottomrule",
        "  \\end{tabular}}",
        "\\end{table}",
        "",
    ])
    return "\n".join(lines)


def render_ablation_table_text(
    rows: Sequence[Mapping[str, object]],
    *,
    n_rounds: int = len(ROUND_TREES),
) -> str:
    """A plain-text rendering of the same rows, for review without a TeX run."""
    header = f"{'Model':<14}{'Condition':<24}{'x̄ (%)':>9}{'δ (pp)':>9}{'Rds/task':>10}{'$/task':>9}{'n':>4}"
    out = [header, "-" * len(header)]
    last = None
    for row in rows:
        model = model_display_name(str(row["model"]))
        if last is not None and model != last:
            out.append("-" * len(header))
        cycles = row.get("cycles_per_task")
        cost = row.get("cost_per_task")
        rounds = int(row["rounds"])
        out.append(
            f"{model if model != last else '':<14}"
            f"{str(row['condition']):<24}"
            f"{float(row['mean']):>9.1f}"
            f"{('—' if row['delta'] is None else format(float(row['delta']), '+.1f')):>9}"
            f"{('—' if cycles is None else format(float(cycles), '.1f')):>10}"
            f"{('—' if cost is None else format(float(cost), '.4f')):>9}"
            f"{rounds:>4}"
        )
        last = model
    return "\n".join(out)


def _import_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_axis_delta_figure(
    panels: Mapping[Tuple[str, str], Sequence[Tuple[Cell, Optional[float]]]],
    models: Sequence[str],
    output_dir: Path,
    *,
    stem: str,
    title: str = FIGURE_TITLE,
) -> List[Path]:
    """Draw the model x axis grid to `<stem>.pdf` and `<stem>.png`."""
    plt = _import_pyplot()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with plt.rc_context(FIGURE_RC_PARAMS):
        fig, axes = plt.subplots(
            len(models), len(AXIS_PANELS),
            figsize=(13.5, 2.3 * len(models)), squeeze=False,
        )
        for row_index, model in enumerate(models):
            for col_index, (panel_title, _modes) in enumerate(AXIS_PANELS):
                ax = axes[row_index][col_index]
                entries = list(panels.get((model, panel_title), []))
                plot_horizontal_delta_bars(
                    ax,
                    [cell.label for cell, _delta in entries],
                    [cell.percent for cell, _delta in entries],
                    [delta for _cell, delta in entries],
                    panel_title if row_index == 0 else "",
                    label_fontsize=11,
                    value_fontsize=10,
                    title_fontsize=13,
                    x_limit=FIGURE_X_LIMIT,
                    inside_label_reserved=FIGURE_INSIDE_LABEL_RESERVED,
                    slots=PANEL_SLOTS,
                )
                ax.spines["left"].set_color(FIGURE_RULE_COLOR)
                ax.spines["left"].set_linewidth(1.1)
                ax.tick_params(axis="y", length=0)
            axes[row_index][0].set_ylabel(
                model_display_name(model),
                fontsize=12, fontweight="bold", labelpad=10,
            )
        if title:
            fig.suptitle(title, fontsize=16)
        fig.tight_layout(rect=(0, 0, 1, 0.985 if not title else 0.96))

        written: List[Path] = []
        for extension, dpi in (("pdf", None), ("png", 200)):
            path = output_dir / f"{stem}.{extension}"
            kwargs = {"bbox_inches": "tight", "pad_inches": 0.04}
            if dpi:
                kwargs["dpi"] = dpi
            fig.savefig(path, **kwargs)
            written.append(path)
        plt.close(fig)
    return written


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--experiment-dir",
        default="experiments/2026-08-31-model-tiers-subset20b",
        help="Sweep directory holding the replicate round trees.",
    )
    parser.add_argument(
        "--rounds", nargs="+", default=list(ROUND_TREES),
        help="Round tree names under --experiment-dir.",
    )
    parser.add_argument("--output-dir", default="paper_figures")
    parser.add_argument("--figure-stem", default="tier_ablation_axis_delta")
    parser.add_argument("--table-name", default="tier_ablation_table.tex")
    parser.add_argument(
        "--metric", choices=("auto", "exact", "semantic"), default="auto",
        help="auto uses semantic match only when every readable cell is audited.",
    )
    parser.add_argument(
        "--min-rounds", type=int, default=1,
        help="Drop a model whose reference cell has fewer rounds than this.",
    )
    parser.add_argument("--min-complete", type=int, default=MIN_COMPLETE)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--print-table", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    experiment_dir = Path(args.experiment_dir)
    models = args.models or discover_models(experiment_dir, rounds=args.rounds)
    if not models:
        print(f"no round trees with models under {experiment_dir}")
        return 1

    metric = (
        resolve_metric(
            experiment_dir, models, rounds=args.rounds, min_complete=args.min_complete
        )
        if args.metric == "auto"
        else args.metric
    )

    table_rows = build_table_rows(
        experiment_dir, models, metric,
        rounds=args.rounds, min_complete=args.min_complete,
    )
    kept = [
        model
        for model in models
        if any(
            row["model"] == model
            and row["condition"] == TABLE_CONDITIONS[0][0]
            and int(row["rounds"]) >= args.min_rounds
            for row in table_rows
        )
    ]
    dropped = [model for model in models if model not in kept]
    table_rows = [row for row in table_rows if row["model"] in kept]
    if not table_rows:
        print(f"no readable reference cells under {experiment_dir}")
        return 1

    panels = build_panel_cells(
        experiment_dir, kept, metric,
        rounds=args.rounds, min_complete=args.min_complete,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / args.table_name
    table_path.write_text(
        render_ablation_table(table_rows, metric=metric, n_rounds=len(args.rounds)),
        encoding="utf-8",
    )
    figures = plot_axis_delta_figure(
        panels, kept, output_dir, stem=args.figure_stem
    )

    print(f"metric={METRIC_COLUMNS[metric]}  models={len(kept)}  rounds<={len(args.rounds)}")
    if dropped:
        print(f"dropped (no reference cell at --min-rounds {args.min_rounds}): {', '.join(dropped)}")
    for model in kept:
        counts = {
            str(row["condition"]): int(row["rounds"])
            for row in table_rows
            if row["model"] == model
        }
        missing = [label for label, _axes in TABLE_CONDITIONS if label not in counts]
        short = {label: n for label, n in counts.items() if n < len(args.rounds)}
        note = ""
        if short:
            note += "  partial: " + ", ".join(f"{label} (n={n})" for label, n in short.items())
        if missing:
            note += "  missing: " + ", ".join(missing)
        print(f"  {model_display_name(model):<14} {len(counts)}/{len(TABLE_CONDITIONS)} cells{note}")
    print(f"wrote {table_path}")
    for path in figures:
        print(f"wrote {path}")
    if args.print_table:
        print()
        print(render_ablation_table_text(table_rows, n_rounds=len(args.rounds)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
