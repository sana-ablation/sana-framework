#!/usr/bin/env python3
"""Retrieval-source panel grid, in the paper's fig21b style.

The tier figure's Search panel varies how the lake is searched. This one varies
what is searched at all -- open web against three lake retrievers -- and pairs
the accuracy with the two recall measures that explain it, so the panels read
left to right as outcome, then cause:

    Semantic match   how often the answer was right
    D_ret            how often the gold dataset appeared in search results
    D_acc            how often the agent actually read it

Arms are ordered by how much of the lake they see, so the baseline row is the
open web and each delta is what moving into the lake buys. The web arm touches
no lake dataset by construction, so its recall is undefined rather than zero and
is printed as n/a instead of drawn as a bar -- a zero-length bar there would read
as "searched the lake and found nothing".

Palette and geometry follow sana_framework_paper/scripts/make_axis_delta_figure.py.

    python make_search_axis_figure.py [--exact]
"""
from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

ROUNDS = ["results", "results-rep2", "results-rep3"]
MODEL = "openai_gpt-5-mini"
STEM = "fig-search-axis"
TASKS = HERE / "inputs" / "benchmarks" / "lakeqa" / "tasks-mini" / "tasks"

# Ordered by how much of the lake the arm can see; the baseline is the first.
ARMS = [
    ("Web",       "search_web__results_naive__profile_standard__compute_standard__nos3__skills_off"),
    ("BM25",      "search_naive__results_naive__profile_standard__compute_standard__skills_off"),
    ("PNEUMA",    "search_standard__results_naive__profile_standard__compute_standard__skills_off"),
    ("Ideal",     "search_ideal__results_naive__profile_standard__compute_standard__skills_off"),
]

GAIN, LOSS, NEUTRAL = "#9ECDAF", "#EFB3B3", "#D3D3D3"
INK, MUTED, RULE, SURFACE = "#111827", "#6B7280", "#333333", "#FFFFFF"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "pdf.fonttype": 42,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


def _num(v, d=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


def rows_for(tree, variant, semantic):
    p = HERE / (f"{tree}_semantic" if semantic else tree) / "modes" / MODEL / variant / "eval_results.csv"
    if not p.is_file():
        return None
    with open(p, newline="") as f:
        rows = [r for r in csv.DictReader(f) if not (r.get("error") or "").strip()]
    return rows if len(rows) >= 15 else None


def all_audited() -> bool:
    for tree in ROUNDS:
        sem, raw = HERE / f"{tree}_semantic" / "modes", HERE / tree / "modes"
        if not sem.is_dir() or not raw.is_dir():
            return False
        if len(list(sem.rglob("eval_results.csv"))) < len(list(raw.rglob("eval_results.csv"))):
            return False
    return True


def accuracy(variant, key, semantic):
    vals = []
    for tree in ROUNDS:
        rows = rows_for(tree, variant, semantic)
        if rows:
            vals.append(100 * sum(1 for r in rows if _num(r.get(key)) >= 1) / len(rows))
    return statistics.mean(vals) if vals else None


def discovery(variant):
    """D_ret / D_acc from the repo's own module, over round-1 traces."""
    try:
        from sana_analysis.running_analysis.discovery_metrics import (
            compute_discovery_metrics, load_traces,
        )
    except Exception:
        return None, None
    d = HERE / "results" / "traces" / "modes" / MODEL / variant
    if not d.is_dir():
        return None, None
    gold = {}
    for tp in TASKS.rglob("*.json"):
        used = list(json.loads(tp.read_text()).get("datasets_used") or [])
        gold[str(tp)] = used
        gold[f"{tp.parent.name}/{tp.stem}"] = used
        gold[tp.stem] = used
    agg = compute_discovery_metrics(load_traces(str(d)), gold).get("aggregate", {})
    if not agg:
        return None, None
    return 100 * agg.get("D_ret", 0), 100 * agg.get("D_acc", 0)


def panel(ax, title, values, *, xmax=105.0):
    """values: [(label, value or None)] top to bottom, baseline first."""
    base = next((v for _, v in values if v is not None), 0.0)
    for row, (label, v) in enumerate(values):
        y = len(values) - 1 - row
        if v is None:
            ax.text(1.5, y, "n/a", ha="left", va="center", fontsize=10,
                    color=MUTED, style="italic", zorder=4)
            continue
        d = v - base
        color = NEUTRAL if abs(d) < 0.05 else (GAIN if d > 0 else LOSS)
        ax.barh(y, v, height=0.62, color=color, zorder=3)
        inside = v >= 0.42 * xmax
        ax.text(v - 1.5 if inside else v + 1.5, y, f"{v:.1f}% ({d:+.1f}%)",
                ha="right" if inside else "left", va="center",
                fontsize=10, color=INK, zorder=4)

    ax.set_yticks(range(len(values)))
    ax.set_yticklabels([label for label, _ in reversed(values)], fontsize=11, color=INK)
    ax.set_ylim(-0.6, len(values) - 0.4)
    ax.set_xlim(0, xmax)
    ax.set_xticks([])
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(RULE)
    ax.spines["left"].set_linewidth(1.1)
    ax.tick_params(axis="y", length=0)
    ax.set_title(title, fontsize=13, color=INK, pad=10)


def main() -> int:
    semantic = "--exact" not in sys.argv and all_audited()
    key = "semantic_match" if semantic else "exact_match"
    title = "Semantic match" if semantic else "Exact match"

    acc = [(label, accuracy(v, key, semantic)) for label, v in ARMS]
    if all(v is None for _, v in acc):
        print("no results found", file=sys.stderr)
        return 1
    disc = {label: discovery(v) for label, v in ARMS}
    # Undefined, not zero: the web arm never touches a lake dataset.
    disc["Web"] = (None, None)

    panels = [
        (title, acc),
        (r"$D_{ret}$  (gold surfaced by search)", [(l, disc[l][0]) for l, _ in ARMS]),
        (r"$D_{acc}$  (gold actually read)",      [(l, disc[l][1]) for l, _ in ARMS]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 2.6), squeeze=False)
    for ax, (t, vals) in zip(axes[0], panels):
        panel(ax, t, vals)
    axes[0][0].set_ylabel("gpt-5-mini", fontsize=12, fontweight="bold", color=INK, labelpad=10)
    fig.tight_layout(rect=(0, 0, 1, 0.985))

    for ext, dpi in (("pdf", None), ("png", 200)):
        kw = {"bbox_inches": "tight", "pad_inches": 0.04}
        if dpi:
            kw["dpi"] = dpi
        fig.savefig(HERE / f"{STEM}.{ext}", **kw)
    plt.close(fig)
    print(f"wrote {STEM}.pdf and {STEM}.png  (metric={key})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
