#!/usr/bin/env python3
"""Per-axis ablation grid in the paper's fig21b style, from the subset20b trees.

Rows are models, columns are the three SANA axes. Each bar is one mode of that
axis with the other two held at Ideal, and its label carries the absolute match
plus the delta against the panel's baseline -- the weakest mode of that axis,
which is the first row of the panel.

The palette and geometry are lifted from
sana_framework_paper/scripts/make_axis_delta_figure.py so the two figures sit
side by side without a visible seam. Two things differ, both because this split
has replicates the paper figure did not: every bar is a mean over the rounds
that exist, and a model is dropped rather than plotted on a different metric
from its neighbours (see METRIC below).

METRIC: semantic match when every round of every plotted model is audited,
otherwise exact match for all of them. Mixing the two within one figure would
put incomparable numbers on a shared axis.

    python make_axis_delta_figure.py [--exact]
"""
from __future__ import annotations

import csv
import re
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROUNDS = ["results", "results-rep2", "results-rep3"]
STEM = "fig-axis-delta"

MODELS = ["openai_gpt-5.4-nano", "openai_gpt-5-mini", "openai_gpt-5.2", "openai_gpt-5.6-luna"]
PRETTY = {"openai_gpt-5.4-nano": "gpt-5.4-nano", "openai_gpt-5-mini": "gpt-5-mini",
          "openai_gpt-5.2": "gpt-5.2", "openai_gpt-5.6-luna": "gpt-5.6-luna"}

# fig21b's status palette: colour reports the sign of the delta, and the signed
# number is printed on every bar too, so direction is never colour-alone.
GAIN, LOSS, NEUTRAL = "#9ECDAF", "#EFB3B3", "#D3D3D3"
INK, RULE, SURFACE = "#111827", "#333333", "#FFFFFF"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "pdf.fonttype": 42,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})

# (panel title, [(printed label, variant regex)]) with the baseline first.
# Preloaded is kept, as in fig21b; fig21c drops it because handing over the gold
# sources is not a retrieval condition.
PANELS = [
    ("Plan", [
        ("No Plan",   r"search_ideal.*profile_naive__compute_ideal"),
        ("Default",   r"search_ideal.*profile_standard__compute_ideal"),
        ("Ideal",     r"search_ideal.*profile_ideal__compute_ideal"),
    ]),
    ("Search", [
        ("BM25",      r"search_naive.*profile_ideal__compute_ideal"),
        ("PNEUMA",    r"search_standard.*profile_ideal__compute_ideal"),
        ("Ideal",     r"search_ideal.*profile_ideal__compute_ideal"),
        ("Preloaded", r"search_preloaded.*profile_ideal__compute_ideal"),
    ]),
    ("Data Analysis", [
        ("Standard",  r"search_ideal.*profile_ideal__compute_standard"),
        ("Ideal",     r"search_ideal.*profile_ideal__compute_ideal"),
    ]),
]
SLOTS = max(len(m) for _, m in PANELS)


def _num(v, d=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


def _cell_rows(tree: str, model: str, pattern: str, semantic: bool):
    root = HERE / (f"{tree}_semantic" if semantic else tree) / "modes" / model
    if not root.is_dir():
        return None
    hits = [d for d in root.iterdir() if d.is_dir() and re.search(pattern, d.name)]
    if not hits:
        return None
    path = hits[0] / "eval_results.csv"
    if not path.is_file():
        return None
    with open(path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if not (r.get("error") or "").strip()]
    return rows if len(rows) >= 15 else None


def model_is_audited(model: str) -> bool:
    """Every round present for this model must also exist in the audited tree."""
    seen = 0
    for tree in ROUNDS:
        for _, modes in PANELS:
            for _, pattern in modes:
                raw = _cell_rows(tree, model, pattern, semantic=False)
                if raw is None:
                    continue
                seen += 1
                if _cell_rows(tree, model, pattern, semantic=True) is None:
                    return False
    return seen > 0


def score(model: str, pattern: str, key: str, semantic: bool):
    """Mean percentage across the rounds that exist, or None."""
    vals = []
    for tree in ROUNDS:
        rows = _cell_rows(tree, model, pattern, semantic)
        if rows:
            vals.append(100 * sum(1 for r in rows if _num(r.get(key)) >= 1) / len(rows))
    return statistics.mean(vals) if vals else None


def panel(ax, model, title, modes, key, semantic, *, show_title, xmax=105.0):
    base = score(model, modes[0][1], key, semantic)
    for row, (label, pattern) in enumerate(modes):
        v = score(model, pattern, key, semantic)
        if v is None:
            continue
        d = v - base
        color = NEUTRAL if abs(d) < 0.05 else (GAIN if d > 0 else LOSS)
        y = SLOTS - 1 - row
        ax.barh(y, v, height=0.62, color=color, zorder=3)
        inside = v >= 0.42 * xmax
        ax.text(v - 1.5 if inside else v + 1.5, y, f"{v:.1f}% ({d:+.1f}%)",
                ha="right" if inside else "left", va="center",
                fontsize=10, color=INK, zorder=4)

    ax.set_yticks([SLOTS - 1 - r for r in range(len(modes))])
    ax.set_yticklabels([label for label, _ in modes], fontsize=11, color=INK)
    ax.set_ylim(-0.6, SLOTS - 0.4)
    ax.set_xlim(0, xmax)
    ax.set_xticks([])
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(RULE)
    ax.spines["left"].set_linewidth(1.1)
    ax.tick_params(axis="y", length=0)
    if show_title:
        ax.set_title(title, fontsize=13, color=INK, pad=10)


def main() -> int:
    present = [m for m in MODELS if score(m, PANELS[0][1][0][1], "exact_match", False) is not None]
    if not present:
        print("no results found", file=sys.stderr)
        return 1

    semantic = "--exact" not in sys.argv and all(model_is_audited(m) for m in present)
    key = "semantic_match" if semantic else "exact_match"
    if not semantic:
        unaudited = [PRETTY[m] for m in present if not model_is_audited(m)]
        print(f"falling back to exact_match; not fully audited: {', '.join(unaudited) or 'n/a'}")

    rounds_seen = max(
        sum(1 for t in ROUNDS if _cell_rows(t, m, PANELS[0][1][0][1], semantic))
        for m in present)

    fig, axes = plt.subplots(len(present), len(PANELS),
                             figsize=(13.5, 2.3 * len(present)), squeeze=False)
    for row, model in enumerate(present):
        for col, (title, modes) in enumerate(PANELS):
            panel(axes[row][col], model, title, modes, key, semantic,
                  show_title=(row == 0))
        axes[row][0].set_ylabel(PRETTY[model], fontsize=12, fontweight="bold",
                                color=INK, labelpad=10)
    fig.tight_layout(rect=(0, 0, 1, 0.985))

    for ext, dpi in (("pdf", None), ("png", 200)):
        kw = {"bbox_inches": "tight", "pad_inches": 0.04}
        if dpi:
            kw["dpi"] = dpi
        fig.savefig(HERE / f"{STEM}.{ext}", **kw)
    plt.close(fig)
    print(f"wrote {STEM}.pdf and {STEM}.png  "
          f"(metric={key}, models={len(present)}, rounds={rounds_seen})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
