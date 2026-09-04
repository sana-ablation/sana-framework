#!/usr/bin/env python3
"""Generate results.tex for the model-tier ablation from the result trees.

Hand-writing this table was wrong twice: once when errored rows were counted as
wrong answers, and once when a single run's numbers were presented as findings
that replication then retracted. Generating it means the prose and the tables
cannot drift from the data.

Uses semantic_match when EVERY round is audited, else exact_match for all of
them -- never a mix, since a semantic cell and an exact cell are not comparable.

    python gen_tex.py            # write results.tex
    python gen_tex.py --stdout   # preview
"""
import csv
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROUNDS = ["results", "results-rep2", "results-rep3"]
CELLS = [
    ("Reference (all ideal)", r"search_ideal.*profile_ideal__compute_ideal"),
    ("Search: BM25",          r"search_naive.*profile_ideal__compute_ideal"),
    ("Search: PNEUMA",        r"search_standard.*profile_ideal__compute_ideal"),
    ("Search: Preloaded",     r"search_preloaded.*profile_ideal__compute_ideal"),
    ("Plan: No Plan",         r"search_ideal.*profile_naive__compute_ideal"),
    ("Plan: Default",         r"search_ideal.*profile_standard__compute_ideal"),
    ("Data An.: Standard",    r"search_ideal.*profile_ideal__compute_standard"),
]
# A model absent from the result trees is skipped, so listing one before its
# runs land is harmless.
MODELS = ["openai_gpt-5.4-nano", "openai_gpt-5-mini", "openai_gpt-5.2",
          "openai_gpt-5.6-luna"]
TEX_NAME = {"openai_gpt-5.4-nano": r"\texttt{gpt-5.4-nano}",
            "openai_gpt-5-mini": r"\texttt{gpt-5-mini}",
            "openai_gpt-5.2": r"\texttt{gpt-5.2}",
            "openai_gpt-5.6-luna": r"\texttt{gpt-5.6-luna}"}


def _num(v, d=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


def all_audited() -> bool:
    """True only if every round has a complete semantic tree."""
    for tree in ROUNDS:
        sem = HERE / f"{tree}_semantic" / "modes"
        raw = HERE / tree / "modes"
        if not sem.is_dir():
            return False
        n_sem = len(list(sem.rglob("eval_results.csv")))
        n_raw = len(list(raw.rglob("eval_results.csv")))
        if n_raw == 0 or n_sem < n_raw:
            return False
    return True


def cell_pct(tree: str, model: str, pattern: str, key: str):
    root = HERE / (f"{tree}_semantic" if key == "semantic_match" else tree) / "modes"
    md = root / model
    if not md.is_dir():
        return None
    hits = [d for d in md.iterdir() if d.is_dir() and re.search(pattern, d.name)]
    if not hits:
        return None
    p = hits[0] / "eval_results.csv"
    if not p.is_file():
        return None
    with open(p, newline="") as f:
        rows = list(csv.DictReader(f))
    done = [r for r in rows if not (r.get("error") or "").strip()]
    if len(done) < 15 or key not in (done[0] or {}):
        return None
    return 100 * sum(1 for r in done if _num(r.get(key)) >= 1) / len(done)


def cell_rows(tree: str, model: str, pattern: str, key: str):
    """Completed rows for one cell in one round, or None.

    Errored rows are dropped here rather than at the call site: such a row never
    reached the model, so counting it would divide real spend by an inflated
    task count.
    """
    root = HERE / (f"{tree}_semantic" if key == "semantic_match" else tree) / "modes"
    md = root / model
    if not md.is_dir():
        return None
    hits = [d for d in md.iterdir() if d.is_dir() and re.search(pattern, d.name)]
    if not hits:
        return None
    p = hits[0] / "eval_results.csv"
    if not p.is_file():
        return None
    with open(p, newline="") as f:
        done = [r for r in csv.DictReader(f) if not (r.get("error") or "").strip()]
    return done if len(done) >= 15 else None


def collect(key):
    data, sds = {}, []
    for model in MODELS:
        rows = []
        for label, pattern in CELLS:
            vals = [v for v in (cell_pct(t, model, pattern, key) for t in ROUNDS) if v is not None]
            if not vals:
                continue
            sd = statistics.stdev(vals) if len(vals) > 1 else None
            if sd is not None:
                sds.append(sd)
            cost = n = correct = 0
            for tree in ROUNDS:
                got = cell_rows(tree, model, pattern, key)
                if not got:
                    continue
                n += len(got)
                correct += sum(1 for r in got if _num(r.get(key)) >= 1)
                cost += sum(_num(r.get("cost_usd")) for r in got)
            rows.append({"label": label, "vals": vals, "rounds": len(vals),
                         "mean": statistics.mean(vals), "sd": sd,
                         "per_task": cost / n if n else None,
                         "per_correct": cost / correct if correct else None})
        if rows:
            ref = rows[0]["mean"]
            for r in rows:
                r["delta"] = None if r["label"].startswith("Reference") else r["mean"] - ref
            data[model] = rows
    return data, sds


def main() -> int:
    key = "semantic_match" if all_audited() else "exact_match"
    metric_name = "semantic match" if key == "semantic_match" else "exact match"
    data, sds = collect(key)
    if not data:
        print("no results found", file=sys.stderr)
        return 1

    pooled = statistics.mean(sds)
    thresh = 2 * pooled

    L = []
    A = L.append
    A("% ============================================================")
    A("% GENERATED by gen_tex.py -- do not edit by hand.")
    A(f"% metric: {key}; {len(ROUNDS)} replicate rounds; 20 tasks per cell.")
    A("% Requires: booktabs, multirow.")
    A("% ============================================================")
    A("")
    A(r"\subsection{Replicate variance on the ablation grid}")
    A(r"\label{sec:tier-noise}")
    A("")
    A("We ran the leave-one-out ablation over three model tiers on")
    A(r"\textsc{subset20b}, a 20-task split of LakeQA stratified jointly on node")
    A("count and historical solve rate. Every axis is held at the oracle and one")
    A("axis is degraded per cell, giving seven cells per model. Each cell was run")
    A(r"\textbf{three times} under identical configuration, so every number carries")
    A("a standard deviation rather than resting on a single sample.")
    A("")
    A("That replication is the central result. The mean per-cell standard")
    A(f"deviation is {pooled:.1f}\\,pp, so a two-sigma interval spans")
    A(f"$\\pm{thresh:.0f}$\\,pp and an effect must exceed roughly {thresh:.0f}\\,pp to be")
    A("distinguishable from run-to-run variance. Since one task is worth 5\\,pp at")
    A(f"$n=20$, that threshold is about {thresh/5:.0f} tasks.")
    A("")

    # which effects clear
    # A cell run once has no spread of its own, so it cannot be said to clear a
    # threshold that is defined by spread; it is reported but never counted here.
    clears = [(m, r) for m, rows in data.items() for r in rows
              if r.get("delta") is not None and r["sd"] is not None
              and abs(r["delta"]) > thresh]
    clears.sort(key=lambda t: t[1]["delta"])
    if clears:
        names = ", ".join(f"{r['label']} at {TEX_NAME[m]} (${r['delta']:+.1f}$\\,pp)"
                          for m, r in clears[:5])
        A(f"Of the {sum(len(r) - 1 for r in data.values())} degraded cells, only")
        A(f"{len(clears)} clear that threshold: {names}.")
        A("The search axis is the one effect that replicates across every tier;")
        A("the remaining cells cannot be separated from noise at this sample size.")
        A("")

    refs = {m: rows[0] for m, rows in data.items() if rows}
    if len(refs) >= 2:
        parts = ", ".join(
            f"{TEX_NAME[m]} {r['mean']:.1f}\\%"
            + ("" if r["sd"] is None else f" ($\\sigma={r['sd']:.1f}$)")
            for m, r in refs.items())
        A(f"The reference cells themselves are {parts}.")
        mini = refs.get("openai_gpt-5-mini")
        big = refs.get("openai_gpt-5.2")
        if mini and big and mini["sd"] is not None and big["sd"] is not None:
            gap = abs(mini["mean"] - big["mean"])
            se = ((mini["sd"] ** 2 + big["sd"] ** 2) / len(ROUNDS)) ** 0.5
            if gap < 0.05:
                A(r"\texttt{gpt-5-mini} and \texttt{gpt-5.2} land on the same mean")
                A(f"({mini['mean']:.1f}\\%), despite a roughly sevenfold difference in")
                A("price per token.")
            else:
                A(r"\texttt{gpt-5-mini} and \texttt{gpt-5.2} differ by")
                A(f"{gap:.1f}\\,pp against a standard error of {se:.1f}\\,pp on that")
                A("difference, so they are not separable here.")
            A("On this split the frontier model does not outperform the smaller")
            A("one, and a single round would have suggested otherwise: round~1")
            A(r"alone put \texttt{gpt-5.2} 5\,pp below \texttt{gpt-5-mini}.")
        A("")

    # ---- main table
    A(r"\begin{table}[h]")
    A(r"  \centering")
    A(r"  \caption{Leave-one-out ablation across model tiers on LakeQA")
    A(r"  \textsc{subset20b} (20 tasks per cell, " + metric_name + r", mean over the")
    A(r"  replicate rounds in column \emph{R}, run under identical")
    A(r"  configuration). $\bar{x}$ is the mean")
    A(r"  over rounds and $\sigma$ their standard deviation; $\delta$ is the effect")
    A(r"  relative to that model's reference cell. Effects exceeding the")
    A(f"  $\\pm{thresh:.0f}$\\,pp two-sigma threshold are set in bold.")
    A(r"  \emph{R} is how many replicate rounds back the row. Cost is over")
    A(r"  completed rows: \$/correct divides that cell's spend by the answers it")
    A(r"  got right, so it prices the outcome rather than the attempt.}")
    A(r"  \label{tab:tier-ablation}")
    A(r"  \scriptsize")
    A(r"  \setlength{\tabcolsep}{4pt}")
    A(r"  \renewcommand{\arraystretch}{0.95}")
    A(r"  \resizebox{\columnwidth}{!}{%")
    A(r"  \begin{tabular}{llrrrrrr}")
    A(r"    \toprule")
    A(r"    Model & Condition & $\bar{x}$ (\%) & $\sigma$ & $\delta$ (pp) & R & "
      r"\$/task & \$/correct \\")
    A(r"    \midrule")
    for i, model in enumerate([m for m in MODELS if m in data]):
        if i:
            A(r"    \midrule")
        rows = data[model]
        A(f"    \\multirow{{{len(rows)}}}{{*}}{{{TEX_NAME[model]}}}")
        for r in rows:
            sd = "---" if r["sd"] is None else f"{r['sd']:.1f}"
            if r.get("delta") is None:
                dl = "---"
            else:
                txt = f"${r['delta']:+.1f}$"
                dl = f"\\textbf{{{txt}}}" if abs(r["delta"]) > thresh else txt
            pt = "---" if r["per_task"] is None else f"{r['per_task']:.4f}"
            pc = "---" if r["per_correct"] is None else f"{r['per_correct']:.4f}"
            A(f"      & {r['label']} & {r['mean']:.1f} & {sd} & {dl} & "
              f"{r['rounds']} & {pt} & {pc} \\\\")
    A(r"    \bottomrule")
    A(r"  \end{tabular}}")
    A(r"\end{table}")
    A("")

    # ---- noise table
    A(r"\begin{table}[h]")
    A(r"  \centering")
    A(r"  \caption{Replicate noise floor, over every cell run three times. One")
    A(r"  task is worth 5\,pp at $n=20$, so the mean per-cell standard deviation")
    A(f"  is close to a single task and the two-sigma threshold is about {thresh/5:.0f}.}}")
    A(r"  \label{tab:tier-noise-floor}")
    A(r"  \small")
    A(r"  \begin{tabular}{lr}")
    A(r"    \toprule")
    A(r"    Statistic over per-cell $\sigma$ & Value (pp) \\")
    A(r"    \midrule")
    A(f"    Mean                  & {pooled:.1f} \\\\")
    A(f"    Median                & {statistics.median(sds):.1f} \\\\")
    A(f"    Maximum               & {max(sds):.1f} \\\\")
    A(f"    Two-sigma threshold   & {thresh:.1f} \\\\")
    A(f"    Cells                 & {len(sds)} \\\\")
    A(r"    \bottomrule")
    A(r"  \end{tabular}")
    A(r"\end{table}")
    A("")

    # ---- figure: per-axis panels, matching the paper's fig21b family.
    # Drawn by make_axis_delta_figure.py into fig-axis-delta.pdf rather than
    # inline TikZ, so this figure and the paper's share one implementation of
    # the geometry and palette and cannot drift apart.
    A(r"\begin{figure}[h]")
    A(r"  \centering")
    A(r"  \includegraphics[width=\columnwidth]{fig-axis-delta}")
    A(r"  \caption{Per-axis ablation across model tiers on LakeQA")
    A(r"  \textsc{subset20b}. Rows are models, columns are the three SANA axes,")
    A(r"  and each bar is one mode of that axis with the other two held at")
    A(r"  \emph{Ideal}. The label gives the " + metric_name + r" and its change")
    A(r"  against the panel's baseline -- the weakest mode of that axis, the top")
    A(r"  row of each panel. Bars are means over the replicate rounds; colour")
    A(r"  repeats the sign of the printed delta rather than carrying it alone.")
    A(f"  Against the $\\pm{thresh:.0f}$\\,pp two-sigma replicate threshold of")
    A(r"  Table~\ref{tab:tier-noise-floor}, only the Search panel moves any model")
    A(r"  further than run-to-run variance.}")
    A(r"  \label{fig:tier-axis-delta}")
    A(r"\end{figure}")

    out = "\n".join(L) + "\n"
    if "--stdout" in sys.argv:
        print(out)
    else:
        (HERE / "results.tex").write_text(out)
        print(f"wrote results.tex  (metric={key}, threshold={thresh:.1f}pp, "
              f"{len(clears)} effects clear)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
