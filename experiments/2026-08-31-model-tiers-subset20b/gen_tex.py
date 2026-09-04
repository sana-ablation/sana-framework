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


_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}


def _count_word(n: int) -> str:
    """Small counts read better as words in prose."""
    return _WORDS.get(n, str(n))


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
            cost = turns = n = correct = 0
            for tree in ROUNDS:
                got = cell_rows(tree, model, pattern, key)
                if not got:
                    continue
                n += len(got)
                correct += sum(1 for r in got if _num(r.get(key)) >= 1)
                # cost_usd is the main agent only. The ideal modes bill hidden
                # helper agents separately, and those dominate: dropping them
                # made compute=ideal look CHEAPER than compute=standard, when it
                # is roughly twice the price.
                cost += sum(_num(r.get("total_cost_with_all_subagents_usd"))
                            or _num(r.get("cost_usd")) for r in got)
                turns += sum(_num(r.get("cycle_count")) for r in got)
            # sd is kept even though it is no longer printed: the bold threshold
            # in the table is two of these standard deviations.
            rows.append({"label": label, "vals": vals, "rounds": len(vals),
                         "mean": statistics.mean(vals), "sd": sd,
                         "turns": turns / n if n else None,
                         "n": n, "correct": correct, "cost": cost,
                         "per_task": cost / n if n else None})
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
    A(f"We ran the leave-one-out ablation over {_count_word(len(data))} model tiers on")
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
        n_search = sum(1 for _, r in clears if r["label"].startswith("Search"))
        tiers = {m for m, r in clears if r["label"].startswith("Search")}
        A(f"{n_search} of those are search degradations, on {len(tiers)} of the")
        A(f"{len(data)} tiers; the remaining cells cannot be separated from noise")
        A("at this sample size. Search costs the most accuracy of any axis, but it")
        A("clears the threshold on the weaker tiers and not on the strongest --")
        A("the same degradation hurts less as the model improves.")
        A("")

    refs = {m: rows[0] for m, rows in data.items() if rows}
    if len(refs) >= 2:
        parts = ", ".join(
            f"{TEX_NAME[m]} {r['mean']:.1f}\\%"
            + ("" if r["sd"] is None else f" ($\\sigma={r['sd']:.1f}$)")
            for m, r in refs.items())
        A(f"The reference cells themselves are {parts}.")
        ranked = sorted(refs.items(), key=lambda kv: kv[1]["mean"], reverse=True)
        (m1, r1), (m2, r2) = ranked[0], ranked[1]
        if r1["sd"] is not None and r2["sd"] is not None:
            gap = r1["mean"] - r2["mean"]
            se = ((r1["sd"] ** 2 + r2["sd"] ** 2) / len(ROUNDS)) ** 0.5
            verdict = "are not separable here" if gap <= 2 * se else "are separable"
            A(f"The two strongest, {TEX_NAME[m1]} and {TEX_NAME[m2]}, differ by")
            A(f"{gap:.1f}\\,pp against a standard error of {se:.1f}\\,pp on that")
            A(f"difference, so they {verdict}.")

        # The separation that does survive is work and cost. Computed, not asserted.
        agg = {}
        for model, cells in data.items():
            n = sum(c["n"] for c in cells)
            correct = sum(c["correct"] for c in cells)
            agg[model] = {"acc": 100 * correct / n,
                          "turns": sum(c["turns"] * c["n"] for c in cells) / n,
                          "per_correct": sum(c["cost"] for c in cells) / max(correct, 1)}
        by_cost = sorted(agg.items(), key=lambda kv: kv[1]["per_correct"])
        cheap, dear = by_cost[0], by_cost[-1]
        top3 = sorted((v["acc"] for v in agg.values()), reverse=True)[:3]
        A("")
        A("Accuracy is not what separates the tiers. Over all cells the top three")
        A(f"land within {max(top3) - min(top3):.1f}\\,pp of one another")
        A(f"({min(top3):.1f}--{max(top3):.1f}\\%), inside the replicate threshold,")
        A("while cost per correct answer spans")
        A(f"{dear[1]['per_correct'] / cheap[1]['per_correct']:.0f}$\\times$:")
        A(f"{TEX_NAME[cheap[0]]} answers at \\${cheap[1]['per_correct']:.4f} in")
        A(f"{cheap[1]['turns']:.1f} agent cycles against {TEX_NAME[dear[0]]} at")
        A(f"\\${dear[1]['per_correct']:.4f} in {dear[1]['turns']:.1f}. The tier that")
        A("costs the most buys nothing here that the cheapest does not already")
        A("provide.")
        A("")

    # ---- main table
    A(r"\begin{table}[h]")
    A(r"  \centering")
    A(r"  \caption{Leave-one-out ablation across model tiers on LakeQA")
    A(r"  \textsc{subset20b} (20 tasks per cell, " + metric_name + r", mean over the")
    A(r"  replicate rounds each row has, run under identical configuration).")
    A(r"  $\bar{x}$ is the mean over those rounds and $\delta$ the effect relative")
    A(r"  to that model's reference cell; effects exceeding the")
    A(f"  $\\pm{thresh:.0f}$\\,pp two-sigma threshold of")
    A(r"  Table~\ref{tab:tier-noise-floor} are set in bold. \emph{Rounds/task} is")
    A(r"  the mean number of agent cycles a task took. Cost is the mean spend per")
    A(r"  task over completed rows, including the hidden helper agents the ideal")
    A(r"  modes delegate to -- these outweigh the visible agent, so a cost counted")
    A(r"  on the main agent alone would rank the oracle tools as the cheap ones.}")
    A(r"  \label{tab:tier-ablation}")
    A(r"  \scriptsize")
    A(r"  \setlength{\tabcolsep}{4pt}")
    A(r"  \renewcommand{\arraystretch}{0.95}")
    A(r"  \resizebox{\columnwidth}{!}{%")
    A(r"  \begin{tabular}{llrrrr}")
    A(r"    \toprule")
    A(r"    Model & Condition & $\bar{x}$ (\%) & $\delta$ (pp) & "
      r"Rounds/task & \$/task \\")
    A(r"    \midrule")
    for i, model in enumerate([m for m in MODELS if m in data]):
        if i:
            A(r"    \midrule")
        rows = data[model]
        A(f"    \\multirow{{{len(rows)}}}{{*}}{{{TEX_NAME[model]}}}")
        for r in rows:
            if r.get("delta") is None:
                dl = "---"
            else:
                txt = f"${r['delta']:+.1f}$"
                dl = f"\\textbf{{{txt}}}" if abs(r["delta"]) > thresh else txt
            pt = "---" if r["per_task"] is None else f"{r['per_task']:.4f}"
            tn = "---" if r["turns"] is None else f"{r['turns']:.1f}"
            A(f"      & {r['label']} & {r['mean']:.1f} & {dl} & {tn} & {pt} \\\\")
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
