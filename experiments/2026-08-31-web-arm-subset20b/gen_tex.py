#!/usr/bin/env python3
"""Generate results.tex for the web-arm retrieval comparison.

Three replicate rounds, semantic match when every round is audited (never a mix
of metrics), and D_ret / D_acc taken from the repo's own discovery_metrics
module rather than a reimplementation.

    python gen_tex.py [--stdout]
"""
import csv
import json
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

ROUNDS = ["results", "results-rep2", "results-rep3"]
MODEL = "openai_gpt-5-mini"
ARMS = [
    ("Ideal",    "search_ideal__results_naive__profile_standard__compute_standard__skills_off"),
    ("Standard", "search_standard__results_naive__profile_standard__compute_standard__skills_off"),
    ("Naive",    "search_naive__results_naive__profile_standard__compute_standard__skills_off"),
    ("Web",      "search_web__results_naive__profile_standard__compute_standard__nos3__skills_off"),
]
TASKS = HERE / "inputs" / "benchmarks" / "lakeqa" / "tasks-mini" / "tasks"


def _num(v, d=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


def all_audited() -> bool:
    for tree in ROUNDS:
        sem, raw = HERE / f"{tree}_semantic" / "modes", HERE / tree / "modes"
        if not sem.is_dir() or not raw.is_dir():
            return False
        if len(list(sem.rglob("eval_results.csv"))) < len(list(raw.rglob("eval_results.csv"))):
            return False
    return True


def rows_for(tree, variant, semantic):
    root = HERE / (f"{tree}_semantic" if semantic else tree) / "modes" / MODEL / variant
    p = root / "eval_results.csv"
    if not p.is_file():
        return None
    with open(p, newline="") as f:
        rows = list(csv.DictReader(f))
    done = [r for r in rows if not (r.get("error") or "").strip()]
    return done if len(done) >= 15 else None


def discovery(variant):
    """D_ret / D_acc from the repo's own module, over round 1 traces."""
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


def main() -> int:
    semantic = all_audited()
    key = "semantic_match" if semantic else "exact_match"
    metric = "semantic match" if semantic else "exact match"

    table, sds = [], []
    for label, variant in ARMS:
        vals, buckets, aux = [], {"c": 0, "i": 0, "b": 0}, []
        for tree in ROUNDS:
            rows = rows_for(tree, variant, semantic)
            if not rows:
                continue
            n = len(rows)
            vals.append(100 * sum(1 for r in rows if _num(r.get(key)) >= 1) / n)
            for r in rows:
                bk = r.get("semantic_bucket", "")
                if bk == "semantic_correct":
                    buckets["c"] += 1
                elif bk == "semantic_incorrect":
                    buckets["i"] += 1
                elif bk == "answer_unknown_blank":
                    buckets["b"] += 1
            aux.append({
                "cyc": sum(_num(r["cycle_count"]) for r in rows) / n,
                "tools": sum(_num(r["tool_calls_total"]) for r in rows) / n,
                "cap": sum(1 for r in rows if _num(r["tool_calls_total"]) >= 30),
                "cost": sum(_num(r["cost_usd"]) for r in rows),
            })
        if not vals:
            continue
        sd = statistics.stdev(vals) if len(vals) > 1 else None
        if sd is not None:
            sds.append(sd)
        dret, dacc = discovery(variant)
        if label == "Web":
            # The web arm touches no lake datasets, so gold recall is undefined,
            # not zero. compute_discovery_metrics returns 0.0; report n/a instead
            # so the table cannot be read as "searched the lake and found nothing".
            dret = dacc = None
        table.append({
            "label": label, "vals": vals, "mean": statistics.mean(vals), "sd": sd,
            "rounds": len(vals), "blank": buckets["b"],
            "cyc": statistics.mean(a["cyc"] for a in aux),
            "tools": statistics.mean(a["tools"] for a in aux),
            "cap": sum(a["cap"] for a in aux) / len(aux),
            "cost": statistics.mean(a["cost"] for a in aux),
            "dret": dret, "dacc": dacc,
        })
    if not table:
        print("no results", file=sys.stderr)
        return 1

    pooled = statistics.mean(sds) if sds else 0.0
    thresh = 2 * pooled
    web = next((t for t in table if t["label"] == "Web"), None)
    lake = [t for t in table if t["label"] != "Web"]
    best_lake = max(lake, key=lambda t: t["mean"]) if lake else None

    L, A = [], None
    A = L.append
    A("% ============================================================")
    A("% GENERATED by gen_tex.py -- do not edit by hand.")
    A(f"% metric: {key}; {len(ROUNDS)} replicate rounds; 20 tasks per arm.")
    A("% D_ret / D_acc via sana_analysis.running_analysis.discovery_metrics.")
    A("% Requires: booktabs.")
    A("% ============================================================")
    A("")
    A(r"\subsection{Retrieval source: open web versus the data lake}")
    A(r"\label{sec:web-arm}")
    A("")
    A("To test whether the retrieval axis reflects the data lake specifically or")
    A("a more general property of the agent, we replaced lake search with")
    A(r"open-web search. The arm uses the Parallel Search API and, under")
    A(r"\texttt{-{}-no-s3}, loses every lake-backed file tool: the agent keeps only")
    A(r"\texttt{download}, which fetches an arbitrary \texttt{http(s)} URL into its")
    A(r"sandbox, and \texttt{execute\_code}. All arms hold the remaining axes at")
    A(r"\texttt{search\_results=naive}, \texttt{profile=standard} and")
    A(r"\texttt{computation\_tool=standard}, since the ideal computation tools")
    A("resolve against authored data-lake records and would make a web run's")
    A(r"outcome independent of what it retrieved. Each arm was run three times.")
    A("")
    if web and best_lake:
        gap = best_lake["mean"] - web["mean"]
        verdict = ("clears" if gap > thresh else "does not clear")
        A(f"The web arm reaches {web['mean']:.1f}\\% against {best_lake['mean']:.1f}\\% for the")
        A(f"strongest lake arm, a gap of {gap:.1f}\\,pp that {verdict} the")
        A(f"$\\pm{thresh:.0f}$\\,pp two-sigma replicate threshold measured across these arms.")
        A("")
    A("The more informative quantity is where retrieval breaks down.")
    A(r"$D_{ret}$ is the fraction of gold datasets that appeared in search")
    A(r"results; $D_{acc}$ is the fraction the agent actually read.")
    if len(lake) == 3:
        spans = ", ".join(f"{t['label']} {t['dret']:.1f}\\%" for t in lake if t["dret"] is not None)
        gaps = ", ".join(f"{t['label']} {t['dret']-t['dacc']:.1f}\\,pp"
                         for t in lake if t["dret"] is not None)
        A(f"All three lake arms surface gold at a similar rate ({spans}), so they")
        A("are not separated by what search returns. They are separated by the")
        A(f"drop between finding and reading ({gaps}): oracle search wins because")
        A("the agent acts on what it was shown, not because it was shown more.")
    A("")
    A(r"For the web arm both quantities are undefined rather than zero: it")
    A("touches no lake datasets by construction, so it can neither retrieve nor")
    A("read a gold source.")
    A("")

    A(r"\begin{table}[h]")
    A(r"  \centering")
    A(r"  \caption{Retrieval-source comparison on LakeQA \textsc{subset20b}")
    A(r"  (\texttt{gpt-5-mini}, 20 tasks per arm, three replicate rounds, " + metric + r").")
    A(r"  $\bar{x}$ is the mean over rounds and $\sigma$ their standard deviation.")
    A(r"  \emph{Turns} is the mean agent cycles per task, averaged over the three")
    A(r"  rounds. $D_{ret}$ and $D_{acc}$ are")
    A(r"  retrieval and access recall of gold datasets, undefined for the web arm.}")
    A(r"  \label{tab:web-arm}")
    A(r"  \scriptsize")
    A(r"  \setlength{\tabcolsep}{4pt}")
    A(r"  \renewcommand{\arraystretch}{0.95}")
    A(r"  \resizebox{\columnwidth}{!}{%")
    A(r"  \begin{tabular}{lrrrrr}")
    A(r"    \toprule")
    A(r"    Arm & $\bar{x}$ (\%) & $\sigma$ & Turns & "
      r"$D_{ret}$ (\%) & $D_{acc}$ (\%) \\")
    A(r"    \midrule")
    for t in table:
        sd = "---" if t["sd"] is None else f"{t['sd']:.1f}"
        dr = "n/a" if t["dret"] is None else f"{t['dret']:.1f}"
        da = "n/a" if t["dacc"] is None else f"{t['dacc']:.1f}"
        mean = f"\\textbf{{{t['mean']:.1f}}}" if t is best_lake else f"{t['mean']:.1f}"
        A(f"    {t['label']} & {mean} & {sd} & {t['cyc']:.1f} & {dr} & {da} \\\\")
    A(r"    \bottomrule")
    A(r"  \end{tabular}}")
    A(r"\end{table}")
    A("")

    # ---- figure: same construction as the tier figure and the paper's fig21b,
    # drawn by make_search_axis_figure.py. Outcome first, then the two recall
    # measures that explain it.
    A(r"\begin{figure}[h]")
    A(r"  \centering")
    A(r"  \includegraphics[width=\columnwidth]{fig-search-axis}")
    A(r"  \caption{Retrieval source as an axis. Arms are ordered by how much of")
    A(r"  the lake they can see, so the baseline row is the open web and each")
    A(r"  delta is what moving into the lake buys. The right two panels are")
    A(r"  baselined on BM25, the weakest arm with a defined recall. Reading them")
    A(r"  together is the point: $D_{ret}$ is flat across the three lake arms --")
    A(r"  oracle search surfaces slightly \emph{less} gold than BM25 -- while")
    A(r"  $D_{acc}$ rises with accuracy. What separates the arms is not what")
    A(r"  search returns but how much of it the agent reads. The web arm touches")
    A(r"  no lake dataset by construction, so its recall is undefined rather than")
    A(r"  zero.}")
    A(r"  \label{fig:web-arm-axis}")
    A(r"\end{figure}")
    A("")
    A(r"\paragraph{What the web arm actually measures.}")
    A(r"\texttt{download} accepts any URL, and the model routinely constructs one")
    A("from pretrained knowledge rather than following a search result: live")
    A(r"Socrata queries against \texttt{data.cityofnewyork.us}, \texttt{data.va.gov}")
    A(r"export endpoints, and a timestamped \texttt{web.archive.org} URL. The arm")
    A("therefore measures an agent with open web access and code execution, not")
    A("web-search retrieval, and its data path may be stronger than the lake")
    A("arms' since it aggregates live authoritative sources rather than a fixed")
    A("snapshot. That liveness also produces false negatives: two failures were")
    A("arithmetically correct against current data but disagreed with gold")
    A("answers derived from the snapshot (2{,}700{,}968 against a gold")
    A("2{,}705{,}988; 2923.59 against 2941).")

    out = "\n".join(L) + "\n"
    if "--stdout" in sys.argv:
        print(out)
    else:
        (HERE / "results.tex").write_text(out)
        print(f"wrote results.tex  (metric={key}, rounds={max(t['rounds'] for t in table)}, "
              f"threshold={thresh:.1f}pp)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
