#!/usr/bin/env python3
"""Analyse every round of an experiment from the experiment directory alone.

`run_mode_analysis` takes four paths that must agree with each other, and
getting `--tasks-dir` wrong does not raise -- `load_task_gold_counts` returns
`{}` and the discovery metrics come back empty. Every one of those paths is
derivable from the experiment directory, so this derives them.

Round naming is not uniform on disk: some experiments put round 1 in `results/`
and later rounds in `results-rep2/`, others use `results-rep1/` throughout. The
suffix is read off the directory name rather than assumed.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional

from sana_analysis.run_mode_analysis import run_analysis
from sana_analysis.semantic_mirror import mirror_is_complete

# Metrics worth a mean and a spread across rounds. `semantic_match` is handled
# specially: an unjudged round's value is the lexical score under a semantic
# name, so it is pooled only across rounds that actually had a judge.
COMBINED_METRICS = [
    "exact_match",
    "semantic_match",
    "D_ret",
    "D_acc",
    "avg_cost_usd",
    "avg_tool_calls_total",
    "avg_search_calls",
    "avg_runtime_seconds",
]
IDENTITY_FIELDS = ["condition_model", "model", "variant"]


@dataclass(frozen=True)
class RoundPaths:
    name: str
    results_dir: Path
    base_results_dir: Path
    traces_dir: Path
    tasks_dir: Optional[str]
    logs_dir: Optional[Path]
    semantic: bool
    semantic_reason: str
    mirror_dir: Optional[Path]
    tasks_dir_spread: list[tuple[str, int]] = field(default_factory=list)


def discover_rounds(exp: Path) -> list[Path]:
    """Every result round in the experiment, in name order.

    `*_semantic` trees are mirrors, not rounds. `results.tex` is not a directory.
    """
    return sorted(
        path
        for path in exp.glob("results*")
        if path.is_dir() and not path.name.endswith("_semantic")
    )


def round_suffix(results_dir: Path) -> str:
    """`results` -> ``, `results-rep2` -> `-rep2`, `results-rep1` -> `-rep1`."""
    return results_dir.name[len("results"):]


def derive_tasks_dir(results_dir: Path) -> tuple[Optional[str], list[tuple[str, int]]]:
    """The task-set root, from the `task_id` paths the run itself recorded.

    `task_id` holds the full repo-relative path to each task JSON, so the run
    tells us which task set it used. Reading it out of `inputs/*.sh` does not
    work: the task set is declared four different ways across four experiments,
    and one round on disk mixes two of them. Majority wins; the spread is
    returned so a caller can say what it ignored.
    """
    counts: Counter = Counter()
    for csv_path in sorted(results_dir.rglob("eval_results.csv")):
        with csv_path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                task_id = row.get("task_id", "")
                if task_id:
                    counts[str(Path(task_id).parent.parent)] += 1
                break
    if not counts:
        return None, []
    return counts.most_common(1)[0][0], counts.most_common()


def resolve_round(
    exp: Path,
    results_dir: Path,
    tasks_dir_override: Optional[str] = None,
) -> RoundPaths:
    suffix = round_suffix(results_dir)
    mirror = exp / f"{results_dir.name}_semantic"
    semantic, problems = mirror_is_complete(results_dir / "modes", mirror / "modes")
    derived, spread = derive_tasks_dir(results_dir)

    # No fallback to `logs/`: those belong to round 1, and labelling them as this
    # round's would put one round's error evidence against another's rows.
    logs = exp / f"logs{suffix}"

    return RoundPaths(
        name=results_dir.name,
        results_dir=(mirror if semantic else results_dir) / "modes",
        base_results_dir=results_dir / "modes",
        traces_dir=results_dir / "traces" / "modes",
        tasks_dir=tasks_dir_override or derived,
        logs_dir=logs if logs.is_dir() else None,
        semantic=semantic,
        semantic_reason=("mirror complete" if semantic else (problems[0] if problems else "no mirror")),
        mirror_dir=mirror if mirror.is_dir() else None,
        tasks_dir_spread=spread,
    )


@lru_cache(maxsize=None)
def _task_set_size(task_root: str) -> Optional[int]:
    """How many task JSONs a complete cell should hold. None if the set is gone."""
    root = Path(task_root)
    if not root.is_dir():
        return None
    return len(list(root.rglob("*.json")))


@dataclass(frozen=True)
class RoundStatus:
    name: str
    number: int
    cells: int
    full_cells: int
    complete: bool
    reason: str


def _round_cells(results_dir: Path) -> list[tuple[int, Optional[int]]]:
    """(rows, expected) per cell.

    `expected` comes from that cell's own task set rather than the round's
    majority: `model-tiers/results-rep3` mixes two, and a majority rule would
    misjudge the odd cell out.
    """
    out: list[tuple[int, Optional[int]]] = []
    for csv_path in sorted(results_dir.rglob("eval_results.csv")):
        with csv_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        expected = None
        if rows and rows[0].get("task_id"):
            expected = _task_set_size(str(Path(rows[0]["task_id"]).parent.parent))
        out.append((len(rows), expected))
    return out


def round_statuses(exp: Path) -> list[RoundStatus]:
    """Every round, numbered by position, with whether its sweep finished.

    A round with a directory is not necessarily a round that completed: an
    OOM-killed sweep leaves one behind. Advancing past it would put a
    permanently half-finished rep into the across-round summary.
    """
    per_round = [(results_dir, _round_cells(results_dir)) for results_dir in discover_rounds(exp)]
    fullest = max((len(cells) for _dir, cells in per_round), default=0)

    statuses: list[RoundStatus] = []
    for number, (results_dir, cells) in enumerate(per_round, start=1):
        full = sum(1 for rows, expected in cells if expected is not None and rows == expected)
        if not cells:
            complete, reason = False, "no eval_results.csv"
        elif any(expected is None for _rows, expected in cells):
            complete, reason = False, "task set not on disk; cannot verify row counts"
        elif full != len(cells):
            complete, reason = False, f"{len(cells) - full} cell(s) short of a full task set"
        elif len(cells) < fullest:
            complete, reason = False, f"{fullest - len(cells)} cell(s) missing vs the fullest round"
        else:
            complete, reason = True, "complete"
        statuses.append(RoundStatus(results_dir.name, number, len(cells), full, complete, reason))
    return statuses


def next_round(exp: Path) -> tuple[int, str]:
    """The round number to run next, and why.

    The earliest incomplete round wins: a gap in round 2 matters more than
    starting round 4. The driver maps the number to a directory using its own
    convention, which is why this returns a number and not a path.
    """
    statuses = round_statuses(exp)
    for status in statuses:
        if not status.complete:
            return status.number, f"resuming round {status.number} ({status.name}): {status.reason}"
    total = len(statuses)
    return total + 1, f"all {total} round(s) complete; starting round {total + 1}"


def format_round_table(statuses: list[RoundStatus]) -> str:
    if not statuses:
        return "  (no rounds yet)"
    return "\n".join(
        f"  round {status.number}  {status.name:<16} {status.full_cells}/{status.cells} cells  "
        + ("complete" if status.complete else f"INCOMPLETE ({status.reason})")
        for status in statuses
    )


def combine_rounds(rounds: list[dict]) -> list[dict]:
    """Mean and spread per condition across rounds.

    `rounds` is `[{"round": name, "semantic": bool, "rows": summary_rows}, ...]`.
    """
    by_key: dict[str, dict] = {}
    for rnd in rounds:
        for row in rnd["rows"]:
            key = str(row["condition_model"])
            entry = by_key.setdefault(key, {
                **{name: row.get(name) for name in IDENTITY_FIELDS},
                "_values": {metric: [] for metric in COMBINED_METRICS},
                "_rounds": [],
                "_n": [],
                "_semantic_rounds": [],
            })
            entry["_rounds"].append(rnd["round"])
            entry["_n"].append(int(row.get("n") or 0))
            if rnd["semantic"]:
                entry["_semantic_rounds"].append(rnd["round"])
            for metric in COMBINED_METRICS:
                if metric == "semantic_match" and not rnd["semantic"]:
                    continue
                value = row.get(metric)
                if value is not None:
                    entry["_values"][metric].append(float(value))

    out: list[dict] = []
    for _key, entry in sorted(by_key.items()):
        row = {name: entry[name] for name in IDENTITY_FIELDS}
        row["n_rounds"] = len(entry["_rounds"])
        row["rounds"] = entry["_rounds"]
        row["n_total"] = sum(entry["_n"])
        row["n_rounds_semantic"] = len(entry["_semantic_rounds"])
        for metric in COMBINED_METRICS:
            values = entry["_values"][metric]
            row[f"{metric}_mean"] = round(statistics.fmean(values), 4) if values else None
            row[f"{metric}_sd"] = (
                round(statistics.stdev(values), 4) if len(values) > 1 else (0.0 if values else None)
            )
            row[f"{metric}_n"] = len(values)
        out.append(row)
    return out


def combined_fieldnames() -> list[str]:
    names = list(IDENTITY_FIELDS) + ["n_rounds", "n_rounds_semantic", "n_total", "rounds"]
    for metric in COMBINED_METRICS:
        names += [f"{metric}_mean", f"{metric}_sd", f"{metric}_n"]
    return names


def analyse_experiment(
    exp: Path,
    *,
    round_filter: Optional[str] = None,
    tasks_dir: Optional[str] = None,
    output_dir: Optional[Path] = None,
    no_figures: bool = False,
) -> dict:
    exp = Path(exp)
    rounds = discover_rounds(exp)
    if round_filter:
        rounds = [r for r in rounds if round_filter in r.name]
    if not rounds:
        raise ValueError(
            f"no result rounds under {exp}: expected directories named results or results-rep<N>"
        )

    out_root = Path(output_dir) if output_dir else exp / "analysis"
    collected: list[dict] = []
    manifest: list[dict] = []

    for results_dir in rounds:
        paths = resolve_round(exp, results_dir, tasks_dir_override=tasks_dir)
        print(f"\n=== {paths.name} ===")
        print(f"  path    : {'semantic' if paths.semantic else 'exact_match only'} ({paths.semantic_reason})")
        print(f"  results : {paths.results_dir}")
        print(f"  traces  : {paths.traces_dir}")
        print(f"  tasks   : {paths.tasks_dir or 'UNKNOWN -- discovery metrics will be empty'}")
        print(f"  logs    : {paths.logs_dir or 'none for this round'}")
        if len(paths.tasks_dir_spread) > 1:
            print(f"  warning : this round mixes {len(paths.tasks_dir_spread)} task sets; using the majority")
            for root, count in paths.tasks_dir_spread:
                print(f"              {count:>4} cells  {root}")
        if paths.tasks_dir and not Path(paths.tasks_dir).is_dir():
            print(f"  warning : task set {paths.tasks_dir} is not on disk; discovery metrics will be empty")

        result = run_analysis(
            results_dir=str(paths.results_dir),
            base_results_dir=str(paths.base_results_dir),
            turn_waste_grouped_dir=None,
            traces_dir=str(paths.traces_dir),
            tasks_dir=paths.tasks_dir or "",
            output_dir=str(out_root / paths.name),
            no_figures=no_figures,
            no_semantic=not paths.semantic,
        )
        collected.append({"round": paths.name, "semantic": paths.semantic, "rows": result["summary"]})
        manifest.append({
            "round": paths.name,
            "semantic": paths.semantic,
            "semantic_reason": paths.semantic_reason,
            "results_dir": str(paths.results_dir),
            "base_results_dir": str(paths.base_results_dir),
            "traces_dir": str(paths.traces_dir),
            "tasks_dir": paths.tasks_dir,
            "tasks_dir_spread": paths.tasks_dir_spread,
            "logs_dir": str(paths.logs_dir) if paths.logs_dir else None,
            "output_dir": str(out_root / paths.name),
        })

    combined = combine_rounds(collected)
    combined_dir = out_root / "combined"
    combined_dir.mkdir(parents=True, exist_ok=True)
    (combined_dir / "combined_summary.json").write_text(json.dumps(combined, indent=2) + "\n")
    with (combined_dir / "combined_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=combined_fieldnames(), extrasaction="ignore")
        writer.writeheader()
        for row in combined:
            writer.writerow({**row, "rounds": ";".join(row["rounds"])})
    (combined_dir / "rounds.json").write_text(
        json.dumps({"experiment": exp.name, "rounds": manifest}, indent=2) + "\n"
    )

    print(f"\n=== combined across {len(collected)} round(s) -> {combined_dir} ===")
    judged = [r for r in collected if r["semantic"]]
    metric = "semantic_match" if judged else "exact_match"
    if judged and len(judged) != len(collected):
        print(f"  note: {len(judged)} of {len(collected)} rounds were judged; "
              "semantic_match is pooled over those only")
    print(f"  {'condition_model':<72} {'rounds':>6} {metric + ' mean':>20} {'sd':>8}")
    for row in combined:
        mean, sd = row[f"{metric}_mean"], row[f"{metric}_sd"]
        mean_text = f"{mean * 100:.1f}%" if mean is not None else "N/A"
        sd_text = f"{sd * 100:.1f}pp" if sd is not None else "N/A"
        print(f"  {str(row['condition_model']):<72} {row['n_rounds']:>6} {mean_text:>20} {sd_text:>8}")

    return {"rounds": manifest, "combined": combined, "output_dir": str(out_root)}


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True,
                        help="Experiment directory, or just its name under experiments/.")
    parser.add_argument("--round", default="",
                        help="Substring filter on round directory name, e.g. rep2.")
    parser.add_argument("--tasks-dir", default="",
                        help="Override the task set derived from the recorded task_id paths.")
    parser.add_argument("--output-dir", default="",
                        help="Default: <experiment>/analysis.")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument(
        "--print-audit-plan",
        action="store_true",
        help="Print one TAB-separated 'round<TAB>source<TAB>mirror<TAB>logs' line "
             "per round that has no complete semantic mirror, then exit. Used by "
             "analyse_with_autoaudit.sh; makes no changes and runs no analysis.",
    )
    parser.add_argument("--print-rounds", action="store_true",
                        help="Print the round status table and exit. Runs nothing.")
    parser.add_argument("--print-next-round", action="store_true",
                        help="Print just the next round number and exit. Used by "
                             "run_experiment.sh run-next.")
    return parser.parse_args(argv)


def resolve_experiment(raw: str) -> Path:
    candidate = Path(raw)
    if candidate.is_dir():
        return candidate
    under_experiments = Path("experiments") / raw
    if under_experiments.is_dir():
        return under_experiments
    raise ValueError(f"no such experiment: {raw} (looked for {candidate} and {under_experiments})")


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if args.print_rounds:
        exp = resolve_experiment(args.experiment)
        print(format_round_table(round_statuses(exp)))
        number, why = next_round(exp)
        print(f"  -> next: round {number} ({why})")
        return 0

    if args.print_next_round:
        print(next_round(resolve_experiment(args.experiment))[0])
        return 0

    if args.print_audit_plan:
        exp = resolve_experiment(args.experiment)
        rounds = discover_rounds(exp)
        if args.round:
            rounds = [r for r in rounds if args.round in r.name]
        for results_dir in rounds:
            paths = resolve_round(exp, results_dir, tasks_dir_override=args.tasks_dir or None)
            if paths.semantic:
                continue
            logs = str(paths.logs_dir) if paths.logs_dir else ""
            print(f"{paths.name}\t{results_dir}\t{exp / (results_dir.name + '_semantic')}\t{logs}")
        return 0
    analyse_experiment(
        resolve_experiment(args.experiment),
        round_filter=args.round or None,
        tasks_dir=args.tasks_dir or None,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        no_figures=args.no_figures,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
