#!/usr/bin/env python3
"""Summarize follow-plan audit CSVs under agent_analysis."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


LABELS = [
    "followed",
    "mostly_followed",
    "partially_followed",
    "did_not_follow",
    "not_comparable",
]
FOLLOWED_STRICT = {"followed"}
FOLLOWED_BROAD = {"followed", "mostly_followed"}
CANONICAL_MODE_LABELS = {
    "search_i_results_i_plani_computei_k5_skills_off": "iii",
    "search_i_results_i_pland_computei_k5_skills_off": "dii",
}


def _benchmark_from_path(path: Path) -> str:
    parts = path.parts
    if "results-kramabench_semantic" in parts:
        return "kramabench"
    if "results_semantic" in parts:
        return "lakeqa"
    return "unknown"


def _safe_float(value: object) -> float | None:
    try:
        text = str(value).strip()
        if text == "":
            return None
        return float(text)
    except (TypeError, ValueError):
        return None


def _read_follow_plan_csv(path: Path) -> list[dict]:
    benchmark = _benchmark_from_path(path)
    with path.open(newline="") as handle:
        rows = []
        for row in csv.DictReader(handle):
            normalized = dict(row)
            normalized["benchmark"] = benchmark
            normalized["source_csv"] = str(path)
            if not normalized.get("model_variant"):
                try:
                    normalized["model_variant"] = path.parts[-3]
                except IndexError:
                    normalized["model_variant"] = "unknown"
            if not normalized.get("mode"):
                try:
                    normalized["mode"] = path.parts[-2]
                except IndexError:
                    normalized["mode"] = "unknown"
            if not normalized.get("plan_family"):
                normalized["plan_family"] = CANONICAL_MODE_LABELS.get(normalized["mode"], "")
            rows.append(normalized)
        return rows


def load_follow_plan_rows(input_root: Path) -> list[dict]:
    return [
        row
        for path in sorted(input_root.rglob("follow_plan.csv"))
        for row in _read_follow_plan_csv(path)
    ]


def _pct(numerator: int, denominator: int) -> float:
    return round(100.0 * numerator / denominator, 1) if denominator else 0.0


def summarize_follow_plan(rows: Iterable[dict]) -> tuple[list[dict], list[dict]]:
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        key = (
            str(row.get("benchmark", "unknown")),
            str(row.get("model_variant", "unknown")),
            str(row.get("mode", "unknown")),
        )
        grouped[key].append(row)

    summary_rows: list[dict] = []
    breakdown_rows: list[dict] = []

    for (benchmark, model_variant, mode), group_rows in sorted(grouped.items()):
        labels = Counter(str(row.get("follow_plan_label", "") or "missing_label") for row in group_rows)
        statuses = Counter(str(row.get("audit_status", "") or "missing_status") for row in group_rows)
        n_total = len(group_rows)
        n_complete = statuses.get("complete", 0)
        n_not_comparable = labels.get("not_comparable", 0)
        n_comparable = n_total - n_not_comparable
        n_strict = sum(labels.get(label, 0) for label in FOLLOWED_STRICT)
        n_broad = sum(labels.get(label, 0) for label in FOLLOWED_BROAD)
        semantic_values = [
            value
            for row in group_rows
            if (value := _safe_float(row.get("semantic_match"))) is not None
        ]
        plan_family = next((str(row.get("plan_family", "")) for row in group_rows if row.get("plan_family")), "")

        summary_row = {
            "benchmark": benchmark,
            "model_variant": model_variant,
            "runner_model": next((str(row.get("runner_model", "")) for row in group_rows if row.get("runner_model")), ""),
            "mode": mode,
            "plan_family": plan_family or CANONICAL_MODE_LABELS.get(mode, ""),
            "n_total": n_total,
            "n_complete": n_complete,
            "n_comparable": n_comparable,
            "n_not_comparable": n_not_comparable,
            "n_followed_strict": n_strict,
            "followed_strict_pct_all": _pct(n_strict, n_total),
            "followed_strict_pct_comparable": _pct(n_strict, n_comparable),
            "n_followed_or_mostly": n_broad,
            "followed_or_mostly_pct_all": _pct(n_broad, n_total),
            "followed_or_mostly_pct_comparable": _pct(n_broad, n_comparable),
            "mean_semantic_match": round(sum(semantic_values) / len(semantic_values), 4) if semantic_values else "",
        }
        for label in LABELS:
            summary_row[f"n_{label}"] = labels.get(label, 0)
            summary_row[f"{label}_pct_all"] = _pct(labels.get(label, 0), n_total)
        summary_rows.append(summary_row)

        for label, count in sorted(labels.items()):
            breakdown_rows.append({
                "benchmark": benchmark,
                "model_variant": model_variant,
                "mode": mode,
                "plan_family": summary_row["plan_family"],
                "follow_plan_label": label,
                "n": count,
                "pct_all": _pct(count, n_total),
                "pct_comparable": _pct(count, n_comparable) if label != "not_comparable" else "",
            })

    return summary_rows, breakdown_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(summary_rows: list[dict], breakdown_rows: list[dict], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "plan_following_summary.csv"
    breakdown_path = output_dir / "plan_following_label_breakdown.csv"
    json_path = output_dir / "plan_following_summary.json"
    write_csv(summary_path, summary_rows)
    write_csv(breakdown_path, breakdown_rows)
    json_path.write_text(
        json.dumps(
            {
                "summary": summary_rows,
                "label_breakdown": breakdown_rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return [summary_path, breakdown_path, json_path]


def _print_summary(summary_rows: list[dict]) -> None:
    print("Plan-following summary:")
    for row in summary_rows:
        mode_label = row["plan_family"] or row["mode"]
        print(
            "  "
            f"{row['benchmark']:<11} {row['model_variant']:<18} {mode_label:<3} "
            f"followed+mostly={row['followed_or_mostly_pct_all']:>5.1f}% "
            f"({row['n_followed_or_mostly']}/{row['n_total']}), "
            f"strict={row['followed_strict_pct_all']:>5.1f}% "
            f"({row['n_followed_strict']}/{row['n_total']}), "
            f"n/c={row['n_not_comparable']}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        default="agent_analysis/follow_plan_analysis",
        help="Root containing follow_plan.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        default="agent_analysis/follow_plan_analysis/summary",
        help="Directory for summary CSV/JSON outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_follow_plan_rows(Path(args.input_root))
    if not rows:
        raise SystemExit(f"No follow_plan.csv rows found under {args.input_root}")
    summary_rows, breakdown_rows = summarize_follow_plan(rows)
    written = write_outputs(summary_rows, breakdown_rows, Path(args.output_dir))
    _print_summary(summary_rows)
    for path in written:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
