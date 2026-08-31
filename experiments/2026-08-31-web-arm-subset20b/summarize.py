#!/usr/bin/env python3
"""Cross-arm table for the subset20b sweep.

Primary metric is `semantic_match`, produced by the semantic-eval-auditor pass.
Until that pass has run, the semantic columns are absent and the table falls back
to exact_match with a warning. F1 is deliberately not reported: on LakeQA most
golds are a single token, where it equals exact match by construction, and where
it diverges it mostly measures whether articles were reproduced.
"""

import csv
import re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
LOGS = HERE / "logs"
MODEL_DIR = "openai_gpt-5-mini"

ARMS = [
    ("web", "search_web__results_naive__profile_standard__compute_standard__nos3__skills_off"),
    ("ideal", "search_ideal__results_naive__profile_standard__compute_standard__skills_off"),
    ("standard", "search_standard__results_naive__profile_standard__compute_standard__skills_off"),
    ("naive", "search_naive__results_naive__profile_standard__compute_standard__skills_off"),
]

BUCKETS = ["semantic_correct", "semantic_incorrect", "answer_unknown_blank"]


def _num(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _median(values):
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def load_rows(variant):
    """Prefer the semantic-audited CSV; fall back to the raw one."""
    for root in (RESULTS.parent / "results_semantic", RESULTS):
        path = root / "modes" / MODEL_DIR / variant / "eval_results.csv"
        if path.is_file():
            with open(path, newline="") as f:
                return list(csv.DictReader(f)), root is not RESULTS
    matches = [m for m in RESULTS.rglob("eval_results.csv") if variant in str(m)]
    if matches:
        with open(matches[0], newline="") as f:
            return list(csv.DictReader(f)), False
    return [], False


def main():
    any_semantic = False
    print(f"{'arm':<10}{'n':>4}{'semantic':>10}{'exact':>8}{'cycles':>8}{'tools':>7}"
          f"{'med_rt':>8}{'cost$':>9}{'blank':>7}")
    print("-" * 71)

    for name, variant in ARMS:
        rows, is_semantic = load_rows(variant)
        if not rows:
            print(f"{name:<10}{'--- no results yet ---':>40}")
            continue
        any_semantic = any_semantic or is_semantic
        n = len(rows)
        has_sem = "semantic_match" in rows[0]
        sem = sum(1 for r in rows if _num(r.get("semantic_match")) >= 1.0) if has_sem else None
        # exact_match is written as a float ("1.0"/"0.0"), not a bool string.
        exact = sum(1 for r in rows if _num(r.get("exact_match")) >= 1.0)
        cycles = sum(_num(r.get("cycle_count")) for r in rows) / n
        tools = sum(_num(r.get("tool_calls_total")) for r in rows) / n
        med_rt = _median([_num(r.get("runtime_seconds")) for r in rows])
        cost = sum(_num(r.get("cost_usd")) for r in rows)
        blank = sum(1 for r in rows if not (r.get("predicted_answer") or "").strip())
        sem_s = f"{sem/n:>9.1%}" if sem is not None else f"{'n/a':>9}"
        print(f"{name:<10}{n:>4}{sem_s}{exact/n:>8.1%}{cycles:>8.1f}{tools:>7.1f}"
              f"{med_rt:>8.0f}{cost:>9.3f}{blank:>7}")

    if not any_semantic:
        print("\n  NOTE: no semantic-audited CSVs found — showing exact_match only.")
        print("  Run the semantic-eval-auditor pass to populate semantic_match.")

    # Semantic bucket breakdown, once available.
    for name, variant in ARMS:
        rows, _ = load_rows(variant)
        if rows and "semantic_bucket" in (rows[0] or {}):
            counts = Counter(r.get("semantic_bucket", "?") for r in rows)
            line = "  ".join(f"{b}={counts.get(b, 0)}" for b in BUCKETS)
            print(f"  {name:<10} {line}")

    web_log = LOGS / "arm-web.log"
    if web_log.is_file():
        text = web_log.read_text(errors="replace")
        print("\n--- web arm specifics ---")
        for label, pat in [("search_web", r"Executing: search_web\("),
                           ("download", r"Executing: download\("),
                           ("execute_code", r"Executing: execute_code\(")]:
            print(f"  {label:<13}: {len(re.findall(pat, text))}")
        bypass = Counter(re.findall(r"curl|wget|urlopen|urllib\.request|requests\.get", text))
        print(f"  {'bypass':<13}: {dict(bypass) if bypass else 'none'}")


if __name__ == "__main__":
    main()
