#!/usr/bin/env python3
"""Summarise the four-arm web-arm sweep: accuracy, cost, and web-specific signals."""

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "tmp" / "results-webarm"
LOGS = ROOT / "tmp" / "logs-webarm"

ARMS = [
    ("web", "search_web__results_naive__profile_standard__compute_standard__nos3__skills_off"),
    ("ideal", "search_ideal__results_naive__profile_standard__compute_standard__skills_off"),
    ("standard", "search_standard__results_naive__profile_standard__compute_standard__skills_off"),
    ("naive", "search_naive__results_naive__profile_standard__compute_standard__skills_off"),
]
MODEL_DIR = "openai_gpt-5-mini"


def _num(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_rows(variant):
    path = RESULTS / "modes" / MODEL_DIR / variant / "eval_results.csv"
    if not path.is_file():
        # results-output-dir layout may nest differently; search for it.
        matches = list(RESULTS.rglob("eval_results.csv"))
        path = next((m for m in matches if variant in str(m)), None)
        if path is None:
            return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main():
    print(f"{'arm':<10}{'n':>4}{'exact':>8}{'f1':>8}{'cycles':>8}{'cost$':>9}{'tok_in':>10}{'errors':>8}")
    print("-" * 65)
    table = {}
    for name, variant in ARMS:
        rows = load_rows(variant)
        if not rows:
            print(f"{name:<10}{'--- no results yet ---':>40}")
            continue
        n = len(rows)
        # exact_match is written as a float ("1.0"/"0.0"), not a bool string.
        exact = sum(1 for r in rows if _num(r.get("exact_match")) >= 1.0)
        f1 = sum(_num(r.get("f1_score")) for r in rows) / n
        cycles = sum(_num(r.get("cycle_count")) for r in rows) / n
        cost = sum(_num(r.get("cost_usd")) for r in rows)
        tok = sum(_num(r.get("input_tokens")) for r in rows) / n
        errs = sum(1 for r in rows if (r.get("error") or "").strip())
        table[name] = rows
        print(f"{name:<10}{n:>4}{exact/n:>8.1%}{f1:>8.3f}{cycles:>8.1f}{cost:>9.3f}{tok:>10.0f}{errs:>8}")

    # Web-specific: download provenance and any route around the download tool.
    web_log = ROOT / "tmp" / "webarm-web.log"
    if web_log.is_file():
        text = web_log.read_text(errors="replace")
        print("\n--- web arm specifics ---")
        dl = len(re.findall(r"Executing: download\(", text))
        sw = len(re.findall(r"Executing: search_web\(", text))
        ec = len(re.findall(r"Executing: execute_code\(", text))
        print(f"search_web calls : {sw}")
        print(f"download calls   : {dl}")
        print(f"execute_code     : {ec}")

        bypass = Counter(re.findall(r"curl|wget|urlopen|urllib\.request|requests\.get", text))
        print(f"bypass attempts  : {dict(bypass) if bypass else 'none'}")

        prov = Counter()
        for trace in (RESULTS / "traces").rglob("*.jsonl") if (RESULTS / "traces").is_dir() else []:
            if "search_web__" not in str(trace):
                continue
            for line in trace.read_text(errors="replace").splitlines():
                try:
                    ev = json.loads(line)
                except Exception:
                    continue
                if ev.get("tool") == "download":
                    prov[ev.get("status", "?")] += 1
        print(f"download outcomes: {dict(prov) if prov else 'n/a'}")


if __name__ == "__main__":
    sys.exit(main())
