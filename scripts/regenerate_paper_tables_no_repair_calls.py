#!/usr/bin/env python3
"""Regenerate paper result tables from existing outputs.

This does not run agents or evaluations. It recomputes analysis summaries from
existing semantic CSVs and trace JSONL files, using discovery metrics that
exclude ideal repair bookkeeping trace events.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sana_analysis.report_generator.export_paper_results import export_paper_results
from sana_analysis.report_generator.paper_figure_generator import _benchmark_defaults
from sana_analysis.run_mode_analysis_semantic import run_analysis


def regenerate_benchmark(benchmark: str, *, include_figures: bool) -> None:
    defaults = _benchmark_defaults(benchmark)
    print(f"Regenerating analysis summary for {benchmark}...")
    run_analysis(
        results_dir=str(defaults.results_dir),
        base_results_dir=str(defaults.base_results_dir),
        turn_waste_grouped_dir=None,
        traces_dir=str(defaults.traces_dir),
        tasks_dir=str(defaults.tasks_dir),
        output_dir=str(defaults.analysis_dir),
        model_filter=None,
        no_figures=not include_figures,
    )
    print(f"Exporting paper tables for {benchmark}...")
    outputs = export_paper_results(
        defaults.analysis_dir,
        Path("sana_framework_paper"),
        benchmark=benchmark,
    )
    print(f"  Wrote {outputs['main_ablation_table']}")
    print(f"  Wrote {outputs['canonical_table']}")
    print(f"  Wrote {outputs['table']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark",
        choices=["lakeqa", "kramabench", "both"],
        default="both",
        help="Benchmark tables to regenerate.",
    )
    parser.add_argument(
        "--include-figures",
        action="store_true",
        help="Also regenerate analysis figures. Tables do not require this.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    benchmarks = ["lakeqa", "kramabench"] if args.benchmark == "both" else [args.benchmark]
    for benchmark in benchmarks:
        regenerate_benchmark(benchmark, include_figures=args.include_figures)


if __name__ == "__main__":
    main()
