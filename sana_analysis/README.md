# sana_analysis

Canonical analysis package for semantic result processing, paper-result
generation, and audit aggregation.

## Contents

- `prepare_sana_result_tree.py`: stages raw SANA result trees into the
  `modes/`/`logs/`/`traces/` layout `run_mode_analysis.py` reads.
- `run_mode_analysis.py`: main mode-analysis entry point for semantic
  result trees.
- `answer_failure_audit_runner.py` and `answer_failure_rerun_queue.py`: answer
  failure audit helpers.
- `metrics/`: reusable metric and validation modules.
- `paper/`: paper figure and export helpers.

Generated artifacts should be written to `analysis_results*`, `agent_analysis/`,
or `paper_figures/`, not back into this package.
