# sana_analysis

Canonical analysis package for semantic result processing, paper-result
generation, and audit aggregation.

## Contents

- `run_mode_analysis_semantic.py`: main mode-analysis entry point for semantic
  result trees.
- `run_sana_mode_analysis.py`: analysis for SANA-specific result layouts.
- `answer_failure_audit_runner.py` and `answer_failure_rerun_queue.py`: answer
  failure audit helpers.
- `running_analysis/`: reusable metric and validation modules.
- `report_generator/`: paper figure, answer-failure, and export helpers.

Generated artifacts should be written to `analysis_results*`, `agent_analysis/`,
or `paper_figures/`, not back into this package.
