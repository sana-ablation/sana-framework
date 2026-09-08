# sana_analysis

Canonical analysis package for semantic result processing, paper-result
generation, and audit aggregation.

## Contents

Mode analysis runs in two steps:

1. `prepare_sana_result_tree.py`: stage raw SANA runs into the
   `modes/`, `logs/`, `traces/` layout the analyzer reads. Only needed for
   result trees that are not already in that shape.
2. `run_mode_analysis.py`: the analysis itself, over semantic-audited
   `eval_results.csv` files.

- `answer_failure/`: a separate pipeline that classifies *why* wrong answers
  were wrong, using external model calls. See its `__init__.py` for the pass
  ordering.
- `metrics/`: reusable metric computations.
- `paper/`: figure generation and paper-ready export.

Generated artifacts should be written to `analysis_results*`, `agent_analysis/`,
or `paper_figures/`, not back into this package.
