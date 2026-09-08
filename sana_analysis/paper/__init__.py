"""Figure generation and paper-ready export for semantic mode analyses.

`figures.py` and `export.py` are the CLIs you run directly. `delta_figures.py`
is not run first -- `run_mode_analysis.py` imports and invokes it during the
analysis itself, which is why its name dropped the `run_` prefix.
`plan_ablation_figure.py` is a one-off, with hardcoded
`agent_analysis/plan_default_analysis/` roots and a two-model list.
"""
