"""Figure generation and paper-ready export for semantic mode analyses.

`figures.py` and `export.py` are the CLIs you run directly. `delta_figures.py`
is not run first -- `run_mode_analysis.py` imports and invokes it during the
analysis itself, which is why its name dropped the `run_` prefix.
`plan_ablation_figure.py` is a one-off, with hardcoded
`agent_analysis/plan_default_analysis/` roots and a two-model list.

`tier_ablation.py` is the other CLI you run directly, and the only module here
that does not read an analysis bundle: its artifacts are means over replicate
rounds, a dimension `summary.json` does not carry, so it reads the round trees
of a sweep directly. It borrows `delta_figures.plot_horizontal_delta_bars` for
the panels and `variants.find_variant` for cell lookup.
"""
