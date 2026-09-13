# paper_figures

Paper-ready figure exports and supporting analysis figures.

Top-level PDFs are final or near-final figure artifacts. The
`agent_analysis/` subdirectory mirrors selected behavioral-analysis outputs used
to create supporting paper figures.

Regenerate these from `sana_analysis.paper` or the relevant
analysis scripts rather than editing figure outputs directly.

## tier_ablation_*

The replicate-averaged model-tier ablation artifacts -- `tier_ablation_table.tex`
and `tier_ablation_axis_delta.{pdf,png}` -- come from one command, which reads a
sweep's replicate round trees directly:

    .venv/bin/python -m sana_analysis.paper.tier_ablation \
        --experiment-dir experiments/2026-08-31-model-tiers-subset20b

Re-running it picks up rounds that have landed since; a cell averaged over fewer
rounds than the sweep has is marked with its own `n` rather than folded in
silently. The PNG is for review, the PDF is the paper artifact.
