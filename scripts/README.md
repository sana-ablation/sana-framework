# scripts

One-off and reusable command-line helpers that sit outside the import packages.

## Common groups

- Benchmark import and manifest helpers: `build_*`, `check_manifest_coverage.py`,
  and `merge_table_descriptions.py`.
- Dataset and artifact profiling: `profile_datasets.py` and
  `sample_unavailable_profiles.py`.
- Remote execution helpers: `remote_setup_run.sh` and `remote_pull_outputs.sh`.
- Smoke checks: `smoke_agent_tools_bucket.py`.
- Upload helpers for benchmark data: `upload_kramabench_*.py`.

When a helper becomes runtime-critical or needs stable imports, move it into
`sana_evaluation/`, `sana_analysis/`, or `dataindexing/`.
