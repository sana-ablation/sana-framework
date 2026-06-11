# benchmarks

Canonical benchmark assets used by SANA evaluation runs.

## Layout

- `lakeqa/tasks-mini/`: maintained LakeQA mini task set.
- `kramabench/tasks-mini/`: maintained Kramabench mini task set.
- `*/tasks-mini/tasks/`: task JSON files grouped by bucket.
- `*/tasks-mini/runtime-profiles/`: ideal-mode runtime profiles for planning,
  search, profile, and compute ablations.
- `*/tasks-mini/artifacts/`: benchmark-local JSONL dependencies, including
  descriptions, snippets, schemas, and file profiles.

These directories are the source of truth for maintained benchmark inputs.
Generated run outputs belong in `results*` or `sana-results*`, not here.
