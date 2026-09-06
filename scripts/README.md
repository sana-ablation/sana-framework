# scripts

Command-line helpers that sit outside the import packages. Two kinds live here,
and they are worth keeping distinct because they run at different times and for
different reasons.

## Running experiments

| script | what it does |
|---|---|
| `run_experiment.sh` | The entry point for a sweep: `bootstrap`, `run`, `status`, `logs`, `pull`, `stop`. Local by default; set `REMOTE_HOST` to run the same command on a remote box. |
| `materialize_task_subset.py` | Build a named task set (e.g. `benchmarks/lakeqa/tasks_20_subset/tasks`) from a manifest of task ids. |
| `smoke_agent_tools_bucket.py` | Pre-run check that the agent's data tools can reach their S3 bucket. |

An experiment is any directory under `experiments/` supplying `inputs/run.sh`;
that file is the only thing `run_experiment.sh` needs to know about it.
`experiments/` is gitignored, so sweeps stay local while the orchestration ships.

## Preparing the corpus and artifacts

These generate the offline artifacts the ideal-mode tools read at runtime. They
are run once per benchmark corpus, not per experiment.

| script | produces |
|---|---|
| `profile_datasets.py` | `table_profiles.jsonl` |
| `sample_unavailable_profiles.py` | retry pass over datasets the profiler could not reach |
| `build_task_file_manifest.py` | `task_file_manifest.jsonl` |
| `build_task_manifest_descriptions.py` | `task_file_manifest_descriptions.jsonl` |
| `merge_table_descriptions.py` | `descriptions.jsonl` |
| `build_snippet_jsonl.py` | `snippets.jsonl` |
| `check_manifest_coverage.py` | coverage audit over the four above |

The last five run in that order; each consumes the previous one's output. See
`dataindexing/README.md`, which owns the upstream parquet stages.

## What is deliberately not here

Remote execution used to be `remote_setup_run.sh` and `remote_pull_outputs.sh`.
Both defaulted to a different repository, a retired host and a key that is not
in this project, and neither could target a task set other than the built-in
default. `run_experiment.sh` replaces them.

The kramabench corpus-import and S3-upload scripts have been removed too. Their
input tree (`other-benchmarks/`) is not part of this repository, their outputs
are already committed under `benchmarks/kramabench/`, and their tests only ever
exercised them against synthetic temporary directories.
