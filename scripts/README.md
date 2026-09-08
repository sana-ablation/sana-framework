# scripts

Command-line helpers that sit outside the import packages. Two kinds live here,
and they are worth keeping distinct because they run at different times and for
different reasons.

## Running experiments

| script | what it does |
|---|---|
| `run_experiment.sh` | The entry point for a sweep: `bootstrap`, `run`, `status`, `logs`, `pull`, `stop`. Local by default; set `REMOTE_HOST` to run the same command on a remote box. |
| `materialize_task_subset.py` | Build a named task set (e.g. `benchmarks/lakeqa/tasks_20_subset/tasks`) from a manifest of task ids. |
| `smoke_lake_bucket.py` | Pre-run check that the agent's data tools can reach their S3 bucket. |

An experiment is any directory under `experiments/` supplying `inputs/run.sh`;
that file is the only thing `run_experiment.sh` needs to know about it.
`experiments/` is gitignored, so sweeps stay local while the orchestration ships.

### After `pull`: watch for the old variant label

The planning axis was renamed from `--profile` to `--plan`, and the variant
directory label with it (`...__profile_ideal__...` became `...__plan_ideal__...`).
Local trees were migrated once, by hand, and that migration is done — the
script that did it is gone. The remote host's trees were never migrated and
cannot be from here: if `REMOTE_HOST` is still running pre-rename code, a
`pull` lands `__profile_` directories beside locally-produced `__plan_` ones
for the same condition, and nothing downstream (aggregation, comparison)
treats them as the same variant. The fix is to update the remote to the
current code so it writes `__plan_` itself; there is no longer a script to
rename a tree after the fact, so a `__profile_` tree found post-pull means the
remote is stale.

## Preparing the corpus and artifacts

Moved to `dataindexing/cli/`, which already owns offline artifact generation.
The five-stage pipeline now has one entry point instead of five scripts that
each hardcoded a lakeqa artifact path:

    python -m dataindexing.cli.benchmark_artifacts --benchmark lakeqa all

Stages: `manifest`, `describe`, `merge-descriptions`, `snippets PARQUET`,
`check`. `all` runs every stage but `snippets`, whose input parquet is not
derivable from the benchmark name. Dataset profiling stays alongside them as
`dataindexing/cli/profile_datasets.py`.

## What is deliberately not here

Remote execution used to be `remote_setup_run.sh` and `remote_pull_outputs.sh`.
Both defaulted to a different repository, a retired host and a key that is not
in this project, and neither could target a task set other than the built-in
default. `run_experiment.sh` replaces them.

The kramabench corpus-import and S3-upload scripts have been removed too. Their
input tree (`other-benchmarks/`) is not part of this repository, their outputs
are already committed under `benchmarks/kramabench/`, and their tests only ever
exercised them against synthetic temporary directories.
