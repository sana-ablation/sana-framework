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

### After `pull`: normalise the variant label

The variant directory label has changed twice. The planning axis was renamed
from `--profile` to `--plan` (`...__profile_ideal__...` became
`...__plan_ideal__...`), and then the layout itself changed: the three ablation
axes lead -- `search`, `plan`, `compute` -- with the search-result richness
modifier trailing them, recorded canonically as `rich`/`minimal` rather than as
whichever alias the caller typed.

So the canonical form is:

    search_ideal__plan_ideal__compute_ideal__results_rich__k5__skills_off

Local trees are migrated. **The remote host's trees are not**, and cannot be
migrated from here. If `REMOTE_HOST` is running pre-rename code, a `pull` lands
old-layout directories beside locally-produced canonical ones for the same
condition, and nothing downstream treats them as the same variant.

After any pull from a host that might be stale, run:

    python scripts/migrate_variant_label_layout.py experiments          # dry run
    python scripts/migrate_variant_label_layout.py experiments --apply

One pass fixes both changes: it accepts `profile_` as well as `plan_`, so a
tree that never saw the first migration does not need two passes. It is
idempotent -- a second run reports nothing to do -- and it renames directories
only, never opening a CSV, JSONL or log. Every rename is appended to
`scripts/variant_label_layout_mapping.tsv` in execution order, so the migration
reverses by replaying that file bottom-to-top with the columns swapped.

`--expect N` refuses to act unless exactly N directories need renaming, which is
worth passing when you already know the count from the dry run.

The durable fix is to update the remote to current code so it writes the
canonical label itself; the script is for trees already written.

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
