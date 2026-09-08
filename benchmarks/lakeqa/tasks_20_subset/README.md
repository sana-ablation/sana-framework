# LakeQA tasks_20_subset

A 20-task subset of `benchmarks/lakeqa/tasks-mini`, stratified jointly on node
count (`k-*`) and dataset count (`d-*`) so the sweep keeps the shape of the full
set rather than sampling uniformly from it. Eleven of the parent set's
`k-<n>-d-<m>` buckets are represented, from `k-3-d-2` to `k-6-d-3`.

It exists because a full leave-one-out ablation is seven cells per model per
round, and at three replicate rounds and four models that is 84 runs. At 20
tasks a cell costs minutes; at the full set it costs hours, which in practice
means a grid gets run once and its variance never measured.

## Running it

    python -m sana_evaluation.cli \
        --benchmark lakeqa --task-set tasks_20_subset --all-tasks ...

`--task-set` accepts either the short name or the full path
(`benchmarks/lakeqa/tasks_20_subset/tasks`).

## Runtime profiles are shared, not copied

Tasks keep their `<bucket>/<task>.json` path relative to the set's `tasks/`
directory, so `runtime_profile_store` resolves them against
`benchmarks/lakeqa/tasks-mini/runtime-profiles` — the parent set's profiles.
There is nothing to regenerate and nothing to keep in sync.

That only works because the profile lookup matches `benchmarks/<benchmark>/<set>/tasks/`
for any set name. It used to hardcode `tasks-mini`, which meant a new set did
not error — it resolved to a *wrong* profile path and failed later with a
confusing missing-file message.

## Reproducing the selection

`manifest.json` lists the 20 task ids. To rebuild the tree from it:

    python scripts/materialize_task_subset.py \
        --manifest benchmarks/lakeqa/tasks_20_subset/manifest.json \
        --source benchmarks/lakeqa/tasks-mini/tasks \
        --out benchmarks/lakeqa/tasks_20_subset/tasks
