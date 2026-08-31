# Model tiers x per-axis ablation — LakeQA subset20b

Run date: planned 2026-08-31 (Azure)
Code: branch `exp/web-arm-subset20` or later (must include `fix/eval-runtime-fixes`)
Status: **planned, not yet run**

## Question

How much does model capability move LakeQA accuracy, and **does it move each
SANA axis equally?** No frontier model has been run on this benchmark — the only
existing results are `gpt-5.4-nano` and `gpt-5-mini`.

The interesting question is not the headline number but whether the bottleneck
*ranking* is model-invariant. If `gpt-5.2` closes the compute gap but not the
search gap, that says the retrieval stage is the durable bottleneck; if it closes
everything, the benchmark is capability-limited rather than pipeline-limited.

## Design — leave-one-out ablation

Every axis is held at the oracle, then **one axis at a time is degraded**. The
accuracy *drop* from each degradation localises the bottleneck.

| cell | `--search_tool` | `--profile` | `--computation_tool` |
|---|---|---|---|
| REFERENCE | ideal | ideal | ideal |
| search=naive | **naive** | ideal | ideal |
| search=standard | **standard** | ideal | ideal |
| profile=naive | ideal | **naive** | ideal |
| profile=standard | ideal | **standard** | ideal |
| compute=standard | ideal | ideal | **standard** |

**6 cells × 3 models × 20 tasks = 360 runs.**

`computation_tool` has only `{standard, ideal}` — there is no `naive` for that
axis — so the compute ablation is a single cell rather than two.

Held fixed everywhere: `--search_results ideal --k 5 --skills off`.
These match the original 135-task experiments exactly, so the historical
`gpt-5-mini` numbers below are a direct baseline for the same configuration.

Variant directory names will therefore match the historical ones
(`search_i_results_i_plani_computei_k5_skills_off` and friends), which keeps
`sana_analysis` and any downstream tooling working unchanged.

## What the ablation already shows at gpt-5-mini

From the historical 135-task runs, same axes, same `k`:

| cell | EM | drop vs reference |
|---|---|---|
| REFERENCE | 72% | — |
| compute=standard | 56% | **−16pp** |
| search=standard | 59% | −13pp |
| search=naive | 60% | −12pp |
| profile=naive | 60% | −12pp |
| profile=standard | 63% | −9pp |

Execution is the largest single bottleneck at mini, search second, planning
smallest. **Whether that ordering survives at gpt-5.2 is the point of this run.**

## Corpus

- Tasks: 20, listed in `inputs/subset20b-manifest.json`
- S3: `lakeqa-yc4103-datalake`, folders `wikipedia`, `datagov`
- Index: `lance_data/` — `search=standard`/`naive` only
- Runtime profiles: `benchmarks/lakeqa/tasks-mini/runtime-profiles` — every
  ideal axis reads these, so all 6 cells depend on them

### Why subset20b

`subset20b` is stratified jointly on node count and historical solve rate;
`subset20` was stratified on node count alone and landed materially easier.

| split | mean solve rate | mean nodes |
|---|---|---|
| `subset20` | 55.6% | 7.45 |
| **`subset20b`** | **42.8%** | **7.10** |
| full 135 (target) | 44.7% | 7.07 |

They overlap by only 4 of 20 tasks.

### Tasks

| task | nodes | wiki | datagov | historical solve |
|---|---|---|---|---|
| `k-3-d-2/task_11` | 4 | 2 | 2 | 38.9% |
| `k-3-d-2/task_12` | 4 | 2 | 2 | 0.0% |
| `k-3-d-2/task_3` | 4 | 2 | 2 | 77.8% |
| `k-3-d-2/task_7` | 4 | 1 | 3 | 16.7% |
| `k-4-d-2/task_7` | 5 | 1 | 4 | 66.7% |
| `k-3-d-4/task_8` | 6 | 2 | 4 | 72.2% |
| `k-4-d-2/task_14` | 6 | 3 | 3 | 72.2% |
| `k-4-d-3/task_5` | 6 | 3 | 3 | 72.2% |
| `k-4-d-4/task_1` | 7 | 1 | 6 | 11.1% |
| `k-5-d-3/task_13` | 7 | 2 | 5 | 38.9% |
| `k-5-d-3/task_14` | 7 | 2 | 5 | 33.3% |
| `k-3-d-4/task_6` | 8 | 0 | 8 | 0.0% |
| `k-4-d-3/task_10` | 8 | 3 | 5 | 72.2% |
| `k-5-d-2/task_6` | 8 | 6 | 2 | 72.2% |
| `k-3-d-3/task_4` | 9 | 0 | 9 | 33.3% |
| `k-5-d-3/task_11` | 9 | 2 | 7 | 33.3% |
| `k-4-d-3/task_6` | 10 | 1 | 9 | 0.0% |
| `k-4-d-5/task_5` | 10 | 4 | 6 | 11.1% |
| `k-6-d-2/task_2` | 10 | 6 | 4 | 66.7% |
| `k-6-d-3/task_1` | 10 | 4 | 6 | 66.7% |

## Cost — read this before launching

Projected from `gpt-5-mini`'s **measured** per-task cost in the historical runs
at this exact configuration, using `total_cost_with_all_subagents_usd` (the
ideal axes spawn judge and repair subagents that roughly double main-agent cost).

| model | 6-cell grid, 20 tasks |
|---|---|
| `gpt-5.4-nano` | ~$8.50 |
| `gpt-5-mini` | ~$10.70 (measured basis) |
| `gpt-5.2` | **$75 – $120** |
| **total** | **$94 – $139** |

`gpt-5.2` is exactly **7× mini** on every token class (1.75/0.25 input,
14/2 output, 0.175/0.025 cached), so there is no cache-driven discount to hope
for. The $120 figure assumes 2.5× output tokens from reasoning.

Two things make this far pricier than a naive estimate:

1. **Ideal subagents.** `compute=ideal` cells cost ~$1.92/20 tasks at mini vs
   $0.82 for `compute=standard` — the semantic judge and repair agents more than
   double it. Five of six cells run `compute=ideal`.
2. **Ideal-compute runs are longer.** ~400k input tokens/task vs ~220k at
   `compute=standard`.

**If $139 is too much**, the cheapest meaningful reductions, in order:
- Run `gpt-5.2` on REFERENCE + `compute=standard` only (2 cells, ~$25). That is
  the largest ablation signal and answers "does a frontier model close the
  execution gap?" — the single most interesting question here.
- Add `--reasoning-effort low` for `gpt-5.2`; output tokens dominate its cost.
- Keep nano and mini at all 6 cells regardless; together they are ~$19.

Runtime: mini's historical median was ~250s/task. At `--parallel 8`, ~15 min per
cell, so **~3-5 h for the full grid**, longer for gpt-5.2.

## Required before running

1. **Fix the timeout.** The web-arm run found `invoke_with_watchdog`'s
   `threading.Timer` firing ~40 min late: two tasks ran 3000s against a 630s
   deadline, a third finished at 1272s uncancelled. At gpt-5.2 prices a hung task
   is expensive. Run with `--timeout 900 --submit-grace-seconds 60` **and** set a
   client-side HTTP timeout (`client_args={"timeout": ...}` reaches
   `OpenAICachedUsageModel` through `extra_model_kwargs`; no CLI flag exists today).
2. **Materialize the subset** with `inputs/materialize_subset.py`. The tree must
   keep the `benchmarks/lakeqa/tasks-mini/tasks` path segment or runtime-profile
   lookup silently resolves to a wrong path (`runtime_profile_store.py:74`).
3. **`HYBRID_TORCH_DTYPE=float32`** on CPU-only Linux for the two `search` cells.
   PyTorch emulates float16 on x86: measured 73 ms/query at float32 vs 143 ms at
   float16. **No GPU needed** — embedding totals ~3s across a 20-task run.
4. **Copy `lance_data/` (760 MB)** — needed by `search=standard`/`naive`.
5. **`.env`** with `OPENAI_API_KEY` + AWS credentials. Azure cannot use an IAM
   instance role, so scope the AWS keys read-only to the bucket.

## Caveats

1. **20 tasks means one task = 5pp.** The historical drops span 9-16pp, so the
   ablation *ordering* may not be resolvable at n=20 even though it is clear at
   n=135. Treat gaps under ~10pp as unresolved.
2. **The historical baseline is 135 tasks, this run is 20.** Absolute numbers are
   not directly comparable to the table above; the within-run drops are.
3. **Solve-rate stratification came from ideal-compute runs**, which is the
   configuration here — so unlike the web-arm experiment, the stratification and
   the measurement are aligned.
4. **The FTS index still lacks positions.** Phrase queries degrade to keyword
   search via the fallback rather than erroring, so `search=standard`/`naive` run
   against a slightly weaker index than intended.
5. **gpt-5.2 cost assumes mini's token profile.** If it reasons much longer, the
   output term dominates and the real figure exceeds the $120 column.

## Layout

- `inputs/` — manifest, materializer, run script
- `results/` — per-cell `eval_results.csv`, `tools_breakdown.csv`, traces
- `logs/` — per-cell stdout and per-task logs
- `archive.sh` — copy scratch output from `tmp/` (gitignored) into here
- `summarize.py` — model × cell table with ablation drops
