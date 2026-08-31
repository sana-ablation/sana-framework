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
| search=naive (BM25) | **naive** | ideal | ideal |
| search=standard (PNEUMA) | **standard** | ideal | ideal |
| search=preloaded | **preloaded** | ideal | ideal |
| profile=naive (No Plan) | ideal | **naive** | ideal |
| profile=standard (Default) | ideal | **standard** | ideal |
| compute=standard | ideal | ideal | **standard** |

**7 cells × 3 models × 20 tasks = 420 runs.**

`computation_tool` has only `{standard, ideal}` — there is no `naive` for that
axis — so the compute ablation is a single cell rather than two.

Held fixed everywhere: `--search_results ideal --k 5 --skills off`.
These match the original 135-task experiments exactly, so the historical
`gpt-5-mini` numbers below are a direct baseline for the same configuration.

Variant directory names will therefore match the historical ones
(`search_i_results_i_plani_computei_k5_skills_off` and friends), which keeps
`sana_analysis` and any downstream tooling working unchanged.

## Metric: semantic match, not exact match

The paper figures report **`semantic_match`**, produced by the
`semantic-eval-auditor` agent skill and written into a parallel
`results_semantic/` tree. It is consistently 2-5pp above raw `exact_match`
because it accepts answers that are equivalent but not string-identical.

Verified: `semantic_match` over `results_semantic/modes/` reproduces all 14
values in figure A2.0 exactly, for both models. Raw `exact_match` does not
(e.g. mini reference is 71.9% EM vs 76.3% semantic).

**This run is therefore a two-stage pipeline**, and stage 2 has a dependency
that is not in this repo — see "Required before running" item 1.

## Axis labels

Figure labels map onto CLI flags as:

| figure | axis | flag |
|---|---|---|
| Plan: No Plan / Default / Ideal | plan | `--profile naive` / `standard` / `ideal` |
| Search: BM25 / PNEUMA / Ideal / Preloaded | search | `--search_tool naive` / `standard` / `ideal` / `preloaded` |
| Data Analysis: Standard / Ideal | execution | `--computation_tool standard` / `ideal` |

## Baseline: what the ablation shows at 135 tasks

Semantic match. Figure convention: gain relative to the weakest option in each
axis, holding the other two axes at ideal.

| cell | nano | mini |
|---|---|---|
| REFERENCE (all ideal) | 37.0% | 76.3% |
| Plan: No Plan | 34.1% | 66.7% |
| Plan: Default | 31.1% (-3.0) | 66.7% (+0.0) |
| Search: BM25 | 23.0% | 63.0% |
| Search: PNEUMA | 26.7% (+3.7) | 61.5% (-1.5) |
| Search: Preloaded | 51.9% (+28.9) | 77.0% (+14.1) |
| Data Analysis: Standard | 28.9% | 57.8% |

Gain from ideal-ising each axis:

| axis | nano | mini |
|---|---|---|
| Plan | +3.0pp | +9.6pp |
| Search | +14.1pp | +13.3pp |
| Data Analysis | +8.1pp | **+18.5pp** |

**The bottleneck ranking is already model-dependent.** At nano, search dominates
(+14.1) and execution is second (+8.1). At mini they invert: execution dominates
(+18.5) and search is second (+13.3). Planning barely matters at nano (+3.0) but
is substantial at mini (+9.6).

That inversion is the reason to run gpt-5.2. If the trend continues, execution
should dominate further; if it reverses, retrieval is the durable bottleneck.

Two anomalies worth noting: `Preloaded` beats the all-ideal reference at mini
(77.0% vs 76.3%), and `PNEUMA` is *below* `BM25` at mini (61.5% vs 63.0%) — the
hybrid index underperforms plain BM25 there.

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

Projected from `gpt-5-mini`'s **measured** per-task
`total_cost_with_all_subagents_usd` in the historical runs at this exact
configuration. The ideal axes spawn semantic-judge and repair subagents that
roughly double main-agent cost, so this is not a token-count estimate.

| cell | mini, 20 tasks |
|---|---|
| REFERENCE | $1.92 |
| Plan: No Plan | $1.82 |
| Plan: Default | $1.90 |
| Search: BM25 | $2.14 |
| Search: PNEUMA | $2.08 |
| Search: Preloaded | $1.80 |
| Data Analysis: Standard | $0.82 |
| **total** | **$12.48** |

| model | 7-cell grid |
|---|---|
| `gpt-5.4-nano` | ~$10 |
| `gpt-5-mini` | ~$12.50 (measured basis) |
| `gpt-5.2` | **$87 – $140** |
| **total** | **$110 – $162** |

`gpt-5.2` is exactly **7× mini** on every token class (1.75/0.25 in, 14/2 out,
0.175/0.025 cached), so no cache behaviour reduces it. The upper figure assumes
2.5× output tokens from reasoning.

Note `Data Analysis: Standard` is the cheapest cell by far ($0.82 vs ~$1.9) —
it is the only cell that does not run ideal computation, so it skips the judge
and repair subagents entirely.

**If $162 is too much**, in order of value retained:
- Run `gpt-5.2` on REFERENCE + `Data Analysis: Standard` only (~$25). That single
  contrast is the largest effect at mini (+18.5pp) and directly tests whether a
  frontier model closes the execution gap.
- Add REFERENCE + `Search: BM25` (~$40 total) to also get the search contrast,
  which is the axis that dominates at nano.
- Add `--reasoning-effort low` for gpt-5.2; output tokens drive its cost.
- Keep nano and mini at all 7 cells regardless — together ~$22.

Runtime: mini's historical median was ~250s/task. At `--parallel 8`, ~15 min per
cell, so **~4-6 h** for the full grid, longer for gpt-5.2.

## Required before running

1. **Semantic scoring is a separate stage, and its tooling is not in this repo.**
   The runs emit raw `results/` with `exact_match`; the paper metric comes from
   the `semantic-eval-auditor` agent skill, which rewrites a parallel
   `results_semantic/` tree adding `semantic_match` / `semantic_reason` /
   `semantic_bucket`. That skill lives in `.agents/skills/semantic-eval-auditor/`
   — gitignored, present only in `exploratory-qa-eval`, absent here. It is why
   `test_semantic_eval_auditor.py` and 8 sibling tests cannot run in this repo.
   Plan for it: either copy `.agents/` onto the box, or pull raw results back and
   score locally. Without this stage the numbers are not comparable to figure A2.0.
2. **Fix the timeout.** The web-arm run found `invoke_with_watchdog`'s
   `threading.Timer` firing ~40 min late: two tasks ran 3000s against a 630s
   deadline, a third finished at 1272s uncancelled. At gpt-5.2 prices a hung task
   is expensive. Run with `--timeout 900 --submit-grace-seconds 60` **and** set a
   client-side HTTP timeout (`client_args={"timeout": ...}` reaches
   `OpenAICachedUsageModel` through `extra_model_kwargs`; no CLI flag exists today).
3. **Materialize the subset** with `inputs/materialize_subset.py`. The tree must
   keep the `benchmarks/lakeqa/tasks-mini/tasks` path segment or runtime-profile
   lookup silently resolves to a wrong path (`runtime_profile_store.py:74`).
4. **`HYBRID_TORCH_DTYPE=float32`** on CPU-only Linux for the two `search` cells.
   PyTorch emulates float16 on x86: measured 73 ms/query at float32 vs 143 ms at
   float16. **No GPU needed** — embedding totals ~3s across a 20-task run.
5. **Copy `lance_data/` (760 MB)** — needed by `search=standard`/`naive`.
6. **`.env`** with `OPENAI_API_KEY` + AWS credentials. Azure cannot use an IAM
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
