# Model tiers on LakeQA — nano vs mini vs gpt-5.2, 20-task subset

Run date: planned 2026-08-31 (Azure)
Code: branch `exp/web-arm-subset20` or later (must include `fix/eval-runtime-fixes`)
Status: **planned, not yet run**

## Question

How much does model capability move LakeQA accuracy, and does it move retrieval
and execution equally? Nobody has run a frontier model on this benchmark — the
only existing LakeQA results are `gpt-5.4-nano` and `gpt-5-mini`. The practical
ask is "when someone asks how a big model does on this, what do we say?"

Secondary: whether the SANA bottleneck story (oracle retrieval beats real
retrieval by a wide margin) survives at a larger model, or whether a stronger
model closes the retrieval gap on its own.

## Conditions

Grid: **3 models × 3 search arms = 9 cells × 20 tasks = 180 runs.**

Held fixed across every cell:
`search_results=naive, profile=standard, computation_tool=standard, skills=off`.

| axis | values |
|---|---|
| `--model-name` | `openai/gpt-5.4-nano`, `openai/gpt-5-mini`, `openai/gpt-5.2` |
| `--search_tool` | `ideal` (profile oracle), `standard` (hybrid LanceDB), `naive` (BM25/FTS) |

These are deliberately the **same fixed axes** as
`experiments/2026-08-30-web-arm-subset20`, so the two experiments compose into
one grid: that one gives `mini × {web, ideal, standard, naive}`, this one gives
`{nano, mini, 5.2} × {ideal, standard, naive}`. The shared
`mini × {ideal, standard, naive}` cells appear in both — on different subsets,
which makes them a direct measurement of the subset effect (see Caveats 1).

`computation_tool=standard`, not `ideal`: `execute_ideal`/`query_ideal` return the
authored `record.answer` regardless of what was retrieved, so an ideal-compute
cell measures the oracle, not the model. It also keeps this comparable to the
web arm, where ideal compute is impossible.

No `--k`: every arm runs at its own default, matching the web-arm experiment.

## Corpus

- Tasks: 20, listed in `inputs/subset20b-manifest.json`, drawn from `benchmarks/lakeqa/tasks-mini/tasks`
- S3: `lakeqa-yc4103-datalake`, folders `wikipedia`, `datagov`
- Index: `lance_data/` (`lakeqa.lance`, `lakeqa_schema.lance`) — `standard`/`naive` only
- Ideal enrichment: `benchmarks/lakeqa/tasks-mini/artifacts/descriptions.jsonl`

### Why subset20b and not subset20

`subset20b` is stratified on **both** node count and historical solve rate;
`subset20` was stratified on node count alone and came out materially easier.

| split | mean solve rate | mean nodes |
|---|---|---|
| `subset20` | 55.6% | 7.45 |
| **`subset20b`** | **42.8%** | **7.10** |
| full 135 (target) | 44.7% | 7.07 |

The two overlap by only 4 of 20 tasks. Since this experiment exists to produce a
number people will quote, it uses the split that is within 2pp of the full
benchmark rather than 11pp above it.

Historical solve rate = mean `exact_match` over ~18 prior runs per task
(`../exploratory-qa-eval/results/modes/`), across mixed models and variants. It
measures difficulty *under the modes already run*, so it is a proxy, not ground truth.

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

## Cost and runtime

Projected from `gpt-5-mini`'s **measured** token counts in the web-arm run at
`compute=standard` (not from an ideal-compute run, which has a different shape).

| arm | nano | mini | gpt-5.2 (same tokens) | gpt-5.2 (2.5× output) |
|---|---|---|---|---|
| ideal | $0.30 | $0.43 | $2.99 | $5.60 |
| standard | $0.37 | $0.51 | $3.54 | $5.84 |
| naive | ~$0.37 | ~$0.51 | ~$3.54 | ~$5.84 |

**Total: ~$8–13.** The 2.5× output column is the realistic one for gpt-5.2,
which emits reasoning tokens.

Runtime: mini's median task was 418s (ideal) / 186s (standard). At
`--parallel 8` expect ~20-35 min per cell, so **4-6 h for the full grid**.
The Azure VM at ~$0.38/h will cost more than the API calls — deallocate when done.

## Required before running

1. **Raise the timeout and add a client-side HTTP timeout.** The web-arm run found
   that `invoke_with_watchdog`'s `threading.Timer` does not reliably fire: two
   tasks ran 3000s against a 630s deadline and a third completed at 1272s without
   being cancelled. Consistent with a provider-side hang inside `agent(prompt)`
   with no client-side HTTP timeout. Without a fix, gpt-5.2 cells can silently
   burn hours. Run with `--timeout 900 --submit-grace-seconds 60`, and set a
   request timeout on the OpenAI client (`client_args={"timeout": ...}` reaches
   `OpenAICachedUsageModel` via `extra_model_kwargs`; there is no CLI flag today).
2. **Materialize the subset** with `inputs/materialize_subset.py` — the task tree
   must keep the `benchmarks/lakeqa/tasks-mini/tasks` path segment or runtime
   profile lookup silently resolves to a wrong path (`runtime_profile_store.py:74`).
3. **`HYBRID_TORCH_DTYPE=float32`** for `standard`/`naive` on a CPU-only Linux box.
   The `qwen3_0_6b` preset hardcodes float16, which PyTorch emulates on x86 —
   measured 73 ms/query at float32 vs 143 ms at float16 on CPU. No GPU needed:
   embedding is ~3s across a whole 20-task run.
4. **Copy `lance_data/` (760 MB)** to the box — `standard`/`naive` need it.
   `ideal` does not, but `--db-path` must still point at an existing directory.
5. **`.env`** with `OPENAI_API_KEY` plus AWS credentials. Azure cannot use an IAM
   instance role, so these are real keys — scope them read-only to the bucket.

## Caveats that will affect interpretation

1. **Cross-experiment comparison is confounded by subset.** The shared
   `mini × {ideal, standard, naive}` cells run on `subset20` there and
   `subset20b` here. Any difference is subset + code drift, not model. This is
   also the cleanest available estimate of how much the subset choice matters —
   worth reporting as such rather than treating as noise.
2. **Historical solve rate was computed from runs that used `compute=ideal`.**
   The stratification is therefore anchored to a slightly different task
   difficulty than this experiment measures. Directionally fine, not exact.
3. **20 tasks is a small sample.** A single task flipping is 5pp. Differences
   under ~10pp between adjacent model tiers should not be read as real.
4. **The FTS index still lacks positions.** Phrase queries degrade to keyword
   search via the fallback in `fix/eval-runtime-fixes` rather than erroring, but
   `standard`/`naive` are running against a slightly weaker index than intended.
5. **Cost projections assume gpt-5.2's token profile resembles mini's.** If it
   reasons substantially longer, the output-token term dominates and the real
   figure lands above the 2.5× column.

## Layout

- `inputs/` — task manifest, materializer, exact run script
- `results/` — per-cell `eval_results.csv`, `tools_breakdown.csv`, JSONL traces
- `logs/` — per-cell stdout and per-task logs
- `archive.sh` — copy scratch output from `tmp/` into here (tmp/ is gitignored)
- `summarize.py` — model × arm table
