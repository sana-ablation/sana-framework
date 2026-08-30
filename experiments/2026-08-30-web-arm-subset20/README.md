# Web-search arm vs data-lake arms — LakeQA 20-task subset

Run date: 2026-08-30
Code: `3ca3c7e` on branch `exp/web-arm-subset20`
(= `fix/eval-runtime-fixes` merged with `feat/web-search-arm`, plus `--no-s3`)

## Question

Compare retrieval conditions when the execution axis is held at `standard`.
`computation_tool=ideal` cannot be implemented for web search — `execute_ideal` /
`query_ideal` return the authored `record.answer`, so the run's outcome would not
depend on what was retrieved — so every arm is pinned to `compute=standard` for
comparability.

## Conditions

Model: `openai/gpt-5-mini`. Held fixed across arms:
`search_results=naive, profile=standard, computation_tool=standard, skills=off`.

| arm | `--search_tool` | extra | data access |
|---|---|---|---|
| web | `web` | `--no-s3` | Parallel Search API + `download` + `execute_code` |
| ideal | `ideal` | | runtime profiles + lake |
| standard | `standard` | | hybrid search over `lance_data` |
| naive | `naive` | | BM25/FTS over `lance_data` |

No `--k`: web search pools results from all queries into a single ranked list,
so a small k silently starves whole sub-questions (measured: k=5 across 2
queries gave one sub-question 0 of 5 slots). Every arm runs at its own default.

## Corpus

- S3: `lakeqa-yc4103-datalake`, folders `wikipedia`, `datagov`
- Index: local `lance_data/` (`lakeqa.lance`, `lakeqa_schema.lance`) — used by naive/standard only
- Ideal enrichment: `benchmarks/lakeqa/tasks-mini/artifacts/descriptions.jsonl` (12,708 URIs)
- Tasks: 20, listed in `inputs/subset20-manifest.json`, drawn from `benchmarks/lakeqa/tasks-mini/tasks`
- The web arm touches none of the above.

## Caveats that affect interpretation

1. **The web arm is not measuring "web search retrieval."** `download` accepts any
   http(s) URL, and gpt-5-mini routinely constructs URLs from pretrained knowledge
   rather than following search results: live Socrata SoQL queries
   (`data.cityofnewyork.us/resource/erm2-nwe9.csv?$select=...&$group=...`),
   `data.va.gov/api/views/<id>/rows.csv`, and even a timestamped
   `web.archive.org/web/20240424132412/...` URL to route around a live page.
   This may be a *stronger* data path than the lake arms get — server-side
   aggregation over the authoritative live dataset vs. the lake's snapshot.
   Read this arm as "agent with open web + code execution", not as weak retrieval.

2. **Answer leakage is expected and accepted.** LakeQA questions are built from
   public data, so the gold answer is often one search away. Example: task
   k-3-d-2/task_11 (gold `115`) has an excerpt stating "The highest temperature
   ever recorded in Wyoming was 115 at Basin" — the state record, which happens to
   equal the county-seat record asked for. Web-arm accuracy should not be read as
   retrieval quality the way the lake arms' can be.

3. **`from_search` provenance is not persisted.** Downloads are tagged
   `from_search: true/false`, but `download` results are not written to the arm log
   and `trace_plugin` records only `tool`/`status`/`latency_ms`. The split can only
   be eyeballed from URLs. Wire this into the trace before any rerun.

4. **The allowlist gate was removed mid-development.** An earlier build restricted
   `download` to URLs `search_web` had returned. That gate is gone; results here are
   from the unrestricted build only (the gated partial run was discarded).

5. **Not comparable to `tmp/searchaxis-naive.log`.** That earlier BM25 sweep ran
   before the FTS phrase-query fallback; this one includes it.

6. **Subset choice.** `subset20`, not `subset20b` — the two overlap by only 4 of 20.

8. **The 600s timeout is not enforced; the ideal arm lost 4 tasks to it.**
   `invoke_with_watchdog` computes `hard_deadline` correctly, but its
   `threading.Timer` fired ~40 min late on two tasks: `k-3-d-2/task_11` and
   `task_12` each ran 3000s against a 630s deadline, and `k-3-d-1/task_1`
   completed at 1272s without ever being cancelled. Those two 3000s tasks were
   in the same directory batch, i.e. concurrent — consistent with a provider-side
   hang inside `agent(prompt)` with no client-side HTTP timeout, not a logic
   error in the deadline math.

   Consequence: ideal scored 11/20 with 4 tasks returning no answer at all, so
   the web-vs-ideal comparison (60% vs 55%) is confounded — if two of the four
   killed tasks would have been correct, ideal leads. Ideal's 3.6 h of task time
   vs web's 64 min is likewise inflated; compare medians (418s vs 167s) instead.
   Needs its own investigation before these numbers go in the paper.

9. **Some web-arm "errors" are snapshot drift, not reasoning failures.**
   `k-5-d-3/task_15` expected 2705988 and the agent computed 2,700,968 (0.19%
   off); `k-4-d-2/task_15` expected 2941 and got 2923.59. The agent aggregated a
   *live* open-data API against a gold answer derived from the lake's frozen
   snapshot. This is structural for any live-data arm and independent of model
   quality. Two further failures were honest abstentions ("Insufficient
   evidence"), which the prompt asks for. So web's 8 failures read as: 2 drift,
   2 abstain, 4 genuinely wrong.

7. **The sandbox network block is incomplete.** `execute_code` patches
   `socket.socket` in-process, but the agent has `subprocess` (it uses `pdftotext`),
   and a subprocess gets its own sockets. Grep runs for `curl|wget|urlopen` to
   confirm nothing routed around `download`.

## Layout

- `inputs/` — task manifest, exact run script
- `results/` — per-condition `eval_results.csv`, `tools_breakdown.csv`, JSONL traces
- `logs/` — per-arm stdout and per-task logs
- `summarize.py` — cross-arm table + web-specific signals
