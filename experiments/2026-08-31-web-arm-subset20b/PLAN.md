# Plan — web-search arm vs data-lake arms, subset20b, semantic-scored

Status: **not yet run.** Blockers in §7 must be resolved first.
Supersedes the 2026-08-30 pilot (`../2026-08-30-web-arm-subset20/`), which ran
locally on subset20 and is treated as a pilot, not a result.

## 1. Question

Hold the execution axis fixed at `standard` and vary only the retrieval
condition. `computation_tool=ideal` cannot be implemented for web search —
`execute_ideal` / `query_ideal` return the authored `record.answer`, so the
outcome would not depend on what was retrieved — so every arm is pinned to
`compute=standard` for comparability.

## 2. What changes from the pilot

| | pilot (2026-08-30) | this run |
|---|---|---|
| task set | subset20 | **subset20b** (overlaps subset20 by 4 of 20) |
| primary metric | exact_match / F1 | **semantic_match** |
| execution | local, `nohup` | **remote EC2, tmux** |
| completeness | web + ideal done, standard 10/20, naive 0/20 | all four arms, all 20 tasks |

### Why F1 is dropped

`compute_f1_score` is token overlap between the predicted and gold answer
strings (`sana_evaluation/helper/metrics.py:43`), not retrieval quality. On
LakeQA most golds are a single token (`1881`, `2941`, `Fire`), where F1 is
identical to exact match by construction. It diverges only on multi-token
entity answers, and there it mostly measures whether the model reproduced
articles: the pilot's only partial credit was gold `the Cut Bank Penguin` vs
predicted `Cut Bank Penguin`, a semantically correct answer scored 0.86 by F1
and 0 by exact match. Reporting F1 alongside exact match implies a second
independent quality signal that does not exist.

Semantic match fixes both failure modes: it credits the article-dropped answer,
and it separates `answer_unknown_blank` from `semantic_incorrect`, so a task
killed by a timeout is no longer scored identically to a confidently wrong
answer.

## 3. Arms

Model `openai/gpt-5-mini`. Fixed across arms:
`search_results=naive, profile=standard, computation_tool=standard, skills=off`.

| arm | `--search_tool` | extra | data access |
|---|---|---|---|
| web | `web` | `--no-s3` | Parallel Search API + `download` + `execute_code` |
| ideal | `ideal` | | runtime profiles + lake |
| standard | `standard` | | hybrid search over `lance_data` |
| naive | `naive` | | BM25/FTS over `lance_data` |

No `--k`. Web search pools results from all queries into one ranked list, so a
small k starves whole sub-questions — measured in the pilot: k=5 across 2
queries gave one sub-question 0 of 5 slots, dropping both veteran-disability
sources while the temperature query took all five. Every arm runs at its own
default.

## 4. Metrics

**Primary: `semantic_match`** (0/1), with `semantic_bucket` in
`{semantic_correct, semantic_incorrect, answer_unknown_blank}`.

**Secondary:** `exact_match` (kept for continuity with earlier sweeps),
`cycle_count`, `cost_usd`, `input_tokens`, **median** `runtime_seconds`
(not mean — the pilot's mean was dominated by a single 3000s stall).

**Dropped:** `f1_score`. Retained in the CSV since `_write_main_csv` emits it,
but not reported.

**Web-arm-specific:** `search_web` / `download` / `execute_code` call counts;
`from_search` provenance split; a grep for `curl|wget|urlopen|requests.get` to
confirm nothing routed around `download`.

## 5. Pipeline

1. **Run** — `run_mode_eval` per arm → `eval_results.csv`, `agent_results.jsonl`, traces
2. **Semantic audit** — `.agents/skills/semantic-eval-auditor/rewrite_semantic_eval_results.py`
   inserts `semantic_match`, `semantic_reason`, `semantic_bucket` after
   `exact_match`, writing the `*_semantic/` tree
3. **Analyse** — `sana_analysis/run_mode_analysis_semantic.py`, which also needs
   `log_error_bucket` and `log_error_evidence`
4. **Pull** — `scripts/remote_pull_outputs.sh`
5. **Archive** — into `results/` and `logs/` here

## 6. Execution (remote)

`scripts/remote_setup_run.sh` starts a detached tmux session per arm and tees to
`run_logs/<session>.log`. Detached matters: the pilot's local run was killed
twice mid-sweep, once losing an in-flight arm.

Run arms **sequentially**, not concurrently — the pilot's two worst stalls were
concurrent tasks in the same batch, and parallel arms would confound the
timeout behaviour further.

## 7. Blockers — resolve before running

1. **SSH identity is missing.** `remote_setup_run.sh` defaults to
   `asw2215.pem`; the repo has only `sana-eval_key.pem`. Need the correct key
   and host, or `REMOTE_HOST` / `REMOTE_IDENTITY` set.

2. **Remote repo path is a different repo.** Default is
   `~/eval_eqa/exploratory-qa-eval`, but this repo is `sana-framework`. Need to
   confirm the remote checkout path and that it is on a branch containing
   `--no-s3` (`exp/web-arm-subset20`, commits `c7d1d77` + `3ca3c7e`).

3. **`setup_run.py` cannot target a task subset.** `_DEFAULT_TASK_SET` is
   hardcoded with no override flag, and `remote_setup_run.sh` calls
   `setup_run.py`. Either add `--task-set` to `setup_run.py` (small, testable)
   or invoke `run_mode_eval` directly on the remote. **Recommend adding the
   flag** — the wrapper gives tmux, logging, and resume for free.

4. **subset20b exists only in local `tmp/`, which is gitignored.** Its 20 task
   files must be committed (`inputs/`) or rsynced to the remote. This is the
   same trap that left subset20 out of PR 2 entirely.

5. **The semantic auditor is not in this checkout.** `.agents/` is gitignored
   and absent locally, which is why 9 tests error. Must confirm it exists on the
   remote, and pin which model it judges with — the judge is part of the
   measurement and belongs in the provenance record.

6. **RESOLVED — the pilot's "timeouts" were the laptop sleeping.** Earlier
   diagnosis in this file said the watchdog was firing late and blamed a
   provider-side hang. That was wrong. `pmset -g log` shows the machine entered
   sleep at 20:21:17 on 2026-08-30 and again 20:38-20:47, 20:48-21:03,
   21:06-21:21. The web arm ran 18:23-19:12, entirely before the first sleep,
   and took 0 timeouts; ideal ran 19:12-21:36 straight through four sleep
   windows and took 4; standard ran 21:36-22:39 and took 1.

   `threading.Timer` waits on a monotonic clock, which does not advance during
   macOS sleep, while `runtime_seconds` uses wall-clock `time.time()`. So the
   timer fired after exactly 630s of *awake* time, as designed, while wall clock
   showed 3000s. `invoke_with_watchdog` is correct and needs no change.

   The pilot's arm comparison is still unusable, but as a scheduling artifact
   rather than an infrastructure fault: web happened to get the pre-sleep slot.
   Running on an always-on remote removes this entirely. If ever running locally
   again, wrap the sweep in `caffeinate -dimsu`.

## 8. Confounds to carry into the write-up

1. **The web arm is not measuring web-search retrieval.** `download` accepts any
   http(s) URL, and gpt-5-mini routinely constructs URLs from pretrained
   knowledge rather than following search results — live Socrata SoQL queries
   (`data.cityofnewyork.us/resource/erm2-nwe9.csv?$select=...&$group=...`),
   `data.va.gov/api/views/<id>/rows.csv`, and a timestamped
   `web.archive.org/web/20240424132412/...` URL to route around a live page.
   That may be a *stronger* data path than the lake arms get: server-side
   aggregation over the authoritative live dataset versus the lake's snapshot.
   Label this arm "agent with open web + code execution", not "weak retrieval".

2. **Snapshot drift produces false negatives.** Two pilot web failures were
   arithmetic-correct against live data: expected 2705988, computed 2,700,968
   (0.19% off); expected 2941, computed 2923.59. Gold answers derive from the
   lake's frozen snapshot. Structural for any live-data arm, independent of
   model quality. Semantic match will not rescue these — a judge should score
   them incorrect — so they need calling out separately.

3. **Answer leakage is expected and accepted.** LakeQA questions are built from
   public data, so the gold is often one search away. Pilot example: gold `115`,
   with an excerpt stating "The highest temperature ever recorded in Wyoming was
   115 at Basin" — the state record, coinciding with the county-seat record
   asked for.

4. **`from_search` provenance is not persisted.** Downloads are tagged, but
   `download` results are not written to the arm log and `trace_plugin` records
   only `tool`/`status`/`latency_ms`. **Wire this into the trace before running**,
   or the provenance split stays un-quantifiable.

5. **The sandbox network block is partial.** `execute_code` patches
   `socket.socket` in-process — which held in the pilot, blocking both
   `requests.get` and `urlopen` attempts — but the agent has `subprocess` (it
   uses `pdftotext`) and a subprocess gets its own sockets. Grep each run to
   confirm nothing routed around `download`.

6. **Not comparable to `tmp/searchaxis-naive.log`.** That BM25 sweep predates the
   FTS phrase-query fallback; this run includes it.

## 9. Deliverables

- `results/` — per-arm `eval_results.csv` (+ `*_semantic/`), `agent_results.jsonl`, traces
- `logs/` — per-arm tmux logs, per-task logs
- `summarize.py` — cross-arm table on semantic_match
- `README.md` — provenance: code SHA, remote host, judge model, corpus, caveats
