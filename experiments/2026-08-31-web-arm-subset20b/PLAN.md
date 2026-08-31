# Plan — web-search arm vs data-lake arms, subset20b, semantic-scored

Status: **ready to run.** All blockers in §7 are resolved; §7 is kept as a record.
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
| execution | local, `nohup` | **remote Azure VM (`sana`), tmux** |
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
`cycle_count`, `tool_calls_total`, `cost_usd`, **median** `runtime_seconds`
(not mean — the pilot's mean was dominated by a single 3000s sleep stall), and a
count of blank predictions.

**Tool-call cap.** `--max-tool-calls` defaults to 30 and the pilot hit it:
4/20 web, 1/20 ideal, 4/17 standard tasks reached the cap. Web is structurally
more exposed — its loop is `search_web -> download -> execute_code`, three calls
per source, against the lake arms' `search -> query_file`. Kept at 30 here to
match the pilot rather than silently change the experiment; `MAXTOOLS=50
./run_remote.sh` raises it. Report the at-cap count per arm either way, since a
truncated task is not a wrong answer.

**Dropped:** `f1_score`. Retained in the CSV since `_write_main_csv` emits it,
but not reported.

**Web-arm-specific:** `search_web` / `download` / `execute_code` call counts;
`from_search` provenance split; a grep for `curl|wget|urlopen|requests.get` to
confirm nothing routed around `download`.

## 5. Pipeline

1. **Run** — `run_mode_eval` per arm → `eval_results.csv`, `agent_results.jsonl`, traces
2. **Semantic audit** — **locally**, after pulling: `./audit_semantic.sh`

   The auditor (`.agents/skills/semantic-eval-auditor/scripts/rewrite_semantic_eval_results.py`)
   is standalone: stdlib plus `from openai import OpenAI`, importing nothing from
   this repo. It needs only the pulled `results/` and `logs/` and an
   `OPENAI_API_KEY`, so the eval box never needs `.agents/` and bootstrap does
   not ship it.

   It inserts `semantic_match`, `semantic_reason`, `semantic_bucket`,
   `log_error_bucket` and `log_error_evidence`, writing a sibling
   `results_semantic/` tree and preserving the lexical `exact_match`. It reads
   log *tails* to classify errors, which is why `logs/` must be pulled too.

   **The judge model is part of the measurement and must be reported.** Default
   `gpt-5.2-codex` at reasoning-effort medium; override with
   `MODEL=... ./audit_semantic.sh`. The skill forbids deterministic string
   normalisation as the primary classifier — judgment is per row, from the model.
3. **Analyse** — `sana_analysis/run_mode_analysis_semantic.py`, which also needs
   `log_error_bucket` and `log_error_evidence`
4. **Pull** — `scripts/remote_pull_outputs.sh`
5. **Archive** — into `results/` and `logs/` here

## 6. Execution (remote)

Three scripts in this directory, run against `sana-framework` directly:

| script | does |
|---|---|
| `bootstrap_remote.sh` | provisions a bare box: prereq check, `git archive` -> tarball, `.env`, `lance_data`, venv, four-arm preflight, semantic-judge check |
| `run_remote.sh` | launches all four arms sequentially in one detached tmux session |
| `watch_and_pull.sh` | polls the driver log and pulls results **after every arm**, printing a summary each time, so progress is visible without waiting for all four |
| `pull_remote.sh` | one-shot pull of `results/` + `logs/` |
| `audit_semantic.sh` | **local** semantic scoring of pulled results -> `results_semantic/` |

All three take `REMOTE_HOST`, `REMOTE_IDENTITY`, `REMOTE_DIR` from the
environment.

**`scripts/remote_setup_run.sh` is deliberately not used.** It only wraps ssh +
tmux + `setup_run.py`, its defaults point at a different repo
(`~/eval_eqa/exploratory-qa-eval`), and the thing it wraps is the problem:
`setup_run.py` hardcodes `_DEFAULT_TASK_SET` with no override, so it cannot
target subset20b at all. `run_mode_eval` already accepts `--task-set`
(`run_mode_eval.py:225`), so calling it directly removes the wrapper, the repo
path mismatch, and the need to add a flag.

The remote needs **no git** — `git archive` runs locally and ships a tarball.

Run arms **sequentially**, not concurrently, so one arm's load cannot distort
another's latency.

## 7. Blockers — resolve before running

1. **RESOLVED — SSH access.** Documented in `../README.md`: Azure VM
   `52.186.168.61` (`sana-eval`), user `asw2215`, key `sana-eval_key.pem`, via
   the `sana` alias in `~/.ssh/config`. Verified reachable: Ubuntu 24.04,
   python3 3.12.3, tar, tmux, 122 GB free. The remote home is empty, so
   `bootstrap_remote.sh` provisions from scratch.

2. **RESOLVED — repo path.** No longer a blocker: `REMOTE_DIR` defaults to
   `~/sana-framework` and `bootstrap_remote.sh` creates it. Nothing needs to
   pre-exist on the box, and it needs no git.

3. **RESOLVED — task subset.** No longer a blocker: `run_mode_eval` already
   takes `--task-set`, so dropping `setup_run.py` removes the need to add a flag.

4. **RESOLVED — subset20b portability.** Its 20 task files are committed under
   `inputs/`, so they ride in the `git archive` tarball. (`tmp/` is gitignored,
   which is the trap that left subset20 out of PR 2 entirely.)

5. **RESOLVED — the semantic auditor.** It was in the old repo at
   `~/Documents/projects/daplab/exploratory-qa-eval/.agents/`, and is now copied
   into this checkout (536 KB, 18 skills). It runs **locally** against pulled
   results and is deliberately not shipped to the box: it is a standalone
   OpenAI-API script with no repo imports, so the remote has no use for it.
   Judge pinned to `gpt-5.2-codex`, reasoning-effort medium.

   Side effect: restoring `.agents/` took the local suite from 549 to 587
   passing, since `task-quality-auditor`, `plan-verifier` and
   `author-ideal-plans` live there too. The remaining 8 failures / 4 errors need
   `other-benchmarks/`, also gitignored.

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
