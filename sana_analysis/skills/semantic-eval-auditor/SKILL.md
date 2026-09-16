---
name: semantic-eval-auditor
description: Audit a `results...` tree file by file and then row by row with an actual model judge, rewriting only mirrored `eval_results.csv` files into a sibling `_semantic` tree while preserving lexical `exact_match` and adding semantic/error columns.
---

# Semantic Eval Auditor

## Rule #1: Start By Choosing A Backend

When this skill is invoked, start by offering exactly two backends:
1. `codex`
2. `api`

Explain them briefly:
- `codex`: use the current Codex session, allow subagents, and process one pending file chosen by the user
- `api`: use the local OpenAI API runner to rewrite files directly from the command line

Do not skip this choice.

This skill is not allowed to use deterministic string-normalization rules as the primary semantic classifier.
Semantic judgment must come from the model row by row.

## Codex Backend

Codex backend is an interactive session workflow, not the Python API runner.

Start by listing unfinished files:

```bash
python sana_analysis/skills/semantic-eval-auditor/scripts/list_pending_eval_files.py --source results-ec2
```

For SANA runs, use:

```bash
python sana_analysis/skills/semantic-eval-auditor/scripts/list_pending_eval_files.py --source sana-results
```

`sana-results` uses the prepared `sana-results/modes/...` view when present and writes pending mirrors under `sana-results_semantic/modes/...`. If `sana-results/modes` is missing, prepare it first with `python -m sana_analysis.prepare_sana_result_tree --source sana-results`.

Then:
1. Show the numbered pending files to the user.
2. Ask the user to pick one file.
3. Pin the exact selected `source_eval` and `mirrored_eval` paths from the pending-list output before auditing. Echo both paths to the user.
4. Process only that selected source file in the current Codex session.
5. You may spawn subagents for disjoint row ranges because the user explicitly requested a Codex-session workflow that can launch subagents.
6. Merge the row decisions and write only the pinned mirrored `eval_results.csv` under the `_semantic` tree.
7. Run the mandatory save gate below before saying the audit is complete.

Do not run the OpenAI API runner when the user picked `codex`.

### Mandatory Codex Save Gate

For Codex backend, completion is blocked until the selected mirror is verified. Do not substitute a sibling file if the pinned mirror is missing.

After writing the mirrored CSV, run:

```bash
python sana_analysis/skills/semantic-eval-auditor/scripts/verify_semantic_mirror.py \
  --source-eval /absolute/path/to/source/eval_results.csv \
  --mirrored-eval /absolute/path/to/output/eval_results.csv
```

Then run the pending-list check with `--include-complete --json` and confirm the selected `mirrored_eval` has `"done": true`.

The final response must include:
- the exact `source_eval`
- the exact `mirrored_eval`
- the verifier result line `Semantic mirror verification OK`
- the row count reported by the verifier

If the save gate fails, do not report completion. Fix the mirror or state that the semantic CSV was not written.

## API Backend

Use the local runner:

```bash
python sana_analysis/skills/semantic-eval-auditor/scripts/rewrite_semantic_eval_results.py --source results-ec2
```

For SANA runs:

```bash
python sana_analysis/skills/semantic-eval-auditor/scripts/rewrite_semantic_eval_results.py --source sana-results
```

This writes only mirrored semantic CSVs under `sana-results_semantic/...`. When `sana-results/modes` exists, the runner audits that mode-compatible view and ignores raw duplicate CSVs under `sana-results/sana/...`.

This path uses the OpenAI SDK plus local API credentials such as `OPENAI_API_KEY`.

## Required Workflow

For each `eval_results.csv` under the source tree:
1. Do a file-level agent pass first.
2. Then do a row-level agent pass for every row in that file.
3. Write only the mirrored `eval_results.csv` into `{source_name}_semantic/.../eval_results.csv`.

Do not copy unrelated files into the `_semantic` tree.

## Row-Level Rules

Keep lexical `exact_match` unchanged.
Add these columns:
- `semantic_match`
- `semantic_reason`
- `semantic_bucket`
- `log_error_bucket`
- `log_error_evidence`

Allowed `semantic_bucket` values:
- `semantic_correct`
- `semantic_incorrect`
- `answer_unknown_blank`

Allowed `log_error_bucket` values:
- `error_turns_exhausted`
- `error_tools_limit`
- `error_tokens_reached`
- `error_context_overflow`
- `error_event_loop`
- `error_unknown`

Use the matching task log under the mirrored `logs...` tree when needed. For `--source sana-results`, the default log root is `sana-results/logs`.

Inspect logs for:
- blank or unknown answers
- semantically incorrect answers

Policy for `log_error_bucket`:
- leave it blank when the answer is wrong but the run still completed without a real execution failure
- use it only for actual execution failures supported by the existing `error` field or the log tail
- use `error_unknown` only when there is a runtime failure signal that does not fit the named buckets

Tail policy:
- check the last `20` lines first
- if they do not contain a useful signal, expand to the last `50` lines

## Judgment Standard

The model must decide semantic equivalence from meaning, not string hacks.

Do not:
- strip suffixes like `County` as a shortcut rule
- rely on initials/full-name hacks as a hardcoded classifier
- promote correctness just because punctuation or brackets differ

Do:
- decide whether the predicted answer and expected answer refer to the same final entity, value, date, place, or set
- treat omitted administrative suffixes as semantically equivalent when the referent is unchanged in context
- treat county answers like `Erie County` vs `Erie` or `Dallas County` vs `Dallas` as `semantic_correct` when the task is clearly asking for the county entity
- treat set answers like `Spokane County, Whitman County` vs `Spokane, Whitman` as `semantic_correct` when the county set is unchanged and only the suffix is omitted
- explain the decision briefly in `semantic_reason`
- use the log tail to classify the dominant execution failure signal only when one actually occurred

## Defaults

The API runner defaults to:
- model: `gpt-5.2-codex`
- reasoning effort: `medium`

Override them with CLI flags if needed.
